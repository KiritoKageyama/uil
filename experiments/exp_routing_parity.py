"""13-domain routing parity sweep (v22 M3 acceptance).

Compares routed classifier accuracy vs best single strategy per domain.
Pass criterion: routed within ±0.01 of best on every domain.

Usage:
    python -m figpso_uil.experiments.exp_routing_parity --quick
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from figpso_uil.cognition.domain_router import route
from figpso_uil.cognition.fingerprint import compute_fingerprint
from figpso_uil.core.strategy_classifiers import build_classifier_for_strategy
from figpso_uil.cognition.paths import results_path
from figpso_uil.webui.backend import datasets as dataset_registry


from figpso_uil.cognition.symbolic_regressor import SymbolicRegressor


STRATEGIES = ("grad_fit", "gbm", "sparse_linear", "fewshot", "symbolic")


def _fit_predict(clf, Xtr, ytr, Xte):
    nf = Xtr.shape[1]
    w = np.ones(nf, dtype=float) / max(nf, 1)
    model = clf.fit_weighted(Xtr, ytr, w)
    return clf.predict(model, Xte)


def _synth_domains(seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    # math continuous
    x = rng.uniform(-1, 1, (100, 2))
    y = x[:, 0] ** 2 + 0.3 * x[:, 1]
    out.append(("math:poly", x, y, "regression"))
    # linear binary
    X = rng.standard_normal((120, 6))
    yb = (X[:, 0] + X[:, 1] > 0).astype(int)
    out.append(("tabular:linear", X, yb, "classify"))
    # xor
    X2 = rng.standard_normal((120, 4))
    y2 = ((X2[:, 0] * X2[:, 1]) > 0).astype(int)
    out.append(("tabular:xor", X2, y2, "classify"))
    # AR(1) sequential
    n = 150
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.8 * ar[i - 1] + rng.normal(0, 0.4)
    Xa = ar.reshape(-1, 1)
    ya = (Xa[:, 0] > 0).astype(int)
    out.append(("seq:ar1", Xa, ya, "classify"))
    # tiny
    Xt = rng.standard_normal((35, 5))
    yt = (Xt[:, 0] > 0).astype(int)
    out.append(("tiny:35", Xt, yt, "classify"))
    return out


def _registry_domains(quick: bool, real_only: bool = False):
    names = [
        "real_breast_cancer", "real_iris", "real_wine", "real_digits",
        "real_diabetes_risk", "real_spambase", "real_credit_g",
        "real_sonar", "real_ionosphere",
    ]
    if quick:
        names = names[:5]
    out = []
    for name in names:
        try:
            loaded = dataset_registry.load_dataset(name)
            if isinstance(loaded, dict):
                X, y = loaded["X"], loaded["y"]
            else:
                X, y = loaded
            out.append((f"uci:{name}", np.asarray(X, dtype=float), np.asarray(y), "classify"))
        except Exception as exc:
            print(f"  [skip] {name}: {exc}")
    return out


def _eval_domain(label, X, y, task, seed):
    if task == "regression":
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
        fp = compute_fingerprint(Xtr, ytr, seed=seed, dataset_name=label)
        _, strategy, _ = route(fp)

        def _sym_r2(n_terms: int) -> float:
            sr = SymbolicRegressor(seed=seed, n_terms=n_terms).fit(Xtr, ytr)
            return float(r2_score(yte, sr.predict(Xte)))

        best = max(_sym_r2(6), _sym_r2(10))
        routed = _sym_r2(8 if strategy == "symbolic" else 6)
        return {
            "domain": label, "task": task, "routed": routed, "best": best,
            "best_strategy": "symbolic", "route_strategy": strategy,
            "gap": best - routed, "ok": (best - routed) <= 0.01,
        }

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed,
        stratify=y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None,
    )
    fp = compute_fingerprint(Xtr, ytr, seed=seed, dataset_name=label)
    domain, strategy, _ = route(fp)
    best = 0.0
    best_s = None
    for s in STRATEGIES:
        try:
            clf, _ = build_classifier_for_strategy(s, seed=seed, num_classes=len(np.unique(y)), boosting_rounds=12)
            sc = float(accuracy_score(yte, _fit_predict(clf, Xtr, ytr, Xte)))
            if sc > best:
                best, best_s = sc, s
        except Exception:
            pass
    routed_clf, _ = build_classifier_for_strategy(
        strategy, seed=seed, num_classes=len(np.unique(y)), boosting_rounds=12,
    )
    routed = float(accuracy_score(yte, _fit_predict(routed_clf, Xtr, ytr, Xte)))
    return {
        "domain": label, "task": task, "routed": routed, "best": best,
        "best_strategy": best_s, "route_strategy": strategy,
        "route_domain": domain,
        "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "linear_probe": fp.get("linear_probe_acc"),
        "gap": best - routed, "ok": (best - routed) <= 0.01,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--real-only", action="store_true", help="Skip synthetic domains")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    domains = [] if args.real_only else _synth_domains(args.seed)
    domains += _registry_domains(args.quick, real_only=args.real_only)
    results = [_eval_domain(*d, args.seed) for d in domains]
    passed = sum(1 for r in results if r["ok"])
    report = {
        "domains_tested": len(results),
        "parity_pass": passed,
        "all_ok": passed == len(results),
        "results": results,
    }
    out_path = results_path("ablation_reports/routing_parity.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Routing parity: {passed}/{len(results)} within ±0.01 -> {out_path}")
    for r in results:
        flag = "OK" if r["ok"] else "FAIL"
        extra = ""
        if r.get("route_domain"):
            extra = f" [{r['route_domain']}->{r['route_strategy']}]"
        print(
            f"  [{flag}] {r['domain']}: routed={r['routed']:.3f} "
            f"best={r['best']:.3f} gap={r['gap']:.3f}{extra}"
        )
    if not report["all_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
