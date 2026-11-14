"""Quick real-dataset smoke: routing + held-out accuracy + short UIL run.

Usage:
    python figpso_uil/experiments/test_real_datasets.py
    python figpso_uil/experiments/test_real_datasets.py --uil-stages 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from figpso_uil.webui.backend.curated_datasets import list_curated, load_curated
from figpso_uil.cognition.fingerprint import compute_fingerprint
from figpso_uil.cognition.domain_router import route, explain
from figpso_uil.core.strategy_classifiers import build_classifier_for_strategy
from figpso_uil.adapters.xgboost_adapter import XGBoostAdapter
from figpso_uil.core.optimizer import UniversalOptimizer
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch
from figpso_uil.cognition.paths import results_path


def _fit_predict(clf, Xtr, ytr, Xte):
    w = np.ones(Xtr.shape[1], dtype=float) / max(Xtr.shape[1], 1)
    model = clf.fit_weighted(Xtr, ytr, w)
    return clf.predict(model, Xte)


class FastCfg:
    stages = 8
    n_particles = 6
    pso_iters = 6
    boosting_rounds = 15
    use_resource_budget = False
    cv_folds = 2


def eval_routing(name: str, seed: int) -> dict:
    X, y = load_curated(name, seed=seed)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed,
        stratify=y if min(np.bincount(y)) >= 2 else None,
    )
    fp = compute_fingerprint(Xtr, ytr, seed=seed, dataset_name=name)
    domain, strategy, _ = route(fp)
    clf, clf_name = build_classifier_for_strategy(
        strategy, seed=seed, num_classes=len(np.unique(y)),
        boosting_rounds=15,
    )
    pred = _fit_predict(clf, Xtr, ytr, Xte)
    acc = float(accuracy_score(yte, pred))
    return {
        "dataset": name,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "route_domain": domain,
        "route_strategy": strategy,
        "classifier": clf_name,
        "held_out_acc": round(acc, 4),
        "linear_probe": fp.get("linear_probe_acc"),
        "sequential": fp.get("sequential"),
        "explain": explain(fp),
    }


def eval_uil_short(name: str, seed: int, stages: int) -> dict:
    X, y = load_curated(name, seed=seed)
    n_classes = len(np.unique(y))
    state_dir = tempfile.mkdtemp(prefix=f"uil_real_{name}_")
    cfg = FastCfg()
    cfg.stages = stages
    opt = UniversalOptimizer(
        cfg=cfg,
        seed=seed,
        dataset_name=name,
        fold=0,
        start_fold=0,
        classifier=XGBoostAdapter(seed=seed, num_classes=n_classes, boosting_rounds=15),
        search=PrincipleGuidedSearch(),
        state_dir=state_dir,
    )
    t0 = time.perf_counter()
    result = opt.fit(X, y)
    elapsed = time.perf_counter() - t0
    return {
        "dataset": name,
        "uil_accuracy": round(float(result["accuracy"]), 4),
        "principles": int(result.get("principles_discovered", 0)),
        "route": getattr(opt, "_route_decision", {}),
        "fingerprint_id": opt._observability_meta.get("fingerprint_id"),
        "seconds": round(elapsed, 1),
        "state_dir": state_dir,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uil-stages", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-uil", action="store_true")
    args = ap.parse_args()

    names = list_curated()
    print("=" * 72)
    print("REAL DATASET ROUTING TEST")
    print("=" * 72)

    routing_results = []
    for name in names:
        try:
            r = eval_routing(name, args.seed)
            routing_results.append(r)
            print(
                f"  {name}: acc={r['held_out_acc']:.3f} "
                f"[{r['route_domain']}->{r['route_strategy']}] "
                f"lin={r['linear_probe']}"
            )
        except Exception as exc:
            print(f"  {name}: FAIL {exc}")
            routing_results.append({"dataset": name, "error": str(exc)})

    uil_results = []
    if not args.skip_uil:
        print("\n" + "=" * 72)
        print(f"SHORT UIL RUNS ({args.uil_stages} stages, isolated state)")
        print("=" * 72)
        for name in ("real_breast_cancer", "real_sonar", "real_wine"):
            try:
                u = eval_uil_short(name, args.seed, args.uil_stages)
                uil_results.append(u)
                rd = u["route"]
                print(
                    f"  {name}: acc={u['uil_accuracy']:.3f} "
                    f"principles={u['principles']} "
                    f"route={rd.get('domain')}->{rd.get('strategy')} "
                    f"({u['seconds']}s)"
                )
            except Exception as exc:
                print(f"  {name}: FAIL {exc}")
                uil_results.append({"dataset": name, "error": str(exc)})

    report = {
        "routing": routing_results,
        "uil_short": uil_results,
    }
    out = results_path("ablation_reports/real_dataset_test.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
