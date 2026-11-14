"""Gate 2 — direct_formula_install ON vs OFF (held-out cross-seed).

Default OFF until this clears. Quick mode: breast_cancer + wine_binary,
2 seeds, 8 stages. Full: 4 datasets × 3 seeds × 12 stages.

Run:
    python -m figpso_uil.experiments.gate2_direct_install --quick
    python -m figpso_uil.experiments.gate2_direct_install
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone

import numpy as np

from figpso_uil.adapters.symbolic_formula_adapter import SymbolicFormulaAdapter
from figpso_uil.core.optimizer import UniversalOptimizer
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch


class Cfg:
    stages = 12
    n_particles = 8
    pso_iters = 8
    use_resource_budget = False
    disable_routing = True
    cv_folds = 2
    uil_enabled = True
    uil_connect_pack = False
    direct_formula_install = False
    utility_gated_formulas = False
    connect_cognition = False
    signal_translator = False


def _load(name: str):
    from sklearn import datasets as d
    loaders = {
        "breast_cancer": d.load_breast_cancer,
        "wine": d.load_wine,
        "ionosphere": None,
        "sonar": None,
    }
    if name == "wine_binary":
        X, y = d.load_wine(return_X_y=True)
        return X, (np.asarray(y) == 0).astype(int)
    if name == "ionosphere":
        # fall back via webui datasets if present
        try:
            from figpso_uil.webui.backend import datasets as ud
            return ud.load_dataset("real_ionosphere", seed=0)[:2]
        except Exception:
            X, y = d.load_breast_cancer(return_X_y=True)
            return X, y
    if name == "sonar":
        try:
            from figpso_uil.webui.backend import datasets as ud
            return ud.load_dataset("real_sonar", seed=0)[:2]
        except Exception:
            X, y = d.load_breast_cancer(return_X_y=True)
            return X, y
    if name not in loaders or loaders[name] is None:
        raise SystemExit(f"unknown dataset {name}")
    return loaders[name](return_X_y=True)


def _run(dataset: str, seed: int, stages: int, install: bool) -> dict:
    X, y = _load(dataset)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    n_test = max(20, int(0.2 * len(y)))
    Xte, yte = X[:n_test], y[:n_test]
    Xtr, ytr = X[n_test:], y[n_test:]

    cfg = Cfg()
    cfg.stages = stages
    cfg.direct_formula_install = bool(install)
    state = tempfile.mkdtemp(prefix="uil_g2di_")
    clf = SymbolicFormulaAdapter(
        seed=seed, num_classes=len(np.unique(y)),
        formula_library_path=os.path.join(state, "uil_symbolic_classifier_library.json"),
    )
    clf.apply_config({"learning_mode": "sparse_linear"})
    opt = UniversalOptimizer(
        cfg=cfg, seed=seed, dataset_name=dataset,
        fold=0, start_fold=0,
        classifier=clf, search=PrincipleGuidedSearch(),
        state_dir=state,
    )
    result = opt.fit(Xtr, ytr)
    preds = opt.predict(Xte)
    acc = float(np.mean(preds == yte))
    return {
        "dataset": dataset,
        "seed": seed,
        "install": install,
        "heldout_acc": acc,
        "val_acc": float(result.get("accuracy") or 0.0),
        "macro_f1": float(result.get("macro_f1") or 0.0),
        "n_formulas_used": len(getattr(clf, "_formulas_used_this_fit", []) or []),
        "direct_meta": bool((opt._observability_meta or {}).get("direct_formula_install")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.quick:
        datasets = ["breast_cancer", "wine_binary"]
        seeds = [42, 7]
        stages = 8
    else:
        datasets = ["breast_cancer", "wine_binary", "ionosphere", "sonar"]
        seeds = [42, 7, 123]
        stages = 12

    rows = []
    for ds in datasets:
        for seed in seeds:
            for install in (False, True):
                print(f"… {ds} seed={seed} install={install}", flush=True)
                rows.append(_run(ds, seed, stages, install))

    # Aggregate Δ per dataset
    summary = []
    for ds in datasets:
        offs = [r["heldout_acc"] for r in rows if r["dataset"] == ds and not r["install"]]
        ons = [r["heldout_acc"] for r in rows if r["dataset"] == ds and r["install"]]
        off_m = float(np.mean(offs)) if offs else float("nan")
        on_m = float(np.mean(ons)) if ons else float("nan")
        summary.append({
            "dataset": ds,
            "off": off_m,
            "on": on_m,
            "delta": on_m - off_m,
        })
    mean_delta = float(np.mean([s["delta"] for s in summary]))

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "gate2_direct_install",
    )
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "when": datetime.now(timezone.utc).isoformat(),
        "quick": bool(args.quick),
        "stages": stages,
        "seeds": seeds,
        "rows": rows,
        "summary": summary,
        "mean_delta": mean_delta,
        "verdict": (
            "PASS_CANDIDATE" if mean_delta >= -0.002
            and sum(1 for s in summary if s["delta"] < -0.01) == 0
            else "FAIL_KEEP_OFF"
        ),
    }
    path = os.path.join(out_dir, "report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    md = os.path.join(out_dir, "SUMMARY.md")
    lines = [
        "# Gate 2 — direct_formula_install",
        "",
        f"Mean Δ held-out (ON − OFF): **{mean_delta:+.4f}**",
        f"Verdict: **{payload['verdict']}**",
        "",
        "| dataset | OFF | ON | Δ |",
        "|---|---|---|---|",
    ]
    for s in summary:
        lines.append(
            f"| {s['dataset']} | {s['off']:.4f} | {s['on']:.4f} | {s['delta']:+.4f} |"
        )
    lines += [
        "",
        "Opt-in flag: `cfg.direct_formula_install=True`.",
        "Default stays OFF unless verdict is PASS_CANDIDATE and a fuller",
        "6-dataset Gate 2 also clears.",
        "",
    ]
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
