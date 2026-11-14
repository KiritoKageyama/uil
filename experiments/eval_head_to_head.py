"""Head-to-head: the full UIL run vs ordinary learners on the SAME splits.

WHY
---
"The cognitive layer ties plain XGBoost+PSO on accuracy" is the project's
standing summary, but `eval_crossseed` only ever ran the SymbolicFormula arm.
The XGBoost side of that comparison came from other harnesses, other splits,
and in some cases other sessions. A tie asserted across two separately-run
tables is not a paired result -- and the Gate 2 re-scoring
(`experiments/rescore_gates.py`) showed how badly unpaired 3-seed means
mislead here.

This runs the full UIL optimizer and every baseline on the *identical*
train/test split, pairs by (dataset, seed), and applies the same interval +
power discipline. `--uil-json` reuses an existing eval_crossseed run so the
expensive arm is not repeated.

Run:
    python -m figpso_uil.experiments.eval_head_to_head --datasets sonar
    python -m figpso_uil.experiments.eval_head_to_head \
        --datasets breast_cancer,wine,ionosphere \
        --uil-json results/compute_cost/uil_on.json
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from figpso_uil.experiments.eval_crossseed import DATASETS, one_split
from figpso_uil.experiments.rescore_gates import analyze

BASELINES = ("xgboost", "random_forest", "logreg", "symbolic_single_fit")


def _baseline(name: str, Xtr, ytr, Xte, seed: int, n_classes: int):
    if name == "xgboost":
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                          random_state=seed, verbosity=0)
    elif name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)
    elif name == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=5000, random_state=seed))
    elif name == "symbolic_single_fit":
        # One ordinary fit of the SAME learner UIL spends its whole search
        # budget tuning -- the most informative baseline on this table.
        from figpso_uil.adapters.symbolic_formula_adapter import SymbolicFormulaAdapter
        import tempfile, os
        sd = tempfile.mkdtemp(prefix="h2h_")
        clf = SymbolicFormulaAdapter(
            seed=seed, num_classes=n_classes, learning_mode="sparse_linear",
            formula_library_path=os.path.join(sd, "lib.json"))
        model = clf.fit_weighted(Xtr, ytr, np.ones(Xtr.shape[1]))
        return clf.predict(model, Xte)
    else:
        raise ValueError(name)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="sonar")
    ap.add_argument("--seeds", default="42,7,123")
    ap.add_argument("--stages", type=int, default=20)
    ap.add_argument("--uil-json", default=None,
                    help="reuse UIL accuracies from an eval_crossseed --json run")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    names = [d.strip() for d in args.datasets.split(",") if d.strip() in DATASETS]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    cached: dict[tuple[str, int], float] = {}
    if args.uil_json:
        with open(args.uil_json, encoding="utf-8") as fh:
            for r in json.load(fh).get("per_seed", []):
                cached[(r["dataset"], int(r["seed"]))] = float(r["acc"])

    uil_map: dict[tuple[str, int], float] = {}
    base_maps = {b: {} for b in BASELINES}
    rows = []

    for name in names:
        loader, mode = DATASETS[name]
        X, y = loader()
        X = np.asarray(X, float); y = np.asarray(y, int)
        n_classes = int(len(np.unique(y)))
        print(f"\n=== {name} ({X.shape[0]}x{X.shape[1]}, {n_classes} cls) ===",
              flush=True)
        for s in seeds:
            key = (name, s)
            # Identical split to eval_crossseed.one_split -- that is what makes
            # this paired rather than two tables laid side by side.
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=s, stratify=y)

            if key in cached:
                uil_acc = cached[key]
                uil_s = float("nan")
            else:
                t0 = time.perf_counter()
                r = one_split(X, y, mode, s, args.stages, 10, 10, 40, n_classes)
                uil_acc = float(r["acc"]); uil_s = time.perf_counter() - t0
            uil_map[key] = uil_acc

            line = f"  seed {s:>4}: uil={uil_acc:.4f}"
            for b in BASELINES:
                t0 = time.perf_counter()
                pred = _baseline(b, Xtr, ytr, Xte, s, n_classes)
                el = time.perf_counter() - t0
                a = float(accuracy_score(yte, pred))
                base_maps[b][key] = a
                rows.append({"dataset": name, "seed": s, "learner": b,
                             "acc": a, "macro_f1": float(
                                 f1_score(yte, pred, average="macro",
                                          zero_division=0)),
                             "fit_s": el})
                line += f"  {b.split('_')[0]}={a:.4f}"
            rows.append({"dataset": name, "seed": s, "learner": "uil",
                         "acc": uil_acc, "fit_s": uil_s})
            print(line, flush=True)

    print(f"\n{'=' * 78}")
    print("UIL minus each baseline, paired by (dataset, seed)")
    print(f"{'=' * 78}")
    reports = {}
    for b in BASELINES:
        rep = analyze(f"uil - {b}", base_maps[b], uil_map)
        reports[b] = rep
        if rep["n_cells"] < 2:
            print(f"{b:<22} insufficient cells"); continue
        w, t, l = rep["win_tie_lose"]
        print(f"{b:<22} mean {rep['mean_delta']:+.4f}  "
              f"CI [{rep['ci95'][0]:+.4f}, {rep['ci95'][1]:+.4f}]  "
              f"{w}W/{t}T/{l}L  need {rep['n_needed_for_0.01']} cells  "
              f"-> {rep['verdict']}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows, "reports": reports,
                       "seeds": seeds, "datasets": names}, fh, indent=2,
                      default=float)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
