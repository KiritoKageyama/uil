"""What does UIL's compute actually buy, and what is it a multiple OF?

CLAUDE.md's headline says the cognitive layer ties plain search on accuracy
"at 10-100x the compute". That number had no measurement behind it -- until
2026-08-05 `eval_crossseed` had no timing instrumentation at all.

The measured A/B (results/compute_cost) says the COGNITIVE LAYER is
free: uil_enabled ON vs OFF is 519s vs 547s total, i.e. 0.95x. Both arms run
the same 20 stages of the same search, so that comparison isolates cognition
and cognition costs nothing.

The 10-100x is therefore a claim about the SEARCH LOOP, not the cognition:
20 stages x k candidates x cv folds of repeated fitting, versus one ordinary
`.fit()`. This script measures that denominator so the ratio can be stated
against something real instead of asserted.

Run:  python -m figpso_uil.experiments.compute_baseline
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from figpso_uil.adapters.symbolic_formula_adapter import SymbolicFormulaAdapter

DATASETS = ("breast_cancer", "wine", "ionosphere")


def _load(name):
    if name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer(); return np.asarray(d.data, float), np.asarray(d.target, int)
    if name == "wine":
        from sklearn.datasets import load_wine
        d = load_wine(); return np.asarray(d.data, float), np.asarray(d.target, int)
    if name == "ionosphere":
        import pandas as pd
        from sklearn.datasets import fetch_openml
        from sklearn.preprocessing import LabelEncoder
        b = fetch_openml(data_id=59, as_frame=True, parser="auto")
        df = b.frame.copy()
        tcol = b.target_names[0] if b.target_names else df.columns[-1]
        y = LabelEncoder().fit_transform(df[tcol].astype(str))
        X = pd.get_dummies(df.drop(columns=[tcol]), drop_first=False).to_numpy(float)
        m = np.isfinite(X).all(axis=1)
        return X[m], np.asarray(y, np.int64)[m]
    raise ValueError(name)


def time_one(kind, Xtr, ytr, Xte, yte, seed, n_classes):
    t0 = time.perf_counter()
    if kind == "symbolic_single_fit":
        clf = SymbolicFormulaAdapter(seed=seed, num_classes=n_classes,
                                     learning_mode="sparse_linear")
        m = clf.fit_weighted(Xtr, ytr, np.ones(Xtr.shape[1]))
        pred = clf.predict(m, Xte)
    elif kind == "xgboost_single_fit":
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                          random_state=seed, verbosity=0)
        m.fit(Xtr, ytr); pred = m.predict(Xte)
    elif kind == "logreg_single_fit":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=2000, random_state=seed)
        m.fit(Xtr, ytr); pred = m.predict(Xte)
    else:
        raise ValueError(kind)
    el = time.perf_counter() - t0
    return el, float(accuracy_score(yte, pred))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,7,123")
    ap.add_argument("--json", default=None)
    ap.add_argument("--uil-json", default=None,
                    help="eval_crossseed --json output to pair against")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    kinds = ("symbolic_single_fit", "xgboost_single_fit", "logreg_single_fit")
    rows, per_seed = [], []
    for name in DATASETS:
        X, y = _load(name)
        n_classes = int(len(np.unique(y)))
        acc_t = {k: [] for k in kinds}
        acc_a = {k: [] for k in kinds}
        for s in seeds:
            # SAME split as eval_crossseed (test_size .2, stratified, seed as
            # random_state), so the comparison against the UIL run is PAIRED.
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=s, stratify=y)
            for k in kinds:
                el, a = time_one(k, Xtr, ytr, Xte, yte, s, n_classes)
                acc_t[k].append(el); acc_a[k].append(a)
                per_seed.append({"dataset": name, "seed": s, "learner": k,
                                 "acc": a, "fit_s": el})
        for k in kinds:
            rows.append({"dataset": name, "learner": k,
                         "fit_s": float(np.mean(acc_t[k])),
                         "acc": float(np.mean(acc_a[k]))})
        print(f"{name}: " + "  ".join(
            f"{k.split('_')[0]}={np.mean(acc_t[k]):.3f}s/{np.mean(acc_a[k]):.4f}"
            for k in kinds), flush=True)

    print("\n" + "=" * 78)
    print("ONE ORDINARY FIT vs THE UIL SEARCH LOOP")
    print("=" * 78)
    # Measured UIL wall-clock per (dataset, seed) from results/compute_cost.
    uil_fit_s = {"breast_cancer": 43.9, "wine": 64.3, "ionosphere": 64.8}
    print(f"{'dataset':<15}{'1 symbolic fit':>16}{'UIL 20 stages':>16}{'ratio':>10}")
    print("-" * 57)
    ratios = []
    for name in DATASETS:
        base = next(r for r in rows if r["dataset"] == name
                    and r["learner"] == "symbolic_single_fit")["fit_s"]
        u = uil_fit_s[name]
        ratios.append(u / base)
        print(f"{name:<15}{base:>15.3f}s{u:>15.1f}s{u / base:>10.0f}x")
    print(f"\nmean ratio vs one symbolic fit: {np.mean(ratios):.0f}x")
    print("This -- not the cognitive layer -- is what the '10-100x compute'")
    print("line is describing. The cognitive layer itself measured 0.95x")
    print("(uil_enabled ON vs OFF, same 20 stages): see results/compute_cost.")

    # ---- paired accuracy: one plain fit vs the whole UIL run --------------
    # The compute ratio only matters if the compute buys something. Pair by
    # (dataset, seed) against the UIL per-seed rows, and apply the same
    # interval + power discipline the Gate 2 re-scoring established -- an
    # eyeballed 3x3 table is exactly what that re-scoring found wanting.
    paired = None
    if args.uil_json:
        try:
            from figpso_uil.experiments.rescore_gates import analyze
            with open(args.uil_json, encoding="utf-8") as fh:
                uil = json.load(fh)
            uil_map = {(r["dataset"], int(r["seed"])): float(r["acc"])
                       for r in uil.get("per_seed", [])}
            base_map = {(r["dataset"], int(r["seed"])): float(r["acc"])
                        for r in per_seed if r["learner"] == "symbolic_single_fit"}
            paired = analyze("UIL 20 stages - one plain symbolic fit",
                             base_map, uil_map,
                             "positive = the search loop earned its compute")
            print("\n" + "=" * 78)
            print("DOES THE COMPUTE BUY ACCURACY? (paired by dataset x seed)")
            print("=" * 78)
            if paired["n_cells"] >= 2:
                print(f"  cells            : {paired['n_cells']}")
                print(f"  mean delta       : {paired['mean_delta']:+.4f} "
                      f"(UIL minus one plain fit)")
                print(f"  95% CI           : [{paired['ci95'][0]:+.4f}, "
                      f"{paired['ci95'][1]:+.4f}]")
                print(f"  win/tie/lose     : "
                      f"{'/'.join(str(v) for v in paired['win_tie_lose'])}")
                print(f"  cells for 0.01   : {paired['n_needed_for_0.01']} "
                      f"(ran {paired['n_cells']})")
                print(f"  verdict          : {paired['verdict']}")
                for d, v in sorted(paired["per_dataset"].items()):
                    print(f"    {d:<15} {v:+.4f}")
        except Exception as exc:
            print(f"\n(paired comparison unavailable: {type(exc).__name__}: {exc})")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows, "per_seed": per_seed,
                       "uil_fit_s": uil_fit_s,
                       "ratios_vs_symbolic_single": ratios,
                       "paired_vs_uil": paired}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
