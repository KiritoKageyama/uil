"""Experiment #1 — does principle mining improve HELD-OUT accuracy?

Reuses the ablation harness's _run_once (which carves a never-touched
15% held-out set BEFORE fit and scores the final model on it exactly
once). Runs only Base vs +Principles on real_spambase at a long enough
horizon that the miner actually accumulates principles, across several
seeds. Reports peak-accuracy (biased) next to held-out (honest) so the
gap is visible, plus how many principles each +Principles run mined and
how many ever fired.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import statistics
import sys
import time

from figpso_uil.experiments.ablation_harness import _run_once
from figpso_uil.cognition.paths import results_path

STAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 150
DATASET = sys.argv[2] if len(sys.argv) > 2 else "real_spambase"
CLASSIFIER = sys.argv[3] if len(sys.argv) > 3 else "XGBoost"
SEEDS = [1, 2, 3]
N_PARTICLES = 8
PSO_ITERS = 8
BOOSTING = 30

STEPS = {
    "Base":        {"principles": False, "network": False, "formulas": False, "causal": False},
    "+Principles": {"principles": True,  "network": False, "formulas": False, "causal": False},
}


def main() -> None:
    print(f"Experiment: {DATASET}  {CLASSIFIER}  stages={STAGES} "
          f"particles={N_PARTICLES} pso_iters={PSO_ITERS} seeds={SEEDS}")
    print("=" * 74)
    results = {step: [] for step in STEPS}
    t0 = time.perf_counter()
    for step, ablation in STEPS.items():
        for seed in SEEDS:
            r = _run_once(DATASET, seed, ablation, CLASSIFIER,
                          STAGES, N_PARTICLES, PSO_ITERS, BOOSTING)
            results[step].append(r)
            mined = max(0, r['n_principles'] - 8)  # 8 seed principles bootstrap every run
            print(f"  {step:12s} seed={seed}  peak={r['accuracy']:.3f}  "
                  f"heldout={r['held_out_accuracy']:.3f}  "
                  f"principles={r['n_principles']} (mined~{mined})  "
                  f"cpu={r['cpu_seconds']:.0f}s")

    def m(step, key):
        return statistics.fmean(x[key] for x in results[step])

    base_ho = [x["held_out_accuracy"] for x in results["Base"]]
    prin_ho = [x["held_out_accuracy"] for x in results["+Principles"]]
    diffs = [p - b for b, p in zip(base_ho, prin_ho)]
    mean_diff = statistics.fmean(diffs)

    print("\n" + "=" * 74)
    print("SUMMARY (held-out is the honest number)")
    print(f"  Base         peak={m('Base','accuracy'):.3f}  "
          f"heldout={m('Base','held_out_accuracy'):.3f}  "
          f"principles={m('Base','n_principles'):.0f}")
    print(f"  +Principles  peak={m('+Principles','accuracy'):.3f}  "
          f"heldout={m('+Principles','held_out_accuracy'):.3f}  "
          f"principles={m('+Principles','n_principles'):.0f}")
    print(f"\n  held-out delta (+Principles - Base): {mean_diff:+.4f}")
    print(f"  per-seed held-out diffs: {[round(d,3) for d in diffs]}")

    try:
        from scipy import stats as st
        if len(diffs) >= 2 and statistics.pstdev(diffs) > 1e-9:
            p = float(st.ttest_rel(prin_ho, base_ho).pvalue)
            print(f"  paired t-test p={p:.4f}" + ("  (significant)" if p < 0.05 else "  (not significant)"))
        else:
            print("  paired t-test: degenerate (no variance in diffs)")
    except Exception as e:
        print("  paired t-test unavailable:", e)

    print(f"\n  total wall clock: {time.perf_counter()-t0:.0f}s")

    out = results_path(f"ablation_reports/exp1_principles_heldout_{CLASSIFIER}_{DATASET}_{STAGES}s.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"dataset": DATASET, "stages": STAGES, "seeds": SEEDS,
                   "results": results, "mean_heldout_diff": mean_diff}, f, indent=2, default=str)
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
