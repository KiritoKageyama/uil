"""Multi-metric 'reasoned vs bulldozer' benchmark.

Instead of judging the cognitive layer on peak accuracy alone (the one
axis where a strong base learner wins for free), this measures the WHOLE
picture, Base vs +Principles, across:

  ACCURACY      peak_val (biased), held_out (honest, never-touched set)
  EFFICIENCY    cpu_seconds, stages_to_90pct (convergence speed),
                heldout_per_1k_cpu (accuracy bought per unit compute)
  SAMPLE-EFF    same metrics but run in TINY-data regimes where a
                brute-force learner overfits and reasoning has room to win
  ROBUSTNESS    cross-seed std of held_out (lower = steadier)
  REASONING     n_mined_principles, mean/final regret, mean model_complexity

Held-out principle gate is forced ON (UIL_HELDOUT_GATE=1) so principles
are credited only for improving generalization.

Usage:
    python -m figpso_uil.experiments.exp_multimetric_benchmark
    python -m figpso_uil.experiments.exp_multimetric_benchmark --stages 150 --seeds 1,2,3 --classifier SymbolicFormula
"""
from __future__ import annotations

import os
os.environ["UIL_HELDOUT_GATE"] = "1"          # force the honest promotion gate
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import json
import statistics
import tempfile
import time
import shutil
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from figpso_uil.webui.backend import datasets as dataset_registry
from figpso_uil.core.classifier_registry import CLASSIFIERS
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch
from figpso_uil.core.optimizer import UniversalOptimizer
from figpso_uil.cognition.paths import results_path

STEPS = {
    "Base":        {"principles": False, "network": False, "formulas": False, "causal": False},
    "+Principles": {"principles": True,  "network": False, "formulas": False, "causal": False},
}

# Regimes: (label, n_rows or None for full). Tiny first -- that's where a
# bulldozer overfits and reasoning has the most room to matter.
REGIMES = [("tiny_60", 60), ("small_150", 150), ("mid_400", 400)]


class _Cfg:
    def __init__(self, stages, n_particles, pso_iters):
        self.stages = stages
        self.n_particles = n_particles
        self.pso_iters = pso_iters
        self.particles = n_particles
        self.pso_iters_per_stage = pso_iters
        self.use_resource_budget = False   # keep budget fixed for fair compute comparison


def _subsample(X, y, n, seed):
    if n is None or n >= len(X):
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, random_state=seed, stratify=y)
    return Xs, ys


def _run_once(dataset, n_rows, seed, ablation, classifier_name, stages, n_particles, pso_iters, boosting):
    X_full, y_full = dataset_registry.load_dataset(dataset, seed=seed)
    X_full, y_full = _subsample(X_full, y_full, n_rows, seed)
    X, X_ho, y, y_ho = train_test_split(X_full, y_full, test_size=0.2, random_state=seed, stratify=y_full)
    n_classes = len(np.unique(y))
    sd = tempfile.mkdtemp(prefix="uil_mm_")
    clf = CLASSIFIERS[classifier_name](seed, n_classes, boosting, state_dir=sd)
    if hasattr(clf, "params") and "n_jobs" in clf.params:
        clf.params["n_jobs"] = 1
    cfg = _Cfg(stages, n_particles, pso_iters)
    try:
        opt = UniversalOptimizer(cfg=cfg, seed=seed, dataset_name=dataset, fold=0, start_fold=0,
                                 classifier=clf, search=PrincipleGuidedSearch(),
                                 ablation=ablation, state_dir=sd)
        t0 = time.perf_counter()
        res = opt.fit(X, y)
        cpu = time.perf_counter() - t0
        ho_acc = float(accuracy_score(y_ho, opt.predict(X_ho)))

        ents = opt.journal.get_dataset_entries(dataset)
        accs = [(e.get("stage", i), e.get("outcome", {}).get("accuracy", 0.0)) for i, e in enumerate(ents)]
        accs.sort()
        peak = res["accuracy"]
        # convergence: first stage reaching 90% of final peak
        target = 0.9 * peak
        stages_to_90 = next((s for s, a in accs if a >= target), len(accs))
        regrets = [e["regret"] for e in ents if "regret" in e]
        complexities = [e.get("outcome", {}).get("model_complexity", 0) for e in ents]

        return {
            "peak_val": peak,
            "held_out": ho_acc,
            "cpu_seconds": cpu,
            "stages_to_90pct": stages_to_90,
            "heldout_per_1k_cpu": ho_acc / cpu * 1000.0 if cpu > 0 else 0.0,
            "n_mined_principles": max(0, len(opt.library.get_principles()) - 8),
            "mean_regret": statistics.fmean(regrets) if regrets else 0.0,
            "final_regret": regrets[-1] if regrets else 0.0,
            "mean_complexity": statistics.fmean(complexities) if complexities else 0.0,
        }
    finally:
        shutil.rmtree(sd, ignore_errors=True)


METRICS = ["peak_val", "held_out", "cpu_seconds", "stages_to_90pct",
           "heldout_per_1k_cpu", "n_mined_principles", "mean_regret",
           "final_regret", "mean_complexity"]
# For these, LOWER is better -- so the +Principles "win" direction flips.
LOWER_BETTER = {"cpu_seconds", "stages_to_90pct", "mean_regret", "final_regret", "mean_complexity"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="real_spambase")
    ap.add_argument("--classifier", default="SymbolicFormula", choices=list(CLASSIFIERS))
    ap.add_argument("--stages", type=int, default=120)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--n-particles", type=int, default=8)
    ap.add_argument("--pso-iters", type=int, default=8)
    ap.add_argument("--boosting", type=int, default=30)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print(f"Multi-metric benchmark  {args.dataset}  {args.classifier}  "
          f"stages={args.stages}  seeds={seeds}  (held-out gate ON)")
    report = {"generated_at": datetime.now().isoformat(), "dataset": args.dataset,
              "classifier": args.classifier, "stages": args.stages, "seeds": seeds, "regimes": {}}
    t_start = time.perf_counter()

    for label, n_rows in REGIMES:
        print("\n" + "=" * 78)
        print(f"REGIME: {label}  (n_rows={n_rows})")
        print("=" * 78)
        per_step = {}
        for step, ablation in STEPS.items():
            runs = [_run_once(args.dataset, n_rows, s, ablation, args.classifier,
                              args.stages, args.n_particles, args.pso_iters, args.boosting)
                    for s in seeds]
            per_step[step] = runs

        def agg(step, m):
            return statistics.fmean(r[m] for r in per_step[step])

        def std(step, m):
            vals = [r[m] for r in per_step[step]]
            return statistics.pstdev(vals) if len(vals) > 1 else 0.0

        header = f"{'metric':22s} {'Base':>12s} {'+Principles':>14s} {'delta':>12s}  verdict"
        print(header)
        print("-" * len(header))
        regime_rec = {"n_rows": n_rows, "Base": per_step["Base"], "+Principles": per_step["+Principles"], "summary": {}}
        for m in METRICS:
            b, p = agg("Base", m), agg("+Principles", m)
            d = p - b
            better = (d < 0) if m in LOWER_BETTER else (d > 0)
            verdict = "WIN " if abs(d) > 1e-9 and better else ("lose" if abs(d) > 1e-9 else "tie ")
            print(f"{m:22s} {b:12.4f} {p:14.4f} {d:+12.4f}  {verdict}")
            regime_rec["summary"][m] = {"base": b, "principles": p, "delta": d, "principles_better": better}
        # robustness: cross-seed std of held_out (lower = steadier)
        rb_b, rb_p = std("Base", "held_out"), std("+Principles", "held_out")
        print(f"{'heldout_std(robust)':22s} {rb_b:12.4f} {rb_p:14.4f} {rb_p-rb_b:+12.4f}  "
              f"{'WIN ' if rb_p < rb_b else 'lose'}")
        regime_rec["summary"]["heldout_std"] = {"base": rb_b, "principles": rb_p}
        report["regimes"][label] = regime_rec

    report["wall_clock_seconds"] = time.perf_counter() - t_start
    out = results_path(f"ablation_reports/multimetric_{args.classifier}_{args.dataset}_{args.stages}s.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nwall clock: {report['wall_clock_seconds']:.0f}s   saved: {out}")


if __name__ == "__main__":
    main()
