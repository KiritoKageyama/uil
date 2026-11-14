"""Run UIL on all curated REAL datasets — builds memory from benchmarks only.

Usage (from Downloads/, parent of figpso_uil):
    python -m figpso_uil.experiments.run_curated_real
    python -m figpso_uil.experiments.run_curated_real --stages 40 --datasets real_iris,real_wine

Events stream to results/uil_live_feed.jsonl (visible in web UI Terminal Runs tab).
"""

from __future__ import annotations
import argparse
import os
import sys
import time

# Ensure live feed publishes
os.environ.setdefault("UIL_LIVE_FEED", "1")

from figpso_uil.webui.backend.curated_datasets import list_curated, load_curated, describe_curated
from figpso_uil.adapters.xgboost_adapter import XGBoostAdapter
from figpso_uil.adapters.random_forest_adapter import RandomForestAdapter
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch
from figpso_uil.core.optimizer import UniversalOptimizer


class Cfg:
    def __init__(self, stages, n_particles=15, pso_iters=15):
        self.stages = stages
        self.n_particles = n_particles
        self.pso_iters = pso_iters
        self.particles = n_particles
        self.pso_iters_per_stage = pso_iters


def run_one(name: str, stages: int, seed: int) -> dict:
    X, y = load_curated(name, seed=seed)
    n_classes = len(set(y.tolist()))
    # XGBoost custom-formula path is binary-focused; RF handles multi-class
    if n_classes > 2:
        clf = RandomForestAdapter(seed=seed, num_classes=n_classes)
    else:
        clf = XGBoostAdapter(seed=seed, num_classes=n_classes, boosting_rounds=30)
    opt = UniversalOptimizer(
        cfg=Cfg(stages=stages),
        seed=seed,
        dataset_name=name,
        fold=0,
        start_fold=0,
        classifier=clf,
        search=PrincipleGuidedSearch(),
    )
    opt._observability_meta["source"] = "terminal"
    opt._observability_meta["script"] = "run_curated_real.py"

    t0 = time.perf_counter()
    result = opt.fit(X, y)
    elapsed = time.perf_counter() - t0
    lab = opt._observability_meta.get("lab_report")
    return {
        "dataset": name,
        "accuracy": result["accuracy"],
        "n_selected": result["n_selected"],
        "principles": result["principles_discovered"],
        "max_stress": result["max_stress"],
        "seconds": round(elapsed, 1),
        "lab_report": lab,
        "route": getattr(opt, "_route_decision", {}),
    }


def main():
    parser = argparse.ArgumentParser(description="UIL curated real-dataset batch runner")
    parser.add_argument("--stages", type=int, default=40, help="Stages per dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", type=str, default="", help="Comma-separated subset")
    args = parser.parse_args()

    if args.datasets.strip():
        names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    else:
        names = list_curated()

    print("=" * 70)
    print("UIL CURATED REAL-DATASET RUN")
    print("=" * 70)
    for d in describe_curated():
        if d["name"] in names:
            print(f"  {d['name']}: {d.get('label', '')}")
    print(f"Stages: {args.stages}  Seed: {args.seed}")
    print()

    results = []
    for name in names:
        print(f"\n--- {name} ---")
        try:
            r = run_one(name, args.stages, args.seed)
            results.append(r)
            print(f"  acc={r['accuracy']:.4f}  features={r['n_selected']}  "
                  f"principles={r['principles']}  stress={r['max_stress']:.3f}  "
                  f"time={r['seconds']}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"dataset": name, "error": str(e)})

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        if "error" in r:
            print(f"  {r['dataset']}: ERROR — {r['error']}")
        else:
            print(f"  {r['dataset']}: acc={r['accuracy']:.4f}  principles={r['principles']}")
    print("\nMemory written to figpso_uil/results/ — check web UI Memory Proof tab.")
    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
