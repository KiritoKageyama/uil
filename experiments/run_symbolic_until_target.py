"""Run SymbolicFormula on a binary dataset until target accuracy or max stages.

Usage (from Downloads/, parent of figpso_uil):
    python -m figpso_uil.experiments.run_symbolic_until_target
    python -m figpso_uil.experiments.run_symbolic_until_target --dataset real_sonar --target 0.95
"""

from __future__ import annotations
import argparse
import os
import time

os.environ.setdefault("UIL_LIVE_FEED", "1")

from figpso_uil.core.optimizer import UniversalOptimizer
from figpso_uil.adapters.symbolic_formula_adapter import SymbolicFormulaAdapter
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch
from figpso_uil.webui.backend.datasets import load_dataset


class Cfg:
    def __init__(self, stages, target, max_stages, n_particles=15, pso_iters=15):
        self.stages = stages
        self.n_particles = n_particles
        self.pso_iters = pso_iters
        self.particles = n_particles
        self.pso_iters_per_stage = pso_iters
        self.target_accuracy = target
        self.max_stages = max_stages


def main():
    parser = argparse.ArgumentParser(description="SymbolicFormula run-until-target")
    parser.add_argument("--dataset", default="real_sonar")
    parser.add_argument("--target", type=float, default=0.95)
    parser.add_argument("--max-stages", type=int, default=500)
    parser.add_argument("--stages", type=int, default=40, help="Minimum planned stages (floor for max)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-particles", type=int, default=15)
    parser.add_argument("--pso-iters", type=int, default=15)
    args = parser.parse_args()

    X, y = load_dataset(args.dataset, seed=args.seed)
    n_classes = len(set(y.tolist()))
    print(f"Loaded {args.dataset}: {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes")

    opt = UniversalOptimizer(
        cfg=Cfg(args.stages, args.target, args.max_stages, args.n_particles, args.pso_iters),
        seed=args.seed,
        dataset_name=args.dataset,
        fold=0,
        start_fold=0,
        classifier=SymbolicFormulaAdapter(seed=args.seed, num_classes=n_classes),
        search=PrincipleGuidedSearch(),
    )
    opt._observability_meta.update({
        "source": "terminal",
        "script": "run_symbolic_until_target.py",
    })

    print("=" * 60)
    print(f"SymbolicFormula · {args.dataset} · target {args.target*100:.0f}% · cap {args.max_stages} stages")
    print("=" * 60)

    t0 = time.perf_counter()
    result = opt.fit(X, y)
    elapsed = time.perf_counter() - t0

    print(f"accuracy:     {result['accuracy']*100:.2f}%")
    print(f"target hit:   {result['target_reached']}")
    print(f"stages run:   {result['stages_run']}")
    print(f"features sel: {result['n_selected']}")
    print(f"principles:   {result['principles_discovered']}")
    print(f"max stress:   {result['max_stress']:.3f}")
    print(f"time:         {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
