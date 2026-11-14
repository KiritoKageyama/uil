"""One-off verification script: run UniversalOptimizer.fit() with the new
SymbolicFormula classifier, isolated state_dir, to confirm the population
actually evolves across stages (not just the standalone adapter smoke test).
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from figpso_uil.webui.backend import datasets
from figpso_uil.core.classifier_registry import CLASSIFIERS
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch
from figpso_uil.core.optimizer import UniversalOptimizer


class _Cfg:
    def __init__(self, stages, n_particles, pso_iters):
        self.stages = stages
        self.n_particles = n_particles
        self.pso_iters = pso_iters
        self.particles = n_particles
        self.pso_iters_per_stage = pso_iters


def main():
    state_dir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="uil_symbolic_")
    os.makedirs(state_dir, exist_ok=True)
    print(f"state_dir = {state_dir}")

    seed = 7
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else "medium_150x40"
    stages = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    X, y = datasets.load_dataset(dataset_name, seed=seed)
    num_classes = len(set(y.tolist()))
    classifier = CLASSIFIERS["SymbolicFormula"](seed, num_classes, 30, state_dir=state_dir)
    search = PrincipleGuidedSearch()
    cfg = _Cfg(stages=stages, n_particles=20, pso_iters=10)

    optimizer = UniversalOptimizer(
        cfg=cfg, seed=seed, dataset_name=dataset_name,
        fold=0, start_fold=0, classifier=classifier, search=search,
        state_dir=state_dir,
    )
    result = optimizer.fit(X, y)
    print("done:", {k: v for k, v in result.items() if k not in ("model", "feature_weights", "selected_features")})
    print(f"symbolic library written to: {os.path.join(state_dir, 'uil_symbolic_classifier_library.json')}")


if __name__ == "__main__":
    main()
