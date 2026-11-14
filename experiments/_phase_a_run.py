"""One-off script: regenerate a fresh formula-population snapshot under the
CURRENT code (post tournament-selection fix, pre-stratification) so Phase A
diagnostics have real data to measure. Isolated state_dir -- never touches
the canonical results/ files. Mirrors the webui run that produced the
96-formula screenshot: brutal_60x120, XGBoost, 300 stages, seed 50.
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
    state_dir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="uil_phase_a_")
    os.makedirs(state_dir, exist_ok=True)
    print(f"state_dir = {state_dir}")

    seed = 50
    X, y = datasets.load_dataset("brutal_60x120", seed=seed)
    num_classes = len(set(y.tolist()))
    classifier = CLASSIFIERS["XGBoost"](seed, num_classes, 30)
    search = PrincipleGuidedSearch()
    cfg = _Cfg(stages=300, n_particles=20, pso_iters=10)

    optimizer = UniversalOptimizer(
        cfg=cfg, seed=seed, dataset_name="brutal_60x120",
        fold=0, start_fold=0, classifier=classifier, search=search,
        state_dir=state_dir,
    )
    result = optimizer.fit(X, y)
    print("done:", {k: v for k, v in result.items() if k not in ("model", "feature_weights")})
    print(f"formula library written to: {os.path.join(state_dir, 'uil_formula_library.json')}")


if __name__ == "__main__":
    main()
