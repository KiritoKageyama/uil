"""Phase C: replicate the Phase A/B diversity diagnostic across 5 seeds for
BOTH the old (pre-stratification) and new (stratified) mutation-parent
selection, to tell whether the structural-diversity dip seen in the n=1
Phase B comparison is a real effect or sampling noise. Diagnostics only --
no accuracy benchmarking. The "before" condition is reproduced via an
in-process monkeypatch of FormulaLibrary.select_mutation_parent (the exact
pre-fix individual-uniform-tournament logic) so the real source code is
never touched/reverted.
"""
import os

for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_env_var, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SEEDS = [10, 20, 30, 40, 50]
STAGES = 300
DATASET = "brutal_60x120"


def _old_select_mutation_parent(self, rng, k: int = 4, progress: float = 0.0):
    """Verbatim pre-V20+.17 logic: individual-uniform tournament, no
    lineage stratification. Monkeypatched in for the "before" condition."""
    pool = [f for f in self._formulas.values() if f["status"] in ("active", "seed", "incubating")]
    if not pool:
        return None
    k = min(k, len(pool))
    tournament = [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]

    def fitness(f):
        evidence_score = min(1.0, f["evidence"] / self.PARENT_EVIDENCE_SATURATION)
        diversity_bonus = 1.0 / (1.0 + f.get("times_used_as_parent", 0))
        return 0.5 * f["confidence"] + 0.3 * evidence_score + 0.2 * diversity_bonus

    winner = max(tournament, key=fitness)
    winner["times_used_as_parent"] = winner.get("times_used_as_parent", 0) + 1
    self._save()
    return winner


def _run_one(seed: int, mode: str, base_tmp: str) -> dict:
    for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                      "NUMEXPR_NUM_THREADS"):
        os.environ[_env_var] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from figpso_uil.webui.backend import datasets
    from figpso_uil.core.classifier_registry import CLASSIFIERS
    from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch
    from figpso_uil.core.optimizer import UniversalOptimizer
    from figpso_uil.cognition.formula_library import FormulaLibrary
    from figpso_uil.experiments.formula_diagnostics import analyze

    if mode == "before":
        FormulaLibrary.select_mutation_parent = _old_select_mutation_parent

    class _Cfg:
        def __init__(self, stages, n_particles, pso_iters):
            self.stages = stages
            self.n_particles = n_particles
            self.pso_iters = pso_iters
            self.particles = n_particles
            self.pso_iters_per_stage = pso_iters

    state_dir = os.path.join(base_tmp, f"{mode}_seed{seed}")
    os.makedirs(state_dir, exist_ok=True)

    X, y = datasets.load_dataset(DATASET, seed=seed)
    num_classes = len(set(y.tolist()))
    classifier = CLASSIFIERS["XGBoost"](seed, num_classes, 30)
    if hasattr(classifier, "params") and "n_jobs" in classifier.params:
        classifier.params["n_jobs"] = 1
    search = PrincipleGuidedSearch()
    cfg = _Cfg(stages=STAGES, n_particles=20, pso_iters=10)

    optimizer = UniversalOptimizer(
        cfg=cfg, seed=seed, dataset_name=DATASET,
        fold=0, start_fold=0, classifier=classifier, search=search,
        state_dir=state_dir,
    )
    optimizer.fit(X, y)

    lib_path = os.path.join(state_dir, "uil_formula_library.json")
    result = analyze(lib_path)
    return {"seed": seed, "mode": mode, **result}


def main() -> None:
    base_tmp = tempfile.mkdtemp(prefix="uil_phase_c_")
    print(f"base_tmp = {base_tmp}")
    jobs = [(seed, mode) for mode in ("before", "after") for seed in SEEDS]

    results = []
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 4)) as ex:
        futures = {ex.submit(_run_one, seed, mode, base_tmp): (seed, mode) for seed, mode in jobs}
        for fut in as_completed(futures):
            seed, mode = futures[fut]
            try:
                r = fut.result()
                print(f"done: seed={seed:<3} mode={mode:<6} "
                      f"lineage_entropy_norm={r['lineage_entropy_normalized']:.3f} "
                      f"shape_ratio={r['structural']['shape_diversity_ratio']:.3f}")
                results.append(r)
            except Exception as e:
                print(f"FAILED: seed={seed} mode={mode}: {type(e).__name__}: {e}")

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "ablation_reports",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase_c_seed_sweep_2026-06-28.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
