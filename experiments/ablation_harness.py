"""Ablation harness — measures what each UIL cognition subsystem
actually contributes, instead of inferring it from a single run.

Runs the SAME dataset/seed/PSO-budget through UniversalOptimizer with
principles/network/formula-evolution/causal toggled on incrementally
(Base -> +Principles -> +Formula Evolution -> +Network -> +Causal),
each run getting its own isolated state_dir (core/optimizer.py's
ablation/state_dir params -- see UniversalOptimizer.__init__) so one
arm's persisted state can never leak into another's. Every step is
repeated across multiple seeds so the report carries mean +/- std, not
a single lucky/unlucky run.

Usage (from Downloads/, matching the rest of this package's import
convention):
    python -m figpso_uil.experiments.ablation_harness
    python -m figpso_uil.experiments.ablation_harness --datasets hard_100x80,brutal_60x120 --seeds 1,2,3,4,5 --stages 150
"""

from __future__ import annotations

# V20+.18: must run before numpy/scipy/sklearn/xgboost are imported
# anywhere in this process -- their BLAS/OpenMP thread pools size
# themselves (to all visible cores, by default) the FIRST time they're
# touched, and that sizing is then inherited by every worker forked or
# spawned afterward. Combined with N worker PROCESSES each separately
# trying to grab all cores for its own BLAS calls, this causes severe
# oversubscription -- confirmed empirically on a shared HPC login node
# (4 worker processes each pegged at 1300%+ CPU, ~60+ cores consumed by
# what was meant to be a tiny smoke test, with the job never finishing
# due to thrashing). Pin every thread pool to 1: the only parallelism
# this harness wants is at the process level (--workers), one worker
# per requested CPU.
import os
for _env_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

# V20+.19: this harness never asks for a GPU (XGBoostAdapter always
# uses tree_method="hist") -- but the pip xgboost wheel still ships
# CUDA-aware code that can probe/touch a node's GPUs regardless, and a
# job that didn't request any ngpus via PBS gets no cgroup GPU
# isolation, so it sees every physical GPU on the node, including
# whichever ones other users' jobs have already filled. Confirmed
# empirically on a shared HPC node: a 300-stage sweep crashed with
# "cudaError 2" (out of memory) from inside XGBoost's internal
# DMatrix/array_interface code, purely from sharing GPU visibility with
# another tenant's job -- never from anything this code itself
# requested. Hiding the devices entirely makes that whole class of
# failure structurally impossible instead of just unlikely.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import json
import math
import shutil
import statistics
import tempfile
import time
from datetime import datetime, timezone

import numpy as np

from figpso_uil.webui.backend import datasets as dataset_registry
from figpso_uil.core.classifier_registry import CLASSIFIERS
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch
from figpso_uil.core.optimizer import UniversalOptimizer
from figpso_uil.cognition.paths import results_path


# Incremental ablation steps, in the order each subsystem is added back.
# Each step's ablation dict is checked against UniversalOptimizer's
# DEFAULT_ABLATION keys at runtime (see _make_ablation below).
ABLATION_STEPS: list[tuple[str, dict[str, bool]]] = [
    ("Base", {"principles": False, "network": False, "formulas": False, "causal": False}),
    ("+Principles", {"principles": True, "network": False, "formulas": False, "causal": False}),
    ("+Formula Evolution", {"principles": True, "network": False, "formulas": True, "causal": False}),
    ("+Network", {"principles": True, "network": True, "formulas": True, "causal": False}),
    ("+Causal", {"principles": True, "network": True, "formulas": True, "causal": True}),
]


class _Cfg:
    def __init__(self, stages: int, n_particles: int, pso_iters: int):
        self.stages = stages
        self.n_particles = n_particles
        self.pso_iters = pso_iters
        self.particles = n_particles
        self.pso_iters_per_stage = pso_iters


def _run_once(
    dataset_name: str,
    seed: int,
    ablation: dict[str, bool],
    classifier_name: str,
    stages: int,
    n_particles: int,
    pso_iters: int,
    boosting_rounds: int,
) -> dict:
    """One (dataset, seed, ablation-config) trial in its own throwaway
    state_dir -- deleted immediately after, since this is scratch
    experiment state, not research data worth keeping (the report
    below is the artifact worth keeping).

    V20+.19: "accuracy" (UniversalOptimizer.fit()'s own return value)
    is the PEAK validation score any single candidate hit during the
    run -- not an average, but still measured on the same 80/20 split
    used to drive search/selection throughout (optimizer.py's
    _split()), every single stage. With dozens of candidates tried
    against that one split, "best" is partly "whichever one got lucky
    on it" -- a real selection-bias risk that grows with how many
    candidates get tried, which is exactly what the formula engine
    does a lot of. "held_out_accuracy" fixes this: a genuinely
    untouched test set, carved out before fit() ever runs, evaluated
    exactly once at the end against whatever the system's FINAL chosen
    model actually is (optimizer.predict(), classifier-agnostic) --
    the "deploy it and ask for its real answer" number, not a
    mid-search peak. Both numbers are reported; "accuracy" stays for
    continuity with every ablation report generated earlier this
    session, "held_out_accuracy" is the more rigorous one going forward.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X_full, y_full = dataset_registry.load_dataset(dataset_name, seed=seed)
    X, X_heldout, y, y_heldout = train_test_split(
        X_full, y_full, test_size=0.15, random_state=seed, stratify=y_full
    )
    num_classes = len(np.unique(y))
    # state_dir created BEFORE the classifier (not after, as originally
    # written) -- SymbolicFormulaAdapter owns its own FormulaLibrary file
    # and needs this path at construction time to isolate per-trial,
    # exactly like UniversalOptimizer's own state isolation below. Every
    # other classifier ignores the extra arg.
    state_dir = tempfile.mkdtemp(prefix="uil_ablation_")
    classifier = CLASSIFIERS[classifier_name](seed, num_classes, boosting_rounds, state_dir=state_dir)
    # XGBoostAdapter hardcodes n_jobs=-1 (correct for a single foreground
    # run -- see adapters/xgboost_adapter.py -- but this harness runs
    # many of these as separate worker PROCESSES at once; without this,
    # each one independently tries to claim every visible core for its
    # own XGBoost threads on top of the process-level parallelism,
    # causing the oversubscription/thrashing confirmed empirically on
    # the HPC login node above). One worker = one logical unit of work.
    if hasattr(classifier, "params") and "n_jobs" in classifier.params:
        classifier.params["n_jobs"] = 1
    search = PrincipleGuidedSearch()
    cfg = _Cfg(stages=stages, n_particles=n_particles, pso_iters=pso_iters)

    try:
        optimizer = UniversalOptimizer(
            cfg=cfg, seed=seed, dataset_name=dataset_name, fold=0, start_fold=0,
            classifier=classifier, search=search,
            ablation=ablation, state_dir=state_dir,
        )
        t0 = time.perf_counter()
        result = optimizer.fit(X, y)
        cpu_seconds = time.perf_counter() - t0

        # V20+.19: the system's actual final answer (its best checkpoint
        # across the whole run -- standard best-checkpoint deployment,
        # not the last stage's transient model), scored on data the
        # search/selection process never saw. See the long docstring
        # note above for why this differs from "accuracy" below.
        held_out_preds = optimizer.predict(X_heldout)
        held_out_accuracy = float(accuracy_score(y_heldout, held_out_preds))

        entries = optimizer.journal.get_dataset_entries(dataset_name)
        regrets = [e["regret"] for e in entries if "regret" in e]
        net_summary = optimizer.network.get_summary()
        causal_summary = optimizer.causal_engine.get_summary()
        formulas = optimizer.formula_library.get_all()

        return {
            "accuracy": result["accuracy"],
            "held_out_accuracy": held_out_accuracy,
            "max_stress": result["max_stress"],
            "stages_run": result["stages_run"],
            "cpu_seconds": cpu_seconds,
            "final_regret": regrets[-1] if regrets else 0.0,
            "mean_regret": statistics.fmean(regrets) if regrets else 0.0,
            "n_principles": len(optimizer.library.get_principles()),
            "n_active_formulas": len([f for f in formulas if f["status"] == "active"]),
            "n_formula_mutations": len(
                [f for f in formulas if f["origin"].startswith("mutation_of_")]
            ),
            "network_nodes": net_summary["nodes"],
            "network_edges": net_summary["edges"],
            "causal_total_hypotheses": causal_summary["total_hypotheses"],
            "causal_confirmed": len(causal_summary["confirmed_causal"]),
        }
    finally:
        shutil.rmtree(state_dir, ignore_errors=True)


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def _ci95(mean: float, std: float, n: int) -> float:
    """Half-width of a 95% CI for the mean (normal approximation)."""
    if n <= 1:
        return 0.0
    return 1.96 * std / math.sqrt(n)


METRIC_KEYS = [
    "accuracy", "held_out_accuracy", "mean_regret", "final_regret", "max_stress", "cpu_seconds",
    "n_principles", "n_active_formulas", "n_formula_mutations",
    "network_nodes", "network_edges", "causal_total_hypotheses", "causal_confirmed",
]


def _paired_comparison(base_values: list[float], step_values: list[float]) -> dict:
    """Paired comparison of one metric, step vs Base, matched by seed
    (every step reuses the same seed list against the same dataset --
    this is a paired design, not independent samples, so a paired t-test
    / Wilcoxon and a paired effect size (Cohen's d_z) are the right
    tools, not a two-sample test)."""
    from scipy import stats as scipy_stats

    diffs = [s - b for b, s in zip(base_values, step_values)]
    mean_diff, std_diff = _mean_std(diffs)
    # A near-zero (but not exactly zero, due to float drift) std_diff
    # makes mean_diff / std_diff blow up to a meaningless huge number
    # instead of correctly being "no variance to normalize by" --
    # confirmed empirically (Cohen's d_z of 9e14 on a 2-seed smoke test
    # where both diffs were +0.05). 1e-9 is well below any real
    # accuracy/regret variance this harness will ever measure.
    cohens_dz = mean_diff / std_diff if std_diff > 1e-9 else None
    base_mean = statistics.fmean(base_values)
    relative_pct = (mean_diff / base_mean * 100.0) if base_mean != 0 else None

    p_ttest = None
    p_wilcoxon = None
    # Same degenerate-variance guard as cohens_dz above -- ttest_rel's
    # t-statistic also divides by std_diff, so near-zero variance
    # produces an artificially tiny (falsely "significant") p-value
    # rather than a meaningful one.
    if len(diffs) >= 2 and std_diff > 1e-9:
        try:
            p_ttest = float(scipy_stats.ttest_rel(step_values, base_values).pvalue)
        except Exception:
            p_ttest = None
        try:
            p_wilcoxon = float(scipy_stats.wilcoxon(diffs).pvalue)
        except Exception:
            # Wilcoxon is undefined when every pair is tied (all diffs == 0)
            p_wilcoxon = None

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "cohens_dz": cohens_dz,
        "relative_pct": relative_pct,
        "p_paired_ttest": p_ttest,
        "p_wilcoxon": p_wilcoxon,
    }


def _build_specs(
    dataset_names: list[str],
    seeds: list[int],
    classifier_name: str,
    stages: int,
    n_particles: int,
    pso_iters: int,
    boosting_rounds: int,
) -> list[dict]:
    """Flatten every (dataset, ablation-step, seed) combination into one
    list of independent work items -- each is a standalone _run_once()
    call with its own throwaway state_dir, so they're safe to execute in
    any order, on any worker, in parallel."""
    specs = []
    for dataset_name in dataset_names:
        for step_name, ablation in ABLATION_STEPS:
            for seed in seeds:
                specs.append({
                    "dataset_name": dataset_name, "step_name": step_name,
                    "ablation": ablation, "seed": seed,
                    "classifier_name": classifier_name, "stages": stages,
                    "n_particles": n_particles, "pso_iters": pso_iters,
                    "boosting_rounds": boosting_rounds,
                })
    return specs


def _run_spec(spec: dict) -> dict:
    """Picklable top-level wrapper around _run_once() for
    ProcessPoolExecutor -- tags the result with which (dataset, step,
    seed) it belongs to, since parallel execution doesn't preserve the
    nested dataset/step/seed loop structure of a sequential run."""
    result = _run_once(
        spec["dataset_name"], spec["seed"], spec["ablation"], spec["classifier_name"],
        spec["stages"], spec["n_particles"], spec["pso_iters"], spec["boosting_rounds"],
    )
    return {
        "dataset_name": spec["dataset_name"],
        "step_name": spec["step_name"],
        "seed": spec["seed"],
        "result": result,
    }


def _aggregate_step(
    step_name: str,
    ablation: dict[str, bool],
    per_seed_runs: list[dict],
    base_raw: list[dict] | None,
) -> dict:
    """Build one step's aggregated record (mean/std/CI95 per metric,
    plus paired comparison vs Base when base_raw is given)."""
    stats = {}
    for key in METRIC_KEYS:
        values = [r[key] for r in per_seed_runs]
        mean, std = _mean_std(values)
        stats[key] = {"mean": mean, "std": std, "ci95": _ci95(mean, std, len(values))}

    record = {
        "step": step_name,
        "ablation": ablation,
        "n_seeds": len(per_seed_runs),
        "stats": stats,
        "raw_runs": per_seed_runs,
    }

    if base_raw is not None:
        record["vs_base"] = {
            "accuracy": _paired_comparison(
                [r["accuracy"] for r in base_raw],
                [r["accuracy"] for r in per_seed_runs],
            ),
            "mean_regret": _paired_comparison(
                [r["mean_regret"] for r in base_raw],
                [r["mean_regret"] for r in per_seed_runs],
            ),
            "cpu_relative": (
                statistics.fmean(r["cpu_seconds"] for r in per_seed_runs)
                / statistics.fmean(r["cpu_seconds"] for r in base_raw)
                if statistics.fmean(r["cpu_seconds"] for r in base_raw) > 0
                else None
            ),
        }
    return record


def run_ablation_parallel(
    dataset_names: list[str],
    seeds: list[int],
    classifier_name: str,
    stages: int,
    n_particles: int,
    pso_iters: int,
    boosting_rounds: int,
    workers: int = 1,
) -> dict[str, list[dict]]:
    """Run every (dataset, ablation-step, seed) combination, optionally
    spread across `workers` worker processes -- this is what lets a
    single PBS/Slurm job request many CPUs and actually use them, since
    a job-level resource limit (e.g. PBS's max_user_run) caps concurrent
    *jobs*, not threads/processes inside one job.

    Returns {dataset_name: [aggregated step record, ...]} in
    ABLATION_STEPS order, same shape run_ablation() used to return for
    one dataset at a time.
    """
    from concurrent.futures import ProcessPoolExecutor

    specs = _build_specs(
        dataset_names, seeds, classifier_name,
        stages, n_particles, pso_iters, boosting_rounds,
    )
    if workers <= 1:
        tagged_results = [_run_spec(s) for s in specs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            tagged_results = list(executor.map(_run_spec, specs))

    # Group by (dataset, step) preserving ABLATION_STEPS order, then
    # within each group preserve the original seed order so vs_base
    # comparisons stay correctly paired by seed index.
    by_dataset_step: dict[tuple[str, str], list[dict]] = {}
    for tagged in tagged_results:
        key = (tagged["dataset_name"], tagged["step_name"])
        by_dataset_step.setdefault(key, []).append(tagged)
    for group in by_dataset_step.values():
        group.sort(key=lambda t: seeds.index(t["seed"]))

    results_by_dataset: dict[str, list[dict]] = {}
    for dataset_name in dataset_names:
        aggregated = []
        base_raw = None
        for step_name, ablation in ABLATION_STEPS:
            per_seed_runs = [
                t["result"] for t in by_dataset_step[(dataset_name, step_name)]
            ]
            record = _aggregate_step(
                step_name, ablation, per_seed_runs,
                base_raw if step_name != "Base" else None,
            )
            if step_name == "Base":
                base_raw = per_seed_runs
            aggregated.append(record)
        results_by_dataset[dataset_name] = aggregated
    return results_by_dataset


def run_ablation(
    dataset_name: str,
    seeds: list[int],
    classifier_name: str,
    stages: int,
    n_particles: int,
    pso_iters: int,
    boosting_rounds: int,
) -> list[dict]:
    """Sequential single-dataset convenience wrapper (kept for quick
    local smoke tests / scripting). See run_ablation_parallel() for the
    multi-dataset, multi-worker version used by main()."""
    aggregated = []
    base_raw = None
    for step_name, ablation in ABLATION_STEPS:
        per_seed_runs = [
            _run_once(
                dataset_name, seed, ablation, classifier_name,
                stages, n_particles, pso_iters, boosting_rounds,
            )
            for seed in seeds
        ]
        stats = {}
        for key in METRIC_KEYS:
            values = [r[key] for r in per_seed_runs]
            mean, std = _mean_std(values)
            stats[key] = {
                "mean": mean, "std": std,
                "ci95": _ci95(mean, std, len(values)),
            }

        record = {
            "step": step_name,
            "ablation": ablation,
            "n_seeds": len(seeds),
            "stats": stats,
            "raw_runs": per_seed_runs,
        }

        # V20+.17: paired comparison vs Base -- effect size + significance
        # for the two outcome metrics (accuracy, regret), plus relative
        # compute cost, so "+1% accuracy at 5x runtime" is visible
        # instead of buried in two separate mean/std blocks.
        if step_name == "Base":
            base_raw = per_seed_runs
        else:
            record["vs_base"] = {
                "accuracy": _paired_comparison(
                    [r["accuracy"] for r in base_raw],
                    [r["accuracy"] for r in per_seed_runs],
                ),
                "mean_regret": _paired_comparison(
                    [r["mean_regret"] for r in base_raw],
                    [r["mean_regret"] for r in per_seed_runs],
                ),
                "cpu_relative": (
                    statistics.fmean(r["cpu_seconds"] for r in per_seed_runs)
                    / statistics.fmean(r["cpu_seconds"] for r in base_raw)
                    if statistics.fmean(r["cpu_seconds"] for r in base_raw) > 0
                    else None
                ),
            }

        aggregated.append(record)
    return aggregated


def _fmt_p(p: float | None) -> str:
    if p is None:
        return "n/a"
    return f"{p:.4f}" + ("*" if p < 0.05 else "")


def print_report(dataset_name: str, aggregated: list[dict]) -> None:
    print("=" * 70)
    print(f"UIL Ablation Report -- {dataset_name}")
    print("=" * 70)
    n_seeds = aggregated[0]["n_seeds"] if aggregated else 0
    if n_seeds < 5:
        print(
            f"WARNING: only {n_seeds} seed(s) -- effect size/significance "
            "below are unreliable at this sample size, treat as a smoke "
            "test, not a result."
        )
    prev_acc = None
    for record in aggregated:
        acc = record["stats"]["accuracy"]
        held_out = record["stats"]["held_out_accuracy"]
        regret = record["stats"]["mean_regret"]
        cpu = record["stats"]["cpu_seconds"]
        delta = "" if prev_acc is None else f"  (delta {acc['mean'] - prev_acc:+.4f})"
        print(f"\n{record['step']}  (n={record['n_seeds']} seeds)")
        print(
            f"  accuracy   {acc['mean']:.4f} +/- {acc['std']:.4f}"
            f"  [95% CI +/-{acc['ci95']:.4f}]{delta}"
        )
        print(
            f"  held_out_accuracy {held_out['mean']:.4f} +/- {held_out['std']:.4f}"
            f"  [95% CI +/-{held_out['ci95']:.4f}]  (final model, never-touched test set)"
        )
        print(f"  mean_regret {regret['mean']:.4f} +/- {regret['std']:.4f}")
        print(f"  cpu_seconds {cpu['mean']:.2f} +/- {cpu['std']:.2f}")
        print(
            f"  principles={record['stats']['n_principles']['mean']:.1f}  "
            f"active_formulas={record['stats']['n_active_formulas']['mean']:.1f}  "
            f"mutations={record['stats']['n_formula_mutations']['mean']:.1f}  "
            f"network=({record['stats']['network_nodes']['mean']:.1f}n/"
            f"{record['stats']['network_edges']['mean']:.1f}e)  "
            f"causal_confirmed={record['stats']['causal_confirmed']['mean']:.1f}"
        )

        # V20+.17: paired comparison vs Base -- effect size, significance,
        # relative compute cost. Only meaningful for non-Base steps.
        vs_base = record.get("vs_base")
        if vs_base is not None:
            acc_cmp = vs_base["accuracy"]
            regret_cmp = vs_base["mean_regret"]
            cpu_rel = vs_base["cpu_relative"]
            dz = acc_cmp["cohens_dz"]
            relative_pct = (
                f" ({acc_cmp['relative_pct']:+.1f}%)"
                if acc_cmp["relative_pct"] is not None else ""
            )
            dz_str = f"{dz:.2f}" if dz is not None else "n/a"
            cpu_rel_str = f"{cpu_rel:.2f}x Base" if cpu_rel is not None else "n/a"
            print(
                f"  vs Base: accuracy {acc_cmp['mean_diff']:+.4f}{relative_pct}"
                f"  Cohen's d_z={dz_str}"
            )
            print(
                f"           paired t-test p={_fmt_p(acc_cmp['p_paired_ttest'])}"
                f"  Wilcoxon p={_fmt_p(acc_cmp['p_wilcoxon'])}"
                f"  (* = p<0.05)"
            )
            print(
                f"           regret {regret_cmp['mean_diff']:+.4f}"
                f"  cpu_relative={cpu_rel_str}"
            )
        prev_acc = acc["mean"]
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", default=",".join(dataset_registry.list_datasets()),
        help="Comma-separated dataset names from the webui registry.",
    )
    parser.add_argument("--seeds", default="1,2,3", help="Comma-separated seeds.")
    parser.add_argument("--classifier", default="XGBoost", choices=list(CLASSIFIERS))
    parser.add_argument("--stages", type=int, default=60)
    parser.add_argument("--n-particles", type=int, default=15)
    parser.add_argument("--pso-iters", type=int, default=15)
    parser.add_argument("--boosting-rounds", type=int, default=30)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Worker processes for parallel execution across "
             "dataset x ablation-step x seed combinations (use this to "
             "actually spend a multi-core PBS/Slurm allocation instead "
             "of running everything on one core).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Where to save the JSON report (default: "
             "results/ablation_reports/<timestamp>.json under the package root).",
    )
    args = parser.parse_args()

    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    full_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classifier": args.classifier,
        "stages": args.stages,
        "n_particles": args.n_particles,
        "pso_iters": args.pso_iters,
        "boosting_rounds": args.boosting_rounds,
        "seeds": seeds,
        "datasets": {},
    }

    t_start = time.perf_counter()
    results_by_dataset = run_ablation_parallel(
        dataset_names, seeds, args.classifier,
        args.stages, args.n_particles, args.pso_iters, args.boosting_rounds,
        workers=args.workers,
    )
    for dataset_name in dataset_names:
        aggregated = results_by_dataset[dataset_name]
        print_report(dataset_name, aggregated)
        full_report["datasets"][dataset_name] = aggregated
    full_report["wall_clock_seconds"] = time.perf_counter() - t_start

    output_path = args.output or results_path(
        f"ablation_reports/ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"Full report saved to {output_path}")


if __name__ == "__main__":
    main()
