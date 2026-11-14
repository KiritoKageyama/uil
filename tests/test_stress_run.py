"""Stress Test — Run UIL on a HARD dataset where it will struggle.

Uses a noisy, high-dimensional dataset with weak signals.
Runs enough stages to generate stress, regret, investigations, and principles.
"""

import numpy as np
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from figpso_uil.adapters.xgboost_adapter import XGBoostAdapter
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch
from figpso_uil.core.optimizer import UniversalOptimizer
from figpso_uil.core.failsafe import FailsafeEngine
from figpso_uil.adapters.random_forest_adapter import RandomForestAdapter
from figpso_uil.adapters.knn_adapter import KNNAdapter


def make_hard_dataset(n_samples=80, n_features=100, seed=42):
    """Create a HARD dataset: few samples, many features, lots of noise.
    
    Only 3 features are informative. The rest are pure noise.
    This forces the system to struggle, generating stress and regret.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    
    # Only features 0, 1, 2 are informative (weak signal)
    signal = 0.3 * X[:, 0] + 0.5 * X[:, 1] - 0.2 * X[:, 2]
    noise = rng.normal(0, 0.8, n_samples)  # Heavy noise
    y = (signal + noise > 0).astype(int)
    
    return X, y


class HardCfg:
    stages = 20           # More stages to build stress
    n_particles = 15
    pso_iters = 20
    particles = 15
    pso_iters_per_stage = 20
    boosting_rounds_per_stage = 30


def run_stress_test():
    print("=" * 70)
    print("  UIL V19 STRESS TEST — Hard Dataset")
    print("  80 samples, 100 features, 3 informative, heavy noise")
    print("=" * 70)
    
    # Clean previous results
    journal_path = "results/uil_journal_v19.json"
    memory_path = "results/uil_journal_v19_memory.json"
    principles_path = "results/uil_principles_v19.json"
    
    for path in [journal_path, memory_path, principles_path]:
        if os.path.exists(path):
            os.remove(path)
    
    X, y = make_hard_dataset()
    print(f"\n  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Class distribution: {np.bincount(y)}")
    print(f"  Signal-to-noise: LOW (only 3/100 features informative)\n")
    
    cfg = HardCfg()
    
    # Run 1: XGBoost with PrincipleGuided search
    print("--- RUN 1: XGBoost + PrincipleGuided (20 stages) ---")
    classifier = XGBoostAdapter(seed=42, num_classes=2, boosting_rounds=30)
    search = PrincipleGuidedSearch()
    
    failsafe = FailsafeEngine(all_adapters=[
        XGBoostAdapter(seed=42, num_classes=2, boosting_rounds=30),
        RandomForestAdapter(seed=42, num_classes=2),
        KNNAdapter(seed=42, num_classes=2),
    ])
    
    optimizer = UniversalOptimizer(
        cfg=cfg, seed=42, dataset_name="hard_noisy_100f",
        fold=0, start_fold=0, classifier=classifier, search=search,
    )
    optimizer.set_failsafe(failsafe)
    result1 = optimizer.fit(X, y)
    
    print(f"\n  Result 1: acc={result1['accuracy']:.4f}, "
          f"f1={result1['f1_score']:.4f}, "
          f"features={result1['n_selected']}/{X.shape[1]}, "
          f"stress={result1['max_stress']:.4f}, "
          f"principles={result1['principles_discovered']}")
    
    # Run 2: Same dataset again — does memory work?
    print("\n--- RUN 2: Same dataset again (memory test) ---")
    classifier2 = XGBoostAdapter(seed=99, num_classes=2, boosting_rounds=30)
    search2 = PrincipleGuidedSearch()
    
    optimizer2 = UniversalOptimizer(
        cfg=cfg, seed=99, dataset_name="hard_noisy_100f",
        fold=1, start_fold=0, classifier=classifier2, search=search2,
    )
    result2 = optimizer2.fit(X, y)
    
    print(f"\n  Result 2: acc={result2['accuracy']:.4f}, "
          f"f1={result2['f1_score']:.4f}, "
          f"features={result2['n_selected']}/{X.shape[1]}, "
          f"stress={result2['max_stress']:.4f}, "
          f"principles={result2['principles_discovered']}")
    
    # Run 3: RandomForest on same dataset — cross-classifier
    print("\n--- RUN 3: RandomForest on same dataset (cross-classifier) ---")
    cfg3 = HardCfg()
    cfg3.stages = 10
    classifier3 = RandomForestAdapter(seed=42, num_classes=2)
    search3 = PrincipleGuidedSearch()
    
    optimizer3 = UniversalOptimizer(
        cfg=cfg3, seed=42, dataset_name="hard_noisy_100f",
        fold=2, start_fold=0, classifier=classifier3, search=search3,
    )
    result3 = optimizer3.fit(X, y)
    
    print(f"\n  Result 3: acc={result3['accuracy']:.4f}, "
          f"f1={result3['f1_score']:.4f}, "
          f"features={result3['n_selected']}/{X.shape[1]}, "
          f"stress={result3['max_stress']:.4f}, "
          f"principles={result3['principles_discovered']}")
    
    # ── ANALYSIS ──
    print("\n" + "=" * 70)
    print("  ANALYSIS: What did the UIL actually learn?")
    print("=" * 70)
    
    # Read journal
    with open(journal_path, "r") as f:
        journal = json.load(f)
    
    print(f"\n  Total journal entries: {len(journal)}")
    
    # Stress analysis
    stresses = [e["system_stress"] for e in journal if e["classifier"] != "FAILSAFE"]
    regrets = [e["regret"] for e in journal if e["classifier"] != "FAILSAFE"]
    accuracies = [e["outcome"]["accuracy"] for e in journal if e["classifier"] != "FAILSAFE"]
    investigations = [e for e in journal if e.get("investigation") is not None]
    
    print(f"  Stress range: {min(stresses):.4f} -> {max(stresses):.4f}")
    print(f"  Regret range: {min(regrets):.4f} -> {max(regrets):.4f}")
    print(f"  Accuracy range: {min(accuracies):.4f} -> {max(accuracies):.4f}")
    print(f"  Investigations triggered: {len(investigations)}")
    
    if investigations:
        print("\n  --- INVESTIGATION EXAMPLES ---")
        for inv in investigations[:3]:
            investigation = inv["investigation"]
            print(f"\n  Stage {inv['stage']}, Fold {inv['fold']}:")
            print(f"    Regret: {inv['regret']:.4f}")
            print(f"    Suspected cause: {investigation['suspected_cause']}")
            print(f"    Stress at time: {investigation['system_stress']:.4f}")
            changes = investigation.get("changes_detected", {})
            for key, detail in list(changes.items())[:3]:
                print(f"    Change: {key} = {detail['at_best']} -> {detail['current']} "
                      f"({detail['pct_change']:.1f}% change)")
    
    # Dataset memory
    with open(memory_path, "r") as f:
        memory = json.load(f)
    
    print(f"\n  --- DATASET MEMORY ---")
    for name, mem in memory.items():
        print(f"  {name}:")
        for k, v in mem.items():
            print(f"    {k}: {v}")
    
    # Principles
    with open(principles_path, "r") as f:
        principles = json.load(f)
    
    print(f"\n  --- PRINCIPLES DISCOVERED: {len(principles)} ---")
    for p in principles[:5]:
        conds = "; ".join(
            f"{c['feature']} {c['op']} {c['threshold']}"
            for c in p["conditions"]
        )
        print(f"\n  {p['principle_id']}:")
        print(f"    Conditions: {conds}")
        print(f"    Override: {p.get('action_override', {})}")
        print(f"    Confidence: {p['confidence']:.3f}, Evidence: {p['evidence_count']}")
        print(f"    Mining level: {p.get('mining_level', 'unknown')}")
        print(f"    Classifiers: {p.get('classifiers_confirmed', [])}")
    
    # Show stress progression
    print(f"\n  --- STRESS PROGRESSION (Run 1, first 10 stages) ---")
    run1_entries = [e for e in journal if e["fold"] == 0 and e["classifier"] != "FAILSAFE"][:10]
    for e in run1_entries:
        bar = "#" * int(e["system_stress"] * 50)
        print(f"  Stage {e['stage']:2d}: stress={e['system_stress']:.4f} "
              f"regret={e['regret']:.4f} acc={e['outcome']['accuracy']:.4f} |{bar}")


def _assert_stress_run_sanity(result: dict, label: str) -> None:
    """Minimal sanity asserts so this file can actually catch regressions."""
    acc = result.get("accuracy", -1)
    assert 0.0 <= acc <= 1.0, f"{label}: accuracy {acc!r} out of [0,1]"
    assert result.get("model") is not None, f"{label}: model is None (fit returned nothing)"
    fw = result.get("feature_weights")
    assert fw is not None and len(fw) > 0, f"{label}: feature_weights missing or empty"


if __name__ == "__main__":
    run_stress_test()
    # Collect results and assert basic sanity.  The individual result dicts
    # are printed above; we re-run a minimal smoke version just for assertions.
    import tempfile
    _X, _y = make_hard_dataset(n_samples=40, n_features=20, seed=0)

    class _MinCfg:
        stages = 5
        max_actions = 3
        state_dir = tempfile.mkdtemp(prefix="uil_stress_assert_")

    _cls = XGBoostAdapter(seed=0, num_classes=2, boosting_rounds=10)
    _opt = UniversalOptimizer(
        cfg=_MinCfg(), seed=0, dataset_name="_stress_assert",
        fold=0, start_fold=0, classifier=_cls,
        search=PrincipleGuidedSearch(),
    )
    _r = _opt.fit(_X, _y)
    _assert_stress_run_sanity(_r, "stress_run_smoke")
    print("ASSERTIONS PASSED")
