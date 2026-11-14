"""V19 UIL Integration Test — End-to-end verification.

Tests:
1. Protocol compliance (all 4 adapters satisfy ClassifierAdapter)
2. Stress engine (decision stress + system stress + regret)
3. Miner discovery (synthetic journal → tree discovers thresholds)
4. Library (promotion, decay, cross-classifier tracking)
5. Executor (math condition evaluation)
6. Search strategies (PSO + PrincipleGuided)
7. Failsafe (diagnosis)
8. Full optimizer loop (end-to-end with real classifier)
"""

import numpy as np
import sys
import os

# Ensure figpso_uil is importable (go up to Downloads/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from figpso_uil.core.classifier_protocol import ClassifierAdapter
from figpso_uil.core.utils import safe_minmax
from figpso_uil.cognition.uil_stress import StressEngine
from figpso_uil.cognition.uil_journal import UIL_Journal
from figpso_uil.cognition.uil_miner import UIL_Miner
from figpso_uil.cognition.uil_library import UIL_Library
from figpso_uil.cognition.uil_executor import UIL_Executor
from figpso_uil.adapters.xgboost_adapter import XGBoostAdapter
from figpso_uil.adapters.random_forest_adapter import RandomForestAdapter
from figpso_uil.adapters.knn_adapter import KNNAdapter
from figpso_uil.search.pso_search import PSOSearch
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch
from figpso_uil.core.failsafe import FailsafeEngine
from figpso_uil.core.optimizer import UniversalOptimizer


def _make_dataset(n_samples=200, n_features=20, n_classes=3, seed=42):
    """Create a synthetic classification dataset."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    # Make first 5 features informative
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_classes):
        mask = X[:, i % 5] > np.percentile(X[:, i % 5], 100 * i / n_classes)
        y[mask] = i
    # Ensure all classes present
    for i in range(n_classes):
        if not np.any(y == i):
            y[i] = i
    return X, y


def test_protocol_compliance():
    """Test 1: All adapters satisfy ClassifierAdapter protocol."""
    print("=" * 60)
    print("TEST 1: Protocol Compliance")
    print("=" * 60)

    adapters = [
        XGBoostAdapter(seed=42, num_classes=3),
        RandomForestAdapter(seed=42, num_classes=3),
        KNNAdapter(seed=42, num_classes=3),
    ]

    for adapter in adapters:
        assert isinstance(adapter, ClassifierAdapter), (
            f"{adapter.name} does not satisfy ClassifierAdapter"
        )
        assert isinstance(adapter.name, str)
        assert isinstance(adapter.supports_warm_start, bool)
        print(f"  ✓ {adapter.name}: protocol compliant, warm_start={adapter.supports_warm_start}")

    print("  PASSED\n")


def test_adapter_fit_predict():
    """Test 2: All adapters can fit and predict."""
    print("=" * 60)
    print("TEST 2: Adapter Fit/Predict")
    print("=" * 60)

    X, y = _make_dataset()
    weights = np.random.default_rng(42).uniform(0, 1, X.shape[1])

    adapters = [
        XGBoostAdapter(seed=42, num_classes=3),
        RandomForestAdapter(seed=42, num_classes=3),
        KNNAdapter(seed=42, num_classes=3),
    ]

    for adapter in adapters:
        model = adapter.fit_weighted(X, y, weights)
        preds = adapter.predict(model, X)
        importance = adapter.get_feature_importance(model, X.shape[1])
        complexity = adapter.get_model_complexity(model)

        assert preds.shape == (len(y),), f"{adapter.name}: prediction shape mismatch"
        assert importance.shape == (X.shape[1],), f"{adapter.name}: importance shape mismatch"
        assert complexity > 0, f"{adapter.name}: complexity should be > 0"

        acc = np.mean(preds == y)
        print(f"  ✓ {adapter.name}: acc={acc:.3f}, complexity={complexity}, importance_sum={importance.sum():.2f}")

    print("  PASSED\n")


def test_stress_engine():
    """Test 3: Stress engine compounding + regret."""
    print("=" * 60)
    print("TEST 3: Stress Engine")
    print("=" * 60)

    engine = StressEngine()

    # Simulate 10 stages of stagnation
    for i in range(10):
        ds = engine.compute_decision_stress(
            previous_accuracy=0.7,
            current_accuracy=0.7,  # No improvement
            cpu_time_this_stage=1.0,
            total_cpu_time=float(i + 1),
            stages_since_best=i,
        )
        ss = engine.update_system_stress(
            decision_stress=ds,
            current_accuracy=0.7,
            miner_r2_score=0.3,
        )

    print(f"  After 10 stages of stagnation:")
    print(f"    System Stress: {engine.system_stress:.4f} (should be HIGH)")
    print(f"    Best Proven: {engine.best_proven_accuracy:.4f}")
    assert engine.system_stress > 0.3, "System stress should compound under stagnation"

    # Test regret
    engine.best_proven_accuracy = 0.85
    regret = engine.compute_regret(0.75)
    print(f"  Regret (best=0.85, current=0.75): {regret:.4f} (should be > 0.10)")
    assert regret > 0.10, "Regret should be significant"

    # Test investigation
    investigation = engine.build_investigation(
        current_state={"penalty": 2.6, "entropy": 0.05},
        best_state={"penalty": 1.8, "entropy": 0.12},
        regret=regret,
    )
    assert investigation is not None, "Investigation should be triggered"
    assert "suspected_cause" in investigation
    print(f"  Investigation suspected cause: {investigation['suspected_cause']}")

    # Test relief after improvement
    stress_before = engine.system_stress
    engine.update_system_stress(
        decision_stress=0.0,
        current_accuracy=0.90,  # Big improvement
        miner_r2_score=0.8,
    )
    print(f"  Stress after improvement: {engine.system_stress:.4f} (was {stress_before:.4f})")
    assert engine.system_stress < stress_before, "Stress should decrease after improvement"

    print("  PASSED\n")


def test_miner_discovery():
    """Test 4: Miner discovers thresholds from synthetic journal."""
    print("=" * 60)
    print("TEST 4: Miner Auto-Discovery")
    print("=" * 60)

    miner = UIL_Miner(min_entries=20)

    # Create synthetic journal entries with a clear pattern:
    # High penalty (>2.0) on small datasets (<50 samples) → high regret
    rng = np.random.default_rng(42)
    entries = []
    for i in range(100):
        penalty = rng.uniform(0.5, 3.5)
        n_samples = rng.integers(20, 200)

        # The pattern: high penalty + small dataset = high regret
        if penalty > 2.0 and n_samples < 50:
            regret = rng.uniform(1.5, 3.0)
        else:
            regret = rng.uniform(0.0, 0.5)

        entries.append({
            "dataset": "synthetic",
            "classifier": "XGBoost",
            "state": {
                "n_samples": int(n_samples),
                "n_features": 20,
                "num_classes": 3,
                "entropy": round(rng.uniform(0, 1), 3),
            },
            "action": {
                "penalty": round(penalty, 3),
            },
            "outcome": {"accuracy": round(0.5 + rng.uniform(0, 0.4), 3)},
            "regret": round(regret, 3),
        })

    rules = miner.mine(entries, stress_level=0.5)
    r2 = miner.get_last_r2_score()

    print(f"  Journal entries: {len(entries)}")
    print(f"  Rules discovered: {len(rules)}")
    print(f"  Miner R² score: {r2:.4f}")

    if rules:
        for i, rule in enumerate(rules[:3]):
            conds = "; ".join(
                f"{c['feature']} {c['op']} {c['threshold']}"
                for c in rule["conditions"]
            )
            print(f"  Rule {i+1}: {conds}")
            print(f"    Predicted regret: {rule['predicted_regret']:.3f}, "
                  f"Confidence: {rule['confidence']:.3f}, "
                  f"Evidence: {rule['evidence_count']}")

    assert len(rules) > 0, "Miner should discover at least 1 rule"
    assert r2 > 0.3, f"R² should be meaningful (got {r2:.4f})"
    print("  PASSED\n")


def test_library_lifecycle():
    """Test 5: Library promotion, decay, cross-classifier tracking."""
    print("=" * 60)
    print("TEST 5: Library Lifecycle")
    print("=" * 60)

    import tempfile
    lib_path = os.path.join(tempfile.gettempdir(), "test_principles.json")
    library = UIL_Library(library_path=lib_path)

    # Create a rule with high confidence → should be promoted
    rules = [{
        "conditions": [
            {"feature": "action.penalty", "op": ">", "threshold": 2.0},
            {"feature": "state.n_samples", "op": "<=", "threshold": 50},
        ],
        "predicted_regret": 2.1,
        "evidence_count": 20,
        "confidence": 0.85,
        "mining_level": "within_dataset",
    }]

    library.update_principles(rules)
    principles = library.get_principles()
    assert len(principles) == 1, "One principle should be promoted"
    print(f"  ✓ Promoted principle: {principles[0]['principle_id']}")
    print(f"    Override: {principles[0]['action_override']}")
    assert "max_penalty" in principles[0]["action_override"]

    # Record applications
    for i in range(25):
        helped = i < 5  # Only helps 5/25 = 20% → below 30% threshold
        library.record_application(principles[0]["principle_id"], helped, "XGBoost")

    # Decay should halve confidence
    library.update_principles([])  # Trigger decay check
    principles = library.get_principles()
    if principles:
        print(f"  After decay: confidence={principles[0]['confidence']:.3f}")
        assert principles[0]["confidence"] < 0.85, "Confidence should decay"

    # Cleanup
    try:
        os.remove(lib_path)
    except OSError:
        pass

    print("  PASSED\n")


def test_executor_math():
    """Test 6: Executor pure math condition evaluation."""
    print("=" * 60)
    print("TEST 6: Executor Math Evaluation")
    print("=" * 60)

    executor = UIL_Executor()

    state = {
        "n_samples": 45,
        "n_features": 200,
        "entropy": 0.08,
        "penalty": 2.6,
    }

    principles = [{
        "principle_id": "P-001",
        "conditions": [
            {"feature": "penalty", "op": ">", "threshold": 2.0},
            {"feature": "n_samples", "op": "<=", "threshold": 50},
        ],
        "action_override": {"max_penalty": 2.0},
        "confidence": 0.90,
    }]

    overrides = executor.get_overrides(state, principles)
    triggered = executor.get_triggered_principles()

    print(f"  State: penalty={state['penalty']}, n_samples={state['n_samples']}")
    print(f"  Triggered: {triggered}")
    print(f"  Overrides: {overrides}")

    assert "P-001" in triggered, "P-001 should trigger (penalty=2.6 > 2.0 AND n_samples=45 <= 50)"
    assert overrides.get("max_penalty") == 2.0, "Should cap penalty at 2.0"

    # Test non-triggering
    state2 = {"n_samples": 100, "penalty": 1.5}
    overrides2 = executor.get_overrides(state2, principles)
    triggered2 = executor.get_triggered_principles()
    assert len(triggered2) == 0, "Should NOT trigger (conditions not met)"
    print(f"  State2: penalty=1.5, n_samples=100 → Triggered: {triggered2} (correct: none)")

    print("  PASSED\n")


def test_search_strategies():
    """Test 7: Both search strategies produce valid actions."""
    print("=" * 60)
    print("TEST 7: Search Strategies")
    print("=" * 60)

    n_features = 20
    guidance = np.random.default_rng(42).uniform(0, 1, n_features)

    # PSO
    pso = PSOSearch()
    pso.initialize(n_features, guidance, seed=42, n_particles=10, pso_iters=5)
    actions_pso = pso.propose_actions({}, [], 0.5, [])
    print(f"  PSO: {len(actions_pso)} actions proposed")
    for a in actions_pso[:2]:
        print(f"    {a['description']}: penalty={a['penalty']:.2f}, weights_sum={a['feature_weights'].sum():.2f}")

    # PrincipleGuided
    pg = PrincipleGuidedSearch()
    pg.initialize(n_features, guidance, seed=42)
    actions_pg = pg.propose_actions({}, [], 0.0, [])
    print(f"  PrincipleGuided (stress=0.0): {len(actions_pg)} actions")
    actions_pg_high = pg.propose_actions({}, [], 0.9, [])
    print(f"  PrincipleGuided (stress=0.9): {len(actions_pg_high)} actions")

    assert len(actions_pso) >= 1, "PSO should return at least 1 action"
    assert len(actions_pg) >= 1, "PrincipleGuided should return actions at low stress"
    assert len(actions_pg_high) >= len(actions_pg), "Higher stress should produce more actions"

    for a in actions_pso + actions_pg:
        assert "feature_weights" in a
        assert "penalty" in a
        assert a["feature_weights"].shape == (n_features,)
        assert np.all(a["feature_weights"] >= 0) and np.all(a["feature_weights"] <= 1)

    print("  PASSED\n")


def test_failsafe():
    """Test 8: Failsafe diagnosis."""
    print("=" * 60)
    print("TEST 8: Failsafe Diagnosis")
    print("=" * 60)

    X, y = _make_dataset(n_samples=200, n_features=20, n_classes=3)
    weights = np.random.default_rng(42).uniform(0, 1, X.shape[1])

    adapters = [
        XGBoostAdapter(seed=42, num_classes=3),
        RandomForestAdapter(seed=42, num_classes=3),
        KNNAdapter(seed=42, num_classes=3),
    ]

    failsafe = FailsafeEngine(all_adapters=adapters)
    diagnosis = failsafe.diagnose(X, y, weights, "XGBoost", seed=42)

    print(f"  Diagnosis: {diagnosis['diagnosis']}")
    print(f"  Data profile: {diagnosis['data_profile']}")
    print(f"  Fingerprint: {diagnosis['dataset_fingerprint']}")
    print(f"  Results:")
    for name, result in diagnosis["all_results"].items():
        print(f"    {name}: acc={result['accuracy']:.3f}, f1={result['f1_score']:.3f}")

    assert "diagnosis" in diagnosis
    assert "data_profile" in diagnosis
    assert "dataset_fingerprint" in diagnosis

    print("  PASSED\n")


def test_full_optimizer():
    """Test 9: End-to-end optimizer with real classifier."""
    print("=" * 60)
    print("TEST 9: Full Optimizer (End-to-End)")
    print("=" * 60)

    X, y = _make_dataset(n_samples=200, n_features=20, n_classes=3, seed=123)

    class SimpleCfg:
        stages = 5
        n_particles = 10
        pso_iters = 5
        particles = 10
        pso_iters_per_stage = 5
        boosting_rounds_per_stage = 20

    cfg = SimpleCfg()
    classifier = XGBoostAdapter(seed=42, num_classes=3, boosting_rounds=20)
    search = PrincipleGuidedSearch()

    optimizer = UniversalOptimizer(
        cfg=cfg,
        seed=42,
        dataset_name="test_synthetic",
        fold=0,
        start_fold=0,
        classifier=classifier,
        search=search,
    )

    result = optimizer.fit(X, y)

    print(f"  Accuracy: {result['accuracy']:.4f}")
    print(f"  F1 Score: {result['f1_score']:.4f}")
    print(f"  Features selected: {result['n_selected']}/{X.shape[1]}")
    print(f"  Max stress: {result['max_stress']:.4f}")
    print(f"  Principles discovered: {result['principles_discovered']}")
    print(f"  Stages run: {result['stages_run']}")

    assert result["model"] is not None, "Model should be trained"
    assert result["accuracy"] > 0.0, "Accuracy should be > 0"
    assert result["feature_weights"] is not None, "Feature weights should exist"

    # Test predict
    preds = optimizer.predict(X)
    assert preds.shape == (len(y),), "Predictions shape mismatch"
    print(f"  Predict OK: {preds.shape}")

    print("  PASSED\n")


def test_backward_compatible_wrapper():
    """Test 10: Old API works unchanged."""
    print("=" * 60)
    print("TEST 10: Backward-Compatible Wrapper")
    print("=" * 60)

    X, y = _make_dataset(n_samples=200, n_features=20, n_classes=3, seed=456)

    class SimpleCfg:
        stages = 3
        particles = 10
        pso_iters_per_stage = 5
        boosting_rounds_per_stage = 15

    from figpso_uil.core.fig_xgboost import FIG_XGBoost_Embedded

    model = FIG_XGBoost_Embedded(
        cfg=SimpleCfg(),
        seed=42,
        dataset_name="test_backward_compat",
        fold=0,
        start_fold=0,
    )
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == (len(y),), "Predictions shape mismatch"
    assert model.final_features is not None, "final_features should be set"
    print(f"  final_features: {len(model.final_features)} selected")
    print(f"  predict shape: {preds.shape}")

    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  V19 UIL INTEGRATION TEST SUITE")
    print("=" * 60 + "\n")

    tests = [
        test_protocol_compliance,
        test_adapter_fit_predict,
        test_stress_engine,
        test_miner_discovery,
        test_library_lifecycle,
        test_executor_math,
        test_search_strategies,
        test_failsafe,
        test_full_optimizer,
        test_backward_compatible_wrapper,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()

    print("=" * 60)
    print(f"  RESULTS: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
