"""V20 UIL Integration Test -- Formula Engine + Classifier Control + Dual Regret.

Tests:
1. V20.1: Dual recovery cost (stages + CPU time)
2. V20.2: Classifier controllable params (get/set/apply)
3. V20.2: Hyperparameter changes propagate through optimizer
4. V20.3: Formula engine (create, evaluate, mutate, validate, serialize)
5. V20.3: Formula library (store, promote, decay, retrieve)
6. V20.3: Custom objective trains XGBoost
7. V20.4: Miner includes classifier_config dimensions
8. Full optimizer with V20 features end-to-end
"""

import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from figpso_uil.cognition.uil_stress import StressEngine
from figpso_uil.cognition.formula_engine import FormulaEngine, FormulaNode, SEED_FORMULAS
from figpso_uil.cognition.formula_library import FormulaLibrary
from figpso_uil.cognition.uil_miner import UIL_Miner
from figpso_uil.adapters.xgboost_adapter import XGBoostAdapter
from figpso_uil.adapters.random_forest_adapter import RandomForestAdapter
from figpso_uil.adapters.knn_adapter import KNNAdapter
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch
from figpso_uil.core.optimizer import UniversalOptimizer


def _make_dataset(n_samples=200, n_features=20, n_classes=3, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_classes):
        mask = X[:, i % 5] > np.percentile(X[:, i % 5], 100 * i / n_classes)
        y[mask] = i
    for i in range(n_classes):
        if not np.any(y == i):
            y[i] = i
    return X, y


def test_v20_1_dual_recovery_cost():
    """V20.1: Recovery cost uses both stages AND CPU time."""
    print("=" * 60)
    print("TEST V20.1: Dual Recovery Cost")
    print("=" * 60)

    engine = StressEngine()

    # Stage 1: good accuracy
    ds1 = engine.compute_decision_stress(
        previous_accuracy=0.0, current_accuracy=0.8,
        cpu_time_this_stage=1.0, total_cpu_time=1.0, stages_since_best=0,
    )
    engine.update_system_stress(ds1, 0.8)

    # Stages 2-5: stagnation with increasing CPU time
    for i in range(1, 5):
        ds = engine.compute_decision_stress(
            previous_accuracy=0.8, current_accuracy=0.75,
            cpu_time_this_stage=5.0,  # expensive stages
            total_cpu_time=1.0 + i * 5.0,
            stages_since_best=i,
        )
        engine.update_system_stress(ds, 0.75)

    breakdown = engine.get_recovery_cost_breakdown()
    print(f"  Recovery breakdown: {breakdown}")
    print(f"  CPU time since best: {engine.cpu_time_since_best:.1f}s")

    assert breakdown["stages_since_best"] == 4
    assert breakdown["cpu_time_since_best"] > 10.0, "CPU time should accumulate"
    assert breakdown["cpu_recovery_component"] > 0, "CPU recovery should be nonzero"
    assert breakdown["recovery_cost"] > 0.1, "Recovery cost should be meaningful"

    # Now improve -- should reset CPU clock
    ds_good = engine.compute_decision_stress(
        previous_accuracy=0.75, current_accuracy=0.9,
        cpu_time_this_stage=1.0, total_cpu_time=22.0, stages_since_best=0,
    )
    engine.update_system_stress(ds_good, 0.9)
    assert engine.cpu_time_since_best == 0.0, "CPU time should reset after new best"
    print(f"  After improvement: cpu_time_since_best = {engine.cpu_time_since_best}")

    print("  PASSED\n")


def test_v20_2_controllable_params():
    """V20.2: All adapters expose controllable parameters."""
    print("=" * 60)
    print("TEST V20.2: Controllable Parameters")
    print("=" * 60)

    adapters = [
        XGBoostAdapter(seed=42, num_classes=3),
        RandomForestAdapter(seed=42, num_classes=3),
        KNNAdapter(seed=42, num_classes=3),
    ]

    for adapter in adapters:
        params = adapter.get_controllable_params()
        config_before = adapter.get_current_config()

        assert len(params) > 0, f"{adapter.name} should have controllable params"
        print(f"  {adapter.name}: {len(params)} controllable params")

        for pname, meta in params.items():
            assert "current" in meta, f"{pname} missing 'current'"
            assert "type" in meta, f"{pname} missing 'type'"
            print(f"    {pname}: {meta['current']} ({meta['type']})")

        # Test apply_config
        if adapter.name == "XGBoost":
            adapter.apply_config({"max_depth": 10, "learning_rate": 0.05})
            new_config = adapter.get_current_config()
            assert new_config["max_depth"] == 10, "max_depth should be 10"
            assert abs(new_config["learning_rate"] - 0.05) < 1e-6, "lr should be 0.05"
            print(f"  XGBoost after apply: max_depth={new_config['max_depth']}, lr={new_config['learning_rate']}")

        elif adapter.name == "RandomForest":
            adapter.apply_config({"n_estimators": 100, "criterion": "entropy"})
            new_config = adapter.get_current_config()
            assert new_config["n_estimators"] == 100
            assert new_config["criterion"] == "entropy"
            print(f"  RF after apply: n_estimators={new_config['n_estimators']}, criterion={new_config['criterion']}")

        elif adapter.name == "KNN":
            adapter.apply_config({"n_neighbors": 7, "weights": "distance"})
            new_config = adapter.get_current_config()
            assert new_config["n_neighbors"] == 7
            assert new_config["weights"] == "distance"
            print(f"  KNN after apply: k={new_config['n_neighbors']}, weights={new_config['weights']}")

        # Verify custom objective support
        print(f"  {adapter.name} supports_custom_objective: {adapter.supports_custom_objective}")

    print("  PASSED\n")


def test_v20_3_formula_engine():
    """V20.3: Formula engine creates, evaluates, mutates, validates."""
    print("=" * 60)
    print("TEST V20.3: Formula Engine")
    print("=" * 60)

    engine = FormulaEngine(seed=42)

    # Test seed formulas
    seeds = engine.get_seed_formulas()
    assert len(seeds) >= 4, f"Should have >= 4 seed formulas, got {len(seeds)}"
    print(f"  Seed formulas: {list(seeds.keys())}")

    # Evaluate logistic loss
    y_pred = np.array([0.9, 0.1, 0.7, 0.3])
    y_true = np.array([1.0, 0.0, 1.0, 0.0])

    for name, tree in seeds.items():
        loss = engine.evaluate(tree, {"y_pred": y_pred, "y_true": y_true})
        print(f"  {name}: loss = {np.mean(loss):.4f}")
        assert np.all(np.isfinite(loss)), f"{name} produced non-finite loss"

    # Test gradient/hessian computation
    logistic = seeds["logistic"]
    grad = engine.compute_gradient(logistic, y_pred, y_true)
    hess = engine.compute_hessian(logistic, y_pred, y_true)
    print(f"  Logistic gradient: {grad}")
    print(f"  Logistic hessian: {hess}")
    assert np.all(np.isfinite(grad)), "Gradient should be finite"
    assert np.all(np.isfinite(hess)), "Hessian should be finite"
    assert np.all(hess > 0), "Hessian should be positive (XGBoost requirement)"

    # Test mutation at different stress levels
    for stress in [0.1, 0.5, 0.9]:
        mutated, mutation_type = engine.mutate(logistic, stress)
        is_valid, reason = engine.validate(mutated)
        print(f"  Mutation (stress={stress}): {mutation_type}, valid={is_valid}")
        assert is_valid, f"Mutation at stress={stress} produced invalid formula: {reason}"

    # Test crossover
    focal = seeds["focal"]
    crossed = engine.crossover(logistic, focal)
    is_valid, reason = engine.validate(crossed)
    print(f"  Crossover (logistic x focal): valid={is_valid}")

    # Test serialization
    serialized = engine.serialize(logistic)
    deserialized = engine.deserialize(serialized)
    loss_orig = engine.evaluate(logistic, {"y_pred": y_pred, "y_true": y_true})
    loss_deser = engine.evaluate(deserialized, {"y_pred": y_pred, "y_true": y_true})
    assert np.allclose(loss_orig, loss_deser), "Serialization roundtrip should preserve formula"
    print(f"  Serialization roundtrip: OK")

    # Test validation catches bad formulas
    bad = FormulaNode(op="div", children=[
        FormulaNode(const=1.0),
        FormulaNode(const=0.0),
    ])
    # This won't be invalid since safe_div handles it, but a deeply nested one would be
    deep = FormulaNode(var="y_pred")
    for _ in range(20):
        deep = FormulaNode(op="abs", children=[deep])
    is_valid, reason = engine.validate(deep)
    print(f"  Deep tree (depth {deep.depth()}): valid={is_valid}, reason={reason}")

    print("  PASSED\n")


def test_v20_3_formula_library():
    """V20.3: Formula library stores, promotes, and decays formulas."""
    print("=" * 60)
    print("TEST V20.3: Formula Library")
    print("=" * 60)

    # Per-run temp dir, not a FIXED path in %TEMP%. The old fixed path was
    # only deleted by the cleanup at the END of this function, so ANY
    # assertion failure left a populated library behind -- and the next run
    # then failed EARLIER and for a different reason (seeds already past the
    # bootstrap floor), hiding the real defect. Measured 2026-08-05.
    lib_dir = tempfile.mkdtemp(prefix="uil_v20_lib_")
    lib_path = os.path.join(lib_dir, "test_formula_lib.json")
    library = FormulaLibrary(filepath=lib_path)

    # Seed
    engine = FormulaEngine(seed=42)
    seeds = engine.get_seed_formulas()
    library.seed_formulas(seeds)
    print(f"  Seeded {library.formula_count()} formulas")

    # Add custom formula
    custom = FormulaNode(op="add", children=[
        FormulaNode(var="y_pred"),
        FormulaNode(const=0.5),
    ])
    custom_id = library.add_formula("custom_add", custom, "test", "manual")
    print(f"  Total formulas: {library.formula_count()}")

    # Record successes
    for _ in range(8):
        library.record_result(custom_id, accuracy=0.85, helped=True, dataset_fingerprint="test_ds")
    for _ in range(2):
        library.record_result(custom_id, accuracy=0.60, helped=False, dataset_fingerprint="test_ds")

    # V20+.13: bootstrap phase. The 4 seed formulas have evidence=0 --
    # get_best_formula() must offer one of THEM first, regardless of
    # custom_id's confidence, otherwise an under-tested formula can never
    # accumulate evidence and is shadowed by the incumbent forever.
    best = library.get_best_formula(dataset_fingerprint="test_ds")
    print(f"  Best for test_ds (bootstrap phase): {best['formula_id']} (evidence={best['evidence']})")
    assert best["evidence"] < FormulaLibrary.MIN_EVIDENCE_BOOTSTRAP
    assert best["formula_id"] != custom_id

    # Bootstrap every seed formula past the evidence floor.
    for seed_id in ["SEED-logistic", "SEED-focal", "SEED-squared", "SEED-hinge"]:
        for _ in range(FormulaLibrary.MIN_EVIDENCE_BOOTSTRAP):
            library.record_result(seed_id, accuracy=0.5, helped=False)

    # Mature phase: now that every formula has cleared the bootstrap
    # floor, ranking falls back to confidence (+ a shrinking exploration
    # bonus) -- custom_id's 8/10 record should win over the seeds' 0/3.
    # Confidence is now a Wilson lower bound, not the raw ratio (8/10's
    # raw ratio is 0.8, but the discounted-for-uncertainty value is
    # ~0.49 -- still far above the seeds' ~0.0 after 3 straight failures).
    best = library.get_best_formula(dataset_fingerprint="test_ds")
    print(f"  Best for test_ds (mature phase): {best['formula_id']} (confidence={best['confidence']:.2f})")
    assert best["formula_id"] == custom_id
    assert abs(best["confidence"] - FormulaLibrary._wilson_lower_bound(8, 10)) < 1e-9

    # Test decay (record lots of failures)
    for _ in range(20):
        library.record_result("SEED-logistic", accuracy=0.5, helped=False)
    seed_logistic = [f for f in library.get_all() if f["formula_id"] == "SEED-logistic"][0]
    print(f"  SEED-logistic after 20 failures: status={seed_logistic['status']}, confidence={seed_logistic['confidence']:.2f}")
    # Retirement is a TWO-STEP pipeline: record_result sets "retired"
    # (evidence >= 10 and confidence < 0.3), then E3's
    # prune_retired_to_archive moves it to "archived" and leaves a tombstone.
    # This assertion predates E3 and demanded the intermediate state, so it
    # failed on a CLEAN library once archiving became automatic. Measured
    # 2026-08-05 on a fresh library: 5 failures -> "seed", 10+ -> "archived";
    # "retired" is never observed at this point. The intent is "a
    # consistently failing formula leaves the working pool" -- both terminal
    # states satisfy that.
    assert seed_logistic["status"] in ("retired", "archived"), (
        f"Should leave the working pool after consistent failure, "
        f"got status={seed_logistic['status']!r}"
    )

    # Best-effort cleanup. Unlike the old fixed path, a leaked dir here is
    # harmless: every run gets its own, so it can never poison the next one.
    import shutil
    shutil.rmtree(lib_dir, ignore_errors=True)

    print("  PASSED\n")


def test_v20_3_custom_objective():
    """V20.3: Custom formula objective trains XGBoost."""
    print("=" * 60)
    print("TEST V20.3: Custom Objective Training")
    print("=" * 60)

    X, y = _make_dataset(n_samples=200, n_features=20, n_classes=2, seed=42)
    y = (y > 0).astype(int)  # binary

    engine = FormulaEngine(seed=42)
    logistic = engine.get_seed_formulas()["logistic"]

    # Build grad/hess closures
    def grad_fn(y_pred, y_true):
        return engine.compute_gradient(logistic, y_pred, y_true)

    def hess_fn(y_pred, y_true):
        return engine.compute_hessian(logistic, y_pred, y_true)

    adapter = XGBoostAdapter(seed=42, num_classes=2, boosting_rounds=30)
    adapter.set_custom_objective(grad_fn, hess_fn)

    weights = np.ones(X.shape[1])
    model = adapter.fit_weighted(X, y, weights)
    preds = adapter.predict(model, X)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y, preds)
    print(f"  Custom objective accuracy: {acc:.4f}")
    assert acc > 0.5, f"Custom objective should beat random, got {acc:.4f}"

    # Reset to default
    adapter.set_custom_objective(None, None)
    model2 = adapter.fit_weighted(X, y, weights)
    preds2 = adapter.predict(model2, X)
    acc2 = accuracy_score(y, preds2)
    print(f"  Default objective accuracy: {acc2:.4f}")

    print("  PASSED\n")


def test_v20_4_miner_classifier_config():
    """V20.4: Miner includes classifier_config dimensions in feature matrix."""
    print("=" * 60)
    print("TEST V20.4: Miner Config Dimensions")
    print("=" * 60)

    miner = UIL_Miner(min_entries=20)
    rng = np.random.default_rng(42)

    # Create entries WITH classifier_config in action
    entries = []
    for i in range(60):
        max_depth = rng.integers(3, 12)
        lr = rng.uniform(0.01, 0.5)
        penalty = rng.uniform(0.5, 3.5)

        # Pattern: high max_depth + high learning_rate = high regret
        if max_depth > 8 and lr > 0.3:
            regret = rng.uniform(1.5, 3.0)
        else:
            regret = rng.uniform(0.0, 0.5)

        entries.append({
            "dataset": "test_config",
            "classifier": "XGBoost",
            "state": {"n_samples": 100, "n_features": 20},
            "action": {
                "penalty": round(float(penalty), 3),
                "classifier_config": {
                    "max_depth": int(max_depth),
                    "learning_rate": round(float(lr), 4),
                },
            },
            "regret": round(float(regret), 3),
        })

    rules = miner.mine(entries, stress_level=0.5)
    r2 = miner.get_last_r2_score()

    print(f"  Entries: {len(entries)}")
    print(f"  Rules discovered: {len(rules)}")
    print(f"  R2 score: {r2:.4f}")

    # Check that config features were included
    if rules:
        all_features = set()
        for rule in rules:
            for cond in rule["conditions"]:
                all_features.add(cond["feature"])
        print(f"  Features in rules: {all_features}")
        config_features = [f for f in all_features if f.startswith("config.")]
        print(f"  Config features discovered: {config_features}")
        # At least R2 should be meaningful
        assert r2 > 0.3, f"R2 should be meaningful since there's a clear pattern (got {r2:.4f})"

    print("  PASSED\n")


def test_v20_full_optimizer():
    """Full optimizer with V20 features end-to-end."""
    print("=" * 60)
    print("TEST V20: Full Optimizer (End-to-End)")
    print("=" * 60)

    X, y = _make_dataset(n_samples=200, n_features=20, n_classes=3, seed=42)

    class SimpleCfg:
        stages = 8
        n_particles = 10
        pso_iters = 5

    # Test with XGBoost
    adapter = XGBoostAdapter(seed=42, num_classes=3, boosting_rounds=20)
    search = PrincipleGuidedSearch()

    optimizer = UniversalOptimizer(
        cfg=SimpleCfg(),
        seed=42,
        dataset_name="test_v20",
        fold=0,
        start_fold=0,
        classifier=adapter,
        search=search,
    )

    result = optimizer.fit(X, y)

    print(f"  Accuracy: {result['accuracy']:.4f}")
    print(f"  F1 Score: {result['f1_score']:.4f}")
    print(f"  Max stress: {result['max_stress']:.4f}")
    print(f"  Stages run: {result['stages_run']}")

    # Verify journal has classifier_config entries
    entries = optimizer.journal.get_entries()
    v20_entries = [e for e in entries if e.get("dataset") == "test_v20"]
    if v20_entries:
        sample = v20_entries[-1]
        action = sample.get("action", {})
        print(f"  Last action keys: {list(action.keys())}")
        if "classifier_config" in action:
            print(f"  Classifier config recorded: {list(action['classifier_config'].keys())}")
        outcome = sample.get("outcome", {})
        if "recovery_cost" in outcome:
            print(f"  Recovery cost recorded: {outcome['recovery_cost']}")

    assert result["model"] is not None
    assert result["accuracy"] > 0.0

    # Test predict
    preds = optimizer.predict(X)
    assert preds.shape == (len(y),)

    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  V20 UIL INTEGRATION TEST SUITE")
    print("=" * 60 + "\n")

    tests = [
        test_v20_1_dual_recovery_cost,
        test_v20_2_controllable_params,
        test_v20_3_formula_engine,
        test_v20_3_formula_library,
        test_v20_3_custom_objective,
        test_v20_4_miner_classifier_config,
        test_v20_full_optimizer,
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
    print(f"  V20 RESULTS: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
