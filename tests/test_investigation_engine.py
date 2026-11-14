"""Test V20+.4: Investigation Engine — all 5 strategies."""
import sys
sys.path.insert(0, ".")
import numpy as np
from figpso_uil.cognition.investigation_engine import (
    InvestigationEngine,
    STRATEGY_REVERT_RECORD,
    STRATEGY_PARTIAL_REVERT,
    STRATEGY_CONTROLLED_RETRY,
    STRATEGY_HYPOTHESIS_TEST,
    STRATEGY_ALTERNATIVE_PATH,
)

ie = InvestigationEngine()

# Shared test data
current_config = {"max_depth": 12, "learning_rate": 0.3, "n_estimators": 200}
best_config = {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 150}
current_weights = np.array([0.8, 0.2, 0.9, 0.1, 0.5])
best_weights = np.array([0.6, 0.4, 0.7, 0.3, 0.5])
investigation_data = {
    "trigger": "regret",
    "regret_value": 0.07,
    "system_stress": 0.4,
    "changes_detected": {
        "max_depth": {"current": 12, "at_best": 6, "delta": 6, "pct_change": 100.0},
        "learning_rate": {"current": 0.3, "at_best": 0.1, "delta": 0.2, "pct_change": 200.0},
    },
    "suspected_cause": "learning_rate",
    "suspected_delta": 0.2,
}

# Test 1: Strategy selection
print("TEST 1: Strategy Selection")
assert ie.select_strategy(0.1, 0.5, 0) == STRATEGY_REVERT_RECORD
assert ie.select_strategy(0.4, 0.5, 0) == STRATEGY_PARTIAL_REVERT
assert ie.select_strategy(0.65, 0.5, 0) == STRATEGY_CONTROLLED_RETRY
assert ie.select_strategy(0.9, 0.5, 0) == STRATEGY_HYPOTHESIS_TEST
assert ie.select_strategy(0.5, 0.5, 3) == STRATEGY_ALTERNATIVE_PATH
print("  All strategy selections correct")

# Test 2: C.1 Revert + Record
print("\nTEST 2: C.1 Revert + Record")
result = ie.execute_investigation(
    STRATEGY_REVERT_RECORD, current_config, best_config,
    current_weights, best_weights, investigation_data,
)
assert result["proposed_config"]["max_depth"] == 6
assert result["proposed_config"]["learning_rate"] == 0.1
assert np.allclose(result["proposed_weights"], best_weights)
assert result["strategy"] == STRATEGY_REVERT_RECORD
print(f"  Rationale: {result['rationale']}")
print("  C.1 OK")

# Test 3: C.2 Partial Revert
print("\nTEST 3: C.2 Partial Revert")
ie2 = InvestigationEngine()
result2 = ie2.execute_investigation(
    STRATEGY_PARTIAL_REVERT, current_config, best_config,
    current_weights, best_weights, investigation_data,
)
# Should revert only the suspected cause (learning_rate)
assert result2["proposed_config"]["learning_rate"] == 0.1  # reverted
assert result2["proposed_config"]["max_depth"] == 12  # kept current
print(f"  Rationale: {result2['rationale']}")
print("  C.2 OK")

# Test 4: C.3 Controlled Retry
print("\nTEST 4: C.3 Controlled Retry")
ie3 = InvestigationEngine()
result3 = ie3.execute_investigation(
    STRATEGY_CONTROLLED_RETRY, current_config, best_config,
    current_weights, best_weights, investigation_data,
)
# Should be halfway between best and current
expected_lr = 0.1 + (0.3 - 0.1) * 0.5  # 0.2
assert abs(result3["proposed_config"]["learning_rate"] - expected_lr) < 0.01
expected_weights = best_weights + (current_weights - best_weights) * 0.5
assert np.allclose(result3["proposed_weights"], expected_weights)
print(f"  Rationale: {result3['rationale']}")
print(f"  learning_rate: {result3['proposed_config']['learning_rate']}")
print("  C.3 OK")

# Test 5: C.4 Hypothesis Test
print("\nTEST 5: C.4 Hypothesis Test")
ie4 = InvestigationEngine()
result4 = ie4.execute_investigation(
    STRATEGY_HYPOTHESIS_TEST, current_config, best_config,
    current_weights, best_weights, investigation_data,
)
assert "hypothesis" in result4
assert result4["hypothesis"]["variable"] == "learning_rate"
assert result4["hypothesis"]["control"] == 0.1
assert result4["hypothesis"]["treatment"] == 0.3
print(f"  Rationale: {result4['rationale']}")
print(f"  Hypothesis: {result4['hypothesis']}")
print("  C.4 OK")

# Test 6: C.5 Alternative Path
print("\nTEST 6: C.5 Alternative Path")
ie5 = InvestigationEngine()
result5 = ie5.execute_investigation(
    STRATEGY_ALTERNATIVE_PATH, current_config, best_config,
    current_weights, best_weights, investigation_data,
)
# Weights should be different from both current and best
assert not np.allclose(result5["proposed_weights"], current_weights)
assert not np.allclose(result5["proposed_weights"], best_weights)
# But should be clipped to [0, 1]
assert np.all(result5["proposed_weights"] >= 0.0)
assert np.all(result5["proposed_weights"] <= 1.0)
print(f"  Rationale: {result5['rationale']}")
print(f"  Orthogonal weights: {result5['proposed_weights']}")
print("  C.5 OK")

# Test 7: Investigation outcome recording
print("\nTEST 7: Outcome Recording")
ie.record_investigation_outcome(0, 0.78, 0.85, True)
summary = ie.get_investigation_summary()
assert summary["total_investigations"] == 1
assert summary["restorations_successful"] == 1
print(f"  Summary: {summary}")
print("  Outcome recording OK")

# Test 8: Escalation across multiple investigations
print("\nTEST 8: Escalation")
ie_esc = InvestigationEngine()
strategies = []
for i in range(5):
    s = ie_esc.select_strategy(0.5, 0.5, i)
    strategies.append(s)
    ie_esc.execute_investigation(
        s, current_config, best_config,
        current_weights, best_weights, investigation_data,
    )
print(f"  Strategies over 5 investigations: {strategies}")
assert strategies[-1] == STRATEGY_ALTERNATIVE_PATH  # Should escalate
assert strategies[-2] == STRATEGY_ALTERNATIVE_PATH
print("  Escalation OK")

print("\n" + "=" * 50)
print("ALL V20+.4 TESTS PASSED")
