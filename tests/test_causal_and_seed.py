"""Quick test for V20+.5 and V20+.6."""
import sys
sys.path.insert(0, ".")
import tempfile, os

# V20+.5: Causal Intervention Engine
from figpso_uil.cognition.causal_intervention import (
    CausalInterventionEngine, STATUS_CONFIRMED_CAUSAL, STATUS_REFUTED
)

# Use a temp file so repeated test runs don't dedup against hypotheses
# persisted by a previous run (see Unit A: CausalInterventionEngine now
# persists to disk, like UIL_Library does below).
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as _f:
    _f.write("[]")
    _ci_tmp_path = _f.name
ci = CausalInterventionEngine(filepath=_ci_tmp_path)

# Form hypothesis from a DT-mined rule
rule = {
    "conditions": [{"feature": "state.n_features", "op": ">", "threshold": 500}],
    "predicted_regret": 0.12,
    "confidence": 0.6,
}
h = ci.form_hypothesis(rule)
assert h is not None
print(f"Hypothesis: {h.hypothesis_id} — '{h.variable}' {h.direction} {h.threshold}")

# Design intervention
plan = ci.design_intervention(h, current_value=600)
print(f"Control value: {plan['control']['value']}, Treatment value: {plan['treatment']['value']}")

# Evaluate — CONFIRMED_CAUSAL now requires majority of last 3 trials,
# not a single lucky comparison. Run the same strong intervention 3 times.
for _trial in range(3):
    result = ci.evaluate_intervention(
        h.hypothesis_id,
        control_regret=0.12, treatment_regret=0.02,
        control_accuracy=0.78, treatment_accuracy=0.85,
    )
# After 3 consistent strong trials → CONFIRMED_CAUSAL
assert result["status"] == STATUS_CONFIRMED_CAUSAL, (
    f"Expected CONFIRMED_CAUSAL after 3 trials, got {result['status']}: "
    f"{result.get('conclusion', '')}"
)
print(f"Status: {result['status']}, Causal strength: {result['causal_strength']}")

# Confidence adjustment — only > 1.0 once CONFIRMED_CAUSAL
adj = ci.get_confidence_adjustment(h.hypothesis_id)
assert adj > 1.0, f"Expected adj > 1.0 after confirmed causal, got {adj}"
print(f"Confidence multiplier: {adj}")

# Test refutation
rule2 = {
    "conditions": [{"feature": "state.entropy", "op": "<", "threshold": 0.3}],
    "predicted_regret": 0.08,
    "confidence": 0.5,
}
h2 = ci.form_hypothesis(rule2)
ci.design_intervention(h2)
result2 = ci.evaluate_intervention(
    h2.hypothesis_id,
    control_regret=0.05, treatment_regret=0.15,
    control_accuracy=0.85, treatment_accuracy=0.75,
)
assert result2["status"] == STATUS_REFUTED
print(f"Refuted hypothesis: {result2['conclusion']}")

summary = ci.get_summary()
print(f"Summary: {summary['total_hypotheses']} hypotheses, {summary['total_interventions']} interventions")
print("V20+.5 PASSED\n")

os.unlink(_ci_tmp_path)

# V20+.6: Seed Knowledge
from figpso_uil.cognition.uil_library import UIL_Library

# Use temp file so we don't affect the real library
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    f.write("[]")
    tmp_path = f.name

try:
    lib = UIL_Library(library_path=tmp_path)
    seeds = [
        {
            "conditions": [{"feature": "state.sparsity", "op": ">", "threshold": 0.7}],
            "action_override": {"max_penalty": 1.5},
            "description": "Sparse datasets need lower penalty",
        },
        {
            "conditions": [{"feature": "state.n_samples", "op": "<", "threshold": 50}],
            "action_override": {"exploration_rate": 0.3},
            "description": "Small datasets need more exploration",
        },
    ]
    count = lib.accept_seed_principles(seeds)
    assert count == 2
    print(f"Accepted {count} seed principles")

    # Verify they start with neutral confidence
    principles = lib.get_principles()
    for p in principles:
        assert p["confidence"] == 0.5
        assert p["source"] == "seed"
        assert p["mining_level"] == "seed"
    print("Seed principles have neutral confidence, source='seed'")

    # Verify duplicate rejection
    count2 = lib.accept_seed_principles(seeds)
    assert count2 == 0
    print("Duplicate seeds correctly rejected")
    print("V20+.6 PASSED")
finally:
    os.unlink(tmp_path)

print("\n" + "=" * 50)
print("V20+.5 AND V20+.6 ALL TESTS PASSED")
