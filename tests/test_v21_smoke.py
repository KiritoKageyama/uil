"""Comprehensive smoke test for all V21 + gap closure components."""
import sys; sys.path.insert(0, ".")
import numpy as np

from figpso_uil.core.optimizer import UniversalOptimizer
from figpso_uil.cognition.structural_representation import StructuralEncoder
from figpso_uil.cognition.journal_forgetting import (
    PrincipleVersioning, ContradictionDetector,
)
from figpso_uil.cognition.meta_principle_merger import MetaPrincipleMerger
from figpso_uil.cognition.abstraction_safeguards import AbstractionSafeguards
from types import SimpleNamespace

# Mock classifier adapter
class MockClassifier:
    name = "knn"
    def fit(self, X, y, w=None): return {"model": True}
    def predict(self, m, X): return np.zeros(len(X))
    def score(self, m, X, y): return 0.5
    def get_controllable_params(self): return {}

# Mock config
cfg = SimpleNamespace(stages=5, n_particles=10, pso_iters=5)

# 1. Instantiate optimizer and verify all components
opt = UniversalOptimizer(cfg=cfg, seed=42, dataset_name="test",
                         fold=0, start_fold=0, classifier=MockClassifier())
print("=== Component Check ===")
components = [
    "journal_forgetting", "principle_versioning", "contradiction_detector",
    "foundation", "structural_encoder", "meta_merger", "safeguards",
    "causal_engine", "episodic_attention",
]
for name in components:
    obj = getattr(opt, name, None)
    status = type(obj).__name__ if obj else "MISSING"
    print(f"  {name}: {status}")
    assert obj is not None, f"Missing component: {name}"
print()

# 2. Test foundation maturity: infant state
maturity = opt.foundation.update_maturity(
    n_principles=0, n_episodes=0, n_formulas=0,
    miner_r2=0.0, network_nodes=0, recent_accuracy=0.0,
)
alloc = opt.foundation.allocate_actions()
print("=== Infant State (0 knowledge) ===")
print(f"  Maturity: {maturity:.3f}, Stage: {alloc['stage']}")
print(f"  PSO: {alloc['pso_actions']}, Reasoning: {alloc['reasoning_actions']}")
assert alloc["stage"] == "infant"
assert alloc["pso_actions"] >= alloc["reasoning_actions"]
print()

# 3. Test foundation maturity: mature state
maturity = opt.foundation.update_maturity(
    n_principles=60, n_episodes=25, n_formulas=15,
    miner_r2=0.85, network_nodes=40, recent_accuracy=0.92,
)
alloc = opt.foundation.allocate_actions()
print("=== Mature State (rich knowledge) ===")
print(f"  Maturity: {maturity:.3f}, Stage: {alloc['stage']}")
print(f"  PSO: {alloc['pso_actions']}, Reasoning: {alloc['reasoning_actions']}")
assert alloc["stage"] in ("mature", "expert")
assert alloc["reasoning_actions"] >= alloc["pso_actions"]
print()

# 4. Test structural encoder
enc = StructuralEncoder()
test_principle = {
    "principle_id": "P-001",
    "conditions": [{"feature": "n_samples", "op": "<", "threshold": 100}],
    "action_override": {"max_penalty": -0.5},
    "confidence": 0.72,
}
sp = enc.encode(test_principle)
print("=== Structural Encoding ===")
print(f"  Context labels: {sp.context_labels}")
print(f"  Action label: {sp.action_label}")
print(f"  Human text: {sp.human_text}")
assert len(sp.context_labels) > 0, "Should infer context labels"
print()

# 5. Test principle versioning
pv = PrincipleVersioning()
pv.record_version({"principle_id": "P-001", "confidence": 0.5})
pv.record_version({"principle_id": "P-001", "confidence": 0.72})
pv.record_version({"principle_id": "P-001", "confidence": 0.65})
print("=== Versioning ===")
print(f"  P-001 versions: {pv.get_latest_version('P-001')}")
print(f"  Regressed: {pv.has_regressed('P-001')}")
summary = pv.get_evolution_summary()
print(f"  Summary: {summary}")
assert pv.get_latest_version("P-001") == 3
assert pv.has_regressed("P-001") == True  # 0.65 < peak of 0.72
print()

# 6. Test contradiction detection
cd = ContradictionDetector()
principles = [
    {
        "principle_id": "P-001",
        "conditions": [{"feature": "penalty", "threshold": 0.5}],
        "action_override": {"max_penalty": 0.8},
        "confidence": 0.7,
    },
    {
        "principle_id": "P-002",
        "conditions": [{"feature": "penalty", "threshold": 0.5}],
        "action_override": {"max_penalty": -0.3},
        "confidence": 0.6,
    },
]
contras = cd.detect_contradictions(principles)
print("=== Contradiction Detection ===")
print(f"  Contradictions found: {len(contras)}")
if contras:
    print(f"  First: {contras[0]['principle_a']} vs {contras[0]['principle_b']}")
assert len(contras) == 1, "Should detect the contradiction"
print()

# 7. Test meta-principle merger
merger = MetaPrincipleMerger(filepath=None)
sps = []
for i in range(6):
    p = enc.encode({
        "principle_id": f"P-{i:03d}",
        "conditions": [{"feature": "n_samples", "op": "<", "threshold": 200}],
        "action_override": {"max_penalty": -0.3},
        "confidence": 0.6 + i * 0.02,
    })
    sps.append(p)
metas = merger.merge(sps)
print("=== Meta-Principle Merging ===")
print(f"  Meta-principles created: {len(metas)}")
if metas:
    print(f"  First meta: {metas[0].human_text}")
    print(f"  Children: {len(metas[0].children)}")
print()

# 8. Test abstraction safeguards
sg = AbstractionSafeguards()
if metas:
    check = sg.check_promotion(metas[0], sps, cd)
    print("=== Safeguard Check ===")
    print(f"  Approved: {check['approved']}")
    for gate, result in check["gates"].items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"    {gate}: {status} (value={result['value']}, threshold={result['threshold']})")
print()

# 9. Test foundation exploration targets
rng = np.random.default_rng(42)
targets = opt.foundation.get_foundation_targets({}, 10, rng)
print("=== Foundation Exploration Targets ===")
print(f"  Targets generated: {len(targets)}")
for t in targets[:2]:
    print(f"  - {t['description']}, weights shape: {t['feature_weights'].shape}")
print()

print("=" * 50)
print("ALL TESTS PASSED!")
print("=" * 50)
