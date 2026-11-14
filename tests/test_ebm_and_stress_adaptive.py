"""Quick unit checks: stress adaptive blend + EBM cyclic fit."""
from __future__ import annotations
import os, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from figpso_uil.cognition.uil_stress import StressEngine
from figpso_uil.adapters.symbolic_formula_adapter import SymbolicFormulaAdapter


def test_stress_adaptive():
    print("--- stress adaptive blend (Category D) ---")
    fixed = StressEngine(adaptive_blend=False)
    adapt = StressEngine(adaptive_blend=True)
    # Easy data: acc=0.99 → deficit≈0; miner_r2=0 → explanation dead
    for eng in (fixed, adapt):
        eng._decision_stress_history = [0.1, 0.1, 0.1]
        eng.update_system_stress(0.1, current_accuracy=0.99, miner_r2_score=0.0)
    fc = fixed._last_pressure_components
    ac = adapt._last_pressure_components
    assert abs(fc["w_deficit"] - 0.25) < 1e-9 if "w_deficit" in fc else True
    # adaptive should zero deficit + explanation and renormalize
    assert ac["w_deficit"] == 0.0, ac
    assert ac["w_explanation_failure"] == 0.0, ac
    assert abs(ac["w_recent"] + ac["w_stagnation"] - 1.0) < 1e-9, ac
    print("PASS: adaptive zeros dead deficit/explanation and renormalizes")


def test_ebm_fit():
    print("--- EBM cyclic fit (research step 5) ---")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    with tempfile.TemporaryDirectory() as td:
        clf = SymbolicFormulaAdapter(
            seed=0, num_classes=2,
            formula_library_path=os.path.join(td, "lib.json"),
        )
        clf.apply_config({
            "learning_mode": "sparse_linear",
            "use_ebm_fit": True,
            "use_stump_basis": False,
            "use_rule_basis": False,
            "use_neural_basis": False,
            "ebm_rounds": 20,
        })
        m1 = clf.fit_weighted(X, y, feature_weights=np.ones(6))
        m2 = clf.fit_weighted(X, y, feature_weights=np.ones(6))
        assert m1.get("fit") == "ebm", m1.get("fit")
        assert m1["kind"] == "sparse_linear"
        # deterministic
        assert m1["coef"] == m2["coef"], (m1["coef"], m2["coef"])
        p = clf.predict(m1, X)
        assert set(np.unique(p)).issubset({0, 1})
        print(f"PASS: ebm fit deterministic, {len(m1['coef'])} terms, "
              f"train acc≈{float(np.mean(p == y)):.3f}")


if __name__ == "__main__":
    test_stress_adaptive()
    test_ebm_fit()
    print("All OK")
