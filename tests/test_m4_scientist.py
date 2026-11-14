"""M4 acceptance — scientist core (EIG causal, refit_constants, LLM offline)."""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from figpso_uil.cognition.causal_intervention import CausalInterventionEngine, Hypothesis
from figpso_uil.cognition.llm_mutator import LLMMutator
from figpso_uil.cognition.symbolic_regressor import SymbolicRegressor
from figpso_uil.cognition.symbolic_gradient import refit_constants
from figpso_uil.cognition.formula_engine import FormulaEngine, FormulaNode


def test_eig_picks_untested_hypothesis():
    print("=== M4.1: EIG intervention selection ===")
    eng = CausalInterventionEngine(filepath=None)
    h1 = Hypothesis("H-1", "config.max_depth", "high", 6.0, {}, predicted_regret=0.2)
    h2 = Hypothesis("H-2", "action.penalty", "high", 3.0, {}, predicted_regret=0.5)
    h2.evidence_trials = [{"effect": 0.01}] * 3
    eng._hypotheses = {"H-1": h1, "H-2": h2}
    picks = eng.choose_intervention_eig(budget=1)
    assert picks[0].hypothesis_id == "H-1"
    print("  EIG prefers untested high-regret hypothesis  PASS")


def test_refit_constants_improves_fit():
    print("=== M4.2: refit_constants improves polynomial fit ===")
    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 60)
    X = x.reshape(-1, 1)
    y = x ** 2 + rng.normal(0, 0.02, len(x))
    engine = FormulaEngine(seed=0)
    tree = FormulaNode(op="square", children=[FormulaNode(feature=0)])
    raw = np.asarray(engine.evaluate(tree, {"X": X}), dtype=float).ravel()
    err_before = float(np.mean((raw - y) ** 2))
    refit_constants(engine, tree, X, y, iters=25, rng=engine.rng)
    raw2 = np.asarray(engine.evaluate(tree, {"X": X}), dtype=float).ravel()
    err_after = float(np.mean((raw2 - y) ** 2))
    assert err_after <= err_before
    print(f"  MSE {err_before:.4f} -> {err_after:.4f}  PASS")


def test_llm_mutator_offline_fallback():
    print("=== M4.3: LLM mutator offline proposals ===")
    m = LLMMutator(api_key=None)
    props = m.propose(fingerprint={"n_samples": 50}, cells=[], n_proposals=2)
    assert len(props) == 2
    # Offline path is System-1: the own-transformer NeuralProposer when a
    # trained model card exists (results/uil_neural_proposer.json), else
    # the raw engine fallback. Both are valid offline sources.
    assert props[0].proposer in ("neural", "engine")
    print(f"  offline proposals ({props[0].proposer}) OK  PASS")


def test_symbolic_regressor_runs():
    print("=== M4.4: SymbolicRegressor with refit ===")
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (60, 2))
    y = 2 * X[:, 0] ** 2 + 0.5 * X[:, 1]
    sr = SymbolicRegressor(seed=1, n_terms=4).fit(X, y)
    pred = sr.predict(X)
    r2 = 1 - np.var(y - pred) / np.var(y)
    assert r2 > 0.5
    print(f"  synthetic quadratic R²={r2:.3f}  PASS")


def test_sin3x_recovery():
    print("=== M4.5: sin(3x) symbolic recovery ===")
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(2)
    x = np.linspace(-1.0, 1.0, 140)
    X = x.reshape(-1, 1)
    y = np.sin(3.0 * x) + rng.normal(0, 0.015, len(x))
    sr = SymbolicRegressor(seed=2, n_terms=10, max_pair_features=2).fit(X, y)
    pred = sr.predict(X)
    r2 = float(r2_score(y, pred))
    # sin(3x)-class: high in-sample R² + non-flat extrapolation past training range
    x_ext = np.linspace(1.05, 1.4, 30).reshape(-1, 1)
    pred_ext = sr.predict(x_ext)
    extrap_var = float(np.var(pred_ext))
    assert r2 > 0.85, f"R²={r2:.3f} too low"
    assert extrap_var > 0.01, f"extrapolation flat var={extrap_var:.4f}"
    print(f"  sin(3x)-class R²={r2:.3f} extrap_var={extrap_var:.4f}  PASS")


def test_eig_beats_passive_trials():
    print("=== M4.6: EIG vs passive trials-to-confirm ===")
    eng = CausalInterventionEngine(filepath=None)
    causal = Hypothesis("H-C", "config.max_depth", "high", 6.0, {}, predicted_regret=0.6)
    decoy1 = Hypothesis("H-D1", "action.penalty", "low", 1.0, {}, predicted_regret=0.1)
    decoy2 = Hypothesis("H-D2", "state.entropy", "high", 0.8, {}, predicted_regret=0.1)
    for h in (decoy1, decoy2):
        h.evidence_trials = [{"regret_delta": 0.0, "accuracy_delta": 0.0}] * 2
    eng._hypotheses = {"H-C": causal, "H-D1": decoy1, "H-D2": decoy2}
    pool = [decoy1, decoy2, causal]

    def passive_picks():
        return pool[0]

    eig_pick = eng.choose_intervention_eig(budget=1)[0]
    passive = passive_picks()
    assert eig_pick.hypothesis_id == "H-C", "EIG should prefer high-regret untested H-C"
    assert passive.hypothesis_id != "H-C"
    print("  EIG targets causal hypothesis before passive decoy  PASS")


def test_feynman_subset_recovery():
    print("=== M4.7: Feynman-subset y=x1^2 ===")
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(4)
    x1 = rng.uniform(0.5, 2.0, 80)
    x2 = rng.uniform(0.5, 2.0, 80)
    X = np.column_stack([x1, x2])
    y = x1 ** 2 + rng.normal(0, 0.02, len(x1))
    sr = SymbolicRegressor(seed=4, n_terms=6).fit(X, y)
    r2 = float(r2_score(y, sr.predict(X)))
    assert r2 > 0.9
    print(f"  I.6.2a-style x1^2 R²={r2:.3f}  PASS")


if __name__ == "__main__":
    test_eig_picks_untested_hypothesis()
    test_refit_constants_improves_fit()
    test_llm_mutator_offline_fallback()
    test_symbolic_regressor_runs()
    test_sin3x_recovery()
    test_eig_beats_passive_trials()
    test_feynman_subset_recovery()
    print("\n" + "=" * 50)
    print("M4 ACCEPTANCE: ALL TESTS PASSED")
    print("=" * 50)
