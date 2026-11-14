"""Research steps 6–8: Thresholdout, TabPFN probe wiring, KAN basis."""

from __future__ import annotations

import numpy as np

from figpso_uil.cognition.thresholdout import beats, beats_bool
from figpso_uil.adapters.symbolic_formula_adapter import SymbolicFormulaAdapter


def test_t1_thresholdout_deterministic():
    a = beats(0.90, 0.88, seed=42, query_id="champ:ds")
    b = beats(0.90, 0.88, seed=42, query_id="champ:ds")
    assert a == b
    assert a["beats"] is True  # +0.02 > tau 0.01 even with small noise


def test_t2_thresholdout_rejects_null_ab():
    """Known-null A/B: identical scores must not 'beat' under Thresholdout."""
    # Many query ids — a null should almost never clear tau=0.01 with
    # noise_scale=0.005. Allow at most rare false positives.
    wins = sum(
        1 for q in range(200)
        if beats_bool(0.85, 0.85, seed=7, query_id=q)
    )
    assert wins < 30, f"null A/B accepted too often: {wins}/200"


def test_t3_thresholdout_clear_win_accepted():
    # Large margin must clear tau despite noise.
    assert beats_bool(0.95, 0.80, seed=1, query_id=0) is True


def test_k1_kan_basis_readable_and_fits():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    # Nonlinear univariate signal: (x0 - 0)^2 style
    y = (X[:, 0] ** 2 + 0.3 * X[:, 1] > 1.0).astype(int)
    clf = SymbolicFormulaAdapter(seed=0, num_classes=2, boosting_rounds=5)
    clf.apply_config({
        "learning_mode": "sparse_linear",
        "use_kan_basis": True,
        "use_stump_basis": False,
        "use_rule_basis": False,
        "use_neural_basis": False,
        "sparse_l1_c": 0.5,
    })
    model = clf.fit_weighted(X, y, np.ones(6))
    assert model.get("kind") == "sparse_linear"
    kinds = {e.get("type") for e in model.get("basis", [])}
    assert "kan" in kinds or "raw" in kinds  # L1 may keep raw; kan must be available
    # Spec must include kan terms when building with flag on
    spec = clf._build_basis_spec(X, y, 6)
    assert any(e.get("type") == "kan" for e in spec)
    text = clf.readable_sparse_linear(model)
    assert "sigmoid" in text
    pred = clf.predict(model, X)
    assert len(np.asarray(pred)) == len(y)


def test_k2_kan_race_axis_present():
    clf = SymbolicFormulaAdapter(seed=0, num_classes=2)
    clf.apply_config({"learning_mode": "sparse_linear"})
    axes = clf.get_architecture_axes()
    assert "use_kan_basis" in axes


def test_f6_logging_policy_footnote():
    """F6: mined rules carry a propensity / logging-policy footnote."""
    from figpso_uil.cognition.uil_miner import UIL_Miner
    from figpso_uil.cognition.uil_library import UIL_Library
    import tempfile, os

    # Plant enough journal rows with known action distribution.
    entries = []
    rng = np.random.default_rng(0)
    for i in range(80):
        pen = float(rng.uniform(0.5, 2.5))
        # High penalty → high regret (planted so the tree splits).
        regret = 0.1 + 0.4 * (pen > 1.5) + float(rng.normal(0, 0.02))
        entries.append({
            "dataset": "toy",
            "classifier": "SymbolicFormula",
            "state": {"entropy": float(rng.uniform(0.3, 0.7)), "n_samples": 100.0},
            "action": {
                "penalty": pen,
                "classifier_config": {
                    "sparse_l1_c": 0.1 if i % 2 == 0 else 0.5,
                    "learning_mode": "sparse_linear",
                },
            },
            "regret": regret,
            "search_mode": "classic",
        })

    miner = UIL_Miner()
    rules = miner.mine(entries, stress_level=0.5, dataset_filter="toy")
    assert rules, "expected at least one high-regret rule from planted data"
    for rule in rules:
        lp = rule.get("logging_policy")
        assert isinstance(lp, dict), rule
        assert lp.get("n_journal_entries") == 80
        assert "action.penalty" in lp
        assert lp["action.penalty"]["n_distinct"] >= 2
        assert "config.sparse_l1_c" in lp
        assert lp["config.sparse_l1_c"]["n_distinct"] == 2
        assert "sparse_linear" in (lp.get("config.learning_mode") or {}).get("values", [])

    # Library must persist the footnote onto the principle.
    sd = tempfile.mkdtemp(prefix="f6_")
    lib = UIL_Library(library_path=os.path.join(sd, "principles.json"))
    lib.update_principles(rules, min_evidence=3)
    principles = lib.get_principles()
    mined = [p for p in principles if p.get("logging_policy")]
    assert mined, "principle missing logging_policy after update"
    assert mined[0]["logging_policy"]["n_journal_entries"] == 80


if __name__ == "__main__":
    test_t1_thresholdout_deterministic()
    test_t2_thresholdout_rejects_null_ab()
    test_t3_thresholdout_clear_win_accepted()
    print("T1–T3 Thresholdout OK")
    test_k1_kan_basis_readable_and_fits()
    test_k2_kan_race_axis_present()
    print("K1–K2 KAN OK")
    test_f6_logging_policy_footnote()
    print("F6 logging_policy OK")
    print("All research 6–8 + F6 unit tests passed")
