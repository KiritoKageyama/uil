"""Signal translator: cognition must flip predictions (existence CONNECTED)."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
from sklearn.datasets import load_breast_cancer

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from figpso_uil.cognition.signal_translator import translate, LearnerDirective
from figpso_uil.core.optimizer import UniversalOptimizer
from figpso_uil.core.classifier_registry import CLASSIFIERS
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch


class _Cfg:
    def __init__(self, **kw):
        self.stages = kw.get("stages", 3)
        self.n_particles = 4
        self.pso_iters = 3
        self.particles = 4
        self.pso_iters_per_stage = 3
        self.boosting_rounds = 10
        self.cv_folds = 1
        self.use_resource_budget = False
        self.disable_routing = True
        self.enable_stagnation_evolution = False
        self.target_accuracy = None
        self.max_stages = self.stages
        self.situation_adaptive = False
        self.uil_enabled = True
        self.uil_connect_pack = kw.get("uil_connect_pack", True)
        self.connect_cognition = False
        self.explore_architecture = False
        self.adaptive_min_evidence = False
        self.utility_gated_formulas = False
        self.signal_translator = False
        self.search_mode = "classic"


def _run(state_dir: str, ablation: dict, seed: int = 42) -> np.ndarray:
    X, y = load_breast_cancer(return_X_y=True)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    te = idx[:80]
    tr = idx[80:280]
    clf = CLASSIFIERS["SymbolicFormula"](seed, 2, 10, state_dir)
    clf.apply_config({"learning_mode": "sparse_linear"})
    search = PrincipleGuidedSearch()
    opt = UniversalOptimizer(
        classifier=clf,
        search=search,
        dataset_name=f"st_bc_{seed}",
        fold=0,
        start_fold=0,
        seed=seed,
        cfg=_Cfg(uil_connect_pack=True, stages=3),
        state_dir=state_dir,
        ablation=ablation,
        classifier_factory=CLASSIFIERS["SymbolicFormula"],
        classifier_build_args=(seed, 2, 10, state_dir),
    )
    opt.fit(X[tr], y[tr])
    return opt.predict(X[te])


def test_t1_translate_principles_concentrate():
    X = np.random.default_rng(0).normal(size=(40, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    flat = translate(
        n_features=6,
        ablation={"principles": False, "network": False, "formulas": True,
                  "episodic": False},
        X_train=X, y_train=y,
    )
    peaked = translate(
        n_features=6,
        ablation={"principles": True, "network": False, "formulas": True,
                  "episodic": False},
        principles=[{"principle_id": "P-1", "action_override": {"max_penalty": 3.0}}],
        overrides={"max_penalty": 3.0},
        triggered_principle_ids=["P-1"],
        X_train=X, y_train=y,
    )
    assert flat.feature_guidance is not None
    assert peaked.feature_guidance is not None
    assert float(np.std(peaked.feature_guidance)) > float(np.std(flat.feature_guidance))
    assert "principles" in peaked.sources


def test_t2_translate_formulas_ablated_strips_basis():
    d_on = translate(
        n_features=4,
        ablation={"principles": False, "network": False, "formulas": True,
                  "episodic": False},
    )
    d_off = translate(
        n_features=4,
        ablation={"principles": False, "network": False, "formulas": False,
                  "episodic": False},
    )
    assert d_on.use_symbolic_basis is True
    assert d_off.use_symbolic_basis is False
    assert d_off.prefer_formula_ids == []


def test_t3_formulas_ablation_flips_predictions():
    """With translator on, formulas OFF must change predictions vs ON."""
    with tempfile.TemporaryDirectory() as d_on, tempfile.TemporaryDirectory() as d_off:
        base = {k: True for k in (
            "principles", "network", "formulas", "causal", "episodic",
            "structural", "meta_principles", "safeguards", "goal",
        )}
        off = dict(base)
        off["formulas"] = False
        p_on = _run(d_on, base)
        p_off = _run(d_off, off)
        assert p_on.shape == p_off.shape
        assert not np.array_equal(p_on, p_off), (
            "formulas still INERT under signal_translator — basis strip failed"
        )


def test_t4_principles_ablation_flips_predictions():
    """With translator + connect_cognition, principles OFF must move preds.

    Isolate the principles channel (network/episodic/causal off) so a
    wash from other subsystems cannot hide the concentration actuator.
    """
    with tempfile.TemporaryDirectory() as d_on, tempfile.TemporaryDirectory() as d_off:
        base = {k: True for k in (
            "principles", "network", "formulas", "causal", "episodic",
            "structural", "meta_principles", "safeguards", "goal",
        )}
        base["network"] = False
        base["episodic"] = False
        base["causal"] = False
        off = dict(base)
        off["principles"] = False
        p_on = _run(d_on, base)
        p_off = _run(d_off, off)
        assert p_on.shape == p_off.shape
        assert not np.array_equal(p_on, p_off), (
            "principles still INERT under signal_translator"
        )


def test_t5_directive_summary():
    d = LearnerDirective(
        feature_guidance=np.array([0.1, 0.9, 0.2]),
        prefer_formula_ids=["F-1"],
        use_symbolic_basis=True,
        sources=["principles"],
    )
    s = d.summary()
    assert s["n_prefer_formulas"] == 1
    assert s["use_symbolic_basis"] is True


if __name__ == "__main__":
    test_t1_translate_principles_concentrate()
    print("  T1 concentrate  PASS")
    test_t2_translate_formulas_ablated_strips_basis()
    print("  T2 formulas strip  PASS")
    test_t5_directive_summary()
    print("  T5 summary  PASS")
    test_t3_formulas_ablation_flips_predictions()
    print("  T3 formulas CONNECTED  PASS")
    test_t4_principles_ablation_flips_predictions()
    print("  T4 principles CONNECTED  PASS")
    print("All signal_translator tests passed")
