"""Master UIL off-switch + connect pack (core wiring opt-in after Gate 2 fail)."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from figpso_uil.core.uil_mode import (
    apply_uil_mode, CONNECT_PACK_FLAGS, CORE_WIRING_FLAGS, ABLATION_KEYS,
)
from figpso_uil.core.optimizer import UniversalOptimizer
from figpso_uil.core.classifier_registry import CLASSIFIERS
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch


class _Cfg:
    def __init__(self, **kw):
        self.stages = kw.get("stages", 2)
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
        self.uil_enabled = kw.get("uil_enabled", True)
        self.uil_connect_pack = kw.get("uil_connect_pack", False)
        self.connect_cognition = False
        self.explore_architecture = False
        self.adaptive_min_evidence = False
        self.utility_gated_formulas = False
        self.signal_translator = False


def test_u1_apply_uil_off():
    cfg = _Cfg(uil_enabled=False, uil_connect_pack=True)  # off wins
    abl = {k: True for k in ABLATION_KEYS}
    out, meta = apply_uil_mode(cfg, abl)
    assert meta["mode"] == "uil_off"
    assert all(v is False for v in out.values())
    for f in CONNECT_PACK_FLAGS:
        assert getattr(cfg, f) is False
    print("  U1 uil_off forces ablation+connect flags False  PASS")


def test_u2_apply_connect_pack():
    cfg = _Cfg(uil_enabled=True, uil_connect_pack=True)
    abl = {k: False for k in ABLATION_KEYS}
    abl["principles"] = True
    out, meta = apply_uil_mode(cfg, abl)
    assert meta["mode"] == "uil_connect_pack"
    assert out["principles"] is True
    assert out["network"] is False
    for f in CONNECT_PACK_FLAGS:
        assert getattr(cfg, f) is True
    print("  U2 connect pack enables flags; preserves ablation  PASS")


def test_u3_default_leaves_wiring_off():
    """Gate 2 failed — default must NOT force translator ON."""
    cfg = _Cfg()
    abl = {k: True for k in ABLATION_KEYS}
    out, meta = apply_uil_mode(cfg, abl)
    assert meta["mode"] == "uil_default"
    assert out == abl
    for f in CORE_WIRING_FLAGS:
        assert getattr(cfg, f) is False, f"{f} must stay OFF by default"
    print("  U3 default leaves core wiring OFF  PASS")


def test_u4_optimizer_uil_off_ablation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 5))
    y = (X[:, 0] > 0).astype(int)
    sd = tempfile.mkdtemp(prefix="uil_off_")
    clf = CLASSIFIERS["SymbolicFormula"](0, 2, 10, sd)
    clf.apply_config({"learning_mode": "sparse_linear"})
    opt = UniversalOptimizer(
        cfg=_Cfg(uil_enabled=False, stages=1),
        seed=0,
        dataset_name="u4",
        fold=0,
        start_fold=0,
        classifier=clf,
        search=PrincipleGuidedSearch(),
        classifier_factory=CLASSIFIERS["SymbolicFormula"],
        classifier_build_args=(0, 2, 10, sd),
        state_dir=sd,
    )
    assert opt._uil_mode_meta["mode"] == "uil_off"
    assert all(v is False for v in opt.ablation.values())
    opt.fit(X, y)
    assert opt._observability_meta.get("uil_mode", {}).get("mode") == "uil_off"
    assert not opt._observability_meta.get("cognition_connected")
    assert not getattr(clf, "use_feature_weight_penalty", False)
    print("  U4 optimizer uil_off end-to-end  PASS")


def test_u5_optimizer_connect_pack_arms_flags():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 5))
    y = (X[:, 0] > 0).astype(int)
    sd = tempfile.mkdtemp(prefix="uil_conn_")
    clf = CLASSIFIERS["SymbolicFormula"](1, 2, 10, sd)
    clf.apply_config({"learning_mode": "sparse_linear"})
    opt = UniversalOptimizer(
        cfg=_Cfg(uil_enabled=True, uil_connect_pack=True, stages=2),
        seed=1,
        dataset_name="u5",
        fold=0,
        start_fold=0,
        classifier=clf,
        search=PrincipleGuidedSearch(),
        classifier_factory=CLASSIFIERS["SymbolicFormula"],
        classifier_build_args=(1, 2, 10, sd),
        state_dir=sd,
    )
    assert opt._uil_mode_meta["mode"] == "uil_connect_pack"
    assert opt.cfg.connect_cognition is True
    assert opt.cfg.signal_translator is True
    assert all(opt.ablation.values())
    opt.fit(X, y)
    assert opt._observability_meta.get("cognition_connected") is True
    assert opt._observability_meta.get("signal_translator") is not None
    print("  U5 connect pack arms learner channels  PASS")


if __name__ == "__main__":
    test_u1_apply_uil_off()
    test_u2_apply_connect_pack()
    test_u3_default_leaves_wiring_off()
    test_u4_optimizer_uil_off_ablation()
    test_u5_optimizer_connect_pack_arms_flags()
    print("All uil_mode tests passed")
