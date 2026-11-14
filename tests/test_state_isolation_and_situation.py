"""Bug #30b isolation + situation_policy unit checks."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np

from figpso_uil.cognition.situation_policy import recommend, apply_to_cfg
from figpso_uil.cognition.paths import results_path
from figpso_uil.core.optimizer import UniversalOptimizer
from figpso_uil.core.classifier_registry import CLASSIFIERS
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch


class _Cfg:
    stages = 2
    n_particles = 4
    pso_iters = 4
    particles = 4
    pso_iters_per_stage = 4
    boosting_rounds = 10
    cv_folds = 1
    use_resource_budget = False
    disable_routing = True
    enable_stagnation_evolution = False
    target_accuracy = None
    max_stages = 2
    search_mode = "classic"
    situation_adaptive = False


def test_i1_fingerprint_and_routing_stay_in_state_dir():
    """Isolated fit must not write the five shared leak files into results/."""
    shared = {
        "uil_fingerprints_v22.json": _mtime_or_0(results_path("uil_fingerprints_v22.json")),
        "uil_routing_memory.json": _mtime_or_0(results_path("uil_routing_memory.json")),
        "uil_governance_manifest.json": _mtime_or_0(results_path("uil_governance_manifest.json")),
        "uil_graph_changelog_v20.json": _mtime_or_0(results_path("uil_graph_changelog_v20.json")),
        "uil_formula_changelog_v20.json": _mtime_or_0(results_path("uil_formula_changelog_v20.json")),
        "uil_live_feed.jsonl": _mtime_or_0(results_path("uil_live_feed.jsonl")),
    }
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    sd = tempfile.mkdtemp(prefix="iso_")
    clf = CLASSIFIERS["SymbolicFormula"](0, 2, 10, sd)
    clf.apply_config({"learning_mode": "sparse_linear"})
    opt = UniversalOptimizer(
        cfg=_Cfg(), seed=0, dataset_name="iso_test", fold=0, start_fold=0,
        classifier=clf, search=PrincipleGuidedSearch(),
        classifier_factory=CLASSIFIERS["SymbolicFormula"],
        classifier_build_args=(0, 2, 10, sd), state_dir=sd,
    )
    opt.fit(X, y)

    assert os.path.isfile(os.path.join(sd, "uil_fingerprints_v22.json"))
    assert os.path.isfile(os.path.join(sd, "uil_routing_memory.json"))
    assert os.path.isfile(os.path.join(sd, "uil_governance_manifest.json"))
    # Shared defaults must be untouched (mtime unchanged or still absent).
    for name, before in shared.items():
        after = _mtime_or_0(results_path(name))
        assert after == before, f"leak into results/{name}: {before} -> {after}"


def test_i2_situation_policy_respects_off_and_applies_when_on():
    class C:
        situation_adaptive = False
        stress_adaptive_blend = False
        maturity_soft_normalization = False
        explore_architecture = False

    off = recommend({"interaction_gap": 0.1}, C(), search_mode="classic")
    assert apply_to_cfg(C(), off) == []

    class On:
        situation_adaptive = True
        stress_adaptive_blend = False
        maturity_soft_normalization = False
        explore_architecture = False

    cfg = On()
    rec = recommend({"interaction_gap": 0.12}, cfg, search_mode="classic")
    applied = apply_to_cfg(cfg, rec)
    assert "stress_adaptive_blend" in applied
    assert "maturity_soft_normalization" in applied
    assert "explore_architecture" in applied
    assert cfg.stress_adaptive_blend is True
    assert any("Category D" in r for r in rec["_reasons"])


def test_i3_operator_lock_blocks_situation_enable():
    class Locked:
        situation_adaptive = True
        stress_adaptive_blend = False
        maturity_soft_normalization = False
        explore_architecture = False
        _situation_lock = frozenset({"stress_adaptive_blend"})

    cfg = Locked()
    rec = recommend({}, cfg, search_mode="classic")
    apply_to_cfg(cfg, rec)
    assert cfg.stress_adaptive_blend is False
    assert cfg.maturity_soft_normalization is True


def _mtime_or_0(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


if __name__ == "__main__":
    test_i2_situation_policy_respects_off_and_applies_when_on()
    test_i3_operator_lock_blocks_situation_enable()
    print("I2 I3 OK — running I1 (short fit)...")
    test_i1_fingerprint_and_routing_stay_in_state_dir()
    print("ALL isolation/situation tests passed")
