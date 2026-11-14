"""M0 acceptance test — v22 blueprint §4 (epistemology repair).

Verifies the four M0 changes:
  M0.1  Regret/mining labels come from multi-split validation:
        journal entries carry fold_scores/fold_spread and accuracy is
        the mean across splits (cv_folds=3 default).
  M0.2  Tiny-data mining: the miner engages between 12 and 39 journal
        entries with shallow, confidence-shrunk rules instead of sleeping.
  M0.3  Multiclass runs green: custom formula objective is a binary-only
        capability; num_classes>2 no longer needs ablation={"formulas": False}.
  M0.4  PSO detachment is earned by demonstrated principle prediction
        skill, not knowledge quantity.

Run from a workspace where `figpso_uil` resolves to the AI copy:
    python figpso_uil/tests/test_m0_epistemology.py
"""

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from figpso_uil.adapters.xgboost_adapter import XGBoostAdapter
from figpso_uil.cognition.uil_miner import UIL_Miner
from figpso_uil.core.optimizer import UniversalOptimizer
from figpso_uil.search.hybrid_foundation import KnowledgeFoundationBuilder
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch


class Cfg:
    stages = 14
    n_particles = 8
    pso_iters = 8
    boosting_rounds = 15
    use_resource_budget = False
    disable_routing = True  # M0 tests pin XGBoost explicitly


def make_tiny_dataset(n=60, f=10, seed=42):
    """n=60 binary dataset — the regime where the miner used to sleep."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, f))
    signal = 0.7 * X[:, 0] + 0.5 * X[:, 1] - 0.4 * X[:, 2]
    y = (signal + rng.normal(0, 0.4, n) > 0).astype(int)
    return X, y


def run_optimizer(X, y, dataset_name, state_dir, stages=14):
    num_classes = len(np.unique(y))
    cfg = Cfg()
    cfg.stages = stages
    opt = UniversalOptimizer(
        cfg=cfg, seed=42, dataset_name=dataset_name,
        fold=0, start_fold=0,
        classifier=XGBoostAdapter(seed=42, num_classes=num_classes,
                                  boosting_rounds=15),
        search=PrincipleGuidedSearch(),
        state_dir=state_dir,
    )
    return opt, opt.fit(X, y)


def test_m01_kfold_provenance_and_m02_tiny_mining():
    print("=== M0.1 + M0.2: K-fold provenance & tiny-data mining (n=60 e2e) ===")
    X, y = make_tiny_dataset()
    state_dir = tempfile.mkdtemp(prefix="uil_m0_")
    opt, result = run_optimizer(X, y, "m0_tiny_60x10", state_dir)

    journal_path = os.path.join(state_dir, "uil_journal_v19.json")
    with open(journal_path) as fh:
        entries = [e for e in json.load(fh)
                   if e.get("classifier") != "FAILSAFE"]
    assert len(entries) >= 12, f"need >=12 journal entries, got {len(entries)}"

    # M0.1: every entry carries fold provenance; accuracy == mean(folds)
    for e in entries:
        out = e["outcome"]
        assert "fold_scores" in out, "journal entry missing fold_scores"
        assert "fold_spread" in out, "journal entry missing fold_spread"
        assert len(out["fold_scores"]) == 3, (
            f"cv_folds=3 default should give 3 fold scores, "
            f"got {len(out['fold_scores'])}"
        )
        assert abs(out["accuracy"] - np.mean(out["fold_scores"])) < 0.01, (
            "outcome accuracy is not the mean of its fold scores"
        )
    spreads = [e["outcome"]["fold_spread"] for e in entries]
    print(f"  entries: {len(entries)}, mean fold spread: {np.mean(spreads):.4f}")
    print("  M0.1 PASS: all entries carry 3-fold provenance, accuracy = fold mean")

    # M0.2 (acceptance: miner produces >0 principles at n=60).
    n_principles = len(opt.library.get_principles())
    print(f"  principles in library after run: {n_principles}")
    assert n_principles > 0, (
        "miner produced 0 principles at n=60 — tiny-data mode not engaging"
    )
    print("  M0.2 PASS: miner engaged below 40 entries")


def test_m02_tiny_mining_unit():
    """Deterministic miner-level check, independent of optimizer dynamics."""
    print("=== M0.2 unit: miner engages at 16 entries, sleeps below 12 ===")
    rng = np.random.default_rng(0)

    def entry(i, regret):
        return {
            "dataset": "synth", "classifier": "XGBoost", "stage": i,
            "state": {"n_samples": 60, "n_features": 10,
                      "entropy": float(rng.uniform(0.5, 1.5))},
            "action": {"penalty": float(1.0 + (i % 4)),
                       "classifier_config": {"max_depth": 3 + (i % 3)}},
            "outcome": {"accuracy": 0.8 - regret},
            "regret": regret,
        }

    # Regret pattern correlated with penalty so a depth-2 tree has signal
    entries = [entry(i, 0.15 if (i % 4) >= 2 else 0.0) for i in range(16)]
    miner = UIL_Miner()

    rules = miner.mine_within_dataset(entries, "synth")
    print(f"  rules mined from 16 entries: {len(rules)}")
    assert len(rules) > 0, "tiny mode produced no rules on separable regret"
    for r in rules:
        assert r.get("tiny_data") is True, "tiny rule not flagged tiny_data"
        assert r["confidence"] <= 0.95, "tiny rule confidence not shrunk"
        print(f"    conditions={r['conditions']} conf={r['confidence']}"
              f" evidence={r['evidence_count']}")

    assert miner.mine_within_dataset(entries[:11], "synth") == [], (
        "miner should still sleep below TINY_MIN_ENTRIES"
    )
    print("  M0.2 unit PASS")


def test_m03_multiclass_green():
    print("=== M0.3: multiclass capability guard + e2e run ===")
    assert XGBoostAdapter(seed=1, num_classes=3).supports_custom_objective \
        is False, "multiclass adapter must not claim custom-objective support"
    assert XGBoostAdapter(seed=1, num_classes=2).supports_custom_objective \
        is True, "binary custom-objective capability regressed"

    # e2e: 3-class run with formulas ENABLED (this used to crash)
    rng = np.random.default_rng(7)
    X = rng.standard_normal((90, 8))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0.5).astype(int) + \
        (X[:, 0] - 0.5 * X[:, 1] > 0.5).astype(int)  # 3 classes
    assert len(np.unique(y)) == 3
    state_dir = tempfile.mkdtemp(prefix="uil_m0_mc_")
    _, result = run_optimizer(X, y, "m0_multiclass_90x8", state_dir, stages=4)
    print(f"  multiclass accuracy: {result['accuracy']:.3f}")
    assert result["accuracy"] > 0.3
    print("  M0.3 PASS: multiclass runs green with formulas enabled")


def test_m04_skill_keyed_detachment():
    print("=== M0.4: detachment earned by prediction skill ===")
    rich = dict(n_principles=60, n_episodes=25, n_formulas=15,
                miner_r2=0.85, network_nodes=40, recent_accuracy=0.92)

    # 1. Rich knowledge, NO demonstrated skill → capped below MATURE.
    f1 = KnowledgeFoundationBuilder()
    m = f1.update_maturity(**rich)
    print(f"  rich knowledge, no outcomes: maturity={m:.3f}")
    assert m < f1.MATURE_THRESHOLD, "accumulation alone must not detach PSO"

    # 2. Rich knowledge + principles that demonstrably win → EXPERT, 0 PSO.
    f2 = KnowledgeFoundationBuilder()
    for _ in range(30):
        f2.record_principle_outcome(True)
    m = f2.update_maturity(**rich)
    alloc = f2.allocate_actions()
    print(f"  rich + 30/30 helped: maturity={m:.3f}, stage={alloc['stage']},"
          f" pso={alloc['pso_actions']}")
    assert m >= f2.MATURE_THRESHOLD, "demonstrated skill should reach expert"
    assert alloc["stage"] == "expert" and alloc["pso_actions"] == 0, (
        "expert stage must run zero PSO"
    )

    # 3. Rich knowledge + principles that mostly LOSE → capped.
    f3 = KnowledgeFoundationBuilder()
    for i in range(30):
        f3.record_principle_outcome(i % 3 == 0)  # ~33% hit rate
    m = f3.update_maturity(**rich)
    print(f"  rich + 33% hit rate: maturity={m:.3f}")
    assert m < f3.MATURE_THRESHOLD, "losing principles must not detach PSO"
    print("  M0.4 PASS: detachment keyed to demonstrated skill")


if __name__ == "__main__":
    test_m02_tiny_mining_unit()
    test_m04_skill_keyed_detachment()
    test_m03_multiclass_green()
    test_m01_kfold_provenance_and_m02_tiny_mining()
    print("\n" + "=" * 50)
    print("M0 ACCEPTANCE: ALL TESTS PASSED")
    print("=" * 50)
