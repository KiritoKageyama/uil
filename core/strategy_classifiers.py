"""Map domain-router strategies to classifier adapters (v22 §3)."""

from __future__ import annotations

import os
from typing import Callable

from figpso_uil.adapters.knn_adapter import KNNAdapter
from figpso_uil.adapters.random_forest_adapter import RandomForestAdapter
from figpso_uil.adapters.symbolic_formula_adapter import SymbolicFormulaAdapter
from figpso_uil.adapters.xgboost_adapter import XGBoostAdapter

# strategy_key -> default classifier name
STRATEGY_TO_CLASSIFIER: dict[str, str] = {
    "grad_fit": "SymbolicFormula",
    "gbm": "XGBoost",
    "symbolic": "SymbolicFormula",
    # The blueprint always described sparse_linear as "sparse linear over a
    # symbolic basis (readable)"; it mapped to XGBoost only because that
    # learner didn't exist yet. It now does — SymbolicFormula's
    # sparse_linear mode (raw + symbolic + stump basis, L1-fit). Measured
    # spambase 0.685->0.949 (beats XGBoost 0.944), a readable weighted sum.
    "sparse_linear": "SymbolicFormula",
    "fewshot": "SymbolicFormula",
    # Research step 7 — optional TabPFN strategy (never default learner
    # when routing is disabled / user picks SymbolicFormula).
    "tabpfn": "TabPFN",
    # gplearn (2026-08-05). Selectable, but NOT wired to any strategy the
    # router currently emits.
    #
    # WHY NOT `"symbolic": "GPLearn"`: the measurement that motivated adding
    # gplearn is about REGRESSION / out-of-range recovery, where it beat UIL's
    # own symbolic regressor 17-4 with 0 blow-ups vs 19
    # (results/extrapolation/SUMMARY.md). There is NO benchmark of gplearn's
    # CLASSIFIER against SymbolicFormula on this project's datasets. Flipping
    # a routing default on evidence from a different task is exactly what E6
    # did (auto-enable rule basis on an interaction-gap signal) and it
    # regressed ionosphere 0.916 -> 0.868. Point this key at "GPLearn" only
    # after a Gate 2, and note the trade: gplearn programs are readable but
    # carry no FormulaLibrary provenance, no VersionStore cells, and
    # cognition/law_export.py does not apply to them.
    "symbolic_gp": "GPLearn",
}

# Per-strategy adapter kwargs (D3 few-shot, D4 sparse readable).
# Basis-family gates express the fingerprint→architecture choice: stumps
# help axis-aligned tabular (spambase) but hurt covertype, so leave them
# on by default here and let router tuning specialize per micro-domain.
STRATEGY_KWARGS: dict[str, dict] = {
    "fewshot": {"learning_mode": "single", "boosting_rounds": 6},
    "sparse_linear": {"learning_mode": "sparse_linear"},
    "symbolic": {"learning_mode": "single", "boosting_rounds": 4},
}


def build_classifier_for_strategy(
    strategy: str,
    *,
    seed: int,
    num_classes: int,
    boosting_rounds: int = 30,
    state_dir: str | None = None,
):
    """Instantiate the classifier adapter for a routing strategy."""
    name = STRATEGY_TO_CLASSIFIER.get(strategy, "XGBoost")
    extra = STRATEGY_KWARGS.get(strategy, {})
    factories: dict[str, Callable] = {
        "XGBoost": lambda: XGBoostAdapter(
            seed=seed, num_classes=num_classes, boosting_rounds=boosting_rounds,
            max_depth=extra.get("max_depth", 6),
            learning_rate=extra.get("learning_rate", 0.1),
        ),
        "KNN": lambda: KNNAdapter(
            seed=seed, num_classes=num_classes,
            n_neighbors=extra.get("n_neighbors", 5),
        ),
        "RandomForest": lambda: RandomForestAdapter(seed=seed, num_classes=num_classes),
        "SymbolicFormula": lambda: SymbolicFormulaAdapter(
            seed=seed,
            num_classes=num_classes,
            boosting_rounds=extra.get("boosting_rounds", boosting_rounds),
            learning_mode=extra.get("learning_mode", "auto"),
            formula_library_path=(
                os.path.join(state_dir, "uil_symbolic_classifier_library.json")
                if state_dir else None
            ),
        ),
    }
    if name == "TabPFN":
        from figpso_uil.adapters.tabpfn_adapter import TabPFNAdapter
        return TabPFNAdapter(seed=seed, num_classes=num_classes), "TabPFN"
    if name == "GPLearn":
        # Optional dep: fall back rather than break a run whose router
        # happened to pick it in an environment without gplearn.
        try:
            from figpso_uil.adapters.gplearn_adapter import GPLearnAdapter
            return GPLearnAdapter(seed=seed, num_classes=num_classes), "GPLearn"
        except ImportError:
            name = "SymbolicFormula"
    factory = factories.get(name, factories["XGBoost"])
    clf = factory()
    return clf, name
