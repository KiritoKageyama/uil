"""Shared classifier-name -> ClassifierAdapter factory registry.

Used by both the webui backend (run_manager.py) and standalone
experiment scripts (experiments/ablation_harness.py) that need to build
a classifier from a name + config without depending on the webui's
FastAPI-based machinery -- ws_hub.py imports fastapi, so importing
run_manager.py just to get this dict would force a web framework
install onto environments (e.g. an HPC cluster) that only run
optimizer experiments and never serve the UI.
"""

import os

from figpso_uil.adapters.xgboost_adapter import XGBoostAdapter
from figpso_uil.adapters.random_forest_adapter import RandomForestAdapter
from figpso_uil.adapters.knn_adapter import KNNAdapter
from figpso_uil.adapters.symbolic_formula_adapter import SymbolicFormulaAdapter
from figpso_uil.adapters.mlp_adapter import MLPAdapter

CLASSIFIERS = {
    "XGBoost": lambda seed, n_classes, boosting_rounds, state_dir=None: XGBoostAdapter(
        seed=seed, num_classes=n_classes, boosting_rounds=boosting_rounds
    ),
    "RandomForest": lambda seed, n_classes, boosting_rounds, state_dir=None: RandomForestAdapter(
        seed=seed, num_classes=n_classes
    ),
    "KNN": lambda seed, n_classes, boosting_rounds, state_dir=None: KNNAdapter(
        seed=seed, num_classes=n_classes
    ),
    # Dense non-linear ceiling probe / honest baseline. NOT readable or
    # editable -- see adapters/mlp_adapter.py for what it trades away.
    "MLP": lambda seed, n_classes, boosting_rounds, state_dir=None: MLPAdapter(
        seed=seed, num_classes=n_classes
    ),
    # state_dir: only SymbolicFormula actually consumes this -- it owns
    # its own FormulaLibrary file, built BEFORE UniversalOptimizer exists
    # (the optimizer isolates everything else internally via its own
    # state_dir param, but a classifier is constructed by the caller
    # before that). Without this, every experiment trial would silently
    # share ONE canonical-path formula population across all
    # datasets/seeds, defeating the whole point of per-trial isolation.
    "SymbolicFormula": lambda seed, n_classes, boosting_rounds, state_dir=None: SymbolicFormulaAdapter(
        seed=seed, num_classes=n_classes,
        boosting_rounds=boosting_rounds,
        formula_library_path=(
            os.path.join(state_dir, "uil_symbolic_classifier_library.json")
            if state_dir else None
        ),
    ),
}

# gplearn — genetic-programming symbolic learner, registered only when the
# optional dep is present (same pattern as TabPFN). Added 2026-08-05 after it
# beat UIL's own symbolic regressor on 17 of 21 out-of-range cells with 0
# blow-ups vs 19 (results/extrapolation/SUMMARY.md).
#
# SCOPE: that evidence is about REGRESSION / extrapolation. This entry exposes
# gplearn's CLASSIFIER, for which there is no UIL benchmark yet. It is
# selectable by name so it can be measured; it deliberately does NOT replace
# SymbolicFormula on any routing strategy (see core/strategy_classifiers.py).
try:
    from figpso_uil.adapters.gplearn_adapter import GPLearnAdapter, _HAS_GPLEARN
    if _HAS_GPLEARN:
        CLASSIFIERS["GPLearn"] = (
            lambda seed, n_classes, boosting_rounds, state_dir=None:
            GPLearnAdapter(seed=seed, num_classes=n_classes)
        )
except Exception:
    pass

# Research step 7 — TabPFN registered only when the optional dep is present.
try:
    from figpso_uil.adapters.tabpfn_adapter import TabPFNAdapter, _HAS_TABPFN
    if _HAS_TABPFN:
        CLASSIFIERS["TabPFN"] = (
            lambda seed, n_classes, boosting_rounds, state_dir=None:
            TabPFNAdapter(seed=seed, num_classes=n_classes)
        )
except Exception:
    pass
