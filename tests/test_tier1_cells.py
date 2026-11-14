"""Tier 1–3 smoke tests — KnowledgeCell wrappers + composite replay."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from figpso_uil.cognition.knowledge_cell import (
    FormulaCell, HypothesisCell, EdgeCell, PrincipleCell, cell_to_readable,
)
from figpso_uil.cognition.cell_store import build_composite_snapshot, cells_for_api
from figpso_uil.cognition.version_store import VersionStore
from figpso_uil.cognition.formula_engine import FormulaNode


def test_formula_cell_readable():
    cell = FormulaCell.from_dict({
        "formula_id": "F-1",
        "name": "quad",
        "tree": FormulaNode(op="square", children=[FormulaNode(feature=0)]).to_dict(),
        "confidence": 0.8,
        "status": "active",
    })
    txt = cell.to_readable()
    assert "quad" in txt and "square" in txt
    print("  FormulaCell.to_readable OK")


def test_hypothesis_cell_readable():
    cell = HypothesisCell.from_dict({
        "hypothesis_id": "H-1",
        "variable": "state.n_features",
        "direction": "high",
        "threshold": 10,
        "status": "pending",
        "causal_strength": 0.3,
    })
    assert "n_features" in cell.to_readable()
    print("  HypothesisCell.to_readable OK")


def test_composite_snapshot_includes_edges():
    from figpso_uil.adapters.xgboost_adapter import XGBoostAdapter
    from figpso_uil.core.optimizer import UniversalOptimizer
    from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch
    import numpy as np

    class Cfg:
        stages = 2
        n_particles = 4
        pso_iters = 4
        use_resource_budget = False
        cv_folds = 2

    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 5))
    y = (X[:, 0] > 0).astype(int)
    d = tempfile.mkdtemp()
    opt = UniversalOptimizer(
        cfg=Cfg(), seed=0, dataset_name="tier_snap",
        fold=0, start_fold=0,
        classifier=XGBoostAdapter(seed=0, num_classes=2, boosting_rounds=5),
        search=PrincipleGuidedSearch(),
        state_dir=d,
    )
    snap = build_composite_snapshot(opt)
    for key in ("node", "edge", "principle", "formula", "causal_hypothesis", "routing_rule", "episode"):
        assert key in snap, f"missing bucket {key}"
    print(f"  composite buckets: {list(snap.keys())} OK")


def test_cells_for_api():
    state = {
        "formula": {"F-1": {"formula_id": "F-1", "name": "t", "confidence": 0.5, "status": "active"}},
        "causal_hypothesis": {"H-1": {"hypothesis_id": "H-1", "variable": "x", "direction": "high", "status": "pending"}},
    }
    rows = cells_for_api(state)
    assert len(rows) == 2
    assert all(r["readable"] for r in rows)
    print("  cells_for_api OK")


if __name__ == "__main__":
    test_formula_cell_readable()
    test_hypothesis_cell_readable()
    test_composite_snapshot_includes_edges()
    test_cells_for_api()
    print("\nTIER 1-3 SMOKE: ALL PASSED")
