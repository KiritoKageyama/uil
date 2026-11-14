"""Formula pyramid: parallel isolated engines + DT merge stay deterministic.

Run:
    python figpso_uil/tests/test_formula_pyramid.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from figpso_uil.cognition.formula_engine import FormulaEngine, FormulaNode
from figpso_uil.cognition.formula_pyramid import (
    PyramidParent,
    _mix_seed,
    _tree_key,
    run_formula_pyramid,
)

_FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        _FAILURES.append(name)


def _toy_parents(n_features: int = 6) -> list[PyramidParent]:
    engine = FormulaEngine(seed=0)
    seeds = engine.get_seed_classifier_formulas(n_features)
    out = []
    for i, (name, tree) in enumerate(list(seeds.items())[:3]):
        out.append(PyramidParent(formula_id=f"P-{i}", name=name, tree=tree))
    return out


def _toy_xy(n: int = 80, d: int = 6, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    # Weak planted signal so DT importances are not pure noise.
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
    return X, y


def test_p1_parallel_matches_serial():
    """Completion order must not change survivors (bug #31 class)."""
    parents = _toy_parents()
    X, y = _toy_xy()
    kwargs = dict(
        parents=parents,
        master_seed=42,
        stress=0.5,
        n_features=X.shape[1],
        feature_bias=np.ones(X.shape[1]),
        X=X,
        y=y,
        n_workers=4,
        n_admit=2,
    )
    a = run_formula_pyramid(**kwargs, parallel=True)
    b = run_formula_pyramid(**kwargs, parallel=False)
    keys_a = [_tree_key(c.tree) for c in a]
    keys_b = [_tree_key(c.tree) for c in b]
    check("P1 parallel==serial", len(a) == len(b) >= 1 and keys_a == keys_b,
          f"n={len(a)}/{len(b)}")


def test_p2_same_seed_byte_identical():
    parents = _toy_parents()
    X, y = _toy_xy()
    kwargs = dict(
        parents=parents,
        master_seed=123,
        stress=0.7,
        n_features=X.shape[1],
        feature_bias=None,
        X=X,
        y=y,
        n_workers=6,
        n_admit=2,
        parallel=True,
    )
    a = run_formula_pyramid(**kwargs)
    b = run_formula_pyramid(**kwargs)
    check(
        "P2 same seed identical",
        [_tree_key(c.tree) for c in a] == [_tree_key(c.tree) for c in b]
        and [c.worker_id for c in a] == [c.worker_id for c in b],
    )


def test_p3_worker_seeds_are_isolated():
    """Two workers must not share one advancing RNG (different mix seeds)."""
    s0, s1 = _mix_seed(42, 0, 0xF0A1), _mix_seed(42, 1, 0xF0A1)
    e0 = FormulaEngine(seed=s0)
    e1 = FormulaEngine(seed=s1)
    base = FormulaNode(op="sigmoid", children=[FormulaNode(feature=0)])
    e0.mutate(base, 0.6, n_features=6)
    e1.mutate(base, 0.6, n_features=6)
    check("P3 worker seeds isolated", s0 != s1, f"{s0} vs {s1}")


def test_p4_admits_at_most_n_admit():
    parents = _toy_parents()
    X, y = _toy_xy()
    out = run_formula_pyramid(
        parents=parents,
        master_seed=9,
        stress=0.4,
        n_features=X.shape[1],
        feature_bias=np.ones(X.shape[1]),
        X=X,
        y=y,
        n_workers=8,
        n_admit=1,
        parallel=False,
    )
    check("P4 n_admit cap", len(out) <= 1, f"got {len(out)}")


if __name__ == "__main__":
    print("Formula pyramid tests")
    test_p1_parallel_matches_serial()
    test_p2_same_seed_byte_identical()
    test_p3_worker_seeds_are_isolated()
    test_p4_admits_at_most_n_admit()
    if _FAILURES:
        print(f"\n{len(_FAILURES)} FAILED: {_FAILURES}")
        sys.exit(1)
    print("\nAll P1-P4 passed.")
