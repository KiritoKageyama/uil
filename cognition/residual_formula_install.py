"""Residual-guided formula install — offline trial-and-error only.

Step 1 of direct formula influence. Builds formulas by mutate → score on
this dataset's residual/target → keep winners. **No API key, no network.**

Uses ``cognition/offline_formula_search.py``. The closed-vocab
``SymbolicRegressor`` path remains available as ``method="greedy"`` for
ablations; product default is ``method="trial"``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from figpso_uil.cognition.formula_engine import FormulaNode
from figpso_uil.cognition.offline_formula_search import trial_and_error_search
from figpso_uil.cognition.symbolic_regressor import SymbolicRegressor, _corr


@dataclass
class InstalledTerm:
    tree: FormulaNode
    gain: float
    formula_id: str | None = None
    mutation_type: str = "offline_trial"


def grow_residual_terms(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    n_terms: int = 3,
    baseline_scores: np.ndarray | None = None,
    method: str = "trial",
    n_trials: int = 64,
    feature_bias: np.ndarray | None = None,
    parents: list[FormulaNode] | None = None,
) -> list[InstalledTerm]:
    """Fit formulas on this dataset's residual/target.

    method:
      - ``"trial"`` (default): offline FormulaEngine trial-and-error
      - ``"greedy"``: SymbolicRegressor closed-vocab (no mutations)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] < 8 or n_terms < 1:
        return []

    if method == "greedy":
        return _grow_greedy(X, y, seed=seed, n_terms=n_terms,
                            baseline_scores=baseline_scores)

    results = trial_and_error_search(
        X, y,
        seed=seed,
        n_features=X.shape[1],
        n_trials=int(n_trials),
        n_keep=int(n_terms),
        baseline_scores=baseline_scores,
        parents=parents,
        feature_bias=feature_bias,
    )
    return [
        InstalledTerm(
            tree=r.tree,
            gain=float(r.score),
            mutation_type=f"offline_trial:{r.mutation_type}",
        )
        for r in results
    ]


def _grow_greedy(
    X, y, *, seed, n_terms, baseline_scores,
) -> list[InstalledTerm]:
    if baseline_scores is not None and len(baseline_scores) == len(y):
        s = np.asarray(baseline_scores, dtype=float).reshape(-1)
        p = 1.0 / (1.0 + np.exp(-np.clip(s, -40.0, 40.0)))
        target = y - p
    else:
        target = y.copy()
    if float(np.std(target)) < 1e-9:
        return []
    reg = SymbolicRegressor(
        seed=int(seed) & 0x7FFFFFFF,
        n_terms=int(n_terms),
        learning_rate=0.85,
        max_pair_features=min(8, X.shape[1]),
        depth_extend=True,
    )
    reg.fit(X, target)
    out: list[InstalledTerm] = []
    for t in reg.terms_:
        tree = t.get("tree")
        if tree is None:
            continue
        gain = float(t.get("gain") or 0.0)
        if gain < 1e-3:
            continue
        out.append(InstalledTerm(
            tree=tree, gain=gain, mutation_type="greedy_residual",
        ))
    return out


def rank_formulas_by_residual(
    engine,
    trees: list[tuple[str, FormulaNode]],
    X: np.ndarray,
    residual: np.ndarray,
    top_k: int = 6,
) -> list[tuple[str, FormulaNode, float]]:
    """Pick the formulas most aligned with a residual (install ranking)."""
    scored: list[tuple[str, FormulaNode, float]] = []
    r = np.asarray(residual, dtype=float).reshape(-1)
    for fid, tree in trees:
        try:
            vals = np.asarray(engine.evaluate(tree, {"X": X}), dtype=float).reshape(-1)
            if vals.shape[0] != r.shape[0]:
                continue
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            g = abs(_corr(vals, r))
        except Exception:
            continue
        if g > 1e-6:
            scored.append((fid, tree, g))
    scored.sort(key=lambda t: (-t[2], t[0]))
    return scored[: max(0, int(top_k))]
