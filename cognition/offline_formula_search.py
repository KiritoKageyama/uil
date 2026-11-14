"""Offline trial-and-error formula builder — no API key, no network.

Mutate with ``FormulaEngine``, score on the dataset residual/target, keep
winners. Deterministic given ``seed``. This is the product path for
building classifier formulas: generate → evaluate → keep/reject, entirely
local.

Never imports LLM clients. Never reads ``OPENAI_API_KEY`` /
``UIL_LLM_API_KEY``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from figpso_uil.cognition.formula_engine import FormulaEngine, FormulaNode
from figpso_uil.cognition.symbolic_regressor import _corr


@dataclass
class TrialResult:
    tree: FormulaNode
    score: float
    mutation_type: str
    trial: int


def _eval_col(engine: FormulaEngine, tree: FormulaNode, X: np.ndarray) -> np.ndarray:
    raw = engine.evaluate(tree, {"X": X})
    col = np.asarray(raw, dtype=float).reshape(-1)
    if col.shape[0] != X.shape[0]:
        col = np.resize(col, X.shape[0])
    return np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)


def _target_from_baseline(
    y: np.ndarray,
    baseline_scores: np.ndarray | None,
) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    if baseline_scores is not None and len(baseline_scores) == len(y):
        s = np.asarray(baseline_scores, dtype=float).reshape(-1)
        p = 1.0 / (1.0 + np.exp(-np.clip(s, -40.0, 40.0)))
        return y - p
    return y.copy()


def trial_and_error_search(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    n_features: int | None = None,
    n_trials: int = 64,
    n_keep: int = 3,
    baseline_scores: np.ndarray | None = None,
    parents: list[FormulaNode] | None = None,
    stress: float = 0.55,
    feature_bias: np.ndarray | None = None,
) -> list[TrialResult]:
    """Build formulas by offline mutate → score → keep.

    Score = |corr(formula(X), residual)|. A trial that fails validation or
    scores below the current weakest keeper is discarded. Surviving
    mutants re-enter the parent pool (trial-and-error hill-climb).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 8:
        return []
    nf = int(n_features if n_features is not None else X.shape[1])
    target = _target_from_baseline(y, baseline_scores)
    if float(np.std(target)) < 1e-9:
        return []

    engine = FormulaEngine(seed=int(seed) & 0x7FFFFFFF)
    if parents:
        pool: list[FormulaNode] = [p for p in parents if p is not None]
    else:
        seeds = engine.get_seed_classifier_formulas(nf)
        pool = list(seeds.values())
    if not pool:
        pool = [FormulaNode(op="sigmoid", children=[FormulaNode(feature=0)])]

    n_trials = max(1, int(n_trials))
    n_keep = max(1, int(n_keep))
    keepers: list[TrialResult] = []

    def _consider(tree: FormulaNode, mtype: str, trial: int) -> None:
        ok, _ = engine.validate_classifier(tree, n_features=nf)
        if not ok:
            return
        col = _eval_col(engine, tree, X)
        if float(np.std(col)) < 1e-12:
            return
        score = abs(_corr(col, target))
        if score < 1e-4:
            return
        cand = TrialResult(tree=tree, score=score, mutation_type=mtype, trial=trial)
        if len(keepers) < n_keep:
            keepers.append(cand)
            keepers.sort(key=lambda r: (-r.score, r.trial))
            pool.append(tree)
            return
        if score > keepers[-1].score + 1e-9:
            keepers[-1] = cand
            keepers.sort(key=lambda r: (-r.score, r.trial))
            pool.append(tree)

    # Score seeds first (trial 0..) so a good seed can win without mutation.
    for i, tree in enumerate(list(pool)):
        _consider(tree, "seed", trial=-1 - i)

    for t in range(n_trials):
        parent = pool[int(engine.rng.integers(0, len(pool)))]
        # Stress ramps slightly so later trials explore more structure.
        s = float(np.clip(stress + 0.25 * (t / max(1, n_trials - 1)), 0.15, 0.95))
        try:
            mut, mtype = engine.mutate(
                parent, s, n_features=nf, feature_bias=feature_bias,
            )
        except Exception:
            continue
        _consider(mut, mtype, t)

    keepers.sort(key=lambda r: (-r.score, r.trial))
    return keepers[:n_keep]
