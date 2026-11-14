"""Gradient-boosting fit for symbolic weak learners.

The legacy SymbolicFormula boosting loop *selected* a fixed-shape formula from
the library and added it with a constant step (``learning_rate * h * 4.0``).
The weak learner's own constants were never optimised and the step was never
line-searched, so the ensemble could only combine pre-existing formula shapes
— the "reachability gap" that left it ~17 accuracy points behind XGBoost on
tabular data.

This module closes that gap by making each boosting stage a *real* functional
gradient step, the way XGBoost fits a tree to the loss gradient:

    1. refit_constants  — sculpt the formula's constant leaves to the current
                          pseudo-residual (numeric-gradient Adam, maximising
                          squared correlation with the residual).
    2. least-squares    — fit scale ``a`` and bias ``b`` mapping the learner's
                          raw output onto the residual.
    3. line search      — Newton step for the optimal multiplier under the
                          logistic loss, then apply shrinkage (learning_rate).

Everything operates in logit space: the caller accumulates ``F`` and applies a
sigmoid. Stages produced here store the numeric fit (``mean/std/scale/bias/
gamma``) so prediction reproduces the same contribution exactly.
"""

from __future__ import annotations

import copy

import numpy as np

from figpso_uil.cognition.formula_engine import FormulaNode

# Raw formula outputs are unbounded (features can be large); clip before any
# standardisation so a single wild column can't blow up the fit.
RAW_CLIP = 50.0
_EPS = 1e-9


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def collect_const_nodes(tree: FormulaNode) -> list[FormulaNode]:
    """All constant-leaf nodes, so their values can be tuned in place."""
    out: list[FormulaNode] = []

    def _walk(node: FormulaNode) -> None:
        if node.is_const:
            out.append(node)
        for child in node.children:
            _walk(child)

    _walk(tree)
    return out


def _raw(engine, tree: FormulaNode, X_eff: np.ndarray) -> np.ndarray:
    raw = np.asarray(engine.evaluate(tree, {"X": X_eff}), dtype=float)
    if raw.shape[0] != X_eff.shape[0]:
        # A degenerate tree (e.g. all-const) may broadcast to length 1.
        raw = np.full(X_eff.shape[0], float(raw.reshape(-1)[0]) if raw.size else 0.0)
    return np.clip(np.where(np.isfinite(raw), raw, 0.0), -RAW_CLIP, RAW_CLIP)


def _corr_sq(raw: np.ndarray, target: np.ndarray) -> float:
    """Squared Pearson correlation — the objective refit maximises."""
    rc = raw - raw.mean()
    tc = target - target.mean()
    denom = float(np.sqrt(np.sum(rc * rc) * np.sum(tc * tc)))
    if denom < _EPS:
        return 0.0
    r = float(np.dot(rc, tc)) / denom
    return r * r


def refit_constants(
    engine,
    tree: FormulaNode,
    X_eff: np.ndarray,
    target: np.ndarray,
    iters: int,
    rng: np.random.Generator,
) -> None:
    """Tune the tree's constant leaves (in place) to fit ``target``.

    Objective is squared correlation between the learner's raw output and the
    pseudo-residual — scale/bias are handled separately by least squares, so
    correlation is the shape-only signal we want the constants to chase.
    Numeric-gradient Adam; cheap because trees are small (few constants) and
    ``evaluate`` is vectorised over samples.
    """
    nodes = collect_const_nodes(tree)
    if not nodes or iters <= 0:
        return
    if float(np.std(target)) < _EPS:
        return

    theta = np.array([n.const for n in nodes], dtype=float)

    def _set(vec: np.ndarray) -> None:
        for node, val in zip(nodes, vec):
            node.const = float(val)

    def _obj(vec: np.ndarray) -> float:
        _set(vec)
        return _corr_sq(_raw(engine, tree, X_eff), target)

    best_theta = theta.copy()
    best_obj = _obj(theta)

    # Finite-difference step scaled per-parameter so large constants get a
    # proportional probe (a fixed eps is meaningless next to a const of 100).
    lr = 0.1
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8

    for t in range(1, iters + 1):
        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            h = 1e-4 * (1.0 + abs(theta[i]))
            up = theta.copy(); up[i] += h
            dn = theta.copy(); dn[i] -= h
            grad[i] = (_obj(up) - _obj(dn)) / (2.0 * h)

        # Ascend (maximise corr^2). Adam.
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad * grad
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        theta = theta + lr * mhat / (np.sqrt(vhat) + eps)
        theta = np.clip(theta, -1e3, 1e3)

        cur = _obj(theta)
        if cur > best_obj:
            best_obj, best_theta = cur, theta.copy()

    _set(best_theta)


def _least_squares_scale_bias(
    base: np.ndarray, target: np.ndarray,
) -> tuple[float, float]:
    """Closed-form (a, b) minimising ||a*base + b - target||^2."""
    bmean = float(base.mean())
    tmean = float(target.mean())
    bc = base - bmean
    denom = float(np.dot(bc, bc))
    if denom < _EPS:
        return 0.0, tmean
    a = float(np.dot(bc, target - tmean)) / denom
    b = tmean - a * bmean
    return a, b


def _logistic_line_search(
    F: np.ndarray, c: np.ndarray, y: np.ndarray, n_iter: int = 8,
) -> float:
    """Newton search for the multiplier g minimising logistic loss of F + g*c.

    This is the step XGBoost performs analytically per leaf; here c is a single
    scalar direction, so it is a 1-D Newton on grad = c·(p - y),
    hess = c^2·p(1-p).
    """
    g = 1.0
    for _ in range(n_iter):
        p = _sigmoid(F + g * c)
        grad = float(np.dot(c, p - y))
        hess = float(np.dot(c * c, p * (1.0 - p)))
        if hess < _EPS:
            break
        step = grad / hess
        g -= step
        if abs(step) < 1e-4:
            break
    # Non-negative (never subtract a learner correlated with the residual) and
    # bounded so a near-separable stage can't take a giant leap.
    return float(np.clip(g, 0.0, 10.0))


def fit_boost_stage(
    engine,
    tree: FormulaNode,
    X_eff: np.ndarray,
    F: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    refit_iters: int,
    rng: np.random.Generator,
) -> tuple[FormulaNode, dict, np.ndarray]:
    """One functional-gradient boosting stage.

    Returns (fitted_tree, params, contribution) where ``contribution`` is the
    logit-space vector to add to F, and ``params`` reproduces it at predict
    time via :func:`stage_contribution`. The tree is deep-copied before any
    constant refit so the shared library formula is never mutated.
    """
    tree = copy.deepcopy(tree)
    p = _sigmoid(F)
    target = y - p  # negative gradient of logistic loss (pseudo-residual)

    if refit_iters > 0:
        refit_constants(engine, tree, X_eff, target, refit_iters, rng)

    raw = _raw(engine, tree, X_eff)
    mean = float(raw.mean())
    std = float(raw.std())
    std = std if std > _EPS else 1.0
    base = (raw - mean) / std

    a, b = _least_squares_scale_bias(base, target)
    c = a * base + b
    rho = _logistic_line_search(F, c, y)
    gamma = float(np.clip(learning_rate, 0.01, 1.0)) * rho

    params = {"mean": mean, "std": std, "scale": a, "bias": b, "gamma": gamma}
    contribution = gamma * c
    return tree, params, contribution


def stage_contribution(
    engine, tree: FormulaNode, X_eff: np.ndarray, params: dict,
) -> np.ndarray:
    """Reproduce a grad_fit stage's logit contribution at predict time."""
    raw = _raw(engine, tree, X_eff)
    std = params.get("std", 1.0) or 1.0
    base = (raw - params.get("mean", 0.0)) / std
    c = params.get("scale", 0.0) * base + params.get("bias", 0.0)
    return params.get("gamma", 0.0) * c
