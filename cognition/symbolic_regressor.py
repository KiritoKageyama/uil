"""Greedy residual-guided symbolic regressor — the D2 (math/equation) learner.

This is the symbolic twin of how a gradient-boosted tree is grown: at each
boosting round it *builds* a small formula term to fit the current residual
(pseudo-gradient of squared loss), scoring candidate structures by their
correlation-gain with the residual — the exact analog of a tree's split gain.
Each term's coefficient is fit by least squares, so the final model is a short,
fully readable additive expression:

    y ≈ base + Σ_t  a_t · term_t(x) + b_t

Every term records the residual-correlation gain that earned it its place, so
the model is not just interpretable but *traceable* (why each piece exists).

Unlike a tree ensemble this extrapolates: outside the training range a tree is
piecewise-constant (flat), but a recovered formula like x0² keeps rising — the
categorical advantage on math/scientific data.
"""

from __future__ import annotations

import numpy as np

from figpso_uil.cognition.formula_engine import FormulaEngine, FormulaNode
from figpso_uil.cognition.symbolic_gradient import refit_constants

_EPS = 1e-12

# Structure-search vocabulary. Kept small and smooth so grown terms stay
# readable and differentiable-ish; sin/cos give periodic structure, square/
# mul give polynomial interactions — enough to recover typical physics forms.
UNARY_OPS = ("square", "sin", "cos", "sqrt", "exp", "abs", "tanh", "neg")
BINARY_OPS = ("mul", "add", "sub", "div")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    ac = a - a.mean()
    bc = b - b.mean()
    denom = float(np.sqrt(np.dot(ac, ac) * np.dot(bc, bc)))
    if denom < _EPS:
        return 0.0
    return float(np.dot(ac, bc)) / denom


def _ls(base: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Least-squares a, b minimising ||a*base + b - target||^2."""
    bmean = float(base.mean())
    bc = base - bmean
    denom = float(np.dot(bc, bc))
    if denom < _EPS:
        return 0.0, float(target.mean())
    a = float(np.dot(bc, target - target.mean())) / denom
    return a, float(target.mean()) - a * bmean


class SymbolicRegressor:
    """Additive boosting of greedily-grown symbolic terms (regression)."""

    def __init__(
        self,
        seed: int = 0,
        n_terms: int = 10,
        learning_rate: float = 0.6,
        max_pair_features: int = 8,
        depth_extend: bool = True,
    ):
        self.engine = FormulaEngine(seed=seed)
        self.n_terms = n_terms
        self.learning_rate = float(learning_rate)
        self.max_pair_features = max_pair_features
        self.depth_extend = depth_extend
        self.base_: float = 0.0
        self.terms_: list[dict] = []  # {tree, a, b, gain}

    # ── structure search ────────────────────────────────────────────────
    def _eval(self, tree: FormulaNode, X: np.ndarray) -> np.ndarray:
        vals = np.asarray(self.engine.evaluate(tree, {"X": X}), dtype=float)
        if vals.shape[0] != X.shape[0]:
            vals = np.full(X.shape[0], float(vals.reshape(-1)[0]) if vals.size else 0.0)
        return np.where(np.isfinite(vals), vals, 0.0)

    def _candidate_trees(self, X: np.ndarray, r: np.ndarray) -> list[FormulaNode]:
        nf = X.shape[1]
        cands: list[FormulaNode] = [FormulaNode(feature=i) for i in range(nf)]
        # unary wraps of each raw feature
        for i in range(nf):
            for op in UNARY_OPS:
                cands.append(FormulaNode(op=op, children=[FormulaNode(feature=i)]))
        # pairwise interactions among the features most correlated with residual
        corrs = np.array([abs(_corr(X[:, i], r)) for i in range(nf)])
        top = list(np.argsort(corrs)[::-1][: self.max_pair_features])
        for ai in top:
            for bi in top:
                if ai < bi:
                    for op in BINARY_OPS:
                        cands.append(FormulaNode(
                            op=op,
                            children=[FormulaNode(feature=int(ai)),
                                      FormulaNode(feature=int(bi))],
                        ))
        return cands

    def _best_term(self, X: np.ndarray, r: np.ndarray) -> tuple[FormulaNode | None, np.ndarray, float]:
        best_tree, best_vals, best_gain = None, None, 0.0
        for tree in self._candidate_trees(X, r):
            vals = self._eval(tree, X)
            g = abs(_corr(vals, r))
            if g > best_gain:
                best_tree, best_vals, best_gain = tree, vals, g
        # one greedy depth-extension: try wrapping the winner in each unary op
        if self.depth_extend and best_tree is not None:
            for op in UNARY_OPS:
                ext = FormulaNode(op=op, children=[best_tree])
                vals = self._eval(ext, X)
                g = abs(_corr(vals, r))
                if g > best_gain + 1e-4:
                    best_tree, best_vals, best_gain = ext, vals, g
        return best_tree, best_vals, best_gain

    # ── boosting loop ───────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SymbolicRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.base_ = float(y.mean())
        F = np.full(len(y), self.base_)
        self.terms_ = []
        for _ in range(self.n_terms):
            r = y - F
            if float(np.std(r)) < 1e-9:
                break
            tree, vals, gain = self._best_term(X, r)
            if tree is None or gain < 1e-3:
                break
            refit_constants(self.engine, tree, X, r, iters=12, rng=self.engine._rng)
            vals = self._eval(tree, X)
            a, b = _ls(vals, r)
            if abs(a) < _EPS:
                break
            F = F + self.learning_rate * (a * vals + b)
            self.terms_.append({
                "tree": tree,
                "a": self.learning_rate * a,
                "b": self.learning_rate * b,
                "gain": round(gain, 4),
            })
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        F = np.full(X.shape[0], self.base_)
        for t in self.terms_:
            F = F + t["a"] * self._eval(t["tree"], X) + t["b"]
        return F

    # ── readout ─────────────────────────────────────────────────────────
    def formula_str(self) -> str:
        parts = [f"{self.base_:+.3f}"]
        for t in self.terms_:
            parts.append(f"{t['a']:+.3f}*({t['tree']!r}) {t['b']:+.3f}")
        return "y = " + " ".join(parts)
