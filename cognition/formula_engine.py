"""UIL Formula Engine — Expression tree construction and evolution.

NOT if-else menu selection. The UIL constructs mathematical formulas
from atomic operations, evolves them via stress-governed mutation,
and evaluates them as custom objectives for classifiers.

A formula is an expression tree: each node is an operation,
leaves are variables or constants. The UIL assembles, mutates,
combines, and invents formulas — recording each in the journal.

V20.3: Core innovation of the UIL.
"""

from __future__ import annotations
import copy
import json
import os
import numpy as np
from typing import Any

from figpso_uil.cognition.failure_point import record_failure_point


# ── Atomic Operations ────────────────────────────────────────────────

# ── Operator guards ──────────────────────────────────────────────────
#
# MEASURED 2026-08-05 (results/extrapolation/SUMMARY.md). Evaluated OUTSIDE
# the region they were fitted in, evolved trees diverged silently: across a
# 21-cell out-of-range benchmark UIL's symbolic regressor emitted predictions
# past 1000x the training scale on 19 seed-runs (R2 ~ -1e35), while gplearn --
# same model family, protected operators -- did so ZERO times.
#
# The old guards were the wrong shape in two ways:
#   * DIV protected only at |b| > 1e-10, so |b| = 1e-6 was "safe" and returned
#     1e6. gplearn's protected division cuts at 1e-3, which is the level that
#     actually bounds the result.
#   * The composition was unguarded. evaluate() replaced NaN/Inf with 0.0, but
#     a product of large FINITE intermediates stays finite -- 1e35 sailed
#     through the isfinite check untouched. That is where the blowups came
#     from, not from any single operator.
#
# Design rule for every constant below: the guard must be INERT in normal
# operation (loss formulas work on logits/probabilities, feature formulas on
# standardized columns) and bind only when a formula is being evaluated far
# outside its fitted region. MAX_MAGNITUDE is ~6 orders of magnitude above any
# legitimate value either path produces.

#: Largest magnitude any formula node may emit (inf = clamp disabled, only
#: the NaN/Inf replacement applies).
#:
#: DEFAULT IS OFF, and that is a MEASURED decision, not an oversight. Tightening
#: these guards was tried against the out-of-range benchmark and no setting
#: dominates -- see results/extrapolation/GUARD_ABLATION.md:
#:
#:   config                                  blowups  kinetic@1.5  gravity@1.5
#:   defaults (guards off)                        19        0.358       -0.177
#:   div 1e-3 + clamp 1e6                          9        0.092       -1.167
#:   div 1e-3 + clamp 1e30 + exp 20               22        0.092       -0.555
#:
#: The tight clamp halves blowups but destroys working cases (clamping a
#: diverging prediction makes it bounded, not correct); relaxing it to keep
#: those cases makes blowups WORSE than doing nothing. The constants also
#: interact -- `exp` reads as innocent under a tight clamp and matters a lot
#: under a loose one -- so this is a four-way trade-off among hand-picked
#: numbers with no winner, on a component that loses 4-17 to off-the-shelf
#: gplearn in EVERY configuration tested. Per HARDCODED_VALUE_AUDIT_QA.md,
#: hand-tuning further is the anti-pattern, not the fix.
#:
#: All four are env-overridable so any row of that table is one variable away.
MAX_MAGNITUDE = float(os.environ.get("UIL_MAX_MAGNITUDE", "inf"))

#: `pow` bounds. Measured to have NO effect on the benchmark either way.
POW_BASE_MAX = float(os.environ.get("UIL_POW_BASE_MAX", "100.0"))
POW_EXP_MAX = float(os.environ.get("UIL_POW_EXP_MAX", "5.0"))

#: Denominator below which division is "protected" (gplearn parity).
#: Denominator below which division is "protected". Left at the original
#: 1e-10: raising it to gplearn's 1e-3 measurably COSTS accuracy here
#: (kinetic@1.5 0.358 -> 0.276) and buys nothing on the head-to-head.
#: NOTE: an earlier analysis claimed this branch "essentially never fires",
#: from counting near-zero values in the RAW DATA (0-1 of 2400). That was the
#: wrong quantity -- _safe_div runs on INTERMEDIATE node values inside evolved
#: trees, which approach zero regardless of the input distribution. The
#: ablation shows it changes results. Measure what the code consumes.
DIV_EPS = float(os.environ.get("UIL_DIV_EPS", "1e-10"))

#: Value a protected division returns. gplearn returns 1.0 (the
#: multiplicative identity, so the term degrades to a pass-through of the
#: numerator); this engine historically returned 0.0, which DELETES the term.
#: Configurable because it is a semantic choice, not a safety one, and it
#: changes the meaning of every persisted formula that contains a division.
DIV_PROTECTED_VALUE = float(os.environ.get("UIL_DIV_PROTECTED_VALUE", "0.0"))

#: Argument cap for `exp`, DERIVED from MAX_MAGNITUDE so that no single
#: operator can exceed the global cap on its own. Deriving it rather than
#: hand-picking a number keeps the two consistent if the cap ever moves --
#: an exp bound above the cap would mean exp always saturates the
#: composition guard instead of being bounded in its own right.
#: `exp` argument cap. Left at the original 100. It reads as "innocent" under
#: a tight MAX_MAGNITUDE (everything is clamped anyway) and turns out to
#: matter under a loose one -- gravity@1.5 was 0.743 with exp effectively
#: unguarded and -0.555 with it at 20, same clamp. Another reason the four
#: constants cannot be tuned one at a time.
EXP_CLIP = float(os.environ.get("UIL_EXP_CLIP", "100"))


def _safe_div(a, b):
    """Protected division: |b| < DIV_EPS yields 0.0 rather than a huge value.

    Uses np.divide(where=...) instead of np.where(cond, a / b, 0.0): the
    latter evaluates a / b for EVERY element including b == 0, which is where
    the `RuntimeWarning: divide by zero` noise in every run log came from.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.abs(b) > DIV_EPS
    out = np.full(np.broadcast(a, b).shape, DIV_PROTECTED_VALUE, dtype=float)
    return np.divide(a, b, out=out, where=ok)

def _safe_log(x):
    """Log with protection against log(0)."""
    return np.log(np.clip(x, 1e-10, None))

def _safe_sqrt(x):
    """Square root with protection against negative values."""
    return np.sqrt(np.clip(x, 0.0, None))

def _bounded(result):
    """Replace NaN/Inf with 0.0 and clamp magnitude to MAX_MAGNITUDE.

    The clamp is the composition guard: individually-bounded operators can
    still multiply into astronomically large finite values, and `isfinite`
    happily passes 1e35.
    """
    result = np.asarray(result, dtype=float)
    result = np.where(np.isfinite(result), result, 0.0)
    return np.clip(result, -MAX_MAGNITUDE, MAX_MAGNITUDE)

def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
        1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
        np.exp(np.clip(x, -500, 500)) / (1.0 + np.exp(np.clip(x, -500, 500))))


# Operation registry: name → (function, arity)
OPS = {
    # Binary operations (2 children)
    "add": (lambda a, b: a + b, 2),
    "sub": (lambda a, b: a - b, 2),
    "mul": (lambda a, b: a * b, 2),
    "div": (_safe_div, 2),
    # 100**5 = 1e10 was reachable and fed straight into the products that
    # produced the 1e35 blowups; 10**3 = 1e3 keeps `pow` expressive while
    # leaving it three orders of magnitude below MAX_MAGNITUDE.
    "pow": (lambda a, b: np.power(np.clip(np.abs(a), 1e-10, POW_BASE_MAX),
                                  np.clip(b, -POW_EXP_MAX, POW_EXP_MAX)), 2),
    "max": (lambda a, b: np.maximum(a, b), 2),
    "min": (lambda a, b: np.minimum(a, b), 2),

    # Unary operations (1 child)
    "neg": (lambda a: -a, 1),
    "abs": (lambda a: np.abs(a), 1),
    "log": (_safe_log, 1),
    # exp(100) = 2.7e43 -- on its own enough to produce the observed 1e35
    # predictions. EXP_CLIP = log(MAX_MAGNITUDE), so exp tops out exactly at
    # the global cap and never relies on the composition guard.
    "exp": (lambda a: np.exp(np.clip(a, -EXP_CLIP, EXP_CLIP)), 1),
    "sqrt": (_safe_sqrt, 1),
    "sigmoid": (_sigmoid, 1),
    "tanh": (lambda a: np.tanh(a), 1),
    "relu": (lambda a: np.maximum(0.0, a), 1),
    "sign": (lambda a: np.sign(a), 1),
    "square": (lambda a: a * a, 1),
    "clip01": (lambda a: np.clip(a, 0.0, 1.0), 1),
}

# Available variable names
VARIABLES = {"y_true", "y_pred", "sample_weight", "n_samples", "class_weight"}

# Mutation types
MUTATION_TYPES = [
    "constant_tweak",
    "operation_swap",
    "term_addition",
    "term_removal",
    "subtree_swap",
]


# ── Formula Tree Node ────────────────────────────────────────────────

class FormulaNode:
    """A single node in a formula expression tree.
    
    Three kinds:
        Op node:   {"op": "add", "children": [left, right]}
        Var node:  {"var": "y_pred"}
        Const node: {"const": 2.3}
    """
    
    def __init__(
        self,
        op: str | None = None,
        var: str | None = None,
        const: float | None = None,
        feature: int | None = None,
        children: list['FormulaNode'] | None = None,
    ):
        self.op = op
        self.var = var
        self.const = const
        self.feature = feature
        self.children = children or []

    @property
    def is_op(self) -> bool:
        return self.op is not None

    @property
    def is_var(self) -> bool:
        return self.var is not None

    @property
    def is_const(self) -> bool:
        return self.const is not None

    @property
    def is_feature(self) -> bool:
        return self.feature is not None

    @property
    def is_leaf(self) -> bool:
        return self.is_var or self.is_const or self.is_feature
    
    def depth(self) -> int:
        if self.is_leaf:
            return 0
        return 1 + max((c.depth() for c in self.children), default=0)
    
    def node_count(self) -> int:
        if self.is_leaf:
            return 1
        return 1 + sum(c.node_count() for c in self.children)
    
    def __repr__(self) -> str:
        if self.is_var:
            return self.var
        if self.is_const:
            return f"{self.const:.3f}"
        if self.is_feature:
            return f"x[{self.feature}]"
        if self.is_op:
            args = ", ".join(repr(c) for c in self.children)
            return f"{self.op}({args})"
        return "???"

    def to_dict(self) -> dict:
        """Serialize for JSON (journal storage)."""
        if self.is_var:
            return {"var": self.var}
        if self.is_const:
            return {"const": self.const}
        if self.is_feature:
            return {"feature": self.feature}
        return {
            "op": self.op,
            "children": [c.to_dict() for c in self.children],
        }

    @staticmethod
    def from_dict(d: dict) -> 'FormulaNode':
        """Deserialize from JSON."""
        if "var" in d:
            return FormulaNode(var=d["var"])
        if "const" in d:
            return FormulaNode(const=d["const"])
        if "feature" in d:
            return FormulaNode(feature=d["feature"])
        return FormulaNode(
            op=d["op"],
            children=[FormulaNode.from_dict(c) for c in d.get("children", [])],
        )


# ── Seed Formulas ────────────────────────────────────────────────────

def _build_logistic_loss() -> FormulaNode:
    """Standard logistic loss: -[y*log(p) + (1-y)*log(1-p)]"""
    return FormulaNode(op="neg", children=[
        FormulaNode(op="add", children=[
            FormulaNode(op="mul", children=[
                FormulaNode(var="y_true"),
                FormulaNode(op="log", children=[FormulaNode(var="y_pred")]),
            ]),
            FormulaNode(op="mul", children=[
                FormulaNode(op="sub", children=[
                    FormulaNode(const=1.0),
                    FormulaNode(var="y_true"),
                ]),
                FormulaNode(op="log", children=[
                    FormulaNode(op="sub", children=[
                        FormulaNode(const=1.0),
                        FormulaNode(var="y_pred"),
                    ]),
                ]),
            ]),
        ]),
    ])

def _build_focal_loss() -> FormulaNode:
    """Focal loss: -alpha * (1-p)^gamma * y * log(p)"""
    return FormulaNode(op="neg", children=[
        FormulaNode(op="mul", children=[
            FormulaNode(op="pow", children=[
                FormulaNode(op="sub", children=[
                    FormulaNode(const=1.0),
                    FormulaNode(var="y_pred"),
                ]),
                FormulaNode(const=2.0),  # gamma — discoverable constant
            ]),
            FormulaNode(op="mul", children=[
                FormulaNode(var="y_true"),
                FormulaNode(op="log", children=[FormulaNode(var="y_pred")]),
            ]),
        ]),
    ])

def _build_squared_loss() -> FormulaNode:
    """Squared loss: (y - p)^2"""
    return FormulaNode(op="square", children=[
        FormulaNode(op="sub", children=[
            FormulaNode(var="y_true"),
            FormulaNode(var="y_pred"),
        ]),
    ])

def _build_hinge_loss() -> FormulaNode:
    """Hinge loss: max(0, 1 - y_signed * p)"""
    return FormulaNode(op="max", children=[
        FormulaNode(const=0.0),
        FormulaNode(op="sub", children=[
            FormulaNode(const=1.0),
            FormulaNode(op="mul", children=[
                FormulaNode(var="y_true"),
                FormulaNode(var="y_pred"),
            ]),
        ]),
    ])

SEED_FORMULAS = {
    "logistic": _build_logistic_loss,
    "focal": _build_focal_loss,
    "squared": _build_squared_loss,
    "hinge": _build_hinge_loss,
}


# ── Formula Engine ───────────────────────────────────────────────────

class FormulaEngine:
    """Constructs, evaluates, mutates, and evolves mathematical formulas.
    
    Stress-governed mutation: low stress → tweak constants,
    high stress → swap subtrees, extreme stress → invent new formulas.
    """
    
    # Safety limits
    MAX_DEPTH = 8
    MAX_NODES = 50
    GRAD_CLIP = 10.0
    HESS_MIN = 1e-6
    
    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        # Optional observe-only guard telemetry (audit Part E). Attached by
        # the optimizer for the loss-formula engine it owns; None everywhere
        # else, so this is a no-op unless explicitly wired. Records how often
        # a mutated tree hits the MAX_DEPTH/MAX_NODES ceiling and has to be
        # pruned -- the evidence for whether the grammar's size caps are
        # load-bearing or never approached.
        self.telemetry = None

    @property
    def rng(self) -> np.random.Generator:
        """Exposed so callers (e.g. tournament-selecting a mutation
        parent in core/optimizer.py) draw from the same seeded stream
        instead of introducing an unseeded source of nondeterminism."""
        return self._rng

    def get_seed_formulas(self) -> dict[str, FormulaNode]:
        """Return fresh copies of all seed formulas."""
        return {name: builder() for name, builder in SEED_FORMULAS.items()}

    # V20+.19: deliberately bounded, not "as many as possible" -- every
    # seed needs MIN_EVIDENCE_BOOTSTRAP (FormulaLibrary) real evaluations
    # before confidence/alignment-based selection even considers it, and
    # only ONE evidence point accumulates per optimizer stage (record_result
    # fires once per stage, on the winning action). N seeds therefore cost
    # ~N * MIN_EVIDENCE_BOOTSTRAP stages of forced bootstrap before the
    # population can be selected on merit at all -- 30 seeds would consume
    # ~90 stages alone, exceeding even a 60-stage run entirely and starving
    # the feature-alignment selection (see symbolic_formula_adapter.py)
    # of any chance to run. Kept comfortably under typical stage budgets.
    MAX_SINGLE_FEATURE_SEEDS = 12
    MAX_PAIR_SEEDS = 4

    def get_seed_classifier_formulas(self, n_features: int) -> dict[str, FormulaNode]:
        """Seed formulas for the direct-classifier population (formula
        IS the prediction, evaluated on raw feature columns) -- a
        separate vocabulary from get_seed_formulas()'s loss functions,
        which stay over y_pred/y_true and are untouched by this. These
        are deliberately weak starting points; the population is
        expected to mutate away from them.

        Spreads single-feature seeds evenly across the feature range
        (not just indices 0..k-1) so the bootstrap phase gets real
        coverage of the dataset instead of always favoring low-index
        features -- mutation/crossover can in principle reach any
        feature eventually, but seeding gives immediate, free coverage
        without waiting on it.
        """
        n_single = min(n_features, self.MAX_SINGLE_FEATURE_SEEDS)
        single_indices = sorted(set(
            int(i) for i in np.linspace(0, n_features - 1, n_single)
        )) if n_features > 0 else []

        seeds: dict[str, FormulaNode] = {
            f"feature_{idx}_threshold": FormulaNode(
                op="sigmoid", children=[FormulaNode(feature=idx)]
            )
            for idx in single_indices
        }

        if n_features >= 2:
            n_pairs = min(self.MAX_PAIR_SEEDS, n_features // 2)
            pair_indices = sorted(set(
                int(i) for i in np.linspace(0, n_features - 2, max(n_pairs, 1))
            ))[:n_pairs]
            for idx in pair_indices:
                seeds[f"feature_{idx}_{idx + 1}_linear"] = FormulaNode(
                    op="sigmoid", children=[
                        FormulaNode(op="add", children=[
                            FormulaNode(op="mul", children=[
                                FormulaNode(const=1.0), FormulaNode(feature=idx),
                            ]),
                            FormulaNode(op="mul", children=[
                                FormulaNode(const=1.0), FormulaNode(feature=idx + 1),
                            ]),
                        ])
                    ],
                )
        from figpso_uil.cognition.method_templates import method_template_seeds
        seeds.update(method_template_seeds(n_features))
        return seeds

    # ── Evaluation ───────────────────────────────────────────────────

    def evaluate(
        self,
        tree: FormulaNode,
        variables: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Evaluate a formula tree with given variable bindings.
        
        Args:
            tree: The formula expression tree.
            variables: Dict mapping variable names to arrays.
                       e.g., {"y_pred": array, "y_true": array}
        
        Returns:
            Result array (same shape as inputs).
        """
        if tree.is_var:
            val = variables.get(tree.var)
            if val is None:
                return np.zeros(1)
            return np.asarray(val, dtype=float)

        if tree.is_const:
            # Return scalar broadcast to length n_samples. Uses shape[0]
            # of the first bound variable, NOT np.full_like's full shape
            # -- np.full_like would copy a 2-D "X" feature matrix's full
            # (n_samples, n_features) shape, breaking element-wise ops
            # against a 1-D feature-column leaf (shape (n_samples,)).
            if not variables:
                return np.array([tree.const])
            sample = np.asarray(next(iter(variables.values())))
            n = sample.shape[0] if sample.ndim >= 1 else 1
            return np.full(n, tree.const, dtype=float)

        if tree.is_feature:
            # Direct-classifier formulas: leaves reference columns of the
            # raw feature matrix passed under "X", not y_pred/y_true.
            X = variables.get("X")
            if X is None:
                return np.zeros(1)
            X = np.asarray(X, dtype=float)
            idx = tree.feature
            if idx < 0 or idx >= X.shape[1]:
                return np.zeros(X.shape[0])
            return X[:, idx]

        if tree.is_op:
            op_fn, arity = OPS.get(tree.op, (None, 0))
            if op_fn is None:
                return np.zeros(1)
            
            child_vals = [self.evaluate(c, variables) for c in tree.children[:arity]]
            
            if len(child_vals) < arity:
                return np.zeros(1)
            
            try:
                result = op_fn(*child_vals)
                # Safety: replace NaN/Inf AND bound the magnitude. The
                # isfinite check alone let finite 1e35 values through, which
                # is what made out-of-range predictions diverge silently.
                return _bounded(result)
            except Exception as e:
                record_failure_point(
                    "formula_engine.evaluate",
                    {
                        "op": tree.op,
                        "arity": arity,
                        "variable_keys": list(variables.keys()),
                        "child_shapes": [
                            np.asarray(c).shape for c in child_vals
                        ],
                    },
                    e,
                )
                raise

        return np.zeros(1)
    
    def compute_gradient(
        self,
        tree: FormulaNode,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        epsilon: float = 1e-5,
    ) -> np.ndarray:
        """Compute gradient of formula w.r.t. y_pred (numerical differentiation).
        
        grad = (loss(y_pred + eps) - loss(y_pred - eps)) / (2 * eps)
        
        Returns clipped gradient array.
        """
        p = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
        
        vars_plus = {"y_pred": p + epsilon, "y_true": y_true}
        vars_minus = {"y_pred": p - epsilon, "y_true": y_true}
        
        loss_plus = self.evaluate(tree, vars_plus)
        loss_minus = self.evaluate(tree, vars_minus)
        
        grad = (loss_plus - loss_minus) / (2.0 * epsilon)
        raw_peak = float(np.max(np.abs(grad))) if grad.size else 0.0
        grad = np.clip(grad, -self.GRAD_CLIP, self.GRAD_CLIP)
        gt = getattr(self, "_guard_telemetry", None)
        if gt is not None:
            clipped_peak = float(np.max(np.abs(grad))) if grad.size else 0.0
            gt.observe("formula_grad_clip", raw_peak, clipped_peak)

        return np.where(np.isfinite(grad), grad, 0.0)
    
    def compute_hessian(
        self,
        tree: FormulaNode,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        epsilon: float = 1e-5,
    ) -> np.ndarray:
        """Compute hessian of formula w.r.t. y_pred (numerical second derivative).
        
        hess = (loss(p+eps) - 2*loss(p) + loss(p-eps)) / eps^2
        
        Returns hessian array, clamped to be positive (XGBoost requirement).
        """
        p = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
        
        vars_center = {"y_pred": p, "y_true": y_true}
        vars_plus = {"y_pred": p + epsilon, "y_true": y_true}
        vars_minus = {"y_pred": p - epsilon, "y_true": y_true}
        
        loss_center = self.evaluate(tree, vars_center)
        loss_plus = self.evaluate(tree, vars_plus)
        loss_minus = self.evaluate(tree, vars_minus)
        
        hess = (loss_plus - 2.0 * loss_center + loss_minus) / (epsilon * epsilon)
        
        # Safety: hessian must be positive for XGBoost
        hess = np.maximum(hess, self.HESS_MIN)
        
        return np.where(np.isfinite(hess), hess, self.HESS_MIN)
    
    # ── Mutation (stress-governed) ───────────────────────────────────

    def mutate(
        self,
        tree: FormulaNode,
        stress_level: float,
        n_features: int | None = None,
        feature_bias: np.ndarray | None = None,
    ) -> tuple[FormulaNode, str]:
        """Mutate a formula tree. Aggressiveness scales with stress.

        n_features/feature_bias: only meaningful for direct-classifier
        formulas (leaves reference feature indices, not y_pred/y_true).
        None reproduces the original loss-formula behavior exactly --
        every existing call site is unaffected.

        Returns:
            (mutated_tree, mutation_type_name)
        """
        tree = copy.deepcopy(tree)

        # Choose mutation based on stress
        if stress_level < 0.3:
            # Low stress: safe, incremental
            mutation = self._rng.choice(["constant_tweak", "constant_tweak", "operation_swap"])
        elif stress_level < 0.6:
            # Medium stress: structural changes
            mutation = self._rng.choice(["constant_tweak", "operation_swap", "term_addition", "term_removal"])
        elif stress_level < 0.85:
            # High stress: aggressive
            mutation = self._rng.choice(["term_addition", "subtree_swap", "operation_swap"])
        else:
            # Extreme stress: try anything
            mutation = self._rng.choice(["subtree_swap", "term_addition", "term_removal", "random_subtree"])

        if mutation == "constant_tweak":
            self._mutate_constant(tree)
        elif mutation == "operation_swap":
            self._mutate_operation(tree)
        elif mutation == "term_addition":
            self._mutate_add_term(tree, n_features, feature_bias)
        elif mutation == "term_removal":
            self._mutate_remove_term(tree, n_features, feature_bias)
        elif mutation == "subtree_swap":
            self._mutate_swap_subtree(tree, n_features, feature_bias)
        elif mutation == "random_subtree":
            tree = self._generate_random_subtree(depth=3, n_features=n_features, feature_bias=feature_bias)
            mutation = "random_invention"

        # Validate size constraints
        if tree.depth() > self.MAX_DEPTH or tree.node_count() > self.MAX_NODES:
            gt = getattr(self, "_guard_telemetry", None)
            if gt is not None:
                d, n = float(tree.depth()), float(tree.node_count())
                gt.observe("formula_max_depth", d, min(d, float(self.MAX_DEPTH)))
                gt.observe("formula_max_nodes", n, min(n, float(self.MAX_NODES)))
            self._prune(tree, n_features=n_features)

        return tree, mutation

    def crossover(
        self,
        tree_a: FormulaNode,
        tree_b: FormulaNode,
    ) -> FormulaNode:
        """Combine two formulas by grafting a subtree from B into A."""
        result = copy.deepcopy(tree_a)
        donor_subtree = self._random_subtree(copy.deepcopy(tree_b))
        
        if donor_subtree is not None:
            self._replace_random_subtree(result, donor_subtree)
        
        if result.depth() > self.MAX_DEPTH or result.node_count() > self.MAX_NODES:
            gt = getattr(self, "_guard_telemetry", None)
            if gt is not None:
                d, n = float(result.depth()), float(result.node_count())
                gt.observe("formula_max_depth", d, min(d, float(self.MAX_DEPTH)))
                gt.observe("formula_max_nodes", n, min(n, float(self.MAX_NODES)))
            self._prune(result)
        
        return result
    
    # ── Validation ───────────────────────────────────────────────────

    def validate(
        self,
        tree: FormulaNode,
        n_samples: int = 20,
    ) -> tuple[bool, str]:
        """Validate a formula tree produces valid gradients and hessians.
        
        Returns:
            (is_valid, reason_if_invalid)
        """
        if tree.depth() > self.MAX_DEPTH:
            return False, f"depth {tree.depth()} exceeds max {self.MAX_DEPTH}"
        if tree.node_count() > self.MAX_NODES:
            return False, f"nodes {tree.node_count()} exceeds max {self.MAX_NODES}"
        
        # Test with dummy data
        y_pred = self._rng.uniform(0.1, 0.9, size=n_samples)
        y_true = self._rng.choice([0.0, 1.0], size=n_samples)
        
        try:
            loss = self.evaluate(tree, {"y_pred": y_pred, "y_true": y_true})
            if not np.all(np.isfinite(loss)):
                return False, "loss contains NaN/Inf"
            
            grad = self.compute_gradient(tree, y_pred, y_true)
            if not np.all(np.isfinite(grad)):
                return False, "gradient contains NaN/Inf"
            
            hess = self.compute_hessian(tree, y_pred, y_true)
            if not np.all(np.isfinite(hess)):
                return False, "hessian contains NaN/Inf"
            
            if np.all(np.abs(grad) < 1e-12):
                return False, "gradient is all zeros (formula has no sensitivity)"
            
            return True, "valid"
        except Exception as e:
            return False, f"evaluation error: {str(e)}"

    def validate_classifier(
        self,
        tree: FormulaNode,
        n_features: int,
        n_samples: int = 20,
    ) -> tuple[bool, str]:
        """Validate a direct-classifier formula tree (evaluated on raw
        feature columns, not y_pred/y_true -- no gradient/hessian check,
        since there's no boosting algorithm consuming them for this
        use case; the formula itself IS the prediction).

        Returns:
            (is_valid, reason_if_invalid)
        """
        if tree.depth() > self.MAX_DEPTH:
            return False, f"depth {tree.depth()} exceeds max {self.MAX_DEPTH}"
        if tree.node_count() > self.MAX_NODES:
            return False, f"nodes {tree.node_count()} exceeds max {self.MAX_NODES}"

        X = self._rng.uniform(-2.0, 2.0, size=(n_samples, n_features))

        try:
            out = self.evaluate(tree, {"X": X})
            if not np.all(np.isfinite(out)):
                return False, "output contains NaN/Inf"
            if np.std(out) < 1e-12:
                return False, "output is constant (formula has no sensitivity to features)"
            return True, "valid"
        except Exception as e:
            return False, f"evaluation error: {str(e)}"

    # ── Serialization ────────────────────────────────────────────────

    def serialize(self, tree: FormulaNode) -> dict:
        """Serialize a formula tree to a JSON-compatible dict."""
        return tree.to_dict()
    
    def deserialize(self, data: dict) -> FormulaNode:
        """Deserialize a formula tree from a dict."""
        return FormulaNode.from_dict(data)
    
    # ── Internal Mutation Operations ─────────────────────────────────

    def _mutate_constant(self, tree: FormulaNode) -> None:
        """Tweak a random constant in the tree."""
        consts = self._collect_nodes(tree, lambda n: n.is_const)
        if consts:
            node = self._rng.choice(consts)
            # Gaussian perturbation (small)
            node.const += self._rng.normal(0, 0.3)
            node.const = float(np.clip(node.const, -10.0, 10.0))
    
    def _mutate_operation(self, tree: FormulaNode) -> None:
        """Swap an operation for another with the same arity."""
        ops = self._collect_nodes(tree, lambda n: n.is_op)
        if not ops:
            return
        node = self._rng.choice(ops)
        _, arity = OPS.get(node.op, (None, 0))
        # Find same-arity operations
        candidates = [name for name, (_, a) in OPS.items() if a == arity and name != node.op]
        if candidates:
            node.op = self._rng.choice(candidates)
    
    def _mutate_add_term(
        self,
        tree: FormulaNode,
        n_features: int | None = None,
        feature_bias: np.ndarray | None = None,
    ) -> None:
        """Add a new term to the formula."""
        # Create a small random term
        new_term = FormulaNode(op="mul", children=[
            FormulaNode(const=float(self._rng.uniform(0.01, 0.5))),
            self._generate_random_subtree(depth=2, n_features=n_features, feature_bias=feature_bias),
        ])
        # Wrap the current tree with an addition
        original = copy.deepcopy(tree)
        tree.op = "add"
        tree.var = None
        tree.const = None
        tree.feature = None
        tree.children = [original, new_term]

    def _mutate_remove_term(
        self,
        tree: FormulaNode,
        n_features: int | None = None,
        feature_bias: np.ndarray | None = None,
    ) -> None:
        """Remove a subtree, replacing with a simpler expression."""
        if tree.is_leaf:
            return
        # Replace a random child with a constant or variable/feature leaf
        if tree.children:
            idx = self._rng.integers(0, len(tree.children))
            tree.children[idx] = self._random_leaf(n_features, feature_bias)

    def _mutate_swap_subtree(
        self,
        tree: FormulaNode,
        n_features: int | None = None,
        feature_bias: np.ndarray | None = None,
    ) -> None:
        """Replace a random subtree with a new random subtree."""
        if tree.is_leaf:
            return
        if tree.children:
            idx = self._rng.integers(0, len(tree.children))
            tree.children[idx] = self._generate_random_subtree(depth=2, n_features=n_features, feature_bias=feature_bias)

    def _random_leaf(
        self,
        n_features: int | None = None,
        feature_bias: np.ndarray | None = None,
    ) -> FormulaNode:
        """A single leaf: y_pred/y_true for loss formulas (n_features=None,
        the original behavior), or a feature-index leaf for direct-
        classifier formulas (n_features given, optionally weighted by
        feature_bias -- e.g. PSO's per-feature relevance signal)."""
        if n_features is not None and n_features > 0:
            if feature_bias is not None and len(feature_bias) == n_features:
                weights = np.clip(np.asarray(feature_bias, dtype=float), 1e-6, None)
                weights = weights / weights.sum()
                idx = int(self._rng.choice(n_features, p=weights))
            else:
                idx = int(self._rng.integers(0, n_features))
            return FormulaNode(feature=idx)
        return FormulaNode(var=self._rng.choice(["y_pred", "y_true"]))

    def _generate_random_subtree(
        self,
        depth: int = 3,
        n_features: int | None = None,
        feature_bias: np.ndarray | None = None,
    ) -> FormulaNode:
        """Generate a random expression subtree."""
        if depth <= 0 or self._rng.random() < 0.3:
            # Leaf
            if self._rng.random() < 0.6:
                return self._random_leaf(n_features, feature_bias)
            else:
                return FormulaNode(const=float(self._rng.uniform(-2, 2)))

        # Pick a random operation
        op_names = list(OPS.keys())
        op_name = self._rng.choice(op_names)
        _, arity = OPS[op_name]

        children = [
            self._generate_random_subtree(depth - 1, n_features=n_features, feature_bias=feature_bias)
            for _ in range(arity)
        ]
        return FormulaNode(op=op_name, children=children)
    
    def _collect_nodes(self, tree: FormulaNode, predicate) -> list[FormulaNode]:
        """Collect all nodes matching a predicate."""
        result = []
        if predicate(tree):
            result.append(tree)
        for child in tree.children:
            result.extend(self._collect_nodes(child, predicate))
        return result
    
    def _random_subtree(self, tree: FormulaNode) -> FormulaNode | None:
        """Pick a random subtree from the tree."""
        all_nodes = self._collect_nodes(tree, lambda n: True)
        if all_nodes:
            return self._rng.choice(all_nodes)
        return None
    
    def _replace_random_subtree(self, tree: FormulaNode, replacement: FormulaNode) -> bool:
        """Replace a random child subtree with the replacement."""
        if tree.is_leaf:
            return False
        if tree.children:
            idx = self._rng.integers(0, len(tree.children))
            tree.children[idx] = replacement
            return True
        return False
    
    def _prune(self, tree: FormulaNode, n_features: int | None = None) -> None:
        """Prune a tree to meet size constraints.

        n_features: if given, prunes into a feature-index leaf instead
        of a y_pred var leaf -- a stray y_pred/y_true leaf would
        silently evaluate to zero under the classifier-formula
        evaluate(tree, {"X": ...}) call convention, since that path
        never binds "y_pred"/"y_true" in variables."""
        while tree.depth() > self.MAX_DEPTH or tree.node_count() > self.MAX_NODES:
            # Replace deepest subtree with a leaf
            if tree.children:
                deepest_idx = max(
                    range(len(tree.children)),
                    key=lambda i: tree.children[i].depth()
                )
                if n_features is not None and n_features > 0:
                    tree.children[deepest_idx] = FormulaNode(feature=0)
                else:
                    tree.children[deepest_idx] = FormulaNode(var="y_pred")
            else:
                break
