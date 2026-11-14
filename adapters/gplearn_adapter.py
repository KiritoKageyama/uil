"""gplearn adapter — genetic-programming symbolic regression/classification.

WHY THIS IS HERE
----------------
Measured 2026-08-05 (`results/extrapolation/SUMMARY.md`): on out-of-range
recovery, gplearn beats UIL's own `cognition/symbolic_regressor.py` on **17 of
21** (task x reach) cells, and beats every NON-symbolic baseline on the hard
long-reach cells (harmonic 0.863 vs MLP 0.463; kinetic 0.735 vs MLP 0.448). It
also never diverges: 0 blow-ups against UIL's 19, across the whole grid.

That result vindicates the project's thesis about the symbolic domain --
symbolic regression really is the right tool where trees and nets cannot
extrapolate -- while falsifying UIL's own realisation of it. Operator guards
were tried as a fix and rejected (`results/extrapolation/GUARD_ABLATION.md`);
the gap is term SELECTION (greedy residual fitting vs population search under
parsimony pressure), which no arithmetic guard closes.

SCOPE -- READ BEFORE ROUTING TO THIS
------------------------------------
The evidence above is entirely about **regression / extrapolation**. This
adapter exposes gplearn's *classifier*, for which there is **no UIL
benchmark at all**. It is registered so it can be measured and chosen
explicitly; it deliberately does NOT replace `SymbolicFormula` on any routing
strategy. Flipping `STRATEGY_TO_CLASSIFIER["symbolic"]` would be exactly the
unvalidated default change that E6 regressed on, and needs its own Gate 2.

WHAT IT TRADES AWAY
-------------------
A gplearn program is readable (`str(model._program)`), but it is NOT part of
the project's editable-knowledge machinery: no FormulaLibrary provenance, no
VersionStore cells, and `cognition/law_export.py` does not apply to it. UIL's
own symbolic learner remains the one with the HITL story. That is the real
division of labour: gplearn for out-of-range accuracy, SymbolicFormula for
inspectable/editable knowledge.
"""

from __future__ import annotations

import numpy as np

from figpso_uil.core.utils import safe_minmax

try:  # optional dependency, mirrors the TabPFN pattern
    from gplearn.genetic import SymbolicClassifier
    _HAS_GPLEARN = True
except Exception:  # pragma: no cover - absence is a valid environment
    SymbolicClassifier = None  # type: ignore[assignment]
    _HAS_GPLEARN = False

#: Matches the symbolic/MLP adapters: a low-weighted feature is shrunk, never
#: deleted, so the search can recover it (the ratchet lesson in CLAUDE.md).
WEIGHT_FLOOR = 0.25

#: gplearn's protected function set. `div` returns 1.0 for |denom| < 0.001 and
#: log/sqrt are domain-guarded -- the property that gave it 0 blow-ups.
DEFAULT_FUNCTIONS = ("add", "sub", "mul", "div", "sqrt", "log", "sin", "cos")


class GPLearnAdapter:
    """Adapts gplearn's SymbolicClassifier to the ClassifierAdapter protocol."""

    def __init__(
        self,
        seed: int,
        num_classes: int,
        population_size: int = 1000,
        generations: int = 20,
        parsimony_coefficient: float = 0.001,
    ):
        if not _HAS_GPLEARN:
            raise ImportError(
                "gplearn is not installed; `pip install gplearn` to use "
                "the GPLearn classifier"
            )
        self.seed = int(seed)
        self.num_classes = int(num_classes)
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.parsimony_coefficient = float(parsimony_coefficient)
        self.function_set = tuple(DEFAULT_FUNCTIONS)

    # ── Protocol properties ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "GPLearn"

    @property
    def supports_warm_start(self) -> bool:
        # gplearn can continue evolution via warm_start, but reusing a
        # population across candidates WITHIN a stage is bug #29's
        # substitution trap: the search would be scoring one drifting
        # population rather than independent proposals.
        return False

    @property
    def supports_custom_objective(self) -> bool:
        # UIL formula trees are not gplearn fitness functions.
        return False

    def set_custom_objective(self, grad_fn, hess_fn) -> None:  # noqa: D102
        return None

    # ── Feature weighting ────────────────────────────────────────────

    def _scale(self, X: np.ndarray, feature_weights) -> np.ndarray:
        """Column scaling — the same channel RandomForest/KNN/MLP use.

        gplearn programs read raw feature values and there is no
        standardization inside the model to cancel the scaling (bug #25), so
        this genuinely changes the function space rather than only a penalty.
        """
        X = np.asarray(X, dtype=float)
        if feature_weights is None:
            return X
        w = np.asarray(feature_weights, dtype=float).ravel()
        if w.size != X.shape[1] or not np.isfinite(w).all():
            return X
        w = safe_minmax(w)
        if float(np.max(w)) < 1e-9:
            return X
        return X * (WEIGHT_FLOOR + (1.0 - WEIGHT_FLOOR) * w)

    # ── Fit / predict ────────────────────────────────────────────────

    def _make(self):
        # bug #31 invariant: every estimator gets an explicit random_state.
        return SymbolicClassifier(
            population_size=self.population_size,
            generations=self.generations,
            function_set=list(self.function_set),
            parsimony_coefficient=self.parsimony_coefficient,
            random_state=self.seed,
            n_jobs=1,
            verbose=0,
        )

    def fit_weighted(self, X, y, feature_weights, warm_start_model=None) -> dict:
        Xs = self._scale(X, feature_weights)
        y = np.asarray(y).ravel()
        classes = np.unique(y)
        # Store the weights so predict() re-applies the IDENTICAL transform
        # (bug #19: any inference path must repeat the fit-time transform).
        model: dict = {
            "kind": "gplearn",
            "feature_weights": (None if feature_weights is None
                                else np.asarray(feature_weights, float).tolist()),
            "classes": [int(c) for c in classes],
            "n_features": int(np.asarray(X).shape[1]),
        }
        if classes.size < 2:
            model["degenerate_class"] = int(classes[0]) if classes.size else 0
            return model

        if classes.size == 2:
            est = self._make()
            est.fit(Xs, (y == classes[1]).astype(int))
            model["estimator"] = est
            return model

        # Multiclass: one-vs-rest. Each head is fit on a DIFFERENT, heavily
        # imbalanced problem, so raw scores are not comparable across heads --
        # that is exactly the collapse of bugs #24/#33. gplearn's
        # predict_proba applies the sigmoid transform its fitness optimized,
        # so heads are compared as probabilities, never as raw programs.
        heads = {}
        for c in classes:
            est = self._make()
            est.fit(Xs, (y == c).astype(int))
            heads[int(c)] = est
        model["heads"] = heads
        return model

    def predict(self, model: object, X: np.ndarray) -> np.ndarray:
        assert isinstance(model, dict)
        Xs = self._scale(X, model.get("feature_weights"))
        n = np.asarray(X).shape[0]
        if "degenerate_class" in model:
            return np.full(n, model["degenerate_class"], dtype=int)

        classes = np.asarray(model.get("classes", [0, 1]))
        if "heads" in model:
            labels = sorted(model["heads"].keys())
            proba = np.column_stack([
                np.asarray(model["heads"][c].predict_proba(Xs))[:, 1]
                for c in labels
            ])
            return np.asarray(labels, dtype=int)[np.argmax(proba, axis=1)]

        est = model.get("estimator")
        if est is None:
            return np.full(n, int(classes[0]), dtype=int)
        pos = np.asarray(est.predict(Xs)).astype(int)
        return np.where(pos == 1, int(classes[1]), int(classes[0]))

    # ── Introspection ────────────────────────────────────────────────

    @staticmethod
    def _programs(model: dict) -> list:
        if "heads" in model:
            return [e._program for e in model["heads"].values()]
        est = model.get("estimator")
        return [est._program] if est is not None else []

    def get_feature_importance(self, model: object, n_features: int) -> np.ndarray:
        """Count how often each feature index appears in the evolved program(s).

        gplearn programs store terminals as ints (feature index) or floats
        (constants), so counting ints is the whole of it.
        """
        imp = np.zeros(int(n_features), dtype=float)
        if not isinstance(model, dict):
            return imp
        for prog in self._programs(model):
            for node in getattr(prog, "program", []):
                if isinstance(node, (int, np.integer)) and 0 <= int(node) < n_features:
                    imp[int(node)] += 1.0
        total = float(imp.sum())
        return imp / total if total > 0 else imp

    def get_model_complexity(self, model: object) -> int:
        if not isinstance(model, dict):
            return 0
        return int(sum(len(getattr(p, "program", [])) for p in self._programs(model)))

    def readable_program(self, model: object) -> str:
        """The evolved expression(s), as text.

        Readable, but NOT part of the editable-knowledge machinery:
        `cognition/law_export.py` does not apply, there is no FormulaLibrary
        provenance, and nothing round-trips through the VersionStore.
        """
        if not isinstance(model, dict):
            return "<no program>"
        progs = self._programs(model)
        if "heads" in model:
            return "\n".join(
                f"class {c}: {p}" for c, p in zip(sorted(model["heads"]), progs)
            )
        return str(progs[0]) if progs else "<no program>"

    # ── V20 control surface ──────────────────────────────────────────

    def get_controllable_params(self) -> dict:
        return {
            "population_size": {"current": self.population_size, "min": 100,
                                "max": 5000, "type": "int"},
            "generations": {"current": self.generations, "min": 5, "max": 100,
                            "type": "int"},
            "parsimony_coefficient": {"current": self.parsimony_coefficient,
                                      "min": 1e-5, "max": 0.05, "type": "float"},
        }

    def apply_config(self, config: dict) -> None:
        if "population_size" in config:
            self.population_size = max(100, int(config["population_size"]))
        if "generations" in config:
            self.generations = max(1, int(config["generations"]))
        if "parsimony_coefficient" in config:
            self.parsimony_coefficient = float(
                np.clip(float(config["parsimony_coefficient"]), 1e-6, 0.1))

    def get_current_config(self) -> dict:
        return {
            "population_size": self.population_size,
            "generations": self.generations,
            "parsimony_coefficient": self.parsimony_coefficient,
        }
