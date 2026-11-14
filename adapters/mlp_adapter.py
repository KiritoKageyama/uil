"""MLP (multi-layer perceptron) adapter for the UIL ClassifierAdapter protocol.

Why this exists: every other learner in the registry is either a tree
ensemble or the symbolic family. An MLP is the one hypothesis class that
learns feature INTERACTIONS densely and continuously, which is the gap E5
identified on sonar (linear models cap ~0.77, non-linear reach ~0.88).

What it costs: readability. A fitted MLP is a weight matrix, not a law you
can read or edit, so it gives up the project's two surviving value claims
(symbolic extrapolation, human-editable knowledge). Use it as a CEILING
PROBE -- "what does dense non-linearity get on this data?" -- and as an
honest baseline, not as the default learner.

Feature weights reach the model the same way they do for RandomForest:
column scaling before fit. Unlike the sparse_linear path there is no
standardization inside the model to cancel it (bug #25), so the scaling
survives -- BUT sklearn's MLP is scale-sensitive, so the weights genuinely
change the optimization landscape rather than just the penalty. That is a
different channel semantics from the L1 path; treat it as such.
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier

from figpso_uil.core.utils import safe_minmax

WEIGHT_FLOOR = 0.25  # matches the symbolic adapter's floor


class MLPAdapter:
    """Adapts sklearn MLPClassifier to the ClassifierAdapter protocol."""

    def __init__(
        self,
        seed: int,
        num_classes: int,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        max_iter: int = 400,
    ):
        self.seed = seed
        self.num_classes = num_classes
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.max_iter = int(max_iter)
        self.alpha = 1e-4          # L2 penalty
        self.learning_rate_init = 1e-3
        self.activation = "relu"
        self._last_loss_curve: list[float] = []

    # ── Protocol properties ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "MLP"

    @property
    def supports_warm_start(self) -> bool:
        # sklearn supports warm_start, but reusing a net across candidates
        # inside one stage is exactly the substitution trap of bug #29:
        # candidates exist to differ, and a warm net biases every one of
        # them toward the first. Resume is not worth that here.
        return False

    @property
    def supports_custom_objective(self) -> bool:
        return False

    # ── Protocol methods ─────────────────────────────────────────────

    def _scale(self, X: np.ndarray, feature_weights: np.ndarray | None) -> np.ndarray:
        if feature_weights is None:
            return X
        w = np.asarray(feature_weights, dtype=float)
        if w.shape[0] != X.shape[1]:
            return X
        if float(np.max(w) - np.min(w)) < 1e-9:
            return X  # uniform weights carry no signal
        mult = WEIGHT_FLOOR + (1.0 - WEIGHT_FLOOR) * safe_minmax(w)
        return X * mult[np.newaxis, :]

    def fit_weighted(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_weights: np.ndarray,
        warm_start_model: object | None = None,
    ) -> object:
        Xw = self._scale(np.asarray(X, dtype=float), feature_weights)
        model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.seed,
            early_stopping=False,
        )
        import warnings
        with warnings.catch_warnings():
            # Non-convergence at a small max_iter is a budget choice here,
            # not an error; the stage loop is what tunes it.
            warnings.filterwarnings("ignore", category=UserWarning)
            model.fit(Xw, y)
        self._last_loss_curve = list(getattr(model, "loss_curve_", []) or [])
        # Stash the weights the net was TRAINED under; predict must apply
        # the identical transform or it serves out-of-distribution rows
        # (the same class of bug as #19).
        model._uil_feature_weights = (
            np.asarray(feature_weights, dtype=float).copy()
            if feature_weights is not None else None
        )
        return model

    def predict(self, model: object, X: np.ndarray) -> np.ndarray:
        Xw = self._scale(
            np.asarray(X, dtype=float),
            getattr(model, "_uil_feature_weights", None),
        )
        return model.predict(Xw)

    def get_feature_importance(
        self, model: object, n_features: int
    ) -> np.ndarray:
        """Input-layer weight magnitude as an importance proxy.

        An MLP has no native importance. Summing |w| over the first layer
        measures how much each input feeds the network -- a real signal, but
        a coarser one than a tree's split gain. It is NOT a causal claim
        about the feature; do not let it be mined as one.
        """
        importance = np.zeros(n_features)
        coefs = getattr(model, "coefs_", None)
        if not coefs:
            return importance
        w0 = np.abs(np.asarray(coefs[0], dtype=float)).sum(axis=1)
        importance[: min(len(w0), n_features)] = w0[:n_features]
        return safe_minmax(importance)

    def get_model_complexity(self, model: object) -> int:
        """Total learnable parameters."""
        coefs = getattr(model, "coefs_", None) or []
        inters = getattr(model, "intercepts_", None) or []
        return int(
            sum(np.asarray(c).size for c in coefs)
            + sum(np.asarray(b).size for b in inters)
        )

    # ── V20: Full Classifier Control ─────────────────────────────────

    def get_controllable_params(self) -> dict:
        return {
            "alpha": {
                "current": self.alpha, "min": 1e-6, "max": 1e-1, "type": "float",
            },
            "learning_rate_init": {
                "current": self.learning_rate_init,
                "min": 1e-5, "max": 1e-1, "type": "float",
            },
            "max_iter": {
                "current": self.max_iter, "min": 50, "max": 2000, "type": "int",
            },
            "activation": {
                "current": self.activation,
                "type": "categorical",
                "options": ["relu", "tanh", "logistic"],
            },
        }

    def apply_config(self, config: dict) -> None:
        if "alpha" in config:
            self.alpha = float(np.clip(config["alpha"], 1e-6, 1e-1))
        if "learning_rate_init" in config:
            self.learning_rate_init = float(
                np.clip(config["learning_rate_init"], 1e-5, 1e-1)
            )
        if "max_iter" in config:
            self.max_iter = int(np.clip(config["max_iter"], 50, 2000))
        if "activation" in config:
            if config["activation"] in ("relu", "tanh", "logistic"):
                self.activation = str(config["activation"])
        # hidden_layer_sizes is ARCHITECTURE, not a hyperparameter: per the
        # bug #21 invariant, a per-stage search must not be able to reset it.
        # Only an explicit constructor argument or the router may set it.

    def get_current_config(self) -> dict:
        return {
            "alpha": self.alpha,
            "learning_rate_init": self.learning_rate_init,
            "max_iter": self.max_iter,
            "activation": self.activation,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
        }

    def set_custom_objective(
        self, grad_fn: object | None, hess_fn: object | None
    ) -> None:
        """MLP does not take a custom objective here. No-op."""
        pass
