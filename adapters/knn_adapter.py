"""KNN adapter for the UIL ClassifierAdapter protocol.

KNN uses feature scaling (X * weights) to bias distance calculations,
does not support warm-start (refits each time, but KNN is fast),
and uses permutation importance for feature importance estimation.

V20: Full hyperparameter control + custom distance metric support.
"""

from __future__ import annotations
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance

from figpso_uil.core.utils import safe_minmax


class KNNAdapter:
    """Adapts sklearn KNeighborsClassifier to the ClassifierAdapter protocol."""

    def __init__(
        self,
        seed: int,
        num_classes: int,
        n_neighbors: int = 5,
    ):
        self.seed = seed
        self.num_classes = num_classes
        self.n_neighbors = n_neighbors
        self._last_X = None
        self._last_y = None
        self._last_weights = None
        # V20: additional controllable params
        self.weights_mode = "uniform"  # 'uniform' or 'distance'
        self.metric = "minkowski"
        self.p = 2  # Minkowski power (2=euclidean, 1=manhattan)
        self._custom_metric = None  # V20: custom distance metric from formula engine

    # ── Protocol properties ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "KNN"

    @property
    def supports_warm_start(self) -> bool:
        return False

    @property
    def supports_custom_objective(self) -> bool:
        return True  # V20: KNN supports custom distance metric

    # ── Protocol methods ─────────────────────────────────────────────

    def fit_weighted(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_weights: np.ndarray,
        warm_start_model: object | None = None,
    ) -> object:
        """Train KNN with feature scaling to bias distance calculations."""
        weights = np.clip(feature_weights, 1e-6, None)
        X_weighted = X * weights[np.newaxis, :]

        k = min(self.n_neighbors, max(1, len(y) - 1))

        # Build KNN with V20 controllable params
        knn_kwargs = {
            "n_neighbors": k,
            "weights": self.weights_mode,
            "n_jobs": -1,
        }

        if self._custom_metric is not None:
            knn_kwargs["metric"] = self._custom_metric
        else:
            knn_kwargs["metric"] = self.metric
            if self.metric == "minkowski":
                knn_kwargs["p"] = self.p

        model = KNeighborsClassifier(**knn_kwargs)
        model.fit(X_weighted, y)

        self._last_X = X_weighted
        self._last_y = y
        self._last_weights = weights
        return model

    def predict(self, model: object, X: np.ndarray) -> np.ndarray:
        """Predict using the trained KNN model."""
        if self._last_weights is not None:
            X_weighted = X * self._last_weights[np.newaxis, :]
        else:
            X_weighted = X
        return model.predict(X_weighted)

    def get_feature_importance(
        self, model: object, n_features: int
    ) -> np.ndarray:
        """Estimate feature importance via permutation importance."""
        if self._last_X is None or self._last_y is None:
            return np.ones(n_features) / n_features

        try:
            n = min(500, len(self._last_y))
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(self._last_y), size=n, replace=False)

            result = permutation_importance(
                model,
                self._last_X[idx],
                self._last_y[idx],
                n_repeats=5,
                random_state=self.seed,
                n_jobs=-1,
            )
            importance = np.zeros(n_features)
            raw = result.importances_mean
            importance[: len(raw)] = np.clip(raw, 0, None)
            return safe_minmax(importance)
        except Exception:
            return np.ones(n_features) / n_features

    def get_model_complexity(self, model: object) -> int:
        """Number of training samples stored by KNN."""
        return getattr(model, "n_samples_fit_", 0)

    # ── V20: Full Classifier Control ─────────────────────────────────

    def get_controllable_params(self) -> dict:
        """All KNN hyperparameters the UIL can adjust."""
        return {
            "n_neighbors": {
                "current": self.n_neighbors,
                "min": 1, "max": 50, "type": "int",
            },
            "weights": {
                "current": self.weights_mode,
                "type": "categorical",
                "options": ["uniform", "distance"],
            },
            "metric": {
                "current": self.metric,
                "type": "categorical",
                "options": ["minkowski", "euclidean", "manhattan", "chebyshev"],
            },
            "p": {
                "current": self.p,
                "min": 1, "max": 5, "type": "int",
            },
        }

    def apply_config(self, config: dict) -> None:
        """Apply hyperparameter configuration from UIL."""
        if "n_neighbors" in config:
            self.n_neighbors = int(np.clip(config["n_neighbors"], 1, 50))
        if "weights" in config:
            if config["weights"] in ("uniform", "distance"):
                self.weights_mode = config["weights"]
        if "metric" in config:
            valid = ["minkowski", "euclidean", "manhattan", "chebyshev"]
            if config["metric"] in valid:
                self.metric = config["metric"]
        if "p" in config:
            self.p = int(np.clip(config["p"], 1, 5))

    def get_current_config(self) -> dict:
        """Report current KNN configuration for journal."""
        return {
            "n_neighbors": self.n_neighbors,
            "weights": self.weights_mode,
            "metric": self.metric,
            "p": self.p,
        }

    def set_custom_objective(
        self,
        grad_fn: object | None,
        hess_fn: object | None,
    ) -> None:
        """Set custom distance metric for KNN.

        For KNN, grad_fn is repurposed as a custom distance callable:
            distance_fn(x, y) -> float
        hess_fn is ignored.
        """
        if grad_fn is not None:
            self._custom_metric = grad_fn
        else:
            self._custom_metric = None
