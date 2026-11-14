"""RandomForest adapter for the UIL ClassifierAdapter protocol.

RandomForest uses feature scaling (X * weights) to bias feature importance,
supports warm-start via sklearn's warm_start parameter, and provides
native feature_importances_.

V20: Full hyperparameter control.
"""

from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from figpso_uil.core.utils import safe_minmax


class RandomForestAdapter:
    """Adapts sklearn RandomForestClassifier to the ClassifierAdapter protocol."""

    def __init__(
        self,
        seed: int,
        num_classes: int,
        trees_per_stage: int = 50,
        max_depth: int | None = None,
    ):
        self.seed = seed
        self.num_classes = num_classes
        self.trees_per_stage = trees_per_stage
        self.max_depth = max_depth
        self._total_estimators = 0
        # V20: additional controllable params
        self.min_samples_split = 2
        self.max_features = "sqrt"
        self.criterion = "gini"

    # ── Protocol properties ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "RandomForest"

    @property
    def supports_warm_start(self) -> bool:
        return True

    @property
    def supports_custom_objective(self) -> bool:
        return False  # sklearn RF doesn't support custom objectives

    # ── Protocol methods ─────────────────────────────────────────────

    def fit_weighted(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_weights: np.ndarray,
        warm_start_model: object | None = None,
    ) -> object:
        """Train RandomForest with feature scaling to bias importance."""
        weights = np.clip(feature_weights, 1e-6, None)
        X_weighted = X * weights[np.newaxis, :]

        if warm_start_model is not None:
            self._total_estimators += self.trees_per_stage
            warm_start_model.n_estimators = self._total_estimators
            warm_start_model.fit(X_weighted, y)
            return warm_start_model

        # Fresh model with V20 controllable params
        self._total_estimators = self.trees_per_stage
        model = RandomForestClassifier(
            n_estimators=self.trees_per_stage,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            criterion=self.criterion,
            warm_start=True,
            random_state=self.seed,
            n_jobs=-1,
        )
        model.fit(X_weighted, y)
        return model

    def predict(self, model: object, X: np.ndarray) -> np.ndarray:
        """Predict using the trained RandomForest model."""
        return model.predict(X)

    def get_feature_importance(
        self, model: object, n_features: int
    ) -> np.ndarray:
        """Extract Gini-based feature importance from RandomForest."""
        raw_importance = model.feature_importances_
        importance = np.zeros(n_features)
        importance[: len(raw_importance)] = raw_importance
        return safe_minmax(importance)

    def get_model_complexity(self, model: object) -> int:
        """Number of estimators (trees) in the forest."""
        return model.n_estimators

    # ── V20: Full Classifier Control ─────────────────────────────────

    def get_controllable_params(self) -> dict:
        """All RandomForest hyperparameters the UIL can adjust."""
        return {
            "n_estimators": {
                "current": self.trees_per_stage,
                "min": 10, "max": 500, "type": "int",
            },
            "max_depth": {
                "current": self.max_depth if self.max_depth is not None else 0,
                "min": 0, "max": 30, "type": "int",  # 0 = None (unlimited)
            },
            "min_samples_split": {
                "current": self.min_samples_split,
                "min": 2, "max": 50, "type": "int",
            },
            "max_features": {
                "current": self.max_features,
                "type": "categorical",
                "options": ["sqrt", "log2", 0.5, 0.7, 1.0],
            },
            "criterion": {
                "current": self.criterion,
                "type": "categorical",
                "options": ["gini", "entropy"],
            },
        }

    def apply_config(self, config: dict) -> None:
        """Apply hyperparameter configuration from UIL."""
        if "n_estimators" in config:
            self.trees_per_stage = int(np.clip(config["n_estimators"], 10, 500))
        if "max_depth" in config:
            val = int(config["max_depth"])
            self.max_depth = val if val > 0 else None
        if "min_samples_split" in config:
            self.min_samples_split = int(np.clip(config["min_samples_split"], 2, 50))
        if "max_features" in config:
            self.max_features = config["max_features"]
        if "criterion" in config:
            if config["criterion"] in ("gini", "entropy"):
                self.criterion = config["criterion"]

    def get_current_config(self) -> dict:
        """Report current RF configuration for journal."""
        return {
            "n_estimators": self.trees_per_stage,
            "max_depth": self.max_depth if self.max_depth is not None else 0,
            "min_samples_split": self.min_samples_split,
            "max_features": self.max_features,
            "criterion": self.criterion,
        }

    def set_custom_objective(
        self,
        grad_fn: object | None,
        hess_fn: object | None,
    ) -> None:
        """RF does not support custom objectives. No-op."""
        pass
