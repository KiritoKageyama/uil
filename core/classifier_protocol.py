"""ClassifierAdapter Protocol — the interface any classifier must implement.

Any classifier that wants to work with the Universal Intelligence Layer
must implement this protocol. The UIL never sees XGBoost, RandomForest, etc.
It only sees ClassifierAdapter.

To add a new classifier, write a ~50-line adapter file. The entire UIL
(journal, miner, library, executor, stress, search) works unchanged.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class ClassifierAdapter(Protocol):
    """Protocol that any classifier adapter must satisfy.
    
    The UIL optimizer talks to classifiers ONLY through this interface.
    This ensures the intelligence layer is truly classifier-agnostic.
    """

    @property
    def name(self) -> str:
        """Human-readable classifier name for journal logging.
        
        Examples: 'XGBoost', 'RandomForest', 'KNN', 'TabPFN'
        """
        ...

    @property
    def supports_warm_start(self) -> bool:
        """Whether this classifier can continue training from a previous model.
        
        True for XGBoost (incremental boosting), RandomForest (warm_start=True).
        False for KNN (no incremental learning), TabPFN (single forward pass).
        """
        ...

    def fit_weighted(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_weights: np.ndarray,
        warm_start_model: object | None = None,
    ) -> object:
        """Train the classifier using feature_weights to bias feature selection.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training labels, shape (n_samples,).
            feature_weights: Array of shape (n_features,) in [0, 1].
                Higher values indicate more important features. How the
                classifier uses these is adapter-specific:
                - XGBoost: DMatrix feature_weights (biases column sampling)
                - RandomForest: Feature scaling (X * weights)
                - KNN: Feature scaling (X * weights)
                - TabPFN: Feature selection mask (weight > threshold)
            warm_start_model: Previous model to continue training from.
                Only used if supports_warm_start is True.

        Returns:
            The trained model object (type is adapter-specific).
        """
        ...

    def predict(self, model: object, X: np.ndarray) -> np.ndarray:
        """Predict class labels using the trained model.

        Args:
            model: The model object returned by fit_weighted.
            X: Test features, shape (n_samples, n_features).

        Returns:
            Predicted labels, shape (n_samples,).
        """
        ...

    def get_feature_importance(
        self, model: object, n_features: int
    ) -> np.ndarray:
        """Extract feature importance scores from the trained model.

        This is how the classifier feeds back into the UIL's search.
        The optimizer uses this to update its guidance vector.

        Args:
            model: The trained model object.
            n_features: Total number of features in the dataset.

        Returns:
            Importance array of shape (n_features,), normalized to [0, 1].
        """
        ...

    def get_model_complexity(self, model: object) -> int:
        """Return a classifier-agnostic complexity measure.

        Used by the journal to track computational cost without
        knowing which classifier is being used.

        Examples:
            - XGBoost: number of trees built
            - RandomForest: number of estimators
            - KNN: number of training samples stored
            - TabPFN: number of features used
        """
        ...

    # ── V20: Full Classifier Control ─────────────────────────────────

    def get_controllable_params(self) -> dict:
        """Return all hyperparameters the UIL can control, with ranges.

        Returns a dict of:
            param_name: {
                "current": current_value,
                "min": minimum_value,
                "max": maximum_value,
                "type": "int" | "float" | "categorical",
                "options": [list of options for categorical]
            }
        """
        ...

    def apply_config(self, config: dict) -> None:
        """Apply a configuration dict of hyperparameters before fitting.

        Only keys present in the dict are updated.
        Unknown keys are silently ignored.

        Args:
            config: Dict of param_name → value.
        """
        ...

    def get_current_config(self) -> dict:
        """Return the current hyperparameter configuration.

        Used by the journal to record what was used.
        Returns a flat dict of param_name → current_value.
        """
        ...

    @property
    def supports_custom_objective(self) -> bool:
        """Whether this classifier can accept UIL formula trees as objectives.

        True for XGBoost (custom obj param).
        False for RandomForest, KNN (sklearn limitation).
        """
        ...

    def set_custom_objective(
        self,
        grad_fn: object | None,
        hess_fn: object | None,
    ) -> None:
        """Set a custom objective function built by the Formula Engine.

        Args:
            grad_fn: Callable (y_pred, y_true) → gradient array.
                     Set to None to use default objective.
            hess_fn: Callable (y_pred, y_true) → hessian array.
                     Set to None to use default objective.
        """
        ...
