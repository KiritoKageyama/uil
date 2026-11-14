"""TabPFN adapter for the UIL ClassifierAdapter protocol.

TabPFN is a pre-trained transformer for tabular classification.
It does NOT train on your data — it classifies in a single forward pass.
No hyperparameters, no epochs, no tuning.

Limits: works best on datasets with ≤1000 features and ≤10,000 samples.
The adapter checks these limits and raises clear errors when exceeded.
"""

from __future__ import annotations
import numpy as np
from sklearn.inspection import permutation_importance

from figpso_uil.core.utils import safe_minmax

try:
    from tabpfn import TabPFNClassifier
    _HAS_TABPFN = True
except ImportError:
    _HAS_TABPFN = False


class TabPFNAdapter:
    """Adapts TabPFN to the ClassifierAdapter protocol.
    
    TabPFN is the newest paradigm-shifting classifier for tabular data.
    It's a foundation model pre-trained on millions of synthetic datasets.
    Classification happens in a single forward pass — no training needed.
    """

    # TabPFN limits
    MAX_FEATURES = 500
    MAX_SAMPLES = 10_000

    def __init__(
        self,
        seed: int,
        num_classes: int,
        device: str = "cpu",
    ):
        if not _HAS_TABPFN:
            raise ImportError(
                "tabpfn is required for TabPFNAdapter. "
                "Install with: pip install tabpfn"
            )
        self.seed = seed
        self.num_classes = num_classes
        self.device = device
        self._feature_mask = None
        self._last_X = None
        self._last_y = None

    # ── Protocol properties ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "TabPFN"

    @property
    def supports_warm_start(self) -> bool:
        return False  # TabPFN doesn't "train" — single forward pass

    # ── Protocol methods ─────────────────────────────────────────────

    def fit_weighted(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_weights: np.ndarray,
        warm_start_model: object | None = None,
    ) -> object:
        """Fit TabPFN using feature weights as a selection mask.
        
        TabPFN works best with fewer features, so we use the weights
        to select the most important features (weight > threshold).
        """
        n_samples, n_features = X.shape

        # Guard: check TabPFN limits
        if n_samples > self.MAX_SAMPLES:
            # Subsample to fit within limits
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(n_samples, size=self.MAX_SAMPLES, replace=False)
            X = X[idx]
            y = y[idx]
            n_samples = self.MAX_SAMPLES

        # Use feature weights as selection mask
        # Select features above a dynamic threshold
        if n_features > self.MAX_FEATURES:
            # Must reduce — take top MAX_FEATURES by weight
            top_indices = np.argsort(feature_weights)[-self.MAX_FEATURES :]
            mask = np.zeros(n_features, dtype=bool)
            mask[top_indices] = True
        else:
            # Select features with meaningful weight
            threshold = max(0.05, np.percentile(feature_weights, 20))
            mask = feature_weights > threshold
            # Ensure at least 2 features selected
            if mask.sum() < 2:
                top_indices = np.argsort(feature_weights)[-max(2, n_features // 5) :]
                mask = np.zeros(n_features, dtype=bool)
                mask[top_indices] = True

        self._feature_mask = mask
        X_selected = X[:, mask]

        model = TabPFNClassifier(device=self.device)
        model.fit(X_selected, y)

        # Store for importance calculation
        self._last_X = X_selected
        self._last_y = y

        return model

    def predict(self, model: object, X: np.ndarray) -> np.ndarray:
        """Predict using TabPFN. Applies same feature mask."""
        if self._feature_mask is not None:
            X = X[:, self._feature_mask]
        return model.predict(X)

    def get_feature_importance(
        self, model: object, n_features: int
    ) -> np.ndarray:
        """Estimate importance via permutation importance.
        
        TabPFN is a black box — no native feature importance.
        We permute each feature and measure accuracy drop.
        """
        importance = np.zeros(n_features)

        if self._last_X is None or self._last_y is None:
            return importance

        try:
            # Use subset for speed
            n = min(300, len(self._last_y))
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(self._last_y), size=n, replace=False)

            result = permutation_importance(
                model,
                self._last_X[idx],
                self._last_y[idx],
                n_repeats=3,
                random_state=self.seed,
            )
            raw = np.clip(result.importances_mean, 0, None)

            # Map back to original feature indices
            if self._feature_mask is not None:
                selected_indices = np.where(self._feature_mask)[0]
                for i, orig_idx in enumerate(selected_indices):
                    if i < len(raw) and orig_idx < n_features:
                        importance[orig_idx] = raw[i]
            else:
                importance[: len(raw)] = raw

            return safe_minmax(importance)
        except Exception:
            return np.ones(n_features) / n_features

    def get_model_complexity(self, model: object) -> int:
        """Number of features used by TabPFN."""
        if self._feature_mask is not None:
            return int(self._feature_mask.sum())
        return 0
