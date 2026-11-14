"""XGBoost adapter for the UIL ClassifierAdapter protocol.

XGBoost uses DMatrix feature_weights to bias column sampling,
supports warm-start (incremental boosting), and provides native
feature importance via gain scores.

V20: Full hyperparameter control + custom formula objectives.
"""

from __future__ import annotations
import numpy as np

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from figpso_uil.core.utils import safe_minmax


class XGBoostAdapter:
    """Adapts XGBoost to the ClassifierAdapter protocol."""

    def __init__(
        self,
        seed: int,
        num_classes: int,
        boosting_rounds: int = 50,
        max_depth: int = 6,
        learning_rate: float = 0.1,
    ):
        if not _HAS_XGB:
            raise ImportError("xgboost is required for XGBoostAdapter. Install with: pip install xgboost")
        self.seed = seed
        self.num_classes = num_classes
        self.boosting_rounds = boosting_rounds
        self._custom_obj = None  # V20: custom formula objective
        self._build_params(max_depth, learning_rate)

    # ── Protocol properties ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "XGBoost"

    @property
    def supports_warm_start(self) -> bool:
        return True

    @property
    def supports_custom_objective(self) -> bool:
        # M0.3 (v22 blueprint §4.4): the formula engine's gradient/Hessian
        # are written for BINARY logistic margins. Under multi:softprob the
        # objective callable receives (n_samples*num_classes)-shaped preds
        # and crashes. Custom objectives are therefore a binary-only
        # capability, not an XGBoost-wide one.
        return self.num_classes <= 2

    # ── Protocol methods ─────────────────────────────────────────────

    def fit_weighted(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_weights: np.ndarray,
        warm_start_model: object | None = None,
    ) -> object:
        """Train XGBoost with feature_weights biasing column sampling."""
        weights = np.clip(feature_weights, 1e-6, None)
        dtrain = xgb.DMatrix(X, label=y, feature_weights=weights)

        train_kwargs = {
            "params": self.params,
            "dtrain": dtrain,
            "num_boost_round": self.boosting_rounds,
            "xgb_model": warm_start_model,
        }

        # V20: custom formula objective (binary-only -- see
        # supports_custom_objective; guarded again here in case a caller
        # sets an objective without checking the capability flag)
        if self._custom_obj is not None and self.num_classes <= 2:
            train_kwargs["obj"] = self._custom_obj

        model = xgb.train(**train_kwargs)
        return model

    def predict(self, model: object, X: np.ndarray) -> np.ndarray:
        """Predict using the trained XGBoost model."""
        dtest = xgb.DMatrix(X)
        preds = model.predict(dtest)
        if self.num_classes > 2:
            return np.asarray(preds, dtype=int)
        else:
            return (np.asarray(preds) > 0.5).astype(int)

    def get_feature_importance(
        self, model: object, n_features: int
    ) -> np.ndarray:
        """Extract gain-based importance from XGBoost."""
        gain_dict = model.get_score(importance_type="gain")
        importance = np.zeros(n_features)
        for feat_name, gain in gain_dict.items():
            try:
                feat_idx = int(feat_name.replace("f", ""))
                if feat_idx < n_features:
                    importance[feat_idx] = gain
            except ValueError:
                pass
        return safe_minmax(importance)

    def get_model_complexity(self, model: object) -> int:
        """Number of trees built."""
        return len(model.get_dump())

    # ── V20: Full Classifier Control ─────────────────────────────────

    def get_controllable_params(self) -> dict:
        """All XGBoost hyperparameters the UIL can adjust."""
        return {
            "max_depth": {
                "current": self.params.get("max_depth", 6),
                "min": 1, "max": 15, "type": "int",
            },
            "learning_rate": {
                "current": self.params.get("learning_rate", 0.1),
                "min": 0.001, "max": 1.0, "type": "float",
            },
            "subsample": {
                "current": self.params.get("subsample", 0.8),
                "min": 0.1, "max": 1.0, "type": "float",
            },
            "colsample_bytree": {
                "current": self.params.get("colsample_bytree", 1.0),
                "min": 0.1, "max": 1.0, "type": "float",
            },
            "colsample_bynode": {
                "current": self.params.get("colsample_bynode", 0.2),
                "min": 0.1, "max": 1.0, "type": "float",
            },
            "min_child_weight": {
                "current": self.params.get("min_child_weight", 1),
                "min": 0, "max": 100, "type": "int",
            },
            "gamma": {
                "current": self.params.get("gamma", 0),
                "min": 0.0, "max": 10.0, "type": "float",
            },
            "reg_alpha": {
                "current": self.params.get("reg_alpha", 0),
                "min": 0.0, "max": 10.0, "type": "float",
            },
            "reg_lambda": {
                "current": self.params.get("reg_lambda", 1),
                "min": 0.0, "max": 10.0, "type": "float",
            },
            "boosting_rounds": {
                "current": self.boosting_rounds,
                "min": 10, "max": 500, "type": "int",
            },
        }

    def apply_config(self, config: dict) -> None:
        """Apply hyperparameter configuration from UIL."""
        controllable = self.get_controllable_params()
        for key, value in config.items():
            if key == "boosting_rounds":
                self.boosting_rounds = int(value)
            elif key in controllable and key in self.params:
                meta = controllable[key]
                if meta["type"] == "int":
                    self.params[key] = int(np.clip(value, meta["min"], meta["max"]))
                elif meta["type"] == "float":
                    self.params[key] = float(np.clip(value, meta["min"], meta["max"]))

    def get_current_config(self) -> dict:
        """Report current XGBoost configuration for journal."""
        return {
            "max_depth": self.params.get("max_depth", 6),
            "learning_rate": self.params.get("learning_rate", 0.1),
            "subsample": self.params.get("subsample", 0.8),
            "colsample_bytree": self.params.get("colsample_bytree", 1.0),
            "colsample_bynode": self.params.get("colsample_bynode", 0.2),
            "min_child_weight": self.params.get("min_child_weight", 1),
            "gamma": self.params.get("gamma", 0),
            "reg_alpha": self.params.get("reg_alpha", 0),
            "reg_lambda": self.params.get("reg_lambda", 1),
            "boosting_rounds": self.boosting_rounds,
        }

    def set_custom_objective(
        self,
        grad_fn: object | None,
        hess_fn: object | None,
    ) -> None:
        """Set custom objective from Formula Engine.

        When set, XGBoost uses this instead of built-in logistic/softmax.
        """
        if grad_fn is not None and hess_fn is not None:
            def _custom_obj(y_pred, dtrain):
                y_true = dtrain.get_label()
                grad = grad_fn(y_pred, y_true)
                hess = hess_fn(y_pred, y_true)
                return grad, hess
            self._custom_obj = _custom_obj
            self.params.pop("objective", None)
        else:
            self._custom_obj = None
            if self.num_classes > 2:
                self.params["objective"] = "multi:softmax"
            else:
                self.params["objective"] = "binary:logistic"

    # ── Internal ─────────────────────────────────────────────────────

    def _build_params(self, max_depth: int, learning_rate: float) -> None:
        """Build XGBoost parameter dict."""
        self.params = {
            "tree_method": "hist",
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
            "colsample_bynode": 0.2,
            # These four were previously absent from self.params, so
            # apply_config()'s `key in self.params` guard silently dropped
            # them — any Miner rule crediting gamma/reg_alpha/reg_lambda/
            # min_child_weight was attributing causation to params that
            # never reached XGBoost. Values here match XGBoost defaults
            # and the fallbacks already in get_controllable_params().
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "min_child_weight": 1,
            "random_state": self.seed,
            "n_jobs": -1,
            "verbosity": 0,
        }
        if self.num_classes > 2:
            self.params["objective"] = "multi:softmax"
            self.params["eval_metric"] = "mlogloss"
            self.params["num_class"] = self.num_classes
        else:
            self.params["objective"] = "binary:logistic"
            self.params["eval_metric"] = "logloss"
