"""FIG-XGBoost Backward-Compatible Wrapper.

This file preserves the external API (FIG_XGBoost_Embedded with fit/predict)
but delegates everything to the new Universal Optimizer + XGBoost Adapter.

Existing code that does:
    model = FIG_XGBoost_Embedded(cfg, seed, dataset, fold, start_fold)
    model.fit(X, y)
    preds = model.predict(X_test)

continues to work unchanged.
"""

from __future__ import annotations
import numpy as np

from figpso_uil.adapters.xgboost_adapter import XGBoostAdapter
from figpso_uil.search.pso_search import PSOSearch
from figpso_uil.core.optimizer import UniversalOptimizer
from figpso_uil.core.failsafe import FailsafeEngine


class FIG_XGBoost_Embedded:
    """Backward-compatible wrapper. Delegates to UniversalOptimizer.
    
    External API unchanged:
        model = FIG_XGBoost_Embedded(cfg, seed, dataset, fold, start_fold)
        model.fit(X, y)
        preds = model.predict(X_test)
    
    Internal: creates XGBoostAdapter + PSOSearch + UniversalOptimizer.
    """

    def __init__(
        self,
        cfg,
        seed: int,
        dataset_name: str,
        fold: int,
        start_fold: float,
    ):
        self.cfg = cfg
        self.seed = seed
        self.dataset_name = dataset_name
        self.fold = fold
        self.start_fold = start_fold

        self.best_model = None
        self.final_features = None
        self.knowledge_graph = []
        self.num_classes = 2

        # These will be set after fit()
        self._optimizer = None
        self._result = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FIG_XGBoost_Embedded":
        """Run co-evolutionary optimization via the Universal Optimizer."""
        self.num_classes = len(np.unique(y))

        # Create XGBoost adapter
        classifier = XGBoostAdapter(
            seed=self.seed,
            num_classes=self.num_classes,
            boosting_rounds=getattr(self.cfg, "boosting_rounds_per_stage", 50),
            max_depth=6,
            learning_rate=0.1,
        )

        # Use PSO search for backward compatibility
        search = PSOSearch()

        # Create failsafe with available adapters
        failsafe_adapters = [classifier]
        try:
            from figpso_uil.adapters.random_forest_adapter import RandomForestAdapter
            failsafe_adapters.append(
                RandomForestAdapter(seed=self.seed, num_classes=self.num_classes)
            )
        except ImportError:
            pass

        try:
            from figpso_uil.adapters.knn_adapter import KNNAdapter
            failsafe_adapters.append(
                KNNAdapter(seed=self.seed, num_classes=self.num_classes)
            )
        except ImportError:
            pass

        try:
            from figpso_uil.adapters.tabpfn_adapter import TabPFNAdapter
            failsafe_adapters.append(
                TabPFNAdapter(seed=self.seed, num_classes=self.num_classes)
            )
        except ImportError:
            pass

        failsafe = FailsafeEngine(all_adapters=failsafe_adapters)

        # Create Universal Optimizer
        optimizer = UniversalOptimizer(
            cfg=self.cfg,
            seed=self.seed,
            dataset_name=self.dataset_name,
            fold=self.fold,
            start_fold=self.start_fold,
            classifier=classifier,
            search=search,
        )
        optimizer.set_failsafe(failsafe)

        # Run optimization
        print(f"    [UIL v19] Starting {getattr(self.cfg, 'stages', 10)} stages...")
        result = optimizer.fit(X, y)

        # Map results back to old API
        self.best_model = result["model"]
        self.final_features = result["selected_features"]
        self._optimizer = optimizer
        self._result = result

        print(
            f"    [UIL v19] Done. "
            f"Accuracy={result['accuracy']:.4f}, "
            f"F1={result['f1_score']:.4f}, "
            f"Features={result['n_selected']}/{X.shape[1]}, "
            f"Stress={result['max_stress']:.2f}, "
            f"Principles={result['principles_discovered']}"
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the best model."""
        if self._optimizer is not None:
            return self._optimizer.predict(X)

        # Fallback (shouldn't happen)
        import xgboost as xgb
        dtest = xgb.DMatrix(X)
        preds = self.best_model.predict(dtest)
        if self.num_classes > 2:
            return np.asarray(preds, dtype=int)
        else:
            return (preds > 0.5).astype(int)
