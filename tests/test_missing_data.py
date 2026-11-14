"""Missing-data perception: train-only impute + indicators (bug #19 family)."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from figpso_uil.core.missing_data import fit_missing_data, MissingDataSpec
from figpso_uil.core.classifier_registry import CLASSIFIERS
from figpso_uil.core.optimizer import UniversalOptimizer
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch


def test_m1_clean_data_identity():
    """Fully finite train → forward leaves values and shape unchanged."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    spec = fit_missing_data(X, add_indicators=True)
    assert spec.indicator_cols == []
    assert spec.n_features_out == 5
    out = spec.forward(X)
    assert out.shape == X.shape
    assert np.allclose(out, X)
    print("  M1 clean identity OK")


def test_m2_median_impute_and_indicator():
    X = np.array([
        [1.0, 10.0],
        [np.nan, 20.0],
        [3.0, np.nan],
        [5.0, 40.0],
    ], dtype=float)
    spec = fit_missing_data(X, add_indicators=True)
    assert spec.n_original == 2
    assert set(spec.indicator_cols) == {0, 1}
    # medians of finite: col0 = 3.0, col1 = 20.0
    assert abs(spec.fills[0] - 3.0) < 1e-9
    assert abs(spec.fills[1] - 20.0) < 1e-9
    out = spec.forward(X)
    assert out.shape == (4, 4)  # 2 data + 2 indicators
    assert np.isfinite(out[:, :2]).all()
    assert out[1, 0] == 3.0 and out[2, 1] == 20.0
    assert out[1, 2] == 1.0 and out[0, 2] == 0.0  # col0 indicator
    assert out[2, 3] == 1.0 and out[0, 3] == 0.0  # col1 indicator
    # Serve-time NaN uses the same fill + indicator schema
    Xte = np.array([[np.nan, 15.0]], dtype=float)
    te = spec.forward(Xte)
    assert te.shape == (1, 4)
    assert te[0, 0] == 3.0 and te[0, 2] == 1.0
    print("  M2 median + indicators OK")


def test_m3_mode_for_integerish():
    col = np.array([1, 1, 1, 2, 2, np.nan], dtype=float).reshape(-1, 1)
    spec = fit_missing_data(col, add_indicators=True)
    assert spec.strategies[0] == "mode"
    assert spec.fills[0] == 1.0
    out = spec.forward(col)
    assert out[5, 0] == 1.0
    print("  M3 mode fill OK")


def test_m4_roundtrip_dict():
    X = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=float)
    spec = fit_missing_data(X)
    again = MissingDataSpec.from_dict(spec.to_dict())
    assert again.fills == spec.fills
    assert again.indicator_cols == spec.indicator_cols
    assert np.allclose(again.forward(X), spec.forward(X))
    print("  M4 dict round-trip OK")


def test_m5_optimizer_train_serve_with_nans():
    """End-to-end: plant NaNs, fit, predict without crash; schema matches."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(120, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    # Plant ~15% missing on two columns
    mask = rng.random(X.shape) < 0.15
    X[mask[:, 0], 0] = np.nan
    X[mask[:, 2], 2] = np.nan

    class _Cfg:
        stages = 2
        n_particles = 4
        pso_iters = 3
        particles = 4
        pso_iters_per_stage = 3
        boosting_rounds = 10
        cv_folds = 1
        use_resource_budget = False
        disable_routing = True
        enable_stagnation_evolution = False
        target_accuracy = None
        max_stages = 2
        handle_missing_data = True
        missing_indicators = True
        situation_adaptive = False
        maturity_soft_normalization = False
        stress_adaptive_blend = False

    sd = tempfile.mkdtemp(prefix="miss_")
    clf = CLASSIFIERS["SymbolicFormula"](7, 2, 10, sd)
    clf.apply_config({"learning_mode": "sparse_linear"})
    opt = UniversalOptimizer(
        cfg=_Cfg(),
        seed=7,
        dataset_name="miss_toy",
        fold=0,
        start_fold=0,
        classifier=clf,
        search=PrincipleGuidedSearch(),
        classifier_factory=CLASSIFIERS["SymbolicFormula"],
        classifier_build_args=(7, 2, 10, sd),
        state_dir=sd,
    )
    result = opt.fit(X, y)
    assert result.get("model") is not None
    meta = opt._observability_meta.get("missing_data") or {}
    assert meta.get("n_cols_with_missing", 0) >= 1
    assert meta.get("n_indicators", 0) >= 1
    # Predict on matrix that still has NaNs
    pred = opt.predict(X[:20])
    assert len(pred) == 20
    assert opt._missing_data is not None
    assert opt._missing_data.n_features_out == 6 + len(opt._missing_data.indicator_cols)
    print(f"  M5 e2e OK (missing_data={meta})")


def test_m6_disable_flag():
    """cfg.handle_missing_data=False leaves _missing_data unset."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 4))
    y = (X[:, 0] > 0).astype(int)

    class _Cfg:
        stages = 1
        n_particles = 3
        pso_iters = 2
        particles = 3
        pso_iters_per_stage = 2
        boosting_rounds = 5
        cv_folds = 1
        use_resource_budget = False
        disable_routing = True
        enable_stagnation_evolution = False
        target_accuracy = None
        max_stages = 1
        handle_missing_data = False
        situation_adaptive = False

    sd = tempfile.mkdtemp(prefix="miss_off_")
    clf = CLASSIFIERS["SymbolicFormula"](1, 2, 5, sd)
    clf.apply_config({"learning_mode": "sparse_linear"})
    opt = UniversalOptimizer(
        cfg=_Cfg(),
        seed=1,
        dataset_name="miss_off",
        fold=0,
        start_fold=0,
        classifier=clf,
        search=PrincipleGuidedSearch(),
        classifier_factory=CLASSIFIERS["SymbolicFormula"],
        classifier_build_args=(1, 2, 5, sd),
        state_dir=sd,
    )
    opt.fit(X, y)
    assert opt._missing_data is None
    print("  M6 disable flag OK")


if __name__ == "__main__":
    test_m1_clean_data_identity()
    test_m2_median_impute_and_indicator()
    test_m3_mode_for_integerish()
    test_m4_roundtrip_dict()
    test_m5_optimizer_train_serve_with_nans()
    test_m6_disable_flag()
    print("All missing-data tests passed")
