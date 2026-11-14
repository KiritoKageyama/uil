"""Curated REAL tabular datasets — no synthetic generators.

All loaders return (X, y) as float64 / int labels from sklearn or OpenML.
Used by the web UI and experiments/run_curated_real.py so UIL memory
is built only from genuine classification benchmarks.
"""

from __future__ import annotations
import numpy as np

# name -> (display_label, category, loader)
_CURATED: dict[str, tuple[str, str, callable]] = {}


def _register(name: str, label: str, category: str, loader):
    _CURATED[name] = (label, category, loader)


def _load_sklearn(name: str, loader_fn, *, max_samples: int | None = None):
    data = loader_fn()
    X = np.asarray(data.data, dtype=np.float64)
    y = np.asarray(data.target, dtype=np.int64)
    if max_samples is not None and len(y) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y), max_samples, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def _init_registry() -> None:
    if _CURATED:
        return

    from sklearn.datasets import (
        load_breast_cancer,
        load_iris,
        load_wine,
        load_digits,
        load_diabetes,
    )

    _register(
        "real_breast_cancer",
        "Breast Cancer Wisconsin (569×30, binary)",
        "medical",
        lambda: _load_sklearn("bc", load_breast_cancer),
    )
    _register(
        "real_iris",
        "Iris (150×4, 3-class)",
        "classic",
        lambda: _load_sklearn("iris", load_iris),
    )
    _register(
        "real_wine",
        "Wine (178×13, 3-class)",
        "classic",
        lambda: _load_sklearn("wine", load_wine),
    )
    _register(
        "real_digits",
        "Digits (1797×64, 10-class)",
        "vision_tabular",
        lambda: _load_sklearn("digits", load_digits, max_samples=900),
    )
    # Diabetes is regression in sklearn — binarize target into high/low
    def _diabetes_binary():
        data = load_diabetes()
        X = np.asarray(data.data, dtype=np.float64)
        med = float(np.median(data.target))
        y = (np.asarray(data.target) > med).astype(np.int64)
        return X, y

    _register(
        "real_diabetes_risk",
        "Diabetes (442×10, binary from regression target)",
        "medical",
        _diabetes_binary,
    )

    # OpenML classics (optional — skip if unavailable offline)
    try:
        import pandas as pd
        from sklearn.datasets import fetch_openml
        from sklearn.preprocessing import LabelEncoder

        def _openml_frame(data_id: int | None = None, name: str | None = None, max_samples: int | None = None):
            if name:
                bunch = fetch_openml(name=name, version=1, as_frame=True, parser="auto")
            else:
                bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
            df = bunch.frame.copy()
            target_col = bunch.target_names[0] if bunch.target_names else df.columns[-1]
            y = LabelEncoder().fit_transform(df[target_col].astype(str))
            X_df = df.drop(columns=[target_col])
            # Encode categoricals; keep numeric columns
            X_df = pd.get_dummies(X_df, drop_first=False)
            X = X_df.to_numpy(dtype=np.float64)
            y = np.asarray(y, dtype=np.int64)
            mask = np.isfinite(X).all(axis=1)
            X, y = X[mask], y[mask]
            if max_samples and len(y) > max_samples:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(y), max_samples, replace=False)
                X, y = X[idx], y[idx]
            return X, y

        _register(
            "real_ionosphere",
            "Ionosphere OpenML (351×34, binary)",
            "physics",
            lambda: _openml_frame(59),
        )
        _register(
            "real_credit_g",
            "German Credit OpenML (~1000×20, binary)",
            "finance",
            lambda: _openml_frame(31, max_samples=800),
        )
        _register(
            "real_spambase",
            "Spambase OpenML (4601×57, binary)",
            "text_tabular",
            lambda: _openml_frame(44, max_samples=1200),
        )
        _register(
            "real_sonar",
            "Sonar OpenML (208×60, binary)",
            "physics",
            lambda: _openml_frame(name="sonar"),
        )
    except Exception:
        pass


def list_curated() -> list[str]:
    _init_registry()
    return list(_CURATED.keys())


def describe_curated() -> list[dict]:
    _init_registry()
    out = []
    for name, (label, category, loader) in _CURATED.items():
        try:
            X, y = loader()
            out.append({
                "name": name,
                "label": label,
                "category": category,
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "n_classes": int(len(np.unique(y))),
            })
        except Exception as e:
            out.append({
                "name": name,
                "label": label,
                "category": category,
                "error": str(e),
            })
    return out


def load_curated(name: str, seed: int = 42):
    _init_registry()
    if name not in _CURATED:
        raise KeyError(f"Unknown curated dataset '{name}'. Known: {list(_CURATED)}")
    _, _, loader = _CURATED[name]
    return loader()
