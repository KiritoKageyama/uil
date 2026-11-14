"""Train-only missing-data handling (impute + optional indicators).

Fits on the training fold only; every later matrix (val, held-out, predict)
must go through the same ``forward``. Mirrors the bug #19 train/serve
invariant that already applies to ``core/normalization.py``.

On fully finite training data this is a no-op for values and (with the
default indicator policy) for shape — clean UCI benchmarks stay
byte-identical to the pre-module path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from figpso_uil.core.utils import sanitize_for_json


def _is_integerish(col: np.ndarray, frac: float = 0.95) -> bool:
    """True when most finite values sit near integers (mode fill preferred)."""
    finite = col[np.isfinite(col)]
    if finite.size == 0:
        return False
    near = np.abs(finite - np.round(finite)) < 1e-6
    return float(np.mean(near)) >= frac


def _fill_value(col: np.ndarray) -> tuple[float, str]:
    """Median for continuous; mode for integer-like; 0 if all missing."""
    finite = col[np.isfinite(col)]
    if finite.size == 0:
        return 0.0, "zero_all_missing"
    if _is_integerish(finite):
        rounded = np.round(finite).astype(np.int64)
        vals, counts = np.unique(rounded, return_counts=True)
        # Require a real plurality — unique one-offs fall back to median.
        if int(counts.max()) >= 2:
            return float(vals[int(np.argmax(counts))]), "mode"
    return float(np.median(finite)), "median"


@dataclass
class MissingDataSpec:
    """Per-column fill values + which columns get a missingness indicator."""

    fills: list[float] = field(default_factory=list)
    strategies: list[str] = field(default_factory=list)
    indicator_cols: list[int] = field(default_factory=list)
    n_original: int = 0
    feature_names: list[str] = field(default_factory=list)
    add_indicators: bool = True
    fitted_on: str = "train_fold"
    # Train missing rates (diagnostic / HITL), not used at transform time.
    missing_rate: list[float] = field(default_factory=list)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Impute non-finite entries; optionally append indicator columns.

        Indicator columns are 1.0 where the *original* cell was non-finite,
        0.0 otherwise. Only columns listed in ``indicator_cols`` (those that
        had ≥1 missing value on the train fold) are appended, in that order,
        so train and serve stay schema-aligned.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D matrix, got shape {X.shape}")
        if X.shape[1] != self.n_original:
            raise ValueError(
                f"MissingDataSpec expects {self.n_original} columns, "
                f"got {X.shape[1]}"
            )
        if not self.fills:
            return X.copy()

        out = X.copy()
        miss_mask = ~np.isfinite(out)
        for j, fill in enumerate(self.fills):
            if j >= out.shape[1]:
                break
            col_miss = miss_mask[:, j]
            if col_miss.any():
                out[col_miss, j] = fill

        if self.add_indicators and self.indicator_cols:
            inds = []
            for j in self.indicator_cols:
                if 0 <= j < miss_mask.shape[1]:
                    inds.append(miss_mask[:, j].astype(float))
                else:
                    inds.append(np.zeros(out.shape[0], dtype=float))
            out = np.column_stack([out] + inds)
        return out

    @property
    def n_features_out(self) -> int:
        n_ind = len(self.indicator_cols) if self.add_indicators else 0
        return self.n_original + n_ind

    def summary(self) -> dict[str, Any]:
        rates = [r for r in self.missing_rate if r > 0]
        return {
            "n_original": self.n_original,
            "n_features_out": self.n_features_out,
            "n_cols_with_missing": int(sum(1 for r in self.missing_rate if r > 0)),
            "n_indicators": len(self.indicator_cols) if self.add_indicators else 0,
            "max_missing_rate": round(float(max(rates)), 4) if rates else 0.0,
            "strategies": sorted(set(self.strategies)),
        }

    def to_dict(self) -> dict:
        return sanitize_for_json({
            "fills": self.fills,
            "strategies": self.strategies,
            "indicator_cols": self.indicator_cols,
            "n_original": self.n_original,
            "feature_names": self.feature_names,
            "add_indicators": self.add_indicators,
            "fitted_on": self.fitted_on,
            "missing_rate": self.missing_rate,
        })

    @classmethod
    def from_dict(cls, d: dict) -> "MissingDataSpec":
        return cls(
            fills=[float(x) for x in d.get("fills", [])],
            strategies=list(d.get("strategies", [])),
            indicator_cols=[int(x) for x in d.get("indicator_cols", [])],
            n_original=int(d.get("n_original", 0)),
            feature_names=list(d.get("feature_names", [])),
            add_indicators=bool(d.get("add_indicators", True)),
            fitted_on=str(d.get("fitted_on", "train_fold")),
            missing_rate=[float(x) for x in d.get("missing_rate", [])],
        )


def fit_missing_data(
    X_train: np.ndarray,
    *,
    add_indicators: bool = True,
    feature_names: list[str] | None = None,
) -> MissingDataSpec:
    """Fit fills (+ indicator schema) on training data only."""
    X = np.asarray(X_train, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got shape {X.shape}")
    n, f = X.shape
    fills: list[float] = []
    strategies: list[str] = []
    rates: list[float] = []
    indicator_cols: list[int] = []
    for j in range(f):
        col = X[:, j]
        fill, strat = _fill_value(col)
        fills.append(fill)
        strategies.append(strat)
        rate = float(np.mean(~np.isfinite(col))) if n else 0.0
        rates.append(rate)
        if add_indicators and rate > 0.0:
            indicator_cols.append(j)
    names = feature_names or [f"f{i}" for i in range(f)]
    return MissingDataSpec(
        fills=fills,
        strategies=strategies,
        indicator_cols=indicator_cols,
        n_original=f,
        feature_names=names,
        add_indicators=bool(add_indicators),
        fitted_on="train_fold",
        missing_rate=rates,
    )
