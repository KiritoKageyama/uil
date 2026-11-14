"""Invertible feature normalization (v22 §2.2).

Fitted on training folds only. Every transform is stored with provenance
so symbolic laws can be back-transformed to original units.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from figpso_uil.core.utils import sanitize_for_json


@dataclass
class FeatureTransform:
    method: str  # zscore | minmax | robust
    mean: float = 0.0
    std: float = 1.0
    median: float = 0.0
    mad: float = 1.0
    vmin: float = 0.0
    vmax: float = 1.0

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.method == "zscore":
            return (x - self.mean) / (self.std or 1.0)
        if self.method == "robust":
            return (x - self.median) / (self.mad or 1.0)
        if self.method == "minmax":
            span = (self.vmax - self.vmin) or 1.0
            return (x - self.vmin) / span
        return x

    def inverse(self, x_scaled: np.ndarray) -> np.ndarray:
        x = np.asarray(x_scaled, dtype=float)
        if self.method == "zscore":
            return x * (self.std or 1.0) + self.mean
        if self.method == "robust":
            return x * (self.mad or 1.0) + self.median
        if self.method == "minmax":
            span = (self.vmax - self.vmin) or 1.0
            return x * span + self.vmin
        return x

    def to_dict(self) -> dict:
        return sanitize_for_json({
            "method": self.method,
            "mean": self.mean, "std": self.std,
            "median": self.median, "mad": self.mad,
            "vmin": self.vmin, "vmax": self.vmax,
        })

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureTransform":
        return cls(**{k: d.get(k, getattr(cls, k, 0)) for k in
                      ("method", "mean", "std", "median", "mad", "vmin", "vmax")})


@dataclass
class NormalizationSpec:
    """Per-feature transforms + provenance."""
    transforms: list[FeatureTransform] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    fingerprint_id: str | None = None
    fitted_on: str = "train_fold"

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = X.copy()
        for j, t in enumerate(self.transforms):
            if j < out.shape[1]:
                out[:, j] = t.forward(out[:, j])
        return out

    def inverse(self, X_scaled: np.ndarray) -> np.ndarray:
        X = np.asarray(X_scaled, dtype=float)
        out = X.copy()
        for j, t in enumerate(self.transforms):
            if j < out.shape[1]:
                out[:, j] = t.inverse(out[:, j])
        return out

    def to_dict(self) -> dict:
        return sanitize_for_json({
            "transforms": [t.to_dict() for t in self.transforms],
            "feature_names": self.feature_names,
            "fingerprint_id": self.fingerprint_id,
            "fitted_on": self.fitted_on,
        })

    @classmethod
    def from_dict(cls, d: dict) -> "NormalizationSpec":
        return cls(
            transforms=[FeatureTransform.from_dict(t) for t in d.get("transforms", [])],
            feature_names=list(d.get("feature_names", [])),
            fingerprint_id=d.get("fingerprint_id"),
            fitted_on=d.get("fitted_on", "train_fold"),
        )


def _pick_method(col: np.ndarray, shape_stats: dict | None) -> str:
    """Choose scaler from axis-3 shape hints."""
    if shape_stats:
        sk = abs(shape_stats.get("skew") or 0.0)
        ku = abs(shape_stats.get("kurtosis") or 0.0)
        if sk > 1.5 or ku > 3.0:
            return "robust"
    col_min, col_max = float(np.min(col)), float(np.max(col))
    if col_min >= 0.0 and col_max <= 1.0:
        return "minmax"
    return "zscore"


def fit_normalization(
    X_train: np.ndarray,
    *,
    fingerprint: dict | None = None,
    feature_names: list[str] | None = None,
) -> NormalizationSpec:
    """Fit transforms on training data only."""
    X = np.asarray(X_train, dtype=float)
    n_features = X.shape[1]
    shape_axis = (fingerprint or {}).get("axes", {}).get("shape", {})
    transforms: list[FeatureTransform] = []
    for j in range(n_features):
        col = X[:, j]
        shape_stats = {
            "skew": shape_axis.get(f"skew_f{j}"),
            "kurtosis": shape_axis.get(f"kurtosis_f{j}"),
        }
        method = _pick_method(col, shape_stats)
        if method == "robust":
            med = float(np.median(col))
            mad = float(np.median(np.abs(col - med))) or 1.0
            transforms.append(FeatureTransform(method="robust", median=med, mad=mad))
        elif method == "minmax":
            transforms.append(FeatureTransform(
                method="minmax",
                vmin=float(np.min(col)),
                vmax=float(np.max(col)),
            ))
        else:
            transforms.append(FeatureTransform(
                method="zscore",
                mean=float(np.mean(col)),
                std=float(np.std(col)) or 1.0,
            ))
    names = feature_names or [f"f{i}" for i in range(n_features)]
    return NormalizationSpec(
        transforms=transforms,
        feature_names=names,
        fingerprint_id=(fingerprint or {}).get("fingerprint_id"),
        fitted_on="train_fold",
    )


def back_transform_formula_readable(
    expr: str,
    spec: NormalizationSpec,
    feature_index: int = 0,
) -> str:
    """Annotate readable formula with inverse-transform hint for one feature."""
    if feature_index >= len(spec.transforms):
        return expr
    t = spec.transforms[feature_index]
    return f"{expr}  [feature {feature_index}: {t.method} inverse available]"
