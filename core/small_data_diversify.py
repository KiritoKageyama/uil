"""Bootstrap + jitter for tiny datasets — feeds seeds, particles, and tournaments."""

from __future__ import annotations

import numpy as np

MIN_EFFECTIVE_ROWS = 15


def ensure_min_rows(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    min_rows: int = MIN_EFFECTIVE_ROWS,
    noise_frac: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Bootstrap with light feature jitter until at least min_rows."""
    n = len(y)
    if n >= min_rows:
        return X, y, False

    idx = rng.integers(0, n, size=min_rows)
    X_out = X[idx].astype(float, copy=True)
    y_out = y[idx].copy()
    scale = np.std(X_out, axis=0, keepdims=True)
    scale = np.where(scale > 1e-9, scale, 1.0)
    X_out += rng.normal(0.0, noise_frac * scale, X_out.shape)
    return X_out, y_out, True


def bootstrap_row_batches(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_batches: int,
    batch_size: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create diversified mini-batches for empty seeds / particle seeds."""
    n = len(y)
    if n == 0:
        return []
    batch_size = batch_size or max(MIN_EFFECTIVE_ROWS, min(n * 2, 32))
    batches: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_batches):
        idx = rng.integers(0, n, size=batch_size)
        Xb = X[idx].astype(float, copy=True)
        yb = y[idx].copy()
        scale = np.std(Xb, axis=0, keepdims=True)
        scale = np.where(scale > 1e-9, scale, 1.0)
        Xb += rng.normal(0.0, 0.008 * scale, Xb.shape)
        batches.append((Xb, yb))
    return batches


def guidance_from_batch(
    X_batch: np.ndarray,
    y_batch: np.ndarray,
    n_features: int,
) -> np.ndarray:
    """Feature emphasis vector from a bootstrapped batch (class separation proxy)."""
    if X_batch.shape[1] != n_features:
        return np.ones(n_features) / n_features
    classes = np.unique(y_batch)
    if len(classes) < 2:
        return np.ones(n_features) / n_features
    means = []
    for c in classes:
        means.append(X_batch[y_batch == c].mean(axis=0))
    diff = np.abs(means[0] - means[1])
    if diff.sum() < 1e-9:
        return np.ones(n_features) / n_features
    w = diff / (diff.sum() + 1e-12)
    return np.clip(w, 0.0, 1.0)


def bootstrap_guidance_weights(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_seeds: int,
) -> list[np.ndarray]:
    """One weight vector per bootstrap batch for search diversification."""
    n_features = X.shape[1]
    batches = bootstrap_row_batches(X, y, rng, n_batches=n_seeds)
    return [guidance_from_batch(xb, yb, n_features) for xb, yb in batches]
