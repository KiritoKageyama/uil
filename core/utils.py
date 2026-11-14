"""Shared utility functions for the UIL framework."""

from __future__ import annotations
import os
from contextlib import contextmanager
import numpy as np

_BLAS_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def blas_threads_for_slots(total_cpus: int, parallel_slots: int) -> int:
    """Divide BLAS/OpenMP threads so nested pools don't oversubscribe."""
    return max(1, int(total_cpus) // max(1, int(parallel_slots)))


@contextmanager
def blas_thread_scope(threads: int):
    """Temporarily pin BLAS thread count for one parallel worker."""
    value = str(max(1, int(threads)))
    saved = {key: os.environ.get(key) for key in _BLAS_ENV_KEYS}
    try:
        for key in _BLAS_ENV_KEYS:
            os.environ[key] = value
        yield
    finally:
        for key, prior in saved.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


def safe_minmax(v: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range. Returns zeros if range is negligible."""
    v = np.asarray(v, dtype=float)
    lo = float(np.min(v))
    hi = float(np.max(v))
    if hi - lo < 1e-12:
        return np.zeros_like(v, dtype=float)
    return (v - lo) / (hi - lo)


def scale_by_max(v: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] by the MAXIMUM only -- never manufactures a zero.

    safe_minmax stretches the range so the smallest element becomes exactly
    0.0 on every call. Applied repeatedly to a running guidance vector (as
    the search strategies do each stage) that is a ratchet: something is
    always pushed to the floor, and a hard `< 0.05 -> 0` clip then removes
    it permanently. Dividing by the max preserves the same ordering and
    scale without deleting the weakest element.
    """
    v = np.asarray(v, dtype=float)
    v = np.clip(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    hi = float(np.max(v)) if v.size else 0.0
    if hi < 1e-12:
        return np.zeros_like(v, dtype=float)
    return v / hi


def apply_guidance_feedback(
    base_guidance: np.ndarray,
    importance: np.ndarray,
    entropy: float,
    *,
    fix_guidance_ratchet: bool,
    guidance_floor: float,
) -> np.ndarray:
    """Blend classifier-importance feedback into a running guidance vector.

    UIL hardcoded-value audit, Category C (2026-07-23): this formula was
    duplicated verbatim in PSOSearch.update_feedback and
    PrincipleGuidedSearch.update_feedback, each carrying its own copy of the
    comment "Fix BOTH copies or you fix only the strategy you are not
    running" -- a warning about itself that was never acted on. Consolidated
    here so there is exactly one place to change the formula, and exactly
    one place a future fix can land instead of two.

    `effective_lr` sets how much this stage's classifier-importance signal
    moves the running guidance vector; it decays as entropy drops (a more
    decisive signal earns a bigger step). Two blend modes:

    - default (fix_guidance_ratchet=False): min-max renormalize after the
      blend, then hard-zero anything under 0.05. This is a RATCHET -- see
      scale_by_max's docstring -- a feature under-credited for 2-3 stages
      in a row is deleted from guidance permanently, and guidance is the
      SOLE input to candidate generation, so it can never be proposed
      again. This is the historical/default behavior, kept for
      backward compatibility.
    - opt-in fix (fix_guidance_ratchet=True): scale by max instead of
      min-max (never manufactures a zero), floor weak features at
      `guidance_floor` instead of deleting them (recoverable), and halve
      the update rate (0.7/stage was rewriting the whole vector every
      ~2 stages). Default OFF pending a cross-seed accuracy sweep, per
      the project's "don't default an unvalidated fix on" convention (E6).
    """
    effective_lr = 0.1 + 0.6 * (1.0 - entropy)
    if fix_guidance_ratchet:
        effective_lr *= 0.5
        updated = (1.0 - effective_lr) * base_guidance + effective_lr * importance
        return np.maximum(scale_by_max(updated), guidance_floor)
    updated = (1.0 - effective_lr) * base_guidance + effective_lr * importance
    updated[updated < 0.05] = 0.0
    return safe_minmax(updated)


def coerce_float_array(value: object | None) -> np.ndarray | None:
    """Restore float vectors from JSON lists or legacy numpy string dumps."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    if isinstance(value, str):
        cleaned = value.strip().replace("\n", " ")
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned[1:-1]
        parts = [p for p in cleaned.replace(",", " ").split() if p]
        if not parts:
            return None
        return np.asarray([float(p) for p in parts], dtype=float)
    return np.asarray(value, dtype=float)


def sanitize_for_json(obj: object) -> object:
    """Make numpy-heavy payloads safe for json.dump (champion checkpoints)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj
