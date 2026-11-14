"""Router threshold tuning on normalized fingerprints (v22 M2).

Re-estimates FORMULA_R2_MATH and LINEAR_ACC_STRUCTURED from a corpus of
labeled fingerprints so routing decisions stay calibrated after normalization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from figpso_uil.cognition import domain_router as dr


@dataclass
class RouterThresholds:
    formula_r2_math: float
    linear_acc_structured: float
    tiny_n: int


def tune_thresholds(
    labeled_fps: list[tuple[dict, str]],
    *,
    margin: float = 0.02,
) -> RouterThresholds:
    """Fit thresholds from (fingerprint, domain_name) examples.

    Uses conservative percentiles: math threshold at p25 of D2 formula_probe_r2,
    structured at p25 of D4 linear_probe_acc — avoids over-routing to specialists.
    """
    math_r2 = [
        fp.get("formula_probe_r2")
        for fp, dom in labeled_fps
        if dom == "D2_math" and fp.get("formula_probe_r2") is not None
    ]
    lin_acc = [
        fp.get("linear_probe_acc")
        for fp, dom in labeled_fps
        if dom == "D4_structured" and fp.get("linear_probe_acc") is not None
    ]
    formula_thr = float(np.percentile(math_r2, 25) - margin) if math_r2 else dr.FORMULA_R2_MATH
    linear_thr = float(np.percentile(lin_acc, 25) - margin) if lin_acc else dr.LINEAR_ACC_STRUCTURED
    return RouterThresholds(
        formula_r2_math=max(0.5, min(0.98, formula_thr)),
        linear_acc_structured=max(0.7, min(0.99, linear_thr)),
        tiny_n=dr.TINY_N,
    )


def apply_thresholds(thresholds: RouterThresholds) -> None:
    """Patch module-level router constants (in-process)."""
    dr.FORMULA_R2_MATH = thresholds.formula_r2_math
    dr.LINEAR_ACC_STRUCTURED = thresholds.linear_acc_structured
    dr.TINY_N = thresholds.tiny_n
    dr.DEFAULT_REGISTRY = dr._default_registry()
