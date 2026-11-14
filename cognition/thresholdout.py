"""Thresholdout — seeded-noise gate for reusable holdout comparisons.

Research step 6 (Dwork et al., Science 2015, *The reusable holdout*).
Every adaptive decision against the same held-out / champion number
leaks information. Thresholdout answers "did A beat B?" through a
noise-adding threshold so repeated queries do not silently overfit the
instrument.

UIL constraints preserved:
- Seeded noise → byte-reproducible given (seed, query_id)
- No opaque model component — only the *comparison instrument* changes
- Logged decision is inspectable (delta, noise, tau, verdict)

Default mechanism for the champion bar (cfg.use_thresholdout). Standing
Gate 2 harnesses can call `beats()` the same way.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# Defaults chosen to be small vs typical held-out std (~0.01–0.04) so a
# real +0.02 win still clears, while a null A/B (delta≈0) is rejected.
DEFAULT_TAU = 0.01
DEFAULT_NOISE_SCALE = 0.005


def beats(
    challenger: float,
    incumbent: float,
    *,
    seed: int,
    query_id: int | str = 0,
    tau: float = DEFAULT_TAU,
    noise_scale: float = DEFAULT_NOISE_SCALE,
) -> dict[str, Any]:
    """Return a structured Thresholdout verdict.

    Verdict is True iff (challenger - incumbent + Lap(noise)) > tau.
    Deterministic for fixed (seed, query_id, inputs, tau, noise_scale).
    """
    c = float(challenger)
    i = float(incumbent)
    raw_delta = c - i
    q = _query_int(query_id)
    # Mix seed with query id so consecutive queries on the same holdout
    # draw independent (but reproducible) noise.
    rng = np.random.default_rng((int(seed) ^ (q * 0x9E3779B9)) & 0xFFFFFFFF)
    noise = float(rng.laplace(0.0, float(noise_scale))) if noise_scale > 0 else 0.0
    noisy_delta = raw_delta + noise
    verdict = bool(noisy_delta > float(tau))
    return {
        "beats": verdict,
        "challenger": c,
        "incumbent": i,
        "raw_delta": raw_delta,
        "noise": noise,
        "noisy_delta": noisy_delta,
        "tau": float(tau),
        "noise_scale": float(noise_scale),
        "query_id": q,
        "seed": int(seed),
    }


def beats_bool(
    challenger: float,
    incumbent: float,
    *,
    seed: int,
    query_id: int | str = 0,
    tau: float = DEFAULT_TAU,
    noise_scale: float = DEFAULT_NOISE_SCALE,
) -> bool:
    """Convenience wrapper returning only the boolean verdict."""
    return bool(beats(
        challenger, incumbent,
        seed=seed, query_id=query_id, tau=tau, noise_scale=noise_scale,
    )["beats"])


def _query_int(query_id: int | str) -> int:
    if isinstance(query_id, int):
        return query_id
    # Stable hash for string query ids (dataset names, gate labels).
    h = 0
    for ch in str(query_id):
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return int(h)
