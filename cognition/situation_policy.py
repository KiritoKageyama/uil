"""Situation-adaptive cfg recommendations from dataset fingerprint.

System Refinement (HARDCODED_VALUE_AUDIT_QA): do NOT dynamize Category-A
safety rails. Category-D / evidence-backed opt-ins may adapt when the
operator enables `cfg.situation_adaptive` (or when Gate 2 has cleared a
flag for default-on).

Every recommendation is logged with a reason so HITL / lab reports stay
transparent. Recommendations never invent new numeric thresholds — they
only flip already-implemented, documented cfg knobs.
"""

from __future__ import annotations

from typing import Any


# Knobs this policy is allowed to touch. Failed Gate-2 research flags
# stay OFF the allow-list until a future gate clears them.
_ALLOWED = frozenset({
    "stress_adaptive_blend",       # Category D #2 — regime-conditional weights
    "maturity_soft_normalization", # Category D #1 — non-saturating counts
    "explore_architecture",        # accuracy-neutral evidence probe (classic)
})


def recommend(
    fingerprint: dict | None,
    cfg: Any,
    *,
    search_mode: str = "classic",
) -> dict[str, Any]:
    """Return {flag: value, ...} plus `_reasons` for observability.

    Only runs when `cfg.situation_adaptive` is truthy (default **ON** after
    Category D Gate 2, 2026-07-24). Explicit cfg values that are already
    True are left alone (operator wins). Explicit False on a Category-D
    flag is also left alone when listed in `cfg._situation_lock`.
    """
    out: dict[str, Any] = {"_reasons": []}
    if not bool(getattr(cfg, "situation_adaptive", True)):
        out["_reasons"].append("situation_adaptive=OFF — no auto knobs")
        return out

    fp = fingerprint or {}
    reasons: list[str] = out["_reasons"]

    # Category D — stress blend: dead components (deficit=0 / explanation=1)
    # are common; adaptive renormalization is the documented fix.
    if _may_enable(cfg, "stress_adaptive_blend"):
        out["stress_adaptive_blend"] = True
        reasons.append(
            "stress_adaptive_blend: regime-conditional stress weights "
            "(Category D #2; skips dead deficit/explanation terms)"
        )

    # Category D — maturity soft counts: production stores saturate legacy
    # min(1, count/cap) at 1.0 across all terms.
    if _may_enable(cfg, "maturity_soft_normalization"):
        out["maturity_soft_normalization"] = True
        reasons.append(
            "maturity_soft_normalization: asymptotic count score "
            "(Category D #1; restores signal after store saturation)"
        )

    # Evidence probe only on classic path (race already varies architecture).
    # Accuracy-neutral by construction (non-competing probe).
    if (
        search_mode == "classic"
        and _may_enable(cfg, "explore_architecture")
    ):
        # Prefer when fingerprint suggests capacity questions (interaction
        # gap present) OR always under situation_adaptive — the probe does
        # not compete for the stage winner.
        gap = fp.get("interaction_gap")
        out["explore_architecture"] = True
        if gap is not None:
            reasons.append(
                f"explore_architecture: classic probe "
                f"(interaction_gap={float(gap):.3f}; journals config variation)"
            )
        else:
            reasons.append(
                "explore_architecture: classic probe "
                "(journals config variation for future mining)"
            )

    return out


def apply_to_cfg(cfg: Any, recommendations: dict[str, Any]) -> list[str]:
    """Mutate cfg in place with allowed recommendations. Returns applied keys."""
    applied: list[str] = []
    for key, val in recommendations.items():
        if key.startswith("_"):
            continue
        if key not in _ALLOWED:
            continue
        setattr(cfg, key, val)
        applied.append(key)
    return applied


def _may_enable(cfg: Any, name: str) -> bool:
    """Enable only if the operator has not already set the flag True/False
    as an explicit sticky choice. `None` / missing → may enable.
    If already True, skip (idempotent). If already False, respect OFF.
    """
    if not hasattr(cfg, name):
        return True
    cur = getattr(cfg, name)
    if cur is True:
        return False  # already on
    if cur is False:
        # Distinguish "default False" from "operator forced OFF":
        # situation_adaptive treats default False as eligible to turn ON.
        # Operator forced OFF is recorded via cfg._situation_lock frozenset.
        locked = getattr(cfg, "_situation_lock", None) or ()
        if name in locked:
            return False
        return True
    return True
