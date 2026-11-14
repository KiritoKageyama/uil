"""Master UIL on/off + optional core wiring / connect pack.

``uil_enabled=False``
    Plain learner + search only. Every cognitive ablation key is forced
    off and all wiring flags are forced off. Honest baseline for A/B.

``uil_enabled=True`` (default)
    Cognition subsystems run. Core wiring flags are left as the caller
    set them (normally False) — Gate 2 (2026-07-25) showed forcing
    translator + connect_cognition + utility_gated ON by default
    regresses accuracy (mean Δ −0.0132 on breast/wine/ionosphere).

``uil_connect_pack=True`` (requires ``uil_enabled``)
    Forces the full wiring bundle for try / existence / research:
      - signal_translator, connect_cognition, utility_gated_formulas
      - explore_architecture, adaptive_min_evidence
    CONNECTED ≠ helpful; pack stays opt-in until a Gate 2 pass.

``learning_mode`` is never unlocked by these policies (bug #21).
"""

from __future__ import annotations

from typing import Any

# Forced ON by uil_connect_pack; NOT forced on default (Gate 2 failed).
CORE_WIRING_FLAGS: tuple[str, ...] = (
    "connect_cognition",
    "utility_gated_formulas",
    "signal_translator",
)

CONNECT_PACK_EXTRA: tuple[str, ...] = (
    "explore_architecture",
    "adaptive_min_evidence",
)

CONNECT_PACK_FLAGS: tuple[str, ...] = CORE_WIRING_FLAGS + CONNECT_PACK_EXTRA

ABLATION_KEYS: tuple[str, ...] = (
    "principles",
    "network",
    "formulas",
    "causal",
    "episodic",
    "structural",
    "meta_principles",
    "safeguards",
    "goal",
)


def apply_uil_mode(cfg: Any, ablation: dict[str, bool]) -> tuple[dict[str, bool], dict]:
    """Apply master / connect-pack policy. Mutates ``cfg``.

    Precedence: ``uil_enabled=False`` wins over everything.
    """
    enabled = bool(getattr(cfg, "uil_enabled", True))
    connect = bool(getattr(cfg, "uil_connect_pack", False))
    out = dict(ablation)
    meta: dict[str, Any] = {
        "uil_enabled": enabled,
        "uil_connect_pack": connect and enabled,
    }

    if not enabled:
        out = {k: False for k in ABLATION_KEYS}
        for flag in CONNECT_PACK_FLAGS:
            setattr(cfg, flag, False)
        meta["mode"] = "uil_off"
        meta["ablation_all_off"] = True
        meta["core_wiring"] = {f: False for f in CORE_WIRING_FLAGS}
        meta["connect_flags"] = {f: False for f in CONNECT_PACK_FLAGS}
        return out, meta

    if connect:
        for flag in CONNECT_PACK_FLAGS:
            setattr(cfg, flag, True)
        meta["mode"] = "uil_connect_pack"
        meta["note"] = (
            "Full wiring ON (opt-in). Gate 2 2026-07-25: default forcing "
            "core wiring regressed mean acc -0.0132 — pack stays try-it."
        )
        meta["core_wiring"] = {f: True for f in CORE_WIRING_FLAGS}
        meta["connect_flags"] = {f: True for f in CONNECT_PACK_FLAGS}
        return out, meta

    meta["mode"] = "uil_default"
    meta["note"] = (
        "Cognition runs; core wiring flags left as set (default OFF). "
        "Use uil_connect_pack or set signal_translator/connect_cognition "
        "explicitly to arm the translator."
    )
    meta["core_wiring"] = {
        f: bool(getattr(cfg, f, False)) for f in CORE_WIRING_FLAGS
    }
    meta["connect_flags"] = {
        f: bool(getattr(cfg, f, False)) for f in CONNECT_PACK_FLAGS
    }
    return out, meta
