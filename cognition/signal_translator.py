"""Signal translator — cognition → learner/search directives.

Turns every live cognitive channel into one ``LearnerDirective`` that the
optimizer applies to particles, seeds, and the SymbolicFormula basis.

Pipeline order (matches the intended agent roles):
  dataset correlation → episodic (middle agent) → network (experience)
  → principles (steer particles / seeds / formulas) → formulas (basis terms)

Always ON when ``uil_enabled`` (see ``core/uil_mode.py`` CORE_WIRING_FLAGS).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LearnerDirective:
    """What the transparent learner / search must honor this stage."""

    feature_guidance: np.ndarray | None = None
    config: dict[str, Any] = field(default_factory=dict)
    prefer_formula_ids: list[str] = field(default_factory=list)
    use_symbolic_basis: bool | None = None
    sources: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        g = self.feature_guidance
        return {
            "sources": list(self.sources),
            "n_prefer_formulas": len(self.prefer_formula_ids),
            "config_keys": sorted(self.config.keys()),
            "use_symbolic_basis": self.use_symbolic_basis,
            "guidance_std": (
                float(np.std(g)) if g is not None and g.size else None
            ),
            "guidance_max": (
                float(np.max(g)) if g is not None and g.size else None
            ),
        }


def _safe_corr_guidance(X: np.ndarray, y: np.ndarray, n_features: int) -> np.ndarray:
    """Dataset instant info: |corr(x_j, y)| as a guidance prior."""
    g = np.ones(n_features, dtype=float)
    if X is None or y is None or X.shape[1] != n_features or len(y) < 8:
        return g
    yv = np.asarray(y, dtype=float)
    if float(np.std(yv)) < 1e-12:
        return g
    for j in range(n_features):
        col = X[:, j]
        if float(np.std(col)) < 1e-12:
            g[j] = 0.05
            continue
        c = float(np.corrcoef(col, yv)[0, 1])
        g[j] = abs(c) if np.isfinite(c) else 0.05
    mx = float(np.max(g))
    if mx > 1e-12:
        g = g / mx
    return np.clip(g, 0.05, 1.0)


def _concentrate(guidance: np.ndarray, strength: float) -> np.ndarray:
    """Higher strength → sharper top features (maps 'use fewer features')."""
    s = float(np.clip(strength, 0.0, 1.0))
    g = np.asarray(guidance, dtype=float)
    if g.size == 0:
        return g
    powered = np.power(np.clip(g, 1e-6, 1.0), 1.0 + 2.0 * s)
    mx = float(np.max(powered))
    if mx > 1e-12:
        powered = powered / mx
    return np.clip(powered, 0.02, 1.0)


def _prefer_seed_formulas(formula_library: Any, out: list[str], limit: int = 8) -> None:
    """Principles steer seeds: prefer SEED-* formulas into the basis."""
    if formula_library is None:
        return
    try:
        for f in formula_library.get_active_formulas():
            fid = f.get("formula_id") or ""
            if fid.startswith("SEED-") or str(f.get("origin", "")).lower() == "seed":
                out.append(fid)
                if len([x for x in out if str(x).startswith("SEED-")]) >= limit:
                    break
    except Exception:
        pass


def translate(
    *,
    n_features: int,
    ablation: dict[str, bool],
    overrides: dict | None = None,
    triggered_principle_ids: list[str] | None = None,
    principles: list[dict] | None = None,
    network_result: dict | None = None,
    episodic_hint: dict | None = None,
    formula_library: Any | None = None,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
) -> LearnerDirective:
    """Compile a stage directive from every live cognitive channel."""
    overrides = overrides or {}
    triggered = list(triggered_principle_ids or [])
    principles = principles or []
    network_result = network_result or {}
    d = LearnerDirective()

    # 1) Dataset — instant characteristic prior
    guidance = _safe_corr_guidance(X_train, y_train, n_features)
    d.sources.append("dataset_correlation")
    used_guidance = False

    # 2) Episodic — agent between dataset and network (process, then hand off)
    if ablation.get("episodic", True) and episodic_hint:
        bw = episodic_hint.get("best_weights")
        if bw is not None:
            w = np.asarray(bw, dtype=float).reshape(-1)
            if w.size == n_features:
                guidance = np.clip(0.45 * guidance + 0.55 * w, 0.02, 1.0)
                used_guidance = True
                d.sources.append("episodic")
        sug = episodic_hint.get("suggested_symbolic_formula")
        if isinstance(sug, str) and sug:
            d.prefer_formula_ids.append(sug)

    # 3) Network — collect / apply experience (after episodic has shaped prior)
    if ablation.get("network", True):
        emph = network_result.get("feature_emphasis")
        if emph is not None:
            e = np.asarray(emph, dtype=float).reshape(-1)
            if e.size == n_features and float(np.max(np.abs(e))) > 1e-12:
                e = e / (float(np.max(e)) + 1e-12)
                guidance = np.clip(0.5 * guidance + 0.5 * e, 0.02, 1.0)
                used_guidance = True
                d.sources.append("network")
        for nid in network_result.get("activated_nodes") or []:
            if isinstance(nid, str) and (
                nid.startswith("F-") or nid.startswith("SEED-")
            ):
                d.prefer_formula_ids.append(nid)

    # 4) Principles — steer particles, seeds, and formula prefs
    if ablation.get("principles", True):
        strength = 0.75
        if "max_penalty" in overrides and isinstance(overrides["max_penalty"], (int, float)):
            strength = float(np.clip(float(overrides["max_penalty"]) / 2.5, 0.55, 0.98))
        elif "min_penalty" in overrides and isinstance(overrides["min_penalty"], (int, float)):
            strength = float(np.clip(1.0 - float(overrides["min_penalty"]) / 5.0, 0.4, 0.9))
        if principles or triggered or overrides:
            guidance = _concentrate(guidance, strength)
            order = np.argsort(-guidance)
            keep = max(2, n_features // 3)
            hard = np.full(n_features, 0.04, dtype=float)
            hard[order[:keep]] = 1.0
            guidance = np.clip(guidance * hard, 0.02, 1.0)
            used_guidance = True
            d.sources.append("principles")
            d.config.setdefault("sparse_l1_c", 0.35)
            # Principles → seeds: keep seed formulas in the competing basis
            _prefer_seed_formulas(formula_library, d.prefer_formula_ids)
        prefer_set = set(triggered) if triggered else None
        for p in principles:
            pid = p.get("principle_id")
            if prefer_set is not None and pid not in prefer_set:
                continue
            ov = p.get("action_override") or {}
            for key in ("prefer_formula", "prefer_formula_id", "formula_id"):
                fid = ov.get(key)
                if isinstance(fid, str) and fid:
                    d.prefer_formula_ids.append(fid)
            prefs = ov.get("prefer_formulas")
            if isinstance(prefs, list):
                d.prefer_formula_ids.extend(str(x) for x in prefs if x)
            if "sparse_l1_c" in ov:
                d.config["sparse_l1_c"] = ov["sparse_l1_c"]

    # 5) Formulas — real symbolic terms into SymbolicFormula (or strip)
    if ablation.get("formulas", True):
        d.use_symbolic_basis = True
        if formula_library is not None:
            try:
                for f in formula_library.get_active_formulas()[:12]:
                    fid = f.get("formula_id")
                    if fid:
                        d.prefer_formula_ids.append(fid)
                if hasattr(formula_library, "get_incubating_formulas"):
                    for f in formula_library.get_incubating_formulas()[:6]:
                        fid = f.get("formula_id")
                        if fid:
                            d.prefer_formula_ids.append(fid)
                d.sources.append("formulas")
            except Exception:
                pass
    else:
        d.use_symbolic_basis = False
        d.prefer_formula_ids = []
        d.sources.append("formulas_ablated")

    for key, value in overrides.items():
        if key.startswith("config."):
            d.config[key.split(".", 1)[1]] = value
    for key in ("sparse_l1_c", "use_rule_basis", "use_stump_basis",
                "use_neural_basis", "mutation_rounds_per_fit", "tournament_k"):
        if key in overrides:
            d.config[key] = overrides[key]

    seen: set[str] = set()
    uniq: list[str] = []
    for fid in d.prefer_formula_ids:
        if fid not in seen:
            seen.add(fid)
            uniq.append(fid)
    d.prefer_formula_ids = uniq
    d.feature_guidance = guidance
    if not used_guidance:
        d.sources.append("guidance_dataset_only")
    return d
