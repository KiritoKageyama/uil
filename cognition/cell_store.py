"""Composite VersionStore snapshot + materialized-view sync (v22 Tier 1)."""

from __future__ import annotations

from typing import Any

from figpso_uil.cognition.knowledge_cell import cell_to_readable


def split_principles(principles: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    """Separate principle vs routing_rule cells for composite replay."""
    pr: dict[str, dict] = {}
    rr: dict[str, dict] = {}
    for p in principles:
        pid = p.get("principle_id", "?")
        if p.get("cell_type") == "routing_rule":
            rr[pid] = dict(p)
        else:
            pr[pid] = dict(p)
    return pr, rr


def build_composite_snapshot(optimizer: Any) -> dict[str, dict[str, dict]]:
    """Full materialized state for VersionStore snapshots / replay."""
    from figpso_uil.cognition.goal_cell import GOAL_CELL_TYPE, GoalStore

    net = optimizer.network.canonical_state()
    principles = optimizer.library._load()
    pr, rr = split_principles(principles)
    episodes: dict[str, dict] = {}
    ep_list = getattr(optimizer.episodic_attention, "_episodes", [])
    for i, ep in enumerate(ep_list[-50:]):
        episodes[f"EP-{i:04d}"] = dict(ep)
    return {
        "node": dict(net.get("node", {})),
        "edge": dict(net.get("edge", {})),
        "principle": pr,
        "routing_rule": rr,
        "formula": optimizer.formula_library.canonical_state(),
        "causal_hypothesis": optimizer.causal_engine.canonical_state(),
        "episode": episodes,
        GOAL_CELL_TYPE: GoalStore().canonical_state(),
    }


def apply_materialized_edit(
    cell_type: str,
    cell_id: str,
    fields: dict,
    *,
    library_path: str | None = None,
    formula_library_path: str | None = None,
    network_path: str | None = None,
    causal_path: str | None = None,
) -> bool:
    """Sync a human_edit event into live JSON materialized views.

    Optional ``*_path`` kwargs scope the write for isolated ``state_dir``
    runs / tests (bug #30b family). Defaults keep the shared ``results/``
    paths the webui uses.
    """
    from figpso_uil.cognition.uil_library import UIL_Library
    from figpso_uil.cognition.formula_library import FormulaLibrary
    from figpso_uil.cognition.causal_intervention import CausalInterventionEngine
    from figpso_uil.cognition.editable_network import EditableNetwork
    from figpso_uil.cognition.paths import results_path

    if cell_type in ("principle", "routing_rule"):
        lib = UIL_Library(
            library_path=library_path or results_path("uil_principles_v19.json"),
        )
        principles = lib._load()
        for p in principles:
            if p.get("principle_id") == cell_id:
                p.update(fields)
                lib._save(principles)
                return True
        return False

    if cell_type == "formula":
        kwargs = {}
        if formula_library_path:
            kwargs["filepath"] = formula_library_path
        flib = FormulaLibrary(**kwargs)
        if cell_id in flib._formulas:
            flib._formulas[cell_id].update(fields)
            flib._save()
            return True
        return False

    if cell_type == "causal_hypothesis":
        eng = (
            CausalInterventionEngine(filepath=causal_path)
            if causal_path
            else CausalInterventionEngine()
        )
        h = eng._hypotheses.get(cell_id)
        if h is None:
            return False
        for k, v in fields.items():
            if hasattr(h, k):
                setattr(h, k, v)
        eng._save()
        return True

    if cell_type == "node":
        net = (
            EditableNetwork(filepath=network_path)
            if network_path
            else EditableNetwork()
        )
        return bool(net.edit_node(cell_id, **fields))

    if cell_type == "edge":
        net = (
            EditableNetwork(filepath=network_path)
            if network_path
            else EditableNetwork()
        )
        src, rest = cell_id.split("->", 1) if "->" in cell_id else ("", "")
        if ":" in rest:
            tgt, rel = rest.rsplit(":", 1)
        else:
            tgt, rel = rest, "leads_to"
        if hasattr(net, "edit_edge"):
            return bool(net.edit_edge(src, tgt, rel, **fields))
        return False

    if cell_type == "goal":
        from figpso_uil.cognition.goal_cell import GoalStore
        return GoalStore().apply_edit(cell_id, fields)

    return False


def cells_for_api(state: dict, limit: int = 80) -> list[dict]:
    """Flatten composite state into HITL-reviewable rows."""
    rows: list[dict] = []
    for ctype, bucket in state.items():
        if not isinstance(bucket, dict):
            continue
        for cid, post in bucket.items():
            if not isinstance(post, dict):
                continue
            rows.append({
                "cell_id": cid,
                "cell_type": ctype,
                "confidence": (
                    post.get("confidence")
                    or post.get("causal_strength")
                    or post.get("weight")
                ),
                "readable": cell_to_readable(ctype, post, cid),
                "post_state": post,
            })
    return rows[:limit]
