"""Inspect/edit live cognition state.

All persisted modules here (principles, network, formulas, causal
hypotheses, episodes) are process-GLOBAL files, not per-dataset -- see
core/optimizer.py's constructor calls. So "live" means "is there any
currently active run, and if so read its in-memory instances" (which
may hold not-yet-saved-to-disk changes); otherwise fall back to a fresh
instance constructed against the default persisted file.

Edits never reinvent mutation logic -- they call each class's own
existing methods (EditableNetwork.edit_node(), etc.) or, where no public
mutator exists (UIL_Library, CausalInterventionEngine, EpisodicAttention
all follow varying load/save conventions), reuse the existing private
_load()/_save() pair directly rather than duplicating their logic.
"""

from __future__ import annotations
import os

from fastapi import APIRouter, HTTPException

from figpso_uil.cognition.uil_library import UIL_Library
from figpso_uil.cognition.editable_network import EditableNetwork
from figpso_uil.cognition.formula_library import FormulaLibrary
from figpso_uil.cognition.causal_intervention import CausalInterventionEngine
from figpso_uil.cognition.episodic_attention import EpisodicAttention

from figpso_uil.webui.backend.run_manager import manager
from figpso_uil.webui.backend.models import StateEditRequest, GoalRequest
from figpso_uil.cognition.version_store import VersionStore
from figpso_uil.cognition.knowledge_cell import cell_to_readable
from figpso_uil.cognition.cell_store import apply_materialized_edit, cells_for_api
from figpso_uil.webui.backend.persistence_stats import get_persistence_stats
from figpso_uil.webui.backend.memory_architecture import (
    get_memory_architecture,
    get_recent_changelog,
    get_consciousness_brief,
)
from figpso_uil.cognition.live_feed import read_since, clear_feed, is_enabled, read_tail
from figpso_uil.cognition.memory_reset import clear_all_memory, clear_and_bootstrap
from figpso_uil.cognition.paths import RESULTS_DIR

router = APIRouter(prefix="/state", tags=["state"])


def _active_optimizer(run_id: str | None):
    if run_id is None:
        return None
    record = manager.get(run_id)
    if record is None or record.status not in ("running", "paused"):
        return None
    return record, record.optimizer


def _json_safe(obj):
    """Replace NaN/Inf with None — episode fingerprints can carry NaN
    (e.g. levene_f0 on constant features) and FastAPI's strict JSON
    encoder refuses them, 500ing the whole state payload."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
        return None
    return obj


# ── GoalCell (LCA-inspired top-down director, 2026-07-18) ────────────

@router.get("/goal")
def get_goal():
    """Return the active GoalCell (or null when none is set)."""
    from figpso_uil.cognition.goal_cell import GoalStore
    active = GoalStore().get_active()
    return {"goal": active.to_dict() if active else None}


@router.post("/goal")
def set_goal(req: GoalRequest):
    """Install a GoalCell via the sanctioned dual write path
    (VersionStore event + materialized JSON — bug #16 invariant)."""
    from figpso_uil.cognition.goal_cell import GoalCell, record_goal_event
    goal = GoalCell(
        directive=req.directive,
        focus_features=list(req.focus_features),
        ignore_features=list(req.ignore_features),
        prefer_domains=list(req.prefer_domains),
        prefer_relations=list(req.prefer_relations),
        goal_keywords=list(req.goal_keywords),
        confidence=max(0.0, min(1.0, req.confidence)),
    )
    seq = record_goal_event(VersionStore.shared(), goal, cause="webui")
    return {"ok": True, "seq": seq, "goal": goal.to_dict()}


@router.post("/goal/clear")
def clear_goal():
    """Archive the active goal (runs return to pure bottom-up attention)."""
    from figpso_uil.cognition.goal_cell import GoalStore, GOAL_CELL_TYPE
    store = GoalStore()
    active = store.get_active()
    if active is None:
        return {"ok": True, "cleared": False}
    store.apply_edit(active.cell_id, {"status": "archived"})
    try:
        VersionStore.shared().record_cell(
            cell_id=active.cell_id,
            cell_type=GOAL_CELL_TYPE,
            op="human_reject",
            post_state={**active.to_dict(), "status": "archived"},
            cause="webui",
        )
    except Exception:
        pass
    return {"ok": True, "cleared": True}


# ── Principles ────────────────────────────────────────────────────────

@router.get("/replay")
def replay_state(seq: int | None = None):
    """Reconstruct VersionStore state at seq (undo preview)."""
    store = VersionStore.shared()
    target = seq if seq is not None else store.last_seq
    return _json_safe(store.reconstruct(target))


@router.get("/cells")
def list_cells(cell_type: str | None = None, limit: int = 80):
    """Reviewable knowledge cells from latest materialized state."""
    store = VersionStore.shared()
    state = store.reconstruct()["state"]
    # Filter BEFORE flattening/truncation — otherwise a rare cell type
    # (e.g. one goal among thousands of nodes) can never surface.
    if cell_type:
        state = {cell_type: state.get(cell_type, {})}
    cells = cells_for_api(state, limit=limit)
    return {"cells": _json_safe(cells), "last_seq": store.last_seq}


@router.patch("/cells/{cell_id}")
def edit_cell(cell_id: str, req: StateEditRequest, cell_type: str = "principle"):
    """Human edit on any cell type — records human_edit event."""
    fields = req.fields
    store = VersionStore.shared()
    state = store.reconstruct()["state"]
    bucket = state.get(cell_type, {})
    target = bucket.get(cell_id)
    if target is None:
        raise HTTPException(status_code=404, detail="cell not found")
    target = dict(target)
    target.update(fields)
    seq = store.record_cell(
        cell_id=cell_id,
        cell_type=cell_type,
        op="human_edit",
        post_state=target,
        cause="webui",
    )
    apply_materialized_edit(cell_type, cell_id, fields)
    return {"ok": True, "seq": seq}


@router.post("/cells/{cell_id}/reject")
def reject_cell(cell_id: str, cell_type: str = "principle"):
    """Human reject — records event; materialized view updated on next prune."""
    store = VersionStore.shared()
    seq = store.record_cell(
        cell_id=cell_id,
        cell_type=cell_type,
        op="human_reject",
        post_state=None,
        cause="webui",
    )
    apply_materialized_edit(cell_type, cell_id, {"status": "rejected"})
    return {"ok": True, "seq": seq}


@router.get("/principles")
def get_principles(run_id: str | None = None, dataset_name: str | None = None):
    live = _active_optimizer(run_id)
    if live is not None:
        record, optimizer = live
        with record.lock:
            return {"principles": optimizer.library.get_principles(dataset_name=dataset_name)}
    library = UIL_Library()
    return {"principles": library.get_principles(dataset_name=dataset_name)}


@router.patch("/principles")
def edit_principle(req: StateEditRequest, run_id: str | None = None):
    live = _active_optimizer(run_id)
    library = live[1].library if live else UIL_Library()
    lock = live[0].lock if live else None
    ctx = lock if lock is not None else _NullLock()
    with ctx:
        principles = library._load()
        target = next((p for p in principles if p.get("principle_id") == req.item_id), None)
        if target is None:
            raise HTTPException(status_code=404, detail="principle not found")
        target.update(req.fields)
        library._save(principles)
        try:
            VersionStore.shared().record_cell(
                cell_id=req.item_id,
                cell_type="principle",
                op="human_edit",
                post_state=target,
                cause="webui",
            )
        except Exception:
            pass
    return {"ok": True}


# ── Network ───────────────────────────────────────────────────────────

@router.get("/network")
def get_network(run_id: str | None = None):
    live = _active_optimizer(run_id)
    if live is not None:
        record, optimizer = live
        with record.lock:
            return {**optimizer.network.to_dict(), "summary": optimizer.network.get_summary()}
    network = EditableNetwork()
    return {**network.to_dict(), "summary": network.get_summary()}


@router.patch("/network/node")
def edit_node(req: StateEditRequest, run_id: str | None = None):
    live = _active_optimizer(run_id)
    network = live[1].network if live else EditableNetwork()
    lock = live[0].lock if live else None
    ctx = lock if lock is not None else _NullLock()
    with ctx:
        ok = network.edit_node(req.item_id, **req.fields)
        if not live:
            network.save()
    if not ok:
        raise HTTPException(status_code=404, detail="node not found")
    return {"ok": True}


# ── Formulas ──────────────────────────────────────────────────────────

@router.get("/formulas")
def get_formulas(run_id: str | None = None):
    live = _active_optimizer(run_id)
    if live is not None:
        record, optimizer = live
        with record.lock:
            symbolic = (
                optimizer.symbolic_library.get_all()
                if optimizer.symbolic_library is not None
                else []
            )
            return {
                "formulas": optimizer.formula_library.get_all(),
                "symbolic_formulas": symbolic,
                "active_formula_id": (
                    optimizer._current_formula_id
                    or getattr(optimizer.classifier, "_last_used_formula_id", None)
                ),
            }
    from figpso_uil.cognition.paths import results_path
    loss = FormulaLibrary().get_all()
    symbolic_lib = FormulaLibrary(filepath=results_path("uil_symbolic_classifier_library.json"))
    return {
        "formulas": loss,
        "symbolic_formulas": symbolic_lib.get_all(),
        "active_formula_id": None,
    }


@router.patch("/formulas")
def edit_formula(req: StateEditRequest, run_id: str | None = None):
    live = _active_optimizer(run_id)
    formula_library = live[1].formula_library if live else FormulaLibrary()
    lock = live[0].lock if live else None
    ctx = lock if lock is not None else _NullLock()
    with ctx:
        entry = formula_library._formulas.get(req.item_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="formula not found")
        entry.update(req.fields)
        formula_library._save()
    return {"ok": True}


# ── Causal hypotheses ────────────────────────────────────────────────

@router.get("/causal")
def get_causal(run_id: str | None = None):
    live = _active_optimizer(run_id)
    engine = live[1].causal_engine if live else CausalInterventionEngine()
    lock = live[0].lock if live else None
    ctx = lock if lock is not None else _NullLock()
    with ctx:
        return {"hypotheses": [h.to_dict() for h in engine._hypotheses.values()]}


@router.patch("/causal")
def edit_hypothesis(req: StateEditRequest, run_id: str | None = None):
    live = _active_optimizer(run_id)
    engine = live[1].causal_engine if live else CausalInterventionEngine()
    lock = live[0].lock if live else None
    ctx = lock if lock is not None else _NullLock()
    with ctx:
        h = engine._hypotheses.get(req.item_id)
        if h is None:
            raise HTTPException(status_code=404, detail="hypothesis not found")
        for key, value in req.fields.items():
            if hasattr(h, key):
                setattr(h, key, value)
        engine._save()
    return {"ok": True}


# ── Episodes (read-only -- KDE-blended data, not hand-edited) ────────

@router.get("/episodes")
def get_episodes(run_id: str | None = None):
    live = _active_optimizer(run_id)
    attention = live[1].episodic_attention if live else EpisodicAttention()
    lock = live[0].lock if live else None
    ctx = lock if lock is not None else _NullLock()
    with ctx:
        return {
            "episodes": [
                {"fingerprint": e["fingerprint"], "trajectory": e["trajectory"], "resolution": e["resolution"]}
                for e in attention._episodes
            ],
            "count": attention.episode_count,
        }


@router.get("/persistence")
def get_persistence():
    """On-disk memory files — proves learning persists between runs."""
    return get_persistence_stats()


@router.get("/memory-architecture")
def get_memory_architecture_route(
    dataset: str | None = None,
    run_id: str | None = None,
):
    """Working vs archive memory, champions, consciousness brief."""
    live_working = None
    live = _active_optimizer(run_id)
    if live is not None:
        record, optimizer = live
        with record.lock:
            from figpso_uil.webui.backend.memory_architecture import _network_counts

            disk = _network_counts()
            live_working = {
                "working_nodes": optimizer.network.node_count,
                "working_edges": len(getattr(optimizer.network, "_edges", [])),
                "disk_working_nodes": disk["working_nodes"],
                "disk_working_edges": disk["working_edges"],
            }
    return get_memory_architecture(dataset, live_working=live_working)


@router.get("/consciousness")
def get_consciousness_route(dataset: str):
    """Abstract exploration memory for a dataset."""
    return get_consciousness_brief(dataset)


@router.get("/changelog")
def get_changelog_route(limit: int = 40):
    """Recent timestamped graph and formula change events."""
    return get_recent_changelog(limit=limit)


@router.get("/journal/recent")
def get_recent_journal(run_id: str | None = None, limit: int = 20):
    live = _active_optimizer(run_id)
    if live is not None:
        record, optimizer = live
        with record.lock:
            entries = optimizer.journal.get_recent(limit)
    else:
        from figpso_uil.cognition.uil_journal import UIL_Journal
        entries = UIL_Journal().get_recent(limit)
    return {"entries": entries, "count": len(entries)}


@router.get("/live-feed")
def get_live_feed(offset: int = 0, max_lines: int = 100, tail: int | None = None):
    """Poll terminal + web runs written to uil_live_feed.jsonl."""
    if tail is not None and tail > 0:
        return read_tail(max_lines=tail)
    return read_since(line_offset=offset, max_lines=max_lines)


@router.post("/live-feed/clear")
def post_clear_live_feed():
    clear_feed()
    return {"ok": True, "enabled": is_enabled()}


@router.post("/memory/clear")
def post_clear_memory(bootstrap: bool = True):
    """Wipe UIL memory; optionally reinstall clean seed principles (default on)."""
    if bootstrap:
        result = clear_and_bootstrap(clear_live_feed=True)
    else:
        result = clear_all_memory(clear_live_feed=True)
    return result


@router.get("/governance")
def get_governance():
    from figpso_uil.cognition.bootstrap_knowledge import load_governance_manifest
    return {"governance": load_governance_manifest()}


@router.get("/ablation-reports")
def list_ablation_reports():
    """List JSON reports from experiments/ablation_harness.py."""
    import json
    import os
    reports_dir = os.path.join(RESULTS_DIR, "ablation_reports")
    if not os.path.isdir(reports_dir):
        return {"reports": []}
    items = []
    for name in sorted(os.listdir(reports_dir), reverse=True):
        if not name.endswith(".json"):
            continue
        path = os.path.join(reports_dir, name)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for i, entry in enumerate(data[:10]):
                    if not isinstance(entry, dict):
                        continue
                    items.append({
                        "filename": f"{name}#{i}",
                        "path": path,
                        "generated_at": entry.get("generated_at"),
                        "classifier": entry.get("classifier"),
                        "stages": entry.get("stages"),
                        "datasets": list((entry.get("datasets") or {}).keys()),
                    })
            elif isinstance(data, dict):
                items.append({
                    "filename": name,
                    "path": path,
                    "generated_at": data.get("generated_at"),
                    "classifier": data.get("classifier"),
                    "stages": data.get("stages"),
                    "datasets": list((data.get("datasets") or {}).keys()),
                })
        except (json.JSONDecodeError, OSError):
            items.append({"filename": name, "path": path, "error": "unreadable"})
    return {"reports": items[:50]}


@router.get("/lab-reports/latest")
def latest_lab_report():
    """Return most recent UIL lab report markdown (v22 M5)."""
    reports_dir = os.path.join(RESULTS_DIR, "reports")
    if not os.path.isdir(reports_dir):
        raise HTTPException(status_code=404, detail="no reports directory")
    md_files = sorted(
        (f for f in os.listdir(reports_dir) if f.endswith(".md")),
        key=lambda n: os.path.getmtime(os.path.join(reports_dir, n)),
        reverse=True,
    )
    if not md_files:
        raise HTTPException(status_code=404, detail="no lab reports")
    path = os.path.join(reports_dir, md_files[0])
    with open(path, encoding="utf-8") as fh:
        return {"filename": md_files[0], "path": path, "markdown": fh.read()}


class _NullLock:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
