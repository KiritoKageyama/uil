"""Aggregate working memory, archive, changelogs, and champions for the web UI."""

from __future__ import annotations

import json
import os
from typing import Any

from figpso_uil.cognition.change_log import FormulaChangeLog, GraphChangeLog
from figpso_uil.cognition.knowledge_archive import KnowledgeArchive
from figpso_uil.cognition.paths import results_path


def _load_json(path: str) -> dict | list | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _network_counts() -> dict:
    data = _load_json(results_path("uil_network_v20.json"))
    if not isinstance(data, dict):
        return {"working_nodes": 0, "working_edges": 0}
    nodes = data.get("nodes", {})
    edges = data.get("edges", [])
    return {
        "working_nodes": len(nodes) if isinstance(nodes, dict) else 0,
        "working_edges": len(edges) if isinstance(edges, list) else 0,
    }


def get_memory_architecture(
    dataset: str | None = None,
    *,
    live_working: dict | None = None,
) -> dict:
    """Working vs archive memory, champions, and changelog stats."""
    archive = KnowledgeArchive.shared()
    graph_log = GraphChangeLog.shared()
    formula_log = FormulaChangeLog.shared()
    arch_stats = archive.get_stats()
    disk = _network_counts()
    net = dict(disk)
    if live_working:
        net["working_nodes"] = int(live_working.get("working_nodes", disk["working_nodes"]))
        net["working_edges"] = int(live_working.get("working_edges", disk["working_edges"]))
        net["source"] = "live_run"
        if live_working.get("disk_working_nodes") is not None:
            net["disk_working_nodes"] = int(live_working["disk_working_nodes"])
            net["disk_working_edges"] = int(live_working["disk_working_edges"])
    else:
        net["source"] = "disk"

    datasets = archive._data.get("datasets", {})
    champions = {}
    for ds_name, bucket in datasets.items():
        champ = bucket.get("champion")
        if champ:
            champions[ds_name] = {
                "best_accuracy": champ.get("best_accuracy"),
                "formula_id": champ.get("formula_id"),
                "saved_at": champ.get("saved_at"),
                "updated_at": champ.get("updated_at"),
                "learning_mode": champ.get("learning_mode"),
            }

    consciousness = None
    last_prune = None
    if dataset:
        consciousness = archive.get_consciousness_brief(dataset)
        events = archive._data.get("datasets", {}).get(dataset, {}).get(
            "graph_prune_events", [],
        )
        if events:
            last_prune = events[-1]

    return {
        "working": {
            **net,
            "description": "Pruned graph used during optimization (live counts when a run is active)",
        },
        "last_prune": last_prune,
        "archive": {
            "archived_nodes": arch_stats.get("archived_nodes", 0),
            "archived_edges": arch_stats.get("archived_edges", 0),
            "archived_formulas": arch_stats.get("archived_formulas", 0),
            "archived_journal": arch_stats.get("archived_journal", 0),
            "archived_principles": arch_stats.get("archived_principles", 0),
            "archived_hypotheses": arch_stats.get("archived_hypotheses", 0),
            "datasets_tracked": arch_stats.get("datasets_tracked", 0),
            "updated_at": arch_stats.get("updated_at"),
            "description": "Full copy of everything pruned from working memory",
        },
        "champions": champions,
        "changelogs": {
            "graph": graph_log.get_stats(),
            "formula": formula_log.get_stats(),
        },
        "consciousness": consciousness,
    }


def get_recent_changelog(limit: int = 40) -> dict:
    """Recent timestamped graph + formula events."""
    limit = max(1, min(int(limit), 200))
    graph_data = _load_json(results_path("uil_graph_changelog_v20.json")) or {}
    formula_data = _load_json(results_path("uil_formula_changelog_v20.json")) or {}

    graph_events = list(graph_data.get("events", []))[-limit:]
    formula_events = list(formula_data.get("events", []))[-limit:]

    return {
        "graph": {
            "updated_at": graph_data.get("updated_at"),
            "total_events": len(graph_data.get("events", [])),
            "events": list(reversed(graph_events)),
        },
        "formula": {
            "updated_at": formula_data.get("updated_at"),
            "total_events": len(formula_data.get("events", [])),
            "events": list(reversed(formula_events)),
        },
    }


def get_consciousness_brief(dataset: str) -> dict:
    archive = KnowledgeArchive.shared()
    return archive.get_consciousness_brief(dataset)
