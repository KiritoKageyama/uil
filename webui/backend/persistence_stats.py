"""Read persisted UIL state from disk — for the Memory Proof panel."""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone

from figpso_uil.cognition.paths import RESULTS_DIR, results_path

PERSISTENCE_FILES = {
    "journal": "uil_journal_v19.json",
    "journal_memory": "uil_journal_v19_memory.json",
    "principles": "uil_principles_v19.json",
    "network": "uil_network_v20.json",
    "causal": "uil_causal_v20.json",
    "episodes": "uil_episodes_v20.json",
    "formulas": "uil_formula_library.json",
    "symbolic_formulas": "uil_symbolic_classifier_library.json",
    "knowledge_archive": "uil_knowledge_archive_v20.json",
    "graph_changelog": "uil_graph_changelog_v20.json",
    "formula_changelog": "uil_formula_changelog_v20.json",
    "meta_principles": "uil_meta_principles.json",
    "failures": "uil_failures_v20.json",
}


def _file_info(path: str) -> dict:
    if not os.path.isfile(path):
        return {
            "path": path,
            "exists": False,
            "bytes": 0,
            "modified_utc": None,
            "record_count": 0,
        }
    stat = os.stat(path)
    modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    count = 0
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            count = len(data)
        elif isinstance(data, dict):
            if "events" in data and isinstance(data["events"], list):
                count = len(data["events"])
            elif "nodes" in data and isinstance(data.get("nodes"), dict):
                # Working KG or archive graph bucket
                edge_n = len(data.get("edges", [])) if isinstance(data.get("edges"), list) else 0
                count = len(data["nodes"]) + edge_n
            elif "graph" in data and isinstance(data["graph"], dict):
                g = data["graph"]
                count = len(g.get("nodes", {})) + len(g.get("edges", []))
            elif "datasets" in data:
                ds = data.get("datasets", {})
                count = len(ds)
                for bucket in ds.values():
                    if isinstance(bucket, dict):
                        count += len(bucket.get("formula_exploration", {}))
            elif "hypotheses" in data:
                count = len(data.get("hypotheses", {}))
            elif "formulas" in data:
                count = len(data.get("formulas", {}))
            else:
                count = len(data)
    except (json.JSONDecodeError, OSError):
        count = -1
    return {
        "path": path,
        "exists": True,
        "bytes": stat.st_size,
        "modified_utc": modified,
        "record_count": count,
    }


def get_persistence_stats() -> dict:
    """Snapshot of all on-disk UIL memory files."""
    files = {}
    total_bytes = 0
    for key, filename in PERSISTENCE_FILES.items():
        info = _file_info(results_path(filename))
        files[key] = info
        total_bytes += info["bytes"]
    return {
        "results_dir": RESULTS_DIR,
        "files": files,
        "total_bytes": total_bytes,
        "learning_active": any(
            f["exists"] and f["bytes"] > 2 for f in files.values()
        ),
    }
