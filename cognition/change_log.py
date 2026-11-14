"""Timestamped change logs for graph and formula mutations.

Working graph nodes stay free of audit fields; every node/edge add, edit,
merge, prune, and removal is recorded here with UTC timestamps. Formula
changes are logged similarly in a dedicated formula changelog file.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from figpso_uil.cognition.paths import results_path

try:
    from figpso_uil.cognition.version_store import VersionStore
except ImportError:
    VersionStore = None  # type: ignore

_MAX_GRAPH_EVENTS = 10_000
_MAX_FORMULA_EVENTS = 10_000

_GRAPH_SHARED: "GraphChangeLog | None" = None
_FORMULA_SHARED: "FormulaChangeLog | None" = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def changelog_beside(sibling_filepath: str | None, filename: str) -> str | None:
    """Path for a changelog co-located with a library/graph file.

    Isolates experiment arms (bug #30b): a library under state_dir must not
    append into the shared results/ changelog.
    """
    if not sibling_filepath:
        return None
    return os.path.join(os.path.dirname(os.path.abspath(sibling_filepath)), filename)


def _is_default_results_dir(path: str) -> bool:
    return os.path.normpath(os.path.dirname(path)) == os.path.normpath(
        os.path.dirname(results_path("uil_graph_changelog_v20.json"))
    )


def scoped_graph_changelog(sibling_filepath: str | None) -> "GraphChangeLog":
    path = changelog_beside(sibling_filepath, "uil_graph_changelog_v20.json")
    if path is None or _is_default_results_dir(path):
        return GraphChangeLog.shared(path)
    return GraphChangeLog(filepath=path)


def scoped_formula_changelog(sibling_filepath: str | None) -> "FormulaChangeLog":
    path = changelog_beside(sibling_filepath, "uil_formula_changelog_v20.json")
    if path is None or _is_default_results_dir(path):
        return FormulaChangeLog.shared(path)
    return FormulaChangeLog(filepath=path)


class GraphChangeLog:
    """Append-only audit trail for node/edge lifecycle (not stored on nodes)."""

    def __init__(self, filepath: str | None = None):
        self.filepath = filepath or results_path("uil_graph_changelog_v20.json")
        self._data: dict[str, Any] = {
            "version": 1,
            "updated_at": None,
            "events": [],
        }
        self._load()

    @classmethod
    def shared(cls, filepath: str | None = None) -> "GraphChangeLog":
        global _GRAPH_SHARED
        path = filepath or results_path("uil_graph_changelog_v20.json")
        if _GRAPH_SHARED is None or _GRAPH_SHARED.filepath != path:
            _GRAPH_SHARED = cls(filepath=path)
        return _GRAPH_SHARED

    def record(
        self,
        action: str,
        entity_type: str,
        entity_id: str,
        *,
        dataset: str | None = None,
        details: dict | None = None,
        snapshot: dict | list | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "timestamp": utc_now(),
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
        }
        if dataset:
            event["dataset"] = dataset
        if details:
            event["details"] = details
        if snapshot is not None:
            event["snapshot"] = snapshot
        events = self._data.setdefault("events", [])
        events.append(event)
        if VersionStore is not None:
            try:
                store = VersionStore.shared()
                store.append(
                    cell_id=entity_id,
                    cell_type=entity_type,
                    op=action,
                    payload={"post_state": snapshot, "details": details or {}},
                    cause=dataset,
                )
            except Exception:
                pass
        if len(events) > _MAX_GRAPH_EVENTS:
            self._data["events"] = events[-_MAX_GRAPH_EVENTS:]
        self._data["updated_at"] = event["timestamp"]

    def save(self) -> None:
        if not self.filepath:
            return
        dirpath = os.path.dirname(self.filepath)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, default=str)

    def get_stats(self) -> dict:
        events = self._data.get("events", [])
        return {
            "event_count": len(events),
            "updated_at": self._data.get("updated_at"),
            "last_event": events[-1] if events else None,
        }

    def _load(self) -> None:
        try:
            with open(self.filepath, encoding="utf-8") as fh:
                self._data = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            pass


class FormulaChangeLog:
    """Timestamped trail of formula library mutations."""

    def __init__(self, filepath: str | None = None):
        self.filepath = filepath or results_path("uil_formula_changelog_v20.json")
        self._data: dict[str, Any] = {
            "version": 1,
            "updated_at": None,
            "events": [],
        }
        self._load()

    @classmethod
    def shared(cls, filepath: str | None = None) -> "FormulaChangeLog":
        global _FORMULA_SHARED
        path = filepath or results_path("uil_formula_changelog_v20.json")
        if _FORMULA_SHARED is None or _FORMULA_SHARED.filepath != path:
            _FORMULA_SHARED = cls(filepath=path)
        return _FORMULA_SHARED

    def record(
        self,
        action: str,
        formula_id: str,
        *,
        dataset: str | None = None,
        details: dict | None = None,
        snapshot: dict | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "timestamp": utc_now(),
            "action": action,
            "formula_id": formula_id,
        }
        if dataset:
            event["dataset"] = dataset
        if details:
            event["details"] = details
        if snapshot is not None:
            event["snapshot"] = snapshot
        events = self._data.setdefault("events", [])
        events.append(event)
        if len(events) > _MAX_FORMULA_EVENTS:
            self._data["events"] = events[-_MAX_FORMULA_EVENTS:]
        self._data["updated_at"] = event["timestamp"]

    def save(self) -> None:
        if not self.filepath:
            return
        dirpath = os.path.dirname(self.filepath)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, default=str)

    def get_stats(self) -> dict:
        events = self._data.get("events", [])
        return {
            "event_count": len(events),
            "updated_at": self._data.get("updated_at"),
            "last_event": events[-1] if events else None,
        }

    def _load(self) -> None:
        try:
            with open(self.filepath, encoding="utf-8") as fh:
                self._data = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
