"""Full knowledge archive + abstract exploration memory for UIL.

Working graph/formula pools are pruned during runs for speed; everything
removed is copied here so nothing is lost. Summaries tell the optimizer
what was already tried and what conclusions prior runs reached.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from figpso_uil.cognition.paths import results_path

_SHARED: "KnowledgeArchive | None" = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class KnowledgeArchive:
    """Persistent archive of pruned KG items, formulas, and run conclusions."""

    LOW_ACCURACY_THRESHOLD = 0.62
    MIN_TRIALS_FOR_AVOID = 4

    def __init__(self, filepath: str | None = None):
        self.filepath = filepath or results_path("uil_knowledge_archive_v20.json")
        self._data: dict[str, Any] = {
            "version": 1,
            "updated_at": None,
            "graph": {"nodes": {}, "edges": []},
            "formulas": {},
            "journal": [],
            "principles": {},
            "hypotheses": {},
            "datasets": {},
        }
        self._load()

    @classmethod
    def shared(cls, filepath: str | None = None) -> "KnowledgeArchive":
        global _SHARED
        path = filepath or results_path("uil_knowledge_archive_v20.json")
        if _SHARED is None or _SHARED.filepath != path:
            _SHARED = cls(filepath=path)
        return _SHARED

    def _dataset_bucket(self, dataset: str) -> dict:
        ds = self._data.setdefault("datasets", {}).setdefault(
            dataset,
            {
                "conclusions": [],
                "formula_exploration": {},
                "graph_prune_events": [],
            },
        )
        return ds

    # ── Graph archive ───────────────────────────────────────────────

    def archive_graph_nodes(self, nodes: dict, reason: str, dataset: str | None = None) -> int:
        archived = 0
        store = self._data.setdefault("graph", {}).setdefault("nodes", {})
        for nid, node_dict in nodes.items():
            entry = dict(node_dict)
            entry["_archived_at"] = _utc_now()
            entry["_archive_reason"] = reason
            if dataset:
                entry["_dataset_context"] = dataset
            store[nid] = entry
            archived += 1
        if archived:
            self._touch()
        return archived

    def archive_graph_edges(self, edges: list[dict], reason: str, dataset: str | None = None) -> int:
        store = self._data.setdefault("graph", {}).setdefault("edges", [])
        for edge in edges:
            entry = dict(edge)
            entry["_archived_at"] = _utc_now()
            entry["_archive_reason"] = reason
            if dataset:
                entry["_dataset_context"] = dataset
            store.append(entry)
        if edges:
            self._touch()
        return len(edges)

    def record_graph_prune(
        self,
        dataset: str,
        *,
        nodes_archived: int,
        edges_archived: int,
        working_nodes: int,
        working_edges: int,
        reason: str,
    ) -> None:
        bucket = self._dataset_bucket(dataset)
        bucket.setdefault("graph_prune_events", []).append({
            "at": _utc_now(),
            "nodes_archived": nodes_archived,
            "edges_archived": edges_archived,
            "working_nodes": working_nodes,
            "working_edges": working_edges,
            "reason": reason,
        })
        bucket["graph_prune_events"] = bucket["graph_prune_events"][-40:]
        self._touch()

    # ── Journal / principles / hypotheses archive ─────────────────────

    def archive_journal_entries(
        self,
        entries: list[dict],
        *,
        reason: str,
        dataset: str | None = None,
    ) -> int:
        store = self._data.setdefault("journal", [])
        for entry in entries:
            full = dict(entry)
            full["_archived_at"] = _utc_now()
            full["_archive_reason"] = reason
            if dataset:
                full["_dataset_context"] = dataset
            store.append(full)
        if entries:
            if len(store) > 50_000:
                self._data["journal"] = store[-50_000:]
            self._touch()
        return len(entries)

    def archive_principle(
        self,
        entry: dict,
        *,
        reason: str,
        dataset: str | None = None,
    ) -> None:
        pid = entry.get("principle_id")
        if not pid:
            return
        full = dict(entry)
        full["_archived_at"] = _utc_now()
        full["_archive_reason"] = reason
        if dataset:
            full["_dataset_context"] = dataset
        self._data.setdefault("principles", {})[pid] = full
        self._touch()

    def archive_hypothesis(
        self,
        entry: dict,
        *,
        reason: str,
        dataset: str | None = None,
    ) -> None:
        hid = entry.get("hypothesis_id")
        if not hid:
            return
        full = dict(entry)
        full["_archived_at"] = _utc_now()
        full["_archive_reason"] = reason
        if dataset:
            full["_dataset_context"] = dataset
        self._data.setdefault("hypotheses", {})[hid] = full
        self._touch()

    def record_cognition_prune(
        self,
        dataset: str,
        *,
        journal_archived: int = 0,
        principles_archived: int = 0,
        hypotheses_archived: int = 0,
        working_journal: int = 0,
        working_principles: int = 0,
        working_hypotheses: int = 0,
        reason: str,
    ) -> None:
        bucket = self._dataset_bucket(dataset)
        bucket.setdefault("cognition_prune_events", []).append({
            "at": _utc_now(),
            "journal_archived": journal_archived,
            "principles_archived": principles_archived,
            "hypotheses_archived": hypotheses_archived,
            "working_journal": working_journal,
            "working_principles": working_principles,
            "working_hypotheses": working_hypotheses,
            "reason": reason,
        })
        bucket["cognition_prune_events"] = bucket["cognition_prune_events"][-40:]
        self._touch()

    # ── Formula archive ─────────────────────────────────────────────

    def archive_formula(
        self,
        entry: dict,
        *,
        reason: str,
        dataset: str | None = None,
    ) -> None:
        fid = entry.get("formula_id")
        if not fid:
            return
        full = dict(entry)
        full["_archived_at"] = _utc_now()
        full["_archive_reason"] = reason
        if dataset:
            full["_dataset_context"] = dataset
        self._data.setdefault("formulas", {})[fid] = full
        self._touch()

    def record_formula_trial(
        self,
        dataset: str,
        formula_id: str,
        accuracy: float,
        helped: bool,
        *,
        formula_name: str | None = None,
    ) -> None:
        bucket = self._dataset_bucket(dataset)
        explor = bucket.setdefault("formula_exploration", {})
        rec = explor.setdefault(
            formula_id,
            {
                "formula_id": formula_id,
                "name": formula_name or formula_id,
                "trials": 0,
                "times_helped": 0,
                "best_accuracy": 0.0,
                "last_accuracy": 0.0,
                "verdict": "unknown",
                "summary": "",
            },
        )
        if formula_name:
            rec["name"] = formula_name
        rec["trials"] += 1
        if helped:
            rec["times_helped"] += 1
        rec["last_accuracy"] = float(accuracy)
        if accuracy > rec["best_accuracy"]:
            rec["best_accuracy"] = float(accuracy)
        rec["verdict"] = self._formula_verdict(rec)
        rec["summary"] = self._formula_summary(rec, dataset)
        rec["last_seen"] = _utc_now()
        rec["updated_at"] = rec["last_seen"]
        self._touch()

    @classmethod
    def _formula_verdict(cls, rec: dict) -> str:
        best = rec.get("best_accuracy", 0.0)
        trials = rec.get("trials", 0)
        if trials < 2:
            return "exploring"
        if best >= 0.78:
            return "strong"
        if best >= 0.68:
            return "moderate"
        if trials >= cls.MIN_TRIALS_FOR_AVOID and best < cls.LOW_ACCURACY_THRESHOLD:
            return "low"
        return "weak"

    @classmethod
    def _formula_summary(cls, rec: dict, dataset: str) -> str:
        verdict = rec.get("verdict", "unknown")
        best = rec.get("best_accuracy", 0.0)
        trials = rec.get("trials", 0)
        name = rec.get("name", rec.get("formula_id", "?"))
        if verdict == "low":
            return (
                f"On {dataset}, formula {name} was tried {trials} times and peaked at "
                f"{best:.1%} — already explored with low accuracy; avoid unless stress forces retry."
            )
        if verdict == "strong":
            return (
                f"On {dataset}, formula {name} reached {best:.1%} in {trials} trials — "
                f"prior strong result; prefer refining this lineage."
            )
        return (
            f"On {dataset}, formula {name}: {trials} trials, best {best:.1%} ({verdict})."
        )

    def record_conclusion(
        self,
        dataset: str,
        text: str,
        metadata: dict | None = None,
    ) -> None:
        bucket = self._dataset_bucket(dataset)
        bucket.setdefault("conclusions", []).append({
            "at": _utc_now(),
            "text": text,
            "metadata": metadata or {},
        })
        bucket["conclusions"] = bucket["conclusions"][-30:]
        self._touch()

    # ── Dataset champion checkpoint ─────────────────────────────────

    def save_champion(self, dataset: str, payload: dict) -> None:
        bucket = self._dataset_bucket(dataset)
        existing = bucket.get("champion")
        if existing and float(existing.get("best_accuracy", 0)) >= float(
            payload.get("best_accuracy", 0)
        ):
            return
        bucket["champion"] = {**payload, "saved_at": _utc_now(), "updated_at": _utc_now()}
        self.record_conclusion(
            dataset,
            (
                f"Champion checkpoint: {payload.get('best_accuracy', 0):.1%} val accuracy "
                f"with formula {payload.get('formula_id', '?')}."
            ),
            metadata={"kind": "champion", "formula_id": payload.get("formula_id")},
        )
        self._touch()

    def load_champion(self, dataset: str) -> dict | None:
        return self._data.get("datasets", {}).get(dataset, {}).get("champion")

    # ── Consciousness brief for optimizer / SymbolicFormula ─────────

    def get_consciousness_brief(self, dataset: str) -> dict:
        bucket = self._data.get("datasets", {}).get(dataset, {})
        explor = bucket.get("formula_exploration", {})
        conclusions = bucket.get("conclusions", [])[-8:]
        champion = bucket.get("champion")

        avoid_formulas = []
        known_good = []
        for fid, rec in explor.items():
            item = {
                "formula_id": fid,
                "name": rec.get("name", fid),
                "trials": rec.get("trials", 0),
                "best_accuracy": rec.get("best_accuracy", 0.0),
                "verdict": rec.get("verdict", "unknown"),
                "summary": rec.get("summary", ""),
            }
            if rec.get("verdict") == "low":
                avoid_formulas.append(item)
            elif rec.get("verdict") in ("strong", "moderate"):
                known_good.append(item)

        avoid_formulas.sort(key=lambda x: x["best_accuracy"])
        known_good.sort(key=lambda x: -x["best_accuracy"])

        graph_arch = self._data.get("graph", {})
        formula_arch = self._data.get("formulas", {})

        summary_lines = [c["text"] for c in conclusions]
        if champion:
            summary_lines.append(
                f"Prior champion on {dataset}: {champion.get('best_accuracy', 0):.1%} "
                f"with {champion.get('formula_id', '?')} — resume and refine, do not restart blind."
            )
        for rec in avoid_formulas[:6]:
            summary_lines.append(rec["summary"])
        for rec in known_good[:3]:
            summary_lines.append(rec["summary"])

        return {
            "dataset": dataset,
            "champion": champion,
            "conclusions": conclusions,
            "avoid_formulas": avoid_formulas[:12],
            "known_good_formulas": known_good[:8],
            "abstract_summary": summary_lines[-14:],
            "archive_stats": {
                "archived_nodes": len(graph_arch.get("nodes", {})),
                "archived_edges": len(graph_arch.get("edges", [])),
                "archived_formulas": len(formula_arch),
                "archived_journal": len(self._data.get("journal", [])),
                "archived_principles": len(self._data.get("principles", {})),
                "archived_hypotheses": len(self._data.get("hypotheses", {})),
            },
        }

    def get_stats(self) -> dict:
        graph = self._data.get("graph", {})
        return {
            "archived_nodes": len(graph.get("nodes", {})),
            "archived_edges": len(graph.get("edges", [])),
            "archived_formulas": len(self._data.get("formulas", {})),
            "archived_journal": len(self._data.get("journal", [])),
            "archived_principles": len(self._data.get("principles", {})),
            "archived_hypotheses": len(self._data.get("hypotheses", {})),
            "datasets_tracked": len(self._data.get("datasets", {})),
            "updated_at": self._data.get("updated_at"),
        }

    def _touch(self) -> None:
        self._data["updated_at"] = _utc_now()

    def save(self) -> None:
        if not self.filepath:
            return
        dirpath = os.path.dirname(self.filepath)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, default=str)

    def _load(self) -> None:
        try:
            with open(self.filepath, encoding="utf-8") as fh:
                self._data = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
