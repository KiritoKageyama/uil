"""Prune working journal, principles, and hypotheses into the archive.

Same pattern as the KG: working pools stay lean for speed; everything
removed is timestamped in KnowledgeArchive with dataset + confidence +
age context. Mining uses JournalForgetting weights — archived entries
are excluded from the working set but remain queryable.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

from figpso_uil.cognition.journal_forgetting import JournalForgetting

if TYPE_CHECKING:
    from figpso_uil.cognition.causal_intervention import CausalInterventionEngine
    from figpso_uil.cognition.knowledge_archive import KnowledgeArchive
    from figpso_uil.cognition.uil_journal import UIL_Journal
    from figpso_uil.cognition.uil_library import UIL_Library


class WorkingMemoryPruner:
    """Archive low-value cognition entries; leave tombstones where needed."""

    MAX_WORKING_JOURNAL = 4000
    JOURNAL_PROTECT_RECENT = 600
    MAX_WORKING_PRINCIPLES = 180
    MAX_WORKING_HYPOTHESES = 100
    STALE_HYPOTHESIS_SEC = 7 * 24 * 3600

    def __init__(self, forgetting: JournalForgetting | None = None):
        self._forgetting = forgetting or JournalForgetting()

    def prune_all(
        self,
        *,
        journal: "UIL_Journal",
        library: "UIL_Library",
        causal_engine: "CausalInterventionEngine",
        archive: "KnowledgeArchive",
        dataset: str,
    ) -> dict[str, int]:
        stats: dict[str, int] = {}
        stats["journal_archived"] = self.prune_journal(journal, archive, dataset)
        stats["principles_archived"] = self.prune_principles(library, archive, dataset)
        stats["hypotheses_archived"] = self.prune_hypotheses(
            causal_engine, archive, dataset,
        )
        if any(stats.values()):
            archive.record_cognition_prune(
                dataset,
                journal_archived=stats["journal_archived"],
                principles_archived=stats["principles_archived"],
                hypotheses_archived=stats["hypotheses_archived"],
                working_journal=journal.get_entry_count(),
                working_principles=len(library._load()),
                working_hypotheses=len(causal_engine._hypotheses),
                reason="periodic_working_prune",
            )
            archive.save()
        return stats

    def prune_journal(
        self,
        journal: "UIL_Journal",
        archive: "KnowledgeArchive",
        dataset: str,
    ) -> int:
        entries = journal._load()
        if len(entries) <= self.MAX_WORKING_JOURNAL:
            return 0

        protect = set(range(max(0, len(entries) - self.JOURNAL_PROTECT_RECENT), len(entries)))
        weights = self._forgetting.compute_weights(entries)
        to_archive: list[dict] = []
        keep: list[dict] = []

        overflow = len(entries) - self.MAX_WORKING_JOURNAL
        candidates: list[tuple[int, float, dict]] = []
        for i, entry in enumerate(entries):
            if i in protect:
                keep.append(entry)
                continue
            age_rank = len(entries) - 1 - i
            acc = float(entry.get("outcome", {}).get("accuracy", 0.0))
            w = float(weights[i]) if i < len(weights) else 0.5
            entry_ds = entry.get("dataset", "")
            score = w
            if entry_ds and entry_ds != dataset:
                score *= 0.85
            if acc < 0.55:
                score *= 0.7
            candidates.append((i, score, entry))

        candidates.sort(key=lambda t: t[1])
        archive_ids = {c[0] for c in candidates[:overflow]}

        for i, entry in enumerate(entries):
            if i in protect or i not in archive_ids:
                keep.append(entry)
            else:
                to_archive.append(entry)

        if not to_archive:
            return 0

        archive.archive_journal_entries(
            to_archive,
            reason="journal_working_cap_age_confidence",
            dataset=dataset,
        )
        journal._save(keep)
        return len(to_archive)

    def prune_principles(
        self,
        library: "UIL_Library",
        archive: "KnowledgeArchive",
        dataset: str,
    ) -> int:
        from figpso_uil.cognition.principle_hierarchy import TIER_EXCEPTION

        principles = library._load()
        if not principles:
            return 0

        archived = 0
        surviving: list[dict] = []
        for p in principles:
            pid = p.get("principle_id", "")
            conf = float(p.get("confidence", 0.5))
            applied = int(p.get("times_applied", 0))
            helped = int(p.get("times_helped", 0))
            help_ratio = helped / max(applied, 1)
            tier = p.get("tier", 3)
            p_ds = p.get("dataset") or p.get("mining_level", "")

            reason = None
            if tier == TIER_EXCEPTION:
                surviving.append(p)
                continue
            if conf < 0.30:
                reason = "low_confidence_decay"
            elif applied >= 20 and help_ratio < 0.25:
                reason = "low_help_ratio"
            elif (
                p_ds
                and p_ds not in (dataset, "seed", "within_dataset", "cross_dataset")
                and conf < 0.45
                and applied >= 8
            ):
                reason = "other_dataset_low_confidence"
            elif (
                dataset
                and library.check_quarantine(pid, dataset)
                and conf < 0.50
                and applied >= 10
            ):
                reason = "quarantined_weak"

            if reason:
                archive.archive_principle(p, reason=reason, dataset=dataset)
                tombstone = {
                    "principle_id": pid,
                    "status": "archived",
                    "confidence": conf,
                    "evidence_count": p.get("evidence_count", 0),
                    "archive_reason": reason,
                    "discovered_at": p.get("discovered_at"),
                    "archived_at": datetime.now(timezone.utc).isoformat(),
                }
                surviving.append(tombstone)
                archived += 1
            else:
                surviving.append(p)

        if len(surviving) > self.MAX_WORKING_PRINCIPLES:
            ranked = sorted(
                [p for p in surviving if p.get("status") != "archived"],
                key=lambda x: (
                    float(x.get("confidence", 0)),
                    int(x.get("times_helped", 0)),
                ),
            )
            drop_n = len(surviving) - self.MAX_WORKING_PRINCIPLES
            drop_ids = {p["principle_id"] for p in ranked[:drop_n]}
            trimmed: list[dict] = []
            for p in surviving:
                if p.get("principle_id") in drop_ids and p.get("status") != "archived":
                    archive.archive_principle(
                        p, reason="principle_working_cap", dataset=dataset,
                    )
                    trimmed.append({
                        "principle_id": p["principle_id"],
                        "status": "archived",
                        "confidence": p.get("confidence", 0),
                        "archive_reason": "principle_working_cap",
                        "archived_at": datetime.now(timezone.utc).isoformat(),
                    })
                    archived += 1
                else:
                    trimmed.append(p)
            surviving = trimmed

        if archived:
            library._save(surviving)
        return archived

    def prune_hypotheses(
        self,
        engine: "CausalInterventionEngine",
        archive: "KnowledgeArchive",
        dataset: str,
    ) -> int:
        from figpso_uil.cognition.causal_intervention import (
            STATUS_CORRELATIONAL,
            STATUS_PENDING,
            STATUS_REFUTED,
        )
        archived = 0
        now = time.time()
        keep: dict[str, Any] = {}

        for hid, hyp in list(engine._hypotheses.items()):
            status = hyp.status
            reason = None
            if status == STATUS_REFUTED:
                reason = "refuted"
            elif status == STATUS_CORRELATIONAL and hyp.causal_strength < 0.15:
                reason = "weak_correlational"
            elif status == STATUS_PENDING and (now - hyp.created_at) > self.STALE_HYPOTHESIS_SEC:
                reason = "stale_pending"

            if reason:
                archive.archive_hypothesis(
                    hyp.to_dict(), reason=reason, dataset=dataset,
                )
                archived += 1
            else:
                keep[hid] = hyp

        if len(keep) > self.MAX_WORKING_HYPOTHESES:
            ranked = sorted(
                keep.values(),
                key=lambda h: (h.causal_strength, -h.created_at),
            )
            drop = ranked[: len(keep) - self.MAX_WORKING_HYPOTHESES]
            for hyp in drop:
                archive.archive_hypothesis(
                    hyp.to_dict(), reason="hypothesis_working_cap", dataset=dataset,
                )
                keep.pop(hyp.hypothesis_id, None)
                archived += 1

        if archived:
            engine._hypotheses = keep
            engine._save()
        return archived
