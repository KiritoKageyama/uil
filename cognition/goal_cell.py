"""GoalCell — LCA-inspired top-down attention channel (2026-07-18).

Conceptual home: the LCA document (`agent_architecture.md`) proposes a
layered agent with an explicit top-down director that tells lower layers
"what to focus on and what to ignore." UIL's existing attention channels
(KDE fingerprint retrieval, lift-based graph edges, co-activation
propagation, ANOVA-F feature preselection) are all BOTTOM-UP: they infer
salience from data. There is no directive channel a human or higher
reasoner can use to say "for this run, prioritize extrapolation" or
"prefer a readable model to a slightly-more-accurate opaque one."

This module adds that channel as a native UIL primitive — a KnowledgeCell
that:

  1. lives in the VersionStore (event-sourced, replayable, byte-identical
     reconstruction at any seq, per M1 invariants);
  2. materializes to a live JSON file (`results/uil_goals.json`) so it
     survives snapshot re-materialization (bug #16 invariant);
  3. is proposed-only by LLMMutator (never written directly by the
     mutator — same safety property as formula proposals);
  4. is CONSULTED by downstream consumers (KG traversal, hypothesis
     chooser, router) — never overrides architectural knobs set
     explicitly by the router/operator (bug #21 invariant).

We deliberately do NOT import LCA-the-agent from agent_architecture.md.
LCA's attention is DOM/screen-region attention; ours is
scientific-hypothesis attention. The spirit is preserved (a top-down
director layer); the vehicle is native.

Data model
----------
A GoalCell carries five parallel "focus" channels — one per downstream
consumer that reads it:

  directive             — one-line human summary (for readability / audit)
  prefer_domains        — router bias (soft, not override)
  focus_features        — features whose usage is UP-weighted in KG traversal
  ignore_features       — features whose usage is DOWN-weighted
  prefer_relations      — edge relation types (leads_to/strengthens/…) to boost
  goal_keywords         — tags for matching hypothesis variables

Any channel left empty is a no-op for that consumer. A run with NO active
goal cell is byte-identical to the pre-GoalCell behavior — this is a
strictly additive extension.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any

from figpso_uil.cognition.knowledge_cell import (
    CellVerdict,
    MutationProposal,
    Provenance,
    StructuralForm,
)
from figpso_uil.cognition.paths import results_path


GOAL_CELL_TYPE = "goal"
ACTIVE_GOAL_ID = "goal:current"

# Bias bounds — a goal SOFTENS the search, it does not replace it.
# 3.0x amplification / 0.1x suppression is the maximum wind-back the
# consumer applies before clamping; a runaway goal never zeroes out a
# node or infinities another.
GOAL_BIAS_MAX = 3.0
GOAL_BIAS_MIN = 0.1


@dataclass
class GoalCell:
    cell_id: str = ACTIVE_GOAL_ID
    cell_type: str = GOAL_CELL_TYPE
    version: int = 1
    provenance: Provenance = field(default_factory=Provenance)
    # A goal is a directive, not a belief. Confidence expresses "how
    # much bias to apply" — 1.0 = full strength, 0.0 = the goal is
    # present but inert (useful for dry-run audits).
    confidence: float = 1.0

    directive: str = ""
    prefer_domains: list[str] = field(default_factory=list)
    focus_features: list[Any] = field(default_factory=list)
    ignore_features: list[Any] = field(default_factory=list)
    prefer_relations: list[str] = field(default_factory=list)
    goal_keywords: list[str] = field(default_factory=list)
    status: str = "active"  # "active" | "archived" | "superseded"

    # ── KnowledgeCell protocol ───────────────────────────────────────

    def evaluate(self, state: dict) -> CellVerdict:
        applies = self.status == "active" and self.confidence > 0.0
        return CellVerdict(
            applies=applies,
            confidence=self.confidence if applies else 0.0,
            message=self.directive or f"goal {self.cell_id}",
        )

    def mutate(self, proposal: MutationProposal) -> "GoalCell":
        payload = proposal.payload or {}
        return GoalCell(
            cell_id=self.cell_id,
            version=self.version + 1,
            provenance=Provenance(
                source=proposal.proposer,
                human_editor=(
                    proposal.proposer if proposal.proposer == "human" else None
                ),
            ),
            confidence=float(payload.get("confidence", self.confidence)),
            directive=str(payload.get("directive", self.directive)),
            prefer_domains=list(payload.get("prefer_domains", self.prefer_domains)),
            focus_features=list(payload.get("focus_features", self.focus_features)),
            ignore_features=list(payload.get("ignore_features", self.ignore_features)),
            prefer_relations=list(payload.get("prefer_relations", self.prefer_relations)),
            goal_keywords=list(payload.get("goal_keywords", self.goal_keywords)),
            status=str(payload.get("status", self.status)),
        )

    def compose(self, other: "GoalCell") -> "GoalCell | None":
        # Composition = union of focus channels; ignore wins over focus
        # on conflict (safer default: an operator asking to ignore a
        # feature should not be overridden by a stale focus entry).
        if not isinstance(other, GoalCell):
            return None
        ignore = list(dict.fromkeys(self.ignore_features + other.ignore_features))
        focus = [
            f for f in dict.fromkeys(self.focus_features + other.focus_features)
            if f not in ignore
        ]
        return GoalCell(
            cell_id=self.cell_id,
            version=max(self.version, other.version) + 1,
            confidence=max(self.confidence, other.confidence),
            directive=(other.directive or self.directive),
            prefer_domains=list(dict.fromkeys(
                self.prefer_domains + other.prefer_domains
            )),
            focus_features=focus,
            ignore_features=ignore,
            prefer_relations=list(dict.fromkeys(
                self.prefer_relations + other.prefer_relations
            )),
            goal_keywords=list(dict.fromkeys(
                self.goal_keywords + other.goal_keywords
            )),
            status=other.status if other.status == "archived" else self.status,
        )

    def to_readable(self) -> str:
        parts = [f"GOAL {self.cell_id} v{self.version}"]
        if self.directive:
            parts.append(f'"{self.directive}"')
        if self.prefer_domains:
            parts.append(f"prefer_domains={self.prefer_domains}")
        if self.focus_features:
            parts.append(f"focus={self.focus_features}")
        if self.ignore_features:
            parts.append(f"ignore={self.ignore_features}")
        if self.prefer_relations:
            parts.append(f"prefer_relations={self.prefer_relations}")
        if self.goal_keywords:
            parts.append(f"keywords={self.goal_keywords}")
        parts.append(f"(conf={self.confidence:.2f}, status={self.status})")
        return " | ".join(parts)

    def to_structural(self) -> StructuralForm:
        return StructuralForm(encoding=self.to_readable())

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "cell_type": self.cell_type,
            "version": self.version,
            "confidence": self.confidence,
            "directive": self.directive,
            "prefer_domains": list(self.prefer_domains),
            "focus_features": list(self.focus_features),
            "ignore_features": list(self.ignore_features),
            "prefer_relations": list(self.prefer_relations),
            "goal_keywords": list(self.goal_keywords),
            "status": self.status,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict, cell_id: str | None = None) -> "GoalCell":
        return cls(
            cell_id=cell_id or d.get("cell_id", ACTIVE_GOAL_ID),
            version=int(d.get("version", 1)),
            confidence=float(d.get("confidence", 1.0)),
            directive=str(d.get("directive", "")),
            prefer_domains=list(d.get("prefer_domains", [])),
            focus_features=list(d.get("focus_features", [])),
            ignore_features=list(d.get("ignore_features", [])),
            prefer_relations=list(d.get("prefer_relations", [])),
            goal_keywords=list(d.get("goal_keywords", [])),
            status=str(d.get("status", "active")),
        )

    # ── Consumer bias functions ──────────────────────────────────────

    def _feature_ids(self, seq: list) -> set:
        """Normalize a mixed list of names/indices to a hashable set."""
        out = set()
        for item in seq:
            if item is None:
                continue
            try:
                out.add(str(item))
            except Exception:
                pass
        return out

    def node_relevance(self, conditions: list, edge_relation: str | None = None) -> float:
        """Attention multiplier for a KG node (consumed by editable_network).

        Returns 1.0 for a "neutral" node — the caller multiplies its
        existing attention weight by this factor. `edge_relation`, when
        provided, additionally applies the relation_bias() boost (kept
        as a kwarg for callers that fold both signals into a single
        multiplier call).
        """
        if self.confidence <= 0.0 or self.status != "active":
            return 1.0
        focus = self._feature_ids(self.focus_features)
        ignore = self._feature_ids(self.ignore_features)
        if not focus and not ignore:
            return self.relation_bias(edge_relation) if edge_relation else 1.0

        multiplier = 1.0
        for cond in conditions or []:
            feat = str(cond.get("feature", "")) if isinstance(cond, dict) else str(cond)
            if feat in focus:
                multiplier *= 1.0 + 0.5 * self.confidence
            if feat in ignore:
                multiplier *= 1.0 - 0.6 * self.confidence
        if edge_relation:
            multiplier *= self.relation_bias(edge_relation)
        return max(GOAL_BIAS_MIN, min(GOAL_BIAS_MAX, multiplier))

    def relation_bias(self, relation: str | None) -> float:
        """Edge-relation multiplier (consumed by KG edge propagation).

        Boost when `relation` is in prefer_relations; neutral otherwise.
        Cannot suppress: goals endorse relations, they do not censor them.
        """
        if (
            self.confidence <= 0.0
            or self.status != "active"
            or not relation
            or not self.prefer_relations
        ):
            return 1.0
        return (
            1.0 + 0.3 * self.confidence
            if relation in self.prefer_relations
            else 1.0
        )

    def hypothesis_score(self, variable: str) -> float:
        """Preference score for a causal hypothesis's variable.

        Returns a score in [0, ~2]: 1.0 for neutral, higher for on-goal
        variables, lower for off-goal. Caller uses this to rank; nothing
        is filtered out unconditionally (a goal biases, does not censor).
        """
        if self.confidence <= 0.0 or self.status != "active":
            return 1.0
        var = str(variable or "").lower()
        focus = {str(f).lower() for f in self.focus_features}
        ignore = {str(f).lower() for f in self.ignore_features}
        keywords = {str(k).lower() for k in self.goal_keywords}

        score = 1.0
        if any(f in var for f in focus if f):
            score += 0.5 * self.confidence
        if any(k in var for k in keywords if k):
            score += 0.3 * self.confidence
        if any(f in var for f in ignore if f):
            score -= 0.5 * self.confidence
        return max(0.05, score)

    def domain_bias(self, candidate_domain: str) -> float:
        """Router bias for a candidate strategy/domain.

        Returns 1.0 for neutral. Only positively biases preferred
        domains — the goal cannot forbid a domain, only endorse one
        (bug #21 invariant: architecture channels are not overridable
        by search-layer state; a goal is closer to a search-layer
        preference than an architecture directive).
        """
        if self.confidence <= 0.0 or self.status != "active":
            return 1.0
        if not self.prefer_domains:
            return 1.0
        return 1.0 + 0.4 * self.confidence if candidate_domain in self.prefer_domains else 1.0


# ── Persistence store ────────────────────────────────────────────────


class GoalStore:
    """Live JSON materialization of GoalCells.

    Analogous to UIL_Library / FormulaLibrary / CausalInterventionEngine:
    a small JSON file that composite snapshots re-materialize from.
    Without this, an event-only goal edit would be overwritten at the
    next snapshot (the bug #16 anti-pattern). Concurrent-safe: appended
    from webui workers and read from optimizer.fit().
    """

    _lock = threading.RLock()

    def __init__(self, filepath: str | None = None):
        self.filepath = filepath or results_path("uil_goals.json")
        self._goals: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.filepath, "r", encoding="utf-8") as fh:
                blob = json.load(fh)
            if isinstance(blob, dict):
                self._goals = {
                    str(k): dict(v) for k, v in blob.items()
                    if isinstance(v, dict)
                }
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self._goals = {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        tmp = self.filepath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._goals, fh, indent=2)
        os.replace(tmp, self.filepath)

    def canonical_state(self) -> dict[str, dict]:
        """Snapshot-ready view (called from build_composite_snapshot)."""
        with self._lock:
            return {k: dict(v) for k, v in self._goals.items()}

    def set(self, goal: GoalCell) -> GoalCell:
        """Install `goal` as the active cell; archive any existing active
        goal at a different cell_id. Returns the persisted GoalCell."""
        with self._lock:
            existing = self._goals.get(goal.cell_id)
            if existing:
                goal.version = int(existing.get("version", 0)) + 1
            for cid, entry in list(self._goals.items()):
                if cid != goal.cell_id and entry.get("status") == "active":
                    entry["status"] = "superseded"
            self._goals[goal.cell_id] = goal.to_dict()
            self._save()
            return goal

    def get(self, cell_id: str = ACTIVE_GOAL_ID) -> GoalCell | None:
        with self._lock:
            d = self._goals.get(cell_id)
            return GoalCell.from_dict(d, cell_id=cell_id) if d else None

    def get_active(self) -> GoalCell | None:
        with self._lock:
            for cid, entry in self._goals.items():
                if entry.get("status") == "active":
                    return GoalCell.from_dict(entry, cell_id=cid)
            return None

    def apply_edit(self, cell_id: str, fields: dict) -> bool:
        """HITL edit path — called from cell_store.apply_materialized_edit."""
        with self._lock:
            entry = self._goals.get(cell_id)
            if entry is None:
                return False
            entry.update(fields)
            entry["version"] = int(entry.get("version", 1)) + 1
            self._save()
            return True


def record_goal_event(
    version_store,
    goal: GoalCell,
    *,
    op: str = "set_goal",
    cause: str | None = None,
) -> int | None:
    """Append a GoalCell state event to VersionStore + persist to disk.

    This is the sanctioned write path (bug #16 invariant): a goal edit
    goes through BOTH the event stream AND the materialized JSON. Any
    caller that skips one side will silently lose the edit on the next
    snapshot re-materialization.
    """
    GoalStore().set(goal)
    if version_store is None:
        return None
    return version_store.record_cell(
        cell_id=goal.cell_id,
        cell_type=GOAL_CELL_TYPE,
        op=op,
        post_state=goal.to_dict(),
        cause=cause,
    )
