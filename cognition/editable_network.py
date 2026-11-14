"""Editable Network — Directed graph of DT rule nodes + KG-inspired edges.

V20+.3: The core knowledge structure of the UIL. Not a neural network.
An explainable, editable, graph-based reasoning system where:

    - Each NODE is a Decision Tree rule (machine-discovered conditions)
    - Each EDGE is a relationship with a type, weight, and optional formula
    - Traversal produces actions: state → match nodes → follow edges → collect

Graph is constrained to prevent infinite loops:
    ω_new = ω_old * 0.5 per revisit, cutoff at 0.01

Edge types (KG-inspired):
    LEADS_TO:     "This condition typically precedes that condition"
    CONTRADICTS:  "These conditions conflict — only one should fire"
    DEPENDS_ON:   "This condition requires that condition to hold"
    STRENGTHENS:  "This condition reinforces that condition's override"

Nodes, edges, and weights are all editable — human or system can
modify thresholds, relationships, and formulas at any time.
"""

from __future__ import annotations
import json
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import time
import numpy as np
from datetime import datetime, timezone

from figpso_uil.cognition.failure_point import record_failure_point
from figpso_uil.cognition.execution_context import flatten_context
from figpso_uil.cognition.paths import results_path
from figpso_uil.cognition.change_log import scoped_graph_changelog
from figpso_uil.cognition.version_store import VersionStore
from figpso_uil.cognition.condition_match import (
    condition_sets_similar,
    sanitize_overrides,
)


# ── Edge Types ───────────────────────────────────────────────────────

EDGE_LEADS_TO = "leads_to"
EDGE_CONTRADICTS = "contradicts"
EDGE_DEPENDS_ON = "depends_on"
EDGE_STRENGTHENS = "strengthens"

VALID_EDGE_TYPES = {EDGE_LEADS_TO, EDGE_CONTRADICTS, EDGE_DEPENDS_ON, EDGE_STRENGTHENS}

# ── Loop Prevention Constants ────────────────────────────────────────

LOOP_DECAY_FACTOR = 0.5   # ω_new = ω_old * 0.5 per revisit
LOOP_CUTOFF = 0.01        # Stop traversing when weight < 0.01

# Metric-based archive: scores below these thresholds are archive candidates.
# Thresholds are adjusted by system_stress at prune time (high stress → keep more).
DEFAULT_EDGE_ARCHIVE_CUT = 0.22
DEFAULT_NODE_ARCHIVE_CUT = 0.24
RECENCY_HALF_LIFE_DAYS = 14.0


# Override keys that express a CONSTRAINT on a classifier knob rather
# than a value for it -- see uil_library._generate_override.
_BOUND_CEILING_PREFIX = "config_max."
_BOUND_FLOOR_PREFIX = "config_min."
# An interval the knob should stay out of, carried as [lo, hi].
_BOUND_AVOID_PREFIX = "config_avoid."


def _is_bound_key(key: str) -> bool:
    return key.startswith((
        _BOUND_CEILING_PREFIX, _BOUND_FLOOR_PREFIX, _BOUND_AVOID_PREFIX,
    )) or key in ("max_penalty", "min_penalty", "penalty_avoid")


def _combine_bound(key: str, current, incoming):
    """Combine two constraints on the same knob into the binding one.

    Both traversals blend ordinary numeric overrides -- one by a weighted
    average, one by accumulating `value * attention`. That is defensible
    for a VALUE (several nodes voting on a setting) and wrong for a
    BOUND. "sparse_l1_c must stay under 0.3" summed with "under 0.5"
    would give 0.8, a ceiling looser than either rule asked for; with
    enough activated nodes it grows without limit and the constraint
    silently stops constraining. That is the silent-no-op family this
    whole vocabulary change exists to escape (bugs #1, #25).

    Constraints do not average -- they intersect. The binding ceiling is
    the lowest, the binding floor is the highest, and a bound is scaled
    by nothing: a rule's confidence decides whether it is REACHED during
    traversal, not how strong its constraint is once it has been.
    """
    # Avoid-regions are [lo, hi] pairs. Two nodes condemning different
    # bands cannot be merged into one band, so the first reached wins --
    # traversal order is by tier, so that is the most authoritative one.
    if key.startswith(_BOUND_AVOID_PREFIX) or key == "penalty_avoid":
        return current if current is not None else incoming
    if not isinstance(incoming, (int, float)) or isinstance(incoming, bool):
        return current
    if not isinstance(current, (int, float)) or isinstance(current, bool):
        return float(incoming)
    is_ceiling = key.startswith(_BOUND_CEILING_PREFIX) or key == "max_penalty"
    return (
        min(current, float(incoming)) if is_ceiling
        else max(current, float(incoming))
    )


def _parse_iso_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def _recency_score(ts: str | None, *, half_life_days: float = RECENCY_HALF_LIFE_DAYS) -> float:
    """1.0 when just used; decays exponentially with idle time."""
    parsed = _parse_iso_timestamp(ts)
    if parsed is None:
        return 0.0
    now = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - parsed).total_seconds() / 86400.0)
    return float(0.5 ** (age_days / half_life_days))


def _dataset_relevance_score(domains: list[str] | str | None, dataset: str | None) -> float:
    """How relevant a node/edge is to the active dataset domain."""
    if not dataset:
        return 0.55
    if not domains:
        return 0.35
    if isinstance(domains, str):
        domains = [domains]
    return 1.0 if dataset in domains else 0.12


def _normalize_domains(raw) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw]
    return list(raw)


class NetworkNode:
    """A node in the Editable Network — wraps a DT-mined rule.

    Each node is a machine-discovered condition set with an action override.
    Equivalent to a principle from the library, but now embedded in a graph
    with explicit relationships to other principles.
    """

    __slots__ = [
        "node_id", "conditions", "action_override", "tier",
        "confidence", "evidence_count", "activation_count",
        "last_activated", "metadata",
    ]

    def __init__(
        self,
        node_id: str,
        conditions: list[dict],
        action_override: dict | None = None,
        tier: int = 3,
        confidence: float = 0.5,
        evidence_count: int = 0,
    ):
        self.node_id = node_id
        self.conditions = conditions
        self.action_override = action_override or {}
        self.tier = tier
        self.confidence = confidence
        self.evidence_count = evidence_count
        self.activation_count = 0
        self.last_activated: str | None = None
        self.metadata: dict = {}

    def matches(self, state: dict) -> bool:
        """Check if this node's conditions match the current state.

        Pure math — evaluates each condition against the state vector.
        """
        import operator
        ops = {
            "<=": operator.le, "<": operator.lt,
            ">=": operator.ge, ">": operator.gt,
            "==": operator.eq, "!=": operator.ne,
        }
        for cond in self.conditions:
            feature = cond.get("feature", "")
            op_str = cond.get("op", "")
            threshold = cond.get("threshold")
            if threshold is None or op_str not in ops:
                return False
            value = self._resolve(feature, state)
            if value is None:
                return False
            try:
                if not ops[op_str](float(value), float(threshold)):
                    return False
            except (ValueError, TypeError):
                return False
        return True

    def _resolve(self, feature: str, state: dict) -> float | None:
        """Resolve a feature name to a value from the state dict."""
        if feature in state:
            val = state[feature]
            if isinstance(val, (int, float)):
                return float(val)
        if "." in feature:
            _, short = feature.rsplit(".", 1)
            if short in state:
                val = state[short]
                if isinstance(val, (int, float)):
                    return float(val)
        return None

    def to_dict(self) -> dict:
        """Serialize node to dict."""
        return {
            "node_id": self.node_id,
            "conditions": self.conditions,
            "action_override": self.action_override,
            "tier": self.tier,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "activation_count": self.activation_count,
            "last_activated": self.last_activated,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NetworkNode":
        """Deserialize node from dict.

        Overrides are repaired on the way in (UIL_Library.
        sanitize_overrides): a node keeps an override only if one of its
        own conditions names the knob being constrained. 18 of the 300
        nodes in the live graph carried penalty bounds inherited from
        DERIVED-feature thresholds (`min_penalty: 1270.45`,
        `max_penalty: 0.0003`) -- harmless while bound aggregation
        SUMMED, actively wrong once it intersects.
        """
        sanitize_overrides(d)
        node = cls(
            node_id=d["node_id"],
            conditions=d.get("conditions", []),
            action_override=d.get("action_override", {}),
            tier=d.get("tier", 3),
            confidence=d.get("confidence", 0.5),
            evidence_count=d.get("evidence_count", 0),
        )
        node.activation_count = d.get("activation_count", 0)
        node.last_activated = d.get("last_activated")
        node.metadata = d.get("metadata", {})
        return node


class NetworkEdge:
    """An edge in the Editable Network — a typed, weighted relationship.

    Edges carry:
        - type:   leads_to | contradicts | depends_on | strengthens
        - weight: confidence/strength of the relationship [0, 1]
        - formula_id: optional formula from FormulaLibrary to apply
        - evidence: how many times this relationship was observed
    """

    __slots__ = [
        "source_id", "target_id", "edge_type", "weight",
        "formula_id", "evidence_count", "times_traversed",
        "times_helped", "last_traversed", "metadata",
    ]

    def __init__(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = EDGE_LEADS_TO,
        weight: float = 0.5,
        formula_id: str | None = None,
    ):
        if edge_type not in VALID_EDGE_TYPES:
            raise ValueError(f"Invalid edge type: {edge_type}")
        self.source_id = source_id
        self.target_id = target_id
        self.edge_type = edge_type
        self.weight = np.clip(weight, 0.0, 1.0)
        self.formula_id = formula_id
        self.evidence_count = 1
        self.times_traversed = 0
        self.times_helped = 0
        self.last_traversed: str | None = None
        self.metadata: dict = {}

    @property
    def edge_key(self) -> str:
        return f"{self.source_id}->{self.target_id}:{self.edge_type}"

    @property
    def help_ratio(self) -> float:
        """Fraction of traversals that helped."""
        if self.times_traversed == 0:
            return 0.5  # Neutral
        return self.times_helped / self.times_traversed

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "formula_id": self.formula_id,
            "evidence_count": self.evidence_count,
            "times_traversed": self.times_traversed,
            "times_helped": self.times_helped,
            "last_traversed": self.last_traversed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NetworkEdge":
        edge = cls(
            source_id=d["source_id"],
            target_id=d["target_id"],
            edge_type=d.get("edge_type", EDGE_LEADS_TO),
            weight=d.get("weight", 0.5),
            formula_id=d.get("formula_id"),
        )
        edge.evidence_count = d.get("evidence_count", 1)
        edge.times_traversed = d.get("times_traversed", 0)
        edge.times_helped = d.get("times_helped", 0)
        edge.last_traversed = d.get("last_traversed")
        edge.metadata = d.get("metadata", {})
        return edge


class EditableNetwork:
    """The UIL's Explainable Editable Network.

    A directed graph where:
        - Nodes are DT-mined rules (conditions → overrides)
        - Edges are typed relationships (leads_to, contradicts, etc.)
        - Traversal is state-driven with loop decay

    Not an NN. No backpropagation. No gradient descent.
    Growth is deterministic: Miner produces nodes, co-occurrence analysis
    produces edges, stress/regret feedback adjusts weights.
    """

    def __init__(
        self,
        formula_engine=None,
        formula_library=None,
        filepath: str | None = results_path("uil_network_v20.json"),
        version_store: "VersionStore | None" = None,
    ):
        self._nodes: dict[str, NetworkNode] = {}
        self._edges: list[NetworkEdge] = []
        # Adjacency for fast lookup: {source_id: [edge_indices]}
        self._adj: dict[str, list[int]] = {}
        # Reverse adjacency: {target_id: [edge_indices]}
        self._rev_adj: dict[str, list[int]] = {}
        # Node counter for ID generation
        self._node_counter = 0
        # V20+.3 edge-formula execution: optional, so a network built
        # without these still works exactly as before (formulas merely
        # get listed in formulas_collected instead of evaluated).
        self._formula_engine = formula_engine
        self._formula_library = formula_library
        # Cross-run co-activation evidence (see discover_relationships) --
        # keyed by "nodeA|nodeB" since JSON dict keys must be strings.
        self._co_activation_tally: dict[str, dict] = {}
        # Persistence: filepath=None disables auto-load/save entirely
        # (used by tests that need a guaranteed-empty graph).
        self.filepath = filepath
        # Co-locate changelog with the graph file (bug #30b isolation).
        self._changelog = (
            scoped_graph_changelog(filepath) if filepath else None
        )
        # M1 (v22 blueprint §1.2): event-sourced lossless history. Every
        # node/edge lifecycle event is appended with its full canonical
        # post-state, so any past network state is reconstructable and
        # belief trajectories are queryable. Mirrors the changelog
        # convention: enabled by default for persistent networks,
        # explicit version_store for tests, disabled when filepath=None.
        self._version_store: VersionStore | None = (
            version_store
            if version_store is not None
            else (VersionStore.shared() if filepath else None)
        )
        if self._version_store is not None:
            self._version_store.set_snapshot_provider(self.canonical_state)
        self._dataset_context: str | None = None
        if self.filepath:
            self._load()

    def set_dataset_context(self, dataset: str | None) -> None:
        """Tag subsequent graph mutations with the active dataset name."""
        self._dataset_context = dataset

    def _tag_node_dataset(self, node: NetworkNode, dataset: str | None = None) -> None:
        ds_name = dataset or self._dataset_context
        if not ds_name:
            return
        domains = _normalize_domains(node.metadata.get("datasets"))
        if ds_name not in domains:
            domains.append(ds_name)
        node.metadata["datasets"] = domains

    def _tag_edge_usage(self, edge: NetworkEdge, dataset: str | None = None) -> None:
        edge.last_traversed = datetime.now(timezone.utc).isoformat()
        ds_name = dataset or self._dataset_context
        if not ds_name:
            return
        domains = _normalize_domains(edge.metadata.get("datasets"))
        if ds_name not in domains:
            domains.append(ds_name)
        edge.metadata["datasets"] = domains

    def node_keep_score(
        self,
        node: NetworkNode,
        *,
        dataset: str | None = None,
        protected: bool = False,
    ) -> float:
        """Composite keep score in [0, 1]. Higher = more worth keeping in working memory."""
        if protected:
            return 1.0

        domains = _normalize_domains(node.metadata.get("datasets"))
        score = 0.0
        score += 0.28 * float(node.confidence)
        score += 0.12 * min(1.0, node.evidence_count / 25.0)
        score += 0.18 * min(1.0, node.activation_count / 12.0)
        score += 0.15 * _recency_score(node.last_activated)
        score += 0.12 * _dataset_relevance_score(domains, dataset)

        pred_regret = node.metadata.get("predicted_regret")
        if pred_regret is not None:
            score += 0.10 * max(0.0, 1.0 - min(1.0, float(pred_regret) * 4.0))
        else:
            score += 0.05

        tier_boost = max(0.0, (6 - min(int(node.tier), 5)) / 5.0)
        score += 0.05 * tier_boost
        return float(np.clip(score, 0.0, 1.0))

    def edge_keep_score(
        self,
        edge: NetworkEdge,
        *,
        dataset: str | None = None,
    ) -> float:
        """Composite keep score in [0, 1] using edge metrics + endpoint node quality."""
        src = self._nodes.get(edge.source_id)
        tgt = self._nodes.get(edge.target_id)
        domains = _normalize_domains(edge.metadata.get("datasets"))

        score = 0.0
        score += 0.30 * float(edge.weight)
        score += 0.26 * float(edge.help_ratio)
        score += 0.12 * min(1.0, edge.evidence_count / 15.0)
        score += 0.10 * min(1.0, edge.times_traversed / 25.0)
        score += 0.10 * _recency_score(edge.last_traversed)
        score += 0.08 * _dataset_relevance_score(domains, dataset)

        if src is not None and tgt is not None:
            score += 0.04 * float(src.confidence)
            score += 0.04 * float(tgt.confidence)

        if edge.times_traversed == 0 and edge.evidence_count <= 1:
            score -= 0.06
        if edge.times_traversed >= 5 and edge.help_ratio < 0.15:
            score -= 0.12

        return float(np.clip(score, 0.0, 1.0))

    def _archive_thresholds(
        self,
        system_stress: float | None = None,
    ) -> tuple[float, float]:
        """Return (edge_cut, node_cut). High stress → lower cut → archive less."""
        stress = 0.5 if system_stress is None else float(np.clip(system_stress, 0.0, 1.0))
        edge_cut = DEFAULT_EDGE_ARCHIVE_CUT * (0.75 + 0.25 * stress)
        node_cut = DEFAULT_NODE_ARCHIVE_CUT * (0.75 + 0.25 * stress)
        return edge_cut, node_cut

    def _edge_should_archive(
        self,
        edge: NetworkEdge,
        score: float,
        edge_cut: float,
    ) -> bool:
        if score >= edge_cut:
            return False
        if edge.times_traversed >= 3:
            return True
        if edge.times_traversed > 0 and edge.help_ratio < 0.15:
            return True
        if edge.times_traversed == 0 and edge.evidence_count <= 1 and edge.weight < 0.35:
            return True
        return score < edge_cut * 0.55

    def _node_should_archive(
        self,
        node: NetworkNode,
        score: float,
        node_cut: float,
        *,
        protected: bool,
        has_edges: bool,
    ) -> bool:
        if protected:
            return False
        if score >= node_cut:
            return False
        if not has_edges:
            return True
        if score < node_cut * 0.50 and node.activation_count == 0:
            return True
        if (
            score < node_cut * 0.65
            and node.activation_count <= 1
            and _recency_score(node.last_activated) < 0.05
        ):
            return True
        return False

    # M1: canonical state = the durable knowledge content of the graph.
    # Volatile usage metadata (activation counts, traversal counters,
    # timestamps, dataset tags) is deliberately EXCLUDED: it mutates on
    # every traversal without lifecycle events, and it is bookkeeping
    # about usage, not knowledge. Replay is byte-identical on canonical
    # fields.
    _REMOVAL_ACTIONS = frozenset({"node_removed", "edge_removed"})

    @staticmethod
    def _canonical_node(node: "NetworkNode") -> dict:
        return {
            "node_id": node.node_id,
            "conditions": node.conditions,
            "action_override": node.action_override,
            "tier": node.tier,
            "confidence": node.confidence,
            "evidence_count": node.evidence_count,
        }

    @staticmethod
    def _canonical_edge(edge: "NetworkEdge") -> dict:
        return {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "edge_type": edge.edge_type,
            "weight": edge.weight,
            "formula_id": edge.formula_id,
            "evidence_count": edge.evidence_count,
        }

    def canonical_state(self) -> dict:
        """Durable knowledge state, JSON-normalized for replay equality."""
        raw = {
            "node": {
                nid: self._canonical_node(n) for nid, n in self._nodes.items()
            },
            "edge": {
                e.edge_key: self._canonical_edge(e) for e in self._edges
            },
        }
        # Round-trip through JSON so live state and replayed state have
        # passed through identical encoding (numpy scalars -> str/float).
        # Empty buckets omitted to match reconstruct()'s normalization.
        raw = {k: v for k, v in raw.items() if v}
        return json.loads(json.dumps(raw, default=str))

    def _canonical_entity(self, entity_type: str, entity_id: str) -> dict | None:
        if entity_type == "node":
            node = self._nodes.get(entity_id)
            if node is None:
                return None
            return json.loads(
                json.dumps(self._canonical_node(node), default=str)
            )
        if entity_type == "edge":
            for e in self._edges:
                if e.edge_key == entity_id:
                    return json.loads(
                        json.dumps(self._canonical_edge(e), default=str)
                    )
        return None

    def _bulk_log(self, op: str, details: dict) -> None:
        """One summary event for a bulk graph operation.

        Bulk prune/decay over a large graph must NOT emit one VersionStore
        event per edge: 70k+ individual appends (each an open+write, every
        500th a full composite snapshot) hang the run for minutes. The
        KnowledgeArchive keeps full copies of everything pruned (nothing
        lost), and the next periodic snapshot re-materializes canonical
        state, so replay stays consistent; the stream records the bulk op
        itself here.
        """
        if self._version_store is None:
            return
        try:
            self._version_store.append(
                cell_id="GRAPH-BULK", cell_type="graph_bulk", op=op,
                payload={"details": details}, cause=self._dataset_context,
            )
        except Exception:
            pass

    def _graph_log(
        self,
        action: str,
        entity_type: str,
        entity_id: str,
        *,
        details: dict | None = None,
        snapshot: dict | list | None = None,
    ) -> None:
        if getattr(self, "_suppress_graph_log", False):
            return
        if self._changelog is not None:
            self._changelog.record(
                action,
                entity_type,
                entity_id,
                dataset=self._dataset_context,
                details=details,
                snapshot=snapshot,
            )
        # M1: emit a state-based event. Removal-family actions delete on
        # replay (post_state None); everything else upserts the entity's
        # canonical post-state looked up NOW (all mutation sites log
        # after mutating; removal sites may log before deleting, which
        # is why removal ops never consult the live lookup).
        if self._version_store is None or entity_type not in ("node", "edge"):
            return
        if action in self._REMOVAL_ACTIONS:
            post = None
        else:
            post = self._canonical_entity(entity_type, entity_id)
            if post is None:
                # Informational only (entity unknown): emit without a
                # post_state key so replay ignores it rather than
                # recording a wrong deletion.
                self._version_store.append(
                    cell_id=entity_id, cell_type=entity_type, op=action,
                    payload={"details": details or {}},
                    cause=self._dataset_context,
                )
                return
        self._version_store.append(
            cell_id=entity_id, cell_type=entity_type, op=action,
            payload={"post_state": post, "details": details or {}},
            cause=self._dataset_context,
        )

    # ── Node Management ─────────────────────────────────────────────

    def add_node(
        self,
        conditions: list[dict],
        action_override: dict | None = None,
        tier: int = 3,
        confidence: float = 0.5,
        evidence_count: int = 0,
        node_id: str | None = None,
    ) -> str:
        """Add a node (DT rule) to the network.

        Returns the node_id of the new or existing node.
        If a node with similar conditions already exists, merges evidence.
        """
        # Check for duplicate
        existing_id = self._find_similar_node(conditions)
        if existing_id is not None:
            existing = self._nodes[existing_id]
            existing.evidence_count += evidence_count
            existing.confidence = min(0.99, 0.6 * existing.confidence + 0.4 * confidence)
            if action_override:
                existing.action_override.update(action_override)
            self._tag_node_dataset(existing)
            self._graph_log(
                "node_merged",
                "node",
                existing_id,
                details={
                    "merged_into": existing_id,
                    "evidence_added": evidence_count,
                    "confidence": existing.confidence,
                },
            )
            return existing_id

        # Create new node
        self._node_counter += 1
        nid = node_id or f"N-{self._node_counter:04d}"
        node = NetworkNode(
            node_id=nid,
            conditions=conditions,
            action_override=action_override,
            tier=tier,
            confidence=confidence,
            evidence_count=evidence_count,
        )
        self._nodes[nid] = node
        self._adj[nid] = []
        self._rev_adj[nid] = []
        self._tag_node_dataset(node)
        self._graph_log(
            "node_added",
            "node",
            nid,
            details={"tier": tier, "confidence": confidence},
            snapshot=node.to_dict(),
        )
        return nid

    def remove_node(
        self,
        node_id: str,
        *,
        reason: str = "explicit_remove",
        merged_into: str | None = None,
    ) -> bool:
        """Remove a node and all its edges."""
        if node_id not in self._nodes:
            return False
        snapshot = self._nodes[node_id].to_dict()
        # Remove edges touching this node
        removed_edges = [
            e.to_dict() for e in self._edges
            if e.source_id == node_id or e.target_id == node_id
        ]
        self._edges = [
            e for e in self._edges
            if e.source_id != node_id and e.target_id != node_id
        ]
        del self._nodes[node_id]
        self._rebuild_adjacency()
        details: dict = {"reason": reason, "edges_removed": len(removed_edges)}
        if merged_into:
            details["merged_into"] = merged_into
        self._graph_log(
            "node_removed",
            "node",
            node_id,
            details=details,
            snapshot=snapshot,
        )
        for edge_dict in removed_edges:
            key = f"{edge_dict['source_id']}->{edge_dict['target_id']}:{edge_dict['edge_type']}"
            self._graph_log(
                "edge_removed",
                "edge",
                key,
                details={"reason": "node_removed", "node_id": node_id},
                snapshot=edge_dict,
            )
        return True

    def get_node(self, node_id: str) -> NetworkNode | None:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def edit_node(self, node_id: str, **kwargs) -> bool:
        """Edit a node's properties. Human or system can call this.

        Supports editing: conditions, action_override, confidence, tier.
        """
        node = self._nodes.get(node_id)
        if node is None:
            return False
        before = node.to_dict()
        for key, value in kwargs.items():
            if hasattr(node, key):
                setattr(node, key, value)
        self._graph_log(
            "node_edited",
            "node",
            node_id,
            details={"fields": list(kwargs.keys())},
            snapshot={"before": before, "after": node.to_dict()},
        )
        return True

    # ── Edge Management ─────────────────────────────────────────────

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = EDGE_LEADS_TO,
        weight: float = 0.5,
        formula_id: str | None = None,
    ) -> int | None:
        """Add an edge between two nodes.

        Returns the edge index, or None if source/target don't exist.
        If a similar edge already exists, strengthens it.
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return None

        # Check for existing edge
        for i, e in enumerate(self._edges):
            if e.source_id == source_id and e.target_id == target_id and e.edge_type == edge_type:
                e.evidence_count += 1
                e.weight = min(0.99, e.weight + 0.02)
                self._graph_log(
                    "edge_strengthened",
                    "edge",
                    e.edge_key,
                    details={
                        "evidence_count": e.evidence_count,
                        "weight": e.weight,
                    },
                )
                return i

        # Create new edge
        edge = NetworkEdge(source_id, target_id, edge_type, weight, formula_id)
        if self._dataset_context:
            edge.metadata["datasets"] = [self._dataset_context]
        idx = len(self._edges)
        self._edges.append(edge)
        self._adj.setdefault(source_id, []).append(idx)
        self._rev_adj.setdefault(target_id, []).append(idx)
        self._graph_log(
            "edge_added",
            "edge",
            edge.edge_key,
            details={"weight": weight, "formula_id": formula_id},
            snapshot=edge.to_dict(),
        )
        return idx

    def remove_edge(self, source_id: str, target_id: str, edge_type: str) -> bool:
        """Remove a specific edge."""
        for i, e in enumerate(self._edges):
            if e.source_id == source_id and e.target_id == target_id and e.edge_type == edge_type:
                snapshot = e.to_dict()
                self._edges.pop(i)
                self._rebuild_adjacency()
                self._graph_log(
                    "edge_removed",
                    "edge",
                    e.edge_key,
                    details={"reason": "explicit_remove"},
                    snapshot=snapshot,
                )
                return True
        return False

    # ── Edge-Formula Execution ──────────────────────────────────────
    #
    # full.txt's "Neural Network Inspiration" section: a traditional NN
    # edge carries a weight that scales how strongly a signal flows;
    # this edge carries a formula that *transforms* it instead. The
    # contract: the formula receives the override value flowing across
    # the edge (bound to the FormulaEngine's existing "y_pred" variable,
    # since these trees were built/mutated as functions of one input)
    # plus edge/node context (anchor "y_true", trust "sample_weight",
    # evidence "n_samples", source confidence "class_weight"), and
    # returns the transformed value that replaces it for the target node.

    def _evaluate_edge_formula(
        self,
        formula_id: str,
        source_node: "NetworkNode",
        edge: "NetworkEdge",
        current_overrides: dict,
    ) -> dict:
        """Apply an edge's formula to the overrides flowing across it.

        A formula evaluation failure here means a mutated/injected
        formula is broken -- that is a real defect in the cognitive
        architecture's output, not a recoverable numerical edge case.
        It is recorded with full context and re-raised rather than
        silently leaving the value untransformed.
        """
        if self._formula_engine is None or self._formula_library is None:
            return current_overrides

        tree = self._formula_library.get_formula_tree(formula_id)
        if tree is None:
            return current_overrides

        anchor = float(source_node.metadata.get("predicted_regret", 0.0))
        transformed = dict(current_overrides)
        for key, value in current_overrides.items():
            if not isinstance(value, (int, float)):
                continue
            variables = {
                "y_pred": np.array([float(value)]),
                "y_true": np.array([anchor]),
                "sample_weight": np.array([edge.weight]),
                "n_samples": np.array([float(source_node.evidence_count)]),
                "class_weight": np.array([source_node.confidence]),
            }
            try:
                result = self._formula_engine.evaluate(tree, variables)
                new_val = float(np.asarray(result).reshape(-1)[0])
                if np.isfinite(new_val):
                    # Sanity band: evolved formulas (square, exp, ...) can
                    # produce values that are finite but astronomically
                    # large. Repeated traversals then compound them
                    # (observed: min_penalty squared stage-over-stage to
                    # 1e273, poisoning actions, the journal, and mined
                    # thresholds). A transform may reshape an override;
                    # it must not launch it outside any physically
                    # meaningful knob range.
                    transformed[key] = float(np.clip(new_val, -1e6, 1e6))
            except Exception as e:
                record_failure_point(
                    "editable_network._evaluate_edge_formula",
                    {
                        "formula_id": formula_id,
                        "source_node_id": source_node.node_id,
                        "edge_target_id": edge.target_id,
                        "edge_type": edge.edge_type,
                        "override_key": key,
                        "override_value": value,
                    },
                    e,
                )
                raise
        return transformed

    # ── Traversal (State-Driven) ────────────────────────────────────

    def traverse(
        self,
        state: "dict | object",
        max_depth: int = 10,
    ) -> dict:
        """Traverse the network given a state vector.

        1. Find all nodes whose conditions match the state (entry points)
        2. Follow edges from matched nodes, collecting overrides
        3. Apply loop decay: ω_new = ω_old * 0.5 per revisit
        4. Cut off when weight < 0.01
        5. Handle CONTRADICTS edges (exclude conflicting nodes)

        Args:
            state: Either a plain state dict (back-compat) or an
                ExecutionContext (see execution_context.py) carrying
                state plus the previous stage's action/config, so
                'action.*'/'config.*' node conditions can match too.

        Returns:
            {
                "overrides": merged action overrides,
                "activated_nodes": list of node_ids that fired,
                "paths": list of traversal paths taken,
                "formulas_collected": list of formula_ids along paths,
                "traversal_time": float (seconds),
            }
        """
        state = flatten_context(state)
        start = time.perf_counter()

        # Phase 1: Find entry points (nodes matching the state)
        entry_nodes = []
        for nid, node in self._nodes.items():
            if node.matches(state):
                entry_nodes.append(nid)

        if not entry_nodes:
            return {
                "overrides": {},
                "activated_nodes": [],
                "paths": [],
                "formulas_collected": [],
                "traversal_time": time.perf_counter() - start,
            }

        # Phase 2: BFS traversal with loop decay
        overrides = {}
        activated = set()
        contradicted = set()  # Nodes excluded by CONTRADICTS edges
        paths = []
        formulas = []

        # visited_weights: tracks accumulated weight per node for loop decay
        visited_weights: dict[str, float] = {}

        # Sort entry points by tier (lower = more authoritative)
        entry_nodes.sort(key=lambda nid: self._nodes[nid].tier)

        # BFS queue: (node_id, current_weight, path)
        queue = [(nid, 1.0, [nid]) for nid in entry_nodes]

        while queue:
            current_id, current_weight, path = queue.pop(0)

            # Loop decay: if revisiting, halve the weight
            if current_id in visited_weights:
                current_weight *= LOOP_DECAY_FACTOR
            visited_weights[current_id] = current_weight

            # Cutoff: weight too low
            if current_weight < LOOP_CUTOFF:
                continue

            # Skip contradicted nodes
            if current_id in contradicted:
                continue

            # Max depth
            if len(path) > max_depth:
                continue

            # Activate this node
            node = self._nodes.get(current_id)
            if node is None:
                continue

            activated.add(current_id)
            node.activation_count += 1
            node.last_activated = datetime.now(timezone.utc).isoformat()
            self._tag_node_dataset(node)

            # Collect overrides (weighted by current_weight and node confidence)
            effective_weight = current_weight * node.confidence
            for key, value in node.action_override.items():
                if _is_bound_key(key):
                    overrides[key] = _combine_bound(key, overrides.get(key), value)
                elif key not in overrides:
                    overrides[key] = value
                elif isinstance(value, (int, float)) and isinstance(overrides[key], (int, float)):
                    # Weighted blend for numeric overrides
                    overrides[key] = (
                        overrides[key] * 0.7 + value * 0.3 * effective_weight
                    )

            # Follow outgoing edges
            for edge_idx in self._adj.get(current_id, []):
                edge = self._edges[edge_idx]
                target_id = edge.target_id

                # Handle edge types
                if edge.edge_type == EDGE_CONTRADICTS:
                    contradicted.add(target_id)
                    continue

                if edge.edge_type == EDGE_DEPENDS_ON:
                    # DEPENDS_ON: only follow if target already activated
                    if target_id not in activated:
                        continue

                # Weight propagation: parent_weight * edge_weight
                propagated_weight = current_weight * edge.weight

                # Track edge traversal
                edge.times_traversed += 1
                self._tag_edge_usage(edge)

                # Collect and EXECUTE the formula if edge has one: it
                # transforms the overrides flowing into the target node.
                if edge.formula_id:
                    formulas.append(edge.formula_id)
                    overrides = self._evaluate_edge_formula(
                        edge.formula_id, node, edge, overrides
                    )

                # Enqueue target
                new_path = path + [target_id]
                queue.append((target_id, propagated_weight, new_path))

            # Record completed path
            if len(path) > 1:
                paths.append(path)

        elapsed = time.perf_counter() - start
        return {
            "overrides": overrides,
            "activated_nodes": sorted(activated),
            "paths": paths,
            "formulas_collected": formulas,
            "traversal_time": round(elapsed, 6),
        }

    # ── V20+.9: Causal Gate Routing ─────────────────────────────────

    def traverse_with_attention(
        self,
        state: "dict | object",
        max_depth: int = 10,
        goal: "Any | None" = None,
    ) -> dict:
        """Traverse using causal gate routing (V20+.9).

        Enhanced traversal with:
        1. Dynamic attention = f(match_score, causal_impact)
        2. Continuous contradiction blending (0.7*Universal + 0.3*Local)
        3. Same loop decay (ω_new = ω_old * 0.5, cutoff 0.01)
        4. Optional goal-prior (LCA-inspired top-down attention, 2026-07-18):
           when `goal` is a GoalCell (or any object with a `node_relevance`
           callable), its returned multiplier scales the node's attention.
           Off by default — goal=None reproduces prior behavior exactly.

        Match score: distance of state value to threshold (closer = lower score)
        Causal impact: historical help_ratio of the edge

        Args:
            state: Either a plain state dict (back-compat) or an
                ExecutionContext (see execution_context.py).
            goal: Optional top-down attention prior. See
                cognition/goal_cell.py — GoalCell.node_relevance(conditions,
                edge_relation) returns a bias in [0.1, 3.0]. Any duck-typed
                object with the same signature works.
        """
        state = flatten_context(state)
        start = time.perf_counter()

        # Phase 1: Score all matching nodes by proximity to threshold
        scored_entries = []
        for nid, node in self._nodes.items():
            match_score = self._compute_match_score(node, state)
            if match_score is not None:
                scored_entries.append((nid, match_score))

        if not scored_entries:
            return {
                "overrides": {},
                "activated_nodes": [],
                "paths": [],
                "formulas_collected": [],
                "attention_weights": {},
                "traversal_time": time.perf_counter() - start,
            }

        # Phase 2: BFS with attention-gated edge weights
        overrides = {}
        activated = set()
        paths = []
        formulas = []
        attention_weights = {}  # node_id → attention weight
        visited_weights: dict[str, float] = {}

        # Sort by tier then match score (lower tier = more authoritative)
        scored_entries.sort(key=lambda x: (self._nodes[x[0]].tier, -x[1]))

        from collections import deque
        queue = deque((nid, score, [nid]) for nid, score in scored_entries)
        # Hard expansion cap: on a dense graph (avg degree = edges/nodes)
        # this weighted BFS re-enqueues every neighbor on every pop, and
        # weight-decay alone bounds it far too late — at ~50k edges the
        # frontier explodes combinatorially (observed: a single traverse
        # hung a stage for minutes). The cap bounds worst-case work while
        # leaving small/normal graphs untouched.
        max_expansions = max(4000, 8 * len(self._nodes))
        expansions = 0

        while queue:
            if expansions >= max_expansions:
                break
            expansions += 1
            current_id, current_weight, path = queue.popleft()

            # Loop decay
            if current_id in visited_weights:
                current_weight *= LOOP_DECAY_FACTOR
            visited_weights[current_id] = current_weight

            if current_weight < LOOP_CUTOFF:
                continue
            if len(path) > max_depth:
                continue

            node = self._nodes.get(current_id)
            if node is None:
                continue

            activated.add(current_id)
            node.activation_count += 1
            node.last_activated = datetime.now(timezone.utc).isoformat()
            self._tag_node_dataset(node)

            # Attention weight for this node
            attention = current_weight * node.confidence
            # LCA-inspired top-down goal prior (2026-07-18). When absent
            # or duck-typed but not implementing node_relevance, this is
            # a no-op — behavior is byte-identical to pre-GoalCell runs.
            if goal is not None:
                try:
                    bias = goal.node_relevance(
                        getattr(node, "conditions", []) or [],
                        edge_relation=None,
                    )
                    attention *= float(bias)
                except Exception:
                    pass
            attention_weights[current_id] = attention

            # Collect overrides (attention-weighted)
            for key, value in node.action_override.items():
                if _is_bound_key(key):
                    # Attention does NOT scale a constraint -- see
                    # _combine_bound. It still decides WHICH nodes are
                    # reached, which is where attention belongs.
                    overrides[key] = _combine_bound(key, overrides.get(key), value)
                elif isinstance(value, (int, float)):
                    if key not in overrides:
                        overrides[key] = value * attention
                    else:
                        # Continuous blending instead of first-wins
                        overrides[key] += value * attention

            # Follow edges with causal gate
            for edge_idx in self._adj.get(current_id, []):
                edge = self._edges[edge_idx]
                target_id = edge.target_id

                # Causal gate: attention = edge_weight * help_ratio
                causal_impact = edge.help_ratio
                gate_value = edge.weight * causal_impact

                if edge.edge_type == EDGE_CONTRADICTS:
                    # Continuous suppression: reduce weight instead of hard block
                    gate_value *= 0.1  # 90% suppression, not 100%

                if edge.edge_type == EDGE_DEPENDS_ON:
                    if target_id not in activated:
                        continue

                if edge.edge_type == EDGE_STRENGTHENS:
                    gate_value *= 1.3  # 30% boost for strengthening edges

                # LCA-inspired goal-relation prior. relation_bias returns
                # 1.0 when goal is None / no prefer_relations set, so this
                # is a no-op for pre-GoalCell runs.
                if goal is not None:
                    try:
                        gate_value *= float(goal.relation_bias(edge.edge_type))
                    except Exception:
                        pass

                propagated_weight = current_weight * gate_value
                edge.times_traversed += 1
                self._tag_edge_usage(edge)

                if edge.formula_id:
                    formulas.append(edge.formula_id)
                    overrides = self._evaluate_edge_formula(
                        edge.formula_id, node, edge, overrides
                    )

                # Enqueue guard: apply the revisit decay the pop WOULD
                # apply and skip weights already below cutoff — otherwise
                # sub-cutoff work floods the queue before being dropped.
                anticipated = propagated_weight
                if target_id in visited_weights:
                    anticipated *= LOOP_DECAY_FACTOR
                if anticipated < LOOP_CUTOFF or len(path) >= max_depth:
                    continue
                queue.append((target_id, propagated_weight, path + [target_id]))

            if len(path) > 1:
                paths.append(path)

        # Normalize overrides by total attention
        total_attention = sum(attention_weights.values()) if attention_weights else 1.0
        if total_attention > 0:
            for key in overrides:
                if isinstance(overrides[key], (int, float)):
                    overrides[key] /= total_attention

        elapsed = time.perf_counter() - start
        return {
            "overrides": overrides,
            "activated_nodes": sorted(activated),
            "paths": paths,
            "formulas_collected": formulas,
            "attention_weights": attention_weights,
            "traversal_time": round(elapsed, 6),
        }

    def _compute_match_score(
        self, node: NetworkNode, state: dict
    ) -> float | None:
        """Compute how well a node's conditions match the state.

        Returns a score in [0, 1] where:
            1.0 = state values far past thresholds (strong match)
            0.0 = state values exactly at thresholds (borderline)
            None = conditions not met

        Used by traverse_with_attention for attention-weighted routing.
        """
        import operator
        ops = {
            "<=": operator.le, "<": operator.lt,
            ">=": operator.ge, ">": operator.gt,
            "==": operator.eq, "!=": operator.ne,
        }
        scores = []
        for cond in node.conditions:
            feature = cond.get("feature", "")
            op_str = cond.get("op", "")
            threshold = cond.get("threshold")
            if threshold is None or op_str not in ops:
                return None
            value = node._resolve(feature, state)
            if value is None:
                return None
            try:
                fval = float(value)
                fthresh = float(threshold)
                if not ops[op_str](fval, fthresh):
                    return None
                # Margin: how far past the threshold
                denom = abs(fthresh) if abs(fthresh) > 1e-6 else 1.0
                margin = abs(fval - fthresh) / denom
                scores.append(min(1.0, margin))
            except (ValueError, TypeError):
                return None
        return float(np.mean(scores)) if scores else 0.5

    # ── Feedback: Strengthen/Weaken Edges ──────────────────────────

    def record_traversal_outcome(
        self,
        activated_nodes: list[str],
        helped: bool,
    ) -> None:
        """Record whether the traversal result helped.

        Strengthens edges along successful paths, weakens failed ones.
        """
        for i, edge in enumerate(self._edges):
            if edge.source_id in activated_nodes and edge.target_id in activated_nodes:
                if helped:
                    edge.times_helped += 1
                    edge.weight = min(0.99, edge.weight + 0.01)
                else:
                    edge.weight = max(0.01, edge.weight - 0.02)

    # ── Edge Decay ─────────────────────────────────────────────────

    def apply_edge_decay(
        self,
        min_traversals: int = 10,
        archive=None,
        dataset: str | None = None,
    ) -> int:
        """Decay edges that consistently fail.

        Weak edges are archived (if archive given) then removed from the
        working graph. Returns the number of edges removed.
        """
        keep: list = []
        removed_edges: list[dict] = []
        for e in self._edges:
            if e.times_traversed < min_traversals or e.help_ratio >= 0.2:
                keep.append(e)
            else:
                removed_edges.append(e.to_dict())
        removed = len(removed_edges)
        if removed > 0:
            if archive is not None:
                archive.archive_graph_edges(
                    removed_edges, reason="edge_decay_low_help_ratio", dataset=dataset,
                )
            # Per-edge events only for small batches; large decays emit one
            # bulk event (see _bulk_log — per-edge appends at 70k+ edges
            # hang the run in VersionStore I/O).
            if removed <= 200:
                for edge_dict in removed_edges:
                    key = f"{edge_dict['source_id']}->{edge_dict['target_id']}:{edge_dict['edge_type']}"
                    self._graph_log(
                        "edge_removed",
                        "edge",
                        key,
                        details={"reason": "edge_decay_low_help_ratio"},
                        snapshot=edge_dict,
                    )
            else:
                self._bulk_log("edge_decay", {
                    "removed": removed,
                    "reason": "edge_decay_low_help_ratio",
                    "archived": archive is not None,
                })
            self._edges = keep
            self._rebuild_adjacency()
        return removed

    def prune_working_graph(
        self,
        archive,
        *,
        max_edges: int = 4000,
        max_nodes: int = 300,
        protect_nodes: set[str] | None = None,
        dataset: str | None = None,
        system_stress: float | None = None,
    ) -> dict:
        """Archive low-utility nodes/edges using metrics, not blind caps.

        Scoring uses confidence, help_ratio, recency (last used), dataset
        domain tags, evidence/activation counts, and predicted_regret.
        Hard caps apply only as a safety ceiling after metric pruning.
        """
        protect_nodes = protect_nodes or set()
        stats = {
            "edges_archived": 0,
            "nodes_archived": 0,
            "edges_decayed": 0,
            "edge_cut": 0.0,
            "node_cut": 0.0,
            "prune_mode": "metric",
        }

        edge_cut, node_cut = self._archive_thresholds(system_stress)
        stats["edge_cut"] = round(edge_cut, 4)
        stats["node_cut"] = round(node_cut, 4)

        decayed = self.apply_edge_decay(
            min_traversals=5, archive=archive, dataset=dataset,
        )
        stats["edges_decayed"] = decayed

        # Metric edge archive: bad help_ratio, stale, wrong domain, low confidence path.
        edge_scores = [
            (self.edge_keep_score(e, dataset=dataset), e)
            for e in self._edges
        ]
        to_drop_edges = [
            e for score, e in edge_scores
            if self._edge_should_archive(e, score, edge_cut)
        ]
        if to_drop_edges:
            drop_ids = {id(e) for e in to_drop_edges}
            dropped_dicts = [e.to_dict() for e in to_drop_edges]
            if archive is not None and dropped_dicts:
                archive.archive_graph_edges(
                    dropped_dicts,
                    reason="metric_low_utility_edge",
                    dataset=dataset,
                )
            # Per-edge events only for small batches (see _bulk_log).
            if len(dropped_dicts) <= 200:
                for edge_dict in dropped_dicts:
                    key = (
                        f"{edge_dict['source_id']}->{edge_dict['target_id']}:"
                        f"{edge_dict['edge_type']}"
                    )
                    self._graph_log(
                        "edge_removed",
                        "edge",
                        key,
                        details={
                            "reason": "metric_low_utility_edge",
                            "keep_score": round(
                                self.edge_keep_score(
                                    NetworkEdge.from_dict(edge_dict), dataset=dataset,
                                ),
                                4,
                            ),
                            "edge_cut": round(edge_cut, 4),
                        },
                        snapshot=edge_dict,
                    )
            else:
                self._bulk_log("edge_prune", {
                    "removed": len(dropped_dicts),
                    "reason": "metric_low_utility_edge",
                    "edge_cut": round(edge_cut, 4),
                    "archived": archive is not None,
                })
            self._edges = [e for e in self._edges if id(e) not in drop_ids]
            stats["edges_archived"] = len(dropped_dicts)
            self._rebuild_adjacency()

        connected: set[str] = set()
        for e in self._edges:
            connected.add(e.source_id)
            connected.add(e.target_id)

        node_edge_count: dict[str, int] = {}
        for e in self._edges:
            node_edge_count[e.source_id] = node_edge_count.get(e.source_id, 0) + 1
            node_edge_count[e.target_id] = node_edge_count.get(e.target_id, 0) + 1

        to_drop_nodes: dict[str, dict] = {}
        for nid, node in list(self._nodes.items()):
            protected = nid in protect_nodes
            score = self.node_keep_score(node, dataset=dataset, protected=protected)
            has_edges = node_edge_count.get(nid, 0) > 0
            if self._node_should_archive(
                node, score, node_cut,
                protected=protected,
                has_edges=has_edges,
            ):
                to_drop_nodes[nid] = node.to_dict()

        if to_drop_nodes:
            if archive is not None:
                archive.archive_graph_nodes(
                    to_drop_nodes,
                    reason="metric_low_utility_node",
                    dataset=dataset,
                )
            for nid, snap in to_drop_nodes.items():
                self._graph_log(
                    "node_removed",
                    "node",
                    nid,
                    details={
                        "reason": "metric_low_utility_node",
                        "keep_score": round(
                            self.node_keep_score(
                                NetworkNode.from_dict(snap), dataset=dataset,
                            ),
                            4,
                        ),
                        "node_cut": round(node_cut, 4),
                    },
                    snapshot=snap,
                )
                self.remove_node(nid, reason="metric_low_utility_node")
            stats["nodes_archived"] = len(to_drop_nodes)
            self._rebuild_adjacency()

        # Safety ceiling only — trim lowest-scored surplus if graph still huge.
        if len(self._edges) > max_edges:
            ranked = sorted(
                self._edges,
                key=lambda e: self.edge_keep_score(e, dataset=dataset),
            )
            drop_n = len(self._edges) - max_edges
            to_drop = ranked[:drop_n]
            drop_ids = {id(e) for e in to_drop}
            dropped_dicts = [e.to_dict() for e in to_drop]
            if archive is not None and dropped_dicts:
                archive.archive_graph_edges(
                    dropped_dicts, reason="safety_edge_ceiling", dataset=dataset,
                )
            # Per-edge events only for small batches (see _bulk_log).
            if len(dropped_dicts) <= 200:
                for edge_dict in dropped_dicts:
                    key = f"{edge_dict['source_id']}->{edge_dict['target_id']}:{edge_dict['edge_type']}"
                    self._graph_log(
                        "edge_removed",
                        "edge",
                        key,
                        details={"reason": "safety_edge_ceiling"},
                        snapshot=edge_dict,
                    )
            else:
                self._bulk_log("edge_prune", {
                    "removed": len(dropped_dicts),
                    "reason": "safety_edge_ceiling",
                    "archived": archive is not None,
                })
            self._edges = [e for e in self._edges if id(e) not in drop_ids]
            stats["edges_archived"] += len(dropped_dicts)
            stats["safety_edge_trim"] = len(dropped_dicts)
            self._rebuild_adjacency()

        connected = set()
        for e in self._edges:
            connected.add(e.source_id)
            connected.add(e.target_id)

        if len(self._nodes) > max_nodes:
            candidates = []
            for nid, node in self._nodes.items():
                if nid in protect_nodes:
                    continue
                score = self.node_keep_score(node, dataset=dataset)
                candidates.append((score, nid, node))
            candidates.sort(key=lambda t: t[0])
            drop_n = min(len(candidates), len(self._nodes) - max_nodes)
            if drop_n > 0:
                drop_nodes = {
                    nid: node.to_dict() for _, nid, node in candidates[:drop_n]
                }
                if archive is not None:
                    archive.archive_graph_nodes(
                        drop_nodes, reason="safety_node_ceiling", dataset=dataset,
                    )
                for nid in drop_nodes:
                    self.remove_node(nid, reason="safety_node_ceiling")
                stats["nodes_archived"] += len(drop_nodes)
                stats["safety_node_trim"] = len(drop_nodes)

        if archive is not None and (stats["edges_archived"] or stats["nodes_archived"]):
            archive.record_graph_prune(
                dataset or "global",
                nodes_archived=stats["nodes_archived"],
                edges_archived=stats["edges_archived"] + stats.get("edges_decayed", 0),
                working_nodes=self.node_count,
                working_edges=self.edge_count,
                reason="metric_working_prune",
            )
            archive.save()
        if stats["edges_archived"] or stats["nodes_archived"] or stats.get("edges_decayed"):
            self._graph_log(
                "graph_pruned",
                "graph",
                "working",
                details=stats,
            )

        return stats

    def consolidate_duplicate_nodes(self, archive=None, dataset: str | None = None) -> int:
        """Merge nodes with identical conditions; archive duplicates."""
        by_sig: dict[str, str] = {}
        merged = 0
        to_archive: dict[str, dict] = {}
        redirect: dict[str, str] = {}

        for nid, node in list(self._nodes.items()):
            sig = json.dumps(node.conditions, sort_keys=True, default=str)
            if sig in by_sig:
                keeper = by_sig[sig]
                keeper_node = self._nodes[keeper]
                keeper_node.evidence_count += node.evidence_count
                keeper_node.confidence = min(
                    0.99,
                    0.6 * keeper_node.confidence + 0.4 * node.confidence,
                )
                keeper_node.activation_count += node.activation_count
                to_archive[nid] = node.to_dict()
                redirect[nid] = keeper
                merged += 1
                # M1: the keeper's canonical fields changed -- emit its
                # post-state so replay tracks the merge.
                self._graph_log(
                    "node_merged",
                    "node",
                    keeper,
                    details={"absorbed": nid},
                )
            else:
                by_sig[sig] = nid

        if not merged:
            return 0

        if archive is not None:
            archive.archive_graph_nodes(
                to_archive, reason="duplicate_node_merge", dataset=dataset,
            )

        for old_id, new_id in redirect.items():
            for edge in self._edges:
                changed = False
                old_key = edge.edge_key
                if edge.source_id == old_id:
                    edge.source_id = new_id
                    changed = True
                if edge.target_id == old_id:
                    edge.target_id = new_id
                    changed = True
                if changed:
                    # M1: an edge's identity (edge_key) changes when it is
                    # redirected -- replay needs a delete of the old key
                    # and an upsert of the new one.
                    self._graph_log(
                        "edge_removed",
                        "edge",
                        old_key,
                        details={"reason": "redirected", "to": edge.edge_key},
                    )
                    self._graph_log(
                        "edge_redirected",
                        "edge",
                        edge.edge_key,
                        details={"from": old_key},
                    )
            self.remove_node(old_id, reason="duplicate_node_merge", merged_into=new_id)

        self._rebuild_adjacency()
        if merged:
            self._graph_log(
                "duplicates_consolidated",
                "graph",
                "working",
                details={"merged_count": merged},
            )
        return merged

    # ── Relationship Discovery ──────────────────────────────────────

    def discover_relationships(
        self,
        journal_entries: list[dict],
        activated_history: list[tuple[int, list[str]]],
        formula_id: str | None = None,
        base_rate: float | None = None,
    ) -> int:
        """Discover edges from co-activation patterns in the journal.

        If node A and node B frequently co-activate:
            - In the same stage → LEADS_TO or STRENGTHENS
            - One helps, the other hurts → CONTRADICTS

        Evidence accumulates in self._co_activation_tally ACROSS calls
        (and, once persisted, across runs) rather than being judged on
        this call's local window alone -- a pair that co-activates only
        1-2 times per call would otherwise never cross the threshold no
        matter how many times that keeps happening across separate runs.

        Args:
            journal_entries: This dataset's journal entries, already
                filtered to the CURRENT fold by the caller (so each
                entry's "stage" field is unique -- two different folds
                can both have a "stage 3", and matching by stage alone
                without fold-filtering would silently pick the wrong
                one).
            activated_history: List of (stage, activated_nodes) tuples
                from recent traversals. Matched against journal_entries
                by stage number, not list position -- the two lists are
                independently windowed/sourced and must not be assumed
                aligned by index.
            formula_id: Optional formula to attach to newly-created edges.

        Returns:
            Number of edges touched (created or strengthened) this call.
        """
        if len(activated_history) < 5:
            return 0

        # Build stage -> regret lookup once, instead of assuming
        # activated_history[i] corresponds to journal_entries[i].
        regret_by_stage: dict[int, float] = {}
        for entry in journal_entries:
            stage = entry.get("stage")
            if stage is not None:
                regret_by_stage[stage] = entry.get("regret", 0.0)

        def _tally_stage(stage_activated: tuple[int, list[str]]) -> tuple[dict, dict]:
            stage, activated = stage_activated
            regret = regret_by_stage.get(stage)
            local_act: dict[tuple, int] = {}
            local_succ: dict[tuple, int] = {}
            for a, b in combinations(activated, 2):
                pair = (a, b) if a < b else (b, a)
                local_act[pair] = local_act.get(pair, 0) + 1
                if regret is not None and regret <= 0:
                    local_succ[pair] = local_succ.get(pair, 0) + 1
            return local_act, local_succ

        co_activation: dict[tuple, int] = {}
        co_success: dict[tuple, int] = {}
        stage_workers = max(1, min(len(activated_history), (os.cpu_count() or 4)))
        if stage_workers > 1:
            with ThreadPoolExecutor(max_workers=stage_workers) as pool:
                tallies = pool.map(_tally_stage, activated_history)
        else:
            tallies = map(_tally_stage, activated_history)
        for local_act, local_succ in tallies:
            for pair, count in local_act.items():
                co_activation[pair] = co_activation.get(pair, 0) + count
                co_success[pair] = co_success.get(pair, 0) + local_succ.get(pair, 0)

        new_edges = 0

        # Accumulate this call's deltas into the persisted cross-run tally.
        for pair, count in co_activation.items():
            key = f"{pair[0]}|{pair[1]}"
            tally = self._co_activation_tally.setdefault(
                key, {"count": 0, "success": 0}
            )
            tally["count"] += count
            tally["success"] += co_success.get(pair, 0)

        # Threshold check reads the ACCUMULATED tally, not the local delta --
        # only pairs touched this call need re-checking.
        #
        # When base_rate is provided (fraction of all stages where regret<=0),
        # classify by lift-over-base-rate rather than absolute success_ratio.
        # Without this, datasets where only ~8% of stages achieve regret<=0
        # mechanically push every pair below the 0.3 "contradicts" threshold
        # regardless of the principles' actual relationship, producing an all-
        # "contradicts" network that tells traversal nothing useful.
        # Lift thresholds: >2x base rate → STRENGTHENS (co-activation doubles
        # success probability), <0.5x → CONTRADICTS (halves it), else LEADS_TO.
        _use_lift = (
            base_rate is not None
            and 0.0 < base_rate < 1.0
        )

        for pair in co_activation:
            key = f"{pair[0]}|{pair[1]}"
            tally = self._co_activation_tally[key]
            count = tally["count"]
            if count >= 3:  # Minimum co-activation threshold
                a, b = pair
                success_ratio = tally["success"] / count

                if _use_lift:
                    lift = success_ratio / base_rate
                    if lift > 2.0:
                        edge_type = EDGE_STRENGTHENS
                        weight = min(0.95, success_ratio)
                    elif lift < 0.5:
                        edge_type = EDGE_CONTRADICTS
                        weight = 1.0 - success_ratio
                    else:
                        edge_type = EDGE_LEADS_TO
                        weight = 0.5
                else:
                    # Absolute fallback (original logic, for balanced datasets)
                    if success_ratio > 0.6:
                        edge_type = EDGE_STRENGTHENS
                        weight = success_ratio
                    elif success_ratio < 0.3:
                        edge_type = EDGE_CONTRADICTS
                        weight = 1.0 - success_ratio
                    else:
                        edge_type = EDGE_LEADS_TO
                        weight = 0.5

                result = self.add_edge(a, b, edge_type, weight=weight, formula_id=formula_id)
                if result is not None:
                    new_edges += 1

        return new_edges

    def _prune_weakest_edges(self, keep: int) -> None:
        """Drop lowest-weight edges when the graph grows too large."""
        if len(self._edges) <= keep:
            return
        ranked = sorted(self._edges, key=lambda e: e.weight)
        drop = len(self._edges) - keep
        for edge in ranked[:drop]:
            self._edges.remove(edge)

    def get_co_activation_bonus(
        self,
        principle_id: str,
        min_co_count: int = 3,
        min_success_rate: float = 0.6,
    ) -> dict[str, float]:
        """Return confidence bonus/penalty weights for co-activation partners.

        When principle A has historically co-activated with principle B and
        those sessions succeeded, B gets a positive bonus (it should be
        boosted when A fires). When B co-activated with A but outcomes were
        poor, B gets a negative penalty (redundancy signal).

        Args:
            principle_id: The principle that just fired.
            min_co_count: Minimum co-activations required to trust the signal.
            min_success_rate: Success rate threshold above which bonus applies.

        Returns:
            Dict mapping partner_id → bonus (positive = boost, negative = penalty).
        """
        bonuses: dict[str, float] = {}
        for key, tally in self._co_activation_tally.items():
            parts = key.split("|")
            if len(parts) != 2:
                continue
            a, b = parts
            if a == principle_id:
                partner = b
            elif b == principle_id:
                partner = a
            else:
                continue
            count = tally.get("count", 0)
            success = tally.get("success", 0)
            if count >= min_co_count:
                rate = success / count
                if rate >= min_success_rate:
                    bonuses[partner] = rate  # strength proportional to success rate
                elif rate < 0.3:
                    bonuses[partner] = -0.2  # redundancy penalty
        return bonuses

    # ── Import from Library ─────────────────────────────────────────

    def import_from_principles(self, principles: list[dict]) -> int:
        """Import existing principles from UIL_Library as nodes.

        Used for bootstrapping — converts the flat principle list
        into graph nodes. Edges are discovered later.

        Returns:
            Number of nodes created.
        """
        count = 0
        for p in principles:
            nid = self.add_node(
                conditions=p.get("conditions", []),
                action_override=p.get("action_override", {}),
                tier=p.get("tier", 3),
                confidence=p.get("confidence", 0.5),
                evidence_count=p.get("evidence_count", 0),
                node_id=p.get("principle_id"),
            )
            if nid:
                # Anchor value for edge-formula evaluation (see
                # _evaluate_edge_formula): what this node's rule was
                # discovered against.
                node = self._nodes.get(nid)
                if node is not None and "predicted_regret" in p:
                    node.metadata["predicted_regret"] = p["predicted_regret"]
                count += 1
        return count

    # ── Diagnostics ─────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def get_summary(self) -> dict:
        """Summary of the network state."""
        edge_types = {}
        for e in self._edges:
            edge_types[e.edge_type] = edge_types.get(e.edge_type, 0) + 1

        tiers = {}
        for n in self._nodes.values():
            t = n.tier
            tiers[t] = tiers.get(t, 0) + 1

        return {
            "nodes": self.node_count,
            "edges": self.edge_count,
            "edge_types": edge_types,
            "node_tiers": tiers,
            "most_activated": self._most_activated_nodes(3),
        }

    def _most_activated_nodes(self, n: int) -> list[dict]:
        """Get the N most frequently activated nodes."""
        sorted_nodes = sorted(
            self._nodes.values(),
            key=lambda node: node.activation_count,
            reverse=True,
        )
        return [
            {"node_id": node.node_id, "activations": node.activation_count}
            for node in sorted_nodes[:n]
        ]

    # ── Serialization ───────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize the entire network to a dict."""
        return {
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
            "edges": [e.to_dict() for e in self._edges],
            "node_counter": self._node_counter,
            "co_activation_tally": self._co_activation_tally,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EditableNetwork":
        """Deserialize network from a dict. Pure deserializer -- never
        touches disk itself (filepath=None), since callers like _load()
        already have their own file open."""
        net = cls(filepath=None)
        for nid, ndata in d.get("nodes", {}).items():
            node = NetworkNode.from_dict(ndata)
            net._nodes[nid] = node
            net._adj[nid] = []
            net._rev_adj[nid] = []
        for edata in d.get("edges", []):
            edge = NetworkEdge.from_dict(edata)
            idx = len(net._edges)
            net._edges.append(edge)
            net._adj.setdefault(edge.source_id, []).append(idx)
            net._rev_adj.setdefault(edge.target_id, []).append(idx)
        net._node_counter = d.get("node_counter", len(net._nodes))
        net._co_activation_tally = d.get("co_activation_tally", {})
        return net

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        """Load nodes/edges/tally from disk, preserving this instance's
        live formula_engine/formula_library references (those are never
        serialized -- see to_dict()/from_dict())."""
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        loaded = EditableNetwork.from_dict(data)
        self._nodes = loaded._nodes
        self._edges = loaded._edges
        self._adj = loaded._adj
        self._rev_adj = loaded._rev_adj
        self._node_counter = loaded._node_counter
        self._co_activation_tally = loaded._co_activation_tally
        # M1: state that predates the event stream (loaded from the
        # legacy JSON) would be invisible to replay. Bootstrap it into
        # the stream ONCE: when the store is empty but the graph is not.
        if (
            self._version_store is not None
            and self._version_store.last_seq == 0
            and (self._nodes or self._edges)
        ):
            for nid in self._nodes:
                self._graph_log("bootstrap", "node", nid,
                                details={"reason": "legacy_json_load"})
            for e in self._edges:
                self._graph_log("bootstrap", "edge", e.edge_key,
                                details={"reason": "legacy_json_load"})

    def _save(self) -> None:
        """Persist nodes/edges/tally to disk (formula_engine/
        formula_library are intentionally excluded -- see to_dict())."""
        if not self.filepath:
            return
        dirpath = os.path.dirname(self.filepath)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        with open(self.filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        if self._changelog is not None:
            self._changelog.save()

    def save(self) -> None:
        """Public hook: persist the current graph. Call this when you
        want network state to survive process exit (e.g. end of fit())."""
        self._save()

    # ── Internal ────────────────────────────────────────────────────

    def _find_similar_node(self, conditions: list[dict]) -> str | None:
        """Find an existing node with similar conditions."""
        for nid, node in self._nodes.items():
            if self._conditions_similar(node.conditions, conditions):
                return nid
        return None

    def _conditions_similar(
        self, conds_a: list[dict], conds_b: list[dict]
    ) -> bool:
        """Check if two condition sets are similar.

        Shared implementation -- see cognition/condition_match.py for why
        this is no longer written out here (bug #17 was the same defect
        in three copies of it).
        """
        return condition_sets_similar(conds_a, conds_b)

    def _rebuild_adjacency(self) -> None:
        """Rebuild adjacency lists from edge list."""
        self._adj = {nid: [] for nid in self._nodes}
        self._rev_adj = {nid: [] for nid in self._nodes}
        for i, e in enumerate(self._edges):
            self._adj.setdefault(e.source_id, []).append(i)
            self._rev_adj.setdefault(e.target_id, []).append(i)
