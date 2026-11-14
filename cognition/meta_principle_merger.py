"""Meta-Principle Merging — Community detection and knowledge compression.

V21.2 (Part 18): When the knowledge base grows to thousands of principles,
many describe the same underlying concept (e.g., 200 rules about "fragile
datasets"). This module clusters similar principles into meta-principles
using structural similarity and graph community detection.

Key operations:
    1. Structural similarity matrix (from StructuralEncoder)
    2. Community detection (greedy modularity on similarity graph)
    3. Meta-principle synthesis (merge cluster → one abstract rule)
    4. Original rules preserved as children of the meta-principle
"""

from __future__ import annotations
import json
import os
import numpy as np
from datetime import datetime, timezone

from figpso_uil.cognition.paths import results_path


class MetaPrinciple:
    """A higher-level abstraction over a cluster of similar principles.

    Contains:
        - summary: aggregated conditions/actions from the cluster
        - children: list of original principle_ids in this cluster
        - confidence: weighted average of children's confidences
        - support: how many principles contribute to this meta-principle
    """

    def __init__(
        self,
        meta_id: str,
        context_labels: list[str],
        action_label: str,
        confidence: float,
        children: list[str],
        human_text: str = "",
    ):
        self.meta_id = meta_id
        self.context_labels = context_labels
        self.action_label = action_label
        self.confidence = confidence
        self.children = children  # principle_ids
        self.support = len(children)
        self.human_text = human_text
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.times_applied = 0
        self.times_helped = 0

    @property
    def help_ratio(self) -> float:
        if self.times_applied == 0:
            return 0.5
        return self.times_helped / self.times_applied

    def to_dict(self) -> dict:
        return {
            "meta_id": self.meta_id,
            "context_labels": self.context_labels,
            "action_label": self.action_label,
            "confidence": self.confidence,
            "children": self.children,
            "support": self.support,
            "human_text": self.human_text,
            "created_at": self.created_at,
            "times_applied": self.times_applied,
            "times_helped": self.times_helped,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MetaPrinciple":
        mp = cls(
            meta_id=d["meta_id"],
            context_labels=d.get("context_labels", []),
            action_label=d.get("action_label", ""),
            confidence=d.get("confidence", 0.5),
            children=d.get("children", []),
            human_text=d.get("human_text", ""),
        )
        mp.support = d.get("support", len(mp.children))
        mp.created_at = d.get("created_at", mp.created_at)
        mp.times_applied = d.get("times_applied", 0)
        mp.times_helped = d.get("times_helped", 0)
        return mp


class MetaPrincipleMerger:
    """Clusters similar principles into meta-principles.

    Uses greedy modularity community detection on the structural
    similarity graph. Produces MetaPrinciple objects that summarize
    each cluster.
    """

    # Minimum similarity to form an edge in the similarity graph
    SIMILARITY_THRESHOLD = 0.6
    # Minimum cluster size to form a meta-principle
    MIN_CLUSTER_SIZE = 3
    # Maximum number of meta-principles to create per merge pass
    MAX_META_PRINCIPLES = 50

    def __init__(self, filepath: str = results_path("uil_meta_principles.json")):
        self._meta_principles: list[MetaPrinciple] = []
        self._meta_counter = 0
        self.filepath = filepath
        self._load()

    # ── Community Detection ─────────────────────────────────────────

    def detect_communities(
        self,
        structural_principles: list,
    ) -> list[list[int]]:
        """Detect communities of similar principles using greedy modularity.

        Args:
            structural_principles: List of StructuralPrinciple objects.

        Returns:
            List of clusters (each cluster is a list of indices).
        """
        n = len(structural_principles)
        if n < self.MIN_CLUSTER_SIZE:
            return []

        # Step 1: Build similarity matrix
        sim_matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                sim = structural_principles[i].structural_similarity(
                    structural_principles[j]
                )
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        # Step 2: Build adjacency (threshold similarity)
        adjacency: dict[int, set[int]] = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= self.SIMILARITY_THRESHOLD:
                    adjacency[i].add(j)
                    adjacency[j].add(i)

        # Step 3: Greedy community detection (label propagation)
        labels = list(range(n))  # Start: each node is its own community
        changed = True
        max_iterations = 20

        for _ in range(max_iterations):
            if not changed:
                break
            changed = False
            # Randomize order to avoid bias
            order = np.random.permutation(n)

            for i in order:
                neighbors = adjacency[int(i)]
                if not neighbors:
                    continue

                # Count neighbor labels, weighted by similarity
                label_scores: dict[int, float] = {}
                for j in neighbors:
                    lbl = labels[j]
                    w = sim_matrix[int(i), j]
                    label_scores[lbl] = label_scores.get(lbl, 0.0) + w

                # Take the label with highest weighted score
                best_label = max(label_scores, key=label_scores.get)
                if labels[int(i)] != best_label:
                    labels[int(i)] = best_label
                    changed = True

        # Step 4: Group indices by label
        clusters_dict: dict[int, list[int]] = {}
        for i, lbl in enumerate(labels):
            clusters_dict.setdefault(lbl, []).append(i)

        # Filter: only keep clusters with MIN_CLUSTER_SIZE+
        clusters = [
            c for c in clusters_dict.values()
            if len(c) >= self.MIN_CLUSTER_SIZE
        ]

        # Sort clusters by size (largest first)
        clusters.sort(key=len, reverse=True)
        return clusters[:self.MAX_META_PRINCIPLES]

    # ── Meta-Principle Synthesis ────────────────────────────────────

    def merge_cluster(
        self,
        cluster_indices: list[int],
        structural_principles: list,
    ) -> MetaPrinciple:
        """Synthesize a meta-principle from a cluster of similar principles.

        Takes the most common context labels, the most common action,
        and the weighted-average confidence.
        """
        self._meta_counter += 1
        meta_id = f"MP-{self._meta_counter:04d}"

        # Collect all context labels, actions, confidences
        all_contexts: dict[str, int] = {}
        all_actions: dict[str, int] = {}
        confidences = []
        children = []

        for idx in cluster_indices:
            sp = structural_principles[idx]
            children.append(sp.principle_id)
            confidences.append(sp.confidence)

            for label in sp.context_labels:
                all_contexts[label] = all_contexts.get(label, 0) + 1
            all_actions[sp.action_label] = all_actions.get(
                sp.action_label, 0
            ) + 1

        # Most common context labels (appearing in >50% of cluster)
        threshold = len(cluster_indices) * 0.5
        common_contexts = [
            label for label, count in all_contexts.items()
            if count >= threshold
        ]

        # Most common action
        common_action = max(all_actions, key=all_actions.get) if all_actions else ""

        # Weighted average confidence
        avg_confidence = float(np.mean(confidences)) if confidences else 0.5

        # Generate human text
        ctx_text = " + ".join(l.replace("_", " ").title() for l in common_contexts) or "General"
        action_text = common_action.replace("_", " ").title()
        human_text = f"Meta-Principle: When [{ctx_text}], {action_text} (from {len(children)} rules)"

        meta = MetaPrinciple(
            meta_id=meta_id,
            context_labels=common_contexts,
            action_label=common_action,
            confidence=avg_confidence,
            children=children,
            human_text=human_text,
        )
        self._meta_principles.append(meta)
        return meta

    # ── Full Merge Pipeline ─────────────────────────────────────────

    def merge(
        self,
        structural_principles: list,
    ) -> list[MetaPrinciple]:
        """Full merge pipeline: detect communities → synthesize meta-principles.

        Args:
            structural_principles: List of StructuralPrinciple objects.

        Returns:
            List of newly created MetaPrinciples.
        """
        clusters = self.detect_communities(structural_principles)
        new_metas = []
        for cluster in clusters:
            meta = self.merge_cluster(cluster, structural_principles)
            new_metas.append(meta)
        if new_metas:
            self._save()
        return new_metas

    # ── Persistence ────────────────────────────────────────────────

    def _load(self) -> None:
        if self.filepath is None:
            return
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
            max_counter = 0
            for d in data:
                mp = MetaPrinciple.from_dict(d)
                self._meta_principles.append(mp)
                try:
                    num = int(mp.meta_id.split("-")[-1])
                    max_counter = max(max_counter, num)
                except (ValueError, IndexError):
                    pass
            self._meta_counter = max_counter
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save(self) -> None:
        if self.filepath is None:
            return
        dirpath = os.path.dirname(self.filepath)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        with open(self.filepath, "w") as f:
            json.dump(
                [mp.to_dict() for mp in self._meta_principles],
                f, indent=2, default=str,
            )

    # ── Diagnostics ────────────────────────────────────────────────

    @property
    def meta_count(self) -> int:
        return len(self._meta_principles)

    def get_all_meta_principles(self) -> list[MetaPrinciple]:
        return list(self._meta_principles)

    def get_summary(self) -> dict:
        return {
            "meta_principles": self.meta_count,
            "total_children": sum(
                m.support for m in self._meta_principles
            ),
            "avg_cluster_size": (
                np.mean([m.support for m in self._meta_principles])
                if self._meta_principles else 0.0
            ),
        }
