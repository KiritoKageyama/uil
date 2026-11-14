"""Structural Principle Representation — Machine-readable concept vocabulary.

V21.1 (Part 17): Store principles as structured concepts, not English text.
Human-readable text exists for explanation only. All reasoning operates
on structured representations.

Concept Vocabulary:
    - Features: n_samples, n_features, sparsity, entropy, feat_ratio, etc.
    - Actions: max_penalty, classifier, tree_depth, regularization, etc.
    - Contexts: dataset_size_class, noise_level, imbalance_level, etc.
    - Outcomes: regret, stress, accuracy, recovery_cost, etc.

Each principle becomes a structured triple:
    (CONTEXT, CONDITION, ACTION) → EXPECTED_OUTCOME
"""

from __future__ import annotations
from datetime import datetime, timezone


# ── Concept Vocabulary ──────────────────────────────────────────────

CONTEXT_VOCABULARY = {
    # Dataset size classes
    "tiny": {"feature": "n_samples", "op": "<", "threshold": 50},
    "small": {"feature": "n_samples", "op": "<", "threshold": 200},
    "medium": {"feature": "n_samples", "op": "<=", "threshold": 1000},
    "large": {"feature": "n_samples", "op": ">", "threshold": 1000},
    # Dimensionality
    "low_dim": {"feature": "n_features", "op": "<", "threshold": 20},
    "mid_dim": {"feature": "n_features", "op": "<=", "threshold": 100},
    "high_dim": {"feature": "n_features", "op": ">", "threshold": 100},
    # Feature ratio classes
    "low_ratio": {"feature": "feat_ratio", "op": "<", "threshold": 0.3},
    "moderate_ratio": {"feature": "feat_ratio", "op": "<=", "threshold": 1.0},
    "high_ratio": {"feature": "feat_ratio", "op": ">", "threshold": 1.0},
    "extreme_ratio": {"feature": "feat_ratio", "op": ">", "threshold": 5.0},
    # Sparsity classes
    "dense": {"feature": "sparsity", "op": "<", "threshold": 0.1},
    "moderate_sparse": {"feature": "sparsity", "op": "<=", "threshold": 0.5},
    "highly_sparse": {"feature": "sparsity", "op": ">", "threshold": 0.5},
    # Entropy classes
    "low_entropy": {"feature": "entropy", "op": "<", "threshold": 0.3},
    "moderate_entropy": {"feature": "entropy", "op": "<=", "threshold": 0.7},
    "high_entropy": {"feature": "entropy", "op": ">", "threshold": 0.7},
}

ACTION_VOCABULARY = {
    "reduce_penalty": {"action": "max_penalty", "direction": "decrease"},
    "increase_penalty": {"action": "max_penalty", "direction": "increase"},
    "reduce_complexity": {"action": "tree_depth", "direction": "decrease"},
    "increase_complexity": {"action": "tree_depth", "direction": "increase"},
    "prefer_simple_model": {"action": "classifier", "value": "knn"},
    "prefer_ensemble": {"action": "classifier", "value": "rf"},
    "prefer_boosting": {"action": "classifier", "value": "xgboost"},
    "increase_regularization": {"action": "regularization", "direction": "increase"},
    "reduce_regularization": {"action": "regularization", "direction": "decrease"},
    "aggressive_search": {"action": "search_scale", "direction": "increase"},
    "conservative_search": {"action": "search_scale", "direction": "decrease"},
}

OUTCOME_VOCABULARY = {
    "low_regret": {"metric": "regret", "direction": "low"},
    "high_regret": {"metric": "regret", "direction": "high"},
    "low_stress": {"metric": "stress", "direction": "low"},
    "high_stress": {"metric": "stress", "direction": "high"},
    "good_accuracy": {"metric": "accuracy", "direction": "high"},
    "poor_accuracy": {"metric": "accuracy", "direction": "low"},
    "fast_convergence": {"metric": "convergence_speed", "direction": "high"},
    "slow_convergence": {"metric": "convergence_speed", "direction": "low"},
}


# ── Structural Principle ────────────────────────────────────────────

class StructuralPrinciple:
    """A principle represented as structured concepts.

    Instead of "Small datasets should avoid deep trees", this stores:
        context: [small]
        conditions: [{feature: n_samples, op: <, threshold: 200}]
        action: reduce_complexity
        expected_outcome: low_regret
        human_text: "Small datasets should avoid deep trees"

    The human_text is for display only. Reasoning uses the structured fields.
    """

    def __init__(
        self,
        principle_id: str,
        context_labels: list[str],
        conditions: list[dict],
        action_label: str,
        action_override: dict,
        expected_outcome: str | None = None,
        human_text: str = "",
        confidence: float = 0.5,
        tier: int = 3,
    ):
        self.principle_id = principle_id
        self.context_labels = context_labels      # e.g., ["small", "high_entropy"]
        self.conditions = conditions               # Raw conditions from DT
        self.action_label = action_label           # e.g., "reduce_penalty"
        self.action_override = action_override     # Raw override dict
        self.expected_outcome = expected_outcome   # e.g., "low_regret"
        self.human_text = human_text               # Display only
        self.confidence = confidence
        self.tier = tier
        self.created_at = datetime.now(timezone.utc).isoformat()

    def structural_similarity(self, other: "StructuralPrinciple") -> float:
        """Compute structural similarity to another principle.

        Measures overlap in context, conditions, and actions.
        Returns a score in [0, 1].
        """
        # Context overlap: Jaccard similarity
        ctx_a = set(self.context_labels)
        ctx_b = set(other.context_labels)
        ctx_union = ctx_a | ctx_b
        ctx_sim = len(ctx_a & ctx_b) / len(ctx_union) if ctx_union else 0.0

        # Action similarity: exact match = 1, else 0
        action_sim = 1.0 if self.action_label == other.action_label else 0.0

        # Condition similarity: feature overlap
        feats_a = {c.get("feature", "") for c in self.conditions}
        feats_b = {c.get("feature", "") for c in other.conditions}
        feat_union = feats_a | feats_b
        cond_sim = len(feats_a & feats_b) / len(feat_union) if feat_union else 0.0

        # Weighted combination
        return 0.35 * ctx_sim + 0.35 * action_sim + 0.30 * cond_sim

    def to_dict(self) -> dict:
        return {
            "principle_id": self.principle_id,
            "context_labels": self.context_labels,
            "conditions": self.conditions,
            "action_label": self.action_label,
            "action_override": self.action_override,
            "expected_outcome": self.expected_outcome,
            "human_text": self.human_text,
            "confidence": self.confidence,
            "tier": self.tier,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StructuralPrinciple":
        sp = cls(
            principle_id=d["principle_id"],
            context_labels=d.get("context_labels", []),
            conditions=d.get("conditions", []),
            action_label=d.get("action_label", ""),
            action_override=d.get("action_override", {}),
            expected_outcome=d.get("expected_outcome"),
            human_text=d.get("human_text", ""),
            confidence=d.get("confidence", 0.5),
            tier=d.get("tier", 3),
        )
        sp.created_at = d.get("created_at", sp.created_at)
        return sp


# ── Structural Encoder ──────────────────────────────────────────────

class StructuralEncoder:
    """Converts raw DT-mined principles into StructuralPrinciples.

    Takes the existing principle dict format (conditions + action_override)
    and enriches it with context labels, action labels, and expected outcomes
    from the concept vocabulary.
    """

    def encode(self, principle: dict) -> StructuralPrinciple:
        """Convert a raw principle dict to a StructuralPrinciple.

        Args:
            principle: Raw dict with conditions, action_override, etc.

        Returns:
            A fully encoded StructuralPrinciple.
        """
        conditions = principle.get("conditions", [])
        action_override = principle.get("action_override", {})

        # Infer context labels from conditions
        context_labels = self._infer_context(conditions)

        # Infer action label from overrides
        action_label = self._infer_action(action_override)

        # Generate human-readable text
        human_text = self._generate_human_text(context_labels, action_label)

        return StructuralPrinciple(
            principle_id=principle.get("principle_id", ""),
            context_labels=context_labels,
            conditions=conditions,
            action_label=action_label,
            action_override=action_override,
            expected_outcome=None,
            human_text=human_text,
            confidence=principle.get("confidence", 0.5),
            tier=principle.get("tier", 3),
        )

    def _infer_context(self, conditions: list[dict]) -> list[str]:
        """Infer context labels from raw conditions.

        Matches condition features/thresholds to the concept vocabulary.
        """
        labels = []
        for cond in conditions:
            feature = cond.get("feature", "")
            op = cond.get("op", "")
            threshold = cond.get("threshold", 0)

            # Strip prefix (e.g., "state.n_samples" → "n_samples")
            short_feature = feature.rsplit(".", 1)[-1] if "." in feature else feature

            # Search vocabulary for matching concept
            best_match = None
            best_distance = float("inf")

            for label, vocab_entry in CONTEXT_VOCABULARY.items():
                if vocab_entry["feature"] == short_feature:
                    # Same feature — check if threshold/op is compatible
                    v_thresh = vocab_entry["threshold"]
                    distance = abs(threshold - v_thresh)
                    if distance < best_distance:
                        best_distance = distance
                        best_match = label

            if best_match:
                labels.append(best_match)

        return labels

    def _infer_action(self, action_override: dict) -> str:
        """Infer action label from raw overrides."""
        for label, vocab_entry in ACTION_VOCABULARY.items():
            action_key = vocab_entry.get("action", "")
            direction = vocab_entry.get("direction", "")
            value = vocab_entry.get("value")

            if action_key in action_override:
                override_val = action_override[action_key]

                if value is not None:
                    # Categorical match
                    if str(override_val) == str(value):
                        return label
                elif direction == "decrease":
                    if isinstance(override_val, (int, float)) and override_val < 0:
                        return label
                elif direction == "increase":
                    if isinstance(override_val, (int, float)) and override_val > 0:
                        return label

        # Fallback: use first override key as label
        if action_override:
            key = list(action_override.keys())[0]
            return f"modify_{key}"
        return "unknown_action"

    def _generate_human_text(
        self, context_labels: list[str], action_label: str
    ) -> str:
        """Generate a human-readable description from structural labels."""
        if not context_labels:
            ctx = "General"
        else:
            ctx = " + ".join(label.replace("_", " ").title()
                             for label in context_labels)

        action = action_label.replace("_", " ").title()
        return f"When [{ctx}]: {action}"

    def batch_encode(self, principles: list[dict]) -> list[StructuralPrinciple]:
        """Encode multiple principles at once."""
        return [self.encode(p) for p in principles]
