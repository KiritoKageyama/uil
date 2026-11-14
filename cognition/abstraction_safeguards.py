"""Abstraction Safeguards — Validation gates before knowledge promotion.

V21.3 (Part 19): Before a meta-principle is accepted, verify that:
    1. Sufficient evidence exists (enough children with enough data)
    2. Contradictions are minimal (cluster is coherent)
    3. Confidence is high (weighted average above threshold)
    4. Exceptions are understood (known failure cases documented)

Meta-principles that pass all gates get promoted.
Those that fail get flagged for investigation or rejected.
Promoted abstractions are continuously monitored and can be
revoked if new evidence contradicts them.
"""

from __future__ import annotations
import numpy as np


class AbstractionSafeguards:
    """Validation gates for meta-principle promotion.

    Prevents incorrect generalizations from spreading through
    the knowledge base. Every abstraction must pass multiple
    evidence checks before it becomes part of reasoning.
    """

    # Gate thresholds
    MIN_SUPPORT = 3              # Minimum children in cluster
    MIN_CONFIDENCE = 0.55        # Average confidence of children
    MAX_CONTRADICTION_RATE = 0.2 # At most 20% of pairs contradict
    MIN_EVIDENCE_COUNT = 10      # Total times_applied across children
    MIN_HELP_RATIO = 0.5         # At least 50% of applications helped

    def __init__(self):
        self._promotion_log: list[dict] = []
        self._rejection_log: list[dict] = []

    # ── Gate Checks ─────────────────────────────────────────────────

    def check_promotion(
        self,
        meta_principle,
        structural_principles: list,
        contradiction_detector=None,
    ) -> dict:
        """Run all safeguard gates on a meta-principle.

        Args:
            meta_principle: MetaPrinciple object.
            structural_principles: All StructuralPrinciple objects
                (needed for contradiction checking).
            contradiction_detector: Optional ContradictionDetector instance.

        Returns:
            {
                "approved": bool,
                "gates": {gate_name: {"passed": bool, "value": ..., "threshold": ...}},
                "reason": str (if rejected),
            }
        """
        gates = {}

        # Gate 1: Minimum support
        gates["min_support"] = {
            "passed": meta_principle.support >= self.MIN_SUPPORT,
            "value": meta_principle.support,
            "threshold": self.MIN_SUPPORT,
        }

        # Gate 2: Minimum confidence
        gates["min_confidence"] = {
            "passed": meta_principle.confidence >= self.MIN_CONFIDENCE,
            "value": round(meta_principle.confidence, 3),
            "threshold": self.MIN_CONFIDENCE,
        }

        # Gate 3: Contradiction rate (within cluster)
        contradiction_rate = self._compute_contradiction_rate(
            meta_principle, structural_principles, contradiction_detector
        )
        gates["max_contradictions"] = {
            "passed": contradiction_rate <= self.MAX_CONTRADICTION_RATE,
            "value": round(contradiction_rate, 3),
            "threshold": self.MAX_CONTRADICTION_RATE,
        }

        # Gate 4: Evidence count
        total_applied = meta_principle.times_applied
        gates["min_evidence"] = {
            "passed": total_applied >= self.MIN_EVIDENCE_COUNT,
            "value": total_applied,
            "threshold": self.MIN_EVIDENCE_COUNT,
        }

        # Gate 5: Help ratio
        help_ratio = meta_principle.help_ratio
        # Relax help ratio gate if not enough applications yet
        if total_applied < self.MIN_EVIDENCE_COUNT:
            gates["min_help_ratio"] = {
                "passed": True,  # Insufficient data — pass by default
                "value": round(help_ratio, 3),
                "threshold": self.MIN_HELP_RATIO,
                "note": "insufficient_data",
            }
        else:
            gates["min_help_ratio"] = {
                "passed": help_ratio >= self.MIN_HELP_RATIO,
                "value": round(help_ratio, 3),
                "threshold": self.MIN_HELP_RATIO,
            }

        # Overall decision: all gates must pass
        all_passed = all(g["passed"] for g in gates.values())
        failed_gates = [name for name, g in gates.items() if not g["passed"]]

        result = {
            "approved": all_passed,
            "gates": gates,
            "reason": "" if all_passed else f"Failed gates: {', '.join(failed_gates)}",
        }

        # Log the decision
        log_entry = {
            "meta_id": meta_principle.meta_id,
            "approved": all_passed,
            "gates_summary": {
                name: g["passed"] for name, g in gates.items()
            },
        }
        if all_passed:
            self._promotion_log.append(log_entry)
        else:
            log_entry["reason"] = result["reason"]
            self._rejection_log.append(log_entry)

        return result

    # ── Continuous Monitoring ───────────────────────────────────────

    def monitor_meta_principle(
        self,
        meta_principle,
        recent_applications: list[dict],
    ) -> dict:
        """Continuously monitor a promoted meta-principle for degradation.

        Checks if recent performance contradicts the abstraction.

        Args:
            meta_principle: MetaPrinciple object.
            recent_applications: Recent application records
                [{applied: bool, helped: bool}, ...]

        Returns:
            {"status": "healthy" | "degrading" | "revoke", "details": ...}
        """
        if not recent_applications:
            return {"status": "healthy", "details": "no_recent_data"}

        recent_helped = sum(
            1 for app in recent_applications if app.get("helped", False)
        )
        recent_total = len(recent_applications)
        recent_ratio = recent_helped / recent_total

        if recent_ratio >= 0.6:
            status = "healthy"
        elif recent_ratio >= 0.4:
            status = "degrading"
        else:
            status = "revoke"

        return {
            "status": status,
            "recent_help_ratio": round(recent_ratio, 3),
            "recent_total": recent_total,
            "meta_id": meta_principle.meta_id,
        }

    # ── Contradiction Rate ──────────────────────────────────────────

    def _compute_contradiction_rate(
        self,
        meta_principle,
        structural_principles: list,
        contradiction_detector,
    ) -> float:
        """Compute the contradiction rate within a cluster.

        Measures how many pairs of children contradict each other.
        """
        if contradiction_detector is None:
            return 0.0

        children_ids = set(meta_principle.children)
        child_principles = [
            sp for sp in structural_principles
            if hasattr(sp, "principle_id") and sp.principle_id in children_ids
        ]

        if len(child_principles) < 2:
            return 0.0

        # Convert to dicts for the contradiction detector
        child_dicts = []
        for sp in child_principles:
            child_dicts.append({
                "principle_id": sp.principle_id,
                "conditions": sp.conditions,
                "action_override": sp.action_override,
                "confidence": sp.confidence,
            })

        contradictions = contradiction_detector.detect_contradictions(child_dicts)
        n_pairs = len(child_dicts) * (len(child_dicts) - 1) / 2
        if n_pairs == 0:
            return 0.0

        return len(contradictions) / n_pairs

    # ── Diagnostics ────────────────────────────────────────────────

    def get_summary(self) -> dict:
        return {
            "promoted": len(self._promotion_log),
            "rejected": len(self._rejection_log),
            "rejection_reasons": self._top_rejection_reasons(),
        }

    def _top_rejection_reasons(self) -> dict:
        reasons: dict[str, int] = {}
        for entry in self._rejection_log:
            for gate, passed in entry.get("gates_summary", {}).items():
                if not passed:
                    reasons[gate] = reasons.get(gate, 0) + 1
        return reasons
