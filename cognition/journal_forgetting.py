"""Journal Forgetting — Weighted recency for mining.

V20+.7: After 10,000 entries, early entries from when the system was
bad pollute the mining. This module provides exponential decay weights
and sliding window support so the Miner prioritizes recent experience.

Design: Exponential decay + sliding window (combined approach).
Old entries are weighted less during mining, not deleted.
Principle versioning is handled by the hierarchy's confidence tracking.
"""

from __future__ import annotations
import numpy as np
from figpso_uil.cognition.condition_match import thresholds_match


class JournalForgetting:
    """Manages recency weighting for journal entries during mining.

    Two mechanisms:
        1. Exponential decay: older entries get lower weight
        2. Sliding window: hard cap on entries fed to the Miner

    Both preserve ALL journal data for auditing. Only the mining
    step sees weighted/windowed data.
    """

    # Default parameters
    DEFAULT_HALF_LIFE = 100      # After 100 entries, weight = 0.5
    DEFAULT_WINDOW_SIZE = 500    # Only mine the last 500 entries
    MIN_WEIGHT = 0.01            # Below this, entries are effectively forgotten

    def __init__(
        self,
        half_life: int = DEFAULT_HALF_LIFE,
        window_size: int = DEFAULT_WINDOW_SIZE,
    ):
        """
        Args:
            half_life: Number of entries for weight to decay to 0.5.
            window_size: Maximum entries to feed to the Miner.
        """
        self.half_life = max(10, half_life)
        self.window_size = max(50, window_size)
        self._decay_constant = np.log(2) / self.half_life

    # ── Sliding Window ──────────────────────────────────────────────

    def apply_window(self, entries: list[dict]) -> list[dict]:
        """Apply sliding window — keep only the most recent entries.

        Returns:
            The last `window_size` entries.
        """
        if len(entries) <= self.window_size:
            return entries
        return entries[-self.window_size:]

    # ── Exponential Decay Weights ───────────────────────────────────

    def compute_weights(self, entries: list[dict]) -> np.ndarray:
        """Compute exponential decay weights for each entry.

        Most recent entry gets weight 1.0.
        Entry N steps ago gets weight exp(-λ * N).

        Returns:
            Array of weights, same length as entries, in [MIN_WEIGHT, 1.0].
        """
        n = len(entries)
        if n == 0:
            return np.array([])

        # Recency: most recent = 0 age, oldest = n-1 age
        ages = np.arange(n - 1, -1, -1, dtype=np.float64)
        weights = np.exp(-self._decay_constant * ages)
        weights = np.clip(weights, self.MIN_WEIGHT, 1.0)
        return weights

    # ── Weighted Sampling ───────────────────────────────────────────

    def weighted_sample(
        self,
        entries: list[dict],
        n_samples: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[dict]:
        """Sample entries with probability proportional to recency weight.

        This is the primary interface for the Miner. Instead of using
        all entries equally, it samples with recency bias.

        Args:
            entries: All journal entries (pre-windowed or full).
            n_samples: How many to sample. Default: min(len(entries), window_size).
            rng: Random generator for reproducibility.

        Returns:
            Sampled entries (may contain duplicates — this is intentional,
            as recent entries should appear more often in the training data).
        """
        if not entries:
            return []

        if rng is None:
            rng = np.random.default_rng(42)

        weights = self.compute_weights(entries)
        probabilities = weights / weights.sum()

        if n_samples is None:
            n_samples = min(len(entries), self.window_size)
        n_samples = min(n_samples, len(entries))

        indices = rng.choice(len(entries), size=n_samples, replace=True, p=probabilities)
        return [entries[i] for i in indices]

    # ── Combined: Window + Weight ───────────────────────────────────

    def prepare_for_mining(
        self, entries: list[dict]
    ) -> tuple[list[dict], np.ndarray]:
        """Apply both window and weights — the full forgetting pipeline.

        Returns:
            (windowed_entries, weights) — ready for the Miner to use.
        """
        windowed = self.apply_window(entries)
        weights = self.compute_weights(windowed)
        return windowed, weights

    # ── Diagnostics ────────────────────────────────────────────────

    def get_effective_memory_span(self, total_entries: int) -> dict:
        """How much of the journal is effectively remembered?"""
        if total_entries == 0:
            return {"total": 0, "windowed": 0, "effective_at_50pct": 0}

        windowed = min(total_entries, self.window_size)
        # Entry count at which weight = 0.5
        half_life_entries = self.half_life
        # Entry count at which weight = MIN_WEIGHT
        forget_horizon = int(np.log(1 / self.MIN_WEIGHT) / self._decay_constant)

        return {
            "total_entries": total_entries,
            "windowed": windowed,
            "half_life": half_life_entries,
            "forget_horizon": forget_horizon,
            "decay_constant": round(self._decay_constant, 6),
        }


# ── Part 12 Gap: Principle Versioning ────────────────────────────────

class PrincipleVersioning:
    """Track principle evolution through versions (v1 → v2 → v3).

    When a principle is superseded, the old version is preserved
    in history. The system always uses the latest version but can
    review evolution history for auditing and learning.
    """

    def __init__(self):
        # {principle_id: [v1_dict, v2_dict, ...]}
        self._history: dict[str, list[dict]] = {}

    def record_version(self, principle: dict) -> int:
        """Record a new version of a principle.

        Returns the version number assigned.
        """
        pid = principle.get("principle_id", "")
        if not pid:
            return 0

        if pid not in self._history:
            self._history[pid] = []

        version = len(self._history[pid]) + 1
        snapshot = {
            "version": version,
            "confidence": principle.get("confidence", 0.5),
            "conditions": principle.get("conditions", []),
            "action_override": principle.get("action_override", {}),
            "tier": principle.get("tier", 3),
            "times_applied": principle.get("times_applied", 0),
            "times_helped": principle.get("times_helped", 0),
        }
        self._history[pid].append(snapshot)
        return version

    def get_history(self, principle_id: str) -> list[dict]:
        """Get all versions of a principle."""
        return self._history.get(principle_id, [])

    def get_latest_version(self, principle_id: str) -> int:
        """Get the latest version number."""
        history = self._history.get(principle_id, [])
        return len(history) if history else 0

    def has_regressed(self, principle_id: str) -> bool:
        """Check if a principle's confidence has decreased over versions.

        Returns True if the latest version is weaker than the peak.
        """
        history = self._history.get(principle_id, [])
        if len(history) < 2:
            return False
        confidences = [v["confidence"] for v in history]
        return confidences[-1] < max(confidences[:-1])

    def get_evolution_summary(self) -> dict:
        """Summary of all principle evolution."""
        total = len(self._history)
        multi_version = sum(
            1 for h in self._history.values() if len(h) > 1
        )
        regressed = sum(
            1 for pid in self._history if self.has_regressed(pid)
        )
        return {
            "total_tracked": total,
            "multi_version": multi_version,
            "regressed": regressed,
        }


# ── Part 12 Gap: Contradiction Detection ─────────────────────────────

class ContradictionDetector:
    """Detect conflicting principles that need investigation.

    Two principles contradict if they:
        1. Share similar conditions (same features, same thresholds)
        2. Recommend opposite actions (e.g., increase vs decrease penalty)

    Contradictions trigger investigation rather than silent coexistence.
    """

    # How close thresholds must be to count as "same condition"
    THRESHOLD_TOLERANCE = 0.15  # 15% relative difference

    def detect_contradictions(
        self, principles: list[dict]
    ) -> list[dict]:
        """Find pairs of principles that contradict each other.

        Returns:
            List of contradiction records:
            {"principle_a", "principle_b", "shared_conditions",
             "conflicting_actions", "severity"}
        """
        contradictions = []
        n = len(principles)

        for i in range(n):
            for j in range(i + 1, n):
                conflict = self._check_pair(principles[i], principles[j])
                if conflict is not None:
                    contradictions.append(conflict)

        # Sort by severity (worst first)
        contradictions.sort(key=lambda c: c["severity"], reverse=True)
        return contradictions

    def _check_pair(self, a: dict, b: dict) -> dict | None:
        """Check if two principles contradict each other."""
        conds_a = a.get("conditions", [])
        conds_b = b.get("conditions", [])
        actions_a = a.get("action_override", {})
        actions_b = b.get("action_override", {})

        if not conds_a or not conds_b or not actions_a or not actions_b:
            return None

        # Step 1: Find shared conditions (same feature, similar threshold)
        shared = []
        for ca in conds_a:
            for cb in conds_b:
                if self._conditions_overlap(ca, cb):
                    shared.append(ca.get("feature", ""))

        if not shared:
            return None

        # Step 2: Find conflicting actions (opposite directions)
        conflicts = []
        for key in actions_a:
            if key in actions_b:
                va, vb = actions_a[key], actions_b[key]
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    # Opposite direction check
                    if va > 0 and vb < 0 or va < 0 and vb > 0:
                        conflicts.append(key)
                    # Large magnitude difference
                    elif abs(va) > 1e-6 and abs(vb) > 1e-6:
                        ratio = max(va, vb) / min(abs(va), abs(vb))
                        if ratio > 3.0:  # 3x difference = conflict
                            conflicts.append(key)

        if not conflicts:
            return None

        # Severity: more shared conditions + more conflicts = worse
        severity = len(shared) * 0.4 + len(conflicts) * 0.6
        conf_a = a.get("confidence", 0.5)
        conf_b = b.get("confidence", 0.5)
        # Both high-confidence = more severe
        severity *= (conf_a + conf_b)

        return {
            "principle_a": a.get("principle_id", "?"),
            "principle_b": b.get("principle_id", "?"),
            "shared_conditions": shared,
            "conflicting_actions": conflicts,
            "severity": round(severity, 3),
            "confidence_a": conf_a,
            "confidence_b": conf_b,
        }

    def _conditions_overlap(self, ca: dict, cb: dict) -> bool:
        """Check if two conditions refer to the same feature/threshold.

        Uses the symmetric ("max" denominator) comparison: overlap is a
        relation between two conditions, with neither one the reference.
        See cognition/condition_match.py -- the string-threshold branch
        this depends on was bug #17, present in all three copies of it.
        """
        if ca.get("feature") != cb.get("feature"):
            return False
        return thresholds_match(
            ca.get("threshold", 0), cb.get("threshold", 0),
            self.THRESHOLD_TOLERANCE, mode="max",
        )

