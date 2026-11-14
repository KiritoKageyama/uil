"""Principle Hierarchy — 4-Tier system with promotion, demotion, and quarantine.

Tier 1: UNIVERSAL — Apply across all datasets. Hardest to override.
Tier 2: CATEGORY  — Apply to a class of datasets (e.g., high-dimensional sparse).
Tier 3: DATASET-SPECIFIC — Apply to one particular dataset or close matches.
Tier 4: EXCEPTION (Quarantine) — Isolated rules for datasets that break all tiers.
         Exception rules do NOT propagate upward and never corrupt universal knowledge.

Promotion/Demotion is entirely evidence-driven:
    - Within-dataset rules start at Tier 3.
    - Cross-dataset rules start at Tier 2.
    - Rules corroborated across many datasets (and classifiers) promote to Tier 1.
    - Rules that fail repeatedly demote or get quarantined.
"""

from __future__ import annotations
from datetime import datetime, timezone


# ── Tier Constants ───────────────────────────────────────────────────

TIER_UNIVERSAL = 1
TIER_CATEGORY = 2
TIER_DATASET_SPECIFIC = 3
TIER_EXCEPTION = 4

TIER_NAMES = {
    TIER_UNIVERSAL: "universal",
    TIER_CATEGORY: "category",
    TIER_DATASET_SPECIFIC: "dataset_specific",
    TIER_EXCEPTION: "exception",
}

# ── Promotion / Demotion Thresholds ─────────────────────────────────

# To promote from Tier 3 → Tier 2:
PROMOTE_3_TO_2_MIN_DATASETS = 3       # Confirmed on ≥3 distinct datasets
PROMOTE_3_TO_2_MIN_CONFIDENCE = 0.80

# To promote from Tier 2 → Tier 1:
PROMOTE_2_TO_1_MIN_DATASETS = 10      # Confirmed on ≥10 distinct datasets
PROMOTE_2_TO_1_MIN_CLASSIFIERS = 2    # Works on ≥2 different classifiers
PROMOTE_2_TO_1_MIN_CONFIDENCE = 0.85
PROMOTE_2_TO_1_MIN_EVIDENCE = 100     # ≥100 evidence observations

# Demotion: if help_ratio drops below threshold after enough applications
DEMOTE_MIN_APPLICATIONS = 15
DEMOTE_HELP_RATIO = 0.25  # Must help >25% of the time or demote

# Quarantine: if a principle consistently fails on a specific dataset
QUARANTINE_FAILURE_THRESHOLD = 5  # ≥5 consecutive failures on one dataset


class PrincipleHierarchy:
    """Manages the 4-tier principle lifecycle.

    Responsibilities:
        1. Assign initial tiers to newly mined rules.
        2. Track per-dataset performance of each principle.
        3. Promote principles that prove themselves across datasets/classifiers.
        4. Demote principles that stop helping.
        5. Quarantine principles that break specific datasets (Tier 4 isolation).
        6. Provide tier-aware retrieval for the Executor.
    """

    def __init__(self):
        # Per-principle dataset tracking:
        # {principle_id: {dataset_name: {"applied": int, "helped": int, "consecutive_failures": int}}}
        self._dataset_performance: dict[str, dict[str, dict]] = {}
        # Track which datasets a principle has been confirmed on
        # {principle_id: set(dataset_names)}
        self._confirmed_datasets: dict[str, set] = {}
        # Quarantine registry: {principle_id: [dataset_names where quarantined]}
        self._quarantine_registry: dict[str, list[str]] = {}

    # ── Tier Assignment ─────────────────────────────────────────────

    def assign_initial_tier(self, rule: dict) -> int:
        """Assign an initial tier to a newly mined rule.

        Within-dataset rules start at Tier 3 (dataset-specific).
        Cross-dataset rules start at Tier 2 (category).
        """
        mining_level = rule.get("mining_level", "unknown")
        if mining_level == "within_dataset":
            return TIER_DATASET_SPECIFIC
        elif mining_level == "cross_dataset":
            return TIER_CATEGORY
        else:
            return TIER_DATASET_SPECIFIC  # Default: conservative

    # ── Performance Recording ───────────────────────────────────────

    def record_application(
        self,
        principle_id: str,
        dataset_name: str,
        helped: bool,
    ) -> None:
        """Record that a principle was applied on a specific dataset.

        Tracks per-dataset performance for promotion/demotion decisions.
        """
        if principle_id not in self._dataset_performance:
            self._dataset_performance[principle_id] = {}

        if dataset_name not in self._dataset_performance[principle_id]:
            self._dataset_performance[principle_id][dataset_name] = {
                "applied": 0,
                "helped": 0,
                "consecutive_failures": 0,
            }

        perf = self._dataset_performance[principle_id][dataset_name]
        perf["applied"] += 1

        if helped:
            perf["helped"] += 1
            perf["consecutive_failures"] = 0
            # Track confirmed datasets
            if principle_id not in self._confirmed_datasets:
                self._confirmed_datasets[principle_id] = set()
            self._confirmed_datasets[principle_id].add(dataset_name)
        else:
            perf["consecutive_failures"] += 1

    # ── Promotion Logic ─────────────────────────────────────────────

    def evaluate_promotion(self, principle: dict) -> int | None:
        """Check if a principle qualifies for tier promotion.

        Returns:
            New tier if promoted, or None if no change.
        """
        pid = principle.get("principle_id", "")
        current_tier = principle.get("tier", TIER_DATASET_SPECIFIC)
        confidence = principle.get("confidence", 0.0)
        evidence = principle.get("evidence_count", 0)
        classifiers = principle.get("classifiers_confirmed", [])
        n_datasets = len(self._confirmed_datasets.get(pid, set()))

        # Tier 3 → Tier 2: confirmed across multiple datasets
        if current_tier == TIER_DATASET_SPECIFIC:
            if (
                n_datasets >= PROMOTE_3_TO_2_MIN_DATASETS
                and confidence >= PROMOTE_3_TO_2_MIN_CONFIDENCE
            ):
                return TIER_CATEGORY

        # Tier 2 → Tier 1: confirmed across many datasets AND classifiers
        elif current_tier == TIER_CATEGORY:
            if (
                n_datasets >= PROMOTE_2_TO_1_MIN_DATASETS
                and len(classifiers) >= PROMOTE_2_TO_1_MIN_CLASSIFIERS
                and confidence >= PROMOTE_2_TO_1_MIN_CONFIDENCE
                and evidence >= PROMOTE_2_TO_1_MIN_EVIDENCE
            ):
                return TIER_UNIVERSAL

        return None

    # ── Demotion Logic ──────────────────────────────────────────────

    def evaluate_demotion(self, principle: dict) -> int | None:
        """Check if a principle should be demoted.

        Returns:
            New tier if demoted, or None if no change.
        """
        current_tier = principle.get("tier", TIER_DATASET_SPECIFIC)
        applied = principle.get("times_applied", 0)
        helped = principle.get("times_helped", 0)

        # Can't demote below Tier 3 (Tier 4 is quarantine, not demotion)
        if current_tier == TIER_DATASET_SPECIFIC:
            return None

        # Check overall help ratio
        if applied >= DEMOTE_MIN_APPLICATIONS:
            help_ratio = helped / max(applied, 1)
            if help_ratio < DEMOTE_HELP_RATIO:
                # Demote by one tier
                return min(current_tier + 1, TIER_DATASET_SPECIFIC)

        return None

    # ── Quarantine Logic ────────────────────────────────────────────

    def check_quarantine(
        self, principle: dict, dataset_name: str
    ) -> bool:
        """Check if a principle should be quarantined for a specific dataset.

        A principle is quarantined when it has failed consecutively
        on a specific dataset beyond the threshold.

        Returns:
            True if the principle should be quarantined for this dataset.
        """
        pid = principle.get("principle_id", "")
        perf = self._dataset_performance.get(pid, {}).get(dataset_name, {})
        consecutive_failures = perf.get("consecutive_failures", 0)

        if consecutive_failures >= QUARANTINE_FAILURE_THRESHOLD:
            # Register quarantine
            if pid not in self._quarantine_registry:
                self._quarantine_registry[pid] = []
            if dataset_name not in self._quarantine_registry[pid]:
                self._quarantine_registry[pid].append(dataset_name)
            return True

        return False

    def is_quarantined(self, principle_id: str, dataset_name: str) -> bool:
        """Check if a principle is quarantined for a specific dataset."""
        return dataset_name in self._quarantine_registry.get(principle_id, [])

    # ── Tier-Aware Retrieval ────────────────────────────────────────

    def filter_by_tier(
        self,
        principles: list[dict],
        max_tier: int = TIER_DATASET_SPECIFIC,
        dataset_name: str | None = None,
    ) -> list[dict]:
        """Filter principles by tier, respecting quarantine.

        Args:
            principles: All active principles.
            max_tier: Maximum tier to include (1=universal only, 3=all).
            dataset_name: If set, exclude principles quarantined for this dataset.

        Returns:
            Filtered and sorted principles (Tier 1 first, highest confidence first).
        """
        filtered = []
        for p in principles:
            tier = p.get("tier", TIER_DATASET_SPECIFIC)
            pid = p.get("principle_id", "")

            # Tier 4 exception rules: include ONLY if they match the dataset
            if tier == TIER_EXCEPTION:
                exception_dataset = p.get("exception_dataset", "")
                if dataset_name and exception_dataset == dataset_name:
                    filtered.append(p)
                continue

            # Skip tiers above max
            if tier > max_tier:
                continue

            # Skip quarantined principles for this dataset
            if dataset_name and self.is_quarantined(pid, dataset_name):
                continue

            filtered.append(p)

        # Sort: Tier 1 first, then by confidence descending
        filtered.sort(key=lambda p: (p.get("tier", 3), -p.get("confidence", 0)))
        return filtered

    # ── Full Lifecycle Update ───────────────────────────────────────

    def update_principle_tiers(self, principles: list[dict]) -> list[dict]:
        """Run promotion and demotion checks on all principles.

        Called periodically (e.g., after every mining cycle).

        Returns:
            Updated list of principles with adjusted tiers.
        """
        for p in principles:
            pid = p.get("principle_id", "")

            # Ensure tier field exists
            if "tier" not in p:
                p["tier"] = self.assign_initial_tier(p)

            # Check promotion
            new_tier = self.evaluate_promotion(p)
            if new_tier is not None:
                old_tier = p["tier"]
                p["tier"] = new_tier
                p["tier_history"] = p.get("tier_history", [])
                p["tier_history"].append({
                    "from": TIER_NAMES.get(old_tier, str(old_tier)),
                    "to": TIER_NAMES.get(new_tier, str(new_tier)),
                    "reason": "promotion",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

            # Check demotion
            new_tier = self.evaluate_demotion(p)
            if new_tier is not None:
                old_tier = p["tier"]
                p["tier"] = new_tier
                p["tier_history"] = p.get("tier_history", [])
                p["tier_history"].append({
                    "from": TIER_NAMES.get(old_tier, str(old_tier)),
                    "to": TIER_NAMES.get(new_tier, str(new_tier)),
                    "reason": "demotion",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        return principles

    # ── Exception Handling Sequence ─────────────────────────────────

    def create_exception_principle(
        self,
        base_principle: dict,
        dataset_name: str,
        exception_overrides: dict,
    ) -> dict:
        """Create a Tier 4 exception principle for a specific dataset.

        When a dataset breaks all principles from Tiers 1-3, the system
        isolates it with dataset-specific rules that do NOT propagate.

        Args:
            base_principle: The principle that failed.
            dataset_name: The dataset it failed on.
            exception_overrides: The action overrides that actually worked.

        Returns:
            A new Tier 4 exception principle.
        """
        exception = dict(base_principle)
        exception["principle_id"] = f"{base_principle['principle_id']}-EX-{dataset_name[:8]}"
        exception["tier"] = TIER_EXCEPTION
        exception["exception_dataset"] = dataset_name
        exception["action_override"] = exception_overrides
        exception["confidence"] = 0.5  # Start neutral
        exception["evidence_count"] = 1
        exception["times_applied"] = 0
        exception["times_helped"] = 0
        exception["created_from"] = base_principle.get("principle_id", "unknown")
        exception["created_at"] = datetime.now(timezone.utc).isoformat()
        exception["tier_history"] = [{
            "from": "none",
            "to": "exception",
            "reason": f"quarantined for {dataset_name}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]
        return exception

    # ── Diagnostics ─────────────────────────────────────────────────

    def get_tier_summary(self, principles: list[dict]) -> dict:
        """Get a summary of principle counts by tier."""
        summary = {name: 0 for name in TIER_NAMES.values()}
        for p in principles:
            tier = p.get("tier", TIER_DATASET_SPECIFIC)
            tier_name = TIER_NAMES.get(tier, "unknown")
            summary[tier_name] = summary.get(tier_name, 0) + 1
        summary["total"] = len(principles)
        summary["quarantined_pairs"] = sum(
            len(datasets) for datasets in self._quarantine_registry.values()
        )
        return summary
