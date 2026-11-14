"""Investigation Engine — Goal-driven regret response strategies.

V20+.4: When regret triggers an investigation, the engine doesn't just
detect what changed — it ACTS on it. Five escalating strategies:

    C.1: Revert + Record   — Undo the bad change, record why       (DEFAULT)
    C.2: Partial Revert    — Revert only the most-changed parameter
    C.3: Controlled Retry  — Same direction, smaller step size
    C.4: Hypothesis Test   — "Was it the penalty, or random noise?"
    C.5: Alternative Path  — Abandon direction entirely, try new approach

Strategy selection is stress-driven:
    Low stress  → C.1 (conservative)
    Mid stress  → C.2, C.3 (surgical)
    High stress → C.4, C.5 (aggressive / desperate)

The investigation is goal-driven: it continues until accuracy recovers
to ≥ best_proven (not a fixed number of experiments).

Records:
    - What parameter changed (from → to)
    - How much regret it caused
    - Whether the strategy restored performance
    - CPU time spent investigating (this IS recovery cost)
"""

from __future__ import annotations
import time
import numpy as np


# ── Strategy Constants ───────────────────────────────────────────────

STRATEGY_REVERT_RECORD = "C.1"
STRATEGY_PARTIAL_REVERT = "C.2"
STRATEGY_CONTROLLED_RETRY = "C.3"
STRATEGY_HYPOTHESIS_TEST = "C.4"
STRATEGY_ALTERNATIVE_PATH = "C.5"

# Stress thresholds for strategy selection
STRESS_LOW = 0.3      # Below: use C.1
STRESS_MID = 0.6      # Below: use C.2 or C.3
STRESS_HIGH = 0.8     # Above: use C.4 or C.5

# Investigation budget limits
MAX_INVESTIGATION_STAGES = 5    # Hard cap on investigation stages
RETRY_STEP_DECAY = 0.5          # Each retry halves the step size


class InvestigationEngine:
    """Goal-driven regret investigation with escalating strategies.

    When the optimizer detects regret (current < best_proven), this engine
    selects and executes an investigation strategy to recover performance.

    It is NOT a random search. Each strategy is deterministic:
        - C.1 reverts to the exact configuration at best.
        - C.2 isolates the most-changed parameter and reverts only it.
        - C.3 retries the same direction with halved step size.
        - C.4 runs A/B comparison to distinguish signal from noise.
        - C.5 proposes a configuration orthogonal to the failing direction.
    """

    def __init__(self):
        # Investigation history for the current optimization run
        self._investigations: list[dict] = []
        self._total_investigation_time: float = 0.0

    # ── Strategy Selection ──────────────────────────────────────────

    def select_strategy(
        self,
        system_stress: float,
        regret: float,
        investigation_count: int,
    ) -> str:
        """Select investigation strategy based on stress and history.

        Args:
            system_stress: Current system stress [0, 1].
            regret: Current regret value.
            investigation_count: How many investigations have been run
                in this optimization session already.

        Returns:
            Strategy code (C.1 through C.5).
        """
        # If we've already investigated and it didn't help, escalate
        if investigation_count >= 3:
            return STRATEGY_ALTERNATIVE_PATH
        if investigation_count >= 2:
            return STRATEGY_HYPOTHESIS_TEST

        # Stress-based selection
        if system_stress < STRESS_LOW:
            return STRATEGY_REVERT_RECORD
        elif system_stress < STRESS_MID:
            # Mid stress: partial revert if we have delta info
            return STRATEGY_PARTIAL_REVERT
        elif system_stress < STRESS_HIGH:
            return STRATEGY_CONTROLLED_RETRY
        else:
            return STRATEGY_HYPOTHESIS_TEST

    # ── Strategy Execution ──────────────────────────────────────────

    def execute_investigation(
        self,
        strategy: str,
        current_config: dict,
        best_config: dict,
        current_weights: np.ndarray,
        best_weights: np.ndarray,
        investigation_data: dict,
    ) -> dict:
        """Execute an investigation strategy.

        Args:
            strategy: Strategy code (C.1 - C.5).
            current_config: Current classifier hyperparameter config.
            best_config: Config at best_proven accuracy.
            current_weights: Current feature weight vector.
            best_weights: Feature weights at best_proven accuracy.
            investigation_data: The investigation record from StressEngine
                (contains changes_detected, suspected_cause, etc.)

        Returns:
            Investigation result dict with:
                - strategy: which strategy was used
                - proposed_config: new config to try
                - proposed_weights: new weights to try
                - rationale: why this was proposed
                - parameter_changes: what was changed and why
        """
        start_time = time.perf_counter()

        if strategy == STRATEGY_REVERT_RECORD:
            result = self._revert_and_record(
                current_config, best_config,
                current_weights, best_weights,
                investigation_data,
            )
        elif strategy == STRATEGY_PARTIAL_REVERT:
            result = self._partial_revert(
                current_config, best_config,
                current_weights, best_weights,
                investigation_data,
            )
        elif strategy == STRATEGY_CONTROLLED_RETRY:
            result = self._controlled_retry(
                current_config, best_config,
                current_weights, best_weights,
                investigation_data,
            )
        elif strategy == STRATEGY_HYPOTHESIS_TEST:
            result = self._hypothesis_test(
                current_config, best_config,
                current_weights, best_weights,
                investigation_data,
            )
        elif strategy == STRATEGY_ALTERNATIVE_PATH:
            result = self._alternative_path(
                current_config, best_config,
                current_weights, best_weights,
                investigation_data,
            )
        else:
            # Fallback to C.1
            result = self._revert_and_record(
                current_config, best_config,
                current_weights, best_weights,
                investigation_data,
            )

        elapsed = time.perf_counter() - start_time
        self._total_investigation_time += elapsed

        # Wrap result with metadata
        result["strategy"] = strategy
        result["investigation_cpu_time"] = round(elapsed, 4)
        result["investigation_index"] = len(self._investigations)

        self._investigations.append(result)
        return result

    # ── C.1: Revert + Record ────────────────────────────────────────

    def _revert_and_record(
        self,
        current_config: dict,
        best_config: dict,
        current_weights: np.ndarray,
        best_weights: np.ndarray,
        investigation_data: dict,
    ) -> dict:
        """C.1: Full revert to best-known configuration.

        The safest strategy. Immediately restores the exact state
        that produced the best result.
        """
        return {
            "proposed_config": dict(best_config) if best_config else dict(current_config),
            "proposed_weights": best_weights.copy() if best_weights is not None else current_weights.copy(),
            "rationale": "Full revert to best-proven configuration",
            "parameter_changes": self._diff_configs(current_config, best_config),
        }

    # ── C.2: Partial Revert ─────────────────────────────────────────

    def _partial_revert(
        self,
        current_config: dict,
        best_config: dict,
        current_weights: np.ndarray,
        best_weights: np.ndarray,
        investigation_data: dict,
    ) -> dict:
        """C.2: Revert only the most-changed parameter.

        Surgical approach: keep the good changes, undo only
        the one that likely caused the regression.
        """
        suspected_cause = investigation_data.get("suspected_cause", "")
        changes = investigation_data.get("changes_detected", {})

        # Start with current config (keep most changes)
        proposed_config = dict(current_config) if current_config else {}
        proposed_weights = current_weights.copy() if current_weights is not None else None

        if suspected_cause and best_config and suspected_cause in best_config:
            # Revert only the suspected parameter
            proposed_config[suspected_cause] = best_config[suspected_cause]

        # If the suspected cause is a feature weight dimension, revert that
        if best_weights is not None and current_weights is not None:
            # Find the dimension with the largest delta
            weight_delta = np.abs(current_weights - best_weights)
            if len(weight_delta) > 0:
                worst_dim = int(np.argmax(weight_delta))
                proposed_weights[worst_dim] = best_weights[worst_dim]

        return {
            "proposed_config": proposed_config,
            "proposed_weights": proposed_weights,
            "rationale": f"Partial revert: reverting suspected cause '{suspected_cause}'",
            "parameter_changes": {
                "reverted_param": suspected_cause,
                "reverted_from": changes.get(suspected_cause, {}).get("current"),
                "reverted_to": changes.get(suspected_cause, {}).get("at_best"),
            },
        }

    # ── C.3: Controlled Retry ──────────────────────────────────────

    def _controlled_retry(
        self,
        current_config: dict,
        best_config: dict,
        current_weights: np.ndarray,
        best_weights: np.ndarray,
        investigation_data: dict,
    ) -> dict:
        """C.3: Same direction, smaller step size.

        The optimizer's direction might be right, but the step was
        too large. Retry with halved deltas from best.
        """
        proposed_config = dict(best_config) if best_config else dict(current_config)
        proposed_weights = best_weights.copy() if best_weights is not None else current_weights.copy()

        # For config params: move halfway from best toward current
        if best_config and current_config:
            for key in current_config:
                if key in best_config:
                    curr = current_config[key]
                    best_val = best_config[key]
                    if isinstance(curr, (int, float)) and isinstance(best_val, (int, float)):
                        delta = curr - best_val
                        proposed_config[key] = best_val + delta * RETRY_STEP_DECAY

        # For weights: move halfway from best toward current
        if best_weights is not None and current_weights is not None:
            delta = current_weights - best_weights
            proposed_weights = best_weights + delta * RETRY_STEP_DECAY

        return {
            "proposed_config": proposed_config,
            "proposed_weights": proposed_weights,
            "rationale": f"Controlled retry: same direction, {RETRY_STEP_DECAY}x step",
            "parameter_changes": {"step_decay": RETRY_STEP_DECAY},
        }

    # ── C.4: Hypothesis Test ───────────────────────────────────────

    def _hypothesis_test(
        self,
        current_config: dict,
        best_config: dict,
        current_weights: np.ndarray,
        best_weights: np.ndarray,
        investigation_data: dict,
    ) -> dict:
        """C.4: A/B comparison — test whether the change was causal.

        Produces TWO configurations to test:
        1. The best config (control)
        2. The best config with ONLY the suspected change (treatment)

        If treatment performs well → the other changes caused the problem.
        If treatment performs poorly → the suspected change WAS the cause.
        """
        suspected_cause = investigation_data.get("suspected_cause", "")
        changes = investigation_data.get("changes_detected", {})

        # Control: pure best config
        control_config = dict(best_config) if best_config else dict(current_config)

        # Treatment: best config + ONLY the suspected change
        treatment_config = dict(control_config)
        if suspected_cause in changes:
            treatment_config[suspected_cause] = changes[suspected_cause].get("current")

        return {
            "proposed_config": control_config,  # Primary: use control
            "proposed_weights": best_weights.copy() if best_weights is not None else current_weights.copy(),
            "rationale": f"Hypothesis test: isolating effect of '{suspected_cause}'",
            "hypothesis": {
                "variable": suspected_cause,
                "control": control_config.get(suspected_cause),
                "treatment": treatment_config.get(suspected_cause),
            },
            "treatment_config": treatment_config,
            "parameter_changes": {"test_variable": suspected_cause},
        }

    # ── C.5: Alternative Path ──────────────────────────────────────

    def _alternative_path(
        self,
        current_config: dict,
        best_config: dict,
        current_weights: np.ndarray,
        best_weights: np.ndarray,
        investigation_data: dict,
    ) -> dict:
        """C.5: Abandon the failing direction entirely.

        At high stress, the current direction might be fundamentally wrong.
        This proposes a configuration that is orthogonal to both
        current and best — a fresh exploration.
        """
        proposed_config = dict(best_config) if best_config else dict(current_config)
        proposed_weights = best_weights.copy() if best_weights is not None else current_weights.copy()

        # Generate orthogonal weights via perturbation
        if best_weights is not None and current_weights is not None:
            # Direction from best to current (the failing direction)
            fail_direction = current_weights - best_weights
            fail_norm = np.linalg.norm(fail_direction)

            if fail_norm > 1e-8:
                # Normalize the failing direction
                fail_unit = fail_direction / fail_norm

                # Generate a random orthogonal direction
                rng = np.random.default_rng(42)
                random_dir = rng.standard_normal(len(best_weights))
                # Gram-Schmidt: remove the failing component
                random_dir -= np.dot(random_dir, fail_unit) * fail_unit
                ortho_norm = np.linalg.norm(random_dir)
                if ortho_norm > 1e-8:
                    random_dir /= ortho_norm
                    # Step in the orthogonal direction from best
                    proposed_weights = best_weights + random_dir * fail_norm * 0.3
                    proposed_weights = np.clip(proposed_weights, 0.0, 1.0)

        return {
            "proposed_config": proposed_config,
            "proposed_weights": proposed_weights,
            "rationale": "Alternative path: orthogonal exploration from best",
            "parameter_changes": {"exploration_type": "orthogonal"},
        }

    # ── Utilities ───────────────────────────────────────────────────

    def _diff_configs(self, config_a: dict, config_b: dict) -> dict:
        """Diff two configurations, returning changes."""
        if not config_a or not config_b:
            return {}
        changes = {}
        all_keys = set(list(config_a.keys()) + list(config_b.keys()))
        for key in all_keys:
            a_val = config_a.get(key)
            b_val = config_b.get(key)
            if a_val != b_val:
                changes[key] = {"from": a_val, "to": b_val}
        return changes

    def record_investigation_outcome(
        self,
        investigation_index: int,
        accuracy_before: float,
        accuracy_after: float,
        restored_performance: bool,
    ) -> None:
        """Record whether the investigation strategy worked.

        Args:
            investigation_index: Which investigation to update.
            accuracy_before: Accuracy before the investigation action.
            accuracy_after: Accuracy after the investigation action.
            restored_performance: Whether accuracy_after >= best_proven.
        """
        if 0 <= investigation_index < len(self._investigations):
            inv = self._investigations[investigation_index]
            inv["outcome"] = {
                "accuracy_before": round(accuracy_before, 4),
                "accuracy_after": round(accuracy_after, 4),
                "delta": round(accuracy_after - accuracy_before, 4),
                "restored": restored_performance,
            }

    def get_investigation_count(self) -> int:
        """Number of investigations performed in this session."""
        return len(self._investigations)

    def get_total_investigation_time(self) -> float:
        """Total CPU time spent on investigations."""
        return self._total_investigation_time

    def get_investigation_summary(self) -> dict:
        """Summary of all investigations for journal recording."""
        if not self._investigations:
            return {}

        strategies_used = [inv.get("strategy", "?") for inv in self._investigations]
        outcomes = [inv.get("outcome", {}) for inv in self._investigations if "outcome" in inv]
        restored_count = sum(1 for o in outcomes if o.get("restored", False))

        return {
            "total_investigations": len(self._investigations),
            "strategies_used": strategies_used,
            "total_cpu_time": round(self._total_investigation_time, 4),
            "restorations_successful": restored_count,
            "restorations_attempted": len(outcomes),
        }
