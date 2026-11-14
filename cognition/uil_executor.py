"""UIL Executor — Pure mathematical condition evaluation.

NO string parsing. NO 'if "N < 50" in p.get("condition")'.
The executor receives a state dict of numbers and compares them
against machine-discovered principle conditions using operators.

Pure transistor logic. Numbers in, overrides out.
"""

from __future__ import annotations
import operator
from typing import Callable

from figpso_uil.cognition.execution_context import flatten_context

# Operator map: string → function
_OPS: dict[str, Callable] = {
    "<=": operator.le,
    "<": operator.lt,
    ">=": operator.ge,
    ">": operator.gt,
    "==": operator.eq,
    "!=": operator.ne,
}


class UIL_Executor:
    """Evaluates principles against current state. Pure math.
    
    No strings. No parsing. No English.
    
    Receives: state dict {n_samples: 45, entropy: 0.08, penalty: 2.6, ...}
    Checks: each principle's conditions against those numbers
    Returns: merged action overrides from triggered principles
    Records: which principles fired (for tracking times_applied/helped)
    """

    def __init__(self):
        self._triggered_principles: list[str] = []

    def get_overrides(
        self, state: "dict | object", principles: list[dict]
    ) -> dict:
        """Evaluate all principles against current state.

        V20+.1: Tier-aware evaluation.
            - Tier 1 (Universal) principles take absolute priority.
            - Tier 2 (Category) fills gaps not covered by Tier 1.
            - Tier 3 (Dataset-Specific) fills remaining gaps.
            - Tier 4 (Exception) only applies if matched to current dataset.
            - Within each tier, higher confidence wins conflicts.

        Args:
            state: Either a plain state dict (flat or namespaced keys,
                e.g. 'penalty' or 'state.entropy') or an ExecutionContext
                (see execution_context.py) carrying state plus the
                previous stage's action/config, so 'action.*'/'config.*'
                conditions can resolve too.
            principles: Active principles from the library (already tier-filtered).

        Returns:
            Merged action overrides from all triggered principles.
            Higher-tier and higher-confidence principles take priority.
        """
        state = flatten_context(state)
        self._triggered_principles = []
        overrides = {}

        # Principles should already be sorted by tier then confidence
        # (from hierarchy.filter_by_tier), but sort again for safety
        sorted_principles = sorted(
            principles,
            key=lambda p: (p.get("tier", 3), -p.get("confidence", 0)),
        )

        for principle in sorted_principles:
            confidence = principle.get("confidence", 0)
            if confidence < 0.50:
                continue  # Skip low-confidence principles

            conditions = principle.get("conditions", [])
            if not conditions:
                continue

            # Check if ALL conditions are met
            all_met = all(
                self._eval_condition(cond, state) for cond in conditions
            )

            if all_met:
                pid = principle.get("principle_id", "unknown")
                self._triggered_principles.append(pid)

                # Apply action overrides (first-wins for conflicts)
                # Because we sorted by tier, Tier 1 always wins
                action_override = principle.get("action_override", {})
                for key, value in action_override.items():
                    if key not in overrides:
                        overrides[key] = value

        return overrides

    def get_triggered_principles(self) -> list[str]:
        """Get list of principle IDs that triggered in the last evaluation.
        
        Used by the optimizer to track times_applied in the library.
        """
        return list(self._triggered_principles)

    # ── Internal ─────────────────────────────────────────────────────

    def _eval_condition(self, condition: dict, state: dict) -> bool:
        """Evaluate a single condition against the state. Pure math.
        
        Handles both flat keys ('penalty') and namespaced keys
        ('action.penalty', 'state.n_samples') by searching all levels.
        """
        feature = condition.get("feature", "")
        op_str = condition.get("op", "")
        threshold = condition.get("threshold")

        if threshold is None or op_str not in _OPS:
            return False

        # Resolve feature value from state
        value = self._resolve_value(feature, state)
        if value is None:
            return False

        try:
            return _OPS[op_str](float(value), float(threshold))
        except (ValueError, TypeError):
            return False

    def _resolve_value(self, feature: str, state: dict) -> float | None:
        """Resolve a feature name to a value from the state dict.
        
        Handles:
        - Flat: 'penalty' → state['penalty']
        - Namespaced: 'action.penalty' → state['penalty'] (strips prefix)
        - Nested: 'action.penalty' → state['action']['penalty']
        """
        # Direct lookup
        if feature in state:
            val = state[feature]
            if isinstance(val, (int, float)):
                return float(val)

        # Strip namespace prefix (e.g., 'state.entropy' → 'entropy')
        if "." in feature:
            _, short_name = feature.rsplit(".", 1)
            if short_name in state:
                val = state[short_name]
                if isinstance(val, (int, float)):
                    return float(val)

            # Nested lookup (e.g., state['action']['penalty'])
            parts = feature.split(".")
            current = state
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            if isinstance(current, (int, float)):
                return float(current)

        return None
