"""UIL Stress Engine — Dual stress model with compounding pressure.

Two distinct stress signals:
- Decision Stress: per-action operational cost (Failure + Recovery + Time Lost)
- System Stress: cumulative, compounding pressure from Decision Stress history

System Stress is easy to accumulate, hard to pay off — like debt.
It governs behavioral escalation across the entire UIL.

Also computes Regret (backward-looking accountability):
    Regret = best_proven - actual
    "I already proved I can do 85%. Why am I at 78%? What changed?"
"""

from __future__ import annotations
import numpy as np


class StressEngine:
    """The UIL's pressure system. Drives behavioral escalation.
    
    Decision Stress measures the immediate cost of each action.
    System Stress accumulates and compounds over time.
    Together they create a ratchet: performance can only go up.
    """

    def __init__(self, adaptive_blend: bool = False):
        self.system_stress: float = 0.0
        self.best_proven_accuracy: float = 0.0
        self.stages_since_best_improved: int = 0
        # Last components of the (Category-D, hand-weighted) pressure blend,
        # populated by update_system_stress for audit Part E telemetry.
        self._last_pressure_components: dict = {}
        # Category D fix #2 (2026-07-24): when True, drop components that
        # are uninformative THIS stage and renormalize the live weights.
        # Measured: deficit is exactly 0 on easy data; explanation_failure
        # is exactly 1 when miner R²=0 — both waste weight as a constant
        # offset. Default False = historical 0.30/0.25/0.25/0.20 blend.
        self.adaptive_blend: bool = bool(adaptive_blend)
        self.cpu_time_since_best: float = 0.0  # V20: CPU time recovery cost
        self._accuracy_history: list[float] = []
        self._decision_stress_history: list[float] = []
        self._recovery_log: dict = {}  # V20: last recovery cost breakdown

    def set_initial_stress(self, stress: float) -> None:
        """Set initial system stress (e.g., from dataset memory).
        
        Used when the journal remembers a dataset was hard.
        """
        self.system_stress = np.clip(stress, 0.0, 1.0)

    # ── Decision Stress (per-action) ─────────────────────────────────

    def compute_decision_stress(
        self,
        previous_accuracy: float,
        current_accuracy: float,
        cpu_time_this_stage: float,
        total_cpu_time: float,
        stages_since_best: int,
    ) -> float:
        """Compute Decision Stress for a single action.
        
        Based on the original UIL formulation:
            Stress = Failure Cost + Recovery Cost + Time Lost
        
        V20: Recovery Cost now uses BOTH stages AND cpu_time since best,
        matching the agreed formula: recovery = stages_to_recover + cpu_time_to_recover.
        
        Args:
            previous_accuracy: Accuracy at the end of the previous stage.
            current_accuracy: Accuracy at the end of this stage.
            cpu_time_this_stage: CPU time consumed by this stage (seconds).
            total_cpu_time: Total CPU time consumed so far (seconds).
            stages_since_best: How many stages since best accuracy improved.
        
        Returns:
            Decision stress value in [0, 1].
        """
        # Track CPU time since best
        self.cpu_time_since_best += cpu_time_this_stage

        # Failure Cost: how much accuracy dropped from previous stage
        failure_cost = max(0.0, previous_accuracy - current_accuracy)
        # Scale: 0.1 accuracy drop → 0.5 failure cost
        failure_cost = min(1.0, failure_cost * 5.0)

        # Recovery Cost: stages + CPU time spent trying to get back to best
        # V20: Dual recovery cost (stages AND cpu time)
        stage_recovery = min(1.0, stages_since_best / 10.0)
        cpu_recovery = min(1.0, self.cpu_time_since_best / max(total_cpu_time * 0.5, 1.0))
        recovery_cost = 0.6 * stage_recovery + 0.4 * cpu_recovery

        # Time Lost: fraction of total time spent without progress
        if total_cpu_time > 0:
            time_lost = min(1.0, cpu_time_this_stage / max(total_cpu_time, 1.0))
        else:
            time_lost = 0.0

        # Store recovery breakdown for journal
        self._recovery_log = {
            "stages_since_best": stages_since_best,
            "cpu_time_since_best": round(self.cpu_time_since_best, 3),
            "stage_recovery_component": round(stage_recovery, 4),
            "cpu_recovery_component": round(cpu_recovery, 4),
            "recovery_cost": round(recovery_cost, 4),
        }

        # Weighted combination
        decision_stress = (
            0.40 * failure_cost
            + 0.35 * recovery_cost
            + 0.25 * time_lost
        )

        self._decision_stress_history.append(decision_stress)
        return float(np.clip(decision_stress, 0.0, 1.0))

    # ── System Stress (cumulative, compounding) ──────────────────────

    def update_system_stress(
        self,
        decision_stress: float,
        current_accuracy: float,
        miner_r2_score: float = 0.5,
    ) -> float:
        """Update System Stress — cumulative and compounding.
        
        System Stress is like debt: easy to accumulate, hard to pay off.
        It takes 1 bad stage to raise stress significantly, but
        10 good stages to bring it back down.
        
        Four pressure signals compound into system stress:
        1. Decision Stress history (recent pain)
        2. Accuracy deficit from 95% target
        3. Stagnation (stages since best improved)
        4. Explanation failure (Miner can't find causes)
        
        Args:
            decision_stress: Current Decision Stress value.
            current_accuracy: Current accuracy.
            miner_r2_score: Miner's Decision Tree R² score.
                High R² = can explain regret = lower stress.
                Low R² = blind to causes = higher stress.
        
        Returns:
            Updated system stress in [0, 1].
        """
        # Track accuracy history
        self._accuracy_history.append(current_accuracy)

        # Update best proven
        improvement = 0.0
        if current_accuracy > self.best_proven_accuracy:
            improvement = current_accuracy - self.best_proven_accuracy
            self.best_proven_accuracy = current_accuracy
            self.stages_since_best_improved = 0
            self.cpu_time_since_best = 0.0  # V20: reset CPU recovery clock
        else:
            self.stages_since_best_improved += 1

        # ── Four pressure signals ──

        # Signal 1: Recent decision stress (rolling average)
        recent_n = min(5, len(self._decision_stress_history))
        recent_stress = (
            np.mean(self._decision_stress_history[-recent_n:])
            if recent_n > 0
            else 0.0
        )

        # Signal 2: Accuracy deficit from 95% target
        deficit = max(0.0, 0.95 - current_accuracy) / 0.95

        # Signal 3: Stagnation
        stagnation = min(1.0, self.stages_since_best_improved / 8.0)

        # Signal 4: Explanation failure (1 - R²)
        explanation_failure = max(0.0, 1.0 - miner_r2_score)

        # Current pressure (all signals combined). WEIGHTS are Category-D
        # of the hardcoded-value audit: hand-picked, never fit to data.
        # Store the components so a consumer can log their distributions
        # (audit Part E) — the prerequisite evidence for ever fitting these
        # four weights instead of guessing them.
        w_recent, w_deficit, w_stag, w_expl = 0.30, 0.25, 0.25, 0.20
        if self.adaptive_blend:
            # Regime-conditional: a component that carries no information
            # this stage gets weight 0; survivors are renormalized. Matches
            # the measured dead-term finding without inventing new numbers.
            if float(deficit) < 1e-12:
                w_deficit = 0.0
            if float(miner_r2_score) <= 0.0:
                w_expl = 0.0
            s = w_recent + w_deficit + w_stag + w_expl
            if s > 1e-12:
                w_recent, w_deficit, w_stag, w_expl = (
                    w_recent / s, w_deficit / s, w_stag / s, w_expl / s,
                )
            else:
                w_recent, w_deficit, w_stag, w_expl = 0.30, 0.25, 0.25, 0.20
        current_pressure = (
            w_recent * recent_stress
            + w_deficit * deficit
            + w_stag * stagnation
            + w_expl * explanation_failure
        )
        self._last_pressure_components = {
            "recent_stress": float(recent_stress),
            "deficit": float(deficit),
            "stagnation": float(stagnation),
            "explanation_failure": float(explanation_failure),
            "current_pressure": float(current_pressure),
            "adaptive_blend": bool(self.adaptive_blend),
            "w_recent": float(w_recent),
            "w_deficit": float(w_deficit),
            "w_stagnation": float(w_stag),
            "w_explanation_failure": float(w_expl),
        }

        # ── Compounding logic ──
        if improvement > 0.01:
            # Relief: stress decays, but SLOWLY
            # Need sustained improvement to bring stress down
            relief_rate = min(0.20, 0.05 * (improvement / 0.05))
            self.system_stress *= (1.0 - relief_rate)
        else:
            # Compounding: stress grows on itself
            # The MORE stressed, the FASTER it grows
            compound_factor = current_pressure * (1.0 + self.system_stress * 0.5)
            self.system_stress += compound_factor * 0.08

        self.system_stress = float(np.clip(self.system_stress, 0.0, 1.0))
        return self.system_stress

    # ── Regret (backward-looking accountability) ─────────────────────

    def compute_regret(self, current_accuracy: float) -> float:
        """Compute Regret — backward-looking accountability.
        
        Regret = best_proven - actual
        
        "I already proved I can do 85%. Why am I at 78%?"
        
        When regret > 0, the system must investigate:
        - What changed since we were at our best?
        - Which parameter shift caused the drop?
        - How do we get back?
        
        Regret is compounded by system stress:
        Higher stress → higher regret multiplier → more aggressive investigation.
        
        Returns:
            Regret value (0 = no regret, higher = worse).
        """
        base_regret = max(0.0, self.best_proven_accuracy - current_accuracy)

        # Stress amplifies regret — compound pressure
        stress_multiplier = 1.0 + (self.system_stress * 2.0)

        return float(base_regret * stress_multiplier)

    # ── Investigation Trigger ────────────────────────────────────────

    def build_investigation(
        self,
        current_state: dict,
        best_state: dict | None,
        regret: float,
    ) -> dict | None:
        """When regret > 0, build an investigation record.
        
        Compares current state to the state when best accuracy was achieved.
        Identifies what changed and by how much.
        
        Returns:
            Investigation dict, or None if no regret.
        """
        if regret <= 0 or best_state is None:
            return None

        changes = {}
        for key in current_state:
            if key in best_state:
                curr_val = current_state[key]
                best_val = best_state[key]
                if isinstance(curr_val, (int, float)) and isinstance(best_val, (int, float)):
                    delta = curr_val - best_val
                    if abs(delta) > 1e-6:
                        changes[key] = {
                            "current": curr_val,
                            "at_best": best_val,
                            "delta": delta,
                            "pct_change": abs(delta) / max(abs(best_val), 1e-6) * 100,
                        }

        if not changes:
            return None

        # Find the most changed variable
        most_changed = max(changes, key=lambda k: changes[k]["pct_change"])

        return {
            "trigger": "regret",
            "regret_value": regret,
            "system_stress": self.system_stress,
            "changes_detected": changes,
            "suspected_cause": most_changed,
            "suspected_delta": changes[most_changed]["delta"],
        }

    # ── Critical Failure Detection ───────────────────────────────────

    def has_failed(
        self,
        miner_r2: float,
        principles_exhausted: bool,
        patience: int = 15,
    ) -> bool:
        """Determine if the system has genuinely failed.
        
        NOT a threshold check — a logical conclusion from multiple evidence:
        1. Stress has been compounding for sustained period
        2. No improvement despite efforts
        3. Miner can't explain WHY (low R²)
        4. Principles have been exhausted
        
        Args:
            miner_r2: Miner's current R² score.
            principles_exhausted: Whether all principles have been tried.
            patience: Minimum stages at high stress before declaring failure.
        
        Returns:
            True if the system has genuinely concluded failure.
        """
        conditions = [
            self.system_stress > 0.75,
            self.stages_since_best_improved >= patience,
            miner_r2 < 0.3,
            principles_exhausted,
        ]
        return all(conditions)

    def get_recovery_cost_breakdown(self) -> dict:
        """V20: Get the last recovery cost breakdown for journal recording."""
        return dict(self._recovery_log)

    def reset(self, keep_knowledge: bool = True) -> None:
        """Reset stress after critical failure.
        
        Stress goes to 0 (fresh start), but knowledge survives
        in the journal and principle library.
        """
        self.system_stress = 0.0
        self.stages_since_best_improved = 0
        self.cpu_time_since_best = 0.0
        self._recovery_log = {}
        if not keep_knowledge:
            self.best_proven_accuracy = 0.0
            self._accuracy_history.clear()
            self._decision_stress_history.clear()
