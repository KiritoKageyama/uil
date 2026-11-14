"""Causal Intervention Engine — Hypothesis-driven experimentation.

V20+.5: When the DT Miner finds a correlation, the UIL should NOT
immediately accept it. It must act like a scientist:

    1. Form a Hypothesis:  "High n_features causes high regret"
    2. Design Intervention: "Restrict n_features to 400"
    3. Run Intervention:    Actually try it
    4. Measure Effect:      Did regret drop?
    5. Conclude:            Causal evidence → promote with high confidence
                            Correlational only → keep but lower confidence

AutoML searches parameters. UIL searches causality.
"""

from __future__ import annotations
import json
import os
import time
import numpy as np

from figpso_uil.cognition.paths import results_path
from figpso_uil.cognition.version_store import VersionStore


# ── Hypothesis Status ────────────────────────────────────────────────

STATUS_PENDING = "pending"
STATUS_TESTING = "testing"
STATUS_CONFIRMED_CAUSAL = "confirmed_causal"
STATUS_CORRELATIONAL = "correlational_only"
STATUS_REFUTED = "refuted"


def controllable_factor(variable: str) -> str | None:
    """The factor of ``variable`` UIL can actually intervene on.

    Direct ``action.penalty`` / ``config.*`` variables return themselves.
    Derived product features (``derived.a*b``, minted by the miner's
    feature synthesis) return their first controllable factor — the
    hypothesis is then tested by moving that factor while the control arm
    keeps it at its current value. ``state.*`` variables describe the
    dataset itself and return None: never intervened on. Without this
    decomposition, most mined hypotheses were born permanently untestable
    (28/42 in the shared store were derived.*) and starved the scientist.
    """
    if variable == "action.penalty" or variable.startswith("config."):
        return variable
    if variable.startswith("derived."):
        for part in variable[len("derived."):].split("*"):
            part = part.strip()
            if part == "action.penalty" or part.startswith("config."):
                return part
    return None


class Hypothesis:
    """A scientific hypothesis about a causal relationship.

    Example:
        variable = "n_features"
        direction = "high"  # "high n_features causes high regret"
        threshold = 500
        condition_from_miner = {"feature": "state.n_features", "op": ">", "threshold": 500}
    """

    def __init__(
        self,
        hypothesis_id: str,
        variable: str,
        direction: str,
        threshold: float,
        source_rule: dict,
        predicted_regret: float = 0.0,
    ):
        self.hypothesis_id = hypothesis_id
        self.variable = variable          # Which variable
        self.direction = direction        # "high" or "low"
        self.threshold = threshold        # The threshold from the DT
        self.source_rule = source_rule    # The original DT-mined rule
        self.predicted_regret = predicted_regret
        self.status = STATUS_PENDING
        self.interventions: list[dict] = []
        # Accumulated per-trial evidence — each entry is one
        # control/treatment pair. CONFIRMED_CAUSAL requires a majority
        # of the last 3 trials to show a real effect; a single lucky
        # noisy trial can't declare causal status. Backwards-compatible:
        # loaded from JSON as [] when the field is absent.
        self.evidence_trials: list[dict] = []
        self.conclusion: str = ""
        self.causal_strength: float = 0.0  # 0-1, how causal is this
        self.created_at = time.time()

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "variable": self.variable,
            "direction": self.direction,
            "threshold": self.threshold,
            "source_rule": self.source_rule,
            "predicted_regret": self.predicted_regret,
            "status": self.status,
            "interventions": self.interventions,
            "evidence_trials": self.evidence_trials,
            "conclusion": self.conclusion,
            "causal_strength": self.causal_strength,
        }


class CausalInterventionEngine:
    """The UIL's scientific method for testing causality.

    Converts DT-mined correlations into testable hypotheses,
    designs interventions, and evaluates causal evidence.
    """

    def __init__(
        self,
        filepath: str | None = results_path("uil_causal_v20.json"),
        version_store: VersionStore | None = None,
    ):
        self.filepath = filepath
        self._hypotheses: dict[str, Hypothesis] = {}
        self._hypothesis_counter = 0
        self._total_interventions = 0
        self._version_store = version_store
        if self.filepath:
            self._load()

    def _cell_store(self) -> VersionStore | None:
        if self._version_store is not None:
            return self._version_store
        try:
            return VersionStore.shared()
        except Exception:
            return None

    def _log_cell(self, hypothesis: Hypothesis, op: str, cause: str | None = None) -> None:
        store = self._cell_store()
        if store is None:
            return
        store.record_cell(
            cell_id=hypothesis.hypothesis_id,
            cell_type="causal_hypothesis",
            op=op,
            post_state=hypothesis.to_dict(),
            cause=cause,
        )

    def canonical_state(self) -> dict[str, dict]:
        return {hid: h.to_dict() for hid, h in self._hypotheses.items()}

    # ── Hypothesis Formation ────────────────────────────────────────

    def form_hypothesis(self, rule: dict) -> Hypothesis | None:
        """Convert a DT-mined rule into a testable hypothesis.

        Args:
            rule: A principle/rule from the Miner containing:
                - conditions: list of {feature, op, threshold}
                - predicted_regret: float
                - confidence: float

        Returns:
            A Hypothesis object, or None if the rule isn't testable.
        """
        conditions = rule.get("conditions", [])
        if not conditions:
            return None

        # Use the primary (first) condition as the hypothesis variable
        primary = conditions[0]
        feature = primary.get("feature", "")
        op = primary.get("op", "")
        threshold = primary.get("threshold")

        if threshold is None:
            return None

        # Determine direction
        if op in (">", ">="):
            direction = "high"
        elif op in ("<", "<="):
            direction = "low"
        else:
            return None

        # Skip if already hypothesized for this variable
        for h in self._hypotheses.values():
            if h.variable == feature and h.direction == direction:
                if abs(h.threshold - threshold) / max(abs(threshold), 1e-6) < 0.15:
                    return None  # Too similar to existing

        self._hypothesis_counter += 1
        hid = f"H-{self._hypothesis_counter:04d}"
        hypothesis = Hypothesis(
            hypothesis_id=hid,
            variable=feature,
            direction=direction,
            threshold=float(threshold),
            source_rule=rule,
            predicted_regret=rule.get("predicted_regret", 0.0),
        )
        self._hypotheses[hid] = hypothesis
        self._log_cell(hypothesis, "create", cause="form_hypothesis")
        self._save()
        return hypothesis

    # ── Intervention Design ─────────────────────────────────────────

    def design_intervention(
        self,
        hypothesis: Hypothesis,
        current_value: float | None = None,
    ) -> dict:
        """Design an intervention to test a hypothesis.

        Produces a control and treatment configuration:
            Control:   Leave the variable at its current value
            Treatment: Move the variable to the opposite side of threshold

        Args:
            hypothesis: The hypothesis to test.
            current_value: Current value of the variable (if known).

        Returns:
            Intervention plan dict with control and treatment configs.
        """
        threshold = hypothesis.threshold
        variable = hypothesis.variable
        factor = controllable_factor(variable) or variable
        # Derived product variables: the threshold is on the PRODUCT, so
        # an absolute target value is meaningless for the single factor
        # we intervene on. Scale semantics instead — halve the factor to
        # push the product below threshold (direction "high": high values
        # cause regret), double it for direction "low". The caller
        # resolves the scale against the factor's live value.
        scale_mode = factor != variable
        if hypothesis.direction == "high":
            treatment_value = threshold * 0.8  # 20% below threshold
            treatment_scale = 0.5
            control_value = current_value if current_value is not None else threshold * 1.2
        else:
            treatment_value = threshold * 1.2
            treatment_scale = 2.0
            control_value = current_value if current_value is not None else threshold * 0.8

        intervention = {
            "hypothesis_id": hypothesis.hypothesis_id,
            "variable": variable,
            "intervention_variable": factor,
            "threshold": threshold,
            "control": {
                "description": f"Keep {factor} at current value",
                "value": control_value,
                "expected_regret": hypothesis.predicted_regret,
            },
            "treatment": {
                "description": (
                    f"Scale {factor} by {treatment_scale}x"
                    if scale_mode
                    else f"Move {variable} to opposite side of threshold"
                ),
                "value": None if scale_mode else treatment_value,
                "scale": treatment_scale if scale_mode else None,
                "expected_regret": 0.0 if hypothesis.predicted_regret > 0 else hypothesis.predicted_regret,
            },
        }

        hypothesis.status = STATUS_TESTING
        self._save()
        return intervention

    # ── Result Evaluation ──────────────────────────────────────────

    def evaluate_intervention(
        self,
        hypothesis_id: str,
        control_regret: float,
        treatment_regret: float,
        control_accuracy: float,
        treatment_accuracy: float,
    ) -> dict:
        """Evaluate the result of an intervention.

        Compares control vs treatment outcomes to determine causality.

        Args:
            hypothesis_id: The hypothesis being tested.
            control_regret: Regret under control conditions.
            treatment_regret: Regret under treatment conditions.
            control_accuracy: Accuracy under control.
            treatment_accuracy: Accuracy under treatment.

        Returns:
            Evaluation result with causal strength and conclusion.
        """
        hypothesis = self._hypotheses.get(hypothesis_id)
        if hypothesis is None:
            return {"error": f"Hypothesis {hypothesis_id} not found"}

        self._total_interventions += 1

        # Compute effect size
        regret_delta = control_regret - treatment_regret
        accuracy_delta = treatment_accuracy - control_accuracy

        # Record this intervention
        intervention_record = {
            "control_regret": round(control_regret, 4),
            "treatment_regret": round(treatment_regret, 4),
            "control_accuracy": round(control_accuracy, 4),
            "treatment_accuracy": round(treatment_accuracy, 4),
            "regret_delta": round(regret_delta, 4),
            "accuracy_delta": round(accuracy_delta, 4),
        }
        hypothesis.interventions.append(intervention_record)

        # Accumulate this trial's deltas into the evidence list before
        # deciding status — CONFIRMED_CAUSAL now requires a majority of
        # the last 3 trials to show a real effect, not just one lucky
        # noisy comparison. A single trial that crosses the delta threshold
        # is a promising signal, not causal evidence.
        hypothesis.evidence_trials.append({
            "regret_delta": round(regret_delta, 4),
            "accuracy_delta": round(accuracy_delta, 4),
        })

        _recent = hypothesis.evidence_trials[-3:]
        _strong_count = sum(
            1 for t in _recent
            if t["regret_delta"] > 0.05 and t["accuracy_delta"] > 0.01
        )
        _refuted_count = sum(1 for t in _recent if t["regret_delta"] < -0.02)

        # Determine causal strength
        if len(hypothesis.evidence_trials) >= 3 and _strong_count >= 2:
            # Majority of last 3 trials showed a real effect → CAUSAL
            avg_regret_delta = float(np.mean([t["regret_delta"] for t in _recent]))
            hypothesis.causal_strength = min(0.95, 0.5 + avg_regret_delta * 2)
            hypothesis.status = STATUS_CONFIRMED_CAUSAL
            hypothesis.conclusion = (
                f"CAUSAL: Controlling '{hypothesis.variable}' reduced regret "
                f"in {_strong_count}/3 trials (avg Δ={avg_regret_delta:.4f})"
            )
        elif regret_delta > 0.05 and accuracy_delta > 0.01:
            # Single strong trial — record as correlational pending replication
            hypothesis.causal_strength = min(0.5, 0.3 + regret_delta)
            hypothesis.status = STATUS_CORRELATIONAL
            hypothesis.conclusion = (
                f"CORRELATIONAL: Controlling '{hypothesis.variable}' reduced "
                f"regret by {regret_delta:.4f} — pending {3 - len(hypothesis.evidence_trials)} "
                f"more trial(s) for causal confirmation"
            )
        elif regret_delta > 0.02:
            # Weak effect
            hypothesis.causal_strength = 0.3 + regret_delta
            hypothesis.status = STATUS_CORRELATIONAL
            hypothesis.conclusion = (
                f"CORRELATIONAL: Controlling '{hypothesis.variable}' "
                f"reduced regret by {regret_delta:.4f} but effect is weak"
            )
        elif _refuted_count >= 2 or regret_delta < -0.02:
            # Counter-evidence: treatment INCREASED regret
            hypothesis.causal_strength = 0.0
            hypothesis.status = STATUS_REFUTED
            hypothesis.conclusion = (
                f"REFUTED: Controlling '{hypothesis.variable}' "
                f"increased regret by {abs(regret_delta):.4f}"
            )
        else:
            # No clear effect
            hypothesis.causal_strength = 0.1
            hypothesis.status = STATUS_CORRELATIONAL
            hypothesis.conclusion = (
                f"INCONCLUSIVE: No significant effect from controlling "
                f"'{hypothesis.variable}'"
            )

        op = "strengthen" if hypothesis.status == STATUS_CONFIRMED_CAUSAL else "mutate"
        if hypothesis.status == STATUS_REFUTED:
            op = "remove"
        self._log_cell(hypothesis, op, cause="evaluate_intervention")
        self._save()
        return {
            "hypothesis_id": hypothesis_id,
            "status": hypothesis.status,
            "causal_strength": hypothesis.causal_strength,
            "conclusion": hypothesis.conclusion,
            "intervention": intervention_record,
        }

    # ── Confidence Adjustment ──────────────────────────────────────

    def get_confidence_adjustment(self, hypothesis_id: str) -> float:
        """Get the confidence adjustment for a principle based on causal testing.

        Called by the Library when updating principle confidence:
            - Confirmed causal → boost confidence significantly
            - Correlational → keep or slightly reduce confidence
            - Refuted → sharply reduce confidence

        Returns:
            Confidence multiplier (0.3 to 1.5).
        """
        hypothesis = self._hypotheses.get(hypothesis_id)
        if hypothesis is None:
            return 1.0

        if hypothesis.status == STATUS_CONFIRMED_CAUSAL:
            return 1.0 + hypothesis.causal_strength * 0.5  # Up to 1.475
        elif hypothesis.status == STATUS_CORRELATIONAL:
            return 0.8  # Mild reduction
        elif hypothesis.status == STATUS_REFUTED:
            return 0.3  # Sharp reduction
        else:
            return 1.0  # Untested

    # ── Batch Processing ───────────────────────────────────────────

    def form_hypotheses_from_rules(self, rules: list[dict]) -> list[Hypothesis]:
        """Convert multiple DT-mined rules into hypotheses.

        Filters for rules that are worth testing:
            - Confidence between 0.3 and 0.8 (too low = noise, too high = already proven)
            - Predicted regret above threshold
        """
        hypotheses = []
        for rule in rules:
            confidence = rule.get("confidence", 0)
            if 0.3 <= confidence <= 0.8:
                h = self.form_hypothesis(rule)
                if h is not None:
                    hypotheses.append(h)
        return hypotheses

    def get_pending_hypotheses(self) -> list[Hypothesis]:
        """Get hypotheses that need testing."""
        return [
            h for h in self._hypotheses.values()
            if h.status == STATUS_PENDING
        ]

    def get_testable_hypotheses(self, goal: "object | None" = None) -> list[Hypothesis]:
        """Hypotheses an intervention can actually be designed for.

        Pending, in-flight (testing — un-sticks hypotheses whose
        evaluation stage never completed), and correlational hypotheses
        whose variable has a controllable factor. state.*-only
        hypotheses are excluded by construction.

        LCA-inspired goal prior (2026-07-18): when `goal` provides a
        `hypothesis_score(variable)` method (see cognition/goal_cell.py),
        the returned list is RANKED by that score (higher = more on-goal),
        and hypotheses scoring below a small cutoff are dropped. Nothing
        gets censored unconditionally — a strong goal biases, does not
        override, and a run with no goal is byte-identical to before.
        """
        candidates = [
            h for h in self._hypotheses.values()
            if h.status in (STATUS_PENDING, STATUS_TESTING, STATUS_CORRELATIONAL)
            and controllable_factor(h.variable) is not None
        ]
        if goal is None or not hasattr(goal, "hypothesis_score"):
            return candidates
        scored: list[tuple[float, Hypothesis]] = []
        for h in candidates:
            try:
                s = float(goal.hypothesis_score(h.variable))
            except Exception:
                s = 1.0
            scored.append((s, h))
        # Cutoff at 0.2 — a goal can push a hypothesis into the "not now"
        # bin (score ~0.15) but not deny that it exists. Higher cutoffs
        # risk starving the EIG chooser on runs where every hypothesis is
        # off-goal.
        scored = [(s, h) for s, h in scored if s >= 0.2]
        scored.sort(key=lambda t: (-t[0], t[1].hypothesis_id))
        return [h for _, h in scored]

    def get_testing_hypotheses(self) -> list[Hypothesis]:
        """Get hypotheses currently under test."""
        return [
            h for h in self._hypotheses.values()
            if h.status == STATUS_TESTING
        ]

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        """Load hypotheses from disk, restoring the counter so new IDs
        never collide with persisted ones."""
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        max_counter = 0
        for entry in data:
            h = Hypothesis(
                hypothesis_id=entry["hypothesis_id"],
                variable=entry["variable"],
                direction=entry["direction"],
                threshold=entry["threshold"],
                source_rule=entry.get("source_rule", {}),
                predicted_regret=entry.get("predicted_regret", 0.0),
            )
            h.status = entry.get("status", STATUS_PENDING)
            h.interventions = entry.get("interventions", [])
            h.evidence_trials = entry.get("evidence_trials", [])
            h.conclusion = entry.get("conclusion", "")
            h.causal_strength = entry.get("causal_strength", 0.0)
            self._hypotheses[h.hypothesis_id] = h
            try:
                num = int(h.hypothesis_id.split("-")[-1])
                max_counter = max(max_counter, num)
            except (ValueError, IndexError):
                pass
        self._hypothesis_counter = max_counter
        self._total_interventions = sum(
            len(h.interventions) for h in self._hypotheses.values()
        )

    def _save(self) -> None:
        """Persist all hypotheses to disk."""
        if not self.filepath:
            return
        dirpath = os.path.dirname(self.filepath)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        with open(self.filepath, "w") as f:
            json.dump(
                [h.to_dict() for h in self._hypotheses.values()],
                f, indent=2, default=str,
            )

    # ── Diagnostics ────────────────────────────────────────────────

    def choose_intervention_eig(
        self,
        pending: list[Hypothesis] | None = None,
        budget: int = 1,
    ) -> list[Hypothesis]:
        """Greedy expected-information-gain acquisition (v22 §5.1).

        Picks hypotheses whose intervention is most likely to resolve
        uncertainty — prefers untested, high predicted_regret, diverse variables.
        """
        pool = pending or [
            h for h in self._hypotheses.values()
            if h.status in (STATUS_PENDING, STATUS_TESTING, STATUS_CORRELATIONAL)
        ]
        if not pool:
            return []

        def _eig_score(h: Hypothesis) -> float:
            n_trials = len(h.evidence_trials)
            uncertainty = 1.0 / (1.0 + n_trials)
            regret_signal = min(1.0, float(h.predicted_regret) * 2.0)
            status_bonus = 0.3 if h.status == STATUS_PENDING else 0.0
            return uncertainty * 0.5 + regret_signal * 0.35 + status_bonus

        ranked = sorted(pool, key=_eig_score, reverse=True)
        return ranked[: max(1, budget)]

    def get_summary(self) -> dict:
        """Summary of all hypotheses and causal testing."""
        status_counts = {}
        for h in self._hypotheses.values():
            status_counts[h.status] = status_counts.get(h.status, 0) + 1

        return {
            "total_hypotheses": len(self._hypotheses),
            "total_interventions": self._total_interventions,
            "status_counts": status_counts,
            "confirmed_causal": [
                h.to_dict() for h in self._hypotheses.values()
                if h.status == STATUS_CONFIRMED_CAUSAL
            ],
        }
