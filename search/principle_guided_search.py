"""Principle-Guided Search — Deep search governed by stress level.

The DEFAULT search strategy. Each proposed configuration is a targeted,
informed perturbation — not random exploration.

Stress level continuously scales aggressiveness:
- Low stress → small perturbations around current best
- Medium stress → principle-guided targeted changes
- High stress → aggressive, diverse configurations
- Critical → full reset

Deep search, not shallow. Each configuration is deeply evaluated.

UIL hardcoded-value audit, Category C (2026-07-23): this docstring used to
say "4 configurations max" while propose_actions() capped at `actions[:6]`
-- a real documentation/behavior mismatch, not a bug in the code. The 4-max
figure describes only the original v19 stress-scaled actions (1-4, below);
V20+.10 and V20+.11 later added two cognition-driven action types
(episodic-guided, network-guided) without the docstring being updated.
propose_actions() returns at most 6 today: up to 4 stress-scaled actions,
plus up to 4 bootstrap-diversified actions on low-data datasets, plus the
episodic- and network-guided actions when their inputs are available --
capped at 6 total.
"""

from __future__ import annotations
import numpy as np

from figpso_uil.core.utils import apply_guidance_feedback


class PrincipleGuidedSearch:
    """Deep search strategy guided by principles and stress.
    
    Instead of exploring thousands of random configurations (AutoML),
    this strategy makes 1-4 targeted, informed changes per stage.
    
    Like a human expert: try one thing, observe, think about WHY,
    make a targeted adjustment, repeat.
    """

    # Opt-in (optimizer sets it from cfg.fix_guidance_ratchet). Default OFF
    # until cross-seed validated -- see the E6 lesson in CLAUDE.md: an
    # unvalidated "improvement" regressed ionosphere 0.916 -> 0.868.
    fix_guidance_ratchet: bool = False
    GUIDANCE_FLOOR: float = 0.02

    def __init__(self):
        self._rng: np.random.Generator | None = None
        self._base_guidance: np.ndarray | None = None
        self._best_weights: np.ndarray | None = None
        self._best_score: float = -np.inf
        self._n_features: int = 0
        self._worst_weights: np.ndarray | None = None
        self._worst_score: float = np.inf
        self._controllable_params: dict | None = None  # V20
        # V20+.10: Cognition-driven search inputs
        self._episodic_hint: dict | None = None
        self._network_overrides: dict | None = None
        # V20+.11: Feature correlation matrix
        self._correlation_matrix: np.ndarray | None = None
        self._propagation_strength: float = 0.15
        self._bootstrap_weights: list[np.ndarray] = []
        self._low_data: bool = False
        self._warm_start_weights: np.ndarray | None = None

    def initialize(
        self,
        n_features: int,
        base_guidance: np.ndarray,
        seed: int,
        n_particles: int = 20,
        pso_iters: int = 30,
    ) -> None:
        """Initialize the search."""
        self._rng = np.random.default_rng(seed)
        self._base_guidance = base_guidance.copy()
        self._best_weights = base_guidance.copy()
        self._n_features = n_features

    def set_bootstrap_diversity(
        self,
        X: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """Seed extra search actions from bootstrapped mini-batches (tiny data)."""
        from figpso_uil.core.small_data_diversify import bootstrap_guidance_weights

        n = len(y)
        self._low_data = n < 80
        if not self._low_data:
            self._bootstrap_weights = []
            return
        n_seeds = max(3, min(8, 80 // max(n, 1)))
        self._bootstrap_weights = bootstrap_guidance_weights(X, y, rng, n_seeds)

    def propose_actions(
        self,
        state: dict,
        principles: list[dict],
        stress_level: float,
        journal_recent: list[dict],
    ) -> list[dict]:
        """Propose 1-4 targeted actions based on stress level.
        
        Stress continuously scales aggressiveness. No discrete buckets.
        """
        if self._rng is None or self._base_guidance is None:
            return []

        actions = []

        # Number of configurations scales with stress
        num_configs = 1 + int(stress_level * 3)  # 1 at calm → 4 at max

        # Perturbation scale scales with stress
        pert_scale = 0.05 + stress_level * 0.45  # 0.05 at calm → 0.50 at max

        # Calculate penalty (may be overridden by principles)
        max_penalty = 3.0
        for p in principles:
            override = p.get("action_override", {})
            if "max_penalty" in override:
                max_penalty = min(max_penalty, override["max_penalty"])

        # Base penalty scales with stress (more aggressive when stressed)
        base_penalty = 1.0 + (1.0 - stress_level) * (max_penalty - 1.0)

        # ── Action 1: Perturbation around best known ──
        if self._best_weights is not None:
            perturbed = self._perturb(self._best_weights, pert_scale * 0.5)
            actions.append({
                "feature_weights": perturbed,
                "penalty": float(base_penalty),
                "classifier_config": self._propose_config(stress_level, 0),
                "description": "Perturbation around best",
            })

        # ── Action 2: Guidance-informed (if stress warrants) ──
        if num_configs >= 2:
            blend = 0.3 + stress_level * 0.4
            guided = (1 - blend) * self._best_weights + blend * self._base_guidance
            guided = np.clip(guided, 0, 1)
            actions.append({
                "feature_weights": guided,
                "penalty": float(base_penalty * 0.8),
                "classifier_config": self._propose_config(stress_level, 1),
                "description": "Guidance-blended",
            })

        # ── Action 3: Step AWAY from the worst candidate (moderate+ stress) ──
        # Was `1 - worst_weights`: in a space of dozens of features the
        # mirror image of a bad point is not a good point, just a different
        # arbitrary one. What the name promises is more useful and costs the
        # same -- take the best known vector and step further along the
        # direction that separates it from the loser.
        if (
            num_configs >= 3
            and self._worst_weights is not None
            and self._best_weights is not None
            and len(self._worst_weights) == len(self._best_weights)
        ):
            direction = self._best_weights - self._worst_weights
            if float(np.max(np.abs(direction))) > 1e-9:
                step = 0.3 + 0.5 * stress_level
                away = np.clip(self._best_weights + step * direction, 0, 1)
            else:
                away = np.clip(self._base_guidance.copy(), 0, 1)
            actions.append({
                "feature_weights": away,
                "penalty": float(max_penalty * 0.7),
                "classifier_config": self._propose_config(stress_level, 2),
                "description": "Away from worst candidate",
            })

        # ── Action 4: Aggressive random restart (high stress only) ──
        if num_configs >= 4:
            random_weights = self._rng.uniform(0.0, 1.0, self._n_features)
            restart = 0.3 * random_weights + 0.7 * self._base_guidance
            restart = np.clip(restart, 0, 1)
            actions.append({
                "feature_weights": restart,
                "penalty": float(max_penalty * 0.5),
                "classifier_config": self._propose_config(stress_level, 3),
                "description": "Aggressive restart",
            })

        # ── Bootstrap-diversified actions (low-data datasets) ──
        for i, bw in enumerate(self._bootstrap_weights[:4]):
            blend = 0.55 * bw + 0.45 * self._best_weights
            blend = np.clip(blend, 0, 1)
            actions.append({
                "feature_weights": blend,
                "penalty": float(base_penalty * 0.9),
                "classifier_config": self._propose_config(stress_level, i % 4),
                "description": f"Bootstrap seed {i + 1} (low-data diversify)",
            })

        # V20+.10: Cognition-driven actions (PSO-free, reason-based)

        # ── Action 5: Episodic-guided (warm-start from similar case) ──
        if self._episodic_hint is not None:
            hint_weights = self._episodic_hint.get("blended_weights")
            if hint_weights is not None:
                hint = np.array(hint_weights, dtype=np.float64)
                if len(hint) == self._n_features:
                    # Blend with current best: 60% hint, 40% best
                    blended = 0.6 * hint + 0.4 * self._best_weights
                    blended = np.clip(blended, 0, 1)
                    actions.append({
                        "feature_weights": blended,
                        "penalty": float(base_penalty),
                        "classifier_config": self._episodic_hint.get(
                            "blended_config", {}
                        ),
                        "description": "Episodic-guided (from similar dataset)",
                    })

        # ── Action 6: Network-guided (from graph traversal) ──
        if self._network_overrides:
            network_weights = self._best_weights.copy()
            # Apply any feature-weight overrides from network traversal
            for key, val in self._network_overrides.items():
                if key.startswith("weight_") and isinstance(val, (int, float)):
                    try:
                        idx = int(key.replace("weight_", ""))
                        if 0 <= idx < self._n_features:
                            network_weights[idx] = float(val)
                    except ValueError:
                        pass
            network_weights = np.clip(network_weights, 0, 1)
            if not np.array_equal(network_weights, self._best_weights):
                actions.append({
                    "feature_weights": network_weights,
                    "penalty": float(self._network_overrides.get(
                        "max_penalty", base_penalty
                    )),
                    "classifier_config": {},
                    "description": "Network-guided (from graph traversal)",
                })

        # V20+.11: Apply correlation propagation to all actions
        if self._correlation_matrix is not None:
            for action in actions:
                action["feature_weights"] = self._apply_correlation_propagation(
                    action["feature_weights"]
                )

        return actions[:6]  # Max 6 configs

    def set_controllable_params(self, params: dict) -> None:
        """V20: Register the classifier's controllable parameters."""
        self._controllable_params = params

    def update_feedback(
        self,
        importance: np.ndarray,
        best_weights: np.ndarray,
        best_score: float,
        entropy: float,
        worst_weights: np.ndarray | None = None,
        worst_score: float | None = None,
    ) -> None:
        """Update with feedback from the optimizer."""
        # Track best
        if best_score > self._best_score:
            self._best_score = best_score
            self._best_weights = best_weights.copy()

        # Track worst. The optimizer now reports the lowest-scoring CANDIDATE
        # of the stage. The old fallback -- treating the stage WINNER as
        # "worst" whenever the stage scored badly -- recorded a good vector as
        # a bad one (and made best == worst on stage 1), so it is no longer
        # used: absent a real loser we simply keep whatever we last had.
        if worst_weights is not None and worst_score is not None:
            if worst_score < self._worst_score:
                self._worst_score = float(worst_score)
                self._worst_weights = np.asarray(worst_weights, dtype=float).copy()

        # Update guidance with classifier importance.
        # UIL hardcoded-value audit, Category C: this formula used to be a
        # verbatim copy of PSOSearch.update_feedback's, each carrying its
        # own copy of a comment warning the other copy needed the same fix.
        # Both now call the one shared implementation -- see
        # apply_guidance_feedback's docstring for what the two modes do.
        if self._base_guidance is not None:
            self._base_guidance = apply_guidance_feedback(
                self._base_guidance, importance, entropy,
                fix_guidance_ratchet=self.fix_guidance_ratchet,
                guidance_floor=self.GUIDANCE_FLOOR,
            )

    # ── Internal ─────────────────────────────────────────────────────

    def _perturb(self, weights: np.ndarray, scale: float) -> np.ndarray:
        """Perturb weights with Gaussian noise."""
        noise = self._rng.normal(0, scale, size=weights.shape)
        return np.clip(weights + noise, 0.0, 1.0)

    def _propose_config(self, stress_level: float, variant: int) -> dict:
        """V20: Propose classifier config changes based on stress level.
        
        Low stress: no changes (empty dict = use defaults)
        Medium+: vary hyperparameters with increasing aggressiveness.
        """
        if self._controllable_params is None or stress_level < 0.2:
            return {}

        config = {}
        perturbation_scale = stress_level * 0.5  # 0 at calm → 0.5 at max

        for param_name, meta in self._controllable_params.items():
            param_type = meta.get("type", "float")
            current = meta.get("current")

            if current is None:
                continue

            # Vary parameters based on stress and variant index
            if param_type in ("int", "float"):
                pmin, pmax = meta.get("min", 0), meta.get("max", 1)
                range_size = pmax - pmin
                noise = self._rng.normal(0, perturbation_scale * range_size * 0.3)
                
                # Different variants try different directions
                if variant == 1:
                    noise = abs(noise)  # try increasing
                elif variant == 2:
                    noise = -abs(noise)  # try decreasing
                # variant 0, 3: random direction

                new_val = current + noise
                if param_type == "int":
                    new_val = int(np.clip(round(new_val), pmin, pmax))
                else:
                    new_val = float(np.clip(new_val, pmin, pmax))
                
                if new_val != current:
                    config[param_name] = new_val
            
            elif param_type == "categorical":
                options = meta.get("options", [])
                if options and stress_level > 0.5 and self._rng.random() < 0.3:
                    config[param_name] = self._rng.choice(options)

            elif param_type == "str":
                options = meta.get("options", [])
                if not options:
                    continue
                if param_name == "learning_mode":
                    if stress_level >= 0.55 and variant >= 2:
                        config[param_name] = "multi"
                    elif stress_level >= 0.35 and variant == 1:
                        config[param_name] = self._rng.choice(["boosting", "bagging", "multi"])
                    elif stress_level >= 0.2 and self._rng.random() < 0.25:
                        config[param_name] = self._rng.choice(options)
                elif stress_level > 0.5 and self._rng.random() < 0.3:
                    config[param_name] = self._rng.choice(options)

        return config

    # ── V20+.10: Cognition Inputs ────────────────────────────────────

    def set_episodic_hint(self, hint: dict | None) -> None:
        """V20+.10: Set the episodic attention warm-start hint.

        Called by the optimizer after episodic retrieval.
        """
        self._episodic_hint = hint
        if hint and hint.get("best_weights"):
            self.set_warm_start_weights(np.asarray(hint["best_weights"], dtype=float))

    def set_warm_start_weights(self, weights: np.ndarray | None) -> None:
        """Seed search around prior champion or episodic best weights."""
        if weights is None:
            return
        w = np.asarray(weights, dtype=float)
        if self._n_features and len(w) == self._n_features:
            self._best_weights = np.clip(w, 0.0, 1.0)
            self._warm_start_weights = self._best_weights.copy()

    def set_network_overrides(self, overrides: dict | None) -> None:
        """V20+.10: Set overrides from network graph traversal.

        Called by the optimizer after network traversal.
        """
        self._network_overrides = overrides

    # ── V20+.11: Feature Correlation Propagation ─────────────────────

    def set_correlation_matrix(
        self, X: np.ndarray, strength: float = 0.15
    ) -> None:
        """V20+.11: Compute and store the feature correlation matrix.

        When a principle alters weight_i, correlated features weight_j
        are propagated by C_{i,j} * strength.

        Args:
            X: Training data (n_samples, n_features).
            strength: How much to propagate (0 = none, 1 = full).
        """
        if X.shape[1] < 2:
            self._correlation_matrix = None
            return

        self._propagation_strength = np.clip(strength, 0.0, 0.5)
        # Compute absolute Pearson correlation
        try:
            corr = np.corrcoef(X.T)
            corr = np.nan_to_num(np.abs(corr), nan=0.0)
            # Zero diagonal — don't self-propagate
            np.fill_diagonal(corr, 0.0)
            self._correlation_matrix = corr
        except Exception:
            self._correlation_matrix = None

    def _apply_correlation_propagation(
        self, weights: np.ndarray
    ) -> np.ndarray:
        """V20+.11: Propagate feature weight changes to correlated features.

        If weight_i was changed from the base, propagate a fraction
        of that change to correlated features.
        """
        if self._correlation_matrix is None:
            return weights
        if self._base_guidance is None:
            return weights
        if len(weights) != len(self._correlation_matrix):
            return weights

        delta = weights - self._base_guidance
        # Propagate: each feature gets sum of (corr * delta) scaled by strength
        propagated = self._correlation_matrix @ delta * self._propagation_strength
        result = weights + propagated
        return np.clip(result, 0.0, 1.0)
