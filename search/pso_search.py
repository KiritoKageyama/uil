"""PSO Search Strategy — Particle Swarm Optimization as a simple, static option.

PSO is preserved as ONE search strategy with FIXED internal parameters.
No stress-scaling. No dynamic particle counts. Simple and deterministic.

The UIL decides WHEN to use PSO, but PSO runs with its own fixed logic.
"""

from __future__ import annotations
import numpy as np

from figpso_uil.core.utils import apply_guidance_feedback


class PSOSearch:
    """Particle Swarm Optimization search strategy.
    
    Simple, static PSO. No stress-scaling internally.
    Returns the top 1-4 solutions from the swarm.
    """

    # Opt-in, set by the optimizer from cfg.fix_guidance_ratchet. Default OFF.
    fix_guidance_ratchet: bool = False
    GUIDANCE_FLOOR: float = 0.02

    def __init__(self):
        self._initialized = False
        self.Xp: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.pbest: np.ndarray | None = None
        self.pbest_fit: np.ndarray | None = None
        self.gbest: np.ndarray | None = None
        self.gbest_fit: float = -np.inf
        self._rng: np.random.Generator | None = None
        self._base_guidance: np.ndarray | None = None
        self._n_particles: int = 20
        self._pso_iters: int = 30

    def initialize(
        self,
        n_features: int,
        base_guidance: np.ndarray,
        seed: int,
        n_particles: int = 20,
        pso_iters: int = 30,
    ) -> None:
        """Initialize the PSO swarm."""
        self._rng = np.random.default_rng(seed)
        self._n_particles = n_particles
        self._pso_iters = pso_iters
        self._base_guidance = base_guidance.copy()

        p = n_particles
        self.Xp = self._rng.uniform(0.0, 1.0, size=(p, n_features))
        self.V = self._rng.normal(0.0, 0.2, size=(p, n_features))

        # Inject base_guidance into some particles
        for i in range(p // 3):
            self.Xp[i] = np.clip(
                base_guidance + self._rng.normal(0, 0.1, n_features), 0, 1
            )

        self.pbest = self.Xp.copy()
        self.pbest_fit = np.full(p, -np.inf)
        self.gbest = self.Xp[0].copy()
        self.gbest_fit = -np.inf
        self._initialized = True

    def propose_actions(
        self,
        state: dict,
        principles: list[dict],
        stress_level: float,
        journal_recent: list[dict],
    ) -> list[dict]:
        """Run PSO iterations and return top solutions.
        
        Returns 1-4 candidate feature weight vectors.
        """
        if not self._initialized or self.Xp is None:
            return []

        rng = self._rng
        guidance = self._base_guidance
        n_features = self.Xp.shape[1]

        # Calculate adaptive penalty (capped by overrides from principles)
        center = self.Xp.mean(axis=0)
        diversity = float(np.mean(np.abs(self.Xp - center)))
        normalized_entropy = float(np.clip(diversity / 0.5, 0.0, 1.0))

        adaptive_penalty = 1.0 + 2.0 * (1.0 - normalized_entropy)
        max_penalty = 3.0  # Default, may be overridden

        for p_dict in principles:
            override = p_dict.get("action_override", {})
            if "max_penalty" in override:
                max_penalty = min(max_penalty, override["max_penalty"])
        adaptive_penalty = min(adaptive_penalty, max_penalty)

        # Run PSO iterations
        for _t in range(self._pso_iters):
            # Fitness = relevance - sparsity_penalty + diversity
            relevance = np.dot(self.Xp, guidance)
            sparsity = adaptive_penalty * np.sum(self.Xp, axis=1) / n_features
            distinctness = np.mean(np.abs(self.Xp - center), axis=1)
            fitness = relevance - sparsity + 0.1 * distinctness

            # Update personal best
            improved = fitness > self.pbest_fit
            self.pbest_fit[improved] = fitness[improved]
            self.pbest[improved] = self.Xp[improved]

            # Update global best
            best_idx = int(np.argmax(self.pbest_fit))
            if self.pbest_fit[best_idx] > self.gbest_fit:
                self.gbest_fit = float(self.pbest_fit[best_idx])
                self.gbest = self.pbest[best_idx].copy()

            # Velocity and position update (fixed parameters)
            r1, r2 = rng.random((2, self._n_particles, n_features))
            self.V = (
                0.7 * self.V
                + 1.5 * r1 * (self.pbest - self.Xp)
                + 1.5 * r2 * (self.gbest - self.Xp)
            )
            self.Xp = np.clip(self.Xp + self.V, 0.0, 1.0)

        # Return top 1-4 solutions
        all_fitness = np.dot(self.Xp, guidance) - adaptive_penalty * np.sum(
            self.Xp, axis=1
        ) / n_features
        top_indices = np.argsort(all_fitness)[-4:][::-1]

        actions = []
        for idx in top_indices:
            actions.append(
                {
                    "feature_weights": self.Xp[idx].copy(),
                    "penalty": float(adaptive_penalty),
                    "description": f"PSO particle {idx}",
                }
            )
        return actions

    def update_feedback(
        self,
        importance: np.ndarray,
        best_weights: np.ndarray,
        best_score: float,
        entropy: float,
        worst_weights: np.ndarray | None = None,
        worst_score: float | None = None,
    ) -> None:
        """Update base guidance with classifier importance feedback."""
        if self._base_guidance is None:
            return

        # UIL hardcoded-value audit, Category C: this formula used to be a
        # verbatim copy of PrincipleGuidedSearch.update_feedback's. Both now
        # call the one shared implementation -- see apply_guidance_feedback's
        # docstring for what the two modes do and why.
        self._base_guidance = apply_guidance_feedback(
            self._base_guidance, importance, entropy,
            fix_guidance_ratchet=self.fix_guidance_ratchet,
            guidance_floor=self.GUIDANCE_FLOOR,
        )
