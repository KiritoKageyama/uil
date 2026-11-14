"""SearchStrategy Protocol — the interface any search strategy must implement.

The UIL optimizer uses search strategies to propose candidate feature weight
vectors. Different strategies can be plugged in:
- PSOSearch: Particle Swarm Optimization (simple, static parameters)
- PrincipleGuidedSearch: Deep search governed by stress level (1-4 configs)

The search strategy is NOT responsible for stress management or principle
discovery. It only proposes actions. The optimizer handles everything else.
"""

from __future__ import annotations
from typing import Protocol
import numpy as np


class SearchStrategy(Protocol):
    """Protocol for pluggable search strategies.
    
    A search strategy proposes 1-4 candidate actions (feature weight vectors)
    at each stage. The optimizer evaluates them with the classifier and
    feeds back the results.
    """

    def initialize(
        self,
        n_features: int,
        base_guidance: np.ndarray,
        seed: int,
        n_particles: int = 20,
        pso_iters: int = 30,
    ) -> None:
        """Initialize the search strategy.

        Args:
            n_features: Number of features in the dataset.
            base_guidance: Initial feature guidance from filter metric (MI).
                Shape (n_features,), values in [0, 1].
            seed: Random seed for reproducibility.
            n_particles: Number of particles (for PSO-based strategies).
            pso_iters: Iterations per stage (for PSO-based strategies).
        """
        ...

    def propose_actions(
        self,
        state: dict,
        principles: list[dict],
        stress_level: float,
        journal_recent: list[dict],
    ) -> list[dict]:
        """Propose 1-4 candidate actions to evaluate.

        Each action is a dict containing at minimum:
            {
                "feature_weights": np.ndarray,  # shape (n_features,), [0, 1]
                "penalty": float,               # penalty value used
                "description": str,             # human-readable label
            }

        The optimizer will evaluate each action by training the classifier
        and measuring accuracy. At most 4 actions — deep search, not shallow.

        Args:
            state: Current state vector (n_samples, n_features, entropy, etc.)
            principles: Active principles from the library.
            stress_level: Current system stress level in [0, 1].
            journal_recent: Recent journal entries for context.

        Returns:
            List of 1-4 action dicts.
        """
        ...

    def update_feedback(
        self,
        importance: np.ndarray,
        best_weights: np.ndarray,
        best_score: float,
        entropy: float,
    ) -> None:
        """Receive feedback from the optimizer after evaluation.

        The search strategy uses this to update its internal state
        (e.g., PSO updates gbest, principle-guided search updates its
        perturbation center).

        Args:
            importance: Feature importance from the classifier, shape (n_features,).
            best_weights: The feature weights that produced the best score.
            best_score: The best score achieved this stage.
            entropy: Current swarm/search diversity metric.
        """
        ...
