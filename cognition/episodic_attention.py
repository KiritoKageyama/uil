"""Episodic Attention — KDE-based retrieval from historical cases.

V20+.8: Resolves the cold-start/bootstrapping problem. When the system
encounters a new dataset, it doesn't start blind — it retrieves the
closest historical cases and uses their configurations as warm-start.

Dual-Encoder Key-Value:
    Key = (static fingerprint + dynamic trajectory signals)
    Value = (configurations + formulas that resolved stress)

Retrieval uses Kernel Density Estimation (KDE) — not hard 1-NN.
The system computes a soft probability distribution over candidate
parameters across top-K matching historical datasets.
"""

from __future__ import annotations
import json
import os
import numpy as np

from figpso_uil.cognition.paths import results_path
from figpso_uil.cognition.version_store import VersionStore


class EpisodicAttention:
    """Episodic memory with soft KDE retrieval for warm-starting.

    Stores episodes (dataset fingerprint → resolution config) and
    retrieves weighted blends of similar historical configurations
    when encountering new datasets.
    """

    # Default parameters
    DEFAULT_BANDWIDTH = 0.3   # KDE bandwidth
    DEFAULT_TOP_K = 5         # Number of nearest episodes to blend

    def __init__(
        self,
        bandwidth: float = DEFAULT_BANDWIDTH,
        top_k: int = DEFAULT_TOP_K,
        filepath: str | None = results_path("uil_episodes_v20.json"),
        version_store: VersionStore | None = None,
    ):
        self.bandwidth = bandwidth
        self.top_k = top_k
        self._episodes: list[dict] = []
        self.filepath = filepath
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

    def _log_episode(self, episode: dict, op: str = "create") -> None:
        store = self._cell_store()
        if store is None:
            return
        eid = f"EP-{len(self._episodes):04d}"
        store.record_cell(
            cell_id=eid,
            cell_type="episode",
            op=op,
            post_state=episode,
            cause="episodic",
        )

    # ── Episode Storage ─────────────────────────────────────────────

    def store_episode(
        self,
        fingerprint: dict,
        trajectory: dict,
        resolution: dict,
    ) -> None:
        """Store a completed optimization episode.

        Args:
            fingerprint: Static dataset dimensions:
                {n_samples, n_features, sparsity, entropy, num_classes, feat_ratio}
            trajectory: Dynamic trajectory signals:
                {max_stress, final_accuracy, stages_to_converge, regret_sum}
            resolution: What worked:
                {best_config, best_weights, formula_id, principle_ids}
        """
        # Build the key vector from fingerprint + trajectory
        key = self._build_key(fingerprint, trajectory)
        if key is None:
            return

        self._episodes.append({
            "key": key,
            "fingerprint": fingerprint,
            "trajectory": trajectory,
            "resolution": resolution,
        })
        self._log_episode(self._episodes[-1])
        self._save()

    # ── KDE Retrieval ───────────────────────────────────────────────

    def retrieve(
        self,
        query_fingerprint: dict,
        query_trajectory: dict | None = None,
    ) -> dict | None:
        """Retrieve a warm-start configuration from historical episodes.

        Uses Gaussian KDE to compute soft similarity weights over
        all stored episodes, then blends the top-K resolutions.

        Args:
            query_fingerprint: Current dataset fingerprint.
            query_trajectory: Current trajectory signals (optional,
                may not be available at cold-start).

        Returns:
            Blended resolution dict, or None if no episodes stored.
        """
        if not self._episodes:
            return None

        query_key = self._build_key(query_fingerprint, query_trajectory or {})
        if query_key is None:
            return None

        # Compute distances to all stored episodes
        distances = []
        for ep in self._episodes:
            d = self._compute_distance(query_key, ep["key"])
            distances.append(d)

        distances = np.array(distances)

        # Gaussian KDE: weight = exp(-d^2 / (2 * h^2))
        weights = np.exp(-distances ** 2 / (2 * self.bandwidth ** 2))

        # Top-K selection
        k = min(self.top_k, len(self._episodes))
        top_indices = np.argsort(-weights)[:k]  # Highest weights first
        top_weights = weights[top_indices]

        # Normalize weights
        weight_sum = top_weights.sum()
        if weight_sum < 1e-10:
            return None
        top_weights /= weight_sum

        # Blend resolutions
        blended = self._blend_resolutions(top_indices, top_weights)
        blended["retrieval_confidence"] = float(top_weights[0])
        blended["episodes_matched"] = k
        return blended

    # ── Key Construction ────────────────────────────────────────────

    def _build_key(
        self,
        fingerprint: dict,
        trajectory: dict,
    ) -> np.ndarray | None:
        """Build a numerical key vector from fingerprint + trajectory.

        Static dimensions (fingerprint):
            n_samples, n_features, sparsity, entropy, num_classes, feat_ratio

        Dynamic dimensions (trajectory):
            max_stress, final_accuracy, stages_to_converge, regret_sum
        """
        static_fields = [
            "n_samples", "n_features", "sparsity", "entropy",
            "num_classes", "feat_ratio",
        ]
        dynamic_fields = [
            "max_stress", "final_accuracy", "stages_to_converge", "regret_sum",
        ]

        values = []
        for field in static_fields:
            val = fingerprint.get(field, 0.0)
            if isinstance(val, (int, float)):
                values.append(float(val))
            else:
                values.append(0.0)

        for field in dynamic_fields:
            val = trajectory.get(field, 0.0)
            if isinstance(val, (int, float)):
                values.append(float(val))
            else:
                values.append(0.0)

        if not values:
            return None

        key = np.array(values, dtype=np.float64)

        # Normalize: z-score each dimension to handle scale differences
        # Use simple min-max scaling per dimension across stored episodes
        if len(self._episodes) > 1:
            all_keys = np.array([ep["key"] for ep in self._episodes])
            mins = all_keys.min(axis=0)
            maxs = all_keys.max(axis=0)
            ranges = maxs - mins
            ranges[ranges < 1e-10] = 1.0
            key = (key - mins) / ranges
        else:
            # No normalization possible with <2 episodes
            norms = np.abs(key)
            norms[norms < 1e-10] = 1.0
            key = key / norms

        return key

    def _compute_distance(
        self, key_a: np.ndarray, key_b: np.ndarray
    ) -> float:
        """Euclidean distance between two key vectors."""
        # Truncate to same length
        min_len = min(len(key_a), len(key_b))
        return float(np.linalg.norm(key_a[:min_len] - key_b[:min_len]))

    # ── Resolution Blending ─────────────────────────────────────────

    def _blend_resolutions(
        self,
        indices: np.ndarray,
        weights: np.ndarray,
    ) -> dict:
        """Blend multiple historical resolutions by weight.

        For numeric config values: weighted average.
        For non-numeric values: use the highest-weighted episode's value.
        """
        blended_config = {}
        blended_weights = None
        best_formula = None
        best_symbolic_formula = None
        best_principles = []

        for idx, w in zip(indices, weights):
            ep = self._episodes[int(idx)]
            resolution = ep["resolution"]

            # Blend config values
            config = resolution.get("best_config", {})
            for key, val in config.items():
                if isinstance(val, (int, float)):
                    if key not in blended_config:
                        blended_config[key] = 0.0
                    blended_config[key] += val * w
                elif key not in blended_config:
                    blended_config[key] = val  # Take first non-numeric

            # Blend feature weights
            ep_weights = resolution.get("best_weights")
            if ep_weights is not None:
                if isinstance(ep_weights, list):
                    ep_weights = np.array(ep_weights, dtype=np.float64)
                if blended_weights is None:
                    blended_weights = np.zeros_like(ep_weights)
                if len(ep_weights) == len(blended_weights):
                    blended_weights += ep_weights * w

            # Take the best episode's formula and principles
            if w == weights.max():
                best_formula = resolution.get("formula_id")
                best_symbolic_formula = resolution.get("symbolic_formula_id")
                best_principles = resolution.get("principle_ids", [])

        result = {"blended_config": blended_config}
        if blended_weights is not None:
            result["blended_weights"] = blended_weights.tolist()
        if best_formula:
            result["suggested_formula"] = best_formula
        if best_symbolic_formula:
            result["suggested_symbolic_formula"] = best_symbolic_formula
        if best_principles:
            result["suggested_principles"] = best_principles

        return result

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        """Load episodes from disk, restoring each "key" vector back to
        an ndarray (JSON only round-trips it as a list)."""
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        for entry in data:
            key = entry.get("key")
            if key is None:
                continue
            self._episodes.append({
                "key": np.array(key, dtype=np.float64),
                "fingerprint": entry.get("fingerprint", {}),
                "trajectory": entry.get("trajectory", {}),
                "resolution": entry.get("resolution", {}),
            })

    def _save(self) -> None:
        """Persist all episodes to disk ("key" ndarrays as plain lists)."""
        if not self.filepath:
            return
        dirpath = os.path.dirname(self.filepath)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        serializable = [
            {
                "key": ep["key"].tolist(),
                "fingerprint": ep["fingerprint"],
                "trajectory": ep["trajectory"],
                "resolution": ep["resolution"],
            }
            for ep in self._episodes
        ]
        with open(self.filepath, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

    # ── Routing memory (merged from domain_router.RoutingMemory) ────

    def store_routing_outcome(
        self,
        fingerprint: dict,
        strategy: str,
        score: float,
    ) -> None:
        """Record which learner won on a fingerprint region."""
        self._episodes.append({
            "key": self._build_key(fingerprint, {"routing_score": score}),
            "fingerprint": fingerprint,
            "trajectory": {"routing": True},
            "resolution": {"strategy": strategy, "score": float(score)},
        })
        self._log_episode(self._episodes[-1], op="mutate")
        self._save()

    def recommend_strategy(self, fingerprint: dict, top_k: int = 6) -> dict | None:
        """kNN over routing episodes — returns best strategy if confident."""
        routing_eps = [
            ep for ep in self._episodes
            if ep.get("trajectory", {}).get("routing")
        ]
        if len(routing_eps) < 1:
            return None
        q = self._build_key(fingerprint, {})
        if q is None:
            return None
        ranked = sorted(
            routing_eps,
            key=lambda ep: float(np.linalg.norm(q - ep["key"])),
        )[:top_k]
        from collections import defaultdict
        agg: dict[str, list] = defaultdict(list)
        for ep in ranked:
            agg[ep["resolution"]["strategy"]].append(ep["resolution"]["score"])
        means = sorted(
            ((s, float(np.mean(v))) for s, v in agg.items()),
            key=lambda t: t[1], reverse=True,
        )
        if len(means) < 1:
            return None
        if len(means) >= 2 and means[0][1] - means[1][1] < 0.03:
            return None
        return {"strategy": means[0][0], "mean_score": means[0][1], "support": len(ranked)}

    # ── Diagnostics ────────────────────────────────────────────────

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    def get_summary(self) -> dict:
        return {
            "episodes": self.episode_count,
            "bandwidth": self.bandwidth,
            "top_k": self.top_k,
        }
