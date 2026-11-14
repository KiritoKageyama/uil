"""Architecture Race — replaces surrogate optimization with a real race.

WHY THIS EXISTS
---------------
`pso_search.propose_actions` scores a candidate feature mask with

    fitness = Xp @ guidance - penalty * count / n_features

which is LINEAR AND SEPARABLE in the binary mask: the optimum is
"include feature j iff guidance[j] > penalty / n_features". There is no
interaction term, no local optimum, nothing for a swarm to discover --
20 particles x 30 iterations were re-deriving an argsort. Worse, the
whole cognitive layer's only channel into the learner is `max_penalty`,
i.e. WHERE that sorted list gets cut (see bug #25 in CLAUDE.md), so even
a perfectly wired cognitive layer could only ever nudge one scalar.

This module replaces that with the design the surrogate was always a
stand-in for:

1. `frontier_masks()` -- the surrogate's optimum is closed-form, so the
   ENTIRE cutoff frontier is one argsort. No particles, strictly more
   coverage than sampling 4 points near one cutoff.

2. `ArchitectureRace.propose()` -- candidates vary the axes the learner
   CANNOT choose for itself (L1 strength, basis families, boosting
   depth), because the L1 fit inside `_fit_sparse_linear` already selects
   features optimally along its own regularization path. Feature masks
   are largely redundant with the learner; architecture is not.

3. Successive halving (driven by `rungs()`, executed by the optimizer)
   -- candidates are judged by REAL held-out accuracy, cheaply: most die
   on a small subsample before anyone pays for a full fit. This is the
   "rough first pass, then accuracy decides" the search was designed
   around.

4. `ArmMemory` -- a per-fingerprint Beta posterior over which arm wins.
   As a fingerprint bucket accumulates outcomes the posterior sharpens
   and the race shrinks (12 arms -> 6 -> 3 -> 1). That is a MEASURABLE
   detachment criterion, replacing the rolling-accuracy heuristic that
   caused bug #4.

Nothing here fits a model or mutates knowledge state; it proposes
candidates and records outcomes. The optimizer owns all fitting.
"""

from __future__ import annotations

import json
import math
import os
import threading

import numpy as np

from figpso_uil.cognition.paths import results_path

# Below this many training rows a subsampled rung is pure noise (sonar is
# n=208), so the race degrades to a single full-fidelity round.
RACE_MIN_ROWS = 300

# Never subsample below this many rows on an elimination rung.
RUNG_MIN_ROWS = 150

# A rung must be at most this fraction of the training set (otherwise it
# costs about as much as the full fit it was supposed to avoid) and at
# least this multiple of the previous rung (otherwise it re-measures the
# same thing at the same noise level).
MAX_RUNG_FRACTION = 0.6
MIN_RUNG_GROWTH = 1.5

# (data_fraction, survivor_fraction) for each elimination rung. The final
# full-data rung is implicit and always runs.
DEFAULT_RUNGS: tuple[tuple[float, float], ...] = ((0.15, 0.5), (0.40, 0.5))

# A race with fewer arms than this has nothing to eliminate.
MIN_ARMS_FOR_RACE = 4

# Arms surviving into the final full-fidelity rung.
MIN_FINAL_ARMS = 2

# Phase-out schedule: (min trials observed in this bucket, max arms).
# As evidence accumulates the system proposes fewer candidates -- it
# "becomes an expert" by needing fewer trials, not by asserting maturity.
PHASEOUT_LADDER: tuple[tuple[int, int], ...] = (
    (0, 12),
    (20, 8),
    (60, 5),
    (150, 3),
    (400, 2),
)


# ── 1. Closed-form frontier ──────────────────────────────────────────


def frontier_masks(
    guidance: np.ndarray,
    n_masks: int = 3,
    *,
    include_full: bool = True,
) -> list[np.ndarray]:
    """Every distinct cutoff of the sorted guidance vector, in closed form.

    The PSO surrogate's optimum for ANY penalty is a threshold on
    `guidance`, so the reachable set of "best" masks is exactly the set
    of prefixes of the descending sort. We return `n_masks` of them
    spread across the frontier instead of running a swarm to rediscover
    one point on it.

    Weights are kept continuous (magnitudes preserved above the cutoff,
    zeroed below) because `fit_weighted` consumes real weights, not a
    binary mask.
    """
    g = np.asarray(guidance, dtype=np.float64).ravel()
    n = g.size
    if n == 0:
        return []
    if n_masks <= 0:
        return []

    masks: list[np.ndarray] = []
    if include_full:
        masks.append(g.copy())

    positive = g[g > 0.0]
    if positive.size == 0:
        # Degenerate guidance (all zero) -- a cutoff sweep is meaningless.
        return masks or [g.copy()]

    order = np.sort(positive)[::-1]  # descending
    remaining = max(0, n_masks - len(masks))
    if remaining == 0:
        return masks

    # Keep-counts spread across the frontier, never emptying the mask.
    max_keep = order.size
    keeps = np.unique(
        np.clip(
            np.linspace(max(1, max_keep // 4), max_keep, remaining + 1)[:-1]
            .round()
            .astype(int),
            1,
            max_keep,
        )
    )
    for k in keeps:
        thr = float(order[int(k) - 1])
        w = g.copy()
        w[g < thr] = 0.0
        if not np.any(w > 0):
            continue
        if any(np.array_equal(w, m) for m in masks):
            continue
        masks.append(w)

    return masks


# ── 2. Successive-halving schedule ───────────────────────────────────


def rungs(
    n_train: int,
    n_arms: int,
    *,
    schedule: tuple[tuple[float, float], ...] = DEFAULT_RUNGS,
) -> list[tuple[int, int]]:
    """Plan the elimination rungs as [(n_rows, n_survivors), ...].

    The final full-data rung is NOT included -- the optimizer always runs
    it. An empty list means "no elimination, evaluate everything at full
    fidelity", which is what tiny datasets and small arm counts get.
    """
    if n_train < RACE_MIN_ROWS or n_arms < MIN_ARMS_FOR_RACE:
        return []

    plan: list[tuple[int, int]] = []
    survivors = n_arms
    prev_rows = 0
    for frac, keep_frac in schedule:
        rows = int(max(RUNG_MIN_ROWS, round(n_train * frac)))
        # A rung only earns its keep if it is genuinely CHEAPER than the
        # full fit and genuinely MORE INFORMATIVE than the rung before
        # it. Without these guards the RUNG_MIN_ROWS floor collapses the
        # schedule on mid-sized data -- n=480 produced rungs of 150 and
        # 192 rows, i.e. a second full round of fits for ~no new signal.
        if rows > n_train * MAX_RUNG_FRACTION:
            break
        if prev_rows and rows < prev_rows * MIN_RUNG_GROWTH:
            continue
        keep = int(max(MIN_FINAL_ARMS, math.ceil(survivors * keep_frac)))
        if keep >= survivors:
            continue
        plan.append((rows, keep))
        survivors = keep
        prev_rows = rows
        if survivors <= MIN_FINAL_ARMS:
            break
    return plan


def stratified_subsample(
    y: np.ndarray, n_rows: int, rng: np.random.Generator
) -> np.ndarray:
    """Row indices for a class-proportional subsample.

    Every class present keeps at least one row, so an elimination rung
    can never hand a learner a single-class problem (which would make the
    whole rung's ranking meaningless).
    """
    y = np.asarray(y).ravel()
    n = y.size
    if n_rows >= n:
        return np.arange(n)

    classes, counts = np.unique(y, return_counts=True)
    picks: list[np.ndarray] = []
    for cls, cnt in zip(classes, counts):
        idx = np.flatnonzero(y == cls)
        take = int(max(1, round(n_rows * cnt / n)))
        take = min(take, idx.size)
        picks.append(rng.choice(idx, size=take, replace=False))
    out = np.concatenate(picks)
    rng.shuffle(out)
    return out


# ── 3. Per-fingerprint arm memory (the phase-out) ────────────────────


class ArmMemory:
    """Beta posterior over "does this arm win?", keyed by fingerprint.

    Persisted so confidence -- and therefore the phase-out -- carries
    across runs, which is the whole point: the system should need fewer
    trials the more it has seen of a given kind of data.
    """

    def __init__(self, filepath: str | None = "results/uil_race_memory.json"):
        if filepath == "results/uil_race_memory.json":
            filepath = results_path("uil_race_memory.json")
        self.filepath = filepath
        self._lock = threading.RLock()
        self._arms: dict[str, dict[str, float]] = {}
        self._load()

    # -- posterior --------------------------------------------------

    @staticmethod
    def _key(bucket: str, arm: str) -> str:
        return f"{bucket}||{arm}"

    def stats(self, bucket: str, arm: str) -> dict[str, float]:
        rec = self._arms.get(self._key(bucket, arm))
        if rec is None:
            return {"wins": 0.0, "trials": 0.0}
        return dict(rec)

    def posterior_mean(self, bucket: str, arm: str) -> float:
        """Beta(1,1) prior => 0.5 for an unseen arm (no false confidence)."""
        rec = self.stats(bucket, arm)
        return (rec["wins"] + 1.0) / (rec["trials"] + 2.0)

    def bucket_trials(self, bucket: str) -> int:
        """Rounds of evidence in this bucket, not arm-trials.

        Measured as the max per-arm trial count: the incumbent races
        every round, so its count IS the number of rounds. Summing across
        arms instead would inflate evidence by the arm count and phase
        the race out roughly an order of magnitude too early.
        """
        prefix = f"{bucket}||"
        counts = [
            r["trials"] for k, r in self._arms.items() if k.startswith(prefix)
        ]
        return int(max(counts)) if counts else 0

    def arm_budget(self, bucket: str, ceiling: int) -> int:
        """How many arms to race, given what we already know here."""
        seen = self.bucket_trials(bucket)
        allowed = PHASEOUT_LADDER[0][1]
        for min_trials, max_arms in PHASEOUT_LADDER:
            if seen >= min_trials:
                allowed = max_arms
        return int(max(1, min(ceiling, allowed)))

    def record(self, bucket: str, arm: str, won: bool) -> None:
        with self._lock:
            key = self._key(bucket, arm)
            rec = self._arms.setdefault(key, {"wins": 0.0, "trials": 0.0})
            rec["trials"] += 1.0
            if won:
                rec["wins"] += 1.0
            self._save()

    def record_round(self, bucket: str, arms: list[str], winner: str) -> None:
        with self._lock:
            for arm in arms:
                key = self._key(bucket, arm)
                rec = self._arms.setdefault(key, {"wins": 0.0, "trials": 0.0})
                rec["trials"] += 1.0
                if arm == winner:
                    rec["wins"] += 1.0
            self._save()

    # -- persistence ------------------------------------------------

    def _load(self) -> None:
        if not self.filepath or not os.path.exists(self.filepath):
            return
        try:
            with open(self.filepath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self._arms = {
                    k: {
                        "wins": float(v.get("wins", 0.0)),
                        "trials": float(v.get("trials", 0.0)),
                    }
                    for k, v in data.get("arms", {}).items()
                    if isinstance(v, dict)
                }
        except Exception:
            self._arms = {}

    def _save(self) -> None:
        if not self.filepath:
            return
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            tmp = self.filepath + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"arms": self._arms}, fh, indent=2)
            os.replace(tmp, self.filepath)
        except Exception:
            pass


# ── 4. Candidate generation ──────────────────────────────────────────


class ArchitectureRace:
    """Generates the candidate set that gets raced.

    Candidates are a one-at-a-time (OAT) design around the incumbent
    config: arm 0 is the incumbent, each other arm changes exactly ONE
    architecture axis. That is deliberate -- it keeps attribution clean,
    so the miner can learn "use_rule_basis=True wins on high-interaction
    fingerprints" instead of trying to untangle a random joint sample.
    This is the widened vocabulary bug #25 asks for: the cognitive layer
    stops emitting one scalar (`max_penalty`) and starts choosing among
    named, readable architecture decisions.
    """

    def __init__(
        self,
        *,
        memory: ArmMemory | None = None,
        seed: int = 0,
    ):
        self.memory = memory if memory is not None else ArmMemory()
        self._rng = np.random.default_rng(seed)

    # -- public ------------------------------------------------------

    def propose(
        self,
        *,
        guidance: np.ndarray,
        axes: dict,
        base_action: dict,
        bucket: str,
        max_arms: int = 12,
        n_frontier: int = 3,
    ) -> list[dict]:
        """Build the candidate arms for one stage.

        `axes` is the classifier's `get_architecture_axes()` output:
            {axis_name: {"values": [...], "current": <v>}}
        `base_action` supplies feature_weights / penalty / description
        defaults so a race arm is drop-in compatible with the actions the
        optimizer already knows how to evaluate.
        """
        budget = self.memory.arm_budget(bucket, max_arms)

        base_config = dict(base_action.get("classifier_config") or {})
        base_weights = np.asarray(
            base_action.get("feature_weights", guidance), dtype=np.float64
        )
        base_penalty = float(base_action.get("penalty", 1.0))

        arms: list[dict] = [
            {
                "feature_weights": base_weights.copy(),
                "penalty": base_penalty,
                "classifier_config": dict(base_config),
                "description": "Race: incumbent",
                "_race_arm": "incumbent",
                "_race_axis": None,
            }
        ]

        # (a) architecture arms -- one axis changed each
        for axis, spec in (axes or {}).items():
            current = spec.get("current", base_config.get(axis))
            for value in spec.get("values", []):
                if value == current:
                    continue
                cfg = dict(base_config)
                cfg[axis] = value
                arms.append(
                    {
                        "feature_weights": base_weights.copy(),
                        "penalty": base_penalty,
                        "classifier_config": cfg,
                        "description": f"Race: {axis}={value}",
                        "_race_arm": f"{axis}={value}",
                        "_race_axis": axis,
                    }
                )

        # (b) cutoff arms -- the closed-form frontier, now judged by real
        #     accuracy instead of by the separable surrogate.
        for i, w in enumerate(
            frontier_masks(guidance, n_frontier, include_full=False)
        ):
            if np.array_equal(w, base_weights):
                continue
            arms.append(
                {
                    "feature_weights": w,
                    "penalty": base_penalty,
                    "classifier_config": dict(base_config),
                    "description": f"Race: frontier cut {i + 1}",
                    "_race_arm": f"frontier_{i + 1}",
                    "_race_axis": "frontier",
                }
            )

        # Rank by what we already believe, incumbent always first so it
        # can never be crowded out of its own defence.
        head, tail = arms[0], arms[1:]
        tail.sort(
            key=lambda a: self.memory.posterior_mean(bucket, a["_race_arm"]),
            reverse=True,
        )
        return [head] + tail[: max(0, budget - 1)]

    def record_round(self, bucket: str, arms: list[dict], winner: dict) -> None:
        """Credit the winning arm; every raced arm counts as a trial."""
        labels = [a.get("_race_arm") for a in arms if a.get("_race_arm")]
        win_label = winner.get("_race_arm")
        if not labels or not win_label:
            return
        self.memory.record_round(bucket, labels, win_label)
