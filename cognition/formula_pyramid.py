"""Parallel formula engines stabilized by a randomized decision-tree pyramid.

Why this exists
---------------
Running several FormulaEngines at once is the natural way to explore more
mutations per stage, but shared RNG / shared library writes make parallel
evolution nondeterministic (same class as bug #31's unseeded liblinear and
hybrid branch clones).

This module keeps workers **isolated** and merges them with a **pyramid of
shallow DecisionTrees** so:

1. Each worker owns `FormulaEngine(seed=mix(master, worker_id))` — no shared
   RNG stream.
2. Mutation results are collected, then sorted by a stable key before any
   selection — completion order cannot change the outcome.
3. A pyramid of randomized DTs (one per level, seeded) ranks formula outputs
   as features against y and keeps the top half each level — cheap, seeded,
   and order-independent once the candidate matrix is sorted.
4. Library writes happen only on the serial admit path after the pyramid
   finishes — workers never touch FormulaLibrary.

Opt-in only (`cfg.parallel_formula_pyramid`). Default stays the single-engine
path until a cross-seed gate clears it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from figpso_uil.cognition.formula_engine import FormulaEngine, FormulaNode


def _mix_seed(*parts: int) -> int:
    """Deterministic 31-bit seed mix (stable across processes)."""
    h = 2166136261
    for p in parts:
        h ^= int(p) & 0xFFFFFFFF
        h = (h * 16777619) & 0xFFFFFFFF
    return h & 0x7FFFFFFF


def _tree_key(tree: FormulaNode) -> str:
    """Stable identity for sorting — dict round-trip is canonical here."""
    try:
        import json
        return json.dumps(tree.to_dict(), sort_keys=True, separators=(",", ":"))
    except Exception:
        return repr(tree)


@dataclass(frozen=True)
class PyramidParent:
    formula_id: str
    name: str
    tree: FormulaNode


@dataclass
class PyramidCandidate:
    tree: FormulaNode
    mutation_type: str
    worker_id: int
    parent_id: str
    parent_name: str
    score: float = 0.0

    def sort_key(self) -> tuple:
        return (_tree_key(self.tree), self.worker_id, self.parent_id)


def _randomize_bias(
    bias: np.ndarray | None,
    n_features: int,
    rng: np.random.Generator,
    keep_frac: float = 0.65,
) -> np.ndarray | None:
    """Per-worker feature subspace — diversity without shared state."""
    if n_features <= 0:
        return bias
    base = (
        np.asarray(bias, dtype=float).copy()
        if bias is not None and len(bias) == n_features
        else np.ones(n_features, dtype=float)
    )
    keep = max(1, int(round(n_features * keep_frac)))
    mask = np.zeros(n_features, dtype=bool)
    mask[rng.choice(n_features, size=keep, replace=False)] = True
    out = base.copy()
    out[~mask] *= 0.05  # near-zero, not hard-zero (preserves grammar reach)
    return out


def _worker_mutate(
    worker_id: int,
    master_seed: int,
    parent: PyramidParent,
    stress: float,
    n_features: int,
    feature_bias: np.ndarray | None,
) -> PyramidCandidate | None:
    engine = FormulaEngine(seed=_mix_seed(master_seed, worker_id, 0xF0A1))
    rng = np.random.default_rng(_mix_seed(master_seed, worker_id, 0xB1A5))
    bias = _randomize_bias(feature_bias, n_features, rng)
    try:
        mut, mtype = engine.mutate(
            parent.tree, stress, n_features=n_features, feature_bias=bias,
        )
        ok, _ = engine.validate_classifier(mut, n_features=n_features)
        if not ok:
            return None
        return PyramidCandidate(
            tree=mut,
            mutation_type=mtype,
            worker_id=worker_id,
            parent_id=parent.formula_id,
            parent_name=parent.name,
        )
    except Exception:
        return None


def _formula_column(engine: FormulaEngine, tree: FormulaNode, X: np.ndarray) -> np.ndarray:
    raw = engine.evaluate(tree, {"X": X})
    col = np.asarray(raw, dtype=float).reshape(-1)
    if col.shape[0] != X.shape[0]:
        col = np.resize(col, X.shape[0])
    col = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
    return col


def _pyramid_select(
    candidates: list[PyramidCandidate],
    X: np.ndarray,
    y: np.ndarray,
    master_seed: int,
    n_admit: int,
) -> list[PyramidCandidate]:
    """Successive DT levels: formula outputs → features → keep top half.

    Each level fits a shallow randomized tree on a bootstrap of rows
    (seeded), uses feature_importances_ to rank formulas, keeps the
    stronger half. Final level returns up to n_admit survivors, sorted
    stably so ties do not depend on worker completion order.
    """
    if not candidates:
        return []
    # Completion-order independence: sort before any selection.
    pool = sorted(candidates, key=lambda c: c.sort_key())
    if len(pool) <= n_admit:
        for i, c in enumerate(pool):
            c.score = float(len(pool) - i)
        return pool

    engine = FormulaEngine(seed=_mix_seed(master_seed, 0xD7EE))
    y_arr = np.asarray(y).reshape(-1)
    multiclass = len(np.unique(y_arr)) > 2
    level = 0
    while len(pool) > n_admit:
        cols = np.column_stack([
            _formula_column(engine, c.tree, X) for c in pool
        ])
        rng = np.random.default_rng(_mix_seed(master_seed, level, 0xD7))
        n = cols.shape[0]
        # Bootstrap rows — randomization for diversity; seed makes it stable.
        idx = rng.integers(0, n, size=n)
        Xb, yb = cols[idx], y_arr[idx]
        depth = min(4, 1 + level)
        if multiclass or set(np.unique(yb)).issubset({0, 1, 0.0, 1.0}):
            # Binary / small-label: classifier importances.
            labels = np.unique(yb)
            if len(labels) < 2:
                # Degenerate — fall back to absolute correlation with y.
                importances = np.array([
                    abs(float(np.corrcoef(cols[:, j], y_arr)[0, 1]))
                    if np.std(cols[:, j]) > 1e-12 else 0.0
                    for j in range(cols.shape[1])
                ])
            else:
                dt = DecisionTreeClassifier(
                    max_depth=depth,
                    random_state=_mix_seed(master_seed, level, 0xC1A5),
                    min_samples_leaf=max(2, n // 40),
                )
                dt.fit(Xb, yb)
                importances = np.asarray(dt.feature_importances_, dtype=float)
        else:
            dt = DecisionTreeRegressor(
                max_depth=depth,
                random_state=_mix_seed(master_seed, level, 0xC1A5),
                min_samples_leaf=max(2, n // 40),
            )
            dt.fit(Xb, yb)
            importances = np.asarray(dt.feature_importances_, dtype=float)

        ranked = sorted(
            zip(pool, importances),
            key=lambda t: (-float(t[1]), t[0].sort_key()),
        )
        keep_n = max(n_admit, (len(ranked) + 1) // 2)
        if keep_n >= len(ranked):
            # Cannot shrink further without dropping below n_admit logic —
            # take top n_admit and stop.
            pool = [c for c, s in ranked[:n_admit]]
            for i, (c, s) in enumerate(ranked[:n_admit]):
                c.score = float(s)
            break
        pool = [c for c, s in ranked[:keep_n]]
        for c, s in ranked[:keep_n]:
            c.score = float(s)
        level += 1
        if level > 8:  # hard cap — pathological plateaus
            pool = [c for c, s in ranked[:n_admit]]
            break

    return sorted(pool, key=lambda c: (-c.score, c.sort_key()))[:n_admit]


def run_formula_pyramid(
    *,
    parents: list[PyramidParent],
    master_seed: int,
    stress: float,
    n_features: int,
    feature_bias: np.ndarray | None,
    X: np.ndarray,
    y: np.ndarray,
    n_workers: int = 4,
    n_admit: int = 2,
    parallel: bool = True,
) -> list[PyramidCandidate]:
    """Mutate with N isolated engines, select survivors via DT pyramid.

    parallel=False runs workers serially (same seeds → byte-identical
    survivors as parallel=True). Used by determinism tests.
    """
    if not parents or n_workers < 1:
        return []
    n_admit = max(1, int(n_admit))
    n_workers = max(1, int(n_workers))

    jobs = []
    for wid in range(n_workers):
        parent = parents[wid % len(parents)]
        jobs.append((wid, parent))

    raw: list[PyramidCandidate] = []
    if parallel and n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [
                pool.submit(
                    _worker_mutate,
                    wid, master_seed, parent, stress, n_features, feature_bias,
                )
                for wid, parent in jobs
            ]
            for fut in as_completed(futs):
                try:
                    c = fut.result()
                except Exception:
                    c = None
                if c is not None:
                    raw.append(c)
    else:
        for wid, parent in jobs:
            c = _worker_mutate(
                wid, master_seed, parent, stress, n_features, feature_bias,
            )
            if c is not None:
                raw.append(c)

    return _pyramid_select(raw, X, y, master_seed, n_admit)


def parents_from_library(
    library,
    rng: np.random.Generator,
    k: int = 4,
    progress: float = 0.0,
) -> list[PyramidParent]:
    """Build pyramid parents from a FormulaLibrary without mutating it twice.

    Uses select_tournament_candidates (roster) so we get lineage diversity
    without burning times_used_as_parent k times.
    """
    roster = library.select_tournament_candidates(rng, k=k, progress=progress)
    out: list[PyramidParent] = []
    for entry in roster:
        tree = library.get_formula_tree(entry["formula_id"])
        if tree is None:
            continue
        out.append(PyramidParent(
            formula_id=entry["formula_id"],
            name=entry.get("name", entry["formula_id"]),
            tree=tree,
        ))
    return out
