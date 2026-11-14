"""Feynman-style equation discovery benchmark (v22 Tier 3).

Tests symbolic recovery + extrapolation on classic physics forms.

Usage:
    python figpso_uil/experiments/exp_feynman_discovery.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from figpso_uil.cognition.symbolic_regressor import SymbolicRegressor
from figpso_uil.cognition.paths import results_path


TASKS = [
    {
        "id": "feynman_x_squared",
        "law": "x0^2",
        "fn": lambda X: X[:, 0] ** 2,
        "train_range": (-1.5, 1.5),
        "test_range": (1.6, 2.2),
    },
    {
        "id": "sin_3x",
        "law": "sin(3*x0)",
        "fn": lambda X: np.sin(3.0 * X[:, 0]),
        "train_range": (-1.0, 1.0),
        "test_range": (1.05, 1.6),
    },
    {
        "id": "linear_2d",
        "law": "2*x0 + 0.5*x1",
        "fn": lambda X: 2 * X[:, 0] + 0.5 * X[:, 1],
        "train_range": (-1.0, 1.0),
        "test_range": (1.1, 1.8),
    },
]


def _make(task, n: int, rng, region):
    lo, hi = region
    if task["id"] == "linear_2d":
        X = rng.uniform(lo, hi, size=(n, 2))
    else:
        X = rng.uniform(lo, hi, size=(n, 1))
    y = task["fn"](X) + rng.normal(0, 0.02, n)
    return X, y


def run_task(task: dict, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    Xtr, ytr = _make(task, 120, rng, task["train_range"])
    Xte, yte = _make(task, 60, rng, task["test_range"])
    sr = SymbolicRegressor(seed=seed, n_terms=10).fit(Xtr, ytr)
    r2_in = float(r2_score(ytr, sr.predict(Xtr)))
    r2_ext = float(r2_score(yte, sr.predict(Xte)))
    ext_ok = bool(np.isfinite(r2_ext) and r2_ext > 0.3)
    return {
        "task": task["id"],
        "law": task["law"],
        "r2_in_sample": round(r2_in, 4),
        "r2_extrapolation": round(r2_ext, 4) if np.isfinite(r2_ext) else None,
        "n_terms": len(sr.terms_),
        "formula": sr.formula_str()[:300],
        "pass": r2_in > 0.85,
        "extrap_ok": ext_ok,
    }


def main():
    results = [run_task(t) for t in TASKS]
    passed = sum(1 for r in results if r["pass"])
    report = {"tasks": len(results), "passed": passed, "results": results}
    out = results_path("ablation_reports/feynman_discovery.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Feynman discovery: {passed}/{len(results)} passed -> {out}")
    for r in results:
        flag = "OK" if r["pass"] else "FAIL"
        print(
            f"  [{flag}] {r['task']}: in={r['r2_in_sample']:.3f} "
            f"extrap={r['r2_extrapolation']:.3f}"
        )


if __name__ == "__main__":
    main()
