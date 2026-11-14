"""HITL pilot protocol script (v22 M5).

Documents and runs a minimal human-in-the-loop vs UIL-alone comparison
on a symbolic recovery task. Outputs a JSON protocol record for audit.

Usage:
    python -m figpso_uil.experiments.hitl_pilot_protocol
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np

from figpso_uil.cognition.symbolic_regressor import SymbolicRegressor
from figpso_uil.cognition.paths import results_path


def _recovery_task(seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, 100)
    X = x.reshape(-1, 1)
    y = np.sin(2.5 * x) + rng.normal(0, 0.02, len(x))
    return X, y


def run_uil_alone(seed: int = 0) -> dict:
    t0 = time.perf_counter()
    X, y = _recovery_task(seed)
    sr = SymbolicRegressor(seed=seed, n_terms=8).fit(X, y)
    pred = sr.predict(X)
    r2 = float(1 - np.var(y - pred) / np.var(y))
    elapsed = time.perf_counter() - t0
    return {
        "mode": "uil_alone",
        "r2": round(r2, 4),
        "n_terms": len(sr.terms_),
        "elapsed_s": round(elapsed, 3),
        "formula": sr.formula_str()[:200],
    }


def run_hitl_simulated(seed: int = 0) -> dict:
    """Simulated HITL: human rejects a bad first term by re-running with more terms."""
    t0 = time.perf_counter()
    X, y = _recovery_task(seed)
    sr = SymbolicRegressor(seed=seed, n_terms=4).fit(X, y)
    r2_first = float(1 - np.var(y - sr.predict(X)) / np.var(y))
    # Human correction: increase capacity after reviewing poor fit
    if r2_first < 0.9:
        sr = SymbolicRegressor(seed=seed, n_terms=10).fit(X, y)
    pred = sr.predict(X)
    r2 = float(1 - np.var(y - pred) / np.var(y))
    elapsed = time.perf_counter() - t0
    return {
        "mode": "hitl_simulated",
        "r2_first_pass": round(r2_first, 4),
        "r2_final": round(r2, 4),
        "human_correction": r2_first < 0.9,
        "elapsed_s": round(elapsed, 3),
        "formula": sr.formula_str()[:200],
    }


def main():
    alone = run_uil_alone()
    hitl = run_hitl_simulated()
    record = {
        "protocol": "v22_m5_hitl_pilot",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "sin(2.5x) recovery on [-1,1]",
        "uil_alone": alone,
        "hitl_simulated": hitl,
        "hitl_recovery_delta": round(hitl["r2_final"] - alone["r2"], 4),
    }
    out = results_path("ablation_reports/hitl_pilot_protocol.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print(json.dumps(record, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
