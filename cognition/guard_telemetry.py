"""Guard telemetry — record when a safety rail actually binds.

UIL hardcoded-value audit, Part E step 2 (2026-07-24). The audit found
~584 hardcoded constants and sorted them into six categories. Category A
is "safety rails" — hard limits drawn against a specific observed failure
(the penalty actuator clip to [0,10], the graph-size ceiling from bug #18's
49k-edge hang, the config-bound domain guard). The audit's recommendation
for every one of them was NOT "make it dynamic" but "leave it fixed and
LOG how often it binds" — because the bind-rate is the evidence that has
to exist before anyone can responsibly loosen or dynamise the value.

This module is that log. It is pure OBSERVATION: the caller still does its
own clamping exactly as before, then hands this module the original and
the clamped value. Nothing here changes a single decision — verified by
the fact that every call site keeps its original clip and only ADDS an
`observe(...)` call afterward.

What it records per named guard:
  - n_seen:  how many times the guard was evaluated at all
  - n_bound: how many times it actually changed the value (the rail bound)
  - the min / max / mean / std of the ORIGINAL magnitudes seen, via
    Welford's online algorithm (so "is the cap ever approached?" and
    "how far past it do values go?" are both answerable from the log)

The optimizer owns one instance scoped to its `state_dir`, so experiment
and audit runs do NOT pollute the shared default log (bug #30b's
disk-leak class). Flushed once per `fit()`, not per call.
"""

from __future__ import annotations

import json
import math
import os
import threading


class GuardTelemetry:
    """Accumulating, thread-safe, observe-only guard-bind log."""

    def __init__(self, filepath: str | None = None):
        self._filepath = filepath
        self._lock = threading.RLock()
        # name -> {n_seen, n_bound, min, max, mean, _m2, count}
        self._stats: dict[str, dict] = {}
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    self._stats = json.load(f)
            except Exception:
                self._stats = {}

    def observe(self, name: str, original: float, clamped: float) -> None:
        """Record one evaluation of a scalar guard.

        `original` is the value the guard was handed; `clamped` is what it
        returned. The rail "bound" iff they differ. Never raises — a
        telemetry failure must never break the run it is measuring.
        """
        try:
            o = float(original)
            c = float(clamped)
        except (TypeError, ValueError):
            return
        if not math.isfinite(o):
            return
        with self._lock:
            s = self._stats.get(name)
            if s is None:
                s = {
                    "n_seen": 0, "n_bound": 0,
                    "min": o, "max": o, "mean": 0.0, "_m2": 0.0,
                }
                self._stats[name] = s
            s["n_seen"] += 1
            if abs(o - c) > 1e-12:
                s["n_bound"] += 1
            # Welford online mean/variance over the ORIGINAL magnitude.
            delta = o - s["mean"]
            s["mean"] += delta / s["n_seen"]
            s["_m2"] += delta * (o - s["mean"])
            s["min"] = min(s["min"], o)
            s["max"] = max(s["max"], o)

    def record_signal(self, name: str, value: float) -> None:
        """Record the distribution of a formula INPUT (not a guard).

        Category-D of the audit is the hand-tuned weighted-sum formulas
        (the stress blend, the maturity blend, the keep-score). Their
        weights can only become evidence-based once the marginal behaviour
        of each component is known — most immediately, whether a component
        is effectively CONSTANT, which makes its weight dead regardless of
        value (the same class of finding as "penalty is the only journal
        column that ever varies"). This accumulates min/max/mean/std of a
        signal with no bind concept (`n_bound` stays 0, `bind_rate` is
        meaningless for signal keys — read `std`/`min`/`max` instead).
        """
        self.observe(name, value, value)

    def observe_array(self, name: str, original, clamped) -> None:
        """Record one evaluation of an array guard (per-element bind-rate).

        Aggregates the whole array into one `observe` for the bind flag
        (bound iff any element changed) plus the array's own extremes, so
        a per-feature clip like the weight floor is summarised without one
        log entry per element.
        """
        try:
            import numpy as np
            o = np.asarray(original, dtype=float)
            c = np.asarray(clamped, dtype=float)
            if o.size == 0:
                return
            bound = bool(np.any(np.abs(o - c) > 1e-12))
            # Feed the extreme original element so min/max track the real
            # range the guard sees; mean uses the array mean.
            with self._lock:
                s = self._stats.get(name)
                if s is None:
                    s = {
                        "n_seen": 0, "n_bound": 0,
                        "min": float(o.min()), "max": float(o.max()),
                        "mean": 0.0, "_m2": 0.0,
                    }
                    self._stats[name] = s
                s["n_seen"] += 1
                if bound:
                    s["n_bound"] += 1
                m = float(np.mean(o))
                delta = m - s["mean"]
                s["mean"] += delta / s["n_seen"]
                s["_m2"] += delta * (m - s["mean"])
                s["min"] = min(s["min"], float(o.min()))
                s["max"] = max(s["max"], float(o.max()))
        except Exception:
            return

    def snapshot(self) -> dict:
        """A JSON-safe copy, with std derived and the internal M2 dropped."""
        with self._lock:
            out = {}
            for name, s in self._stats.items():
                n = s.get("n_seen", 0)
                var = (s.get("_m2", 0.0) / n) if n > 1 else 0.0
                out[name] = {
                    "n_seen": n,
                    "n_bound": s.get("n_bound", 0),
                    "bind_rate": round(s["n_bound"] / n, 4) if n else 0.0,
                    "min": round(s.get("min", 0.0), 6),
                    "max": round(s.get("max", 0.0), 6),
                    "mean": round(s.get("mean", 0.0), 6),
                    "std": round(math.sqrt(max(0.0, var)), 6),
                }
            return out

    def flush(self) -> None:
        """Persist accumulated stats. No-op when no filepath is set."""
        if not self._filepath:
            return
        with self._lock:
            try:
                tmp = self._filepath + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self._stats, f, indent=2)
                os.replace(tmp, self._filepath)
            except Exception:
                pass
