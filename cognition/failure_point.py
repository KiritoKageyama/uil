"""Failure Point Capture — diagnostic recording before fail-loud propagation.

V20+.E: This is a researcher's codebase, not a production service. When a
core mechanism breaks (a custom formula throws, a classifier rejects a
config a principle produced), the old behavior was to swallow the
exception and silently degrade (skip the action, return a zero, leave a
value untransformed). That hides exactly the failures most worth knowing
about.

record_failure_point() does the opposite: it captures *what* broke,
*where*, and the surrounding state, appends it to an unmissable
append-only log, and returns that record so the caller can re-raise with
it attached. It does NOT catch anything itself and does NOT make the run
survive -- the caller still raises. It only makes the failure legible.
"""

from __future__ import annotations
import json
import os
import time
import traceback

from figpso_uil.cognition.paths import results_path


def record_failure_point(
    subsystem: str,
    context: dict,
    exc: Exception,
    filepath: str = results_path("uil_failures_v20.json"),
) -> dict:
    """Capture and persist a structured failure record, then return it.

    Args:
        subsystem: Where the failure occurred, e.g. "optimizer.action_fit".
        context: Whatever state was in play -- stage, dataset, action,
            formula_id, the variables a formula was evaluated against,
            etc. Should be JSON-safe (numpy arrays will be coerced via
            str() by the json.dump default= fallback).
        exc: The exception that was caught.
        filepath: Append-only log location.

    Returns:
        The failure record dict that was written.
    """
    record = {
        "subsystem": subsystem,
        "context": context,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
        "timestamp": time.time(),
    }

    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    data.append(record)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return record
