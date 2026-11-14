"""Reset all persisted UIL memory to empty defaults."""

from __future__ import annotations
import json
import os

from figpso_uil.cognition.paths import RESULTS_DIR, results_path
from figpso_uil.cognition.live_feed import clear_feed, enable as enable_live_feed

# Empty defaults matching each module's _ensure_file / _load expectations
EMPTY_STATES: dict[str, object] = {
    "uil_journal_v19.json": [],
    "uil_journal_v19_memory.json": {},
    "uil_principles_v19.json": [],
    "uil_causal_v20.json": [],
    "uil_episodes_v20.json": [],
    "uil_formula_library.json": [],
    "uil_meta_principles.json": [],
    "uil_failures_v20.json": [],
    "uil_symbolic_classifier_library.json": [],
    "uil_network_v20.json": {
        "nodes": {},
        "edges": [],
        "node_counter": 0,
        "co_activation_tally": {},
    },
}


def clear_all_memory(clear_live_feed: bool = True) -> dict:
    """Wipe shared UIL memory files. Returns summary of cleared paths."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cleared = []
    for filename, empty in EMPTY_STATES.items():
        path = results_path(filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(empty, f, indent=2)
        cleared.append(path)

    if clear_live_feed:
        clear_feed()

    # Re-enable feed so next runs appear in UI
    enable_live_feed()

    return {
        "ok": True,
        "cleared_files": cleared,
        "results_dir": RESULTS_DIR,
        "message": "All UIL memory reset. Episodic, journal, principles, KG, causal — empty.",
    }


def clear_and_bootstrap(clear_live_feed: bool = True) -> dict:
    """Wipe memory then install clean governance + seed principles."""
    from figpso_uil.cognition.bootstrap_knowledge import apply_bootstrap
    from figpso_uil.cognition.uil_library import UIL_Library
    from figpso_uil.cognition.editable_network import EditableNetwork
    from figpso_uil.cognition.formula_library import FormulaLibrary
    from figpso_uil.cognition.formula_engine import FormulaEngine

    result = clear_all_memory(clear_live_feed=clear_live_feed)
    lib = UIL_Library()
    net = EditableNetwork()
    sym = FormulaLibrary(filepath=results_path("uil_symbolic_classifier_library.json"))
    engine = FormulaEngine(seed=42)
    boot = apply_bootstrap(lib, net, symbolic_library=sym, symbolic_engine=engine, force=True)
    result["bootstrap"] = boot
    result["message"] = (
        "Memory cleared and Path-A bootstrap installed "
        f"({boot['principles_added']} seed principles, governance manifest)."
    )
    return result
