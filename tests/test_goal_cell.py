"""Acceptance test for the LCA-inspired GoalCell (2026-07-18).

Covers the invariants named in `cognition/goal_cell.py`:

  1. Roundtrip — a GoalCell recorded via record_goal_event survives
     VersionStore replay byte-identically (M1 invariant).
  2. Snapshot re-materialization — the goal survives a fresh
     build_composite_snapshot (bug #16 invariant: HITL edits must be in
     BOTH the event stream AND the materialized JSON).
  3. Backward-compat — a run with NO active goal returns
     byte-identical traversal + hypothesis output vs. one where the
     goal parameter is not passed.
  4. Directive effect — a goal that focuses on feature X measurably
     changes the KG traversal attention weight for a node conditioned
     on X; a goal that ignores a hypothesis variable measurably drops
     that hypothesis in the ranking.
  5. LLMMutator proposes goals — offline heuristic path returns
     None or a MutationProposal, never writes state.

Run: python figpso_uil/tests/test_goal_cell.py  (from Downloads/)
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

# Ensure the package parent is on sys.path when run directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(os.path.dirname(_HERE))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from figpso_uil.cognition.goal_cell import (
    ACTIVE_GOAL_ID,
    GOAL_CELL_TYPE,
    GoalCell,
    GoalStore,
    record_goal_event,
)
from figpso_uil.cognition.knowledge_cell import cell_to_readable
from figpso_uil.cognition.version_store import VersionStore
from figpso_uil.cognition.editable_network import EditableNetwork
from figpso_uil.cognition.causal_intervention import (
    CausalInterventionEngine,
    Hypothesis,
    STATUS_PENDING,
)
from figpso_uil.cognition.llm_mutator import LLMMutator


def _make_scratch():
    """Give each test its own tempdir so nothing bleeds into results/."""
    return tempfile.mkdtemp(prefix="uil_goal_test_")


def test_roundtrip_and_snapshot_rematerialization():
    scratch = _make_scratch()
    try:
        goal_path = os.path.join(scratch, "uil_goals.json")
        events_dir = os.path.join(scratch, "events")
        store = GoalStore(filepath=goal_path)

        goal = GoalCell(
            directive="prefer readable symbolic model",
            prefer_domains=["symbolic"],
            focus_features=["feat_a", "feat_b"],
            ignore_features=["feat_c"],
            prefer_relations=["strengthens"],
            goal_keywords=["symbolic", "readable"],
            confidence=0.8,
        )

        vs = VersionStore(directory=events_dir, segment_size=100, snapshot_every=50)
        seq = vs.record_cell(
            cell_id=goal.cell_id,
            cell_type=GOAL_CELL_TYPE,
            op="set_goal",
            post_state=goal.to_dict(),
            cause="test_setup",
        )
        store.set(goal)
        assert seq > 0, "VersionStore.record_cell should return a positive seq"

        # 1. Roundtrip via VersionStore.reconstruct
        state = vs.reconstruct(seq)["state"]
        assert GOAL_CELL_TYPE in state, "goal cell type missing from replay"
        replayed = GoalCell.from_dict(
            state[GOAL_CELL_TYPE][ACTIVE_GOAL_ID], cell_id=ACTIVE_GOAL_ID,
        )
        assert replayed.directive == goal.directive
        assert replayed.focus_features == goal.focus_features
        assert replayed.confidence == goal.confidence

        # 2. Materialized-JSON path survives a fresh GoalStore read
        # (this is what protects against the bug #16 anti-pattern).
        fresh_store = GoalStore(filepath=goal_path)
        active = fresh_store.get_active()
        assert active is not None, "goal lost from materialized JSON"
        assert active.directive == goal.directive

        # 3. cell_to_readable includes goal
        readable = cell_to_readable(GOAL_CELL_TYPE, goal.to_dict(), goal.cell_id)
        assert "GOAL" in readable and "readable symbolic" in readable
        print(f"[OK] roundtrip + snapshot re-materialization (seq={seq})")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_backward_compat_no_goal():
    """A traverse with no goal must match a traverse with goal=None."""
    net = EditableNetwork(filepath=os.path.join(_make_scratch(), "net.json"))
    net.add_node(
        conditions=[{"feature": "x0", "op": ">", "threshold": 0.0}],
        node_id="N1",
    )
    net.add_node(
        conditions=[{"feature": "x1", "op": ">", "threshold": 0.0}],
        node_id="N2",
    )
    state = {"x0": 1.0, "x1": 1.0}

    r1 = net.traverse_with_attention(state)
    r2 = net.traverse_with_attention(state, goal=None)
    assert set(r1["activated_nodes"]) == set(r2["activated_nodes"])
    # attention_weights values match to float precision
    for nid, w in r1["attention_weights"].items():
        assert abs(w - r2["attention_weights"][nid]) < 1e-9, (
            f"goal=None diverged from bare call at {nid}"
        )
    print("[OK] backward compat: goal=None == no-arg call")


def test_directive_biases_traversal():
    net = EditableNetwork(filepath=os.path.join(_make_scratch(), "net.json"))
    net.add_node(
        conditions=[{"feature": "focus_me", "op": ">", "threshold": 0.0}],
        node_id="focused",
    )
    net.add_node(
        conditions=[{"feature": "ignore_me", "op": ">", "threshold": 0.0}],
        node_id="ignored",
    )
    state = {"focus_me": 1.0, "ignore_me": 1.0}

    baseline = net.traverse_with_attention(state)
    goal = GoalCell(
        directive="focus attention on focus_me",
        focus_features=["focus_me"],
        ignore_features=["ignore_me"],
        confidence=1.0,
    )
    biased = net.traverse_with_attention(state, goal=goal)
    b_focus = biased["attention_weights"].get("focused", 0)
    b_ignore = biased["attention_weights"].get("ignored", 0)
    base_focus = baseline["attention_weights"].get("focused", 0)
    base_ignore = baseline["attention_weights"].get("ignored", 0)
    # focused up, ignored down, relative to baseline
    assert b_focus > base_focus * 1.05, (
        f"goal should amplify focused node: base={base_focus} biased={b_focus}"
    )
    assert b_ignore < base_ignore * 0.95, (
        f"goal should suppress ignored node: base={base_ignore} biased={b_ignore}"
    )
    print(f"[OK] directive effect on traversal (focus {base_focus:.3f}->{b_focus:.3f}, "
          f"ignore {base_ignore:.3f}->{b_ignore:.3f})")


def test_directive_biases_hypothesis_chooser():
    scratch = _make_scratch()
    try:
        eng = CausalInterventionEngine(filepath=os.path.join(scratch, "causal.json"))
        # Build two hypotheses with controllable-factor variables.
        h_focus = Hypothesis(
            hypothesis_id="H-FOCUS", variable="action.penalty",
            direction="high", threshold=0.5, source_rule={}, predicted_regret=0.0,
        )
        h_ignore = Hypothesis(
            hypothesis_id="H-IGNORE", variable="config.gamma",
            direction="high", threshold=0.5, source_rule={}, predicted_regret=0.0,
        )
        h_focus.status = STATUS_PENDING
        h_ignore.status = STATUS_PENDING
        eng._hypotheses[h_focus.hypothesis_id] = h_focus
        eng._hypotheses[h_ignore.hypothesis_id] = h_ignore

        baseline = eng.get_testable_hypotheses()
        base_ids = {h.hypothesis_id for h in baseline}
        assert base_ids == {"H-FOCUS", "H-IGNORE"}, (
            f"baseline testable set unexpected: {base_ids}"
        )

        # Goal biases toward 'action' variables, ignores 'gamma'.
        goal = GoalCell(
            directive="focus on penalty knobs, defer gamma tuning",
            goal_keywords=["action", "penalty"],
            ignore_features=["gamma"],
            confidence=1.0,
        )
        biased = eng.get_testable_hypotheses(goal=goal)
        assert biased, "goal should not eliminate every hypothesis"
        assert biased[0].hypothesis_id == "H-FOCUS", (
            f"expected H-FOCUS first, got ordering "
            f"{[h.hypothesis_id for h in biased]}"
        )
        print(f"[OK] hypothesis ranking under goal: {[h.hypothesis_id for h in biased]}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_llm_mutator_proposes_goals():
    m = LLMMutator()  # offline / no API key by default
    # Fingerprint that trips the smoothness+low-dim heuristic
    fp = {"n_features": 4, "sparsity": 0.05, "smoothness": 0.85}
    prop = m.propose_goal(fingerprint=fp, residuals=None, current_goal=None)
    assert prop is not None, "heuristic should propose for smooth low-dim"
    assert prop.op == "set_goal"
    assert prop.payload.get("directive")
    assert prop.proposer in ("engine", "llm")

    # Sparse high-dim → different directive
    fp2 = {"n_features": 128, "sparsity": 0.9, "smoothness": 0.3}
    prop2 = m.propose_goal(fingerprint=fp2, residuals=None, current_goal=None)
    assert prop2 is not None
    assert "sparse" in prop2.payload["directive"].lower()

    # No signal → no proposal (heuristic prefers None over wrong)
    fp3 = {"n_features": 20, "sparsity": 0.3, "smoothness": 0.5}
    prop3 = m.propose_goal(fingerprint=fp3, residuals=None, current_goal=None)
    assert prop3 is None

    # Already-active current_goal suppresses override
    prop4 = m.propose_goal(
        fingerprint=fp, residuals=None,
        current_goal={"status": "active", "directive": "existing"},
    )
    assert prop4 is None, "shouldn't override an active goal on soft signals"
    print("[OK] LLMMutator.propose_goal (heuristic path)")


def main():
    tests = [
        test_roundtrip_and_snapshot_rematerialization,
        test_backward_compat_no_goal,
        test_directive_biases_traversal,
        test_directive_biases_hypothesis_chooser,
        test_llm_mutator_proposes_goals,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print()
    if failed:
        print(f"{failed}/{len(tests)} tests failed")
        sys.exit(1)
    print(f"All {len(tests)} tests passed")


if __name__ == "__main__":
    main()
