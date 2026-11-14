"""M1 acceptance test — v22 blueprint §1.2 (VersionStore event sourcing).

Acceptance criteria:
  1. VersionStore replay reconstructs network canonical state at
     arbitrary seq BYTE-IDENTICALLY (json.dumps equality).
  2. Decay/belief trajectories are queryable per cell.
  3. Uncapped segment rotation works; store state survives re-open.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from figpso_uil.cognition.editable_network import EditableNetwork
from figpso_uil.cognition.version_store import VersionStore


def canon(state: dict) -> str:
    return json.dumps(state, sort_keys=True)


def test_store_segments_and_reopen():
    print("=== M1.3a: segment rotation + re-open ===")
    d = tempfile.mkdtemp(prefix="uil_m1_store_")
    store = VersionStore(directory=d, segment_size=10, snapshot_every=10_000)
    for i in range(25):
        store.append(cell_id=f"C-{i % 4}", cell_type="node", op="mutate",
                     payload={"post_state": {"confidence": i / 25.0}})
    segs = [f for f in os.listdir(d) if f.startswith("events_")]
    assert len(segs) == 3, f"expected 3 segments for 25 events @10, got {segs}"
    assert store.last_seq == 25

    reopened = VersionStore(directory=d, segment_size=10)
    assert reopened.last_seq == 25, "seq lost on re-open"
    # versions restored: C-0 appeared at i=0,4,8,12,16,20,24 -> 7 versions
    assert reopened._versions["C-0"] == 7, reopened._versions
    print(f"  segments: {sorted(segs)}, last_seq={store.last_seq}  PASS")


def test_network_byte_identical_replay():
    print("=== M1.3b: byte-identical network replay ===")
    d = tempfile.mkdtemp(prefix="uil_m1_net_")
    store = VersionStore(directory=d, segment_size=10, snapshot_every=7)
    net = EditableNetwork(filepath=None, version_store=store)

    checkpoints: list[tuple[int, str]] = []

    def checkpoint():
        checkpoints.append((store.last_seq, canon(net.canonical_state())))

    n1 = net.add_node([{"feature": "state.entropy", "op": ">", "threshold": 0.5}],
                      action_override={"max_penalty": 2.0}, confidence=0.9)
    n2 = net.add_node([{"feature": "state.n_samples", "op": "<", "threshold": 100}],
                      confidence=0.7)
    checkpoint()

    net.add_edge(n1, n2, "leads_to", weight=0.6)
    net.edit_node(n1, confidence=0.6)
    checkpoint()

    # Duplicate-conditions node -> merge path
    net.add_node([{"feature": "state.entropy", "op": ">", "threshold": 0.5}],
                 confidence=0.8, evidence_count=3)
    net.edit_node(n1, confidence=0.3)
    checkpoint()

    n3 = net.add_node([{"feature": "config.max_depth", "op": ">", "threshold": 6}],
                      confidence=0.55)
    net.add_edge(n2, n3, "strengthens", weight=0.4)
    net.add_edge(n1, n3, "leads_to", weight=0.5)
    net.remove_edge(n2, n3, "strengthens")
    checkpoint()

    # Exact-duplicate node then consolidation (merge + redirects)
    dup = net.add_node([{"feature": "config.max_depth", "op": ">", "threshold": 6}],
                       confidence=0.5, node_id="N-DUP")
    if dup != n3:  # only if it wasn't merged on similarity already
        net.add_edge(n1, dup, "leads_to", weight=0.9)
        net.consolidate_duplicate_nodes()
    checkpoint()

    net.remove_node(n2)
    checkpoint()

    snaps = [f for f in os.listdir(d) if f.startswith("snapshot_")]
    assert snaps, "no snapshots were written (snapshot_every=7)"

    for seq, expected in checkpoints:
        got = canon(store.reconstruct(seq)["state"])
        assert got == expected, (
            f"replay at seq {seq} diverged from live canonical state\n"
            f"expected: {expected[:300]}\n     got: {got[:300]}"
        )
    print(f"  {len(checkpoints)} checkpoints byte-identical "
          f"(final seq {store.last_seq}, {len(snaps)} snapshots)  PASS")


def test_belief_trajectory():
    print("=== M1.3c: belief trajectory queryable ===")
    d = tempfile.mkdtemp(prefix="uil_m1_traj_")
    store = VersionStore(directory=d)
    net = EditableNetwork(filepath=None, version_store=store)

    nid = net.add_node([{"feature": "state.entropy", "op": ">", "threshold": 0.9}],
                       confidence=0.9)
    net.edit_node(nid, confidence=0.6)
    net.edit_node(nid, confidence=0.3)

    traj = [t["confidence"] for t in store.trajectory(nid)
            if t["confidence"] is not None]
    assert traj == [0.9, 0.6, 0.3], f"trajectory wrong: {traj}"
    versions = [t["version"] for t in store.trajectory(nid)]
    assert versions == sorted(versions) and len(set(versions)) == len(versions)
    print(f"  confidence trajectory {traj} with versions {versions}  PASS")
    # The decayed belief (0.9->0.3) is now distinguishable from one that
    # was always 0.3 -- the M1 requirement.


def test_formula_causal_version_store():
    print("=== M1.3d: formula + causal cells in VersionStore ===")
    d = tempfile.mkdtemp(prefix="uil_m1_fc_")
    store = VersionStore(directory=d, snapshot_every=10_000)
    from figpso_uil.cognition.formula_library import FormulaLibrary
    from figpso_uil.cognition.formula_engine import FormulaNode
    from figpso_uil.cognition.causal_intervention import CausalInterventionEngine

    flib = FormulaLibrary(
        filepath=os.path.join(d, "formulas.json"),
        version_store=store,
    )
    fid = flib.add_formula("t", FormulaNode(feature=0), "seed", "none")
    assert store.trajectory(fid)[0]["op"] == "create"

    ceng = CausalInterventionEngine(filepath=None, version_store=store)
    h = ceng.form_hypothesis({
        "conditions": [{"feature": "state.n_features", "op": ">", "threshold": 10}],
        "predicted_regret": 0.3,
    })
    assert h is not None
    assert store.trajectory(h.hypothesis_id)[0]["op"] == "create"
    ceng.evaluate_intervention(h.hypothesis_id, 0.5, 0.2, 0.6, 0.7)
    traj = store.trajectory(h.hypothesis_id)
    assert len(traj) >= 2
    print(f"  formula + causal trajectories len={len(traj)}  PASS")


if __name__ == "__main__":
    test_store_segments_and_reopen()
    test_network_byte_identical_replay()
    test_belief_trajectory()
    test_formula_causal_version_store()
    print("\n" + "=" * 50)
    print("M1 ACCEPTANCE: ALL TESTS PASSED")
    print("=" * 50)
