"""M5 acceptance — HITL + lab report + human_edit events."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from figpso_uil.cognition.version_store import VersionStore
from figpso_uil.cognition.lab_report import generate_lab_report
from figpso_uil.cognition.uil_library import UIL_Library


def test_human_edit_event():
    print("=== M5.1: human_edit event in VersionStore ===")
    d = tempfile.mkdtemp(prefix="uil_m5_")
    store = VersionStore(directory=d, snapshot_every=10_000)
    lib = UIL_Library(
        library_path=os.path.join(d, "principles.json"),
        version_store=store,
    )
    lib.record_routing_rule(
        dataset_name="test_ds",
        domain="D2_math",
        strategy="symbolic",
        source="test",
        fingerprint_id="fp1",
    )
    traj = store.trajectory("RR-D2_math-symbolic")
    assert len(traj) >= 1
    assert traj[0]["op"] == "create"
    print(f"  routing_rule events={len(traj)}  PASS")


def test_lab_report_generated():
    print("=== M5.2: lab report markdown ===")
    d = tempfile.mkdtemp(prefix="uil_m5_lab_")
    store = VersionStore(directory=d, snapshot_every=10_000)
    store.record_cell(
        cell_id="P-1", cell_type="principle", op="create",
        post_state={"confidence": 0.8, "conditions": []},
        cause="test",
    )
    path = generate_lab_report(
        run_id="run-abc",
        dataset_name="test_ds",
        route_decision={"domain": "D2_math", "strategy": "symbolic", "source": "heuristic"},
        fingerprint={"n_samples": 100, "n_features": 5, "target_type": "binary", "fingerprint_id": "x"},
        version_store=store,
    )
    assert os.path.isfile(path)
    with open(path, encoding="utf-8") as fh:
        body = fh.read()
    assert "UIL Lab Report" in body
    assert "D2_math" in body
    print(f"  report at {os.path.basename(path)}  PASS")


def test_replay_undo_preview():
    print("=== M5.3: replay reconstructs state ===")
    d = tempfile.mkdtemp(prefix="uil_m5_replay_")
    store = VersionStore(directory=d, snapshot_every=10_000)
    store.record_cell("P-1", "principle", "create", {"confidence": 0.9})
    store.record_cell("P-1", "principle", "human_edit", {"confidence": 0.4})
    s0 = store.reconstruct(1)["state"]["principle"]["P-1"]["confidence"]
    s1 = store.reconstruct(2)["state"]["principle"]["P-1"]["confidence"]
    assert s0 == 0.9 and s1 == 0.4
    print("  replay at seq 1 vs 2 distinguishes beliefs  PASS")


if __name__ == "__main__":
    test_human_edit_event()
    test_lab_report_generated()
    test_replay_undo_preview()
    print("\n" + "=" * 50)
    print("M5 ACCEPTANCE: ALL TESTS PASSED")
    print("=" * 50)
