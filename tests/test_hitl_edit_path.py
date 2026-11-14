"""HITL edit path: event + materialized file both update; replay sees it.

Bug #16 note: human edits must go through BOTH ``record_cell`` AND
``apply_materialized_edit`` — snapshots re-materialize from library files,
so an event-only edit is overwritten. Mirrors webui ``edit_cell``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from figpso_uil.cognition.version_store import VersionStore
from figpso_uil.cognition.uil_library import UIL_Library
from figpso_uil.cognition.cell_store import apply_materialized_edit, build_composite_snapshot
from figpso_uil.cognition.editable_network import EditableNetwork
from figpso_uil.core.classifier_registry import CLASSIFIERS
from figpso_uil.core.optimizer import UniversalOptimizer
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch


def test_h1_edit_survives_replay_and_file():
    print("=== H1: human_edit on principle survives event + file ===")
    d = tempfile.mkdtemp(prefix="hitl_h1_")
    store = VersionStore(directory=os.path.join(d, "events"), snapshot_every=10_000)
    lib_path = os.path.join(d, "uil_principles_v19.json")
    lib = UIL_Library(library_path=lib_path, version_store=store)

    rule = {
        "conditions": [
            {"feature": "action.penalty", "op": ">", "threshold": 2.0},
        ],
        "predicted_regret": 0.5,
        "confidence": 0.9,
        "evidence_count": 20,
        "mining_level": "within_dataset",
        "dataset": "toy",
        "logging_policy": {"n_journal_entries": 20},
    }
    lib.update_principles([rule], min_evidence=3)
    principles = lib.get_principles()
    mined = [p for p in principles if p.get("dataset") == "toy"]
    assert mined, "expected minted principle"
    pid = mined[0]["principle_id"]

    # Human edits confidence + override (webui edit_cell shape).
    fields = {
        "confidence": 0.12,
        "action_override": {"max_penalty": 0.5},
        "status": "human_edited",
    }
    target = dict(mined[0])
    target.update(fields)
    seq = store.record_cell(
        cell_id=pid, cell_type="principle", op="human_edit",
        post_state=target, cause="hitl_test",
    )
    # Point apply_materialized_edit at our isolated library via env/path —
    # it uses default results paths, so write the file directly then call.
    apply_materialized_edit("principle", pid, fields, library_path=lib_path)

    on_disk = UIL_Library(library_path=lib_path)._load()
    row = next(p for p in on_disk if p["principle_id"] == pid)
    assert abs(float(row["confidence"]) - 0.12) < 1e-9, row
    assert row.get("action_override", {}).get("max_penalty") == 0.5

    recon = store.reconstruct(seq)["state"]["principle"][pid]
    assert abs(float(recon["confidence"]) - 0.12) < 1e-9
    print(f"  edited {pid} at seq={seq}; file+replay agree  PASS")


def test_h2_reject_marks_status():
    print("=== H2: human_reject marks status ===")
    d = tempfile.mkdtemp(prefix="hitl_h2_")
    store = VersionStore(directory=os.path.join(d, "events"), snapshot_every=10_000)
    lib_path = os.path.join(d, "principles.json")
    # Seed one principle on disk
    seed = [{
        "principle_id": "P-HITL",
        "tier": 2,
        "conditions": [{"feature": "action.penalty", "op": ">", "threshold": 1.0}],
        "action_override": {"max_penalty": 1.0},
        "confidence": 0.8,
        "evidence_count": 12,
        "status": "active",
    }]
    with open(lib_path, "w", encoding="utf-8") as fh:
        json.dump(seed, fh)
    store.record_cell(
        cell_id="P-HITL", cell_type="principle", op="human_reject",
        post_state=None, cause="hitl_test",
    )
    apply_materialized_edit(
        "principle", "P-HITL", {"status": "rejected"}, library_path=lib_path,
    )
    on_disk = json.load(open(lib_path, encoding="utf-8"))
    assert on_disk[0]["status"] == "rejected"
    print("  reject status on disk  PASS")


def test_h3_delete_principles_ab_predictions():
    """Paired A/B: full principles vs stripped library — do predictions move?

    CLAUDE.md recorded 9/9 delta +0.0000 for deleting principles. Re-check
    under current defaults on a short deterministic run.
    """
    print("=== H3: principle-library wipe A/B (existence-style) ===")
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = load_breast_cancer(return_X_y=True)
    X = np.asarray(X, float)
    y = np.asarray(y, int)

    class _Cfg:
        stages = 4
        n_particles = 6
        pso_iters = 5
        particles = 6
        pso_iters_per_stage = 5
        boosting_rounds = 20
        cv_folds = 1
        use_resource_budget = False
        disable_routing = True
        enable_stagnation_evolution = False
        target_accuracy = None
        max_stages = 4
        situation_adaptive = False
        maturity_soft_normalization = False
        stress_adaptive_blend = False
        handle_missing_data = True

    def _run(ablation: dict, tag: str):
        sd = tempfile.mkdtemp(prefix=f"hitl_ab_{tag}_")
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )
        clf = CLASSIFIERS["SymbolicFormula"](42, 2, 20, sd)
        clf.apply_config({"learning_mode": "sparse_linear"})
        opt = UniversalOptimizer(
            cfg=_Cfg(),
            seed=42,
            dataset_name="hitl_ab",
            fold=0,
            start_fold=0,
            classifier=clf,
            search=PrincipleGuidedSearch(),
            classifier_factory=CLASSIFIERS["SymbolicFormula"],
            classifier_build_args=(42, 2, 20, sd),
            state_dir=sd,
            ablation=ablation,
        )
        opt.fit(Xtr, ytr)
        pred = opt.predict(Xte)
        return {
            "acc": float(accuracy_score(yte, pred)),
            "pred": tuple(np.asarray(pred).tolist()),
            "n_principles": len(opt.library.get_principles()),
        }

    full = _run(
        {k: True for k in UniversalOptimizer.DEFAULT_ABLATION}, "full",
    )
    no_prin = _run(
        {**{k: True for k in UniversalOptimizer.DEFAULT_ABLATION},
         "principles": False},
        "noprin",
    )
    same_pred = full["pred"] == no_prin["pred"]
    delta = full["acc"] - no_prin["acc"]
    print(f"  full acc={full['acc']:.4f} n_prin={full['n_principles']}")
    print(f"  no_prin acc={no_prin['acc']:.4f} n_prin={no_prin['n_principles']}")
    print(f"  predictions identical={same_pred}  Δacc={delta:+.4f}")
    if same_pred:
        print("  H3 PASS — principles INERT on this short run (matches prior HITL A/B)")
    else:
        print("  H3 NOTE — principles CONNECTED (predictions moved)")
    return same_pred, delta


def test_h4_pilot_protocol():
    print("=== H4: hitl_pilot_protocol symbolic recovery ===")
    from figpso_uil.experiments.hitl_pilot_protocol import (
        run_uil_alone, run_hitl_simulated,
    )
    alone = run_uil_alone(seed=0)
    hitl = run_hitl_simulated(seed=0)
    delta = hitl["r2_final"] - alone["r2"]
    print(f"  alone R²={alone['r2']}  hitl R²={hitl['r2_final']}  "
          f"correction={hitl['human_correction']}  Δ={delta:+.4f}")
    assert alone["r2"] > 0.5
    assert hitl["r2_final"] > 0.5
    # Correction only fires when first pass is weak (<0.9); when it does,
    # capacity bump must not hurt the first pass.
    if hitl["human_correction"]:
        assert hitl["r2_final"] >= hitl["r2_first_pass"] - 1e-6
    print("  H4 PASS")


if __name__ == "__main__":
    test_h1_edit_survives_replay_and_file()
    test_h2_reject_marks_status()
    test_h3_delete_principles_ab_predictions()
    test_h4_pilot_protocol()
    print("\nHITL edit-path checks done")
