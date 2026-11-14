"""UIL v22 milestone audit — runs M0–M5 acceptance checks and blueprint criteria.

Usage (from parent of figpso_uil):
    python figpso_uil/experiments/v22_milestone_audit.py
    python figpso_uil/experiments/v22_milestone_audit.py --full
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


TESTS = [
    ("M0", "tests/test_m0_epistemology.py"),
    ("M1", "tests/test_m1_version_store.py"),
    ("M2", "tests/test_m2_perception.py"),
    ("M3", "tests/test_m3_routing.py"),
    ("M4", "tests/test_m4_scientist.py"),
    ("M5", "tests/test_m5_hitl.py"),
]


def _run_test(rel_path: str) -> tuple[bool, str]:
    path = os.path.join(ROOT, rel_path)
    proc = subprocess.run(
        [sys.executable, path],
        cwd=PARENT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out[-2000:]


def _module_exists(dotted: str) -> bool:
    return importlib.util.find_spec(dotted) is not None


def audit_blueprint_files() -> dict:
    required = [
        "cognition/version_store.py",
        "cognition/fingerprint.py",
        "core/normalization.py",
        "cognition/knowledge_cell.py",
        "cognition/domain_router.py",
        "cognition/llm_mutator.py",
        "cognition/lab_report.py",
        "cognition/router_tuning.py",
        "core/strategy_classifiers.py",
        "webui/frontend/js/hitl-review.js",
        "experiments/exp_routing_parity.py",
        "experiments/hitl_pilot_protocol.py",
    ]
    missing = [p for p in required if not os.path.isfile(os.path.join(ROOT, p))]
    return {"required_files": len(required), "missing": missing, "ok": not missing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="Also run routing parity sweep")
    args = ap.parse_args()

    t0 = time.perf_counter()
    report = {
        "audit": "uil_v22_milestones",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "milestones": {},
        "blueprint_files": audit_blueprint_files(),
        "modules": {
            "figpso_uil.cognition.version_store": _module_exists("figpso_uil.cognition.version_store"),
            "figpso_uil.cognition.fingerprint": _module_exists("figpso_uil.cognition.fingerprint"),
            "figpso_uil.cognition.lab_report": _module_exists("figpso_uil.cognition.lab_report"),
        },
    }

    all_ok = True
    for label, rel in TESTS:
        ok, tail = _run_test(rel)
        report["milestones"][label] = {"pass": ok, "tail": tail}
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label} {rel}")

    if args.full:
        proc = subprocess.run(
            [sys.executable, "-m", "figpso_uil.experiments.exp_routing_parity", "--quick"],
            cwd=PARENT,
            capture_output=True,
            text=True,
            timeout=900,
        )
        report["routing_parity"] = {
            "pass": proc.returncode == 0,
            "tail": (proc.stdout or "")[-1500:],
        }
        all_ok = all_ok and proc.returncode == 0
        print(f"[{'PASS' if proc.returncode == 0 else 'FAIL'}] routing parity sweep")

    report["all_milestones_pass"] = all_ok and report["blueprint_files"]["ok"]
    report["elapsed_s"] = round(time.perf_counter() - t0, 2)

    out_dir = os.path.join(ROOT, "results", "ablation_reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "v22_milestone_audit.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\n" + "=" * 60)
    print(f"v22 AUDIT: {'ALL PASS' if report['all_milestones_pass'] else 'GAPS REMAIN'}")
    print(f"Report: {out_path} ({report['elapsed_s']}s)")
    print("=" * 60)
    if not report["all_milestones_pass"]:
        if report["blueprint_files"]["missing"]:
            print("Missing files:", report["blueprint_files"]["missing"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
