"""One-shot: start a short web-style SymbolicFormula run and poll until done."""
from __future__ import annotations

import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000/api"


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=30) as r:
        return json.loads(r.read())


def main() -> None:
    for _ in range(30):
        try:
            get("/runs")
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError("server not up")

    rec = post("/runs", {
        "dataset_name": "real_breast_cancer",
        "classifier": "SymbolicFormula",
        "symbolic_learning_mode": "multi",
        "seed": 42,
        "stages": 5,
        "n_particles": 8,
        "pso_iters": 8,
        "boosting_rounds": 6,
    })
    run_id = rec["run_id"]
    print(f"started {run_id}")

    for i in range(120):
        time.sleep(3)
        detail = get(f"/runs/{run_id}")
        st = detail.get("status")
        snap = detail.get("last_snapshot") or {}
        stage = snap.get("stage")
        acc = snap.get("accuracy")
        best = snap.get("best_accuracy")
        cpu = snap.get("stage_cpu_seconds")
        blend = snap.get("hybrid_blend")
        print(f"  [{i}] status={st} stage={stage} acc={acc} best={best} cpu={cpu}s blend={blend}")
        if st in ("completed", "stopped", "error"):
            print("done", json.dumps(detail, indent=2)[:800])
            return
    print("timeout waiting for run")


if __name__ == "__main__":
    main()
