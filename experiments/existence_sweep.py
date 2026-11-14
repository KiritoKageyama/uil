"""Graded existence audit across datasets AND stage budgets.

The project's central negative result -- "the cognitive subsystems do not
reach the model" -- currently rests on TWO datasets at ONE short budget each
(breast_cancer 5 stages, wine_binary 6 stages; see
results/component_hitl/SUMMARY.md). Two observations is a thin base for the
most important claim in the codebase, and the existing audit is known to be
budget-sensitive: at those budgets even `causal` reads INERT, contradicting
an earlier audit that found it CONNECTED.

Budget matters mechanically, not just statistically. Several subsystems
cannot act early:

  * principles need enough journal rows to mine (MIN_EVIDENCE; ~200 stages
    at the fixed bar, fewer with adaptive_min_evidence);
  * meta_principles need >= 5 principles to cluster;
  * the graph prune/decay cadence first fires around stage 5;
  * evolution fires on a stress peak that a short run may never reach.

So "INERT at 5 stages" and "INERT at 40 stages" are different claims, and
only the second one is interesting. This sweep measures both, and reports
the GRADED metric (share of probe predictions moved) rather than a binary
verdict, so "moved one borderline row" is distinguishable from "rewrote the
model".

Run:
    python -m figpso_uil.experiments.existence_sweep
    python -m figpso_uil.experiments.existence_sweep \
        --datasets breast_cancer,wine --stages 5,20 --out results/existence_sweep
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cell(dataset: str, stages: int, seed: int, out_dir: str,
             only: str | None) -> dict:
    tag = f"{dataset}_s{stages}"
    jpath = os.path.join(out_dir, f"{tag}.json")
    lpath = os.path.join(out_dir, f"{tag}.log")
    cmd = [sys.executable, "-m", "figpso_uil.experiments.existence_audit",
           "--dataset", dataset, "--stages", str(stages), "--seed", str(seed),
           "--json", jpath]
    if only:
        cmd += ["--only", only]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = os.path.dirname(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                          cwd=os.path.dirname(_ROOT))
    elapsed = time.perf_counter() - t0
    with open(lpath, "w", encoding="utf-8") as fh:
        fh.write(proc.stdout + "\n---- stderr ----\n" + proc.stderr)

    if proc.returncode == 2:
        return {"dataset": dataset, "stages": stages, "status": "CONTROL FAILED",
                "elapsed_s": elapsed}
    if proc.returncode != 0 or not os.path.exists(jpath):
        return {"dataset": dataset, "stages": stages, "status": "ERROR",
                "elapsed_s": elapsed,
                "detail": (proc.stderr or proc.stdout).strip()[-300:]}
    with open(jpath, encoding="utf-8") as fh:
        data = json.load(fh)
    data.update({"status": "OK", "elapsed_s": elapsed})
    return data


def render(cells: list[dict]) -> str:
    subs: list[str] = []
    for c in cells:
        for r in c.get("rows", []):
            if r["subsystem"] not in subs:
                subs.append(r["subsystem"])

    out = ["# Graded existence audit sweep", "",
           "Cell value = share of probe predictions that MOVED when the",
           "subsystem was removed. `.` means byte-identical (INERT).",
           "A CONTROL FAILED cell means two identical runs disagreed, so",
           "nothing in that column can be read as evidence.", ""]

    ok = [c for c in cells if c["status"] == "OK"]
    bad = [c for c in cells if c["status"] != "OK"]

    if ok:
        head = "| subsystem | " + " | ".join(
            f"{c['dataset']}@{c['stages']}" for c in ok) + " |"
        out += [head, "|---" * (len(ok) + 1) + "|"]
        for s in subs:
            cellstrs = []
            for c in ok:
                row = next((r for r in c["rows"] if r["subsystem"] == s), None)
                if row is None:
                    cellstrs.append("—")
                elif row["verdict"] == "INERT":
                    cellstrs.append(".")
                else:
                    f = row["pred_frac_changed"]
                    cellstrs.append(
                        f"**{100.0 * f:.1f}%**" if f else
                        f"state only ({', '.join(row['changed'])[:24]})")
            out.append(f"| {s} | " + " | ".join(cellstrs) + " |")
        out.append("")

        out.append("## Per-cell summary")
        out.append("")
        out.append("| cell | connected / total | max preds moved | wall |")
        out.append("|---|---:|---:|---:|")
        for c in ok:
            rows = c["rows"]
            conn = sum(1 for r in rows if r["verdict"] == "CONNECTED")
            mx = max((r["pred_frac_changed"] or 0.0) for r in rows) if rows else 0.0
            out.append(f"| {c['dataset']}@{c['stages']} | {conn}/{len(rows)} | "
                       f"{100.0 * mx:.1f}% | {c['elapsed_s']:.0f}s |")
        out.append("")

    if bad:
        out.append("## Cells that produced no evidence")
        out.append("")
        for c in bad:
            out.append(f"- **{c['dataset']}@{c['stages']}**: {c['status']}"
                       + (f" — {c.get('detail', '')}" if c.get("detail") else ""))
        out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="breast_cancer,wine_binary")
    ap.add_argument("--stages", default="5,20")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=os.path.join(_ROOT, "results",
                                                  "existence_sweep"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    budgets = [int(s) for s in args.stages.split(",") if s.strip()]

    cells = []
    for ds in datasets:
        for st in budgets:
            print(f"--- {ds} @ {st} stages ---", flush=True)
            c = run_cell(ds, st, args.seed, args.out, args.only)
            print(f"    {c['status']} in {c['elapsed_s']:.0f}s", flush=True)
            cells.append(c)

    text = render(cells)
    with open(os.path.join(args.out, "SUMMARY.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.out, "cells.json"), "w", encoding="utf-8") as fh:
        json.dump(cells, fh, indent=2)
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
