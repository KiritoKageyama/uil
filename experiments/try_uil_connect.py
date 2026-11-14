"""Accuracy A/B: uil-off vs default (core wiring ON) vs full connect pack.

Default UIL now arms signal_translator + connect_cognition +
utility_gated_formulas. This script measures whether that wiring helps
or hurts held-out accuracy (cross-seed).

Usage (from Downloads/):
    python -m figpso_uil.experiments.try_uil_connect --datasets breast_cancer,wine,ionosphere
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

from figpso_uil.experiments.eval_crossseed import evaluate, DATASETS
from figpso_uil.cognition.paths import results_path


def _arm(name: str, seeds, stages, **kw):
    t0 = time.perf_counter()
    row = evaluate(
        name, seeds, stages, 10, 10, 40,
        search_mode="classic",
        **kw,
    )
    elapsed = time.perf_counter() - t0
    row["wall_s"] = round(elapsed, 2)
    return row


def _write_summary(report: dict, path: str) -> None:
    lines = [
        "# Accuracy A/B — core wiring vs off",
        "",
        f"Generated: {report['generated_at']}",
        f"Stages: {report['stages']}  |  Seeds: {report['seeds']}",
        "",
        "Arms:",
        "- **uil_off** — plain learner + search",
        "- **default** — UIL core wiring ON (translator + cognition→L1 + utility formulas)",
        "- **connect** — core + explore_architecture + adaptive_min_evidence",
        "",
        "| dataset | off acc | default acc | d default-off | connect acc | d connect-off | wall default/off |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    deltas = []
    for name, block in report["datasets"].items():
        off = block["uil_off"]
        dflt = block["default"]
        conn = block["connect"]
        d_def = dflt["acc_mean"] - off["acc_mean"]
        d_con = conn["acc_mean"] - off["acc_mean"]
        deltas.append(d_def)
        wall_r = dflt["wall_s"] / max(off["wall_s"], 1e-9)
        lines.append(
            f"| {name} | {off['acc_mean']:.4f}+/-{off['acc_std']:.3f} | "
            f"{dflt['acc_mean']:.4f}+/-{dflt['acc_std']:.3f} | {d_def:+.4f} | "
            f"{conn['acc_mean']:.4f}+/-{conn['acc_std']:.3f} | {d_con:+.4f} | "
            f"{wall_r:.2f}x |"
        )
    mean_d = float(sum(deltas) / len(deltas)) if deltas else 0.0
    n_win = sum(1 for d in deltas if d > 1e-6)
    n_tie = sum(1 for d in deltas if abs(d) <= 1e-6)
    n_lose = sum(1 for d in deltas if d < -1e-6)
    lines.extend([
        "",
        f"**Mean dAcc (default - off):** {mean_d:+.4f}  "
        f"(win {n_win} / tie {n_tie} / lose {n_lose})",
        "",
        "Primary question: does core wiring help accuracy? "
        "CONNECTED != helpful -- this table is Gate 2 for the default flip.",
        "",
    ])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="breast_cancer,wine,ionosphere")
    ap.add_argument("--seeds", default="42,7,123")
    ap.add_argument("--stages", type=int, default=12)
    args = ap.parse_args()
    names = [d.strip() for d in args.datasets.split(",") if d.strip() in DATASETS]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    arms = [
        ("uil_off", dict(uil_enabled=False, uil_connect_pack=False)),
        ("default", dict(uil_enabled=True, uil_connect_pack=False)),
        ("connect", dict(uil_enabled=True, uil_connect_pack=True)),
    ]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stages": args.stages,
        "seeds": seeds,
        "note": "default = CORE_WIRING_FLAGS ON (signal_translator etc.)",
        "datasets": {},
    }

    print(f"=== accuracy A/B  stages={args.stages} seeds={seeds} ===\n")
    for name in names:
        print(f"\n######## {name} ########", flush=True)
        rows = {}
        for tag, kw in arms:
            print(f"\n--- arm={tag} ---", flush=True)
            force = "sparse_linear" if name == "sonar" else None
            r = _arm(name, seeds, args.stages, force_mode=force, **kw)
            rows[tag] = r
            print(
                f"  [{tag}] acc={r['acc_mean']:.4f}+/-{r['acc_std']:.4f}  "
                f"mf1={r['mf1_mean']:.4f}  wall={r['wall_s']:.1f}s",
                flush=True,
            )
        off, dflt, conn = rows["uil_off"], rows["default"], rows["connect"]
        print(
            f"\n  dAcc default-off   = {dflt['acc_mean'] - off['acc_mean']:+.4f}\n"
            f"  dAcc connect-off   = {conn['acc_mean'] - off['acc_mean']:+.4f}\n"
            f"  dWall default/off  = {dflt['wall_s'] / max(off['wall_s'], 1e-9):.2f}x",
            flush=True,
        )
        report["datasets"][name] = {
            tag: {
                "acc_mean": rows[tag]["acc_mean"],
                "acc_std": rows[tag]["acc_std"],
                "mf1_mean": rows[tag]["mf1_mean"],
                "wall_s": rows[tag]["wall_s"],
                "gap": rows[tag]["gap"],
            }
            for tag in rows
        }
        report["datasets"][name]["delta"] = {
            "acc_default_minus_off": dflt["acc_mean"] - off["acc_mean"],
            "acc_connect_minus_off": conn["acc_mean"] - off["acc_mean"],
            "wall_ratio_default_over_off": round(
                dflt["wall_s"] / max(off["wall_s"], 1e-9), 3
            ),
        }

    out_dir = results_path("try_uil_connect")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    summary = os.path.join(out_dir, "SUMMARY.md")
    _write_summary(report, summary)
    print(f"\nWrote {out}\nWrote {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
