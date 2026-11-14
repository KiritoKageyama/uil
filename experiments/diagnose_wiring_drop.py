"""Diagnose why core wiring dropped accuracy (Gate 2 2026-07-25).

Isolates each flag + each translator mechanism on the worst cells:
  wine seed 123 (0.944 -> 0.861) and breast_cancer seed 42 (0.956 -> 0.939).

Usage (from Downloads/):
    python -m figpso_uil.experiments.diagnose_wiring_drop
"""

from __future__ import annotations

import json
import os

import numpy as np

from figpso_uil.experiments.eval_crossseed import one_split, DATASETS
from figpso_uil.cognition.paths import results_path
from figpso_uil.cognition import signal_translator as st


_ORIG_TRANSLATE = st.translate


def _patched(flags: dict):
    """Wrap translate() to surgically disable one mechanism."""
    def _wrapped(**kwargs):
        # Optionally neuter principle reshape BEFORE translate builds it
        if flags.get("no_principles"):
            abl = dict(kwargs.get("ablation") or {})
            abl["principles"] = False
            kwargs = {**kwargs, "ablation": abl, "principles": [],
                      "triggered_principle_ids": [], "overrides": {}}
        if flags.get("no_formulas"):
            abl = dict(kwargs.get("ablation") or {})
            abl["formulas"] = False
            kwargs = {**kwargs, "ablation": abl, "formula_library": None}
        if flags.get("no_network"):
            abl = dict(kwargs.get("ablation") or {})
            abl["network"] = False
            kwargs = {**kwargs, "ablation": abl,
                      "network_result": {"overrides": {}, "activated_nodes": []}}
        if flags.get("no_episodic"):
            abl = dict(kwargs.get("ablation") or {})
            abl["episodic"] = False
            kwargs = {**kwargs, "ablation": abl, "episodic_hint": None}

        d = _ORIG_TRANSLATE(**kwargs)

        if flags.get("no_l1_bump"):
            d.config.pop("sparse_l1_c", None)
        if flags.get("no_prefer"):
            d.prefer_formula_ids = []
        if flags.get("no_hard_topk") and d.feature_guidance is not None:
            # Recompute soft concentration only (no hard mask)
            n = kwargs["n_features"]
            g = st._safe_corr_guidance(
                kwargs.get("X_train"), kwargs.get("y_train"), n,
            )
            if not flags.get("no_concentrate"):
                g = st._concentrate(g, 0.45)
            d.feature_guidance = g
        if flags.get("no_concentrate") and flags.get("no_hard_topk"):
            n = kwargs["n_features"]
            d.feature_guidance = st._safe_corr_guidance(
                kwargs.get("X_train"), kwargs.get("y_train"), n,
            )
        if flags.get("corr_only"):
            n = kwargs["n_features"]
            d.feature_guidance = st._safe_corr_guidance(
                kwargs.get("X_train"), kwargs.get("y_train"), n,
            )
            d.config = {}
            d.prefer_formula_ids = []
            d.sources = ["dataset_correlation"]
        return d
    return _wrapped


def _cell(dataset: str, seed: int, stages: int, **kw) -> dict:
    loader, mode = DATASETS[dataset]
    X, y = loader()
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    n_classes = int(len(np.unique(y)))
    r = one_split(
        X, y, mode, seed, stages, 10, 10, 40, n_classes,
        search_mode="classic",
        **kw,
    )
    return {
        "acc": float(r["acc"]),
        "macro_f1": float(r["macro_f1"]),
        "gap": float(r["internal_val"] - r["acc"]),
        "translator": (r.get("observability") or {}).get("signal_translator"),
        "cognition_connected": (r.get("observability") or {}).get(
            "cognition_connected"
        ),
    }


def main() -> int:
    stages = 12
    cells = [("wine", 123), ("breast_cancer", 42)]

    flag_arms = [
        ("off", dict(uil_enabled=False)),
        ("translator_only", dict(
            uil_enabled=True, signal_translator=True,
            connect_cognition=False, utility_gated_formulas=False,
        )),
        ("connect_only", dict(
            uil_enabled=True, signal_translator=False,
            connect_cognition=True, utility_gated_formulas=False,
        )),
        ("utility_only", dict(
            uil_enabled=True, signal_translator=False,
            connect_cognition=False, utility_gated_formulas=True,
        )),
        ("translator+connect", dict(
            uil_enabled=True, signal_translator=True,
            connect_cognition=True, utility_gated_formulas=False,
        )),
        ("full_core", dict(
            uil_enabled=True, signal_translator=True,
            connect_cognition=True, utility_gated_formulas=True,
        )),
    ]

    mech_arms = [
        ("full", {}),
        ("no_l1_bump", {"no_l1_bump": True}),
        ("no_hard_topk", {"no_hard_topk": True}),
        ("no_concentrate_hard", {"no_hard_topk": True, "no_concentrate": True}),
        ("no_prefer", {"no_prefer": True}),
        ("corr_only", {"corr_only": True}),
        ("no_principles", {"no_principles": True}),
        ("no_formulas", {"no_formulas": True}),
    ]

    report = {"stages": stages, "phase1_flags": {}, "phase2_mechanisms": {}}

    print("=== Phase 1: which cfg flag? ===\n", flush=True)
    for ds, seed in cells:
        key = f"{ds}/seed{seed}"
        report["phase1_flags"][key] = {}
        print(f"\n## {key}", flush=True)
        off_acc = None
        for tag, kw in flag_arms:
            r = _cell(ds, seed, stages, **kw)
            report["phase1_flags"][key][tag] = r
            if tag == "off":
                off_acc = r["acc"]
            d = r["acc"] - off_acc if off_acc is not None else 0.0
            print(
                f"  {tag:20s}  acc={r['acc']:.4f}  d={d:+.4f}  "
                f"gap={r['gap']:+.4f}",
                flush=True,
            )

    print("\n=== Phase 2: which translator mechanism? (full_core base) ===\n",
          flush=True)
    for ds, seed in cells:
        key = f"{ds}/seed{seed}"
        report["phase2_mechanisms"][key] = {}
        print(f"\n## {key}", flush=True)
        off = _cell(ds, seed, stages, uil_enabled=False)
        report["phase2_mechanisms"][key]["off"] = off
        print(f"  {'off':20s}  acc={off['acc']:.4f}", flush=True)
        for tag, flags in mech_arms:
            st.translate = _patched(flags)
            try:
                r = _cell(
                    ds, seed, stages,
                    uil_enabled=True,
                    signal_translator=True,
                    connect_cognition=True,
                    utility_gated_formulas=True,
                )
            finally:
                st.translate = _ORIG_TRANSLATE
            report["phase2_mechanisms"][key][tag] = r
            d = r["acc"] - off["acc"]
            print(
                f"  {tag:20s}  acc={r['acc']:.4f}  d={d:+.4f}  "
                f"gap={r['gap']:+.4f}",
                flush=True,
            )

    out_dir = results_path("diagnose_wiring_drop")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    lines = [
        "# Wiring drop diagnosis",
        "",
        "Phase 1 = which cfg flag. Phase 2 = which translator mechanism.",
        "",
    ]
    for phase in ("phase1_flags", "phase2_mechanisms"):
        lines.append(f"## {phase}")
        lines.append("")
        for key, arms in report[phase].items():
            lines.append(f"### {key}")
            off_acc = arms.get("off", {}).get("acc")
            for tag, r in arms.items():
                d = ""
                if off_acc is not None and tag != "off":
                    d = f"  (d={r['acc'] - off_acc:+.4f}, gap={r['gap']:+.4f})"
                lines.append(f"- `{tag}`: acc={r['acc']:.4f}{d}")
            lines.append("")
    summary = os.path.join(out_dir, "SUMMARY.md")
    with open(summary, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nWrote {out}\nWrote {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
