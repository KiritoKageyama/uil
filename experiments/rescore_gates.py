"""Re-score every stored Gate 2 verdict with paired statistics and power.

WHY
---
Every "PASSED / FAILED" verdict in CLAUDE.md was decided by comparing two
arm MEANS over 3 seeds x a handful of datasets, with no interval and no power
check. The per-cell standard deviation in those same artifacts runs 0.013 to
0.047 accuracy. A 3-seed mean cannot resolve an effect of 0.002 against that
noise -- so several recorded verdicts ("mean -0.0013, FAILED", "mean -0.0022,
FAILED") are not measurements of absence, they are absence of measurement.

This script re-reads the artifacts already on disk -- no new compute -- and
for each experiment reports:

  * the paired per-cell differences (dataset x seed), which is the correct
    unit: both arms saw the same split, so pairing removes dataset and seed
    variance that the unpaired mean leaves in;
  * a 95% CI on the paired mean, and a Wilcoxon signed-rank p-value;
  * the MINIMUM DETECTABLE EFFECT at 80% power for the n actually run, and
    the n that WOULD have been needed to detect delta = 0.01;
  * a verdict of HARM / BENEFIT / UNDERPOWERED that distinguishes "we showed
    it does nothing" from "we could not have seen anything".

Run:  python -m figpso_uil.experiments.rescore_gates
      python -m figpso_uil.experiments.rescore_gates --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")

#: Target effect the project actually cares about (one accuracy point).
TARGET_DELTA = 0.01

_DATASET_RE = re.compile(r"===\s+([A-Za-z0-9_]+)\s+\((\d+)\s+seeds?\)\s+===")
_SEED_RE = re.compile(r"seed\s+(\d+):\s*acc=([0-9.]+)(?:\s+macroF1=([0-9.]+))?")


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------

def read_crossseed_log(path: str) -> dict[tuple[str, int], float]:
    """Parse per-(dataset, seed) accuracy out of an eval_crossseed transcript.

    The logs are UTF-16 PowerShell transcripts interleaved with warnings, so
    parse line-wise and ignore everything that is not a header or a seed row.
    """
    out: dict[tuple[str, int], float] = {}
    text = None
    for enc in ("utf-16", "utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(path, encoding=enc) as fh:
                text = fh.read()
            if text.count("\x00") > len(text) // 4:
                continue
            if "seed" in text:
                break
        except (UnicodeError, OSError):
            continue
    if not text:
        return out

    current = None
    for line in text.splitlines():
        m = _DATASET_RE.search(line)
        if m:
            current = m.group(1)
            continue
        m = _SEED_RE.search(line)
        if m and current:
            out[(current, int(m.group(1)))] = float(m.group(2))
    return out


def read_direct_install(path: str) -> tuple[dict, dict]:
    with open(path, encoding="utf-8") as fh:
        rep = json.load(fh)
    off, on = {}, {}
    for row in rep.get("rows", []):
        key = (row["dataset"], int(row["seed"]))
        (on if row.get("install") else off)[key] = float(row["heldout_acc"])
    return off, on


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def _t_ppf(p: float, df: int) -> float:
    """Student-t quantile without scipy (Cornish-Fisher on the normal)."""
    z = _z_ppf(p)
    if df <= 0:
        return z
    g1 = (z ** 3 + z) / 4.0
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96.0
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384.0
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3


def _z_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _wilcoxon_p(diffs: list[float]) -> float | None:
    """Exact two-sided Wilcoxon signed-rank p for small n (zeros dropped)."""
    nz = [d for d in diffs if abs(d) > 1e-12]
    n = len(nz)
    if n == 0 or n > 20:
        return None
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(nz[order[j + 1]]) - abs(nz[order[i]])) < 1e-12:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    # Exact null: enumerate all 2^n sign assignments of the observed ranks.
    counts: dict[float, int] = {0.0: 1}
    for r in ranks:
        nxt: dict[float, int] = defaultdict(int)
        for s, c in counts.items():
            nxt[s] += c
            nxt[s + r] += c
        counts = dict(nxt)
    total = 2 ** n
    mean_w = sum(ranks) / 2.0
    dev = abs(w_plus - mean_w)
    extreme = sum(c for s, c in counts.items() if abs(s - mean_w) >= dev - 1e-9)
    return min(1.0, extreme / total)


def analyze(name: str, off: dict, on: dict, note: str = "") -> dict:
    cells = sorted(set(off) & set(on))
    diffs = [on[k] - off[k] for k in cells]
    n = len(diffs)
    res: dict = {
        "experiment": name, "note": note, "n_cells": n,
        "datasets": sorted({d for d, _ in cells}),
        "cells": [
            {"dataset": d, "seed": s, "off": off[(d, s)], "on": on[(d, s)],
             "delta": on[(d, s)] - off[(d, s)]}
            for d, s in cells
        ],
    }
    if n < 2:
        res["verdict"] = "INSUFFICIENT DATA"
        return res

    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n) if n else float("nan")
    tcrit = _t_ppf(0.975, n - 1)
    lo, hi = mean - tcrit * se, mean + tcrit * se

    # MDE at 80% power, alpha .05 two-sided, for the n actually run.
    mde = (_t_ppf(0.975, n - 1) + _t_ppf(0.80, n - 1)) * sd / math.sqrt(n) if sd else 0.0
    need = (((_z_ppf(0.975) + _z_ppf(0.80)) * sd / TARGET_DELTA) ** 2) if sd else 0.0

    n_win = sum(1 for d in diffs if d > 1e-12)
    n_tie = sum(1 for d in diffs if abs(d) <= 1e-12)
    n_lose = n - n_win - n_tie

    # Verdict. The distinction that matters is between "we showed the effect
    # is smaller than we care about" (an equivalence result, which needs the
    # CI to sit INSIDE +/-TARGET_DELTA) and "we could not have seen it".
    if lo > 0:
        verdict = "BENEFIT (CI excludes 0)"
    elif hi < 0:
        verdict = "HARM (CI excludes 0)"
    elif hi < TARGET_DELTA and lo > -TARGET_DELTA:
        verdict = f"NULL CONFIRMED (equivalent within +/-{TARGET_DELTA})"
    elif hi < TARGET_DELTA:
        verdict = "NO MEANINGFUL BENEFIT (harm not ruled out)"
    elif lo > -TARGET_DELTA:
        verdict = "NO MEANINGFUL HARM (benefit not ruled out)"
    else:
        verdict = "UNDERPOWERED (cannot distinguish from zero)"

    # Dataset-level analysis. Seeds within a dataset are replicates; datasets
    # are NOT replicates of each other, so pooling all cells overstates n
    # whenever the effect is dataset-dependent -- which, for this project, it
    # visibly is (E6/E7: helps sonar, hurts ionosphere). Treating the
    # per-dataset mean as the unit is the conservative reading.
    per_ds = {
        d: sum(c["delta"] for c in res["cells"] if c["dataset"] == d)
           / sum(1 for c in res["cells"] if c["dataset"] == d)
        for d in res["datasets"]
    }
    ds_vals = list(per_ds.values())
    ds_n = len(ds_vals)
    ds_block = None
    if ds_n >= 2:
        ds_mean = sum(ds_vals) / ds_n
        ds_sd = math.sqrt(sum((v - ds_mean) ** 2 for v in ds_vals) / (ds_n - 1))
        ds_se = ds_sd / math.sqrt(ds_n)
        ds_t = _t_ppf(0.975, ds_n - 1)
        ds_block = {
            "n_datasets": ds_n, "mean": ds_mean, "sd": ds_sd,
            "ci95": [ds_mean - ds_t * ds_se, ds_mean + ds_t * ds_se],
            "heterogeneous": (max(ds_vals) - min(ds_vals)) > 2 * TARGET_DELTA,
        }

    res.update({
        "mean_delta": mean, "sd_delta": sd, "se": se,
        "ci95": [lo, hi], "wilcoxon_p": _wilcoxon_p(diffs),
        "mde_80pct_power": mde, "n_needed_for_0.01": math.ceil(need) if need else 0,
        "win_tie_lose": [n_win, n_tie, n_lose], "verdict": verdict,
        "per_dataset": per_ds, "dataset_level": ds_block,
    })
    return res


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def collect() -> tuple[list[dict], list[str]]:
    reports: list[dict] = []
    skipped: list[str] = []

    def _log(*parts):
        return os.path.join(RESULTS, *parts)

    pairs = [
        ("KAN basis (use_kan_basis)",
         _log("gate2_kan", "OFF_sparse_linear.log"),
         _log("gate2_kan", "ON_sparse_linear.log"),
         "recorded verdict: FAILED, mean -0.0022"),
        ("Architecture race @ incumbent margin 0.01",
         _log("gate2_incumbent_margin", "classic.log"),
         _log("gate2_incumbent_margin", "race_m001.log"),
         "recorded verdict: FAILED, mean +0.0010"),
        ("EBM cyclic fit (sonar only)",
         _log("gate2_ebm", "sonar_sl_OFF.log"),
         _log("gate2_ebm", "sonar_sl_ON.log"),
         "recorded verdict: sonar +0.0080"),
    ]
    for name, off_p, on_p, note in pairs:
        if not (os.path.exists(off_p) and os.path.exists(on_p)):
            skipped.append(f"{name}: missing log file")
            continue
        off, on = read_crossseed_log(off_p), read_crossseed_log(on_p)
        if not (off and on):
            skipped.append(f"{name}: no per-seed rows parsed")
            continue
        reports.append(analyze(name, off, on, note))

    di = _log("gate2_direct_install", "report.json")
    if os.path.exists(di):
        off, on = read_direct_install(di)
        reports.append(analyze("Direct formula install", off, on,
                               "per-seed rows stored in JSON"))

    # Artifacts that stored ONLY arm means -- re-analysis is impossible.
    for path, why in [
        (_log("try_uil_connect", "report.json"),
         "stores acc_mean/acc_std only, no per-seed values"),
        (_log("gate2_utility", "SUMMARY.md"), "no OFF-arm log retained"),
        (_log("gate2_category_d", "SUMMARY.md"), "no OFF-arm log retained"),
    ]:
        if os.path.exists(path):
            skipped.append(f"{os.path.relpath(path, RESULTS)}: {why}")

    return reports, skipped


def render(reports: list[dict], skipped: list[str]) -> str:
    out = ["# Gate 2 re-scoring — paired statistics and power",
           "",
           "Re-analysis of stored artifacts only. No new runs.",
           f"Target effect for the power calculation: delta = {TARGET_DELTA}.",
           "",
           "A verdict of NULL CONFIRMED means the experiment genuinely showed",
           "the effect is smaller than we care about. UNDERPOWERED means it",
           "could not have shown that either way — the recorded number is",
           "compatible with a large effect in both directions.",
           "",
           "| experiment | cells | mean | 95% CI | needed for 0.01 | verdict |",
           "|---|---:|---:|---|---:|---|"]
    for r in reports:
        if r["n_cells"] < 2:
            out.append(f"| {r['experiment']} | {r['n_cells']} | — | — | — | "
                       f"{r['verdict']} |")
            continue
        out.append(
            f"| {r['experiment']} | {r['n_cells']} | {r['mean_delta']:+.4f} | "
            f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | "
            f"{r['n_needed_for_0.01']} | {r['verdict']} |"
        )
    out.append("")

    for r in reports:
        out.append(f"## {r['experiment']}")
        if r.get("note"):
            out.append(f"_{r['note']}_")
        out.append("")
        if r["n_cells"] < 2:
            out.append(f"**{r['verdict']}** — only {r['n_cells']} paired cell(s).")
            out.append("")
            continue
        w, t, l = r["win_tie_lose"]
        out.append(f"- paired cells: **{r['n_cells']}** "
                   f"({len(r['datasets'])} datasets: {', '.join(r['datasets'])})")
        out.append(f"- mean paired delta: **{r['mean_delta']:+.4f}** "
                   f"(sd {r['sd_delta']:.4f})")
        out.append(f"- 95% CI: **[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]**")
        p = r.get("wilcoxon_p")
        out.append(f"- Wilcoxon signed-rank p: "
                   f"{'%.3f' % p if p is not None else 'n/a'}")
        out.append(f"- win/tie/lose: {w}/{t}/{l}")
        out.append(f"- minimum detectable effect at this n (80% power): "
                   f"**{r['mde_80pct_power']:.4f}**")
        out.append(f"- cells needed to detect {TARGET_DELTA}: "
                   f"**{r['n_needed_for_0.01']}** (ran {r['n_cells']})")
        out.append(f"- **verdict: {r['verdict']}**")
        dsb = r.get("dataset_level")
        if dsb:
            out.append(f"- dataset-level (n={dsb['n_datasets']} datasets, the "
                       f"conservative unit): mean {dsb['mean']:+.4f}, "
                       f"95% CI [{dsb['ci95'][0]:+.4f}, {dsb['ci95'][1]:+.4f}]")
            if dsb["heterogeneous"]:
                spread = max(r["per_dataset"].values()) - min(r["per_dataset"].values())
                out.append(f"- **heterogeneous across datasets (spread "
                           f"{spread:.4f}) — the pooled mean hides opposite "
                           f"effects; do not quote it alone**")
        out.append("")
        out.append("| dataset | mean delta |")
        out.append("|---|---:|")
        for d, v in sorted(r["per_dataset"].items()):
            out.append(f"| {d} | {v:+.4f} |")
        out.append("")
    if skipped:
        out.append("## Not re-analysable from stored artifacts")
        out.append("")
        for s in skipped:
            out.append(f"- {s}")
        out.append("")
        out.append("These recorded a verdict but did not retain the per-seed "
                   "values needed to reproduce it. Future gates should write "
                   "per-(dataset, seed) rows to JSON, not just arm means.")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="also write raw results here")
    ap.add_argument("--out", default=None, help="write the markdown report here")
    args = ap.parse_args()

    reports, skipped = collect()
    text = render(reports, skipped)
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"reports": reports, "skipped": skipped}, fh, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
