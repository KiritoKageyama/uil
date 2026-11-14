"""One-off analysis: ablation lift, KG quality, SymbolicFormula vs XGBoost."""
from __future__ import annotations
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "results" / "ablation_reports"
NETWORK = ROOT / "results" / "uil_network_v20.json"


def load_json(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def ablation_lift(report_path: Path) -> list[dict]:
    data = load_json(report_path)
    if not isinstance(data, dict):
        return []
    datasets = data.get("datasets", {})
    if not isinstance(datasets, dict):
        return []
    rows = []
    for dataset, steps in datasets.items():
        by_step = {s["step"]: s for s in steps}
        base_acc = by_step.get("Base", {}).get("stats", {}).get("accuracy", {}).get("mean")
        if base_acc is None:
            continue
        for step_name, step in by_step.items():
            acc = step["stats"]["accuracy"]["mean"]
            regret = step["stats"].get("mean_regret", {}).get("mean")
            cpu = step["stats"].get("cpu_seconds", {}).get("mean")
            rows.append({
                "report": report_path.name,
                "dataset": dataset,
                "step": step_name,
                "accuracy": acc,
                "delta_vs_base": acc - base_acc if step_name != "Base" else 0.0,
                "mean_regret": regret,
                "cpu_seconds": cpu,
                "n_principles": step["stats"].get("n_principles", {}).get("mean", 0),
                "network_edges": step["stats"].get("network_edges", {}).get("mean", 0),
            })
    return rows


def summarize_ablations():
    all_rows = []
    for p in sorted(REPORTS.glob("*.json")):
        try:
            all_rows.extend(ablation_lift(p))
        except Exception as e:
            print(f"SKIP {p.name}: {e}")

    # Layer incremental lifts (only reports with standard 5-step ladder)
    ladder = ["Base", "+Principles", "+Formula Evolution", "+Network", "+Causal"]
    incremental = []
    for p in sorted(REPORTS.glob("ablation_full*.json")):
        data = load_json(p)
        if not isinstance(data, dict):
            continue
        datasets = data.get("datasets", {})
        if not isinstance(datasets, dict):
            continue
        for dataset, steps in datasets.items():
            by_step = {s["step"]: s for s in steps}
            if not all(s in by_step for s in ladder):
                continue
            prev_acc = None
            for step in ladder:
                acc = by_step[step]["stats"]["accuracy"]["mean"]
                if prev_acc is not None:
                    incremental.append({
                        "report": p.name,
                        "dataset": dataset,
                        "transition": f"{ladder[ladder.index(step)-1]} -> {step}",
                        "delta": acc - prev_acc,
                        "acc": acc,
                    })
                prev_acc = acc

    # Best/worst full-stack vs base
    full_vs_base = []
    for p in sorted(REPORTS.glob("*.json")):
        data = load_json(p)
        if not isinstance(data, dict):
            continue
        datasets = data.get("datasets", {})
        if not isinstance(datasets, dict):
            continue
        for dataset, steps in datasets.items():
            by_step = {s["step"]: s for s in steps}
            if "Base" not in by_step:
                continue
            full = by_step.get("+Causal") or by_step.get("Full")
            if not full:
                continue
            b = by_step["Base"]["stats"]["accuracy"]["mean"]
            f = full["stats"]["accuracy"]["mean"]
            full_vs_base.append({
                "report": p.name,
                "dataset": dataset,
                "base": b,
                "full": f,
                "delta": f - b,
                "classifier": data.get("classifier", "?"),
                "stages": data.get("stages"),
            })

    return all_rows, incremental, full_vs_base


def analyze_network():
    data = load_json(NETWORK)
    nodes = data.get("nodes", {})
    edges = data.get("edges", [])
    co_act = data.get("co_activation_tally", {})

    node_confs = [n.get("confidence", 0) for n in nodes.values()]
    node_evidence = [n.get("evidence_count", 0) for n in nodes.values()]
    node_activations = [n.get("activation_count", 0) for n in nodes.values()]
    node_tiers = Counter(n.get("tier", 0) for n in nodes.values())
    empty_overrides = sum(1 for n in nodes.values() if not n.get("action_override"))

    edge_types = Counter(e.get("edge_type", "?") for e in edges)
    edge_weights = [e.get("weight", 0) for e in edges]
    edge_conf = [e.get("confidence", 0) for e in edges]
    edge_help = []
    never_traversed = 0
    low_help = 0
    for e in edges:
        t = e.get("times_traversed", 0)
        h = e.get("times_helped", 0)
        if t == 0:
            never_traversed += 1
            edge_help.append(0.5)
        else:
            hr = h / t
            edge_help.append(hr)
            if hr < 0.2:
                low_help += 1

    def pct(xs, p):
        if not xs:
            return 0
        xs = sorted(xs)
        return xs[int(len(xs) * p / 100)]

    return {
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_co_activation_pairs": len(co_act),
        "node_conf_median": pct(node_confs, 50),
        "node_conf_p90": pct(node_confs, 90),
        "node_evidence_median": pct(node_evidence, 50),
        "node_activation_median": pct(node_activations, 50),
        "node_activation_p90": pct(node_activations, 90),
        "never_activated": sum(1 for a in node_activations if a == 0),
        "empty_action_override": empty_overrides,
        "node_tiers": dict(node_tiers),
        "edge_types": dict(edge_types),
        "edge_weight_median": pct(edge_weights, 50),
        "edge_conf_median": pct(edge_conf, 50),
        "low_conf_edges_lt_0.3": sum(1 for c in edge_conf if c < 0.3),
        "high_conf_edges_ge_0.7": sum(1 for c in edge_conf if c >= 0.7),
        "help_ratio_median": pct(edge_help, 50) if edge_help else None,
        "edges_never_traversed": never_traversed,
        "edges_low_help_ratio_lt_0.2": low_help,
    }


def compare_classifiers():
    pairs = [
        ("symbolicformula_vs_xgboost_2026-06-30.json", "head-to-head 60 stages"),
        ("symbolicformula_matched_2026-06-30.json", None),  # may not exist
        ("xgboost_matched_2026-06-30.json", None),
        ("symbolicformula_postfix_2026-06-30.json", "postfix"),
        ("xgboost_postfix_2026-06-30.json", "postfix"),
        ("symbolicformula_heldout_2026-06-30.json", "heldout"),
        ("xgboost_heldout_2026-06-30.json", "heldout"),
        ("symbolicformula_moreseed_2026-06-30.json", "more seeds"),
    ]
    rows = []
    for fname, tag in pairs:
        p = REPORTS / fname.replace("symbolicformula_matched", "symbolicformula_vs_xgboost")
        if fname == "symbolicformula_matched_2026-06-30.json":
            continue
        p = REPORTS / fname
        if not p.exists():
            continue
        data = load_json(p)
        clf = data.get("classifier", "?")
        for dataset, steps in data.get("datasets", {}).items():
            for step in steps:
                acc = step["stats"]["accuracy"]["mean"]
                held = step["stats"].get("held_out_accuracy", {}).get("mean")
                rows.append({
                    "file": fname,
                    "tag": tag or fname,
                    "classifier": clf,
                    "dataset": dataset,
                    "step": step["step"],
                    "accuracy": acc,
                    "held_out": held,
                    "cpu": step["stats"].get("cpu_seconds", {}).get("mean"),
                    "n_seeds": step.get("n_seeds", 1),
                })
    return rows


def main():
    print("=" * 70)
    print("1. ABLATION LIFT")
    print("=" * 70)
    _, incremental, full_vs_base = summarize_ablations()

    if incremental:
        print("\nIncremental layer deltas (ablation_full reports):")
        by_trans = {}
        for r in incremental:
            key = (r["dataset"], r["transition"])
            by_trans.setdefault(key, []).append(r["delta"])
        for (ds, trans), deltas in sorted(by_trans.items(), key=lambda x: -max(x[1])):
            avg = sum(deltas) / len(deltas)
            print(f"  {ds:20s} {trans:35s} avg d_acc={avg:+.4f}  (n={len(deltas)})")

    print("\nFull stack (+Causal) vs Base — all reports:")
    for r in sorted(full_vs_base, key=lambda x: x["delta"], reverse=True):
        print(
            f"  {r['classifier']:16s} {r['dataset']:18s} "
            f"base={r['base']:.3f} full={r['full']:.3f} d={r['delta']:+.3f}  "
            f"({r['report']}, stages={r['stages']})"
        )

    print("\n" + "=" * 70)
    print("2. KNOWLEDGE GRAPH QUALITY")
    print("=" * 70)
    kg = analyze_network()
    for k, v in kg.items():
        print(f"  {k}: {v}")

    if kg["n_nodes"]:
        noise_signals = []
        if kg["never_activated"] / kg["n_nodes"] > 0.5:
            noise_signals.append(f"{kg['never_activated']/kg['n_nodes']*100:.0f}% nodes never activated")
        if kg["empty_action_override"] / kg["n_nodes"] > 0.5:
            noise_signals.append(f"{kg['empty_action_override']/kg['n_nodes']*100:.0f}% nodes have empty action_override")
        if kg["n_edges"] and kg.get("low_conf_edges_lt_0.3", 0) / kg["n_edges"] > 0.3:
            noise_signals.append(f"{kg['low_conf_edges_lt_0.3']/kg['n_edges']*100:.0f}% edges conf < 0.3")
        print("\n  Noise heuristics:", noise_signals or ["none triggered"])

    print("\n" + "=" * 70)
    print("3. SYMBOLICFORMULA vs XGBOOST")
    print("=" * 70)
    clf_rows = compare_classifiers()

    # Head-to-head on same datasets/steps
    sym = { (r["file"], r["dataset"], r["step"]): r for r in clf_rows if "Symbolic" in r["classifier"] }
    xgb = { (r["file"].replace("symbolic", "xgboost"), r["dataset"], r["step"]): r for r in clf_rows if "XGBoost" in r["classifier"] or r["classifier"] == "XGBoost" }

    # Direct comparison files
    for compare_file in ["symbolicformula_vs_xgboost_2026-06-30.json"]:
        sym_data = load_json(REPORTS / compare_file)
        xgb_file = REPORTS / "xgboost_matched_2026-06-30.json"
        if not xgb_file.exists():
            # try postfix pair
            sym_r = [r for r in clf_rows if r["file"] == compare_file]
            print(f"\nFrom {compare_file} (SymbolicFormula only in file):")
            for r in sym_r:
                print(f"  {r['dataset']:18s} {r['step']:20s} acc={r['accuracy']:.3f} held={r['held_out']}")

    # Matched pair comparison
    sym_m = load_json(REPORTS / "symbolicformula_postfix_2026-06-30.json") if (REPORTS / "symbolicformula_postfix_2026-06-30.json").exists() else None
    xgb_m = load_json(REPORTS / "xgboost_postfix_2026-06-30.json") if (REPORTS / "xgboost_postfix_2026-06-30.json").exists() else None
    if sym_m and xgb_m:
        print("\nPostfix matched runs (same config):")
        for ds in sym_m.get("datasets", {}):
            sym_steps = {s["step"]: s for s in sym_m["datasets"][ds]}
            xgb_steps = {s["step"]: s for s in xgb_m["datasets"][ds]}
            for step in sym_steps:
                if step not in xgb_steps:
                    continue
                sa = sym_steps[step]["stats"]["accuracy"]["mean"]
                xa = xgb_steps[step]["stats"]["accuracy"]["mean"]
                sh = sym_steps[step]["stats"].get("held_out_accuracy", {}).get("mean")
                xh = xgb_steps[step]["stats"].get("held_out_accuracy", {}).get("mean")
                print(f"  {ds:18s} {step:22s} Sym={sa:.3f} XGB={xa:.3f} d={sa-xa:+.3f}  held Sym={sh} XGB={xh}")

    sym_h = REPORTS / "symbolicformula_heldout_2026-06-30.json"
    xgb_h = REPORTS / "xgboost_heldout_2026-06-30.json"
    if sym_h.exists() and xgb_h.exists():
        sd, xd = load_json(sym_h), load_json(xgb_h)
        print("\nHeld-out accuracy focus:")
        for ds in sd.get("datasets", {}):
            for step_s, step_x in zip(sd["datasets"][ds], xd["datasets"][ds]):
                sa = step_s["stats"]["accuracy"]["mean"]
                xa = step_x["stats"]["accuracy"]["mean"]
                sh = step_s["stats"].get("held_out_accuracy", {}).get("mean")
                xh = step_x["stats"].get("held_out_accuracy", {}).get("mean")
                print(f"  {ds:18s} {step_s['step']:22s} train Sym={sa:.3f} XGB={xa:.3f} | held Sym={sh:.3f} XGB={xh:.3f}")


if __name__ == "__main__":
    main()
