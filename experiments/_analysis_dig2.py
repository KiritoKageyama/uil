"""Supplementary dig: XGBoost ablations, active KG subgraph, regret/CPU."""
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "results" / "ablation_reports"
NETWORK = ROOT / "results" / "uil_network_v20.json"

def pct(xs, p):
    xs = sorted(xs)
    return xs[int(len(xs) * p / 100)] if xs else 0

def xgboost_ladder():
    p = REPORTS / "ablation_full_300_2026-06-28.json"
    data = json.load(open(p, encoding="utf-8"))
    ladder = ["Base", "+Principles", "+Formula Evolution", "+Network", "+Causal"]
    print("XGBoost 300-stage ladder (mean over 5 seeds):")
    print(f"{'dataset':18s} {'step':22s} {'acc':>6s} {'d_acc':>7s} {'regret':>7s} {'cpu_s':>8s} {'principles':>10s} {'edges':>8s}")
    for ds, steps in data["datasets"].items():
        by = {s["step"]: s for s in steps}
        base = by["Base"]["stats"]["accuracy"]["mean"]
        prev = None
        for step in ladder:
            s = by[step]["stats"]
            acc = s["accuracy"]["mean"]
            d = acc - base if step != "Base" else 0
            dp = acc - prev if prev is not None else 0
            print(
                f"{ds:18s} {step:22s} {acc:6.3f} {d:+7.3f} "
                f"{s['mean_regret']['mean']:7.4f} {s['cpu_seconds']['mean']:8.1f} "
                f"{s.get('n_principles',{}).get('mean',0):10.1f} "
                f"{s.get('network_edges',{}).get('mean',0):8.1f}"
            )
            prev = acc
        print()

def kg_active_subgraph():
    data = json.load(open(NETWORK, encoding="utf-8"))
    nodes = data["nodes"]
    edges = data["edges"]
    active = {nid for nid, n in nodes.items() if n.get("activation_count", 0) > 0}
    dormant = set(nodes) - active
    print(f"KG nodes: {len(nodes)} total, {len(active)} activated (>0), {len(dormant)} dormant")
    act_confs = [nodes[n]["confidence"] for n in active]
    dor_confs = [nodes[n]["confidence"] for n in dormant]
    act_overrides = sum(1 for n in active if nodes[n].get("action_override"))
    print(f"  Active node conf median: {pct(act_confs,50):.3f}  dormant: {pct(dor_confs,50):.3f}")
    print(f"  Active nodes with action_override: {act_overrides}/{len(active)} ({100*act_overrides/max(1,len(active)):.0f}%)")

    active_edges = [e for e in edges if e["source_id"] in active and e["target_id"] in active]
    touch_active = [e for e in edges if e["source_id"] in active or e["target_id"] in active]
    print(f"  Edges fully within active subgraph: {len(active_edges)}/{len(edges)}")
    print(f"  Edges touching at least one active node: {len(touch_active)}/{len(edges)}")

    def help_stats(es):
        hrs = []
        for e in es:
            t, h = e.get("times_traversed", 0), e.get("times_helped", 0)
            hrs.append(0.5 if t == 0 else h / t)
        return {
            "n": len(hrs),
            "help_med": pct(hrs, 50),
            "help_lt_0.2": sum(1 for x in hrs if x < 0.2),
            "never_traversed": sum(1 for e in es if e.get("times_traversed", 0) == 0),
        }
    for label, es in [("all edges", edges), ("active subgraph", active_edges), ("touch active", touch_active)]:
        st = help_stats(es)
        print(f"  {label}: help_ratio median={st['help_med']:.3f}, "
              f"low_help={st['help_lt_0.2']}/{st['n']} ({100*st['help_lt_0.2']/max(1,st['n']):.0f}%), "
              f"never_traversed={st['never_traversed']}")

    et = Counter(e["edge_type"] for e in edges)
    print(f"  Edge types: {dict(et)}")
    print(f"  Contradicts share: {et['contradicts']/len(edges)*100:.0f}%")

def sym_xgb_headline():
    pairs = [
        ("symbolicformula_postfix_2026-06-30.json", "xgboost_postfix_2026-06-30.json"),
        ("symbolicformula_heldout_2026-06-30.json", "xgboost_heldout_2026-06-30.json"),
    ]
    for sf, xf in pairs:
        sd = json.load(open(REPORTS / sf, encoding="utf-8"))
        xd = json.load(open(REPORTS / xf, encoding="utf-8"))
        print(f"\nHeadline: {sf} vs {xf} (+Causal step, 3 seeds)")
        for ds in sd["datasets"]:
            sym = next(s for s in sd["datasets"][ds] if s["step"] == "+Causal")
            xgb = next(s for s in xd["datasets"][ds] if s["step"] == "+Causal")
            sa, xa = sym["stats"]["accuracy"]["mean"], xgb["stats"]["accuracy"]["mean"]
            sh = sym["stats"].get("held_out_accuracy", {}).get("mean")
            xh = xgb["stats"].get("held_out_accuracy", {}).get("mean")
            sc = sym["stats"]["cpu_seconds"]["mean"]
            xc = xgb["stats"]["cpu_seconds"]["mean"]
            print(f"  {ds:18s} train Sym={sa:.3f} XGB={xa:.3f} | held Sym={sh} XGB={xh} | cpu Sym={sc:.2f}s XGB={xc:.2f}s")

if __name__ == "__main__":
    xgboost_ladder()
    print("=" * 70)
    kg_active_subgraph()
    print("=" * 70)
    sym_xgb_headline()
