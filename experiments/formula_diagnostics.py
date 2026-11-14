"""Phase A evolution diagnostics: measure the formula population's lineage
and structural diversity from a saved FormulaLibrary JSON file. Read-only --
makes no changes to selection/mutation logic. Run this BEFORE and AFTER any
algorithm change (e.g. stratified tournament) to confirm the change actually
moved the numbers, instead of assuming it did.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from figpso_uil.cognition.paths import results_path


def load_library(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def tree_depth(tree: dict) -> int:
    children = tree.get("children")
    if not children:
        return 1
    return 1 + max(tree_depth(c) for c in children)


def tree_node_count(tree: dict) -> int:
    children = tree.get("children")
    if not children:
        return 1
    return 1 + sum(tree_node_count(c) for c in children)


def tree_shape_hash(tree: dict) -> str:
    """Canonical structural signature -- op names and shape, ignoring which
    leaf variable/constant is used, so two formulas with the same skeleton
    but different leaves are NOT counted as structurally distinct."""
    if "op" in tree:
        kids = ",".join(tree_shape_hash(c) for c in tree.get("children", []))
        return f"{tree['op']}({kids})"
    return "L"  # any leaf (var or const) collapses to one symbol


def shannon_entropy(counts: list[int]) -> tuple[float, float]:
    """Returns (raw_entropy_bits, normalized_0_to_1)."""
    total = sum(counts)
    if total == 0:
        return 0.0, 0.0
    probs = [c / total for c in counts if c > 0]
    raw = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    normalized = raw / max_entropy if max_entropy > 0 else 0.0
    return raw, normalized


def trace_root_lineage(formulas: list[dict]) -> dict[str, str]:
    """Map every formula_id -> the seed formula_id its lineage traces back
    to, by walking the origin chain (origin is either 'seed' or
    'mutation_of_<parent_id>')."""
    by_id = {f["formula_id"]: f for f in formulas}

    def root_of(fid: str, _seen: set[str] | None = None) -> str:
        _seen = _seen or set()
        if fid in _seen:
            return fid  # cycle guard, shouldn't happen
        _seen.add(fid)
        entry = by_id.get(fid)
        if entry is None:
            return fid
        origin = entry.get("origin", "seed")
        if origin == "seed" or not origin.startswith("mutation_of_"):
            return fid
        parent_id = origin[len("mutation_of_"):]
        if parent_id not in by_id:
            return fid
        return root_of(parent_id, _seen)

    return {f["formula_id"]: root_of(f["formula_id"]) for f in formulas}


def analyze(path: str) -> dict:
    formulas = load_library(path)
    root_of = trace_root_lineage(formulas)

    lineage_counts = Counter(root_of.values())
    lineage_root_names = {
        fid: next((f["name"] for f in formulas if f["formula_id"] == fid), fid)
        for fid in lineage_counts
    }

    raw_h, norm_h = shannon_entropy(list(lineage_counts.values()))

    # Parent-selection entropy: how concentrated are tournament wins?
    parent_use_counts = [f.get("times_used_as_parent", 0) for f in formulas]
    parent_raw_h, parent_norm_h = shannon_entropy(
        [c for c in parent_use_counts if c > 0] or [0]
    )
    n_distinct_parents_used = sum(1 for c in parent_use_counts if c > 0)

    # Structural diversity
    depths = [tree_depth(f["tree"]) for f in formulas]
    node_counts = [tree_node_count(f["tree"]) for f in formulas]
    shapes = [tree_shape_hash(f["tree"]) for f in formulas]
    distinct_shapes = len(set(shapes))

    # Survival by lineage: active+seed+incubating vs retired
    survival: dict[str, dict[str, int]] = defaultdict(lambda: {"alive": 0, "retired": 0})
    for f in formulas:
        root = root_of[f["formula_id"]]
        key = "retired" if f["status"] == "retired" else "alive"
        survival[root][key] += 1

    # Age proxy: numeric suffix of F-#### ids (SEED-* treated as age 0)
    def age(fid: str) -> int:
        if fid.startswith("F-"):
            try:
                return int(fid.split("-")[1])
            except (IndexError, ValueError):
                return -1
        return 0

    ages_by_lineage: dict[str, list[int]] = defaultdict(list)
    for f in formulas:
        ages_by_lineage[root_of[f["formula_id"]]].append(age(f["formula_id"]))

    # True stage-based lifespan (created_at_stage), when present. Older
    # saved libraries predate this field -- falls back to "n/a" rather
    # than silently mixing in the ID-ordinal proxy as if it were stages.
    stages_by_lineage: dict[str, list[int]] = defaultdict(list)
    has_stage_field = any(f.get("created_at_stage") is not None for f in formulas)
    if has_stage_field:
        for f in formulas:
            s = f.get("created_at_stage")
            if s is not None:
                stages_by_lineage[root_of[f["formula_id"]]].append(s)

    return {
        "n_formulas": len(formulas),
        "n_lineages_with_descendants": sum(1 for c in lineage_counts.values() if c > 1),
        "n_seed_lineages_total": len(lineage_counts),
        "lineage_sizes": {
            lineage_root_names[fid]: count for fid, count in lineage_counts.items()
        },
        "lineage_entropy_bits": round(raw_h, 4),
        "lineage_entropy_normalized": round(norm_h, 4),
        "parent_selection_entropy_bits": round(parent_raw_h, 4),
        "parent_selection_entropy_normalized": round(parent_norm_h, 4),
        "n_distinct_formulas_ever_used_as_parent": n_distinct_parents_used,
        "structural": {
            "depth_mean": round(sum(depths) / len(depths), 2),
            "depth_min": min(depths),
            "depth_max": max(depths),
            "node_count_mean": round(sum(node_counts) / len(node_counts), 2),
            "node_count_min": min(node_counts),
            "node_count_max": max(node_counts),
            "distinct_tree_shapes": distinct_shapes,
            "shape_diversity_ratio": round(distinct_shapes / len(formulas), 4),
        },
        "survival_by_lineage": {
            lineage_root_names[root]: dict(v) for root, v in survival.items()
        },
        "age_range_by_lineage": {
            lineage_root_names[root]: (min(ages), max(ages))
            for root, ages in ages_by_lineage.items()
        },
        "stage_lifespan_by_lineage": (
            {
                lineage_root_names[root]: {
                    "birth_stage": min(stages),
                    "last_active_stage": max(stages),
                    "lifespan": max(stages) - min(stages),
                }
                for root, stages in stages_by_lineage.items()
            }
            if has_stage_field else None
        ),
    }


def print_report(result: dict, label: str = "") -> None:
    print("=" * 70)
    if label:
        print(f"FORMULA POPULATION DIAGNOSTICS -- {label}")
    print(f"Total formulas: {result['n_formulas']}")
    print(f"Seed lineages with at least one descendant: "
          f"{result['n_lineages_with_descendants']} / {result['n_seed_lineages_total']}")
    print()
    print("Lineage sizes (formulas per root seed):")
    for name, size in sorted(result["lineage_sizes"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:<12} {size}")
    print()
    print(f"Lineage entropy: {result['lineage_entropy_bits']:.3f} bits "
          f"(normalized {result['lineage_entropy_normalized']:.3f} -- "
          f"1.0 = perfectly even across all seed lineages, 0.0 = total collapse)")
    print(f"Parent-selection entropy (among formulas ever chosen as a parent): "
          f"{result['parent_selection_entropy_bits']:.3f} bits "
          f"(normalized {result['parent_selection_entropy_normalized']:.3f})")
    print(f"Distinct formulas ever used as a mutation parent: "
          f"{result['n_distinct_formulas_ever_used_as_parent']}")
    print()
    s = result["structural"]
    print(f"Tree depth: mean={s['depth_mean']} range=[{s['depth_min']}, {s['depth_max']}]")
    print(f"Node count: mean={s['node_count_mean']} range=[{s['node_count_min']}, {s['node_count_max']}]")
    print(f"Distinct tree shapes: {s['distinct_tree_shapes']} / {result['n_formulas']} "
          f"(ratio {s['shape_diversity_ratio']:.3f} -- 1.0 = every formula structurally unique)")
    print()
    print("Survival by lineage (alive = active/seed/incubating):")
    for name, counts in result["survival_by_lineage"].items():
        alive, retired = counts.get("alive", 0), counts.get("retired", 0)
        total = alive + retired
        rate = alive / total if total else 0.0
        print(f"  {name:<12} alive={alive:<4} retired={retired:<4} survival_rate={rate:.2f}")
    print()
    print("Formula-ID age range by lineage (min, max -- proxy for when each lineage was active):")
    for name, (lo, hi) in result["age_range_by_lineage"].items():
        print(f"  {name:<12} [{lo}, {hi}]")
    print()
    if result.get("stage_lifespan_by_lineage") is not None:
        print("Lineage lifespan (true optimizer stage, not mutation-order proxy):")
        print(f"  {'lineage':<12} {'birth':>6} {'last_active':>12} {'lifespan':>9}")
        for name, v in result["stage_lifespan_by_lineage"].items():
            print(f"  {name:<12} {v['birth_stage']:>6} {v['last_active_stage']:>12} {v['lifespan']:>9}")
    else:
        print("Lineage lifespan (true stage): n/a -- this file predates the "
              "created_at_stage field, only the ID-ordinal proxy above is available.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Formula population diversity diagnostics (read-only)")
    parser.add_argument("--path", default=results_path("uil_formula_library.json"))
    parser.add_argument("--label", default="")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    result = analyze(args.path)
    print_report(result, label=args.label or args.path)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))
        print(f"\nSaved JSON to {args.json_out}")


if __name__ == "__main__":
    main()
