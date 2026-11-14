"""Comprehensive UIL Test — Multiple datasets, per-stage tracking, honest reporting.

Tests on 4 datasets of varying difficulty:
1. Easy:   200 samples, 15 features, strong signal
2. Medium: 150 samples, 40 features, moderate signal
3. Hard:   100 samples, 80 features, weak signal, noise
4. Brutal: 60 samples, 120 features, 3 informative, heavy noise

Shows per-stage: accuracy, stress, regret, investigations, principles applied.
"""

import numpy as np
import sys, os, json, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from figpso_uil.adapters.xgboost_adapter import XGBoostAdapter
from figpso_uil.adapters.random_forest_adapter import RandomForestAdapter
from figpso_uil.adapters.knn_adapter import KNNAdapter
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch
from figpso_uil.core.optimizer import UniversalOptimizer
from figpso_uil.core.failsafe import FailsafeEngine
from figpso_uil.cognition.uil_miner import UIL_Miner
from figpso_uil.cognition.uil_library import UIL_Library
from sklearn.metrics import accuracy_score


# ── Dataset Generators ──────────────────────────────────────────────

def make_easy(seed=42):
    """Easy: 200 samples, 15 features, 8 informative, clear separation."""
    rng = np.random.default_rng(seed)
    n, f = 200, 15
    X = rng.standard_normal((n, f))
    # Strong signal in first 8 features
    signal = 0.8*X[:,0] + 0.6*X[:,1] + 0.5*X[:,2] + 0.4*X[:,3] + \
             0.3*X[:,4] + 0.3*X[:,5] + 0.2*X[:,6] + 0.2*X[:,7]
    noise = rng.normal(0, 0.3, n)
    y = (signal + noise > 0).astype(int)
    return X, y, "easy_200x15"


def make_medium(seed=42):
    """Medium: 150 samples, 40 features, 5 informative, some noise."""
    rng = np.random.default_rng(seed)
    n, f = 150, 40
    X = rng.standard_normal((n, f))
    signal = 0.5*X[:,0] + 0.4*X[:,1] + 0.3*X[:,2] + 0.2*X[:,3] + 0.15*X[:,4]
    noise = rng.normal(0, 0.5, n)
    y = (signal + noise > 0).astype(int)
    return X, y, "medium_150x40"


def make_hard(seed=42):
    """Hard: 100 samples, 80 features, 3 informative, weak signal."""
    rng = np.random.default_rng(seed)
    n, f = 100, 80
    X = rng.standard_normal((n, f))
    signal = 0.3*X[:,0] + 0.3*X[:,1] + 0.2*X[:,2]
    noise = rng.normal(0, 0.7, n)
    y = (signal + noise > 0).astype(int)
    return X, y, "hard_100x80"


def make_brutal(seed=42):
    """Brutal: 60 samples, 120 features, 3 informative, heavy noise."""
    rng = np.random.default_rng(seed)
    n, f = 60, 120
    X = rng.standard_normal((n, f))
    signal = 0.25*X[:,0] + 0.2*X[:,1] - 0.15*X[:,2]
    noise = rng.normal(0, 0.9, n)
    y = (signal + noise > 0).astype(int)
    return X, y, "brutal_60x120"


class Cfg:
    stages = 15
    n_particles = 15
    pso_iters = 20
    particles = 15
    pso_iters_per_stage = 20
    boosting_rounds_per_stage = 30


def run_one_dataset(X, y, dataset_name, classifier_name="XGBoost", seed=42):
    """Run UIL on one dataset and return per-stage results."""
    num_classes = len(np.unique(y))

    if classifier_name == "XGBoost":
        classifier = XGBoostAdapter(seed=seed, num_classes=num_classes, boosting_rounds=30)
    elif classifier_name == "RandomForest":
        classifier = RandomForestAdapter(seed=seed, num_classes=num_classes)
    elif classifier_name == "KNN":
        classifier = KNNAdapter(seed=seed, num_classes=num_classes)

    failsafe_adapters = [
        XGBoostAdapter(seed=seed, num_classes=num_classes, boosting_rounds=30),
        RandomForestAdapter(seed=seed, num_classes=num_classes),
        KNNAdapter(seed=seed, num_classes=num_classes),
    ]
    failsafe = FailsafeEngine(all_adapters=failsafe_adapters)

    search = PrincipleGuidedSearch()
    optimizer = UniversalOptimizer(
        cfg=Cfg(), seed=seed, dataset_name=dataset_name,
        fold=0, start_fold=0, classifier=classifier, search=search,
    )
    optimizer.set_failsafe(failsafe)

    result = optimizer.fit(X, y)
    return result


def analyze_journal_for_dataset(dataset_name, journal_path="results/uil_journal_v19.json"):
    """Extract per-stage data for a specific dataset."""
    with open(journal_path, "r") as f:
        data = json.load(f)
    
    entries = [e for e in data if e["dataset"] == dataset_name and e["classifier"] != "FAILSAFE"]
    return entries


def print_per_stage(entries, dataset_name, label=""):
    """Print per-stage accuracy/stress/regret for a dataset."""
    if not entries:
        print(f"  No entries for {dataset_name}")
        return

    print(f"\n  {'Stage':>5} | {'Accuracy':>8} | {'Stress':>8} | {'Regret':>8} | {'D.Stress':>8} | {'Penalty':>8} | {'Notes'}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*20}")
    
    prev_acc = 0.0
    for e in entries:
        stage = e["stage"]
        acc = e["outcome"]["accuracy"]
        ss = e["system_stress"]
        reg = e["regret"]
        ds = e["decision_stress"]
        pen = e["action"]["penalty"]
        
        notes = []
        if acc > prev_acc:
            notes.append("IMPROVED")
        elif acc < prev_acc:
            notes.append("DROPPED")
        if reg > 0:
            notes.append(f"REGRET")
        if e.get("investigation"):
            notes.append("INVESTIGATE")
        
        stress_bar = "#" * int(ss * 20)
        note_str = ", ".join(notes)
        
        print(f"  {stage:5d} | {acc:8.4f} | {ss:8.4f} | {reg:8.4f} | {ds:8.4f} | {pen:8.3f} | {note_str:20s} |{stress_bar}")
        prev_acc = acc


def main():
    print("=" * 90)
    print("  UIL V19 COMPREHENSIVE TEST — 4 Datasets, Per-Stage Tracking")
    print("=" * 90)

    # Clean previous results
    for path in ["results/uil_journal_v19.json", "results/uil_journal_v19_memory.json", 
                  "results/uil_principles_v19.json"]:
        if os.path.exists(path):
            os.remove(path)

    datasets = [
        make_easy(),
        make_medium(),
        make_hard(),
        make_brutal(),
    ]

    all_results = {}

    # ── Run all datasets with XGBoost ──
    for X, y, name in datasets:
        print(f"\n{'='*90}")
        print(f"  DATASET: {name}")
        print(f"  Shape: {X.shape[0]} samples x {X.shape[1]} features")
        print(f"  Classes: {np.bincount(y)}")
        print(f"{'='*90}")

        t0 = time.perf_counter()
        result = run_one_dataset(X, y, name, "XGBoost", seed=42)
        elapsed = time.perf_counter() - t0

        print(f"\n  RESULT: acc={result['accuracy']:.4f}, f1={result['f1_score']:.4f}, "
              f"features={result['n_selected']}/{X.shape[1]}, "
              f"stress={result['max_stress']:.4f}, "
              f"principles={result['principles_discovered']}, "
              f"time={elapsed:.1f}s")

        # Print per-stage
        entries = analyze_journal_for_dataset(name)
        print_per_stage(entries, name)

        all_results[name] = result

    # ── Run brutal dataset with different classifiers ──
    X_brutal, y_brutal, _ = make_brutal(seed=42)
    for clf_name in ["RandomForest", "KNN"]:
        ds_name = f"brutal_60x120_{clf_name}"
        print(f"\n{'='*90}")
        print(f"  DATASET: brutal_60x120 with {clf_name}")
        print(f"{'='*90}")

        t0 = time.perf_counter()
        result = run_one_dataset(X_brutal, y_brutal, ds_name, clf_name, seed=42)
        elapsed = time.perf_counter() - t0

        print(f"\n  RESULT: acc={result['accuracy']:.4f}, f1={result['f1_score']:.4f}, "
              f"features={result['n_selected']}/{X_brutal.shape[1]}, "
              f"stress={result['max_stress']:.4f}, time={elapsed:.1f}s")

        entries = analyze_journal_for_dataset(ds_name)
        print_per_stage(entries, ds_name)

        all_results[ds_name] = result

    # ── Summary Table ──
    print(f"\n\n{'='*90}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*90}")
    print(f"\n  {'Dataset':<30} | {'Classifier':<12} | {'Accuracy':>8} | {'F1':>8} | {'Features':>10} | {'MaxStress':>9} | {'Principles':>10}")
    print(f"  {'-'*30}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*9}-+-{'-'*10}")
    for name, r in all_results.items():
        clf = "XGBoost" if "XGBoost" not in name else "XGBoost"
        if "RandomForest" in name: clf = "RF"
        if "KNN" in name: clf = "KNN"
        print(f"  {name:<30} | {clf:<12} | {r['accuracy']:8.4f} | {r['f1_score']:8.4f} | {r['n_selected']:>10} | {r['max_stress']:9.4f} | {r['principles_discovered']:>10}")

    # ── Analyze overall journal ──
    print(f"\n\n{'='*90}")
    print(f"  JOURNAL & LEARNING ANALYSIS")
    print(f"{'='*90}")

    with open("results/uil_journal_v19.json", "r") as f:
        journal = json.load(f)
    normal = [e for e in journal if e["classifier"] != "FAILSAFE"]
    failsafes = [e for e in journal if e["classifier"] == "FAILSAFE"]

    print(f"\n  Total journal entries: {len(normal)}")
    print(f"  Failsafe diagnoses: {len(failsafes)}")
    print(f"  Entries with regret > 0: {sum(1 for e in normal if e['regret'] > 0)}")
    print(f"  Investigations triggered: {sum(1 for e in normal if e.get('investigation'))}")
    print(f"  Unique datasets: {len(set(e['dataset'] for e in normal))}")
    print(f"  Unique classifiers: {set(e['classifier'] for e in normal)}")

    # Memory
    mem_path = "results/uil_journal_v19_memory.json"
    if os.path.exists(mem_path):
        with open(mem_path, "r") as f:
            memory = json.load(f)
        print(f"\n  Dataset Memory:")
        for name, mem in memory.items():
            override = mem.get("initial_stress_override", 0)
            pref = mem.get("preferred_classifier", "none")
            print(f"    {name}: best_acc={mem.get('last_best_accuracy', 0):.4f}, "
                  f"stress_override={override:.3f}, preferred={pref}")

    # Principles
    lib_path = "results/uil_principles_v19.json"
    if os.path.exists(lib_path):
        with open(lib_path, "r") as f:
            principles = json.load(f)
        print(f"\n  Principles discovered: {len(principles)}")
        for p in principles[:5]:
            conds = "; ".join(f"{c['feature']} {c['op']} {c['threshold']}" for c in p["conditions"])
            print(f"    {p['principle_id']}: {conds}")
            print(f"      confidence={p['confidence']:.3f}, evidence={p['evidence_count']}, "
                  f"override={p.get('action_override', {})}")

    # Run miner manually on full journal to see what it would find
    print(f"\n  Manual Miner Run (all {len(normal)} entries):")
    miner = UIL_Miner(min_entries=15)
    rules = miner.mine(normal, stress_level=0.8)
    r2 = miner.get_last_r2_score()
    print(f"    R2 score: {r2:.4f}")
    print(f"    Rules discovered: {len(rules)}")
    promotable = [r for r in rules if r["confidence"] >= 0.75 and r["evidence_count"] >= 10]
    print(f"    Promotable rules (conf>=0.75, evidence>=10): {len(promotable)}")
    for r in promotable:
        conds = "; ".join(f"{c['feature']} {c['op']} {c['threshold']}" for c in r["conditions"])
        print(f"      {conds}")
        print(f"        regret={r['predicted_regret']:.4f}, conf={r['confidence']:.3f}, evidence={r['evidence_count']}")

    return all_results


if __name__ == "__main__":
    _all_results = main()
    # Sanity asserts — previously this file had zero asserts so it could
    # never catch regressions. These check the minimum expected outputs.
    if _all_results:
        assert len(_all_results) > 0, "main() returned empty results"
        for _name, _r in _all_results.items():
            assert "accuracy" in _r, f"result for {_name!r} missing 'accuracy'"
            _acc = _r["accuracy"]
            assert 0.0 <= _acc <= 1.0, f"{_name}: accuracy {_acc!r} out of [0,1]"
            assert _r.get("model") is not None, f"{_name}: model is None"
        print(f"ASSERTIONS PASSED ({len(_all_results)} datasets validated)")
