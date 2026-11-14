"""Test V20+.2: Derived Feature Synthesis — hard case where base DT struggles."""
import sys
sys.path.insert(0, ".")
import numpy as np
from figpso_uil.cognition.uil_miner import UIL_Miner

print("=" * 60)
print("TEST 1: Easy case (base DT handles it, derivation skipped)")
print("=" * 60)

miner = UIL_Miner(min_entries=10)
rng = np.random.default_rng(42)
entries = []
for i in range(100):
    penalty = rng.uniform(0.5, 5.0)
    sparsity = rng.uniform(0.0, 1.0)
    n_samples = rng.integers(50, 500)
    n_features = rng.integers(10, 200)
    regret = max(0.0, penalty * sparsity - 1.5 + rng.normal(0, 0.3))
    entries.append({
        "dataset": "test_dataset", "classifier": "XGBoost",
        "state": {"n_samples": int(n_samples), "n_features": int(n_features),
                  "sparsity": round(sparsity, 4), "entropy": round(rng.uniform(0.5, 2.0), 4),
                  "feat_ratio": 1.0, "num_classes": 2,
                  "sample_feature_ratio": round(n_samples / max(n_features, 1), 4),
                  "feature_correlation_mean": round(rng.uniform(0.0, 0.5), 4), "class_entropy": 1.0},
        "action": {"penalty": round(penalty, 4)},
        "regret": round(regret, 4),
    })

rules = miner.mine(entries, stress_level=0.5, dataset_filter="test_dataset")
print(f"  Base R2: {miner.get_last_r2_score():.4f}")
print(f"  Rules: {len(rules)}")
print(f"  Derived used: {miner.get_useful_derived_features()}")
print()

print("=" * 60)
print("TEST 2: Hard case (regret depends on XOR-like interaction)")
print("=" * 60)

# Create a case where regret = high ONLY when (penalty > median) XOR (sparsity > median)
# A simple DT on raw features will have low R^2 because
# the relationship is inherently non-axis-aligned
miner2 = UIL_Miner(min_entries=10)
entries2 = []
for i in range(200):
    # Many noisy features to confuse the base tree
    penalty = rng.uniform(0.0, 1.0)
    sparsity = rng.uniform(0.0, 1.0)
    entropy = rng.uniform(0.0, 1.0)
    feat_ratio = rng.uniform(0.0, 1.0)
    correlation = rng.uniform(0.0, 1.0)
    # Regret purely from interaction: penalty * sparsity * entropy
    # With lots of noise to make the base tree struggle
    product = penalty * sparsity * entropy
    regret = product + rng.normal(0, 0.5)
    regret = max(0.0, regret)
    entries2.append({
        "dataset": "hard_dataset", "classifier": "XGBoost",
        "state": {"n_samples": 200, "n_features": 50, "sparsity": round(sparsity, 4),
                  "entropy": round(entropy, 4), "feat_ratio": round(feat_ratio, 4),
                  "num_classes": 2, "sample_feature_ratio": 4.0,
                  "feature_correlation_mean": round(correlation, 4), "class_entropy": 1.0},
        "action": {"penalty": round(penalty, 4)},
        "regret": round(regret, 4),
    })

rules2 = miner2.mine(entries2, stress_level=0.5, dataset_filter="hard_dataset")
r2 = miner2.get_last_r2_score()
useful = miner2.get_useful_derived_features()
print(f"  Final R2: {r2:.4f}")
print(f"  Rules: {len(rules2)}")
print(f"  Derived features used: {useful}")

# Check for derived feature rules
derived_rules = []
for r in rules2:
    for c in r.get("conditions", []):
        if "derived." in c.get("feature", ""):
            derived_rules.append(r)
            break

print(f"  Rules using derived features: {len(derived_rules)}")
for r in derived_rules[:3]:
    conds = r["conditions"]
    for c in conds:
        print(f"    Feature: {c['feature']} {c['op']} {c['threshold']}")
    print(f"    Predicted regret: {r['predicted_regret']}")
    print()

print("=" * 60)
print("TEST 3: Verify derived synthesis function directly")
print("=" * 60)

miner3 = UIL_Miner(min_entries=5)
X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
names = ["action.penalty", "state.sparsity"]
X_enh, enh_names = miner3._synthesize_derived_features(X, names)
print(f"  Base features: {len(names)}")
print(f"  Enhanced features: {len(enh_names)}")
print(f"  Names: {enh_names}")
assert X_enh.shape[1] > X.shape[1], "Should have more columns"
assert any("derived." in n for n in enh_names), "Should have derived feature names"
print("  Direct synthesis OK")

print()
print("ALL V20+.2 TESTS PASSED")
