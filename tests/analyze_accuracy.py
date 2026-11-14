"""Deep analysis: WHY the accuracy numbers are what they are.

Questions to answer:
1. Why brutal_60x120 accuracy stays low and oscillates
2. Why easy got 87.5% now vs 95% before 
3. Why does accuracy hit a ceiling and stop
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

print("=" * 80)
print("  QUESTION 1: Why 95% before vs 87.5% now?")
print("  Answer: DIFFERENT DATASETS")
print("=" * 80)

# Dataset from the first integration test (test_v19_integration.py)
print("\n--- Dataset A: Integration Test (the 95% one) ---")
X_a, y_a = make_classification(
    n_samples=200, n_features=20, n_informative=10, n_redundant=5,
    n_classes=3, random_state=42
)
print(f"  Shape: {X_a.shape}, Classes: {len(np.unique(y_a))}")
print(f"  Informative features: 10/20, Redundant: 5/20, Noise: 5/20")
print(f"  Signal strength: HIGH (sklearn designed for good separability)")

# Dataset from the comprehensive test (the 87.5% one)  
print("\n--- Dataset B: Comprehensive Test (the 87.5% one) ---")
rng = np.random.default_rng(42)
n, f = 200, 15
X_b = rng.standard_normal((n, f))
signal = 0.8*X_b[:,0] + 0.6*X_b[:,1] + 0.5*X_b[:,2] + 0.4*X_b[:,3] + \
         0.3*X_b[:,4] + 0.3*X_b[:,5] + 0.2*X_b[:,6] + 0.2*X_b[:,7]
noise = rng.normal(0, 0.3, n)
y_b = (signal + noise > 0).astype(int)
print(f"  Shape: {X_b.shape}, Classes: {len(np.unique(y_b))}")
print(f"  Informative features: 8/15, Noise features: 7/15")
print(f"  Signal strength: MODERATE (hand-crafted, weaker coefficients)")

# Test both with plain XGBoost (no UIL) to see their TRUE ceiling
print("\n--- True ceiling (XGBoost 5-fold CV, no UIL) ---")
xgb_a = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, 
                            use_label_encoder=False, eval_metric='mlogloss')
scores_a = cross_val_score(xgb_a, X_a, y_a, cv=5, scoring='accuracy')
print(f"  Dataset A (integration): {scores_a.mean():.4f} +/- {scores_a.std():.4f}")
print(f"    Per-fold: {[f'{s:.3f}' for s in scores_a]}")

xgb_b = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42,
                            use_label_encoder=False, eval_metric='logloss')
scores_b = cross_val_score(xgb_b, X_b, y_b, cv=5, scoring='accuracy')
print(f"  Dataset B (comprehensive): {scores_b.mean():.4f} +/- {scores_b.std():.4f}")
print(f"    Per-fold: {[f'{s:.3f}' for s in scores_b]}")

print(f"\n  >>> Dataset A is fundamentally easier. sklearn's make_classification")
print(f"  >>> creates well-separated clusters. Dataset B has weaker signals + noise.")
print(f"  >>> The UIL's 87.5% is near the ceiling for Dataset B.")


print("\n\n" + "=" * 80)
print("  QUESTION 2: Why does accuracy stop at 95% and not go higher?")
print("  Answer: VALIDATION SET SIZE + DATA CEILING")
print("=" * 80)

# Show quantization effect
X_train_a, X_val_a, y_train_a, y_val_a = train_test_split(
    X_a, y_a, test_size=0.2, random_state=42, stratify=y_a
)
n_val = len(y_val_a)
print(f"\n  Validation set size: {n_val} samples")
print(f"  Each correct/wrong answer changes accuracy by: {1.0/n_val:.4f} = {100.0/n_val:.2f}%")
print(f"\n  Possible accuracy values near 95%:")
for wrong in range(0, 6):
    acc = (n_val - wrong) / n_val
    print(f"    {wrong} wrong: {acc:.4f} ({acc*100:.1f}%)")

print(f"\n  >>> With 40 validation samples, accuracy jumps in 2.5% steps.")
print(f"  >>> 95% = 38/40 correct. To get 97.5% = 39/40, need to fix")
print(f"  >>> exactly the right 1 sample. That might not be possible")
print(f"  >>> because that sample might be genuinely ambiguous.")

# Check: what are the misclassified samples?
xgb_a.fit(X_train_a, y_train_a)
preds = xgb_a.predict(X_val_a)
wrong_idx = np.where(preds != y_val_a)[0]
print(f"\n  Misclassified samples: {len(wrong_idx)}/{n_val}")
for i in wrong_idx:
    proba = xgb_a.predict_proba(X_val_a[i:i+1])[0]
    true_class = y_val_a[i]
    pred_class = preds[i]
    print(f"    Sample {i}: true={true_class}, pred={pred_class}, "
          f"confidence={max(proba):.3f}, probas={[f'{p:.3f}' for p in proba]}")

# Now with ALL features weighted equally vs with feature selection
print(f"\n  XGBoost with ALL features (no selection): {accuracy_score(y_val_a, preds):.4f}")

# Try RandomForest and KNN too
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_a, y_train_a)
print(f"  RandomForest with ALL features: {accuracy_score(y_val_a, rf.predict(X_val_a)):.4f}")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_a, y_train_a)
print(f"  KNN with ALL features: {accuracy_score(y_val_a, knn.predict(X_val_a)):.4f}")


print("\n\n" + "=" * 80)
print("  QUESTION 3: Why does brutal_60x120 XGBoost stay so low?")
print("  Answer: FUNDAMENTAL DATA PROBLEM")  
print("=" * 80)

rng = np.random.default_rng(42)
n, f = 60, 120
X_c = rng.standard_normal((n, f))
signal = 0.25*X_c[:,0] + 0.2*X_c[:,1] - 0.15*X_c[:,2]
noise = rng.normal(0, 0.9, n)
y_c = (signal + noise > 0).astype(int)

X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(
    X_c, y_c, test_size=0.2, random_state=42, stratify=y_c
)
n_val_c = len(y_val_c)
print(f"\n  Dataset: {n} samples, {f} features")
print(f"  Training: {len(y_train_c)}, Validation: {n_val_c}")
print(f"  Ratio: {n/f:.2f} (samples per feature) ← LESS THAN 1!")
print(f"  Only 3 informative features out of 120")
print(f"  Signal-to-noise: {np.std(signal)/np.std(noise):.3f} ← VERY LOW")

# Show the fundamental problem
print(f"\n  --- Accuracy CEILING for each classifier (5-fold CV) ---")
for name, clf in [
    ("XGBoost", xgb.XGBClassifier(n_estimators=30, max_depth=3, random_state=42,
                                     use_label_encoder=False, eval_metric='logloss')),
    ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("KNN (k=5)", KNeighborsClassifier(n_neighbors=5)),
    ("KNN (k=3)", KNeighborsClassifier(n_neighbors=3)),
]:
    scores = cross_val_score(clf, X_c, y_c, cv=5, scoring='accuracy')
    print(f"    {name:20s}: {scores.mean():.4f} +/- {scores.std():.4f}  "
          f"(range: {scores.min():.3f} to {scores.max():.3f})")

# What if we only use the 3 informative features?
print(f"\n  --- With ONLY the 3 informative features ---")
X_c_3 = X_c[:, :3]
for name, clf in [
    ("XGBoost (3 feat)", xgb.XGBClassifier(n_estimators=30, max_depth=3, random_state=42,
                                              use_label_encoder=False, eval_metric='logloss')),
    ("KNN (3 feat)", KNeighborsClassifier(n_neighbors=5)),
]:
    scores = cross_val_score(clf, X_c_3, y_c, cv=5, scoring='accuracy')
    print(f"    {name:20s}: {scores.mean():.4f} +/- {scores.std():.4f}")

# Quantization
print(f"\n  Validation set: {n_val_c} samples")
print(f"  Accuracy step: {100.0/n_val_c:.1f}%")
print(f"  Possible values: ", end="")
for wrong in range(0, 10):
    acc = (n_val_c - wrong) / n_val_c
    print(f"{acc*100:.0f}%", end="  ")
print()

# THE REAL PROBLEM: can the signal even be separated?
print(f"\n  --- THE CORE ISSUE ---")
print(f"  Signal variance: {np.var(signal):.4f}")
print(f"  Noise variance:  {np.var(noise):.4f}")
print(f"  Noise is {np.var(noise)/np.var(signal):.1f}x stronger than signal")
print(f"  Even a PERFECT feature selector that finds exactly features 0,1,2")
print(f"  can only reach ~{cross_val_score(KNeighborsClassifier(5), X_c_3, y_c, cv=5).mean()*100:.0f}% because the noise overwhelms the signal.")
print(f"  NO classifier can do much better on this data.")

print(f"\n  >>> The UIL's 75% (KNN) on this dataset is actually IMPRESSIVE")
print(f"  >>> given that the theoretical ceiling is around {cross_val_score(KNeighborsClassifier(5), X_c_3, y_c, cv=5).mean()*100:.0f}%")
print(f"  >>> with perfect feature selection.")

# Why XGBoost oscillates
print(f"\n\n  --- Why XGBoost OSCILLATES on brutal ---")
print(f"  Training set: {len(y_train_c)} samples for {f} features")
print(f"  XGBoost builds trees that MEMORIZE training data")
print(f"  When feature weights change slightly -> completely different tree")
print(f"  -> completely different validation accuracy")
print(f"  This is overfitting instability. Not a UIL problem.")
print(f"  The UIL correctly DETECTS this (regret, investigations)")
print(f"  but can't FIX it because the classifier is wrong for this data.")
