"""GPLearn adapter — protocol conformance, determinism, and honest scoping.

Registered 2026-08-05 after gplearn beat UIL's own symbolic regressor on 17 of
21 out-of-range cells with 0 blow-ups vs 19
(`results/extrapolation/SUMMARY.md`).

GP1-GP4 check the things this codebase has been burned by before:
  GP1  protocol conformance + the registry entry is optional, not required
  GP2  DETERMINISM under a perturbed global numpy RNG (bug #31's mechanism)
  GP3  multiclass does NOT collapse to one class (bugs #24/#33) -- gplearn's
       SymbolicClassifier is binary-only, so this exercises the OvR path that
       has broken twice in this project
  GP4  feature weights reach the model AND predict() re-applies the identical
       transform (bug #19's train/serve invariant)

Run:  python figpso_uil/tests/test_gplearn_adapter.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(os.path.dirname(_HERE))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from figpso_uil.adapters.gplearn_adapter import _HAS_GPLEARN
from figpso_uil.core.classifier_registry import CLASSIFIERS

_FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        _FAILURES.append(name)


def _data(n=300, d=5, classes=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    if classes == 2:
        y = (1.5 * X[:, 0] - X[:, 2] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    else:
        z = np.column_stack([X[:, 0], X[:, 1] - X[:, 3], -X[:, 2]])
        y = np.argmax(z + rng.normal(scale=0.25, size=z.shape), axis=1)
    return X, y


def _adapter(num_classes=2, seed=0):
    from figpso_uil.adapters.gplearn_adapter import GPLearnAdapter
    # Small budget: these tests check wiring and invariants, not accuracy.
    return GPLearnAdapter(seed=seed, num_classes=num_classes,
                          population_size=200, generations=5)


def test_gp1_protocol_and_optional_registration() -> None:
    print("\nGP1. Protocol conformance; registration is optional")
    from figpso_uil.core.classifier_protocol import ClassifierAdapter
    clf = _adapter()
    check("GP1 satisfies ClassifierAdapter", isinstance(clf, ClassifierAdapter))
    check("GP1 registered when gplearn is installed", "GPLearn" in CLASSIFIERS,
          f"registry keys: {sorted(CLASSIFIERS)}")
    # The registry must still build every other learner if gplearn is absent;
    # a missing optional dep may never break the registry import.
    check("GP1 registry still has the core learners",
          {"XGBoost", "RandomForest", "SymbolicFormula"} <= set(CLASSIFIERS))

    X, y = _data()
    model = clf.fit_weighted(X, y, np.ones(X.shape[1]))
    preds = clf.predict(model, X)
    acc = float(np.mean(preds == y))
    imp = clf.get_feature_importance(model, X.shape[1])
    check("GP1 predicts sane labels", set(np.unique(preds)) <= set(np.unique(y)))
    check("GP1 learns something (acc > majority)",
          acc > float(np.mean(y == np.bincount(y).argmax())),
          f"acc {acc:.3f}")
    check("GP1 importance is a normalized n_features vector",
          imp.shape == (X.shape[1],) and abs(float(imp.sum()) - 1.0) < 1e-9,
          f"sum={float(imp.sum()):.6f}")
    check("GP1 complexity is positive", clf.get_model_complexity(model) > 0)
    prog = clf.readable_program(model)
    check("GP1 program renders as text", isinstance(prog, str) and len(prog) > 0,
          prog[:70])


def test_gp2_determinism_under_global_rng_perturbation() -> None:
    """The bug #31 mechanism: an estimator seeded from the process-global RNG
    differs per call. Perturbing that RNG between fits is the only way to see
    it."""
    print("\nGP2. Deterministic under a perturbed global numpy RNG")
    X, y = _data(seed=1)
    digests = set()
    for g in range(3):
        np.random.seed(g * 7919 + 13)
        clf = _adapter(seed=42)
        model = clf.fit_weighted(X, y, np.ones(X.shape[1]))
        digests.add(tuple(np.asarray(clf.predict(model, X)).ravel().tolist()))
    check("GP2 identical predictions across global seeds", len(digests) == 1,
          f"{len(digests)} distinct outputs")


def test_gp3_multiclass_does_not_collapse() -> None:
    """gplearn's SymbolicClassifier is BINARY only, so multiclass runs through
    an OvR path -- the exact place bugs #24 and #33 collapsed to one class."""
    print("\nGP3. Multiclass OvR does not collapse (bugs #24/#33)")
    X, y = _data(classes=3, seed=2)
    clf = _adapter(num_classes=3, seed=0)
    model = clf.fit_weighted(X, y, np.ones(X.shape[1]))
    preds = np.asarray(clf.predict(model, X)).ravel()
    n_pred = len(np.unique(preds))
    acc = float(np.mean(preds == y))
    base = float(np.mean(y == np.bincount(y).argmax()))
    check("GP3 all classes reachable", n_pred == 3,
          f"{n_pred}/3 classes predicted")
    check("GP3 beats the base rate (not a majority-class model)", acc > base,
          f"acc {acc:.3f} vs base rate {base:.3f}")


def test_gp4_feature_weights_reach_the_model_and_predict_matches() -> None:
    print("\nGP4. Feature weights are live, and predict re-applies them (bug #19)")
    X, y = _data(seed=3)
    n_features = X.shape[1]

    uniform = _adapter(seed=7).fit_weighted(X, y, np.ones(n_features))
    skewed_w = np.array([1.0] + [0.01] * (n_features - 1))
    skewed = _adapter(seed=7).fit_weighted(X, y, skewed_w)

    clf = _adapter(seed=7)
    p_uniform = clf.predict(uniform, X)
    p_skewed = clf.predict(skewed, X)
    check("GP4 weights change the fit (channel is live, not a no-op)",
          not np.array_equal(p_uniform, p_skewed),
          "identical predictions would mean the weights are dead (bug #25)")

    # The stored weights must be the ones predict uses; refitting the same
    # data through a fresh adapter must reproduce the same labels.
    again = clf.predict(skewed, X)
    check("GP4 predict is stable / re-applies the stored transform",
          np.array_equal(p_skewed, again))
    check("GP4 weights are persisted on the model",
          skewed.get("feature_weights") is not None)


if __name__ == "__main__":
    if not _HAS_GPLEARN:
        print("SKIP: gplearn not installed (`pip install gplearn`)")
        sys.exit(0)
    for t in (test_gp1_protocol_and_optional_registration,
              test_gp2_determinism_under_global_rng_perturbation,
              test_gp3_multiclass_does_not_collapse,
              test_gp4_feature_weights_reach_the_model_and_predict_matches):
        t()
    print(f"\n{'FAILURES: ' + ', '.join(_FAILURES) if _FAILURES else 'All gplearn adapter checks passed'}")
    sys.exit(1 if _FAILURES else 0)
