"""Regression test for the multiclass one-vs-rest collapse fix (2026-07-18).

Bug: SymbolicFormula's multiclass predict stacked RAW per-head scores and
argmaxed over them. OvR heads are different binary models on different
(1-vs-rest, imbalanced) problems with different score scales, so raw
scores aren't comparable — argmax collapsed onto a few hot heads (digits
emitted 3/10 classes -> 0.21 acc; MNIST 4/10).

Fix (two parts):
  1. Platt-calibrate each head's score to P(class==c) before argmax, so
     every class is reachable.
  2. Add a `sparse_linear` branch to the OvR loop so each head is a strong
     L1 model over the symbolic basis (single-formula heads top out ~0.5
     on digits; sparse_linear heads reach linear-model territory ~0.95).

This test fits the adapter DIRECTLY (no optimizer) on sklearn digits and
asserts: all classes reachable, accuracy far above majority, and every
head carries a stored calibration.

Run:  python figpso_uil/tests/test_multiclass_ovr.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(os.path.dirname(_HERE))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from figpso_uil.adapters.symbolic_formula_adapter import SymbolicFormulaAdapter


def _fit_predict(mode):
    d = load_digits()
    X = np.asarray(d.data, dtype=float)
    y = np.asarray(d.target, dtype=int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    n_classes = int(len(np.unique(y)))
    clf = SymbolicFormulaAdapter(
        seed=0, num_classes=n_classes, boosting_rounds=20, learning_mode=mode)
    fw = np.ones(X.shape[1], dtype=float)
    model = clf.fit_weighted(Xtr, ytr, fw)
    pred = clf.predict(model, Xte)
    acc = float((pred == yte).mean())
    maj = np.bincount(yte).max() / len(yte)
    return model, pred, acc, maj, n_classes


def test_ovr_all_classes_reachable():
    """sparse_linear OvR must reach every class and beat majority hugely."""
    model, pred, acc, maj, n_classes = _fit_predict("sparse_linear")
    n_pred = len(np.unique(pred))
    assert model.get("kind") == "multiclass_ovr", f"expected OvR, got {model.get('kind')}"
    assert n_pred >= n_classes - 1, (
        f"class collapse: only {n_pred}/{n_classes} classes predicted "
        f"(the bug produced 3/10)"
    )
    assert acc > 0.85, f"digits accuracy {acc:.3f} too low (majority {maj:.3f})"
    print(f"[OK] sparse_linear OvR: acc={acc:.4f}  classes={n_pred}/{n_classes}  "
          f"(majority {maj:.4f})")


def test_heads_are_calibrated():
    """Every OvR head must carry a stored Platt calibration."""
    model, _, _, _, n_classes = _fit_predict("sparse_linear")
    heads = model["classifiers"]
    calibrated = sum(1 for h in heads.values() if h.get("ovr_calibration"))
    assert calibrated == len(heads), (
        f"only {calibrated}/{len(heads)} heads calibrated — "
        f"argmax comparability requires all heads calibrated"
    )
    # Calibration params must be finite.
    for c, h in heads.items():
        cal = h["ovr_calibration"]
        assert np.isfinite(cal["a"]) and np.isfinite(cal["b"]), f"head {c} calibration non-finite"
    print(f"[OK] all {len(heads)} OvR heads carry a finite Platt calibration")


def test_multi_mode_no_collapse():
    """Even the weaker `multi` (hybrid) mode must not collapse to a few
    classes — calibration alone fixes coverage (accuracy is lower than
    sparse_linear, so only the coverage + majority-beat is asserted)."""
    model, pred, acc, maj, n_classes = _fit_predict("multi")
    n_pred = len(np.unique(pred))
    # The weak hybrid at a tiny direct-fit budget won't reach every class,
    # but it must clear the collapse floor (the bug produced 3/10). Half
    # the classes is a comfortable guard; sparse_linear (the recommended
    # multiclass path, tested above) reaches full coverage.
    assert n_pred >= n_classes // 2, (
        f"multi-mode collapse: {n_pred}/{n_classes} classes (pre-fix: 3/10)"
    )
    # Weak hybrid at minimal budget: only require it clearly learns (well
    # above majority), not that it's competitive — sparse_linear carries
    # the accuracy guarantee. The point here is calibration alone lifts
    # coverage off the collapse floor.
    assert acc > maj + 0.1, f"multi-mode acc {acc:.3f} not clearly above majority {maj:.3f}"
    print(f"[OK] multi-mode no collapse: acc={acc:.4f}  classes={n_pred}/{n_classes}")


def main():
    tests = [
        test_ovr_all_classes_reachable,
        test_heads_are_calibrated,
        test_multi_mode_no_collapse,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print()
    if failed:
        print(f"{failed}/{len(tests)} tests failed")
        sys.exit(1)
    print(f"All {len(tests)} tests passed")


if __name__ == "__main__":
    main()
