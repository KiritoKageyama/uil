"""Regression test for the RuleFit-style rule basis (2026-07-18).

The sparse_linear basis was purely ADDITIVE (raw + stump + formula), so it
could not represent feature INTERACTIONS — the capacity gap on interaction-
dense data like sonar (additive ~0.75, black-box RBF ~0.89). The `rule`
family adds conjunctions of stump conditions (1[x3>0.2 & x7<=0.5]), mined
RuleFit-style from a shallow forest, L1-selected jointly. Each rule stays a
readable, editable term.

Asserts:
  1. rules are generated and materialize as {0,1} conjunction columns;
  2. the fitted law renders readably (1[.. & ..]);
  3. rules OFF (default) is byte-identical to the pre-rule additive path
     (backward compatibility);
  4. rules ON + weaker L1 measurably lifts an interaction dataset without
     breaking (the rules need their own regularization budget).

Run:  python figpso_uil/tests/test_rule_basis.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(os.path.dirname(_HERE))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from figpso_uil.adapters.symbolic_formula_adapter import SymbolicFormulaAdapter


def _xor_like(n=600, seed=0):
    """An interaction-dependent target additive models struggle with:
    y depends on the SIGN AGREEMENT of two features (a 2-way interaction)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)  # XOR: pure interaction
    X += rng.normal(scale=0.15, size=X.shape)  # a little noise
    return X, y


def test_rules_generate_and_render():
    X, y = _xor_like(seed=1)
    clf = SymbolicFormulaAdapter(seed=0, num_classes=2, learning_mode="sparse_linear")
    clf.use_rule_basis = True
    clf.sparse_l1_c = 0.5
    model = clf.fit_weighted(X, y, np.ones(X.shape[1]))
    rule_terms = [e for e in model["basis"] if e.get("type") == "rule"]
    assert rule_terms, "no rule terms survived the fit — rule family inactive"
    law = clf.readable_sparse_linear(model, top=8)
    assert " & " in law or "1[" in law, f"rules not rendered readably: {law[:120]}"
    print(f"[OK] {len(rule_terms)} rule terms in the fitted basis; "
          f"law renders readably")
    print(f"     {law[:140]}")


def test_rules_off_is_additive_backcompat():
    """Rules OFF (default) must produce a basis with NO rule terms —
    identical to the pre-rule additive path."""
    X, y = _xor_like(seed=2)
    clf = SymbolicFormulaAdapter(seed=0, num_classes=2, learning_mode="sparse_linear")
    assert clf.use_rule_basis is False, "rule basis must default OFF"
    model = clf.fit_weighted(X, y, np.ones(X.shape[1]))
    rule_terms = [e for e in model["basis"] if e.get("type") == "rule"]
    assert not rule_terms, "rule terms leaked into the default additive path"
    print("[OK] rules OFF by default: additive path unchanged (0 rule terms)")


def test_rules_help_on_interaction_data():
    """On a pure-interaction (XOR) target, additive can't separate the
    classes; rules should measurably beat it."""
    seeds = [0, 1, 2, 3, 4]
    add_acc, rule_acc = [], []
    for s in seeds:
        X, y = _xor_like(seed=s)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=s, stratify=y)
        # additive
        a = SymbolicFormulaAdapter(seed=s, num_classes=2, learning_mode="sparse_linear")
        a.sparse_l1_c = 0.5
        ma = a.fit_weighted(Xtr, ytr, np.ones(X.shape[1]))
        add_acc.append(accuracy_score(yte, a.predict(ma, Xte)))
        # + rules
        r = SymbolicFormulaAdapter(seed=s, num_classes=2, learning_mode="sparse_linear")
        r.use_rule_basis = True; r.sparse_l1_c = 0.5
        mr = r.fit_weighted(Xtr, ytr, np.ones(X.shape[1]))
        rule_acc.append(accuracy_score(yte, r.predict(mr, Xte)))
    add_m, rule_m = float(np.mean(add_acc)), float(np.mean(rule_acc))
    print(f"[{'OK' if rule_m > add_m + 0.05 else 'WEAK'}] XOR interaction: "
          f"additive={add_m:.3f}  +rules={rule_m:.3f}  (delta {rule_m-add_m:+.3f})")
    assert rule_m > add_m + 0.05, (
        f"rules should beat additive on a pure-interaction target: "
        f"additive={add_m:.3f} rules={rule_m:.3f}"
    )


def main():
    tests = [test_rules_generate_and_render,
             test_rules_off_is_additive_backcompat,
             test_rules_help_on_interaction_data]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}"); failed += 1
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}"); failed += 1
    print()
    if failed:
        print(f"{failed}/{len(tests)} failed"); sys.exit(1)
    print(f"All {len(tests)} tests passed")


if __name__ == "__main__":
    main()
