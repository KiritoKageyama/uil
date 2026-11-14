"""Fidelity of the readable model output — does the rendered law IS the model?

The project's surviving value claim is inspectable, human-editable knowledge.
Until now that claim had ZERO tests: `readable_sparse_linear` produced a
string and nothing checked the string described the function `predict()`
computes.

This file settles it in both directions.

  F1-F5  the EXPORTED law (cognition/law_export.py) reproduces predict()
         exactly, across every basis family and for multiclass OvR.
  F6     the OLD renderer does NOT — read literally it computes a different
         function. Asserted as a known defect so it cannot regress silently
         if someone "simplifies" the exporter back into it.
  F7     ACTIONABILITY: editing a coefficient in the law moves predictions by
         exactly the amount the law says it should. Readability without this
         is decoration.
  F8     the law survives a JSON round-trip (it is an editable artifact, so
         it has to survive being written to disk and read back).

Run:  python figpso_uil/tests/test_law_fidelity.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(os.path.dirname(_HERE))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from figpso_uil.adapters.symbolic_formula_adapter import SymbolicFormulaAdapter
from figpso_uil.cognition.law_export import (
    LawExportError,
    evaluate_law,
    export_law,
    law_is_legible,
    law_is_self_contained,
    render_law,
)

_TMP = os.path.join(_HERE, "_law_fidelity_lib.json")


def _data(n=400, d=8, seed=0, classes=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    if classes == 2:
        z = 1.4 * X[:, 0] - 0.9 * X[:, 3] + 0.6 * (X[:, 1] > 0) * X[:, 2]
        y = (z + rng.normal(scale=0.4, size=n) > 0).astype(int)
    else:
        z = np.column_stack([X[:, 0], X[:, 1] - X[:, 2], -X[:, 3]])
        y = np.argmax(z + rng.normal(scale=0.3, size=z.shape), axis=1)
    return X, y


def _adapter(num_classes=2, **gates):
    clf = SymbolicFormulaAdapter(
        seed=0,
        num_classes=num_classes,
        learning_mode="sparse_linear",
        formula_library_path=_TMP,
    )
    # Symbolic (formula-tree) family off unless a test asks for it, so the
    # law under test is genuinely self-contained.
    clf.use_symbolic_basis = False
    for k, v in gates.items():
        setattr(clf, k, v)
    return clf


def _check_roundtrip(clf, model, X, label, tree_eval=None):
    """The one assertion that matters: law predictions == model predictions."""
    law = export_law(model)
    got = evaluate_law(law, X, tree_eval=tree_eval)
    want = clf.predict(model, X)
    n_diff = int(np.sum(got != want))
    assert n_diff == 0, (
        f"{label}: law disagrees with predict() on {n_diff}/{len(want)} rows"
    )
    n_terms = sum(len(h["terms"]) for h in law["heads"])
    print(f"[OK] {label}: {n_terms} terms, {len(want)} rows, 0 disagreements")
    return law


# --------------------------------------------------------------------------
# F1-F5 — the exported law is the model
# --------------------------------------------------------------------------

def test_f1_raw_and_stump_basis():
    X, y = _data(seed=1)
    clf = _adapter(use_raw_basis=True, use_stump_basis=True)
    model = clf.fit_weighted(X, y, np.ones(X.shape[1]))
    law = _check_roundtrip(clf, model, X, "F1 raw+stump")
    assert law_is_self_contained(law) and law_is_legible(law), (
        "raw+stump law must be both self-contained and legible"
    )


def test_f2_rule_and_kan_basis():
    X, y = _data(seed=2)
    clf = _adapter(use_rule_basis=True, use_kan_basis=True, sparse_l1_c=0.5)
    model = clf.fit_weighted(X, y, np.ones(X.shape[1]))
    kinds = {e.get("type") for e in model["basis"]}
    law = _check_roundtrip(clf, model, X, f"F2 rule+kan (families={sorted(kinds)})")
    assert law_is_legible(law), "rule/kan terms must stay legible"


def test_f3_neural_basis_is_exact_but_not_legible():
    X, y = _data(seed=3)
    clf = _adapter(use_neural_basis=True, sparse_l1_c=0.5)
    model = clf.fit_weighted(X, y, np.ones(X.shape[1]))
    law = _check_roundtrip(clf, model, X, "F3 +neural")
    if any(t["type"] == "neural" for h in law["heads"] for t in h["terms"]):
        assert law_is_self_contained(law), "neural units are exactly specifiable"
        assert not law_is_legible(law), (
            "a law containing neural units must NOT claim full legibility"
        )
        print("     neural terms present: exact, flagged not-legible (honest)")


def test_f4_formula_terms_are_not_self_contained():
    X, y = _data(seed=4)
    clf = _adapter(use_symbolic_basis=True)
    model = clf.fit_weighted(X, y, np.ones(X.shape[1]))
    law = export_law(model)
    has_formula = any(
        t["type"] == "formula" for h in law["heads"] for t in h["terms"]
    )
    if not has_formula:
        print("[SKIP] F4: no formula term survived L1 on this data")
        return
    assert not law_is_self_contained(law), (
        "a law leaning on evolved formula trees is NOT self-contained"
    )
    # Refusing to guess is part of fidelity: without the engine it must raise.
    try:
        evaluate_law(law, X)
        raise AssertionError("evaluate_law silently guessed a formula term")
    except LawExportError:
        pass

    fw = model.get("feature_weights")
    tree_eval = lambda tree, Xr: clf._raw_from_tree(tree, Xr, fw)  # noqa: E731
    _check_roundtrip(clf, model, X, "F4 +formula (with engine)", tree_eval=tree_eval)


def test_f5_multiclass_ovr_law():
    X, y = _data(seed=5, classes=3)
    clf = _adapter(num_classes=3, use_raw_basis=True, use_stump_basis=True)
    model = clf.fit_weighted(X, y, np.ones(X.shape[1]))
    if model.get("kind") != "multiclass_ovr":
        print(f"[SKIP] F5: fit produced kind={model.get('kind')!r}")
        return
    law = _check_roundtrip(clf, model, X, "F5 multiclass OvR")
    assert len(law["heads"]) == len(np.unique(y)), "a head per class"
    got = evaluate_law(law, X)
    assert len(np.unique(got)) > 1, "OvR law collapsed to one class (bug #24)"


# --------------------------------------------------------------------------
# F6 — the OLD renderer is unfaithful (falsification, pinned)
# --------------------------------------------------------------------------

def test_f6_shipped_renderer_is_faithful_and_the_old_one_was_not():
    """Both halves, pinned together.

    OLD: `readable_sparse_linear` printed STANDARDIZED coefficients against
    RAW-space term names, omitted the fitted threshold, and truncated at
    `top`. Read literally it computed a different function.

    NEW: the shipped renderer round-trips through the exported law.
    """
    X, y = _data(n=500, d=10, seed=6)
    clf = _adapter(use_raw_basis=True, use_stump_basis=True)
    model = clf.fit_weighted(X, y, np.ones(X.shape[1]))
    truth = clf.predict(model, X)

    # --- the defect the fix removed -------------------------------------
    defects = []
    if not np.allclose(model["mu"], 0.0) or not np.allclose(model["sd"], 1.0):
        defects.append("omitted mu/sd standardization")
    if abs(float(model["threshold"]) - 0.5) > 1e-9:
        defects.append(f"omitted fitted threshold ({model['threshold']:.3g} != 0.5)")
    assert defects, "model has trivial mu/sd/threshold — test proves nothing here"

    naive = np.full(X.shape[0], float(model["intercept"]))
    for e, c in zip(model["basis"], model["coef"]):
        if e["type"] == "raw":
            naive = naive + c * X[:, e["feature"]]
        elif e["type"] == "stump":
            naive = naive + c * (X[:, e["feature"]] > e["threshold"]).astype(float)
    naive_pred = (1.0 / (1.0 + np.exp(-np.clip(naive, -30, 30))) >= 0.5).astype(int)
    n_diff = int(np.sum(naive_pred != truth))
    assert n_diff > 0, "old arithmetic no longer differs — re-derive this test"
    print(f"[OK] F6 old renderer {', '.join(defects)}; read literally it "
          f"mislabelled {n_diff}/{len(truth)} rows ({100.0*n_diff/len(truth):.1f}%)")

    legacy = clf._readable_sparse_linear_legacy(model, top=12)
    assert f"{model['threshold']:.3g}" not in legacy, "legacy string leaked threshold"

    # --- what ships now --------------------------------------------------
    text = clf.readable_sparse_linear(model, top=None)
    assert "sigmoid" in text and "TRUNCATED" not in text
    assert f"{float(model['threshold']):.6g}" in text, (
        "shipped renderer must print the fitted decision threshold"
    )
    assert np.array_equal(evaluate_law(export_law(model), X), truth)
    print("     shipped renderer now carries the threshold and round-trips exactly")

    # truncation must announce itself
    short = clf.readable_sparse_linear(model, top=1)
    if len(model["coef"]) > 1:
        assert "TRUNCATED" in short, "silent truncation is back"
        print("     truncation is labelled, not silent")


# --------------------------------------------------------------------------
# F7 — actionability: an edit does what the law says it will
# --------------------------------------------------------------------------

def test_f7_editing_a_coefficient_is_predictable():
    X, y = _data(seed=7)
    clf = _adapter(use_raw_basis=True, use_stump_basis=True)
    model = clf.fit_weighted(X, y, np.ones(X.shape[1]))
    law = export_law(model)
    head = law["heads"][0]

    idx = int(np.argmax([abs(t["coef"]) for t in head["terms"]]))
    term = head["terms"][idx]

    # Independent prediction of the effect, computed from the law's own text:
    # deleting term j must shift the logit by exactly -coef_j * value_j(x).
    from figpso_uil.cognition.law_export import _term_values, head_scores

    sel = law.get("selected_features")
    X_red = X[:, np.asarray(sel, dtype=int)] if sel is not None else X
    vals = _term_values(term, X, X_red, None)
    before = head_scores(head, X, X_red, None)
    logit_before = np.log(np.clip(before, 1e-12, 1 - 1e-12) /
                          np.clip(1 - before, 1e-12, 1))
    expected_logit = logit_before - term["coef"] * vals

    saved = term["coef"]
    term["coef"] = 0.0
    after = head_scores(head, X, X_red, None)
    logit_after = np.log(np.clip(after, 1e-12, 1 - 1e-12) /
                         np.clip(1 - after, 1e-12, 1))

    # Only compare where the sigmoid has not saturated (clip destroys the
    # logit identity at the tails — that is arithmetic, not infidelity).
    live = (before > 1e-6) & (before < 1 - 1e-6) & (after > 1e-6) & (after < 1 - 1e-6)
    err = float(np.max(np.abs(logit_after[live] - expected_logit[live])))
    assert err < 1e-6, f"edit effect off by {err:.2e} — law is not actionable"

    changed = int(np.sum((after >= head["threshold"]) != (before >= head["threshold"])))
    term["coef"] = saved
    print(f"[OK] F7 edit is predictable: max logit error {err:.2e}; "
          f"deleting '{term['expr']}' flips {changed}/{len(X)} predictions")
    assert changed > 0, (
        "the single largest-coefficient term changed NO predictions — the law "
        "reports importance it does not have"
    )


# --------------------------------------------------------------------------
# F8 — the law is a durable artifact
# --------------------------------------------------------------------------

def test_f8_law_survives_json_roundtrip():
    X, y = _data(seed=8)
    clf = _adapter(use_raw_basis=True, use_stump_basis=True, use_kan_basis=True)
    model = clf.fit_weighted(X, y, np.ones(X.shape[1]))
    law = export_law(model)
    reloaded = json.loads(json.dumps(law))
    assert np.array_equal(evaluate_law(reloaded, X), clf.predict(model, X)), (
        "law changed meaning through JSON (bug #28 family: dtype/key loss)"
    )
    text = render_law(reloaded)
    assert "TRUNCATED" not in text, "full render must not truncate"
    print("[OK] F8 law survives JSON round-trip; full render is untruncated")
    print("     " + text.splitlines()[-1][:120])


def _cleanup():
    for p in (_TMP, _TMP + ".bak"):
        try:
            os.remove(p)
        except OSError:
            pass


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001 — report, don't abort the suite
            failed += 1
            print(f"[FAIL] {t.__name__}: {type(exc).__name__}: {exc}")
    _cleanup()
    print(f"\n{len(tests) - failed}/{len(tests)} law-fidelity tests passed")
    sys.exit(1 if failed else 0)
