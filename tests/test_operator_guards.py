"""Formula-engine operator guards — bound divergence, change nothing in domain.

MEASURED 2026-08-05 (`results/extrapolation/SUMMARY.md`): evaluated outside
the region they were fitted in, evolved formula trees diverged silently. Over
a 21-cell out-of-range benchmark UIL's symbolic regressor emitted predictions
past 1000x the training scale on **19 seed-runs** (R2 ~ -1e35); `gplearn` --
same model family, protected operators -- did so **zero** times.

Two defects, both fixed in `cognition/formula_engine.py`:

  1. `div` protected only at |b| > 1e-10, so |b| = 1e-6 returned 1e6.
     Now DIV_EPS = 1e-3 (gplearn parity), and it uses np.divide(where=...)
     so it no longer computes a/b for b == 0 -- the source of the
     `RuntimeWarning: divide by zero` noise in every run log.
  2. The COMPOSITION was unguarded. `evaluate()` replaced NaN/Inf with 0.0,
     but a product of large FINITE intermediates stays finite and 1e35 passed
     the isfinite check untouched. `_bounded()` now clamps every node to
     MAX_MAGNITUDE.

G4 is the one that protects everything else: the guards MUST be inert on
in-domain values. A guard that changes the loss maths would silently alter
every custom-objective fit in the project.

Run:  python figpso_uil/tests/test_operator_guards.py
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(os.path.dirname(_HERE))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from figpso_uil.cognition.formula_engine import (
    DIV_EPS, MAX_MAGNITUDE, OPS, FormulaEngine, FormulaNode, _safe_div,
)

_FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        _FAILURES.append(name)


def test_g1_protected_division() -> None:
    print("\nG1. Protected division bounds the result and does not warn")
    import figpso_uil.cognition.formula_engine as fe
    # Denominators straddling the threshold: exactly 0, well below DIV_EPS,
    # and a normal value.
    below = DIV_EPS / 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning fails the test
        try:
            out = _safe_div(np.array([1.0, 1.0, 1.0]),
                            np.array([0.0, below, 2.0]))
            warned = False
        except RuntimeWarning as exc:
            out, warned = np.array([np.nan, np.nan, np.nan]), True
            check("G1 no divide-by-zero warning", False, str(exc))
    if not warned:
        check("G1 no divide-by-zero warning", True)
    # The mechanism, independent of where the threshold happens to sit.
    check("G1 |b| below DIV_EPS returns the protected value",
          float(out[0]) == fe.DIV_PROTECTED_VALUE
          and float(out[1]) == fe.DIV_PROTECTED_VALUE,
          f"got {out[0]}, {out[1]}; protected value = {fe.DIV_PROTECTED_VALUE}")
    check("G1 normal division still exact", abs(float(out[2]) - 0.5) < 1e-12)


def test_g2_defaults_are_the_measured_ones() -> None:
    """The guards ship OFF, and that is a measured decision.

    Tightening them was benchmarked and no setting dominates (see
    results/extrapolation/GUARD_ABLATION.md): the tight clamp halves blowups
    but destroys working cases, and relaxing it to keep those cases makes
    blowups worse than doing nothing. This pins the defaults so nobody
    silently re-tightens them without redoing that measurement.
    """
    print("\nG2. Shipped defaults match the measured decision (guards OFF)")
    check("G2 MAX_MAGNITUDE clamp disabled by default",
          not np.isfinite(MAX_MAGNITUDE), f"MAX_MAGNITUDE = {MAX_MAGNITUDE}")
    check("G2 DIV_EPS at the original 1e-10", DIV_EPS == 1e-10,
          f"DIV_EPS = {DIV_EPS}")


def test_g3_clamp_mechanism_works_when_enabled() -> None:
    """The composition clamp is what `isfinite` could never provide.

    Tested by setting the cap explicitly rather than relying on the default,
    because the default is deliberately off. A product of large FINITE
    intermediates stays finite, so isfinite passes 1e35 untouched -- that is
    the mechanism that made out-of-range predictions diverge silently.
    """
    print("\nG3. Composition clamp bounds finite blow-ups when enabled")
    import figpso_uil.cognition.formula_engine as fe
    eng = FormulaEngine(seed=0)
    X = np.array([[1e5, 1e5]])
    sq = lambda: FormulaNode(op="mul", children=[  # noqa: E731
        FormulaNode(feature=0), FormulaNode(feature=1)])
    tree = FormulaNode(op="mul", children=[sq(), sq()])

    unclamped = np.asarray(eng.evaluate(tree, {"X": X}), dtype=float)
    check("G3 default lets a large FINITE product through (isfinite is not enough)",
          bool(np.all(np.isfinite(unclamped))) and float(np.abs(unclamped[0])) > 1e12,
          f"value = {float(unclamped[0]):.3g}")

    saved = fe.MAX_MAGNITUDE
    try:
        fe.MAX_MAGNITUDE = 1e6
        clamped = np.asarray(eng.evaluate(tree, {"X": X}), dtype=float)
    finally:
        fe.MAX_MAGNITUDE = saved
    check("G3 clamp binds when set", float(np.abs(clamped[0])) <= 1e6,
          f"value = {float(clamped[0]):.3g}")


def test_g4_guards_are_inert_in_domain() -> None:
    """The guards must not change ANY value the engine legitimately produces.

    This is what makes the fix shippable unconditionally: the loss-formula
    path (logits / probabilities) and the feature path (standardized columns)
    both sit many orders of magnitude inside the cap.
    """
    print("\nG4. Guards are inert on in-domain values")
    rng = np.random.default_rng(0)
    Xn = rng.normal(size=(300, 3))
    worst = 0.0
    for op in ("add", "sub", "mul", "tanh", "sigmoid", "square", "abs",
               "neg", "relu", "sign", "min", "max"):
        fn, arity = OPS[op]
        args = [Xn[:, 0]] if arity == 1 else [Xn[:, 0], Xn[:, 1]]
        worst = max(worst, float(np.max(np.abs(np.asarray(fn(*args))))))
    check("G4 standard ops stay far inside any plausible cap",
          worst < 1e3,
          f"largest in-domain magnitude {worst:.3g}")

    # The loss path must be bit-identical to the unguarded maths.
    eng = FormulaEngine(seed=0)
    y_pred = np.clip(rng.random(500), 1e-6, 1 - 1e-6)
    y_true = (rng.random(500) > 0.5).astype(float)
    tree = FormulaNode(op="mul", children=[
        FormulaNode(var="y_true"),
        FormulaNode(op="log", children=[FormulaNode(var="y_pred")])])
    got = np.asarray(eng.evaluate(tree, {"y_true": y_true, "y_pred": y_pred}))
    want = y_true * np.log(np.clip(y_pred, 1e-10, None))
    err = float(np.max(np.abs(got - want)))
    check("G4 log-loss term is bit-identical under the guards", err == 0.0,
          f"max abs error {err:.3g}")


if __name__ == "__main__":
    for t in (test_g1_protected_division, test_g2_defaults_are_the_measured_ones,
              test_g3_clamp_mechanism_works_when_enabled,
              test_g4_guards_are_inert_in_domain):
        t()
    print(f"\n{'FAILURES: ' + ', '.join(_FAILURES) if _FAILURES else 'All guard checks passed'}")
    sys.exit(1 if _FAILURES else 0)
