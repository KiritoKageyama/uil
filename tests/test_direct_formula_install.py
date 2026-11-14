"""Direct formula influence: residual grow + forced install (steps 1–3).

Step 1 — formulas fit on this dataset's residual/target
Step 2 — forced into the model (dedicated symbolic stage, not shared L1)
Step 3 — existence CONNECTED (predictions move) + Gate 2 harness

Run:
    python figpso_uil/tests/test_direct_formula_install.py
    python -m figpso_uil.experiments.gate2_direct_install --quick
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from figpso_uil.adapters.symbolic_formula_adapter import SymbolicFormulaAdapter
from figpso_uil.cognition.residual_formula_install import grow_residual_terms

_FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        _FAILURES.append(name)


def _xor(n=400, d=6, seed=0):
    """Additive families are near chance; residual formulas must help."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X, y


def test_d1_residual_grows_on_target():
    X, y = _xor()
    terms = grow_residual_terms(X, y, seed=42, n_terms=3)
    check("D1 residual grow nonempty", len(terms) >= 1, f"n={len(terms)}")


def test_d2_forced_install_puts_formula_in_model():
    X, y = _xor()
    split = int(0.7 * len(y))
    state = tempfile.mkdtemp(prefix="uil_di_")
    clf = SymbolicFormulaAdapter(
        seed=42, num_classes=2,
        formula_library_path=os.path.join(state, "lib.json"),
    )
    clf.apply_config({
        "learning_mode": "sparse_linear",
        "direct_formula_install": True,
        "use_stump_basis": False,
        "use_rule_basis": False,
        "use_neural_basis": False,
    })
    model = clf.fit_weighted(X[:split], y[:split], np.ones(X.shape[1]))
    formula_terms = [
        e for e in (model.get("basis") or [])
        if e.get("type") == "formula"
    ]
    check(
        "D2 model carries formula terms",
        bool(model.get("direct_install")) and len(formula_terms) >= 1,
        f"direct={model.get('direct_install')} n_formula={len(formula_terms)} "
        f"used={clf._formulas_used_this_fit}",
    )


def test_d3_existence_connected():
    """Ablating formulas under direct_install must change predictions."""
    X, y = _xor(n=500, seed=7)
    split = int(0.75 * len(y))
    Xtr, ytr, Xte = X[:split], y[:split], X[split:]

    def _fit(install: bool, use_sym: bool):
        state = tempfile.mkdtemp(prefix="uil_di_")
        clf = SymbolicFormulaAdapter(
            seed=7, num_classes=2,
            formula_library_path=os.path.join(state, "lib.json"),
        )
        clf.apply_config({
            "learning_mode": "sparse_linear",
            "direct_formula_install": install,
            "use_symbolic_basis": use_sym,
            "use_stump_basis": True,
            "use_rule_basis": False,
            "use_neural_basis": False,
        })
        model = clf.fit_weighted(Xtr, ytr, np.ones(X.shape[1]))
        preds = clf.predict(model, Xte)
        return preds, model

    preds_on, model_on = _fit(True, True)
    preds_off, model_off = _fit(False, True)
    # Direct install ON vs classic (utility-off) path
    moved_vs_classic = not np.array_equal(preds_on, preds_off)
    # Direct install with symbolic OFF should differ from ON (formulas gated)
    preds_nosym, _ = _fit(True, False)
    moved_vs_nosym = not np.array_equal(preds_on, preds_nosym)
    n_form = sum(1 for e in (model_on.get("basis") or []) if e.get("type") == "formula")
    check(
        "D3 existence CONNECTED",
        moved_vs_classic or moved_vs_nosym,
        f"vs_classic={moved_vs_classic} vs_nosym={moved_vs_nosym} "
        f"n_form={n_form} off_direct={model_off.get('direct_install')}",
    )


def test_d4_deterministic():
    X, y = _xor(seed=3)
    split = int(0.7 * len(y))

    def _one():
        state = tempfile.mkdtemp(prefix="uil_di_")
        clf = SymbolicFormulaAdapter(
            seed=3, num_classes=2,
            formula_library_path=os.path.join(state, "lib.json"),
        )
        clf.apply_config({
            "learning_mode": "sparse_linear",
            "direct_formula_install": True,
        })
        model = clf.fit_weighted(X[:split], y[:split], np.ones(X.shape[1]))
        return clf.predict(model, X[split:])

    a, b = _one(), _one()
    check("D4 same seed identical preds", np.array_equal(a, b))


if __name__ == "__main__":
    print("Direct formula install tests")
    test_d1_residual_grows_on_target()
    test_d2_forced_install_puts_formula_in_model()
    test_d3_existence_connected()
    test_d4_deterministic()
    if _FAILURES:
        print(f"\n{len(_FAILURES)} FAILED: {_FAILURES}")
        sys.exit(1)
    print("\nAll D1-D4 passed.")
