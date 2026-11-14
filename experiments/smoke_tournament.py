"""Smoke test: tournament formula selection + snapshot fields."""
from __future__ import annotations

import numpy as np

from figpso_uil.adapters.symbolic_formula_adapter import SymbolicFormulaAdapter
from figpso_uil.webui.backend import datasets


def main() -> None:
    X, y = datasets.load_dataset("real_breast_cancer", seed=42)
    split = int(0.7 * len(y))
    Xtr, ytr = X[:split], y[:split]
    Xte, yte = X[split:], y[split:]
    fw = np.ones(X.shape[1])

    for mode in ("single", "boosting", "auto"):
        clf = SymbolicFormulaAdapter(seed=42, num_classes=2, boosting_rounds=3)
        clf.apply_config({"learning_mode": mode})
        model = clf.fit_weighted(Xtr, ytr, fw)
        acc = float(np.mean(clf.predict(model, Xte) == yte))
        print(
            f"{mode}: acc={acc:.3f} winner={clf._last_used_formula_id} "
            f"used={clf._formulas_used_this_fit} "
            f"tournament={clf._last_tournament[:3]}"
        )


if __name__ == "__main__":
    main()
