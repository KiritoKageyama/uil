"""Quick smoke test for SymbolicFormula learning modes."""
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

    for mode in ("single", "boosting", "bagging", "auto"):
        clf = SymbolicFormulaAdapter(seed=42, num_classes=2, boosting_rounds=5)
        clf.apply_config({"learning_mode": mode})
        model = clf.fit_weighted(Xtr, ytr, fw)
        acc = float(np.mean(clf.predict(model, Xte) == yte))
        print(f"{mode}: acc={acc:.3f} resolved={model.get('learning_mode')}")


if __name__ == "__main__":
    main()
