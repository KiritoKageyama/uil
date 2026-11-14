"""Train the offline own-transformer proposer (R1 §5.4) on the system's history.

Reads the formula libraries (classifier + loss), builds (tree, quality) pairs,
trains the TinyTransformer, and writes the open-box model card to
results/uil_neural_proposer.json. Re-run this whenever the libraries have
grown enough to be worth re-distilling.

    python -m figpso_uil.experiments.train_neural_proposer
"""

from __future__ import annotations

import json
import os

from figpso_uil.cognition.neural_proposer import NeuralProposer
from figpso_uil.cognition.paths import results_path


def _load(name: str) -> list:
    path = results_path(name)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def main() -> None:
    formulas = _load("uil_symbolic_classifier_library.json")
    formulas += _load("uil_formula_library.json")
    print(f"training corpus: {len(formulas)} formula cells")

    proposer = NeuralProposer.train_from_libraries(formulas, seed=42)
    if proposer is None:
        print("Not enough history yet (need >=20 formula cells with trees). "
              "The mutator stays on the engine fallback until then.")
        return

    stats = proposer.model.training_stats
    print(f"trained: train_acc={stats['train_acc']} holdout_acc={stats['holdout_acc']} "
          f"(n_train={stats['n_train']}, n_holdout={stats['n_holdout']})")
    print("distilled-out token preferences (top 10):")
    for tok, delta in proposer.model.explain()[:10]:
        print(f"  {delta:+0.4f}  {tok}")
    print(f"model card: {results_path('uil_neural_proposer.json')}")


if __name__ == "__main__":
    main()
