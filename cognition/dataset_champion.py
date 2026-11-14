"""Per-dataset champion checkpoints — resume accuracy across runs."""

from __future__ import annotations

from typing import Any

import numpy as np

from figpso_uil.cognition.knowledge_archive import KnowledgeArchive
from figpso_uil.core.utils import coerce_float_array, sanitize_for_json


class DatasetChampionStore:
    """Save/load the best model + weights + config per dataset_name."""

    def __init__(self, archive: KnowledgeArchive | None = None):
        self._archive = archive or KnowledgeArchive.shared()

    def save(
        self,
        dataset_name: str,
        *,
        best_accuracy: float,
        best_weights: list[float] | None,
        best_config: dict | None,
        model: dict | Any | None,
        formula_id: str | None,
        learning_mode: str | None = None,
    ) -> None:
        payload = {
            "best_accuracy": float(best_accuracy),
            "best_weights": (
                sanitize_for_json(best_weights) if best_weights is not None else []
            ),
            "best_config": best_config or {},
            "formula_id": formula_id,
            "learning_mode": learning_mode,
        }
        if model is not None and isinstance(model, dict):
            payload["model"] = sanitize_for_json(model)
        self._archive.save_champion(dataset_name, payload)

    def load(self, dataset_name: str) -> dict | None:
        return self._archive.load_champion(dataset_name)
