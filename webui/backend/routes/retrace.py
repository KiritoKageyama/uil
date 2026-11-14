"""Retrace: pure read of the journal for stage-by-stage playback of a
completed run. No new persistence -- UIL_Journal.get_dataset_entries()
already exists and already returns exactly the ordered per-stage
state/action/outcome/regret/stress/investigation records needed.
"""

from __future__ import annotations
from fastapi import APIRouter

from figpso_uil.cognition.uil_journal import UIL_Journal

router = APIRouter(prefix="/retrace", tags=["retrace"])


@router.get("/{dataset_name}")
def retrace(dataset_name: str, fold: int | None = None):
    journal = UIL_Journal()
    entries = journal.get_dataset_entries(dataset_name)
    if fold is not None:
        entries = [e for e in entries if e.get("fold") == fold]
    entries.sort(key=lambda e: (e.get("fold", 0), e.get("stage", 0)))
    return {"dataset_name": dataset_name, "fold": fold, "entries": entries}
