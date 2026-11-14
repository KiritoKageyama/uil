"""UIL Experience Journal — Raw numerical recording. Zero English.

The journal is the UIL's brain. It stores every experience as a pure
numerical vector: state, action, outcome, stress, regret.

No hardcoded if-statements. No English lessons. No human intelligence.
Just numbers. The Miner discovers patterns from these numbers.

The journal PERSISTS across runs. It is append-only — the growing brain.
"""

from __future__ import annotations
import os
import json
from datetime import datetime, timezone

from figpso_uil.cognition.paths import results_path


class UIL_Journal:
    """The UIL's persistent memory. Append-only numerical recording.
    
    Each entry is a pure numerical vector:
    {
        "dataset": str,
        "fold": int,
        "stage": int,
        "classifier": str,
        "state": {n_samples, n_features, num_classes, entropy, feat_ratio, ...},
        "action": {penalty, ...},
        "outcome": {accuracy, cpu_time, model_complexity, ...},
        "decision_stress": float,
        "system_stress": float,
        "regret": float,
        "investigation": {...} or null,
        "timestamp": str
    }
    
    No English. No lessons. No hardcoded rules. Pure data.
    """

    def __init__(self, filepath: str = results_path("uil_journal_v19.json")):
        self.filepath = filepath
        self._ensure_file()
        self._dataset_memory: dict = {}
        self._load_dataset_memory()

    # ── Core Recording ───────────────────────────────────────────────

    def record(
        self,
        dataset: str,
        fold: int,
        stage: int,
        classifier: str,
        state: dict,
        action: dict,
        outcome: dict,
        decision_stress: float,
        system_stress: float,
        regret: float,
        investigation: dict | None = None,
    ) -> None:
        """Record a single experience entry. Pure numbers.
        
        Args:
            dataset: Dataset name.
            fold: Cross-validation fold number.
            stage: Co-evolution stage number.
            classifier: Name of the classifier adapter used.
            state: State vector {n_samples, n_features, num_classes, entropy, ...}.
            action: Action vector {penalty, ...}.
            outcome: Outcome vector {accuracy, cpu_time, model_complexity, ...}.
            decision_stress: Per-action stress value.
            system_stress: Cumulative system stress.
            regret: Backward-looking regret value.
            investigation: Investigation record if regret triggered one.
        """
        entry = {
            "dataset": dataset,
            "fold": fold,
            "stage": stage,
            "classifier": classifier,
            "state": state,
            "action": action,
            "outcome": outcome,
            "decision_stress": float(decision_stress),
            "system_stress": float(system_stress),
            "regret": float(regret),
            "investigation": investigation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        data = self._load()
        data.append(entry)
        self._save(data)

    # ── Query Methods ────────────────────────────────────────────────

    def get_entries(self, dataset: str | None = None) -> list[dict]:
        """Get all journal entries, optionally filtered by dataset."""
        data = self._load()
        if dataset is not None:
            return [e for e in data if e.get("dataset") == dataset]
        return data

    def get_recent(self, n: int = 20) -> list[dict]:
        """Get the N most recent entries."""
        data = self._load()
        return data[-n:]

    def get_dataset_entries(self, dataset: str) -> list[dict]:
        """Get all entries for a specific dataset."""
        return self.get_entries(dataset=dataset)

    def get_best_proven_accuracy(self, dataset: str | None = None) -> float:
        """Get the best accuracy ever achieved (overall or per dataset)."""
        entries = self.get_entries(dataset)
        if not entries:
            return 0.0
        return max(e.get("outcome", {}).get("accuracy", 0.0) for e in entries)

    def get_entry_count(self) -> int:
        """Total number of journal entries."""
        return len(self._load())

    # ── Dataset Memory ───────────────────────────────────────────────

    def get_dataset_prior_stress(self, dataset: str) -> float:
        """Get the initial stress override for a dataset.
        
        If the dataset was previously hard (reached high stress),
        start with elevated stress next time.
        """
        return self._dataset_memory.get(dataset, {}).get(
            "initial_stress_override", 0.0
        )

    def update_dataset_memory(
        self,
        dataset: str,
        max_stress_reached: float,
        best_accuracy: float,
        preferred_classifier: str | None = None,
    ) -> None:
        """Update dataset memory after a run.
        
        If stress was high, remember this dataset is hard.
        Next time, start with elevated stress — skip conservative phase.
        """
        if dataset not in self._dataset_memory:
            self._dataset_memory[dataset] = {}

        mem = self._dataset_memory[dataset]
        mem["last_best_accuracy"] = best_accuracy
        mem["last_max_stress"] = max_stress_reached

        # If stress was high, set initial override for next time
        if max_stress_reached > 0.5:
            mem["initial_stress_override"] = min(0.5, max_stress_reached * 0.6)
        else:
            mem["initial_stress_override"] = 0.0

        if preferred_classifier:
            mem["preferred_classifier"] = preferred_classifier

        self._save_dataset_memory()

    def get_dataset_memory(self, dataset: str) -> dict:
        """Get full dataset memory record."""
        return self._dataset_memory.get(dataset, {})

    # ── State at Best ────────────────────────────────────────────────

    def get_state_at_best(self, dataset: str) -> dict | None:
        """Get the state vector from when best accuracy was achieved.
        
        Used by the Stress Engine for regret investigation:
        comparing current state to state-at-best to find what changed.
        """
        entries = self.get_dataset_entries(dataset)
        if not entries:
            return None

        best_entry = max(
            entries, key=lambda e: e.get("outcome", {}).get("accuracy", 0.0)
        )
        return best_entry.get("state")

    def get_action_at_best(self, dataset: str) -> dict | None:
        """Get the action vector from when best accuracy was achieved."""
        entries = self.get_dataset_entries(dataset)
        if not entries:
            return None

        best_entry = max(
            entries, key=lambda e: e.get("outcome", {}).get("accuracy", 0.0)
        )
        return best_entry.get("action")

    # ── Failsafe Recording ───────────────────────────────────────────

    def record_failsafe(
        self,
        dataset: str,
        diagnosis: dict,
    ) -> None:
        """Record a failsafe diagnosis result."""
        entry = {
            "dataset": dataset,
            "fold": -1,
            "stage": -1,
            "classifier": "FAILSAFE",
            "state": {},
            "action": {},
            "outcome": diagnosis,
            "decision_stress": 0.0,
            "system_stress": 0.0,
            "regret": 0.0,
            "investigation": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        data = self._load()
        data.append(entry)
        self._save(data)

    # ── Internal ─────────────────────────────────────────────────────

    def _ensure_file(self) -> None:
        """Ensure journal file and directory exist."""
        dirpath = os.path.dirname(self.filepath)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w") as f:
                json.dump([], f)

    def _load(self) -> list[dict]:
        """Load journal entries from file."""
        try:
            with open(self.filepath, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save(self, data: list[dict]) -> None:
        """Save journal entries to file."""
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_dataset_memory(self) -> None:
        """Load dataset memory from a companion file."""
        mem_path = self.filepath.replace(".json", "_memory.json")
        try:
            with open(mem_path, "r") as f:
                self._dataset_memory = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            self._dataset_memory = {}

    def _save_dataset_memory(self) -> None:
        """Save dataset memory to a companion file."""
        mem_path = self.filepath.replace(".json", "_memory.json")
        dirpath = os.path.dirname(mem_path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        with open(mem_path, "w") as f:
            json.dump(self._dataset_memory, f, indent=2)
