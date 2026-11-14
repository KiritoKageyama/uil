"""RunManager -- owns every active/completed run this server process
has started. Each run executes UniversalOptimizer.fit() on a daemon
thread (fit() is synchronous, CPU-bound XGBoost/etc. training; running
it directly in an async request handler would block the event loop).

Cross-thread access to a run's live cognition-module instances (library,
network, formula_library, causal_engine, episodic_attention) is GIL-safe
for simple reads/reassignment, but iterating a dict while the worker
thread inserts into it can raise "dictionary changed size during
iteration". RunRecord.lock (an RLock) guards both the on_stage snapshot
build and every state-route read/edit section -- see routes/state.py.
"""

from __future__ import annotations
import threading
import traceback
import uuid

import numpy as np

from figpso_uil.core.classifier_registry import CLASSIFIERS
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch
from figpso_uil.core.optimizer import UniversalOptimizer

from figpso_uil.webui.backend import datasets
from figpso_uil.webui.backend.ws_hub import hub


class _Cfg:
    def __init__(
        self,
        stages,
        n_particles,
        pso_iters,
        target_accuracy=None,
        max_stages=500,
        boosting_rounds=30,
        use_resource_budget=True,
        resource_fraction=0.75,
    ):
        self.stages = stages
        self.n_particles = n_particles
        self.pso_iters = pso_iters
        self.particles = n_particles
        self.pso_iters_per_stage = pso_iters
        self.target_accuracy = target_accuracy
        self.max_stages = max_stages
        self.boosting_rounds = boosting_rounds
        self.use_resource_budget = use_resource_budget
        self.resource_fraction = resource_fraction
        # Webui users pick a classifier explicitly — the M3 domain router
        # must not silently replace it (same philosophy as removing the
        # old XGBoost→RandomForest multiclass swap). The route decision
        # is still computed, recorded, and minted as a routing rule.
        self.disable_routing = True
        # Core UIL wiring (signal translator + cognition→L1 + utility
        # formulas) is applied by UniversalOptimizer via apply_uil_mode —
        # web/desktop runs get the same connected modules as CLI.
        self.uil_enabled = True
        self.uil_connect_pack = False


class RunRecord:
    def __init__(self, run_id: str, dataset_name: str, classifier_name: str, stages_total: int):
        self.run_id = run_id
        self.dataset_name = dataset_name
        self.classifier_name = classifier_name
        self.stages_total = stages_total
        self.status = "running"
        self.last_snapshot: dict | None = None
        self.error: str | None = None
        self.optimizer: UniversalOptimizer | None = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()  # set = not paused
        self.lock = threading.RLock()

    def to_summary(self) -> dict:
        with self.lock:
            snap = self.last_snapshot
        return {
            "run_id": self.run_id,
            "dataset_name": self.dataset_name,
            "classifier": self.classifier_name,
            "status": self.status,
            "stages_total": self.stages_total,
            "last_stage": snap.get("stage") if snap else None,
            "error": self.error,
        }


class RunManager:
    def __init__(self):
        self._runs: dict[str, RunRecord] = {}

    def list_runs(self) -> list[dict]:
        return [r.to_summary() for r in self._runs.values()]

    def get(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    def start_run(
        self,
        dataset_name: str,
        classifier_name: str,
        seed: int,
        stages: int,
        n_particles: int,
        pso_iters: int,
        boosting_rounds: int,
        target_accuracy: float | None = None,
        max_stages: int = 500,
        symbolic_learning_mode: str = "auto",
    ) -> tuple[RunRecord, list[str]]:
        # Only one heavy UIL worker at a time — multiple runs each spawn
        # parallel thread pools and can freeze the web server.
        stopped_previous: list[str] = []
        for prev in list(self._runs.values()):
            if prev.status in ("running", "paused"):
                prev.stop_event.set()
                prev.pause_event.set()
                with prev.lock:
                    prev.status = "stopped"
                stopped_previous.append(prev.run_id)

        if dataset_name not in datasets.list_datasets():
            raise KeyError(f"Unknown dataset '{dataset_name}'")
        X, y = datasets.load_dataset(dataset_name, seed=seed)

        num_classes = len(np.unique(y))
        # XGBoost multi-class is safe since the M0.3 capability guard
        # (supports_custom_objective is False for num_classes > 2, so the
        # binary-only loss-formula path never engages) — run the classifier
        # the user actually asked for.
        effective_classifier = classifier_name
        classifier_factory = CLASSIFIERS.get(effective_classifier)
        if classifier_factory is None:
            raise KeyError(f"Unknown classifier '{effective_classifier}'")
        classifier = classifier_factory(seed, num_classes, boosting_rounds)
        if hasattr(classifier, "apply_config") and effective_classifier == "SymbolicFormula":
            classifier.apply_config({"learning_mode": symbolic_learning_mode})
        search = PrincipleGuidedSearch()
        cfg = _Cfg(
            stages=stages,
            n_particles=n_particles,
            pso_iters=pso_iters,
            target_accuracy=target_accuracy,
            max_stages=max_stages,
            boosting_rounds=boosting_rounds,
        )

        run_id = uuid.uuid4().hex[:12]
        optimizer = UniversalOptimizer(
            cfg=cfg, seed=seed, dataset_name=dataset_name,
            fold=0, start_fold=0, classifier=classifier, search=search,
            classifier_factory=classifier_factory,
            classifier_build_args=(seed, num_classes, boosting_rounds),
        )
        stages_total = max_stages if target_accuracy is not None else stages
        optimizer._observability_meta = {
            "source": "webui",
            "run_id": run_id,
            "pid": __import__("os").getpid(),
            "dataset": dataset_name,
            "classifier": effective_classifier,
            "requested_classifier": classifier_name,
            "isolated_state": False,
        }

        record = RunRecord(run_id, dataset_name, effective_classifier, stages_total)
        record.optimizer = optimizer
        self._runs[run_id] = record

        def on_stage(snapshot: dict) -> None:
            with record.lock:
                record.last_snapshot = snapshot
                if snapshot.get("target_reached"):
                    record.status = "completed"
            hub.push(run_id, {"type": snapshot.get("event", "stage"), **snapshot})

        def worker() -> None:
            try:
                result = optimizer.fit(
                    X, y,
                    on_stage=on_stage,
                    stop_event=record.stop_event,
                    pause_event=record.pause_event,
                )
                with record.lock:
                    record.status = "stopped" if record.stop_event.is_set() else "completed"
                hub.push(run_id, {
                    "type": "finished", "status": record.status,
                    "accuracy": result.get("accuracy"),
                    "stages_run": result.get("stages_run"),
                    "target_accuracy": result.get("target_accuracy"),
                    "target_reached": result.get("target_reached"),
                })
            except Exception as e:
                with record.lock:
                    record.status = "error"
                    record.error = f"{type(e).__name__}: {e}"
                hub.push(run_id, {
                    "type": "error", "error": record.error,
                    "traceback": traceback.format_exc(),
                })

        thread = threading.Thread(target=worker, daemon=True)
        record.thread = thread
        thread.start()
        return record, stopped_previous

    def stop_run(self, run_id: str) -> None:
        record = self._require(run_id)
        record.stop_event.set()
        record.pause_event.set()  # don't leave it stuck paused
        with record.lock:
            record.status = "stopped"

    def pause_run(self, run_id: str) -> None:
        record = self._require(run_id)
        record.pause_event.clear()
        with record.lock:
            record.status = "paused"

    def resume_run(self, run_id: str) -> None:
        record = self._require(run_id)
        record.pause_event.set()
        with record.lock:
            record.status = "running"

    def _require(self, run_id: str) -> RunRecord:
        record = self._runs.get(run_id)
        if record is None:
            raise KeyError(f"Unknown run_id '{run_id}'")
        return record

    def find_active_for_dataset(self, dataset_name: str) -> RunRecord | None:
        for record in self._runs.values():
            if record.dataset_name == dataset_name and record.status in ("running", "paused"):
                return record
        return None


manager = RunManager()
