"""Universal Optimizer — The central engine of the UIL.

Never mentions any specific classifier or search strategy.
Talks only through protocols. Wires together:
- Journal (memory)
- StressEngine (pressure)
- Miner (cause discovery)
- Library (principles)
- Executor (principle application)
- SearchStrategy (action proposal)
- ClassifierAdapter (model training)
- FailsafeEngine (last resort diagnosis)
"""

from __future__ import annotations
import copy
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import mutual_info_classif

from figpso_uil.core.utils import blas_thread_scope, blas_threads_for_slots, safe_minmax, coerce_float_array
from figpso_uil.cognition.uil_journal import UIL_Journal
from figpso_uil.cognition.uil_stress import StressEngine
from figpso_uil.cognition.uil_miner import UIL_Miner
from figpso_uil.cognition.uil_library import UIL_Library
from figpso_uil.cognition.uil_executor import UIL_Executor
from figpso_uil.cognition.formula_engine import FormulaEngine
from figpso_uil.cognition.formula_library import FormulaLibrary
from figpso_uil.cognition.investigation_engine import InvestigationEngine
from figpso_uil.cognition.editable_network import EditableNetwork
from figpso_uil.cognition.bootstrap_knowledge import apply_bootstrap
from figpso_uil.cognition.knowledge_archive import KnowledgeArchive
from figpso_uil.cognition.guard_telemetry import GuardTelemetry
from figpso_uil.cognition.dataset_champion import DatasetChampionStore
from figpso_uil.cognition.journal_forgetting import (
    JournalForgetting, PrincipleVersioning, ContradictionDetector,
)
from figpso_uil.cognition.working_memory_pruner import WorkingMemoryPruner
from figpso_uil.cognition.episodic_attention import EpisodicAttention
from figpso_uil.cognition.causal_intervention import CausalInterventionEngine
from figpso_uil.cognition.live_feed import publish as publish_live_feed
from figpso_uil.cognition.failure_point import record_failure_point
from figpso_uil.cognition.execution_context import ExecutionContext
from figpso_uil.cognition.paths import results_path
from figpso_uil.cognition.structural_representation import StructuralEncoder
from figpso_uil.core.resource_budget import (
    apply_budget_to_classifier,
    detect_resource_budget,
)
from figpso_uil.cognition.meta_principle_merger import MetaPrincipleMerger
from figpso_uil.cognition.abstraction_safeguards import AbstractionSafeguards
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch
from figpso_uil.search.hybrid_foundation import KnowledgeFoundationBuilder
from figpso_uil.search.architecture_race import (
    ArchitectureRace, ArmMemory, rungs as race_rungs, stratified_subsample,
)
from figpso_uil.cognition.fingerprint import compute_fingerprint, save_fingerprint
from figpso_uil.core.normalization import fit_normalization, NormalizationSpec
from figpso_uil.core.missing_data import fit_missing_data, MissingDataSpec
from figpso_uil.core.uil_mode import apply_uil_mode
from figpso_uil.cognition.signal_translator import translate as translate_signals
from figpso_uil.cognition.domain_router import (
    RoutedController, RoutingMemory, explain as route_explain,
)
from figpso_uil.core.strategy_classifiers import build_classifier_for_strategy
from figpso_uil.cognition.llm_mutator import LLMMutator
from figpso_uil.cognition.lab_report import generate_lab_report
from figpso_uil.cognition.situation_policy import (
    recommend as recommend_situation,
    apply_to_cfg as apply_situation_cfg,
)
from figpso_uil.cognition.thresholdout import beats as thresholdout_beats
from figpso_uil.cognition.cell_store import build_composite_snapshot
from figpso_uil.cognition.version_store import VersionStore
from figpso_uil.cognition.goal_cell import GoalStore


class UniversalOptimizer:
    """The UIL's central optimization engine.
    
    Orchestrates the co-evolution loop:
    1. Build state vector
    2. Get overrides from discovered principles
    3. Search proposes 1-4 actions
    4. Evaluate each with classifier
    5. Record outcome + stress + regret to journal
    6. Mine for new principles
    7. Repeat
    
    Args:
        cfg: Configuration object with at least:
            - stages: int (number of co-evolution stages)
            - n_particles: int (for PSO)
            - pso_iters: int (for PSO)
        seed: Random seed.
        dataset_name: Name of the dataset.
        fold: Current CV fold number.
        start_fold: Starting fold.
        classifier: Any ClassifierAdapter implementation.
        search: Any SearchStrategy implementation (default: PrincipleGuidedSearch).
    """

    # V20+.16: ablation harness support (experiments/ablation_harness.py).
    # Every key defaults True -- omitting `ablation` entirely reproduces
    # today's behavior exactly, same backward-compat guarantee as
    # fit()'s on_stage/stop_event/pause_event.
    # 2026-07-23: the existence audit could only reach four of these. The
    # other five ran on every fit with NO way to remove them, so "is this
    # subsystem connected to anything?" was an unanswerable question for
    # more than half the cognitive layer. Each new key gates every call
    # site of exactly one subsystem, so `experiments/existence_audit.py`
    # can put an identity test on all nine.
    #
    # Dependency note: `meta_principles` and `safeguards` consume
    # StructuralPrinciple objects, so turning `structural` off necessarily
    # removes their input too. The audit reports that honestly rather than
    # pretending three independent switches exist where there are two.
    DEFAULT_ABLATION = {
        "principles": True,
        "network": True,
        "formulas": True,
        "causal": True,
        "episodic": True,
        "structural": True,
        "meta_principles": True,
        "safeguards": True,
        "goal": True,
    }

    # Working knowledge-graph size management (bug #18: an unbounded working
    # graph reached ~49k edges and made every traversal pathologically slow
    # — the "142-minute hang"). A prune is TRIGGERED when the live edge
    # count exceeds GRAPH_PRUNE_TRIGGER_EDGES and reduces the graph to
    # GRAPH_MAX_EDGES / GRAPH_MAX_NODES (everything pruned is archived,
    # nothing lost). The gap between the trigger (6000) and the target
    # (4000) is deliberate hysteresis so a graph hovering near the cap is
    # not re-pruned on consecutive stages.
    #
    # UIL hardcoded-value audit, Category C (2026-07-24): these three values
    # were previously written out as bare literals at four call sites in
    # fit() (`> 6000` twice, `max_edges=4000, max_nodes=300` three times),
    # so a future change to the graph budget had to be made in four places
    # to be made at all. Consolidated here. They are also the natural home
    # for a future evidence-based dynamic budget (Category A of the audit:
    # a safety rail whose bind-rate should be logged before it is loosened).
    GRAPH_PRUNE_TRIGGER_EDGES = 6000
    GRAPH_MAX_EDGES = 4000
    GRAPH_MAX_NODES = 300

    def __init__(
        self,
        cfg,
        seed: int,
        dataset_name: str,
        fold: int,
        start_fold: int,
        classifier,  # ClassifierAdapter
        search=None,  # SearchStrategy
        ablation: dict | None = None,
        state_dir: str | None = None,
        classifier_factory=None,
        classifier_build_args: tuple | None = None,
        classifier_build_kwargs: dict | None = None,
    ):
        self.cfg = cfg
        self.seed = seed
        self.dataset_name = dataset_name
        self.fold = fold
        self.start_fold = start_fold

        # Classifier adapter (never knows what's behind this)
        self.classifier = classifier
        self._classifier_factory = classifier_factory
        self._classifier_build_args = classifier_build_args or ()
        self._classifier_build_kwargs = dict(classifier_build_kwargs or {})
        self._parallel_action_workers = 1

        # Search strategy (default: PrincipleGuidedSearch)
        self.search = search or PrincipleGuidedSearch()
        # Opt-in guidance-ratchet fix (see PrincipleGuidedSearch.
        # update_feedback). Default OFF -- flip on for the A/B, and only
        # default it on after a cross-seed sweep, per the E6 lesson.
        # (Covers an injected PSOSearch too -- fig_xgboost.py passes one in.)
        if getattr(self.cfg, "fix_guidance_ratchet", False) and hasattr(
            self.search, "fix_guidance_ratchet"
        ):
            self.search.fix_guidance_ratchet = True

        # V20+.16: which cognition subsystems actually run this fit().
        self.ablation = {**self.DEFAULT_ABLATION, **(ablation or {})}
        self._state_dir = state_dir

        # Master UIL off-switch + experimental connect pack (see core/uil_mode.py).
        # Applied after the caller ablation merge so uil_enabled=False can
        # force a pure learner+search baseline even if ablation was omitted.
        self.ablation, self._uil_mode_meta = apply_uil_mode(self.cfg, self.ablation)

        # V20+.16: isolated persisted state for ablation/experiment runs
        # that must never share a file with another run -- otherwise a
        # "principles OFF" arm could still read principles a "principles
        # ON" arm wrote earlier into the shared default path. None (the
        # default) preserves today's behavior: everything resolves
        # against the package root via paths.results_path().
        def _state_path(filename: str) -> str:
            return (
                os.path.join(state_dir, filename)
                if state_dir is not None
                else results_path(filename)
            )
        # Expose for fit()-time writers (fingerprint, routing, live feed).
        self._state_path = _state_path

        # Cognition layer
        self.journal = UIL_Journal(filepath=_state_path("uil_journal_v19.json"))
        self.stress_engine = StressEngine(
            # Category D #2 — Gate 2 PASSED (2026-07-24): 5/6 exact ties vs
            # classic, spambase +0.0004, mean ≈ 0. Default ON.
            adaptive_blend=bool(getattr(cfg, "stress_adaptive_blend", True))
        )
        self.miner = UIL_Miner()
        self.library = UIL_Library(
            library_path=_state_path("uil_principles_v19.json"),
            version_store=VersionStore.shared() if state_dir is None else None,
        )
        self.executor = UIL_Executor()

        # V20: Formula engine + library
        self.formula_engine = FormulaEngine(seed=seed)
        self.formula_library = FormulaLibrary(
            filepath=_state_path("uil_formula_library.json"),
            version_store=VersionStore.shared() if state_dir is None else None,
        )

        # V20+.18: SymbolicFormulaAdapter (formula-as-classifier) owns its
        # OWN engine/library pair, physically separate from the
        # loss-function population above -- duck-typed, None for every
        # other classifier, so this never imports the adapter class here.
        self.symbolic_engine = getattr(self.classifier, "formula_engine", None)
        self.symbolic_library = getattr(self.classifier, "formula_library", None)

        # Failsafe (populated lazily if adapters available)
        self.failsafe = None

        # V20+.4: Investigation engine
        self.investigation_engine = InvestigationEngine()

        # V20+.3: Editable Network (graph of DT rules)
        # Formula deps wired in so edges can actually execute their
        # formula_id instead of merely listing it (see editable_network.py).
        self.network = EditableNetwork(
            formula_engine=self.formula_engine,
            formula_library=self.formula_library,
            filepath=_state_path("uil_network_v20.json"),
        )

        # V20+.5: Causal Intervention Engine
        self.causal_engine = CausalInterventionEngine(
            filepath=_state_path("uil_causal_v20.json"),
            version_store=VersionStore.shared() if state_dir is None else None,
        )

        # V20+.7: Journal Forgetting (recency-weighted mining)
        self.journal_forgetting = JournalForgetting()
        self.working_memory_pruner = WorkingMemoryPruner(self.journal_forgetting)
        self.principle_versioning = PrincipleVersioning()
        self.contradiction_detector = ContradictionDetector()

        # V20+.8: Episodic Attention (KDE warm-start)
        self.episodic_attention = EpisodicAttention(
            filepath=_state_path("uil_episodes_v20.json"),
            version_store=VersionStore.shared() if state_dir is None else None,
        )
        self.knowledge_archive = KnowledgeArchive.shared(
            _state_path("uil_knowledge_archive_v20.json")
        )
        self.champion_store = DatasetChampionStore(self.knowledge_archive)

        # Guard telemetry (audit Part E step 2): observe-only bind-rate log
        # for the Category-A safety rails, scoped to this run's state_dir so
        # experiment/audit runs don't pollute the shared default log. Pure
        # observation — never changes a decision. Flushed once at end of fit.
        self.guard_telemetry = GuardTelemetry(
            _state_path("uil_guard_telemetry.json")
        )

        # Part 15: Hybrid Foundation Builder (PSO↔reasoning transition)
        self.foundation = KnowledgeFoundationBuilder()
        # Category D #1 — Gate 2 PASSED (2026-07-24) with stress_adaptive.
        # Soft count normalization no longer saturates production stores.
        # Default ON; set cfg.maturity_soft_normalization=False to roll back.
        self.foundation.soft_normalization = bool(
            getattr(self.cfg, "maturity_soft_normalization", True)
        )

        # V21.1: Structural Encoder (concept vocabulary)
        self.structural_encoder = StructuralEncoder()

        # V21.2: Meta-Principle Merger (community detection)
        self.meta_merger = MetaPrincipleMerger(
            filepath=_state_path("uil_meta_principles.json")
        )

        # V21.3: Abstraction Safeguards
        self.safeguards = AbstractionSafeguards()

        # LCA-inspired top-down goal channel (2026-07-18) — a small
        # KnowledgeCell store holding an operator- or LLM-proposed
        # directive that biases KG traversal + hypothesis ranking. Read
        # at fit() step 0; a None active goal reproduces prior behavior.
        self._goal_store = GoalStore(
            filepath=_state_path("uil_goals.json")
        )

        # Architecture race (cfg.search_mode == "race"). Default "classic"
        # keeps the PSO/principle-guided path byte-identical -- the race is
        # opt-in until it clears the standing 3-seed x 6-dataset gate.
        self._search_mode = str(getattr(self.cfg, "search_mode", "classic"))
        self._race: ArchitectureRace | None = None
        if self._search_mode == "race":
            self._race = ArchitectureRace(
                memory=ArmMemory(filepath=_state_path("uil_race_memory.json")),
                seed=self.seed,
            )
        self._race_stats: dict = {}

        # Internal state
        self._active_goal = None
        self._best_model = None
        # THIS RUN's best only -- never seeded from a stored champion.
        # See the champion-load block in fit() (bug #26).
        self._best_accuracy = 0.0
        # The stored champion, kept beside the run for the end-of-run
        # comparison rather than used as a gate on it.
        self._champion_bar: float | None = None
        self._champion_payload: dict | None = None
        # Bug #29: warm start is for RESUMING a run, not for making the
        # candidates inside a stage cheap. One warm-started fit per run.
        self._warm_start_consumed = False
        self._best_f1 = 0.0        # weighted F1 (frequency-weighted)
        self._best_macro_f1 = 0.0  # macro F1 (unweighted mean over classes)
        self._best_feature_weights = None
        self._best_state = None
        self._best_config = None  # V20: best classifier config
        self._total_cpu_time = 0.0
        self._previous_accuracy = 0.0
        # EXPERIMENTAL (held-out principle gate): when UIL_HELDOUT_GATE=1,
        # a never-touched held-out split is carved off before the train/val
        # split, and a triggered principle earns "helped" credit only if it
        # improves HELD-OUT accuracy -- not the validation score the search
        # already optimizes. This tests whether principles improve
        # generalization or merely overfit the validation split. Off by
        # default so production behaviour and existing tests are unchanged.
        self._heldout_gate = os.environ.get("UIL_HELDOUT_GATE", "0") == "1"
        self._X_heldout = None
        self._y_heldout = None
        self._previous_heldout = 0.0
        # M0.1 (v22 blueprint §4.1): extra resampled train/val splits.
        # The winning action is re-scored on these each stage so regret,
        # mining labels, and principle credit reflect multi-split
        # validation instead of the one split the search loop maximizes.
        self._extra_val_splits: list = []
        self._base_guidance = None
        self._max_stress_reached = 0.0
        self._current_formula_id = None  # V20: active formula ID
        # V20+.12: action/config that won the PREVIOUS stage's candidate
        # competition -- the only action/config data causally available
        # before this stage's own candidates exist. Fed into
        # ExecutionContext for the executor/network (see execution_context.py).
        self._last_executed_action: dict = {}
        self._last_executed_config: dict = {}
        # Principles that fired in the previous stage — used to propagate
        # co-activation bonuses: when A fired last stage, A's successful
        # co-activation partners get a temporary confidence boost this stage
        # so the executor is more likely to trigger them (Part 16).
        self._last_triggered_principles: list[str] = []
        # Rolling window of per-stage accuracies (last 10) — fed to
        # update_maturity as recent_accuracy so maturity reflects actual
        # sustained performance, not the all-time peak (_best_accuracy).
        # Without this, a lucky stage-0 peak permanently inflates the
        # accuracy_score component of maturity, causing exploration_ratio
        # to decay as if the system has reliably mastered the problem even
        # when 90%+ of subsequent stages are well below that peak.
        self._recent_stage_accuracies: list[float] = []
        # Stages since best accuracy last improved -- drives the formula/
        # neural-proposer evolution trigger (stagnation means the current
        # population has plateaued, which is exactly when to invent new
        # structure, regardless of raw stress).
        self._stages_since_improvement: int = 0
        # Per-channel last-evolution stage for the birth-rate limiter
        # (see _want_new_structure) -- prevents the plateau runaway.
        self._last_structure_stage: dict[str, int] = {}
        # Highest system_stress seen so far in THIS run. The evolution
        # trigger is relative to it -- see _want_new_structure for the
        # measurement that motivated that.
        self._stress_peak: float = 0.0
        # Fitted at step 0 of fit(); predict() must apply the same forward
        # transform or the model is served out-of-distribution.
        self._normalization = None
        # Fitted just before normalization; same train/serve rule as above.
        self._missing_data = None

        # Observability: web UI sets run_id; terminal tests use defaults
        self._observability_meta: dict = {
            "source": "terminal",
            "pid": os.getpid(),
            "uil_mode": getattr(self, "_uil_mode_meta", None),
        }

    def set_failsafe(self, failsafe) -> None:
        """Set the failsafe engine (optional)."""
        self.failsafe = failsafe

    # ── Main Entry Point ─────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        on_stage: Callable[[dict], None] | None = None,
        stop_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
    ) -> dict:
        """Run the full co-evolution optimization loop.

        Args:
            on_stage: Optional callback invoked once per completed stage
                with a flat, JSON-safe snapshot dict (stress/regret/
                maturity/active formula/network activations/etc.) --
                used by webui/ for live WebSocket streaming. Stage
                snapshots are ALSO written to results/uil_live_feed.jsonl
                when the web console is running (or UIL_LIVE_FEED=1),
                so terminal tests appear in the UI without using on_stage.
            stop_event: Optional cooperative-cancellation flag, checked
                at the top of each stage. If set, the loop breaks before
                that stage's work begins.
            pause_event: Optional cooperative-pause flag, waited on at
                the top of each stage. Starts "set" (= not paused) by
                convention -- callers should construct it pre-set.

        Returns dict with:
        - 'model': best trained model
        - 'accuracy': best validation accuracy
        - 'f1_score': best F1 score
        - 'feature_weights': best feature weight vector
        - 'selected_features': indices of selected features
        - 'n_selected': number of selected features
        - 'stages_run': number of stages completed
        - 'max_stress': maximum system stress reached
        - 'principles_discovered': count of principles
        """
        n_samples, n_features = X.shape
        num_classes = len(np.unique(y))
        planned_stages = getattr(self.cfg, "stages", 10)
        target_accuracy = getattr(self.cfg, "target_accuracy", None)
        # When target_accuracy is set, keep the same train/val split and
        # co-evolution loop running until hit or max_stages (safety cap).
        max_stages = getattr(self.cfg, "max_stages", planned_stages)
        if target_accuracy is not None:
            max_stages = max(max_stages, planned_stages)
        else:
            max_stages = planned_stages
        stages = max_stages  # observability / episodic "stages_to_converge"

        self._observability_meta["script"] = (
            os.path.basename(sys.argv[0]) if sys.argv else "unknown"
        )
        self._observability_meta["dataset"] = self.dataset_name
        self._observability_meta["classifier"] = getattr(
            self.classifier, "name", "unknown"
        )
        self._observability_meta["isolated_state"] = self._state_dir is not None

        # Validation split. EXPERIMENTAL (held-out principle gate): when
        # enabled, first carve a never-touched held-out set (used only to
        # judge whether principles generalize, never for training/selection),
        # then split the remainder into train/val exactly as before.
        if self._heldout_gate:
            from sklearn.model_selection import train_test_split as _tts
            X_rest, self._X_heldout, y_rest, self._y_heldout = _tts(
                X, y, test_size=0.15, random_state=self.seed, stratify=y
            )
            _X_pool, _y_pool = X_rest, y_rest
        else:
            _X_pool, _y_pool = X, y
        X_train, X_val, y_train, y_val = self._split(_X_pool, _y_pool)

        # M0.1 (v22 blueprint §4.1): build cv_folds-1 extra resampled
        # splits from the same pool (never the held-out set). The primary
        # split above still drives action selection and model artifacts;
        # the winner's score is averaged across all splits so regret and
        # everything mined from it measure generalization, not one split.
        # cv_folds=1 restores legacy single-split behavior.
        cv_folds = max(1, int(getattr(self.cfg, "cv_folds", 3)))
        # Auto-scale CV cost by dataset size. k-fold regret re-scores the
        # winner on every extra split, so cv_folds=3 TRIPLES per-stage eval
        # cost -- impractical at scale (covertype 100k x 7-class OvR x PSO
        # ran >40 min CPU with cv_folds=3). Large datasets already get low-
        # variance estimates from sheer sample count, so one split suffices;
        # small datasets keep the full k-fold for a stable generalization
        # signal. Opt out with cfg.auto_cv_scale = False.
        if getattr(self.cfg, "auto_cv_scale", True) and cv_folds > 1:
            if n_samples > 20000:
                cv_folds = 1
            elif n_samples > 8000:
                cv_folds = min(cv_folds, 2)
            if cv_folds != int(getattr(self.cfg, "cv_folds", 3)):
                self._observability_meta["cv_folds_autoscaled"] = {
                    "from": int(getattr(self.cfg, "cv_folds", 3)),
                    "to": cv_folds, "n_samples": int(n_samples),
                }
        # A rotating within-stage split needs at least one FIXED fold to
        # compare stages against, or "best so far" becomes "luckiest draw".
        if getattr(self.cfg, "rotate_val_split", False):
            cv_folds = max(cv_folds, 2)
        self._extra_val_splits = []
        if cv_folds > 1:
            from sklearn.model_selection import train_test_split as _tts_extra
            for _i in range(cv_folds - 1):
                try:
                    _Xt, _Xv, _yt, _yv = _tts_extra(
                        _X_pool, _y_pool, test_size=0.2,
                        random_state=self.seed + 101 * (_i + 1),
                        stratify=_y_pool,
                    )
                except ValueError:
                    # Too few samples per class to stratify another split;
                    # fall back to unstratified rather than dying.
                    _Xt, _Xv, _yt, _yv = _tts_extra(
                        _X_pool, _y_pool, test_size=0.2,
                        random_state=self.seed + 101 * (_i + 1),
                    )
                self._extra_val_splits.append((_Xt, _yt, _Xv, _yv))

        # ── Step 0 (v22 §2–§3): impute → fingerprint → normalize → route ─
        # Missing-data first so fingerprint probes and normalization never
        # see NaN/Inf. Fit on X_train only; every other matrix + predict()
        # reuse the same forward (bug #19's train/serve invariant).
        if getattr(self.cfg, "handle_missing_data", True):
            self._missing_data = fit_missing_data(
                X_train,
                add_indicators=bool(
                    getattr(self.cfg, "missing_indicators", True)
                ),
            )
            X_train = self._missing_data.forward(X_train)
            X_val = self._missing_data.forward(X_val)
            if self._heldout_gate:
                self._X_heldout = self._missing_data.forward(self._X_heldout)
            self._extra_val_splits = [
                (
                    self._missing_data.forward(_Xt), _yt,
                    self._missing_data.forward(_Xv), _yv,
                )
                for _Xt, _yt, _Xv, _yv in self._extra_val_splits
            ]
            n_features = int(X_train.shape[1])
            self._observability_meta["missing_data"] = self._missing_data.summary()
        else:
            self._missing_data = None

        self._dataset_fingerprint = compute_fingerprint(
            X_train, y_train, seed=self.seed, dataset_name=self.dataset_name,
        )
        save_fingerprint(
            self._dataset_fingerprint,
            filepath=self._state_path("uil_fingerprints_v22.json"),
        )
        # Situation-adaptive knobs (System Refinement Category D / evidence
        # probes). Opt-in via cfg.situation_adaptive; never touches Category-A
        # safety rails. Logged for HITL transparency.
        _sit = recommend_situation(
            self._dataset_fingerprint,
            self.cfg,
            search_mode=str(getattr(self.cfg, "search_mode", "classic")),
        )
        _applied = apply_situation_cfg(self.cfg, _sit)
        self._observability_meta["situation_policy"] = {
            "applied": _applied,
            "reasons": list(_sit.get("_reasons") or []),
        }
        if "stress_adaptive_blend" in _applied:
            self.stress_engine.adaptive_blend = True
        if "maturity_soft_normalization" in _applied:
            self.foundation.soft_normalization = True
        self._normalization = fit_normalization(
            X_train, fingerprint=self._dataset_fingerprint,
        )
        X_train = self._normalization.forward(X_train)
        X_val = self._normalization.forward(X_val)
        if self._heldout_gate:
            self._X_heldout = self._normalization.forward(self._X_heldout)
        self._extra_val_splits = [
            (
                self._normalization.forward(_Xt), _yt,
                self._normalization.forward(_Xv), _yv,
            )
            for _Xt, _yt, _Xv, _yv in self._extra_val_splits
        ]
        # ── Rotating validation split (cfg.rotate_val_split) ──────────
        # The split above is drawn ONCE and every action of every stage is
        # then scored against it -- ~6 candidates x N stages maximized over
        # one 20% slice. On small data that slice is tiny (sonar val ~35
        # rows, one row = 2.9%), and "keep the best of 60 tries against one
        # answer key" is exactly how the +0.03-0.06 optimism gap CLAUDE.md
        # calls unavoidable gets manufactured. Architecture Race measured
        # the same thing from the other side: corr(delta, val-gap swing) =
        # -0.977.
        #
        # Fix: rotate the WITHIN-STAGE split (drawn from seed + stage, so
        # replay stays byte-identical) and judge ACROSS stages on the fixed
        # extra folds, which are the same for every stage and therefore
        # comparable. Splits are stored as row INDICES, not row copies --
        # 56 KB rather than 44 MB per split on MNIST, and safe to hand to
        # workers.
        #
        # Normalization is deliberately still fit ONCE on the initial
        # X_train (routing, fingerprint and predict() all depend on a single
        # transform). Rows that later rotate into a validation fold did
        # inform that transform, which is a small, bounded leak -- far
        # smaller than the selection bias it replaces.
        self._rotate_val = bool(getattr(self.cfg, "rotate_val_split", False))
        self._pool_X = self._pool_y = None
        if self._rotate_val:
            self._pool_X = np.vstack([X_train, X_val])
            self._pool_y = np.concatenate([y_train, y_val])

        self._route_controller = RoutedController(
            memory=RoutingMemory(
                filepath=self._state_path("uil_routing_memory.json"),
            ),
            episodic_attention=self.episodic_attention,
        )
        self._route_decision = self._route_controller.decide(self._dataset_fingerprint)
        self._observability_meta["route"] = self._route_decision
        self._observability_meta["fingerprint_id"] = self._dataset_fingerprint.get(
            "fingerprint_id",
        )

        # Category-E telemetry (audit Part E). `RoutingMemory` already records
        # fingerprint -> strategy -> outcome, so WHICH route won is known.
        # What was missing is evidence about the CUTOFFS themselves
        # (FORMULA_R2_MATH = 0.90, LINEAR_ACC_STRUCTURED = 0.92): they cite
        # real cross-seed numbers but from a handful of benchmarks, so the
        # open question is whether datasets ever land NEAR them. Logging the
        # probe values answers it: if every dataset sits far from a cutoff,
        # its exact value is irrelevant and it can stay fixed; if datasets
        # cluster at the boundary, the cutoff is load-bearing and deserves
        # to be learned. `bound` marks which side of the cutoff we fell on.
        try:
            from figpso_uil.cognition.domain_router import (
                FORMULA_R2_MATH, LINEAR_ACC_STRUCTURED,
            )
            _fp = self._dataset_fingerprint or {}
            _r2 = _fp.get("formula_probe_r2")
            _lin = _fp.get("linear_probe_acc")
            if isinstance(_r2, (int, float)):
                self.guard_telemetry.record_signal("route_probe.formula_r2", _r2)
                self.guard_telemetry.observe(
                    "route_cutoff.formula_r2_math", float(_r2),
                    min(float(_r2), FORMULA_R2_MATH),
                )
            if isinstance(_lin, (int, float)):
                self.guard_telemetry.record_signal("route_probe.linear_acc", _lin)
                self.guard_telemetry.observe(
                    "route_cutoff.linear_acc_structured", float(_lin),
                    min(float(_lin), LINEAR_ACC_STRUCTURED),
                )
        except Exception:
            pass

        self.network.set_dataset_context(self.dataset_name)

        # M3: apply routed classifier + mint routing_rule cell (unless disabled)
        strategy = self._route_decision.get("strategy", "grad_fit")
        if not getattr(self.cfg, "disable_routing", False):
            num_classes = len(np.unique(y_train))
            routed_clf, routed_name = build_classifier_for_strategy(
                strategy,
                seed=self.seed,
                num_classes=num_classes,
                boosting_rounds=getattr(self.cfg, "boosting_rounds", 30),
                state_dir=self._state_dir,
            )
            self.classifier = routed_clf
            self.symbolic_engine = getattr(self.classifier, "formula_engine", None)
            self.symbolic_library = getattr(self.classifier, "formula_library", None)
            self._observability_meta["classifier"] = routed_name
            # Parallel workers must spawn the ROUTED adapter type. The
            # caller's factory builds the PRE-route classifier, whose
            # models the routed adapter cannot predict with (e.g. a
            # SymbolicFormula dict-model fed to XGBoostAdapter.predict).
            if self._classifier_factory is not None:
                _routed_strategy = strategy
                _routed_rounds = getattr(self.cfg, "boosting_rounds", 30)
                self._classifier_factory = (
                    lambda *_a, **_kw: build_classifier_for_strategy(
                        _routed_strategy,
                        seed=self.seed,
                        num_classes=num_classes,
                        boosting_rounds=_routed_rounds,
                        state_dir=self._state_dir,
                    )[0]
                )
                self._classifier_build_args = ()
                self._classifier_build_kwargs = {}
        if self.ablation.get("principles", True):
            self.library.record_routing_rule(
                dataset_name=self.dataset_name,
                domain=self._route_decision.get("domain", "?"),
                strategy=strategy,
                source=self._route_decision.get("source", "heuristic"),
                fingerprint_id=self._dataset_fingerprint.get("fingerprint_id"),
            )

        # E6: interaction-aware rule basis — OFF BY DEFAULT. The intent was
        # to auto-enable the E5 rule family when the fingerprint's non-linear
        # probe beats its linear probe (interaction-dense data). But a
        # rigorous 3-seed cross-seed sweep (2026-07-18) showed the auto-
        # trigger causes NET REGRESSIONS: ionosphere (gap ~0.06) dropped
        # 0.916→0.868 because rules + the weaker L1 overfit near-linear data,
        # and sonar (gap ~0.07) — which the rules DO help in isolation
        # (0.748→0.786) — got no benefit through the full optimizer (0.722,
        # worse than boosting's 0.746). No clean gap threshold separates
        # "helps" from "hurts". So the auto-enable is opt-in only; the rule
        # basis (E5) remains a proven, manually-enableable capability, and
        # interaction_gap stays on the fingerprint as a diagnostic signal.
        # Set cfg.auto_rule_basis = True to re-enable (raise the gate above
        # ~0.08 first, and prefer it on larger datasets where the optimizer
        # overhead doesn't dominate).
        if getattr(self.cfg, "auto_rule_basis", False):
            gap = self._dataset_fingerprint.get("interaction_gap")
            if (
                gap is not None and gap >= 0.05
                and hasattr(self.classifier, "use_rule_basis")
                and hasattr(self.classifier, "apply_config")
            ):
                self.classifier.apply_config({
                    "use_rule_basis": True,
                    "sparse_l1_c": max(
                        float(getattr(self.classifier, "sparse_l1_c", 0.1)), 0.5,
                    ),
                })
                self._observability_meta["rule_basis_enabled"] = {
                    "interaction_gap": gap,
                    "nonlinear_probe_acc": self._dataset_fingerprint.get("nonlinear_probe_acc"),
                    "linear_probe_acc": self._dataset_fingerprint.get("linear_probe_acc"),
                }

        # Bug #25: connect the cognitive layer to the learner. Its ONLY
        # channel is the search's feature weights, which sparse_linear
        # currently ignores (basis reads unweighted X; standardization
        # cancels column scaling) — measured ablation delta 0.0000. When
        # cfg.connect_cognition is on AND any cognitive subsystem is
        # active, tell the adapter to fold those weights into the L1
        # penalty so they can actually move the fit. Default OFF until
        # cross-seed validated (see the E6 lesson).
        if getattr(self.cfg, "connect_cognition", False):
            cognition_on = any(
                self.ablation.get(k, True)
                for k in ("principles", "network", "causal", "formulas")
            )
            if cognition_on and hasattr(self.classifier, "apply_config") \
                    and hasattr(self.classifier, "use_feature_weight_penalty"):
                self.classifier.apply_config({"use_feature_weight_penalty": True})
                self._observability_meta["cognition_connected"] = True

        # Research step 4 — DreamCoder-style utility-gated formula admission.
        # Opt-in; default OFF pending Gate 2 / existence-audit CONNECTED.
        if getattr(self.cfg, "utility_gated_formulas", False):
            if hasattr(self.classifier, "apply_config"):
                self.classifier.apply_config({"utility_gated_formulas": True})
            self._observability_meta["utility_gated_formulas"] = True
            self._utility_admit_stats = {"tested": 0, "admitted": 0, "rejected": 0}

        # Direct formula influence: residual-guided grow + forced install
        # into the sparse_linear law (does not share L1 with stumps).
        # Opt-in; Gate 2 + existence CONNECTED required before default ON.
        if getattr(self.cfg, "direct_formula_install", False):
            if hasattr(self.classifier, "apply_config"):
                self.classifier.apply_config({"direct_formula_install": True})
            self._observability_meta["direct_formula_install"] = True

        # Research step 5 — EBM cyclic fit over the composed basis.
        # Opt-in; default OFF until Gate 2. Only affects sparse_linear path.
        if getattr(self.cfg, "use_ebm_fit", False):
            if hasattr(self.classifier, "apply_config"):
                self.classifier.apply_config({"use_ebm_fit": True})
            self._observability_meta["use_ebm_fit"] = True

        # Research step 8 — KAN-style truncated-power basis family.
        # Opt-in; default OFF until Gate 2. Only affects sparse_linear path.
        if getattr(self.cfg, "use_kan_basis", False):
            if hasattr(self.classifier, "apply_config"):
                self.classifier.apply_config({"use_kan_basis": True})
            self._observability_meta["use_kan_basis"] = True

        self._llm_mutator = LLMMutator(
            # Formula building is offline trial-and-error by default —
            # never call an external API even if OPENAI_API_KEY is set.
            offline_only=bool(getattr(self.cfg, "formula_offline_only", True)),
        )
        self._version_store = VersionStore.shared() if self._state_dir is None else None
        if self._version_store is not None:
            def _composite_snapshot() -> dict:
                return build_composite_snapshot(self)
            self._version_store.set_snapshot_provider(_composite_snapshot)
            if hasattr(self.network, "_version_store") and self.network._version_store is None:
                self.network._version_store = self._version_store

        # Load the active LCA-inspired GoalCell (2026-07-18). None when no
        # operator/LLM has set one — consumers become no-ops in that case,
        # so runs without a goal are byte-identical to pre-GoalCell.
        try:
            self._active_goal = (
                self._goal_store.get_active()
                if self.ablation["goal"] else None
            )
        except Exception:
            self._active_goal = None
        if self._active_goal is not None:
            self._observability_meta["active_goal"] = self._active_goal.to_readable()

        # Category-A telemetry: thread the optimizer's scoped GuardTelemetry
        # into engines/adapters (no module singleton — that reintroduces #30b).
        for _eng in (self.formula_engine, self.symbolic_engine):
            if _eng is not None:
                _eng._guard_telemetry = self.guard_telemetry
        try:
            self.classifier._guard_telemetry = self.guard_telemetry
            _se = getattr(self.classifier, "formula_engine", None)
            if _se is not None:
                _se._guard_telemetry = self.guard_telemetry
        except Exception:
            pass

        # Scale workload to ~75% of machine CPU/RAM/GPU headroom
        self._resource_budget = None
        if getattr(self.cfg, "use_resource_budget", True):
            budget = detect_resource_budget(
                base_particles=getattr(self.cfg, "n_particles", 15),
                base_pso_iters=getattr(self.cfg, "pso_iters", 15),
                base_boosting=getattr(self.cfg, "boosting_rounds", 30),
                target_fraction=getattr(self.cfg, "resource_fraction", 0.75),
                telemetry=self.guard_telemetry,
            )
            self.cfg.n_particles = budget.n_particles
            self.cfg.pso_iters = budget.pso_iters
            if hasattr(self.cfg, "boosting_rounds"):
                self.cfg.boosting_rounds = budget.boosting_stages
            apply_budget_to_classifier(self.classifier, budget)
            intra_threads = blas_threads_for_slots(
                budget.n_jobs, budget.parallel_eval_workers,
            )
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                os.environ[key] = str(intra_threads)
            self._resource_budget = budget.to_dict()
            self._observability_meta["resource_budget"] = self._resource_budget
            self._parallel_action_workers = max(1, budget.parallel_eval_workers)
            self._intra_fit_threads = intra_threads

        # Compute base guidance via Mutual Information
        self._base_guidance = self._compute_mutual_info(X_train, y_train)

        # Initialize search strategy
        self.search.initialize(
            n_features=n_features,
            base_guidance=self._base_guidance,
            seed=self.seed,
            n_particles=getattr(self.cfg, "n_particles", 20),
            pso_iters=getattr(self.cfg, "pso_iters", 30),
        )

        if hasattr(self.search, "set_bootstrap_diversity"):
            import numpy as _np
            self.search.set_bootstrap_diversity(
                X_train, y_train, _np.random.default_rng(self.seed + 7),
            )

        # V20: Register classifier's controllable params with search
        if hasattr(self.classifier, 'get_controllable_params'):
            controllable = self.classifier.get_controllable_params()
            if hasattr(self.search, 'set_controllable_params'):
                self.search.set_controllable_params(controllable)

        # Check dataset memory — start with prior stress if dataset was hard
        prior_stress = self.journal.get_dataset_prior_stress(self.dataset_name)
        if prior_stress > 0:
            self.stress_engine.set_initial_stress(prior_stress)

        # V20: Initialize formula library with seed formulas
        seed_formulas = self.formula_engine.get_seed_formulas()
        self.formula_library.seed_formulas(seed_formulas)

        bootstrap_info = apply_bootstrap(
            self.library,
            self.network,
            symbolic_library=self.symbolic_library,
            symbolic_engine=self.symbolic_engine,
            force=False,
            state_dir=self._state_dir,
        )
        self._observability_meta["bootstrap"] = bootstrap_info

        # V20+.3: Bootstrap network from existing principles
        existing_principles = (
            self.library.get_principles(dataset_name=self.dataset_name)
            if self.ablation["principles"] else []
        )
        if (
            existing_principles
            and self.ablation["network"]
            and self.network.node_count == 0
        ):
            self.network.import_from_principles(existing_principles)

        # V20+.3: Track activation history for relationship discovery.
        # (stage, activated_nodes) tuples -- matched to journal entries by
        # stage number in discover_relationships(), not list position.
        _activation_history: list[tuple[int, list[str]]] = []

        # V20+.8: Query episodic attention for warm-start
        fingerprint = {
            "n_samples": n_samples, "n_features": n_features,
            "sparsity": float(np.mean(X_train == 0)),
            "entropy": float(np.mean(np.std(X_train, axis=0))),
            "num_classes": num_classes,
            "feat_ratio": n_features / max(n_samples, 1),
        }
        episodic_hint = (
            self.episodic_attention.retrieve(fingerprint)
            if self.ablation["episodic"] else None
        )
        if episodic_hint is not None and hasattr(self.search, "set_episodic_hint"):
            self.search.set_episodic_hint(episodic_hint)
        if episodic_hint is not None and hasattr(self.classifier, "set_episodic_hint"):
            self.classifier.set_episodic_hint(episodic_hint)

        consciousness = self.knowledge_archive.get_consciousness_brief(self.dataset_name)
        self.network.set_dataset_context(self.dataset_name)
        # Load-time guard: short runs never reach the stage%15 graph prune,
        # so the shared working graph can arrive here far past its cap
        # (observed 49k edges) and make every traverse pathologically slow.
        # Prune once up front; everything goes to the archive, nothing lost.
        _n_edges = len(self.network._edges)
        # Observe-only (audit Part E): does the graph ever actually arrive
        # oversized? Records the edge count seen at load and whether the
        # bug-#18 ceiling bound. `max` on this guard answers "how large does
        # the working graph really get" — the evidence for whether 6000 is
        # the right trigger or the graph is always tiny in normal runs.
        self.guard_telemetry.observe(
            "graph_load_prune_trigger",
            _n_edges,
            min(_n_edges, self.GRAPH_PRUNE_TRIGGER_EDGES),
        )
        if _n_edges > self.GRAPH_PRUNE_TRIGGER_EDGES:
            self.network.prune_working_graph(
                self.knowledge_archive,
                max_edges=self.GRAPH_MAX_EDGES,
                max_nodes=self.GRAPH_MAX_NODES,
                protect_nodes=set(),
                dataset=self.dataset_name,
                system_stress=0.0,
            )
        champion = self.champion_store.load(self.dataset_name)
        if champion:
            # Bug #26: the champion's accuracy is a RECORD, not a GATE.
            # It used to be loaded straight into self._best_accuracy, which
            # is also the condition for saving this stage's model. A lucky
            # previous run therefore locked the current one out: every stage
            # did full work and was discarded, and fit() returned either
            # last run's model (when the champion serialized one) or stage
            # 1's (via the need_model fallback below) -- neither of which is
            # "the best model". Worse, the bar was measured on a DIFFERENT
            # random split, so "can't beat 0.958" often just meant "the old
            # split was easier". Keep it beside the run instead: this run
            # builds its own best from nothing, and the two are compared
            # once at the end (see the champion save at the bottom of fit).
            self._champion_bar = float(champion.get("best_accuracy", 0.0))
            self._champion_payload = champion
            if champion.get("best_weights"):
                self._best_feature_weights = coerce_float_array(
                    champion["best_weights"],
                )
            if champion.get("model"):
                self._best_model = champion["model"]
                if hasattr(self.classifier, "_repair_model_trees"):
                    self.classifier._repair_model_trees(self._best_model)
                fw = self._best_model.get("feature_weights")
                if fw is not None:
                    self._best_model["feature_weights"] = coerce_float_array(fw)
            if champion.get("best_config") and hasattr(self.classifier, "apply_config"):
                self.classifier.apply_config(champion["best_config"])
            warm_w = self._best_feature_weights
            if warm_w is not None:
                if hasattr(self.search, "set_warm_start_weights"):
                    self.search.set_warm_start_weights(warm_w)
                if hasattr(self.foundation, "set_warm_start_weights"):
                    self.foundation.set_warm_start_weights(warm_w)
        elif episodic_hint and episodic_hint.get("best_weights"):
            warm_w = np.asarray(episodic_hint["best_weights"], dtype=float)
            if hasattr(self.search, "set_warm_start_weights"):
                self.search.set_warm_start_weights(warm_w)
            if hasattr(self.foundation, "set_warm_start_weights"):
                self.foundation.set_warm_start_weights(warm_w)

        if hasattr(self.classifier, "set_exploration_hint"):
            self.classifier.set_exploration_hint({
                **consciousness,
                "champion": champion,
            })
        self._observability_meta["consciousness"] = {
            "abstract_lines": len(consciousness.get("abstract_summary", [])),
            "avoid_formulas": len(consciousness.get("avoid_formulas", [])),
            "champion_accuracy": (
                champion.get("best_accuracy") if champion else None
            ),
        }

        # V20+.18: same optional hook, extended to the classifier --
        # only SymbolicFormulaAdapter implements it (warm-starts its
        # formula selection from a similar past dataset's regime).
        # (episodic set_episodic_hint above; exploration hint merges champion)

        # V20+.11: Compute feature correlation matrix
        if hasattr(self.search, "set_correlation_matrix"):
            self.search.set_correlation_matrix(X_train)

        # Per-stage mining summary for webui decision trace
        self._last_stage_mining: dict | None = None

        if on_stage is not None:
            self._emit_observability(
                on_stage,
                event="started",
                stage=-1,
                stages_total=stages,
                state={},
                principles=existing_principles,
                overrides={},
                network_result={"activated_nodes": [], "overrides": {}},
                allocation={"stage": "infant", "pso_actions": 0, "reasoning_actions": 0},
                reasoning_actions=[],
                pso_actions=[],
                triggered=[],
                decision_stress=0.0,
                system_stress=self.stress_engine.system_stress,
                regret=0.0,
                best_action_score=0.0,
                miner_r2=0.0,
                investigation=None,
                active_intervention=None,
                stage_cpu=0.0,
                current_formula_depth=None,
                current_formula_node_count=None,
                episodic_hint=episodic_hint,
                fingerprint=fingerprint,
                prior_stress=prior_stress,
            )

        # ── Co-evolution loop ──
        stages_completed = 0
        target_reached = False
        for stage in range(max_stages):
            # Cooperative stop/pause (webui/ control panel). A stop
            # issued while paused must not wait out the full pause --
            # re-check stop_event on every wake instead of one blocking
            # wait() call.
            if stop_event is not None and stop_event.is_set():
                break
            if pause_event is not None:
                while not pause_event.wait(timeout=0.5):
                    if stop_event is not None and stop_event.is_set():
                        break
                if stop_event is not None and stop_event.is_set():
                    break

            stage_start = time.perf_counter()

            # Rotate this stage's answer key (cfg.rotate_val_split). Nothing
            # can grind against a split it only sees once.
            if self._rotate_val and self._pool_X is not None:
                _tr, _va = self._stage_split(stage)
                X_train, y_train = self._pool_X[_tr], self._pool_y[_tr]
                X_val, y_val = self._pool_X[_va], self._pool_y[_va]

            # 1. Build state vector (pure numbers)
            state = self._build_state(
                n_samples, n_features, num_classes, X_train, y_train
            )

            # 2. Get overrides from discovered principles
            # V20+.1: Tier-aware retrieval — filters by quarantine for this dataset
            # V20+.16: ablation gate -- "principles OFF" must mean the
            # executor never sees any, not just that none get mined.
            principles = (
                self.library.get_principles(dataset_name=self.dataset_name)
                if self.ablation["principles"] else []
            )
            # Part 16: Propagate co-activation attention from last stage's
            # triggered principles. Each principle that fired last stage
            # may have successful co-activation partners — temporarily boost
            # their effective confidence so the executor is more likely to
            # trigger them this stage. Uses a copy to avoid mutating the
            # library's in-memory state (bonuses are stage-local).
            if self._last_triggered_principles and self.ablation["network"]:
                _co_bonuses: dict[str, float] = {}
                for _pid in self._last_triggered_principles:
                    for _partner, _bonus in self.network.get_co_activation_bonus(_pid).items():
                        _co_bonuses[_partner] = max(_co_bonuses.get(_partner, 0.0), _bonus)
                if _co_bonuses:
                    principles = [
                        {**p, "confidence": min(0.99, p.get("confidence", 0.5) + _co_bonuses[p["principle_id"]] * 0.15)}
                        if p.get("principle_id") in _co_bonuses else p
                        for p in principles
                    ]
            # V20+.12: carry the previous stage's winning action/config
            # alongside state, so 'action.*'/'config.*' conditions --
            # mined by the Miner just as often as 'state.*' ones -- can
            # actually match (see execution_context.py).
            exec_context = ExecutionContext(
                state=state,
                previous_action=self._last_executed_action,
                previous_config=self._last_executed_config,
                stage=stage,
            )
            overrides = self.executor.get_overrides(exec_context, principles)

            # V20+.3/V20+.9: Traverse network with causal gate attention.
            # V20+.16: ablation gate -- skip traversal entirely so a
            # disabled network contributes zero overrides/activations.
            if self.ablation["network"]:
                network_result = self.network.traverse_with_attention(
                    exec_context, goal=self._active_goal,
                )
                network_overrides = network_result.get("overrides", {})
                # Merge: principle overrides take priority, network fills gaps
                for key, value in network_overrides.items():
                    if key not in overrides:
                        overrides[key] = value
                # V20+.10: Feed network overrides to search
                if hasattr(self.search, "set_network_overrides"):
                    self.search.set_network_overrides(network_overrides)
                if hasattr(self.classifier, "set_graph_hint"):
                    self.classifier.set_graph_hint({
                        "activated_nodes": network_result.get("activated_nodes", []),
                        "overrides": network_overrides,
                        "feature_emphasis": self._graph_feature_emphasis(
                            network_result, n_features,
                        ),
                    })
            else:
                network_result = {"overrides": {}, "activated_nodes": []}
                network_overrides = {}
            # Track activated nodes for relationship discovery
            _activation_history.append(
                (stage, network_result.get("activated_nodes", []))
            )

            # Signal translator: cognition → seeds / particles / formula basis.
            # Opt-in (cfg.signal_translator / uil_connect_pack). Gate 2
            # 2026-07-25: forcing ON by default hurt accuracy (mean -0.0132).
            self._stage_directive = None
            stage_guidance = self._base_guidance
            if getattr(self.cfg, "signal_translator", False):
                net_payload = dict(network_result)
                if self.ablation["network"]:
                    net_payload["feature_emphasis"] = self._graph_feature_emphasis(
                        network_result, n_features,
                    )
                sym_lib = self.symbolic_library or getattr(
                    self.classifier, "formula_library", None
                )
                directive = translate_signals(
                    n_features=n_features,
                    ablation=self.ablation,
                    overrides=overrides,
                    triggered_principle_ids=self.executor.get_triggered_principles(),
                    principles=principles,
                    network_result=net_payload,
                    episodic_hint=episodic_hint,
                    formula_library=sym_lib if self.ablation["formulas"] else None,
                    X_train=X_train,
                    y_train=y_train,
                )
                self._stage_directive = directive
                if directive.feature_guidance is not None:
                    stage_guidance = np.clip(
                        0.35 * np.asarray(self._base_guidance, dtype=float)
                        + 0.65 * np.asarray(directive.feature_guidance, dtype=float),
                        0.02, 1.0,
                    )
                    if hasattr(self.search, "_base_guidance"):
                        self.search._base_guidance = stage_guidance.copy()
                if hasattr(self.classifier, "set_learner_directive"):
                    self.classifier.set_learner_directive(directive)
                cfg_payload = dict(directive.config)
                if directive.use_symbolic_basis is not None:
                    cfg_payload["use_symbolic_basis"] = directive.use_symbolic_basis
                if cfg_payload and hasattr(self.classifier, "apply_config"):
                    self.classifier.apply_config(cfg_payload)
                self._observability_meta["signal_translator"] = directive.summary()

            # 3. Search proposes actions (hybrid: PSO foundation + reasoning)
            stress_level = self.stress_engine.system_stress
            recent_journal = self.journal.get_recent(20)
            miner_r2 = self.miner.get_last_r2_score()

            # Part 15: Update maturity and get action allocation
            _rolling_recent_acc = (
                float(np.mean(self._recent_stage_accuracies[-10:]))
                if self._recent_stage_accuracies else 0.0
            )
            self.foundation.update_maturity(
                n_principles=len(principles),
                n_episodes=self.episodic_attention.episode_count,
                n_formulas=self.formula_library.formula_count(),
                miner_r2=miner_r2,
                network_nodes=self.network.node_count
                if hasattr(self.network, 'node_count') else 0,
                recent_accuracy=_rolling_recent_acc,
            )
            allocation = self.foundation.allocate_actions()

            # Category-D telemetry (audit Part E): log the marginal
            # distribution of each maturity-blend sub-score, same rationale
            # as the stress blend — a sub-score stuck at a constant (e.g.
            # network_score at 0 on a fresh graph, or formula_score capped)
            # makes its weight dead in the PSO-detachment decision. NaN
            # skill_score (no principle outcomes yet) is skipped by observe.
            for _mname, _mval in getattr(
                self.foundation, "_last_maturity_components", {}
            ).items():
                self.guard_telemetry.record_signal(
                    f"maturity_component.{_mname}", _mval,
                )

            # Reasoning-driven actions (from principle-guided search)
            reasoning_actions = []
            if allocation["reasoning_actions"] > 0:
                reasoning_actions = self.search.propose_actions(
                    state, principles, stress_level, recent_journal
                )

            # PSO/foundation exploration actions
            pso_actions = []
            if allocation["pso_actions"] > 0:
                rng = np.random.default_rng(self.seed + stage)
                pso_actions = self.foundation.get_foundation_targets(
                    state, n_features, rng
                )

            # Merge: reasoning first, then PSO fills remaining slots
            actions = reasoning_actions + pso_actions

            # ── Architecture race (cfg.search_mode == "race") ──────────
            # Replace the surrogate-optimized candidate set with one that
            # varies ARCHITECTURE (axes the learner cannot pick for
            # itself) plus the closed-form cutoff frontier. The winner is
            # then decided by real held-out accuracy in _race_actions(),
            # not by the separable surrogate the swarm was maximizing.
            # The cognitive layer still speaks: its `overrides` shape the
            # incumbent arm, which is arm 0 and always survives to the
            # final rung, so cognition proposes and measurement disposes.
            if self._race is not None:
                axes = (
                    self.classifier.get_architecture_axes()
                    if hasattr(self.classifier, "get_architecture_axes")
                    else {}
                )
                # Interaction-gap gate (E7's named next fix, converts the
                # one race regression). The judge fix made the race net
                # +0.0080 but it still LOST ionosphere -0.047: near-linear
                # data where the interaction-hungry basis families
                # (use_rule_basis / use_neural_basis) look better on the
                # internal folds and overfit the true test -- the exact E6
                # finding. When the fingerprint says interactions are weak,
                # don't even RACE those axes; the race then varies only
                # cutoffs + L1 strength + stumps, and the incumbent
                # sparse_linear wins ionosphere as it should. High-gap data
                # (sonar) keeps them. Opt-in default ON *within race mode*
                # -- race itself is opt-in, so this only ever fires when the
                # user already chose the race; it strictly narrows a search
                # that was overfitting, so it does not need its own gate to
                # be safe, but its ACCURACY effect is measured before the
                # race as a whole is defaulted on (still opt-in).
                axes = self._gate_interaction_axes(axes)
                incumbent = (
                    actions[0]
                    if actions
                    else {
                        "feature_weights": stage_guidance.copy(),
                        "penalty": 1.0,
                        "classifier_config": {},
                    }
                )
                actions = self._race.propose(
                    guidance=stage_guidance,
                    axes=axes,
                    base_action=incumbent,
                    bucket=self._race_bucket(),
                    max_arms=int(getattr(self.cfg, "race_max_arms", 12)),
                )

            if not actions:
                # Fallback: use base guidance directly
                actions = [{
                    "feature_weights": stage_guidance.copy(),
                    "penalty": 1.0,
                    "description": "Fallback base guidance",
                }]

            # V20.3: Select a formula for the custom objective. Stress-
            # governed like every other exploration knob here: a calm run
            # keeps the classifier's default loss, only a stressed run
            # spends effort trying the library's best alternative.
            self._current_formula_id = None
            # webui/ snapshot fields -- only bound below when a formula
            # is actually selected this stage, so default to None here.
            current_formula_depth = None
            current_formula_node_count = None
            if (
                self.ablation["formulas"]
                and getattr(self.classifier, "supports_custom_objective", False)
                and stress_level >= 0.3
            ):
                best_formula = self.formula_library.get_best_formula(
                    dataset_fingerprint=self.dataset_name
                )
                if best_formula is not None:
                    formula_tree = self.formula_library.get_formula_tree(
                        best_formula["formula_id"]
                    )
                    if formula_tree is not None:
                        self._current_formula_id = best_formula["formula_id"]
                        # V20+.14: structural numeric proxies for which
                        # formula was active -- the Miner can't use the
                        # raw formula_id (it's a string, and a fresh one
                        # every mutation), but depth/node_count are real
                        # numbers it CAN mine against, e.g. "deep formulas
                        # correlate with regret." Without this, swapping
                        # the objective is a hidden variable the Miner
                        # can't explain outcomes by.
                        current_formula_depth = formula_tree.depth()
                        current_formula_node_count = formula_tree.node_count()
                        tree_ref = formula_tree
                        self.classifier.set_custom_objective(
                            lambda yp, yt, _t=tree_ref: self.formula_engine.compute_gradient(_t, yp, yt),
                            lambda yp, yt, _t=tree_ref: self.formula_engine.compute_hessian(_t, yp, yt),
                        )
            if self._current_formula_id is None and hasattr(self.classifier, "set_custom_objective"):
                self.classifier.set_custom_objective(None, None)

            # V20+.5: Causal Intervention — turn one pending hypothesis
            # into a real control/treatment comparison this stage by
            # reusing an action slot already being spent this stage, so
            # testing causality costs zero extra retrains. Only ever
            # intervenes on variables UIL actually controls (never
            # state.* — see _set_intervention_value).
            active_intervention = None
            if self.ablation["causal"]:
                # Only hypotheses with a controllable factor enter the EIG
                # pool (derived.* products decompose to their config./
                # action.penalty factor; state.* never qualifies), and the
                # factor must be a knob THIS classifier exposes — otherwise
                # apply_config would silently no-op and record a fake
                # "no effect" trial.
                testable = [
                    hyp for hyp in self.causal_engine.get_testable_hypotheses(
                        goal=self._active_goal,
                    )
                    if self._intervention_applicable(hyp)
                ]
                eig_picks = self.causal_engine.choose_intervention_eig(
                    pending=testable, budget=1,
                ) if testable else []
                active_intervention = eig_picks[0] if eig_picks else None

            intervention_results: dict[str, dict] = {}
            if active_intervention is not None and actions:
                plan = self.causal_engine.design_intervention(active_intervention)
                intervention_variable = plan.get(
                    "intervention_variable", active_intervention.variable,
                )
                treatment_value = plan["treatment"]["value"]
                if treatment_value is None:
                    # Derived-variable plan: scale the factor's live value.
                    current = self._get_intervention_value(
                        actions[0], intervention_variable,
                    )
                    scale = plan["treatment"].get("scale") or 0.5
                    if current is None or current == 0.0:
                        treatment_value = scale if scale > 1.0 else 0.0
                    else:
                        treatment_value = current * scale
                control_action = dict(actions[0])
                treatment_action = self._set_intervention_value(
                    dict(actions[0]), intervention_variable,
                    treatment_value,
                )
                control_action["_intervention_role"] = "control"
                treatment_action["_intervention_role"] = "treatment"
                actions[0] = control_action
                if len(actions) > 1:
                    actions[-1] = treatment_action
                else:
                    actions.append(treatment_action)

            # 4. Apply overrides and evaluate each action
            best_action = None
            best_action_score = -np.inf
            best_action_model = None
            # The config actually in effect when best_action was fit --
            # NOT necessarily best_action["classifier_config"]: PSO
            # actions (search/hybrid_foundation.py) never set that key at
            # all, so when one wins, the classifier was still running
            # under whatever an EARLIER candidate's apply_config() left
            # it at (or last stage's, if no candidate this stage set
            # one). That ambient/inherited config is the ground truth of
            # what produced the recorded outcome, not an empty dict.
            best_action_effective_config = {}

            parallel_action_scoring = (
                self._classifier_factory is not None
                and self._parallel_action_workers > 1
                and len(actions) > 1
            )
            best_action_result = None
            if self._race is not None:
                action_results = self._race_actions(
                    actions,
                    overrides,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    stage=stage,
                )
            else:
                action_results = self._evaluate_stage_actions(
                    actions,
                    overrides,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    stage=stage,
                )
            best_selection_score = -np.inf
            for result in action_results:
                intervention_role = result.get("intervention_role")
                if intervention_role:
                    # Causal comparison stays on the primary split: control
                    # and treatment both reach full fidelity, so this is a
                    # same-split, same-fidelity comparison by construction.
                    intervention_results[intervention_role] = result["score"]

            # Winner selection is centralized in one helper (Category-C
            # discipline): plain argmax on select_score by default -- the
            # judge fix's multi-split mean, falling back to the primary
            # score, byte-identical to the prior inline loop -- plus the
            # opt-in race incumbent-margin. select_score is absent on the
            # classic path and at cv_folds == 1, so this stays unchanged
            # there.
            best_action_result = self._pick_winner_with_margin(action_results)
            if best_action_result is not None:
                best_action = best_action_result["action"]
                best_action_score = best_action_result["score"]
                best_action_model = best_action_result["model"]
                best_action_effective_config = best_action_result["effective_config"]
                best_selection_score = best_action_result.get(
                    "select_score", best_action_score,
                )

            if best_action is None:
                continue

            # Re-apply the winning action's effective config. The scoring
            # loop above calls apply_config() for EVERY candidate in turn
            # (so each can be fit_weighted()'d under its own config) --
            # without this, the classifier's live config is left at
            # whichever candidate was scored LAST, not the one that
            # actually won. Everything downstream that reads
            # get_current_config() (journal recording at step 9,
            # investigation's current_cfg below, and therefore every
            # "config.*" rule the Miner later learns from) depends on
            # this reflecting best_action, not a loser.
            winning_config = best_action_effective_config
            if winning_config and hasattr(self.classifier, "apply_config"):
                self.classifier.apply_config(winning_config)
            if parallel_action_scoring and best_action_result is not None:
                worker = best_action_result.get("worker_classifier")
                if worker is not None:
                    for attr in (
                        "_formulas_used_this_fit",
                        "_last_used_formula_id",
                        "_last_tournament",
                        "_last_hybrid_blend",
                    ):
                        if hasattr(worker, attr):
                            setattr(self.classifier, attr, getattr(worker, attr))

            # V20+.5: Evaluate the causal intervention, if both control
            # and treatment actions actually completed this stage. Uses
            # self._best_accuracy from BEFORE this stage's update below,
            # matching the "best proven" anchor used everywhere else.
            if active_intervention is not None and "control" in intervention_results and "treatment" in intervention_results:
                ctrl_acc = intervention_results["control"]
                trt_acc = intervention_results["treatment"]
                control_regret = max(0.0, self._best_accuracy - ctrl_acc)
                treatment_regret = max(0.0, self._best_accuracy - trt_acc)
                self.causal_engine.evaluate_intervention(
                    hypothesis_id=active_intervention.hypothesis_id,
                    control_regret=control_regret,
                    treatment_regret=treatment_regret,
                    control_accuracy=ctrl_acc,
                    treatment_accuracy=trt_acc,
                )
                multiplier = self.causal_engine.get_confidence_adjustment(
                    active_intervention.hypothesis_id
                )
                self.library.apply_causal_adjustment(
                    active_intervention.source_rule.get("conditions", []),
                    multiplier,
                )

            # M0.1 (v22 blueprint §4.1): re-score the WINNING action on the
            # extra resampled splits and replace its score with the
            # multi-split mean. Everything downstream of this point --
            # best-overall update, rolling maturity accuracies, stress,
            # regret, the journal entry the miner learns from, and
            # principle helped/hurt credit -- now measures multi-split
            # generalization instead of one fixed validation split.
            # (Selection among this stage's candidates: the classic path
            # still selects on the primary split; RACE finalists are
            # selected on the multi-split mean -- the E7 judge fix, see
            # _race_actions. The causal control/treatment comparison stays
            # single-split-internal by design in both modes.)
            # With a ROTATING within-stage split the primary score is not
            # comparable across stages (stage 3 may simply have drawn easier
            # rows), so "best stage so far" must be judged on the FIXED
            # extra folds only -- same rows for every stage. Without
            # rotation the primary split is fixed and still counts, which is
            # the original behavior.
            fold_scores = [float(best_action_score)]
            if self._extra_val_splits:
                # Judge fix: when the race already scored this winner on the
                # extra splits (to SELECT it), reuse those scores instead of
                # re-fitting -- same computation, deterministic, half the
                # cost. The classic path still computes them here.
                extra = (
                    best_action_result.get("extra_scores")
                    if best_action_result is not None
                    else None
                )
                if extra is None:
                    extra = self._score_extra_splits(
                        best_action, best_action_effective_config
                    )
                fold_scores = extra if self._rotate_val else fold_scores + extra
                best_action_score = float(np.mean(fold_scores))
            fold_spread = float(max(fold_scores) - min(fold_scores))

            # 5. Update best overall
            stage_cpu = time.perf_counter() - stage_start
            self._total_cpu_time += stage_cpu

            improved = best_action_score > self._best_accuracy
            # _best_accuracy is now this run's own, starting at 0.0 (bug
            # #26), so any stage scoring above zero improves and stage 1
            # effectively always sets the model. This fallback -- which
            # existed to stop a champion-blocked run returning model=None
            # and breaking predict() (bug #16) -- is therefore near-dead,
            # but it is kept for the degenerate case of a run whose every
            # stage scores exactly 0.0. Cheap, and still the right answer.
            need_model = self._best_model is None and best_action_model is not None
            if improved or need_model:
                if improved:
                    self._best_accuracy = best_action_score
                self._best_model = best_action_model
                self._best_feature_weights = best_action["feature_weights"]
                self._best_state = state.copy()
                self._best_config = winning_config

                preds = self.classifier.predict(best_action_model, X_val)
                self._best_f1 = float(f1_score(y_val, preds, average="weighted"))
                # Macro-F1 (unweighted mean over classes) surfaces imbalance
                # that accuracy AND weighted-F1 both hide: on a 76%-majority
                # set, a model weak on the minority class still scores high
                # on both. macro-F1 is the honest metric there (Finding 4).
                self._best_macro_f1 = float(
                    f1_score(y_val, preds, average="macro", zero_division=0)
                )

            # Stagnation tracking for the evolution trigger (below).
            if improved:
                self._stages_since_improvement = 0
            else:
                self._stages_since_improvement += 1

            # Rolling recent accuracy for maturity (last 10 stages).
            self._recent_stage_accuracies.append(best_action_score)
            if len(self._recent_stage_accuracies) > 10:
                self._recent_stage_accuracies = self._recent_stage_accuracies[-10:]

            # 6. Get feature importance feedback
            importance = self.classifier.get_feature_importance(
                best_action_model, n_features
            )

            # 7. Compute stress and regret
            decision_stress = self.stress_engine.compute_decision_stress(
                previous_accuracy=self._previous_accuracy,
                current_accuracy=best_action_score,
                cpu_time_this_stage=stage_cpu,
                total_cpu_time=self._total_cpu_time,
                stages_since_best=self.stress_engine.stages_since_best_improved,
            )

            miner_r2 = self.miner.get_last_r2_score()
            system_stress = self.stress_engine.update_system_stress(
                decision_stress=decision_stress,
                current_accuracy=best_action_score,
                miner_r2_score=miner_r2,
            )

            regret = self.stress_engine.compute_regret(best_action_score)

            # Category-D telemetry (audit Part E): log the marginal
            # distribution of each component of the hand-weighted stress
            # blend. Purely observational. The immediate payoff is spotting
            # a DEAD component: if e.g. explanation_failure has std ~0
            # across a run, its 0.20 weight contributes nothing regardless
            # of value, and the blend is effectively 3-term — exactly the
            # kind of finding needed before any of these weights is touched.
            for _cname, _cval in getattr(
                self.stress_engine, "_last_pressure_components", {}
            ).items():
                self.guard_telemetry.record_signal(
                    f"stress_component.{_cname}", _cval,
                )

            self._max_stress_reached = max(
                self._max_stress_reached, system_stress
            )

            # 8. Investigation (if regret > 0)
            # V20+.4: Full investigation engine with C.1-C.5 strategies
            investigation = None
            if regret > 0:
                best_state_record = self.journal.get_state_at_best(
                    self.dataset_name
                )
                investigation = self.stress_engine.build_investigation(
                    current_state=state,
                    best_state=best_state_record,
                    regret=regret,
                )

                # V20+.4: Execute investigation strategy
                if investigation is not None:
                    strategy = self.investigation_engine.select_strategy(
                        system_stress=system_stress,
                        regret=regret,
                        investigation_count=self.investigation_engine.get_investigation_count(),
                    )

                    # Get best config for comparison
                    best_action_record = self.journal.get_action_at_best(self.dataset_name)
                    best_config_at_best = (
                        best_action_record.get("classifier_config", {})
                        if best_action_record else {}
                    )
                    current_cfg = {}
                    if hasattr(self.classifier, 'get_current_config'):
                        current_cfg = self.classifier.get_current_config()

                    inv_result = self.investigation_engine.execute_investigation(
                        strategy=strategy,
                        current_config=current_cfg,
                        best_config=best_config_at_best,
                        current_weights=best_action["feature_weights"],
                        best_weights=(
                            self._best_feature_weights
                            if self._best_feature_weights is not None
                            else best_action["feature_weights"]
                        ),
                        investigation_data=investigation,
                    )

                    # Apply the proposed config if the strategy recommends it
                    proposed_config = inv_result.get("proposed_config", {})
                    if proposed_config and hasattr(self.classifier, 'apply_config'):
                        self.classifier.apply_config(proposed_config)

                    # Merge investigation result into investigation record
                    investigation["strategy"] = inv_result.get("strategy", "")
                    investigation["rationale"] = inv_result.get("rationale", "")
                    investigation["investigation_cpu_time"] = inv_result.get(
                        "investigation_cpu_time", 0.0
                    )

            # 9. Record to journal (pure numbers)
            # V20: include classifier config and formula info.
            # Use winning_config directly (re-applied right after the
            # scoring loop, above) rather than re-reading
            # get_current_config() -- that call can also reflect an
            # investigation's proposed_config for the NEXT stage (step 8
            # above), which must not leak into the record of what
            # actually produced THIS stage's outcome.
            classifier_config = winning_config

            action_record = {
                "penalty": best_action["penalty"],
                "description": best_action.get("description", ""),
            }
            # V20: Include classifier config in action
            if classifier_config:
                action_record["classifier_config"] = classifier_config
            # V20: Include formula metadata
            if self._current_formula_id:
                action_record["formula_id"] = self._current_formula_id
                # V20+.14: numeric proxies so the Miner can explain
                # outcomes by formula complexity (see selection block
                # above) -- formula_id itself is a string and unusable.
                action_record["formula_depth"] = current_formula_depth
                action_record["formula_node_count"] = current_formula_node_count

            # V20+.12: carry forward into next stage's ExecutionContext.
            self._last_executed_action = {
                k: v for k, v in action_record.items()
                if isinstance(v, (int, float))
            }
            self._last_executed_config = classifier_config

            self.journal.record(
                dataset=self.dataset_name,
                fold=self.fold,
                stage=stage,
                classifier=self.classifier.name,
                state=state,
                action=action_record,
                outcome={
                    "accuracy": round(best_action_score, 4),
                    "f1_score": round(self._best_f1, 4),
                    "macro_f1": round(self._best_macro_f1, 4),
                    "cpu_time": round(stage_cpu, 3),
                    "model_complexity": self.classifier.get_model_complexity(
                        best_action_model
                    ),
                    "recovery_cost": self.stress_engine.get_recovery_cost_breakdown(),
                    # M0.1: fold provenance -- accuracy above is the mean
                    # of these per-split scores (single element = legacy).
                    "fold_scores": [round(s, 4) for s in fold_scores],
                    "fold_spread": round(fold_spread, 4),
                },
                decision_stress=decision_stress,
                system_stress=system_stress,
                regret=regret,
                investigation=investigation,
            )

            # Research step 2 (PBT-style default-path architecture variation,
            # opt-in): fit ONE architecture-perturbed probe and journal it as
            # evidence, so the miner finally sees architecture knobs VARY
            # instead of being constant (F4's root cause). Never competes for
            # the stage winner, so it cannot regress accuracy. Skipped in
            # race mode, which supplies its own architecture variation.
            if (
                getattr(self.cfg, "explore_architecture", False)
                and self._race is None
            ):
                probe_rec = self._architecture_exploration_probe(
                    stage, best_action, overrides, state,
                    X_train, y_train, X_val, y_val, system_stress,
                )
                if probe_rec is not None:
                    self._observability_meta.setdefault(
                        "arch_probes", []
                    ).append(probe_rec["description"])

            # EXPERIMENTAL (held-out principle gate): score the winning
            # model on the never-touched held-out set so principle credit
            # below can be judged on generalization, not the validation
            # score the search already maximizes.
            heldout_score = None
            if self._heldout_gate and self._X_heldout is not None:
                _ho_preds = self.classifier.predict(best_action_model, self._X_heldout)
                heldout_score = float(accuracy_score(self._y_heldout, _ho_preds))

            # 10. Track triggered principles (for library feedback)
            triggered = self.executor.get_triggered_principles()
            for pid in triggered:
                if heldout_score is not None:
                    # A principle "helped" only if it did not hurt held-out
                    # accuracy vs the previous stage -- the generalization
                    # test, not the validation-split test.
                    helped = heldout_score >= self._previous_heldout
                else:
                    helped = best_action_score >= self._previous_accuracy
                # V20+.1: Pass dataset_name for per-dataset tier tracking
                self.library.record_application(
                    pid, helped, self.classifier.name,
                    dataset_name=self.dataset_name,
                )
                # M0.4: the same helped signal drives maturity, so PSO
                # detachment is earned by principles that demonstrably
                # predict wins (see hybrid_foundation.update_maturity).
                self.foundation.record_principle_outcome(helped)
            # Part 16: Carry triggered principles forward so co-activation
            # bonuses can be applied on the NEXT stage's principle retrieval.
            self._last_triggered_principles = triggered

            # V20.3: Record formula outcome so retirement evidence
            # actually accumulates (previously orphaned — set but never
            # recorded against).
            if self._current_formula_id:
                self.formula_library.record_result(
                    formula_id=self._current_formula_id,
                    accuracy=best_action_score,
                    helped=best_action_score >= self._previous_accuracy,
                    dataset_fingerprint=self.dataset_name,
                )

            # V20+.18: parallel read-back for SymbolicFormulaAdapter --
            # formula *selection* happens inside fit_weighted() (the
            # adapter), not in this loop, so the optimizer reads the
            # choice back out by convention to record its outcome. Same
            # pattern KNN's distance-metric repurposing already uses.
            last_symbolic_id = getattr(self.classifier, "_last_used_formula_id", None)
            if self.symbolic_library is not None:
                used_ids = getattr(self.classifier, "_formulas_used_this_fit", None) or []
                if not used_ids and last_symbolic_id:
                    used_ids = [last_symbolic_id]
                for fid in dict.fromkeys(used_ids):
                    self.symbolic_library.record_result(
                        formula_id=fid,
                        accuracy=best_action_score,
                        helped=best_action_score >= self._previous_accuracy,
                        dataset_fingerprint=self.dataset_name,
                    )

            # 11. Update search strategy with feedback
            center = np.mean(
                [a["feature_weights"] for a in actions], axis=0
            )
            diversity = float(np.mean(
                [np.mean(np.abs(a["feature_weights"] - center)) for a in actions]
            ))
            # The genuinely worst-scoring candidate THIS stage. Previously
            # only the winner was reported back, so the search inferred
            # "worst" as "the winning weights of the lowest-scoring stage" --
            # a decent vector, not a bad one (and on stage 1, literally the
            # same vector as best). Its "Opposite of worst" action was
            # therefore stepping away from something good, and the journal
            # recorded that wrong label for the miner to learn from.
            worst_weights = None
            worst_score = None
            if action_results:
                _worst = min(action_results, key=lambda r: r["score"])
                if _worst["score"] < best_action_score:
                    worst_weights = _worst["action"]["feature_weights"]
                    worst_score = float(_worst["score"])
            self.search.update_feedback(
                importance=importance,
                best_weights=best_action["feature_weights"],
                best_score=best_action_score,
                entropy=np.clip(diversity / 0.5, 0.0, 1.0),
                worst_weights=worst_weights,
                worst_score=worst_score,
            )

            # 12. Mine for principles (frequency scales with stress)
            mine_interval = max(1, int(5 * (1.0 - system_stress)))
            self._last_stage_mining = None
            if stage % mine_interval == 0 or stage == stages - 1:
                all_entries = self.journal.get_entries()
                principles_before = len(self.library.get_principles())

                # V20+.7: Apply journal forgetting — recency-weighted window
                mining_entries, _ = self.journal_forgetting.prepare_for_mining(
                    all_entries
                )

                # Within-dataset mining (stronger causal evidence).
                # V20+.16: mining itself is cheap analysis over journal
                # entries -- only APPLYING the rules is gated, so
                # "principles OFF" means discovered rules are never
                # written anywhere, not that mining never runs.
                within_rules = self.miner.mine_within_dataset(
                    mining_entries, self.dataset_name, system_stress
                )
                cross_rules: list = []
                if within_rules and self.ablation["principles"]:
                    self.library.update_principles(
                        within_rules,
                        min_evidence=self._promotion_min_evidence(
                            len(mining_entries),
                        ),
                    )
                    # V20+.3: Add new rules as nodes in the network
                    if self.ablation["network"]:
                        self.network.import_from_principles(within_rules)
                    # V20+.5: Form causal hypotheses from new rules
                    if self.ablation["causal"]:
                        self.causal_engine.form_hypotheses_from_rules(within_rules)
                    # Part 12: Version tracking for each updated principle
                    for rule in within_rules:
                        self.principle_versioning.record_version(rule)

                # Cross-dataset mining (correlational, needs more evidence)
                if len(mining_entries) >= 50:
                    cross_rules = self.miner.mine_cross_dataset(
                        mining_entries, system_stress
                    )
                    if cross_rules and self.ablation["principles"]:
                        # Part 5: Cross-dataset rules represent weaker causal
                        # evidence — discount confidence before library update
                        # so they must prove themselves further to affect
                        # principle selection over within-dataset rules.
                        for _r in cross_rules:
                            _r["confidence"] = round(_r.get("confidence", 0.5) * 0.75, 4)
                        self.library.update_principles(
                            cross_rules,
                            min_evidence=self._promotion_min_evidence(
                                len(mining_entries),
                            ),
                        )
                        if self.ablation["network"]:
                            self.network.import_from_principles(cross_rules)

                # V20.3: Cognitive novelty — under high stress, mutate the
                # current best formula instead of relying on PSO alone
                # (Part 15: novelty should come from formula invention,
                # not just random search). Validated before being stored
                # so a broken mutation never reaches set_custom_objective.
                # V20+.14: gated on the incubation queue, not just stress --
                # every mutation forces MIN_EVIDENCE_BOOTSTRAP real training
                # trials under a possibly-bad objective (see
                # formula_library.py), and a bad mutation raising stress
                # would otherwise spawn more mutations with no limit,
                # a runaway feedback loop discovered empirically this
                # session. Capping how many can be incubating at once
                # throttles the birth rate directly, independent of stress.
                if (
                    self.ablation["formulas"]
                    and self._want_new_structure(system_stress, stage, "loss")
                    and len(self.formula_library.get_incubating_formulas())
                    < FormulaLibrary.MAX_INCUBATING
                ):
                    # V20+.15/.17: tournament-selected, lineage-stratified
                    # mutation parent, NOT get_best_formula() -- using the
                    # single best formula as the mutation base every time
                    # turns evolution into hill-climbing along one lineage
                    # (see formula_library.select_mutation_parent
                    # docstring). progress feeds the stratification decay.
                    base_formula = self.formula_library.select_mutation_parent(
                        self.formula_engine.rng,
                        progress=stage / max(1, stages - 1),
                    )
                    if base_formula is not None:
                        base_tree = self.formula_library.get_formula_tree(
                            base_formula["formula_id"]
                        )
                        if base_tree is not None:
                            mutated = None
                            mutation_type = "engine"
                            # Default: offline FormulaEngine only. LLM is
                            # opt-in via cfg.formula_offline_only=False
                            # (and still needs an API key).
                            use_llm = (
                                not getattr(self.cfg, "formula_offline_only", True)
                                and self._llm_mutator.has_proposer
                            )
                            if use_llm:
                                proposals = self._llm_mutator.propose(
                                    fingerprint=self._dataset_fingerprint,
                                    cells=[p for p in self.library.get_principles()[:5]],
                                    n_proposals=1,
                                )
                                if proposals and proposals[0].payload.get("tree"):
                                    from figpso_uil.cognition.formula_engine import FormulaNode
                                    try:
                                        mutated = FormulaNode.from_dict(proposals[0].payload["tree"])
                                        mutation_type = proposals[0].proposer
                                    except Exception:
                                        mutated = None
                            if mutated is None:
                                mutated, mutation_type = self.formula_engine.mutate(
                                    base_tree, system_stress
                                )
                            is_valid, _ = self.formula_engine.validate(mutated)
                            if is_valid:
                                self.formula_library.add_formula(
                                    name=f"{base_formula['name']}_mut",
                                    tree=mutated,
                                    origin=f"mutation_of_{base_formula['formula_id']}",
                                    mutation_type=mutation_type,
                                    created_at_stage=stage,
                                )

                # V20+.18: same generation-advance, extended to the
                # SymbolicFormula classifier's own population (duck-typed
                # via self.symbolic_engine/self.symbolic_library -- None
                # for every other classifier, so this block is a no-op
                # for them). fit_weighted() never mutates its own
                # population; this is the one place it advances, exactly
                # like the loss-formula block above. feature_bias lets
                # PSO's per-feature relevance signal shape which features
                # new formulas reference (see search/pso_search.py).
                if (
                    self.ablation["formulas"]
                    and self.symbolic_engine is not None
                    and self.symbolic_library is not None
                    and self._want_new_structure(system_stress, stage, "symbolic")
                    and len(self.symbolic_library.get_incubating_formulas())
                    < FormulaLibrary.MAX_INCUBATING
                ):
                    # Opt-in parallel pyramid (cfg.parallel_formula_pyramid):
                    # N isolated FormulaEngines + randomized DT pyramid merge.
                    # Default OFF — single-engine path unchanged.
                    if getattr(self.cfg, "parallel_formula_pyramid", False):
                        self._admit_symbolic_pyramid(
                            system_stress=system_stress,
                            stage=stage,
                            stages=stages,
                            n_features=n_features,
                            feature_weights=best_action.get("feature_weights"),
                            X_train=X_train,
                            y_train=y_train,
                        )
                    else:
                        # Offline trial-and-error on this dataset (default).
                        # No API key. Mutate → score on residual → keep.
                        if getattr(self.cfg, "formula_offline_only", True):
                            from figpso_uil.cognition.offline_formula_search import (
                                trial_and_error_search,
                            )
                            parents = []
                            sym_base = self.symbolic_library.select_mutation_parent(
                                self.symbolic_engine.rng,
                                progress=stage / max(1, stages - 1),
                            )
                            if sym_base is not None:
                                bt = self.symbolic_library.get_formula_tree(
                                    sym_base["formula_id"]
                                )
                                if bt is not None:
                                    parents.append(bt)
                            trials = trial_and_error_search(
                                X_train, y_train,
                                seed=int(self.seed) ^ (stage * 0x9E37) ^ 0x0FF1,
                                n_features=n_features,
                                n_trials=int(getattr(
                                    self.cfg, "formula_offline_trials", 48)),
                                n_keep=1,
                                parents=parents or None,
                                feature_bias=best_action.get("feature_weights"),
                                stress=float(system_stress),
                            )
                            if trials:
                                parent_meta = sym_base or {
                                    "formula_id": "seed",
                                    "name": "offline_trial",
                                }
                                self._admit_symbolic_mutant(
                                    tree=trials[0].tree,
                                    mutation_type=f"offline_trial:{trials[0].mutation_type}",
                                    parent=parent_meta,
                                    stage=stage,
                                    X_train=X_train,
                                    y_train=y_train,
                                    feature_weights=best_action.get(
                                        "feature_weights"
                                    ),
                                )
                        else:
                            sym_base = self.symbolic_library.select_mutation_parent(
                                self.symbolic_engine.rng,
                                progress=stage / max(1, stages - 1),
                            )
                            if sym_base is not None:
                                sym_base_tree = self.symbolic_library.get_formula_tree(
                                    sym_base["formula_id"]
                                )
                                if sym_base_tree is not None:
                                    sym_mutated, sym_mutation_type = self._neural_pick_classifier_mutation(
                                        sym_base_tree, system_stress, n_features,
                                        best_action["feature_weights"],
                                    )
                                    sym_is_valid, _ = self.symbolic_engine.validate_classifier(
                                        sym_mutated, n_features=n_features
                                    )
                                    if sym_is_valid:
                                        self._admit_symbolic_mutant(
                                            tree=sym_mutated,
                                            mutation_type=sym_mutation_type,
                                            parent=sym_base,
                                            stage=stage,
                                            X_train=X_train,
                                            y_train=y_train,
                                            feature_weights=best_action.get(
                                                "feature_weights"
                                            ),
                                        )

                # V20+.3: Discover relationships from co-activation patterns.
                # Attach the current best formula so edges actually carry
                # one for traverse() to execute (see editable_network.py).
                if self.ablation["network"] and len(_activation_history) >= 5:
                    best_formula = (
                        self.formula_library.get_best_formula(
                            dataset_fingerprint=self.dataset_name
                        )
                        if self.ablation["formulas"] else None
                    )
                    # Filter to THIS fold so "stage" is unique within the
                    # entries discover_relationships() matches against --
                    # stage resets to 0 each fold, so an unfiltered
                    # cross-fold list would let a different fold's stage-3
                    # entry silently stand in for this fold's stage-3.
                    fold_entries = [
                        e for e in self.journal.get_dataset_entries(self.dataset_name)
                        if e.get("fold") == self.fold
                    ]
                    # Base-rate: fraction of this fold's stages where the
                    # system actually improved (regret <= 0). Passed to
                    # discover_relationships() so edge-type thresholds are
                    # lift-over-base-rate rather than absolute, preventing
                    # every edge from becoming "contradicts" when the
                    # overall success rate is low (e.g. 8% on hard datasets).
                    _fold_regrets = [e.get("regret", 1.0) for e in fold_entries]
                    _net_base_rate = (
                        sum(1 for r in _fold_regrets if r <= 0) / len(_fold_regrets)
                        if _fold_regrets else None
                    )
                    self.network.discover_relationships(
                        fold_entries,
                        _activation_history[-25:],
                        formula_id=best_formula["formula_id"] if best_formula else None,
                        base_rate=_net_base_rate,
                    )

                cross_count = len(cross_rules)

                rule_previews = [
                    {
                        "conditions": r.get("conditions", [])[:3],
                        "confidence": r.get("confidence"),
                        "predicted_regret": r.get("predicted_regret"),
                        "evidence_status": r.get("evidence_status", "hypothesis"),
                    }
                    for r in (within_rules or [])[:3]
                ]
                self._last_stage_mining = {
                    "mined_this_stage": True,
                    "journal_entries_used": len(mining_entries),
                    "within_rules_found": len(within_rules or []),
                    "cross_rules_found": cross_count,
                    "principles_before": principles_before,
                    "principles_after": len(self.library.get_principles()),
                    "miner_r2": round(self.miner.get_last_r2_score(), 4),
                    "rule_previews": rule_previews,
                    "network_nodes": self.network.node_count,
                    "causal_hypotheses": len(self.causal_engine._hypotheses),
                }

            # V20+.3: Record traversal outcome for edge feedback
            if self.ablation["network"]:
                helped = best_action_score >= self._previous_accuracy
                activated = network_result.get("activated_nodes", [])
                if activated:
                    self.network.record_traversal_outcome(activated, helped)

                # V20+.3: Periodic edge decay + working-graph prune (every 10 stages)
                if stage > 0 and stage % 10 == 0:
                    self.network.apply_edge_decay(
                        min_traversals=5,
                        archive=self.knowledge_archive,
                        dataset=self.dataset_name,
                    )
                # Learning-preserving formula prune on a SHORT cadence so the
                # accelerated evolution from stagnation-triggered births (b)
                # can't bloat the working pool even in short runs. Keeps
                # seeds + proven formulas; archives the weakest mutants to the
                # (queryable) KnowledgeArchive. Retired formulas archived too.
                if stage > 0 and stage % 5 == 0:
                    for _lib in (self.symbolic_library, self.formula_library):
                        if _lib is None:
                            continue
                        if hasattr(_lib, "prune_retired_to_archive"):
                            _lib.prune_retired_to_archive(self.dataset_name)
                        if hasattr(_lib, "prune_to_cap"):
                            _lib.prune_to_cap(self.dataset_name)
                # Size-triggered graph prune: the stage%15 cadence below
                # never fires in short runs, so across many short runs the
                # shared working graph grew unboundedly (observed 49k edges
                # -> traverse_with_attention exploded, hanging stages for
                # minutes). Prune to cap whenever oversized, on the same
                # short cadence as the formula prune. Everything archived,
                # nothing lost.
                if (
                    stage > 0 and stage % 5 == 0
                    and len(self.network._edges) > self.GRAPH_PRUNE_TRIGGER_EDGES
                ):
                    _recent: set[str] = set()
                    for _, nodes in _activation_history[-12:]:
                        _recent.update(nodes)
                    self.network.prune_working_graph(
                        self.knowledge_archive,
                        max_edges=self.GRAPH_MAX_EDGES,
                        max_nodes=self.GRAPH_MAX_NODES,
                        protect_nodes=_recent,
                        dataset=self.dataset_name,
                        system_stress=system_stress,
                    )
                if stage > 0 and stage % 15 == 0:
                    recent_nodes: set[str] = set()
                    for _, nodes in _activation_history[-12:]:
                        recent_nodes.update(nodes)
                    self.network.consolidate_duplicate_nodes(
                        self.knowledge_archive, dataset=self.dataset_name,
                    )
                    self.network.prune_working_graph(
                        self.knowledge_archive,
                        max_edges=self.GRAPH_MAX_EDGES,
                        max_nodes=self.GRAPH_MAX_NODES,
                        protect_nodes=recent_nodes,
                        dataset=self.dataset_name,
                        system_stress=system_stress,
                    )
                    lib = self.symbolic_library or self.formula_library
                    if hasattr(lib, "prune_retired_to_archive"):
                        lib.prune_retired_to_archive(self.dataset_name)
                    self.working_memory_pruner.prune_all(
                        journal=self.journal,
                        library=self.library,
                        causal_engine=self.causal_engine,
                        archive=self.knowledge_archive,
                        dataset=self.dataset_name,
                    )
                    events = self.knowledge_archive._data.get("datasets", {}).get(
                        self.dataset_name, {},
                    ).get("graph_prune_events", [])
                    if events:
                        self._last_prune = {**events[-1], "stage": stage}

            # 13. Check for failure → failsafe (LAST RESORT only)
            if (
                self.failsafe is not None
                and not self.failsafe.was_already_run()
                and self.stress_engine.has_failed(
                    miner_r2=miner_r2,
                    principles_exhausted=self.library.all_principles_exhausted(),
                )
            ):
                failsafe_weights = (
                    self._best_feature_weights
                    if self._best_feature_weights is not None
                    else self._base_guidance
                )
                diagnosis = self.failsafe.diagnose(
                    X, y,
                    failsafe_weights,
                    self.classifier.name,
                    self.seed,
                )
                self.journal.record_failsafe(self.dataset_name, diagnosis)

                # If classifier mismatch was found, record as lesson
                if diagnosis.get("diagnosis") == "classifier_mismatch":
                    preferred = diagnosis.get("recommendation")
                    self.journal.update_dataset_memory(
                        self.dataset_name,
                        self._max_stress_reached,
                        self._best_accuracy,
                        preferred_classifier=preferred,
                    )

                # Reset and continue with accumulated knowledge
                self.stress_engine.reset(keep_knowledge=True)

            # 14. Update previous accuracy for next iteration
            self._previous_accuracy = best_action_score
            if heldout_score is not None:
                self._previous_heldout = heldout_score
            stages_completed = stage + 1

            if (
                target_accuracy is not None
                and self._best_accuracy >= target_accuracy
            ):
                target_reached = True
                self._emit_observability(
                    on_stage,
                    event="finished",
                    stage=stage,
                    stages_total=max_stages,
                    state=state,
                    principles=principles,
                    overrides=overrides,
                    network_result=network_result,
                    allocation=allocation,
                    reasoning_actions=reasoning_actions,
                    pso_actions=pso_actions,
                    triggered=triggered,
                    decision_stress=decision_stress,
                    system_stress=system_stress,
                    regret=regret,
                    best_action_score=best_action_score,
                    miner_r2=miner_r2,
                    investigation=investigation,
                    active_intervention=active_intervention,
                    stage_cpu=stage_cpu,
                    current_formula_depth=current_formula_depth,
                    current_formula_node_count=current_formula_node_count,
                    episodic_hint=None,
                    fingerprint=fingerprint,
                    prior_stress=0.0,
                    target_reached=True,
                )
                break

            # webui/ + terminal live feed (see cognition/live_feed.py).
            self._emit_observability(
                on_stage,
                event="stage",
                stage=stage,
                stages_total=max_stages,
                state=state,
                principles=principles,
                overrides=overrides,
                network_result=network_result,
                allocation=allocation,
                reasoning_actions=reasoning_actions,
                pso_actions=pso_actions,
                triggered=triggered,
                decision_stress=decision_stress,
                system_stress=system_stress,
                regret=regret,
                best_action_score=best_action_score,
                miner_r2=miner_r2,
                investigation=investigation,
                active_intervention=active_intervention,
                stage_cpu=stage_cpu,
                current_formula_depth=current_formula_depth,
                current_formula_node_count=current_formula_node_count,
                episodic_hint=None,
                fingerprint=fingerprint,
                prior_stress=0.0,
            )

        # ── End of optimization ──

        # Update dataset memory
        self.journal.update_dataset_memory(
            self.dataset_name,
            self._max_stress_reached,
            self._best_accuracy,
        )

        # V20+.8: Store this episode for future warm-starts.
        # Gated with retrieval: an "episodic OFF" arm must not leave an
        # episode behind for a later arm to retrieve (bug #30b's leak
        # class), otherwise the switch only half-removes the subsystem.
        if self.ablation["episodic"]:
            self.episodic_attention.store_episode(
                fingerprint=fingerprint,
                trajectory={
                    "max_stress": self._max_stress_reached,
                    "final_accuracy": self._best_accuracy,
                    "stages_to_converge": stages,
                    "regret_sum": self.stress_engine.cumulative_regret
                    if hasattr(self.stress_engine, "cumulative_regret") else 0.0,
                },
                resolution={
                    "best_config": self._best_config or {},
                    "best_weights": (
                        self._best_feature_weights.tolist()
                        if self._best_feature_weights is not None else []
                    ),
                    "principle_ids": [
                        p["principle_id"] for p in self.library.get_principles()
                    ],
                    # V20+.18: lets a future similar-regime dataset's
                    # SymbolicFormulaAdapter.set_episodic_hint() warm-start
                    # from whatever formula this run last used (see
                    # episodic_attention.py's _blend_resolutions).
                    "symbolic_formula_id": getattr(
                        self.classifier, "_last_used_formula_id", None
                    ),
                },
            )

        # V20+.3: Persist the graph (nodes/edges/co-activation tally) --
        # saved once per fit() rather than per-mutation, see editable_network.py.
        self.network.save()

        # Bug #26: the champion comparison happens HERE, once, instead of
        # gating every stage. Both numbers are this-run-vs-stored on the
        # same measure (the k-fold mean when cv_folds > 1). A worse run
        # simply doesn't overwrite the record -- it is no longer prevented
        # from doing its own work first.
        #
        # Deliberately NOT compared on self._X_heldout: that set is the one
        # measurement in the system nothing has selected against, and it
        # stays that way. A two-way choice would leak little, but it costs
        # nothing to keep it clean.
        #
        # Research step 6 (Dwork Thresholdout): when enabled, "did this run
        # beat the stored champion?" goes through a seeded-noise threshold
        # so repeated adaptive comparisons against the same bar do not
        # silently overfit the instrument. Deterministic given the seed.
        if self._champion_bar is None:
            beats_champion = True
        elif getattr(self.cfg, "use_thresholdout", True):
            _to = thresholdout_beats(
                float(self._best_accuracy),
                float(self._champion_bar),
                seed=self.seed,
                query_id=f"champion:{self.dataset_name}",
                tau=float(getattr(self.cfg, "thresholdout_tau", 0.01)),
                noise_scale=float(getattr(self.cfg, "thresholdout_noise", 0.005)),
            )
            beats_champion = bool(_to["beats"])
            self._observability_meta["thresholdout_champion"] = _to
        else:
            beats_champion = self._best_accuracy >= self._champion_bar
        if not beats_champion:
            self._observability_meta["champion_kept"] = {
                "stored": round(float(self._champion_bar), 4),
                "this_run": round(float(self._best_accuracy), 4),
            }
        if self._best_model is not None and self._best_accuracy > 0 and beats_champion:
            self.champion_store.save(
                self.dataset_name,
                best_accuracy=self._best_accuracy,
                best_weights=(
                    self._best_feature_weights.tolist()
                    if self._best_feature_weights is not None else None
                ),
                best_config=self._best_config or {},
                model=self._best_model if isinstance(self._best_model, dict) else None,
                formula_id=getattr(self.classifier, "_last_used_formula_id", None),
                learning_mode=(
                    self._best_model.get("learning_mode")
                    if isinstance(self._best_model, dict) else None
                ),
            )
        self.knowledge_archive.record_conclusion(
            self.dataset_name,
            (
                f"Run finished at {self._best_accuracy:.1%} val accuracy "
                f"(formula {getattr(self.classifier, '_last_used_formula_id', '?')})."
            ),
            metadata={
                "best_accuracy": self._best_accuracy,
                "stages_run": stages_completed,
            },
        )
        # Bug #27: flush the archive AFTER the champion + conclusion are
        # written into it, not before. save() used to run above the champion
        # block, so the champion only ever reached _data in memory -- it
        # survived to disk purely by accident, when some LATER component
        # (formula_library.prune_*, working_memory_pruner, the next fit())
        # happened to flush the same singleton. A process that optimized one
        # dataset and exited persisted no champion at all, which is why
        # champion-resume looked like it worked in multi-dataset benchmarks
        # and did nothing in single runs.
        self.knowledge_archive.save()

        lib = self.symbolic_library or self.formula_library
        if hasattr(lib, "prune_retired_to_archive"):
            lib.prune_retired_to_archive(self.dataset_name)

        # V21 Pipeline: Structural → Meta → Safeguards (skip when target hit —
        # post-run merge on a 5k-edge KG can block the worker for minutes).
        all_principles = self.library.get_principles()
        if (
            not target_reached
            and len(all_principles) >= 5
            and self.ablation["structural"]
        ):
            # V21.1: Encode all principles structurally
            structural_principles = self.structural_encoder.batch_encode(
                all_principles
            )

            # V21.2: Merge similar principles into meta-principles.
            # `merge()` PERSISTS to uil_meta_principles.json (bug #8), so
            # this has to be skipped, not just ignored, for the switch to
            # actually remove the subsystem.
            new_metas = (
                self.meta_merger.merge(structural_principles)
                if self.ablation["meta_principles"] else []
            )

            # V21.3: Safeguard each meta-principle before promotion
            if self.ablation["safeguards"]:
                for meta in new_metas:
                    check = self.safeguards.check_promotion(
                        meta, structural_principles, self.contradiction_detector
                    )
                    # Log but don't discard — investigation can revisit

            # Part 12: Detect contradictions in the full principle set
            contradictions = self.contradiction_detector.detect_contradictions(
                all_principles
            )

        # Feature selection: PSO weights + formula tree leaves (SymbolicFormula)
        if (
            self._best_model is not None
            and hasattr(self.classifier, "get_selected_feature_indices")
        ):
            formula_feats = self.classifier.get_selected_feature_indices(self._best_model)
            if self._best_feature_weights is not None:
                threshold = max(0.05, np.percentile(self._best_feature_weights, 30))
                pso_feats = np.where(self._best_feature_weights > threshold)[0]
                selected = np.unique(np.concatenate([formula_feats, pso_feats]))
            else:
                selected = formula_feats
            if len(selected) == 0:
                selected = formula_feats if len(formula_feats) else np.arange(n_features)
        elif self._best_feature_weights is not None:
            threshold = max(0.05, np.percentile(self._best_feature_weights, 30))
            selected = np.where(self._best_feature_weights > threshold)[0]
            if len(selected) == 0:
                selected = np.arange(n_features)
        else:
            selected = np.arange(n_features)

        # M5: lab report from event stream
        try:
            report_path = generate_lab_report(
                run_id=f"{self.dataset_name}-{self.seed}",
                dataset_name=self.dataset_name,
                route_decision=getattr(self, "_route_decision", None),
                fingerprint=getattr(self, "_dataset_fingerprint", None),
                version_store=self._version_store,
            )
            self._observability_meta["lab_report"] = report_path
        except Exception:
            pass

        # M3: record routing outcome for episodic memory
        if hasattr(self, "_route_controller") and self._best_accuracy is not None:
            self._route_controller.observe(
                self._dataset_fingerprint,
                self._route_decision.get("strategy", "grad_fit"),
                float(self._best_accuracy),
            )

        # Audit Part E: persist the run's guard-bind observations (accumulates
        # across runs into uil_guard_telemetry.json) and surface this run's
        # snapshot for the web UI / caller. Pure observation, never gated.
        try:
            self.guard_telemetry.flush()
            self._observability_meta["guard_telemetry"] = (
                self.guard_telemetry.snapshot()
            )
        except Exception:
            pass
        if getattr(self, "_utility_admit_stats", None) is not None:
            self._observability_meta["utility_admit"] = dict(
                self._utility_admit_stats
            )

        return {
            "model": self._best_model,
            "accuracy": self._best_accuracy,
            "f1_score": self._best_f1,
            "macro_f1": self._best_macro_f1,
            "feature_weights": self._best_feature_weights,
            "selected_features": selected,
            "n_selected": len(selected),
            "stages_run": stages_completed,
            "max_stress": self._max_stress_reached,
            "principles_discovered": len(self.library.get_principles()),
            "target_accuracy": target_accuracy,
            "target_reached": target_reached if target_accuracy is not None else None,
            "network_summary": self.network.get_summary(),
            "causal_summary": self.causal_engine.get_summary(),
            "episodic_count": self.episodic_attention.episode_count,
            "foundation_stage": self.foundation.get_summary(),
            "meta_principles": self.meta_merger.get_summary(),
            "safeguards": self.safeguards.get_summary(),
            "versioning": self.principle_versioning.get_evolution_summary(),
        }

    # ── Predict ──────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the best model.

        Applies the SAME missing-data + normalization transforms fitted at
        step 0 of fit(). Without this, the model is trained in imputed /
        normalized space (v22 §2.2) but served raw — a train/serve
        distribution mismatch that silently corrupts every prediction on
        unseen data (bank_marketing: trained on mean 0.57/std 3.11, served
        mean 0.0/std 1.0).
        """
        if self._best_model is None:
            raise RuntimeError("Must call fit() before predict().")
        X = self._perception_forward(X)
        return self.classifier.predict(self._best_model, X)

    def _perception_forward(self, X: np.ndarray) -> np.ndarray:
        """Train-serve perception chain: impute (+indicators) then normalize."""
        if getattr(self, "_missing_data", None) is not None:
            X = self._missing_data.forward(X)
        if getattr(self, "_normalization", None) is not None:
            X = self._normalization.forward(X)
        return X

    # ── Internal Helpers ─────────────────────────────────────────────

    def _emit_observability(
        self,
        on_stage: Callable[[dict], None] | None,
        **snapshot_kwargs,
    ) -> dict:
        """Publish snapshot to shared live feed + optional WebSocket callback."""
        snap = self._build_observability_snapshot(**snapshot_kwargs)
        publish_live_feed(
            snap,
            meta=dict(self._observability_meta),
            filepath=(
                self._state_path("uil_live_feed.jsonl")
                if self._state_dir is not None
                else None
            ),
        )
        if on_stage is not None:
            on_stage(snap)
        return snap

    def _build_observability_snapshot(
        self,
        *,
        event: str,
        stage: int,
        stages_total: int,
        state: dict,
        principles: list,
        overrides: dict,
        network_result: dict,
        allocation: dict,
        reasoning_actions: list,
        pso_actions: list,
        triggered: list,
        decision_stress: float,
        system_stress: float,
        regret: float,
        best_action_score: float,
        miner_r2: float,
        investigation: dict | None,
        active_intervention,
        stage_cpu: float,
        current_formula_depth,
        current_formula_node_count,
        episodic_hint: dict | None,
        fingerprint: dict,
        prior_stress: float,
        target_reached: bool = False,
    ) -> dict:
        """Rich per-stage payload for the web UI (decision trace + memory)."""
        activated = network_result.get("activated_nodes", [])
        net_overrides = network_result.get("overrides", {})
        merged_keys = sorted(set(list(overrides.keys()) + list(net_overrides.keys())))

        triggered_detail = []
        for pid in triggered:
            p = next((x for x in principles if x.get("principle_id") == pid), None)
            if p:
                triggered_detail.append({
                    "id": pid,
                    "tier": p.get("tier"),
                    "confidence": round(p.get("confidence", 0), 3),
                    "evidence_status": p.get("evidence_status", "hypothesis"),
                    "conditions": p.get("conditions", [])[:2],
                })
            else:
                triggered_detail.append({"id": pid})

        pipeline = [
            {
                "layer": "ROUTE",
                "icon": "🧭",
                "title": "Domain router (step 0)",
                "detail": (
                    f"{getattr(self, '_route_decision', {}).get('domain', '?')}"
                    f" -> {getattr(self, '_route_decision', {}).get('strategy', '?')}"
                    f" ({getattr(self, '_route_decision', {}).get('source', 'heuristic')})"
                    if event == "started"
                    else "Using routed classifier from fit() step 0"
                ),
                "data": {
                    "fingerprint_id": self._observability_meta.get("fingerprint_id"),
                    "route": getattr(self, "_route_decision", {}),
                    "linear_probe": fingerprint.get("linear_probe_acc"),
                    "formula_probe_r2": fingerprint.get("formula_probe_r2"),
                    "sequential": fingerprint.get("sequential"),
                },
            },
            {
                "layer": "STATE",
                "icon": "📊",
                "title": "Observe dataset + trajectory",
                "detail": (
                    f"n={state.get('n_samples', fingerprint.get('n_samples', '?'))}, "
                    f"features={state.get('n_features', fingerprint.get('n_features', '?'))}, "
                    f"sparsity={state.get('sparsity', fingerprint.get('sparsity', '?'))}"
                    if state else
                    f"Fingerprint: {fingerprint.get('n_samples')}×{fingerprint.get('n_features')}, "
                    f"prior_stress={prior_stress:.2f}"
                ),
            },
            {
                "layer": "EPISODIC",
                "icon": "🧠",
                "title": "Episodic memory warm-start",
                "detail": (
                    "Retrieved similar past run — blending historical config"
                    if episodic_hint
                    else (
                        "No similar episode yet (cold start)"
                        if event == "started"
                        else "Using in-run memory only"
                    )
                ),
                "active": episodic_hint is not None,
            },
            {
                "layer": "DT→PRINCIPLES",
                "icon": "🌳",
                "title": "Decision Tree rules → Principle executor",
                "detail": (
                    f"{len(principles)} principles in library, "
                    f"{len(triggered)} matched this stage"
                ),
                "data": triggered_detail,
            },
            {
                "layer": "KG",
                "icon": "🔗",
                "title": "Knowledge graph traversal (NN attention)",
                "detail": (
                    f"{len(activated)} nodes activated"
                    + (f" → overrides: {list(net_overrides.keys())}" if net_overrides else "")
                ),
                "path": activated,
            },
            {
                "layer": "SEARCH",
                "icon": "🎯",
                "title": f"Hybrid search ({allocation.get('stage', '?')} maturity)",
                "detail": (
                    f"{len(reasoning_actions)} reasoning + {len(pso_actions)} PSO actions, "
                    f"merged overrides: {merged_keys or '(none)'}"
                ),
            },
            {
                "layer": "STRESS",
                "icon": "⚡",
                "title": "Stress + regret feedback",
                "detail": (
                    f"decision={decision_stress:.3f}, system={system_stress:.3f}, "
                    f"regret={regret:.3f}, miner R²={miner_r2:.3f}"
                ),
            },
        ]

        if investigation:
            pipeline.append({
                "layer": "INVESTIGATE",
                "icon": "🔬",
                "title": "Regret investigation",
                "detail": investigation.get("strategy", "detected"),
                "data": investigation.get("changes_detected", {}),
            })

        if active_intervention is not None:
            pipeline.append({
                "layer": "CAUSAL",
                "icon": "🧪",
                "title": "Causal intervention test",
                "detail": (
                    f"{active_intervention.hypothesis_id}: "
                    f"{active_intervention.variable} {active_intervention.direction}"
                ),
            })

        if self._last_stage_mining:
            m = self._last_stage_mining
            pipeline.append({
                "layer": "DT-MINER",
                "icon": "⛏️",
                "title": "DT Miner discovered patterns",
                "detail": (
                    f"{m['within_rules_found']} within-dataset rules, "
                    f"R²={m['miner_r2']}, "
                    f"principles {m['principles_before']}→{m['principles_after']}"
                ),
                "data": m.get("rule_previews", []),
            })

        memory = {
            "journal_entries": len(self.journal.get_entries()),
            "principles": len(self.library.get_principles()),
            "network_nodes": self.network.node_count,
            "network_edges": len(getattr(self.network, "_edges", [])),
            "episodes": self.episodic_attention.episode_count,
            "causal_hypotheses": len(self.causal_engine._hypotheses),
            "formulas": self.formula_library.formula_count(),
            "symbolic_formulas": (
                self.symbolic_library.formula_count()
                if self.symbolic_library is not None
                else 0
            ),
            "miner_r2": round(miner_r2, 4),
            "archive_nodes": self.knowledge_archive.get_stats().get("archived_nodes", 0),
            "archive_edges": self.knowledge_archive.get_stats().get("archived_edges", 0),
            "consciousness_lines": self._observability_meta.get("consciousness", {}).get(
                "abstract_lines", 0,
            ),
            "champion_accuracy": self._observability_meta.get("consciousness", {}).get(
                "champion_accuracy",
            ),
            "last_prune": getattr(self, "_last_prune", None),
        }

        symbolic_id = getattr(self.classifier, "_last_used_formula_id", None)
        symbolic_ids = list(dict.fromkeys(
            getattr(self.classifier, "_formulas_used_this_fit", []) or []
        ))
        tournament = getattr(self.classifier, "_last_tournament", []) or []
        target_accuracy = getattr(self.cfg, "target_accuracy", None)

        return {
            "event": event,
            "stage": stage,
            "stages_total": stages_total,
            "dataset": self.dataset_name,
            "fold": self.fold,
            "decision_stress": decision_stress,
            "system_stress": system_stress,
            "regret": regret,
            "accuracy": best_action_score,
            "best_accuracy": self._best_accuracy,
            "f1_score": self._best_f1,
            "macro_f1": self._best_macro_f1,
            "accuracy_kind": (
                "kfold_mean" if self._extra_val_splits else "validation_holdout"
            ),
            "maturity": self.foundation.maturity,
            "foundation_stage": allocation.get("stage"),
            "pso_actions": allocation.get("pso_actions", 0),
            "reasoning_actions": allocation.get("reasoning_actions", 0),
            "formula_id": self._current_formula_id,
            "symbolic_formula_id": symbolic_id,
            "symbolic_formula_ids": symbolic_ids,
            "formula_tournament": tournament[:8],
            "hybrid_blend": getattr(self.classifier, "_last_hybrid_blend", None),
            "formula_depth": current_formula_depth,
            "formula_node_count": current_formula_node_count,
            "activated_nodes": activated,
            "network_overrides_keys": list(net_overrides.keys()),
            "triggered_principles": triggered,
            "action_overrides": merged_keys,
            "intervention_hypothesis_id": (
                active_intervention.hypothesis_id if active_intervention else None
            ),
            "intervention_variable": (
                active_intervention.variable if active_intervention else None
            ),
            "investigation_strategy": (
                investigation.get("strategy") if investigation else None
            ),
            "stage_cpu_seconds": stage_cpu,
            "decision_trace": {"pipeline": pipeline},
            "memory": memory,
            "episodic_warm_start": episodic_hint is not None,
            "resource_budget": getattr(self, "_resource_budget", None),
            "train_augmented": getattr(self.classifier, "_train_augmented", False),
            "target_accuracy": target_accuracy,
            "target_reached": target_reached if target_accuracy is not None else None,
        }

    def _graph_feature_emphasis(
        self, network_result: dict, n_features: int,
    ) -> np.ndarray:
        """Map KG activations + overrides to per-feature tournament weights."""
        emphasis = np.zeros(n_features, dtype=float)
        for key, val in network_result.get("overrides", {}).items():
            for prefix in ("weight_", "feature_weight_", "feat_weight_"):
                if key.startswith(prefix):
                    try:
                        idx = int(key[len(prefix):])
                        if 0 <= idx < n_features:
                            emphasis[idx] += abs(float(val))
                    except ValueError:
                        pass
        if hasattr(self.network, "get_node"):
            for nid in network_result.get("activated_nodes", []):
                node = self.network.get_node(nid)
                if node is None:
                    continue
                for cond in getattr(node, "conditions", []):
                    feat = str(cond.get("feature", ""))
                    for token in feat.replace(".", "_").split("_"):
                        if token.isdigit():
                            idx = int(token)
                            if 0 <= idx < n_features:
                                emphasis[idx] += 0.5
        if self._best_feature_weights is not None:
            w = np.asarray(self._best_feature_weights, dtype=float)
            if w.size == n_features and np.max(w) > 1e-9:
                emphasis += 0.35 * (w / (w.sum() + 1e-12))
        if emphasis.sum() > 0:
            return emphasis / emphasis.sum()
        return emphasis

    def _split(self, X: np.ndarray, y: np.ndarray):
        """Validation split."""
        from sklearn.model_selection import train_test_split
        return train_test_split(
            X, y, test_size=0.2, random_state=self.seed, stratify=y
        )

    def _stage_split(self, stage: int):
        """Row INDICES for this stage's train/val split (rotating).

        Derived from `seed + stage`, so a given stage draws the same rows on
        every replay -- reproducible without being frozen. Indices rather
        than row copies: on MNIST one validation fold is ~44 MB of data but
        ~56 KB of indices, and Windows ships a full copy to every worker.
        """
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(self._pool_y))
        try:
            tr, va = train_test_split(
                idx, test_size=0.2, random_state=self.seed + stage,
                stratify=self._pool_y,
            )
        except ValueError:
            # Too few members in some class to stratify this draw.
            tr, va = train_test_split(
                idx, test_size=0.2, random_state=self.seed + stage,
            )
        return tr, va

    def _compute_mutual_info(
        self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Compute Mutual Information as base guidance."""
        mi = mutual_info_classif(X, y, random_state=self.seed, n_neighbors=5)
        return safe_minmax(mi)

    def _build_state(
        self,
        n_samples: int,
        n_features: int,
        num_classes: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict:
        """Build the rich state vector. Pure numbers.
        
        This vector defines the ceiling of what the Miner can discover.
        More signals = higher ceiling.
        """
        state = {
            "n_samples": n_samples,
            "n_features": n_features,
            "num_classes": num_classes,
            "sample_feature_ratio": round(n_samples / max(n_features, 1), 4),
        }

        # Entropy from search diversity (if available from search feedback)
        if self._best_feature_weights is not None:
            selected = np.sum(self._best_feature_weights > 0.1)
            state["feat_ratio"] = round(float(selected / n_features), 4)
        else:
            state["feat_ratio"] = 1.0

        # Sparsity
        state["sparsity"] = round(float(np.mean(X == 0)), 4)

        # Feature correlation (sample for speed)
        try:
            if n_features > 100:
                rng = np.random.default_rng(self.seed)
                idx = rng.choice(n_features, 100, replace=False)
                X_sub = X[:, idx]
            else:
                X_sub = X
            corr = np.corrcoef(X_sub.T)
            np.fill_diagonal(corr, 0)
            state["feature_correlation_mean"] = round(
                float(np.nanmean(np.abs(corr))), 4
            )
        except Exception:
            state["feature_correlation_mean"] = 0.0

        # Class balance entropy
        class_counts = np.bincount(y.astype(int))
        class_probs = class_counts / class_counts.sum()
        class_probs = class_probs[class_probs > 0]
        state["class_entropy"] = round(
            float(-np.sum(class_probs * np.log2(class_probs + 1e-10))), 4
        )

        # Entropy placeholder (from search diversity)
        state["entropy"] = round(state.get("feat_ratio", 1.0), 4)

        return state

    def _spawn_eval_classifier(self, *, branch_parallel: bool) -> object:
        """Worker classifier for parallel PSO action scoring."""
        if self._classifier_factory is None:
            clf = self.classifier
        else:
            clf = self._classifier_factory(
                *self._classifier_build_args,
                **self._classifier_build_kwargs,
            )
            if hasattr(clf, "apply_config") and hasattr(self.classifier, "get_current_config"):
                cfg = dict(self.classifier.get_current_config())
                workers = max(1, self._parallel_action_workers)
                if "parallel_eval_workers" in cfg:
                    cfg["parallel_eval_workers"] = max(
                        1, int(cfg["parallel_eval_workers"]) // workers,
                    )
                clf.apply_config(cfg)
            if hasattr(self.classifier, "params") and hasattr(clf, "params"):
                if isinstance(clf.params, dict) and "n_jobs" in clf.params:
                    n_jobs = 1
                    if self._resource_budget:
                        n_jobs = max(1, int(self._resource_budget["n_jobs"]) // workers)
                    clf.params["n_jobs"] = n_jobs
        if hasattr(clf, "_branch_parallel"):
            clf._branch_parallel = branch_parallel
        if hasattr(clf, "set_graph_hint") and getattr(self.classifier, "_graph_hint", None):
            clf.set_graph_hint(self.classifier._graph_hint)
        if hasattr(clf, "set_episodic_hint"):
            hint = getattr(self.classifier, "_episodic_suggestion", None)
            if hint is not None:
                clf.set_episodic_hint({
                    "suggested_symbolic_formula": hint,
                    "retrieval_confidence": 1.0,
                })
        if hasattr(clf, "set_exploration_hint"):
            hint = getattr(self.classifier, "_exploration_hint", None)
            if hint is not None:
                clf.set_exploration_hint(hint)
        if hasattr(clf, "set_learner_directive"):
            directive = getattr(self.classifier, "_learner_directive", None)
            if directive is not None:
                clf.set_learner_directive(directive)
        return clf

    def _score_extra_splits(
        self, action: dict, effective_config: dict
    ) -> list[float]:
        """M0.1: fit the winning action on each extra resampled split.

        Returns one validation accuracy per extra split. Uses a spawned
        worker classifier when a factory exists; when it falls back to
        the main classifier, per-fit bookkeeping attributes are saved and
        restored so the winning fit's formula/tournament records (read
        back later in the stage loop) are not clobbered.
        """
        _shadow_attrs = (
            "_formulas_used_this_fit",
            "_last_used_formula_id",
            "_last_tournament",
            "_last_hybrid_blend",
        )
        scores: list[float] = []
        for X_tr, y_tr, X_va, y_va in self._extra_val_splits:
            clf = self._spawn_eval_classifier(branch_parallel=True)
            saved = None
            if clf is self.classifier:
                saved = {
                    a: getattr(clf, a)
                    for a in _shadow_attrs
                    if hasattr(clf, a)
                }
            try:
                if effective_config and hasattr(clf, "apply_config"):
                    clf.apply_config(effective_config)
                model = clf.fit_weighted(
                    X_tr, y_tr, action["feature_weights"],
                    warm_start_model=None,
                )
                preds = clf.predict(model, X_va)
                scores.append(float(accuracy_score(y_va, preds)))
            finally:
                if saved is not None:
                    for a, v in saved.items():
                        setattr(clf, a, v)
        return scores

    def _blas_threads_for_slots(self, slots: int) -> int:
        total = (
            int(self._resource_budget["n_jobs"])
            if self._resource_budget
            else (os.cpu_count() or 4)
        )
        return blas_threads_for_slots(total, slots)

    # Race axes whose ONLY job is to buy feature-interaction capacity.
    # E6/E8 measured that they help interaction-dense data (sonar rules
    # +0.038, XOR neural +large) and HURT near-linear data (ionosphere
    # rules -0.048). The interaction-gap gate keeps them only when the
    # fingerprint says interactions actually exist.
    _INTERACTION_AXES = ("use_rule_basis", "use_neural_basis", "use_kan_basis")

    def _gate_interaction_axes(self, axes: dict) -> dict:
        """Drop the interaction-hungry race axes on near-linear data.

        `interaction_gap = nonlinear_probe_acc - linear_probe_acc`
        (fingerprint axis 4). A small gap means a linear model already
        captures the boundary, so racing rule/neural basis families only
        gives the winner's-curse a more expressive way to overfit the
        internal folds -- which is exactly how the race lost ionosphere.
        Below `cfg.interaction_gate` (default 0.05, the same threshold the
        E5 rule-basis auto-enable used) those axes are removed from the
        race entirely. Returns a filtered COPY; never mutates the input.
        """
        if not axes or getattr(self.cfg, "race_all_axes", False):
            return axes
        gap = None
        fp = getattr(self, "_dataset_fingerprint", None) or {}
        g = fp.get("interaction_gap")
        if isinstance(g, (int, float)):
            gap = float(g)
        # No probe value (probe skipped / errored) -> be conservative and
        # KEEP the axes: a missing signal must not silently disable a
        # capability the user opted into by choosing the race.
        if gap is None:
            return axes
        # Default 0.07, MEASURED to separate the two discriminating cases:
        # ionosphere interaction_gap is 0.051-0.066 across seeds (near-
        # linear, its rule/neural arms overfit -> should be gated OUT) and
        # sonar is 0.072-0.077 (interaction-dense, its rules help -> kept).
        # The gap between those ranges is real (~0.006) but narrow, so this
        # threshold IS fit to these two datasets -- it must clear the full
        # 6-dataset Gate 2 before `search_mode="race"` is defaulted on, and
        # `cfg.interaction_gate` is exposed precisely so that value is not
        # frozen. (E5's 0.05 was too low -- it sat below ionosphere's own
        # gap, so the gate never fired where it was needed.)
        threshold = float(getattr(self.cfg, "interaction_gate", 0.07))
        if gap >= threshold:
            return axes  # interactions present -> race the full set
        gated = {k: v for k, v in axes.items() if k not in self._INTERACTION_AXES}
        dropped = [k for k in axes if k in self._INTERACTION_AXES]
        if dropped:
            self._observability_meta.setdefault("interaction_gated_axes", {})[
                "dropped"
            ] = dropped
            self._observability_meta["interaction_gated_axes"]["gap"] = round(gap, 4)
        return gated

    def _pick_winner_with_margin(self, results: list[dict]) -> dict | None:
        """Choose the stage winner from evaluated action results.

        Default (`cfg.race_incumbent_margin` == 0) is a plain argmax on
        `select_score` (the multi-split mean from the judge fix, falling
        back to the primary score) -- byte-identical to the prior loop.

        Opt-in winner's-curse mitigation for the race: arm 0 is the
        INCUMBENT, which carries the plain sparse_linear that cognition
        proposed. A challenger that beats it by only a hair on the
        internal folds has most likely won by noise and will overfit the
        external test -- this is exactly how the race lost ionosphere
        (challenger looks better on 3 internal folds, worse on the true
        test, and the interaction_gap proxy was too noisy to catch it).
        Requiring a challenger to clear the incumbent by a noise-floor
        margin attacks that directly. Only fires in race mode (an
        incumbent arm exists); needs its own Gate 2 before defaulting on.
        """
        if not results:
            return None

        def sel(r: dict) -> float:
            return r.get("select_score", r["score"])

        best = max(results, key=sel)
        margin = float(getattr(self.cfg, "race_incumbent_margin", 0.0))
        if margin <= 0.0 or self._race is None:
            return best
        inc = next(
            (r for r in results if r["action"].get("_race_arm") == "incumbent"),
            None,
        )
        if inc is not None and best is not inc and sel(best) < sel(inc) + margin:
            return inc
        return best

    def _architecture_exploration_probe(
        self,
        stage: int,
        best_action: dict,
        overrides: dict,
        state: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        system_stress: float,
    ) -> dict | None:
        """PBT-style default-path architecture variation (research step 2).

        ROOT CAUSE of the inert cognitive layer (RESEARCH_FAILURE_ANALYSIS
        F4): in classic search the per-stage candidates vary only `penalty`;
        every architecture knob is CONSTANT in the journal, so the miner can
        never split on one and no principle can ever mention architecture.
        This adds ONE probe per stage that perturbs ONE architecture axis
        (round-robin) and JOURNALS the (config, outcome) pair as evidence
        the miner can later learn from.

        Winner's-curse-safe BY CONSTRUCTION: the probe NEVER competes for
        the stage winner -- it is fit purely to enrich the journal, so it
        cannot cause the E7 selection-overfitting regression (the research
        doc's explicit warning: variation must not ride the broken judge).
        Bug #21 respected: `get_architecture_axes()` excludes learning_mode.
        Opt-in (`cfg.explore_architecture`, default off): it adds journal
        entries that feed FUTURE mining, so it is an evidence-gathering
        change with a cross-run side effect, not an accuracy change.

        Returns the probe's recorded action_record (for observability), or
        None when no axis was available to perturb.
        """
        axes = {}
        if hasattr(self.classifier, "get_architecture_axes"):
            try:
                axes = self.classifier.get_architecture_axes() or {}
            except Exception:
                axes = {}
        if not axes:
            return None
        names = sorted(axes)
        axis = names[stage % len(names)]
        values = list(axes[axis].get("values", []))
        current = axes[axis].get("current")
        alts = [v for v in values if v != current]
        if not alts:
            return None
        # Rotate through the alternatives on successive visits to this axis,
        # so over a run every (axis, value) is exercised deterministically.
        chosen = alts[(stage // len(names)) % len(alts)]

        probe = dict(best_action)
        probe_cfg = dict(best_action.get("classifier_config") or {})
        probe_cfg[axis] = chosen
        probe["classifier_config"] = probe_cfg

        # Fit in isolation. With a factory the worker is already isolated;
        # without one the probe runs on the main classifier, so its config
        # and per-fit bookkeeping are snapshotted and restored (same
        # discipline the race rungs use -- a probe must never corrupt the
        # run it enriches).
        use_main = self._classifier_factory is None
        saved = self._shadowed_attrs(self.classifier) if use_main else None
        saved_cfg = None
        if use_main and hasattr(self.classifier, "get_current_config"):
            try:
                saved_cfg = dict(self.classifier.get_current_config())
            except Exception:
                saved_cfg = None
        try:
            result = self._evaluate_one_action(
                probe, overrides, X_train, y_train, X_val, y_val,
                branch_parallel=True,
                use_main_classifier=use_main,
                allow_warm_start=False,
            )
        except Exception:
            return None
        finally:
            if saved is not None:
                for a, v in saved.items():
                    setattr(self.classifier, a, v)
            if saved_cfg is not None and hasattr(self.classifier, "apply_config"):
                try:
                    self.classifier.apply_config(saved_cfg)
                except Exception:
                    pass

        probe_acc = float(result.get("score", 0.0))
        probe_regret = max(0.0, self._best_accuracy - probe_acc)
        eff_cfg = result.get("effective_config") or probe_cfg
        action_record = {
            "penalty": probe.get("penalty", best_action.get("penalty", 3.0)),
            "description": f"arch_probe:{axis}={chosen}",
            "classifier_config": eff_cfg,
        }
        try:
            self.journal.record(
                dataset=self.dataset_name, fold=self.fold, stage=stage,
                classifier=self.classifier.name, state=state,
                action=action_record,
                outcome={"accuracy": round(probe_acc, 4)},
                decision_stress=0.0, system_stress=system_stress,
                regret=probe_regret, investigation=None,
            )
        except Exception:
            return None
        return action_record

    def _evaluate_one_action(
        self,
        action: dict,
        overrides: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        *,
        branch_parallel: bool,
        use_main_classifier: bool = False,
        blas_slots: int = 1,
        allow_warm_start: bool = True,
    ) -> dict:
        intervention_role = action.get("_intervention_role")
        action = self._apply_overrides(action, overrides)
        classifier_config = action.get("classifier_config", {})
        if use_main_classifier:
            clf = self.classifier
        else:
            clf = self._spawn_eval_classifier(branch_parallel=branch_parallel)
        if classifier_config and hasattr(clf, "apply_config"):
            clf.apply_config(classifier_config)
        threads = self._blas_threads_for_slots(blas_slots)
        with blas_thread_scope(threads):
            model = clf.fit_weighted(
                X_train,
                y_train,
                action["feature_weights"],
                warm_start_model=(
                    self._best_model
                    if (clf.supports_warm_start and allow_warm_start)
                    else None
                ),
            )
            preds = clf.predict(model, X_val)
        acc = float(accuracy_score(y_val, preds))
        effective_config = (
            clf.get_current_config() if hasattr(clf, "get_current_config") else {}
        )
        return {
            "action": action,
            "score": acc,
            "model": model,
            "effective_config": effective_config,
            "intervention_role": intervention_role,
            "worker_classifier": clf if not use_main_classifier else None,
        }

    # ── Architecture race ────────────────────────────────────────────

    def _race_bucket(self) -> str:
        """Memory key for arm outcomes.

        Keyed on the ROUTED DOMAIN, not the dataset name, so evidence
        generalizes ("on D4_structured data, rules lose") instead of
        restarting from scratch on every new dataset -- which is the only
        way the phase-out can ever fire on unseen data.
        """
        route = getattr(self, "_route_decision", None) or {}
        domain = route.get("domain") or "unrouted"
        clf = getattr(self.classifier, "name", "?")
        return f"{domain}|{clf}"

    def _shadowed_attrs(self, clf) -> dict:
        """Snapshot the per-fit bookkeeping an elimination rung must not
        clobber (same attribute set `_score_extra_splits` protects).

        Elimination rungs are throwaway measurements. When workers are
        off they run on the MAIN classifier, and `_evaluate_one_action`
        is NOT side-effect-free -- it writes formula-usage records that
        the stage loop reads back later. A probe that corrupts the run it
        is measuring is worthless (that lesson cost two failed attempts
        at the verify-before-apply gate, bug #25).
        """
        names = (
            "_formulas_used_this_fit",
            "_last_used_formula_id",
            "_last_tournament",
            "_last_hybrid_blend",
        )
        return {a: getattr(clf, a) for a in names if hasattr(clf, a)}

    def _race_actions(
        self,
        actions: list[dict],
        overrides: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        *,
        stage: int,
    ) -> list[dict]:
        """Successive halving: cheap subsampled rungs, then a full fit.

        Returns exactly what `_evaluate_stage_actions` returns, scored at
        FULL fidelity, so every downstream consumer (best-action pick,
        journal, causal comparison, champion) is unchanged.

        Two invariants:
        - Elimination rungs never mutate the run. Adapter bookkeeping is
          saved and restored around them.
        - Causal intervention arms are never eliminated. Control and
          treatment must be compared at the SAME fidelity or the causal
          verdict is an artifact of the rung they died on.
        """
        n_train = len(y_train)
        survivors = list(actions)
        plan = race_rungs(n_train, len(survivors))
        rng = np.random.default_rng(self.seed + stage)

        for rung_idx, (rows, keep) in enumerate(plan):
            if len(survivors) <= keep:
                continue
            idx = stratified_subsample(y_train, rows, rng)
            Xs, ys = X_train[idx], y_train[idx]
            if len(np.unique(ys)) < 2:
                break  # degenerate subsample -- skip straight to full fit

            clf = self.classifier
            saved = self._shadowed_attrs(clf)
            try:
                rung_results = self._evaluate_stage_actions(
                    survivors, overrides, Xs, ys, X_val, y_val, stage=stage,
                )
            except Exception:
                # A rung is an optimization, never a reason to fail a
                # stage: fall through and score everyone at full fidelity.
                break
            finally:
                for attr, val in saved.items():
                    setattr(clf, attr, val)

            scored = [(r["score"], i) for i, r in enumerate(rung_results)]
            scored.sort(reverse=True)
            keep_ids = {i for _, i in scored[:keep]}
            # Protected arms always reach the FULL-fidelity rung:
            #  - the causal control/treatment pair, because comparing
            #    them across different rungs makes the causal verdict an
            #    artifact of where each one died;
            #  - the INCUMBENT, because it is the arm carrying the
            #    cognitive layer's proposal (principle overrides, warm
            #    start, episodic hint). Letting a 150-row subsample veto
            #    cognition before it is ever measured properly would
            #    reintroduce exactly the failure this work exists to fix.
            #    Cognition gets its say at full fidelity, then loses
            #    honestly if it loses.
            keep_ids |= {
                i
                for i, r in enumerate(rung_results)
                if r["action"].get("_intervention_role")
                or r["action"].get("_race_arm") == "incumbent"
            }
            next_survivors = [
                rung_results[i]["action"] for i in sorted(keep_ids)
            ]
            self._race_stats[f"stage{stage}_rung{rung_idx}"] = {
                "rows": int(rows),
                "in": len(survivors),
                "out": len(next_survivors),
                "eliminated": [
                    rung_results[i]["action"].get("_race_arm")
                    for i in range(len(rung_results))
                    if i not in keep_ids
                ],
            }
            survivors = next_survivors
            if len(survivors) <= 1:
                break

        results = self._evaluate_stage_actions(
            survivors, overrides, X_train, y_train, X_val, y_val, stage=stage,
        )

        # THE JUDGE FIX (E7 Gate 2 failure -> Cawley & Talbot 2010;
        # RESEARCH_FAILURE_ANALYSIS.md step 1). Gate 2 measured
        # corr(Δaccuracy, val-gap swing) = -0.977: essentially every point
        # the race lost was selection overfitting -- best-of-N judged on
        # the ONE primary split, which on small data is ~56 rows. The
        # nested-CV remedy: the FINAL rung's verdict comes from the mean
        # over every split available, not the split the rungs already
        # used. Affordable because only ~2-3 finalists reach this rung.
        #
        # `score` (primary-split) is left untouched on every result: the
        # causal control/treatment comparison reads it and stays
        # single-split-internal by design -- both arms are protected to
        # the final rung, so theirs is a same-split, same-fidelity
        # comparison either way. Selection reads `select_score`.
        # `cfg.race_select_on_extra_splits = False` restores the old
        # (diagnosed-broken) selection for A/B comparison.
        if (
            results
            and self._extra_val_splits
            and getattr(self.cfg, "race_select_on_extra_splits", True)
        ):
            for r in results:
                try:
                    extra = self._score_extra_splits(
                        r["action"], r["effective_config"],
                    )
                except Exception:
                    continue  # this arm is judged on the primary split only
                if not extra:
                    continue
                r["extra_scores"] = extra
                folds = extra if self._rotate_val else [r["score"]] + extra
                r["select_score"] = float(np.mean(folds))

        # Credit the winner -- the arm SELECTION will actually pick, i.e.
        # judged multi-split when the judge fix is on. Only the arms that
        # reached the final rung are scored here, but ALL proposed arms
        # raced, so all get a trial -- otherwise an arm that keeps dying
        # early would keep its 0.5 prior forever and never phase out.
        if results:
            # Same selection rule the stage loop uses (Category-C: one
            # helper, not two copies), so arm memory learns from the arm
            # that actually won -- including the incumbent-margin when on.
            chosen = self._pick_winner_with_margin(results)
            if chosen is not None:
                try:
                    self._race.record_round(
                        self._race_bucket(), actions, chosen["action"],
                    )
                except Exception:
                    pass
        return results

    def _evaluate_stage_actions(
        self,
        actions: list[dict],
        overrides: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        *,
        stage: int,
    ) -> list[dict]:
        workers = 1
        if (
            self._classifier_factory is not None
            and self._parallel_action_workers > 1
            and len(actions) > 1
        ):
            workers = min(self._parallel_action_workers, len(actions))

        branch_slots = 1
        if hasattr(self.classifier, "hybrid_branch_count"):
            branch_slots = max(
                1,
                min(
                    int(getattr(self.classifier, "parallel_eval_workers", 1)),
                    int(getattr(self.classifier, "hybrid_branch_count", 5)),
                ),
            )

        # Bug #29: warm start has two jobs and only one is legitimate.
        # RESUMING across runs (start from the champion instead of zero) is
        # useful. SPEEDING UP the 4-6 fits inside a stage is not: those fits
        # exist to differ from each other, and handing them all the same
        # frozen model guaranteed they mostly did not -- for every mode
        # except sparse_linear (the one that opts out, and the one that
        # benchmarks well) the stage loop stopped training after fit 1.
        # So: exactly ONE warm-started fit per run, the first, deterministic
        # (index 0 of the first evaluated stage) so replay is unaffected.
        # Default OFF: the 3-seed A/B did NOT support it (breast_cancer
        # -0.0015, wine -0.0093, digits +0.0000; mean -0.0036). The
        # STRUCTURAL defect it fixes is real and confirmed -- see bug #29 --
        # but recovering candidate diversity did not buy accuracy here, so
        # per the E6 lesson it does not ship on. Revisit together with the
        # fixed-validation-split fix; see the hypothesis in CLAUDE.md #29.
        warm_idx = -1
        if getattr(self.cfg, "warm_start_first_fit_only", False):
            if not self._warm_start_consumed:
                warm_idx = 0
                self._warm_start_consumed = True
        else:
            warm_idx = None  # legacy: every action warm-starts

        def _warm(idx: int) -> bool:
            return True if warm_idx is None else idx == warm_idx

        if workers <= 1:
            results = []
            for _idx, action in enumerate(actions):
                try:
                    results.append(
                        self._evaluate_one_action(
                            action,
                            overrides,
                            X_train,
                            y_train,
                            X_val,
                            y_val,
                            branch_parallel=True,
                            use_main_classifier=True,
                            blas_slots=branch_slots,
                            allow_warm_start=_warm(_idx),
                        )
                    )
                except Exception as e:
                    record_failure_point(
                        "optimizer.action_fit",
                        {
                            "dataset": self.dataset_name,
                            "fold": self.fold,
                            "stage": stage,
                            "action": {
                                k: v for k, v in action.items()
                                if k != "feature_weights"
                            },
                            "classifier": self.classifier.name,
                            "intervention_role": action.get("_intervention_role"),
                        },
                        e,
                    )
                    raise
            return results

        results: list[dict | None] = [None] * len(actions)

        def _run(idx_action: tuple[int, dict]) -> tuple[int, dict]:
            idx, action = idx_action
            return idx, self._evaluate_one_action(
                action,
                overrides,
                X_train,
                y_train,
                X_val,
                y_val,
                branch_parallel=False,
                blas_slots=workers,
                allow_warm_start=_warm(idx),
            )

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for idx, result in pool.map(_run, list(enumerate(actions))):
                    results[idx] = result
        except Exception as e:
            failed = next(a for a in actions if a is not None)
            record_failure_point(
                "optimizer.action_fit",
                {
                    "dataset": self.dataset_name,
                    "fold": self.fold,
                    "stage": stage,
                    "classifier": self.classifier.name,
                    "parallel_workers": workers,
                },
                e,
            )
            raise
        return [r for r in results if r is not None]

    @staticmethod
    def _step_out_of_region(current, region):
        """Move a knob just outside an interval a mined rule marked bad.

        A tree path bounded on both sides ("regret is high for
        boosting_stages in (14.5, 19.5]") condemns a REGION. Escaping it
        is possible in either direction, so the choice is made here,
        where the knob's live value is known: leave by the NEARER edge,
        which is the smallest change that satisfies the rule. Returns
        None when the knob already sits outside, so a satisfied rule
        costs nothing.
        """
        if not isinstance(region, (list, tuple)) or len(region) != 2:
            return None
        try:
            lo, hi = float(region[0]), float(region[1])
        except (TypeError, ValueError):
            return None
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            return None
        if not (lo < current <= hi):  # already outside the bad region
            return None
        eps = 1e-6
        if isinstance(current, int):
            below, above = int(math.floor(lo)), int(math.ceil(hi)) + 1
        else:
            below, above = lo - eps, hi + eps
        return below if (current - lo) <= (hi - current) else above

    def _resolve_config_bounds(self, action: dict, overrides: dict) -> dict:
        """Turn `config_max.X` / `config_min.X` into concrete config values.

        A mined rule constrains a knob ("regret rises once sparse_l1_c goes
        above 0.3"); it does not know what the knob is set to right now.
        Resolution therefore has to happen at the actuator, against the
        value this action would actually run with: the action's own
        classifier_config if it carries one (a race arm does), otherwise
        the classifier's live config.

        Booleans are handled by KIND, not by name, and by solving the
        constraint over the only two values the knob can take rather than
        by comparing against 0.5. A gate journals as 0.0/1.0, and a tree
        can split it anywhere in between, so "ceiling 0.3" and "ceiling
        0.5" must both mean OFF. Emitting the raw bound instead would
        hand `apply_config` the float 0.5, whose `bool()` is True -- the
        silent-no-op family of bug #1.

        The result is deliberately NOT applied to an architecture-race
        arm's own axis: `_apply_overrides` drops the raced key after
        merging, so a principle cannot set every arm identically and
        collapse the race (E7's invariant).
        """
        if not overrides:
            return {}
        live = {}
        if hasattr(self.classifier, "get_current_config"):
            try:
                live = dict(self.classifier.get_current_config() or {})
            except Exception:
                live = {}
        live.update(action.get("classifier_config") or {})

        # Inert-knob filter. A mined rule may constrain any `config.*`
        # column the miner saw vary, but a knob that does nothing under the
        # ACTIVE learning mode yields a "true but useless" override -- the
        # step-2+3 chain's very first architecture rule was exactly that
        # (`config_min.bagging_estimators` under sparse_linear, provably a
        # no-op since `_fit_sparse_linear` never reads it). The adapter is
        # the authority on what affects its own fit. Absent the method,
        # nothing is filtered, so other adapters behave exactly as before.
        # Accuracy-neutral by construction: the dropped knobs cannot change
        # the model -- what it prevents is journalling a knob that did
        # nothing, which would otherwise feed more spurious rules.
        active = None
        if hasattr(self.classifier, "get_active_config_knobs"):
            try:
                active = set(self.classifier.get_active_config_knobs())
            except Exception:
                active = None

        def _is_inert(param: str) -> bool:
            if active is None or param in active:
                return False
            # Observe-only: how often do mined rules constrain a knob that
            # is dead for the current mode? A high rate means the miner is
            # learning about knobs that cannot matter (audit Part E).
            self.guard_telemetry.observe(
                "inert_config_override_dropped", 1.0, 0.0,
            )
            return True

        resolved: dict = {}
        for key, bound in overrides.items():
            if key.startswith("config_avoid."):
                param = key.split(".", 1)[1]
                if _is_inert(param):
                    continue
                out = self._step_out_of_region(live.get(param), bound)
                if out is not None:
                    resolved[param] = out
                continue
            if key.startswith("config_max."):
                param, is_ceiling = key.split(".", 1)[1], True
            elif key.startswith("config_min."):
                param, is_ceiling = key.split(".", 1)[1], False
            else:
                continue
            if _is_inert(param):
                continue
            if param not in live or not isinstance(bound, (int, float)):
                # Unknown to this classifier -- a rule mined while a
                # different adapter was in play. Silently inapplicable,
                # not an error.
                continue
            current = live[param]
            if isinstance(current, bool):
                # Solve the constraint over the only two values the knob
                # can take. A ceiling B admits {v in {0,1} : v <= B}; a
                # floor B admits {v : v >= B}. When both survive, the rule
                # says nothing and the current value stands.
                allowed = [
                    v for v in (False, True)
                    if (float(v) <= bound if is_ceiling else float(v) >= bound)
                ]
                if len(allowed) == 1:
                    resolved[param] = allowed[0]
            elif isinstance(current, int):
                # Round INTO the feasible region, never out of it:
                # floor a ceiling, ceil a floor.
                resolved[param] = (
                    min(current, int(math.floor(bound))) if is_ceiling
                    else max(current, int(math.ceil(bound)))
                )
            elif isinstance(current, float):
                resolved[param] = float(
                    min(current, bound) if is_ceiling else max(current, bound)
                )
            # Strings (learning_mode) are unorderable and are filtered out
            # upstream by NON_CONSTRAINABLE_CONFIG; skip defensively.
        return resolved

    def _apply_overrides(self, action: dict, overrides: dict) -> dict:
        """Apply principle overrides to a proposed action.
        
        V20: Now also applies hyperparameter overrides from principles.
        """
        action = dict(action)  # Don't mutate original

        # Drop penalty bounds outside the knob's own domain. 18 of the
        # 300 nodes in the live graph carry values like
        # `min_penalty: 1270.45` / `max_penalty: 0.0003`, minted before
        # uil_library.PENALTY_FEATURES stopped the substring match that
        # let DERIVED products (`derived.action.penalty*config.x`) pose
        # as penalty thresholds. Those are corrupt data, not constraints.
        # This mattered more after bound aggregation started INTERSECTING
        # rather than summing: summing hid them (a cap of 1223 caps
        # nothing), while a min would let one bogus 0.0003 pin penalty to
        # zero for the whole stage.
        lo_ok, hi_ok = UIL_Library.PENALTY_BOUND_RANGE
        _kept = {}
        for k, v in overrides.items():
            corrupt = (
                k in ("max_penalty", "min_penalty")
                and isinstance(v, (int, float))
                and not (lo_ok <= v <= hi_ok)
            )
            if k in ("max_penalty", "min_penalty") and isinstance(v, (int, float)) and math.isfinite(v):
                # Observe-only (audit Part E): bind == an out-of-domain
                # penalty bound was dropped. Encode the drop as the in-domain
                # value it would have had to be (differs from v iff corrupt),
                # so the bind flag is correct and finite. A nonzero bind-rate
                # here means corrupt derived-feature overrides are still
                # reaching the actuator (should be ~0 after PENALTY_FEATURES).
                self.guard_telemetry.observe(
                    "penalty_bound_domain_guard",
                    v, float(min(max(v, lo_ok), hi_ok)),
                )
            if not corrupt:
                _kept[k] = v
        overrides = _kept

        # Cap penalty if principles say so.
        # Observe-only (audit Part E): `cognition_penalty_override` binds iff
        # a principle/graph override ACTUALLY moved the penalty search
        # proposed. This is the single most important measurement of the
        # inert-layer investigation: `max_penalty`/`min_penalty` is the
        # cognitive layer's ONE channel to the learner (bug #25), so this
        # bind-rate is literally "how often does cognition change what the
        # learner does at all." A near-zero rate here IS the inert layer,
        # quantified per run instead of argued.
        _pen_before = action.get("penalty", 3.0)
        if "max_penalty" in overrides:
            action["penalty"] = min(
                action.get("penalty", 3.0), overrides["max_penalty"]
            )
        if "min_penalty" in overrides:
            action["penalty"] = max(
                action.get("penalty", 0.0), overrides["min_penalty"]
            )
        if "max_penalty" in overrides or "min_penalty" in overrides:
            self.guard_telemetry.observe(
                "cognition_penalty_override",
                float(_pen_before), float(action.get("penalty", _pen_before)),
            )
        # Same interval semantics as config_avoid.*: a tree path that
        # bounds penalty on BOTH sides condemns a band, not a direction.
        # Previously such a path emitted max_penalty AND min_penalty and
        # the two lines above applied them in order, so penalty was pinned
        # to the upper edge by accident of statement order rather than by
        # any decision.
        if "penalty_avoid" in overrides:
            moved = self._step_out_of_region(
                float(action.get("penalty", 3.0)), overrides["penalty_avoid"],
            )
            if moved is not None:
                action["penalty"] = float(moved)
        # Actuator boundary: penalty is a bounded control knob (PSO and
        # reasoning propose 0.5-3.0). Cognition upstream (edge formulas,
        # override blending) may propose anything; the actuator clamps.
        # Without this, one runaway network override poisons the fitted
        # model, the journal record, and every threshold mined from it.
        if "penalty" in action and isinstance(action["penalty"], (int, float)):
            _pen_in = float(action["penalty"])
            action["penalty"] = float(np.clip(_pen_in, 0.0, 10.0))
            # Observe-only (audit Part E): how often, and by how much, does
            # cognition propose a penalty outside the actuator's domain?
            # The clamp above is unchanged; this only records it.
            self.guard_telemetry.observe(
                "penalty_actuator_clip", _pen_in, action["penalty"],
            )

        # V20: Apply classifier config overrides from principles
        config_overrides = {}
        for key, value in overrides.items():
            if key.startswith("config."):
                param_name = key.split(".", 1)[1]
                config_overrides[param_name] = value

        # Bug #25 vocabulary widening: mined rules also emit BOUNDS on
        # classifier config (`config_max.sparse_l1_c`, `config_min.
        # boosting_stages`) rather than assignments -- see
        # uil_library._generate_override for why. A bound needs the live
        # value to resolve against, which only exists here.
        bounded = self._resolve_config_bounds(action, overrides)
        for param_name, value in bounded.items():
            # An explicit assignment from a principle is more specific
            # than a bound, so it wins.
            config_overrides.setdefault(param_name, value)

        if config_overrides:
            # An architecture-race arm OWNS the one axis it exists to
            # test. A principle override on that same key would set it
            # identically on every arm, collapsing the race back into a
            # single candidate evaluated N times -- the exact one-knob
            # failure this search mode was built to escape. Cognition
            # still shapes every OTHER key, and still fully shapes the
            # incumbent arm (which races no axis of its own).
            raced_axis = action.get("_race_axis")
            if raced_axis:
                config_overrides.pop(raced_axis, None)
            # Copy the nested dict: `action = dict(action)` above is a
            # SHALLOW copy, so updating classifier_config in place edits
            # the caller's action -- which for a race arm persists across
            # rungs and quietly accumulates overrides.
            existing_config = dict(action.get("classifier_config") or {})
            existing_config.update(config_overrides)
            action["classifier_config"] = existing_config

        # Signal translator: reshape particle weights toward the stage
        # directive (principles / network / episodic / dataset prior).
        # This is the strong connection the thin max_penalty channel lacked.
        directive = getattr(self, "_stage_directive", None)
        if (
            directive is not None
            and directive.feature_guidance is not None
            and action.get("feature_weights") is not None
        ):
            fw = np.asarray(action["feature_weights"], dtype=float)
            g = np.asarray(directive.feature_guidance, dtype=float)
            if fw.size == g.size:
                # Principles own the blend when they contributed — otherwise
                # soft-merge so network/episodic can still speak.
                alpha = 0.85 if "principles" in (directive.sources or []) else 0.6
                action["feature_weights"] = np.clip(
                    (1.0 - alpha) * fw + alpha * g, 0.02, 1.0,
                )

        # Translator may also pin symbolic-basis / config on the action so
        # worker classifiers (which don't share main apply_config) honor it.
        if directive is not None:
            cfg = dict(action.get("classifier_config") or {})
            changed = False
            if directive.use_symbolic_basis is not None:
                if cfg.get("use_symbolic_basis") != directive.use_symbolic_basis:
                    cfg["use_symbolic_basis"] = directive.use_symbolic_basis
                    changed = True
            for k, v in directive.config.items():
                if k not in cfg:
                    cfg[k] = v
                    changed = True
            if changed:
                action["classifier_config"] = cfg

        return action

    def _admit_symbolic_mutant(
        self,
        *,
        tree,
        mutation_type: str,
        parent: dict,
        stage: int,
        X_train,
        y_train,
        feature_weights,
    ) -> bool:
        """Serial admit path for one classifier-formula mutant.

        Shared by the single-engine evolution path and the parallel
        pyramid so utility-gating / incubating caps stay identical.
        Returns True if the formula was written to the library.
        """
        if self.symbolic_library is None:
            return False
        slots = (
            FormulaLibrary.MAX_INCUBATING
            - len(self.symbolic_library.get_incubating_formulas())
        )
        if slots <= 0:
            return False
        admit_status = "incubating"
        if getattr(self.cfg, "utility_gated_formulas", False) \
                and hasattr(self.classifier, "utility_test_candidate"):
            stats = getattr(self, "_utility_admit_stats", None)
            if stats is not None:
                stats["tested"] += 1
            try:
                ok, delta = self.classifier.utility_test_candidate(
                    tree, X_train, y_train, feature_weights,
                )
            except Exception:
                ok, delta = False, 0.0
            if not ok:
                if stats is not None:
                    stats["rejected"] += 1
                self._observability_meta["utility_admit_last_delta"] = float(delta)
                return False
            admit_status = "active"
            if stats is not None:
                stats["admitted"] += 1
            self._observability_meta["utility_admit_last_delta"] = float(delta)
        self.symbolic_library.add_formula(
            name=f"{parent['name']}_mut",
            tree=tree,
            origin=f"mutation_of_{parent['formula_id']}",
            mutation_type=mutation_type,
            created_at_stage=stage,
            status=admit_status,
        )
        return True

    def _admit_symbolic_pyramid(
        self,
        *,
        system_stress: float,
        stage: int,
        stages: int,
        n_features: int,
        feature_weights,
        X_train,
        y_train,
    ) -> int:
        """Parallel isolated FormulaEngines + randomized DT pyramid merge.

        Opt-in via cfg.parallel_formula_pyramid. Workers never touch the
        shared library; admit is serial and still respects MAX_INCUBATING.
        """
        from figpso_uil.cognition.formula_pyramid import (
            parents_from_library,
            run_formula_pyramid,
        )

        if self.symbolic_library is None or self.symbolic_engine is None:
            return 0
        progress = stage / max(1, stages - 1)
        parents = parents_from_library(
            self.symbolic_library,
            self.symbolic_engine.rng,
            k=max(2, int(getattr(self.cfg, "formula_pyramid_parents", 4))),
            progress=progress,
        )
        if not parents:
            return 0
        n_workers = max(2, int(getattr(self.cfg, "formula_pyramid_workers", 4)))
        n_admit = max(1, int(getattr(self.cfg, "formula_pyramid_admit", 2)))
        slots = (
            FormulaLibrary.MAX_INCUBATING
            - len(self.symbolic_library.get_incubating_formulas())
        )
        if slots <= 0:
            return 0
        n_admit = min(n_admit, slots)
        survivors = run_formula_pyramid(
            parents=parents,
            master_seed=int(self.seed) ^ (stage * 0x9E37),
            stress=float(system_stress),
            n_features=n_features,
            feature_bias=feature_weights,
            X=X_train,
            y=y_train,
            n_workers=n_workers,
            n_admit=n_admit,
            parallel=True,
        )
        admitted = 0
        for cand in survivors:
            parent = {
                "formula_id": cand.parent_id,
                "name": cand.parent_name,
            }
            if self._admit_symbolic_mutant(
                tree=cand.tree,
                mutation_type=f"pyramid_{cand.mutation_type}",
                parent=parent,
                stage=stage,
                X_train=X_train,
                y_train=y_train,
                feature_weights=feature_weights,
            ):
                admitted += 1
        self._observability_meta["formula_pyramid_last"] = {
            "workers": n_workers,
            "survivors": len(survivors),
            "admitted": admitted,
            "stage": stage,
        }
        return admitted

    def _neural_pick_classifier_mutation(
        self, base_tree, system_stress, n_features, feature_bias,
    ):
        """Neural-ranked mutation of a SymbolicFormula-classifier tree.

        Generates several grammar-safe candidate mutations and lets the
        offline own-transformer (neural_proposer, via LLMMutator) pick the
        highest-scored. Falls back to a single engine mutation when no
        trained model card exists — behavior-identical to the old path.
        Returns (tree, mutation_type) where mutation_type is 'neural' when
        the ranker chose, else the engine operator name.
        """
        engine = self.symbolic_engine
        neural = (
            self._llm_mutator._neural_proposer()
            if getattr(self, "_llm_mutator", None) is not None else None
        )
        if neural is None:
            return engine.mutate(
                base_tree, system_stress,
                n_features=n_features, feature_bias=feature_bias,
            )
        candidates = []
        for _ in range(6):
            mut, mtype = engine.mutate(
                base_tree, system_stress,
                n_features=n_features, feature_bias=feature_bias,
            )
            ok, _ = engine.validate_classifier(mut, n_features=n_features)
            if ok:
                candidates.append((mut, mtype))
        if not candidates:
            return engine.mutate(
                base_tree, system_stress,
                n_features=n_features, feature_bias=feature_bias,
            )
        ranked = neural.rank_candidates(candidates)
        best_tree, _engine_type, _score = ranked[0]
        return best_tree, "neural"

    EVOLUTION_MIN_GAP = 3  # stages between births per channel (rate limit)

    def _promotion_min_evidence(self, n_entries: int) -> int | None:
        """Evidence a mined rule needs before it becomes a principle.

        None = keep the library's fixed `MIN_EVIDENCE` (10), which is the
        default and today's behaviour.

        MEASURED 2026-07-23: `evidence_count` is a decision-tree LEAF
        SIZE, so it is bounded by the journal. Against a planted signal
        the largest leaf holds 4 rows at 16 journal entries, 9 at 100,
        and only at ~200 does anything clear 10. Every run with an
        isolated `state_dir` starts from an empty journal, so no
        principle is EVER mined in an experiment or an audit arm -- the
        library holds bootstrap seeds only, and those emit `max_penalty`
        alone. The cognitive layer therefore reads INERT for a reason
        that has nothing to do with how much vocabulary the miner can
        express.

        Opt-in scaling mirrors M0.2's tiny-mining mode: allow a smaller
        leaf when the journal is small, never more than the fixed bar.
        The miner already shrinks such a rule's confidence toward 0.5, so
        weak evidence buys a weak principle rather than a confident
        wrong one -- but this still changes what counts as knowledge, so
        it stays off until it clears a cross-seed gate (the E6 lesson).
        """
        if not getattr(self.cfg, "adaptive_min_evidence", False):
            return None
        return max(3, min(UIL_Library.MIN_EVIDENCE, n_entries // 8))

    def _want_new_structure(
        self, system_stress: float, stage: int, channel: str = "loss",
    ) -> bool:
        """Should formula evolution / the neural proposer fire this stage?

        Fires on high stress (the original signal) OR on stagnation --
        no accuracy improvement for >=2 stages -- BUT throttled to at most
        one birth per channel every EVOLUTION_MIN_GAP stages. Without the
        throttle, a run whose accuracy plateaus (common) triggers
        stagnation on EVERY subsequent stage; each birth grows the formula
        library, and fit_weighted cost scales with library size, so
        per-stage time compounds superlinearly (adult: stage 0→1.8s,
        1→7.8s, 2→24.1s ... → a 142-min hang on 8 stages). MAX_INCUBATING
        caps *concurrent* incubation but not this per-stage fit blow-up.
        The rate limit + the 5-stage prune-to-cap keep both bounded.
        """
        # The stress trigger, made reachable (2026-07-23).
        #
        # MEASURED, on the 1,299-entry journal covering 37 runs: only TWO
        # runs ever reached system_stress 0.6. Run-max percentiles are
        # p25 0.085, p50 0.169, p75 0.254, p90 0.431. The distribution is
        # bimodal -- a run either saturates at 1.0 or stays under ~0.43 --
        # so an absolute 0.6 is not a conservative threshold, it is an
        # unreachable one for ~95% of runs. That is why the existence
        # audit reads `formulas` INERT: the engine is correct and simply
        # never fires. Evolution was not disconnected, it was asleep.
        #
        # So gate on stress that is high FOR THIS RUN -- a new stress peak
        # -- which is what "the system is struggling, invent new
        # structure" was always meant to mean. `absolute` stays an upper
        # bound, so a saturating run behaves exactly as before, and the
        # floor keeps a flat, calm run from evolving on numerical noise.
        # Firing is still rate-limited to one birth per channel per
        # EVOLUTION_MIN_GAP stages, and E3's prune_to_cap still bounds the
        # library, so this cannot reintroduce the birth runaway.
        absolute = float(getattr(self.cfg, "evolution_stress_gate", 0.6))
        floor = float(getattr(self.cfg, "evolution_stress_floor", 0.05))
        if getattr(self.cfg, "evolution_absolute_gate_only", False):
            stress_fire = system_stress >= absolute
        else:
            effective_gate = min(absolute, max(floor, self._stress_peak))
            stress_fire = system_stress >= effective_gate
        self._stress_peak = max(self._stress_peak, float(system_stress))
        # Stagnation is powerful but destabilizing: a single birth can
        # trigger a pathological slowdown in fit_weighted (adult hung on
        # the first stagnation birth), and on a plateaued run it fires
        # every stage. It is therefore OPT-IN via cfg.enable_stagnation_
        # evolution (default off = original stable behavior). When on, the
        # rate limiter below still bounds births to one per EVOLUTION_MIN_GAP.
        stagnation_fire = (
            getattr(self.cfg, "enable_stagnation_evolution", False)
            and self._stages_since_improvement >= 2
        )
        if not (stress_fire or stagnation_fire):
            return False
        last = self._last_structure_stage.get(channel, -(10 ** 9))
        if stage - last < self.EVOLUTION_MIN_GAP:
            return False
        self._last_structure_stage[channel] = stage
        return True

    def _intervention_applicable(self, hypothesis) -> bool:
        """Can this classifier actually move the hypothesis's factor?

        action.penalty is always ours. config.* factors must be a knob
        the CURRENT classifier exposes — intervening on e.g.
        config.mutation_rounds_per_fit under XGBoost would silently
        no-op in apply_config and record a fake "no effect" trial.
        """
        from figpso_uil.cognition.causal_intervention import controllable_factor

        factor = controllable_factor(hypothesis.variable)
        if factor is None:
            return False
        if factor == "action.penalty":
            return True
        param = factor.split(".", 1)[1]
        if hasattr(self.classifier, "get_controllable_params"):
            try:
                return param in self.classifier.get_controllable_params()
            except Exception:
                return False
        if hasattr(self.classifier, "get_current_config"):
            try:
                return param in self.classifier.get_current_config()
            except Exception:
                return False
        return False

    def _get_intervention_value(
        self, action: dict, variable: str
    ) -> float | None:
        """Live value of an intervention variable for this action."""
        if variable == "action.penalty":
            v = action.get("penalty")
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None
        if variable.startswith("config."):
            param = variable.split(".", 1)[1]
            cfg = action.get("classifier_config") or {}
            source = None
            if param in cfg:
                source = cfg[param]
            elif hasattr(self.classifier, "get_current_config"):
                source = self.classifier.get_current_config().get(param)
            try:
                return float(source) if source is not None else None
            except (TypeError, ValueError):
                return None
        return None

    def _set_intervention_value(
        self, action: dict, variable: str, value: float
    ) -> dict:
        """Pin a causal hypothesis's variable to a specific value.

        Used by the V20+.5 causal-intervention wiring in fit() to build
        the treatment action of a control/treatment pair. Only variables
        UIL actually controls are ever intervened on (action.penalty or
        a config.* hyperparameter) — state.* variables describe the
        dataset itself and must never be touched (see full.txt's
        "UIL must not manipulate the dataset" principle).
        """
        if variable == "action.penalty":
            action["penalty"] = float(value)
        elif variable.startswith("config."):
            param = variable.split(".", 1)[1]
            cfg = dict(action.get("classifier_config", {}))
            cfg[param] = float(value)
            action["classifier_config"] = cfg
        return action
