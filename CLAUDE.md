# UIL (Universal Intelligence Layer) — Developer Reference

## What this project is

UIL is a cognitive meta-learning system layered on top of standard ML classifiers (XGBoost, RandomForest, KNN, TabPFN, SymbolicFormula). It watches the classifier's performance over time, discovers causal rules about why it succeeds or fails, and applies those rules to improve future runs — without human labeling.

Two generations coexist in this codebase:

- **v19–v21 (the 20-part architecture)** — the original PSO→journal→miner→library→executor loop plus 19 cognitive extensions. Documented in `full.txt` (14,465 lines). All 20 parts implemented.
- **v22 "The Editable Scientist" (M0–M5)** — event-sourced knowledge history, dataset fingerprinting + domain routing, epistemology repairs, symbolic regression, and human-in-the-loop editability. Documented in `UIL_V22_SCIENTIST_BLUEPRINT.md`. All six milestones implemented and passing.

**Honest empirical status** (multi-benchmark, held-out, cross-seed): the cognitive layer **ties plain XGBoost+PSO on accuracy** for saturated tabular data (mean Δ ≈ +0.004 across small/medium/big/text datasets). **Compute, measured 2026-08-05 (`results/compute_cost/SUMMARY.md`) — the old "10–100×" line was unsupported and wrong in both directions:** the cognitive layer itself costs **0.95×** (`uil_enabled` ON vs OFF over the same 20 stages: 519s vs 547s), while the **search loop** costs **~3900×** one ordinary `fit_weighted` of the same learner (47× / 3794× / 7997× on breast_cancer / wine / ionosphere). And that compute buys nothing measurable: paired over 9 (dataset, seed) cells, UIL-20-stages minus one plain fit is **−0.0087, 95% CI [−0.0193, +0.0019]**, 1 win / 3 tie / 5 lose — **a +0.01 benefit is statistically excluded**. Its defensible value is (a) ~~the symbolic/extrapolation domain, where symbolic regression categorically beats XGBoost out-of-range~~ **— NARROWED 2026-08-05, see `results/extrapolation/SUMMARY.md`. Beating XGBoost out of range is a theorem about piecewise-constant models, not a finding. Cross-seed (7 laws × 5 seeds × 3 extrapolation reaches), UIL's symbolic regressor beats the best *extrapolating* baseline in **0 of 21 cells**. Against **`gplearn`** — an off-the-shelf GP symbolic regressor, same model family — UIL loses **4–17**, and blows up (predictions past 1000× the training scale, R² ≈ −1e35) on **19 seed-runs vs gplearn's 0**; **The THESIS is vindicated and the IMPLEMENTATION is not:** gplearn beats every non-symbolic baseline on the hard long-reach cells (harmonic 0.863 vs MLP 0.463, kinetic 0.735 vs MLP 0.448), so symbolic regression genuinely IS the right tool for out-of-range recovery — UIL's own regressor is just a weak instance of it. **Operator guards were the obvious hypothesis and were TRIED AND REJECTED (`results/extrapolation/GUARD_ABLATION.md`, 2026-08-05): no setting of the four constants dominates — a tight `MAX_MAGNITUDE` halves blowups (19→9) but destroys working cases (gravity@1.5 0.743→−1.167), and relaxing it to recover them makes blowups WORSE than no guards (22). The head-to-head stays 4–17 in every arm. Defaults are reverted; the constants are env-overridable and `tests/test_operator_guards.py` G2 pins them. The gap is term SELECTION (greedy residual fitting picks terms that fit in-range with wrong asymptotics) vs gplearn's population search with parsimony pressure — not arithmetic hygiene.** — and (b) inspectable, replayable, human-editable knowledge (VersionStore + HITL), **whose renderer was itself unfaithful until bug #32 and is now round-trip tested**. Don't pitch or re-test raw accuracy; the answer is stable.

---

## How to run

```bash
# From c:\Users\arinb\Downloads\ (the package parent directory)

# v19–v21 architecture tests
python figpso_uil/tests/test_v21_smoke.py        # Full architecture smoke test
python figpso_uil/tests/test_causal_and_seed.py  # Causal engine + seed knowledge
python figpso_uil/tests/test_editable_network.py # Knowledge graph
python figpso_uil/tests/test_comprehensive.py    # 4-dataset end-to-end
python figpso_uil/tests/test_stress_run.py       # Hard dataset stress test

# v22 milestone acceptance tests (all must pass)
python figpso_uil/tests/test_m0_epistemology.py  # k-fold regret, tiny mining, multiclass guard
python figpso_uil/tests/test_m1_version_store.py # Event-sourced lossless history
python figpso_uil/tests/test_m2_perception.py    # Fingerprint + invertible normalization
python figpso_uil/tests/test_m3_routing.py       # Domain router parity
python figpso_uil/tests/test_m4_scientist.py     # Active experiments + Feynman rediscovery
python figpso_uil/tests/test_m5_hitl.py          # Replay, lab reports, human edits
python figpso_uil/tests/test_multiclass_ovr.py   # Multiclass OvR: calibration + no collapse (bug #24)
python figpso_uil/tests/test_rule_basis.py       # Rule basis: readable interactions, backcompat (E5)
python figpso_uil/tests/test_law_fidelity.py     # Readable law IS the model: round-trip + editability (bug #32)
python figpso_uil/tests/test_determinism_matrix.py # Modes x arities x processes x thread counts (M1-M5)

# Re-score stored Gate 2 verdicts with paired stats + power (no new compute)
python -m figpso_uil.experiments.rescore_gates --out results/rescore_gates/REPORT.md

# Graded existence audit (share of predictions moved, across datasets x budgets)
python -m figpso_uil.experiments.existence_sweep --datasets breast_cancer,wine_binary --stages 5,20

# Out-of-range symbolic eval vs baselines that CAN extrapolate (not just XGBoost)
python -m figpso_uil.experiments.eval_extrapolation --seeds 42,7,123,5,99 --reach 1.2,1.5,2.0

# What the search loop costs vs one ordinary fit, and whether it buys anything
python -m figpso_uil.experiments.compute_baseline --uil-json results/compute_cost/uil_on.json

# Graded existence audit (share of predictions moved, across datasets x budgets)
python -m figpso_uil.experiments.existence_sweep --datasets breast_cancer,wine_binary --stages 5,20

# Out-of-range symbolic eval vs baselines that CAN extrapolate (not just XGBoost)
python -m figpso_uil.experiments.eval_extrapolation --seeds 42,7,123,5,99 --reach 1.2,1.5,2.0

# What the search loop costs vs one ordinary fit, and whether it buys anything
python -m figpso_uil.experiments.compute_baseline --uil-json results/compute_cost/uil_on.json

# Cross-seed evaluation — the ONLY sanctioned way to quote accuracy (never a single split)
python -m figpso_uil.experiments.eval_crossseed   # mean±std acc + macro-F1 + val-heldout gap, SymbolicFormula only

# Run the web UI (from c:\Users\arinb\Downloads\)
python -m uvicorn figpso_uil.webui.backend.main:app --port 8000
# Open http://localhost:8000

# Run the DESKTOP app (native window via pywebview; needs `pip install pywebview`)
python figpso_uil/desktop.py
# Same backend + the modern animated frontend at /desktop/ (webui/desktop/).
# Falls back to the default browser if pywebview is missing.
```

---

## Project layout

```
figpso_uil/
├── core/
│   ├── optimizer.py           # UniversalOptimizer — main orchestrator (~2400 lines; fit() is the 18-step loop)
│   ├── failsafe.py            # FailsafeEngine — classifier emergency fallback
│   ├── classifier_registry.py # {name → adapter factory}: XGBoost, RandomForest, KNN, MLP, SymbolicFormula; TabPFN and GPLearn register only when their optional dep is installed
│   ├── classifier_protocol.py # ClassifierAdapter protocol
│   ├── strategy_classifiers.py# Domain-router strategy → adapter mapping (v22 §3)
│   ├── normalization.py       # Invertible per-feature transforms with provenance (v22 §2.2)
│   ├── missing_data.py        # Train-only median/mode impute + missingness indicators (predict reuses same forward)
│   ├── uil_mode.py            # Master uil_enabled off-switch + experimental uil_connect_pack bundle
│   ├── resource_budget.py     # Machine resource detection → workload scaling
│   ├── small_data_diversify.py# Bootstrap + jitter for tiny datasets
│   └── fig_xgboost.py         # Backward-compatible FIG_XGBoost_Embedded wrapper
├── adapters/
│   ├── xgboost_adapter.py     # Full hyperparameter control; custom objective is BINARY-ONLY (guarded)
│   ├── random_forest_adapter.py
│   ├── knn_adapter.py
│   ├── tabpfn_adapter.py      # Pre-trained tabular transformer (no per-dataset training)
│   ├── gplearn_adapter.py     # Genetic-programming symbolic learner (optional dep; registered as "GPLearn")
│   └── symbolic_formula_adapter.py # Formula-as-classifier: single|boosting|bagging|multi|auto modes
├── cognition/
│   ├── uil_journal.py         # Persistent stage-by-stage record
│   ├── uil_stress.py          # Stress + regret computation
│   ├── uil_miner.py           # Decision-tree rule miner (held-out R²)
│   ├── uil_library.py         # Principle library (persistence + tier mgmt)
│   ├── uil_executor.py        # Evaluates principles against current state
│   ├── editable_network.py    # Knowledge graph with typed edges
│   ├── causal_intervention.py # Causal hypothesis testing (majority-of-3)
│   ├── episodic_attention.py  # Case-based memory retrieval
│   ├── investigation_engine.py# 5-strategy regret investigation
│   ├── principle_hierarchy.py # 4-tier promotion system
│   ├── journal_forgetting.py  # Exponential decay over journal entries
│   ├── meta_principle_merger.py # Cluster similar rules into meta-principles
│   ├── structural_representation.py # Structural encoding + vocabulary
│   ├── abstraction_safeguards.py # 5-gate promotion guard
│   ├── formula_engine.py      # Symbolic formula evolution + mutation (loss formulas)
│   ├── formula_library.py     # Tournament-selection library for formulas
│   ├── execution_context.py   # Shared context for executor + network overrides
│   ├── failure_point.py       # Diagnostic capture before fail-loud propagation
│   ├── guard_telemetry.py     # Observe-only bind-rate log for safety rails (audit Part E step 2)
│   ├── live_feed.py           # results/uil_live_feed.jsonl — terminal runs visible in web UI
│   ├── paths.py               # Results directory anchor
│   ├── signal_translator.py   # Cognition → LearnerDirective (particles / formulas / config)
│   │   # ── v22 modules ──
│   ├── knowledge_cell.py      # KnowledgeCell protocol — smallest self-modifiable unit (§1.1)
│   ├── cell_store.py          # Composite VersionStore snapshot + materialized views
│   ├── version_store.py       # Event-sourced lossless history; replayable at any seq (§1.2)
│   ├── change_log.py          # Timestamped graph/formula change logs
│   ├── fingerprint.py         # Five-axis orthogonal statistical fingerprint (§2.1)
│   ├── domain_router.py       # Meta-controller: fingerprint → learner strategy (4 domains)
│   ├── router_tuning.py       # Router threshold re-calibration on normalized fingerprints
│   ├── knowledge_archive.py   # Nothing-is-lost archive for pruned working state
│   ├── working_memory_pruner.py # Prunes journal/principles/hypotheses into the archive
│   ├── dataset_champion.py    # Per-dataset champion checkpoints — resume accuracy across runs
│   ├── symbolic_regressor.py  # Greedy residual-guided symbolic regressor — the D2 math learner
│   ├── symbolic_gradient.py   # Gradient-fit for symbolic weak learners (grad_fit)
│   ├── llm_mutator.py         # Mutation proposals (§5.2) — dual-loop: external LLM (System 2, needs API key) → neural_proposer (System 1, offline) → raw engine ops; proposals only, never writes state
│   ├── neural_proposer.py     # R1 §5.4 stage 1 — own tiny transformer (pure numpy, open-box JSON weights) trained on the system's own formula history; generate-and-rank over the formula grammar
│   ├── law_export.py          # Faithful model → executable law + independent evaluator (bug #32)
│   ├── lab_report.py          # Lab reports compiled from the VersionStore event stream (§6.1)
│   ├── method_templates.py    # Formula templates from well-known ML methods
│   ├── bootstrap_knowledge.py # Clean seed knowledge without fake journal rows
│   └── memory_reset.py        # Reset all persisted UIL memory to empty defaults
├── search/
│   ├── principle_guided_search.py # Hybrid PSO + reasoning search
│   ├── hybrid_foundation.py   # PSO detachment (maturity stages)
│   └── pso_search.py          # Core PSO implementation
├── experiments/               # Ablations, benchmarks, research runs (see ablation harness below)
├── webui/backend/             # FastAPI + WebSocket UI (app, run_manager, ws_hub, routes/)
├── tests/                     # v19–v21 tests + M0–M5 acceptance suites
├── results/                   # All persisted state (see table below)
├── full.txt                   # 14,465-line v19–v21 design spec
└── UIL_V22_SCIENTIST_BLUEPRINT.md # v22 design spec (§ references above point here)
```

---

## The 20-part architecture (v19–v21)

All parts implemented. Audited July 2026.

| Part | Name | File | Status |
|------|------|------|--------|
| 1 | V19 Foundation (PSO→journal→miner→library→executor loop) | `core/optimizer.py` | ✅ |
| 2 | Dual Stress & Regret | `cognition/uil_stress.py` | ✅ |
| 3 | Full Classifier Control (XGBoost gamma/reg_alpha/reg_lambda/min_child_weight) | `adapters/xgboost_adapter.py` | ✅ fixed |
| 4 | Formula Engine (7 mutations, NaN/Inf/gradient/Hessian safety) | `cognition/formula_engine.py` | ✅ |
| 5 | Within/Cross-Dataset Mining (cross rules penalised 0.75×) | `cognition/uil_miner.py` | ✅ fixed |
| 6 | Principle Hierarchy (4-tier, promotion/demotion/quarantine) | `cognition/principle_hierarchy.py` | ✅ |
| 7 | Derived Feature Synthesis (held-out R² gating) | `cognition/uil_miner.py` | ✅ fixed |
| 8 | Editable Network (lift-based edges, causal gate traversal, decay) | `cognition/editable_network.py` | ✅ fixed |
| 9 | Investigation Engine (5 escalating strategies) | `cognition/investigation_engine.py` | ✅ |
| 10 | Causal Intervention Engine (majority-of-3 for CONFIRMED_CAUSAL) | `cognition/causal_intervention.py` | ✅ fixed |
| 11 | Seed Knowledge (confidence=0.5, duplicate rejection) | `cognition/uil_library.py` | ✅ |
| 12 | Journal Forgetting (exponential decay, window 500) | `cognition/journal_forgetting.py` | ✅ |
| 13 | Episodic Memory (KDE fingerprint retrieval, warm-start) | `cognition/episodic_attention.py` | ✅ |
| 14 | Graph Attention (top-k edge scoring) | `cognition/editable_network.py` | ✅ |
| 15 | PSO Detachment (4 maturity stages, rolling-10 accuracy) | `search/hybrid_foundation.py` | ✅ fixed |
| 16 | Causal Co-activation (tracking + attention-bonus propagation) | `cognition/editable_network.py` + `core/optimizer.py` | ✅ fixed |
| 17 | Structural Representation | `cognition/structural_representation.py` | ✅ |
| 18 | Meta-Principle Merging (persists across runs) | `cognition/meta_principle_merger.py` | ✅ fixed |
| 19 | Abstraction Safeguards (5 gates) | `cognition/abstraction_safeguards.py` | ✅ |
| 20 | Full Pipeline (18-step cycle wired in `fit()`) | `core/optimizer.py` | ✅ |

## The v22 milestones (merged into this copy 2026-07-14)

| Milestone | Name | Key files | Status |
|-----------|------|-----------|--------|
| M0 | Epistemology repair — k-fold regret, tiny-data mining, multiclass capability guard, skill-keyed detachment | `core/optimizer.py`, `adapters/xgboost_adapter.py`, `cognition/uil_miner.py` | ✅ |
| M1 | VersionStore — event-sourced lossless history, byte-identical replay, belief trajectories | `cognition/version_store.py`, `cognition/change_log.py` | ✅ |
| M2 | Perception — five-axis fingerprint + invertible normalization + router tuning | `cognition/fingerprint.py`, `core/normalization.py`, `cognition/router_tuning.py` | ✅ |
| M3 | Routing — domain router as step 0 of `fit()` (fingerprint → 4 domains → strategy) | `cognition/domain_router.py`, `core/strategy_classifiers.py` | ✅ |
| M4 | Scientist core — active experiment design, symbolic regressor, Feynman-law rediscovery, LLM mutation proposals | `cognition/symbolic_regressor.py`, `cognition/llm_mutator.py`, `cognition/causal_intervention.py` | ✅ |
| M5 | HITL — replay/reconstruct at any seq, lab reports, human edit/reject ops | `cognition/lab_report.py`, `cognition/version_store.py`, webui | ✅ |

Acceptance: `tests/test_m0_epistemology.py` … `test_m5_hitl.py` — all passing as of 2026-07-14.

---

## Bugs fixed in development (do not reintroduce)

### 1 — XGBoost silent no-ops (`adapters/xgboost_adapter.py`)
`gamma`, `reg_alpha`, `reg_lambda`, `min_child_weight` were not in `self.params`, so `apply_config()`'s `key in self.params` guard silently dropped them. All 4 are now seeded with XGBoost defaults in `_build_params()`.

### 2 — Miner circular R² (`cognition/uil_miner.py`)
`tree.score()` was called on the same data as `tree.fit()`. Fixed: 75/25 held-out split when `len(X_meta) >= 40`; feature synthesis also gated on held-out improvement > 0.05.

### 3 — Causal single-trial (`cognition/causal_intervention.py`)
A single control/treatment comparison declared `CONFIRMED_CAUSAL`. Fixed: requires majority of last 3 trials (`_strong_count >= 2`).

### 4 — Maturity sticky-peak bug (`search/hybrid_foundation.py` + `core/optimizer.py`)
`update_maturity()` was called with the all-time peak accuracy, hitting EXPERT early and dropping PSO permanently. Fixed: rolling mean of last 10 stages.

### 5 — Network all-contradicts bug (`cognition/editable_network.py`)
On hard datasets, absolute thresholds classified all edges as CONTRADICTS. Fixed: lift-over-base-rate classification (base_rate passed from optimizer).

### 6 — Co-activation tracking without propagation (Part 16)
`_co_activation_tally` accumulated but nothing read it. Fixed: `get_co_activation_bonus()` wired in `optimizer.py` (up to 15% confidence boost per stage).

### 7 — Cross-dataset confidence not penalised (Part 5)
Fixed: 0.75× multiplier applied in `optimizer.py` before `library.update_principles(cross_rules)`.

### 8 — Meta-principles ephemeral (Part 18)
Fixed: `_save()`/`_load()` added, default path `results/uil_meta_principles.json`. `MetaPrincipleMerger(filepath=None)` disables persistence (use in tests).

### 9 — run_manager read-without-lock race (`webui/backend/run_manager.py`)
`to_summary()` read `last_snapshot` without `self.lock` while the worker wrote with it. Fixed: read wrapped in `with self.lock`.

### 10 — Multiclass formula crash (M0.3, `adapters/xgboost_adapter.py`)
`supports_custom_objective` returned `True` unconditionally, so on `num_classes > 2` the binary-only formula gradient received `(n, k)`-shaped `y_pred` and crashed with a broadcast `ValueError`. Never caught pre-v22 because every test used binary synthetic data. Fixed: `supports_custom_objective` returns `self.num_classes <= 2`, double-guarded in `fit_weighted()`. The old webui workaround (silently swapping XGBoost→RandomForest on multiclass) was removed from `run_manager.py` — users get the classifier they asked for.

### 11 — VersionStore unsynchronized singleton (`cognition/version_store.py`)
`VersionStore.shared()` is appended to from webui worker threads, and a stopped run's thread can overlap the next run's for up to a full stage (stop_event is only checked at stage boundaries). Unguarded `_last_seq += 1` + segment appends could produce duplicate seqs / interleaved JSONL. Fixed: `threading.RLock` around `append()` and `write_snapshot()`, lock-guarded `shared()` construction.

### 12 — Router swap desyncs the parallel-worker factory (`core/optimizer.py`)
The M3 router replaces `self.classifier` at step 0 but left `self._classifier_factory` building the PRE-route type, so parallel action workers returned models the routed adapter couldn't predict with (`'dict' object has no attribute 'predict'` when SymbolicFormula workers fed XGBoostAdapter). Fixed: the routing block rebinds the factory to `build_classifier_for_strategy` for the routed strategy. Additionally, the webui sets `cfg.disable_routing = True` — a user's explicit classifier choice must not be silently replaced (route decision is still computed and recorded).

### 13 — SymbolicFormula classifier boosting never wired to grad_fit (`adapters/symbolic_formula_adapter.py`)
The classifier's boosting loop still used the legacy constant-step update (`learning_rate * (score−0.5) * 4.0`) with unfitted library formulas, producing near-constant logits on real data — every prediction collapsed to the majority class (sonar 0.46, breast_cancer 0.63 = exact majority rates). `cognition/symbolic_gradient.py` (grad_fit: constant refit vs pseudo-residual + normalize + least-squares scale + Newton line search) existed but was only imported by the regression path. **`experiments/smoke_symbolic_modes.py` was blind to this**: its unstratified sequential split has a 77.2% majority class, so all-majority predictions scored "0.772 acc" and looked fine. Fixed: `_fit_boosting` uses `fit_boost_stage` (stages carry a `grad_fit` params dict), predict uses `stage_contribution`; legacy stage dicts without `grad_fit` still predict via the old path. After (single-split, at fix time): sonar 0.83, breast_cancer 0.92, ionosphere 0.91. New boosting-stage code must go through grad_fit, and any accuracy smoke test must check the prediction distribution, not just accuracy. **CORRECTION (2026-07-18 cross-seed audit):** those were single-split numbers and OVERSTATE. Cross-seed (mean±std over seeds 42/7/123, full budget): sonar boosting **0.746±0.030** (a genuine weak spot, not 0.83), breast_cancer sparse_linear 0.974±0.012, ionosphere sparse_linear 0.916±0.030. **Never quote a single split — use `experiments/eval_crossseed.py`, which reports mean±std + macro-F1 + the val-heldout optimism gap.** Small datasets (sonar, wine) show self-reported val running ~0.03 OPTIMISTIC vs true held-out (inherent to CV on small n); trust the held-out number.

### 14 — Causal scientist starved by untestable hypotheses (`cognition/causal_intervention.py` + `core/optimizer.py`)
28/42 hypotheses were on `derived.a*b` product variables (miner feature synthesis), which the intervention scheduler's `action.penalty`/`config.*` prefix filter permanently excluded — hypotheses were born untestable and died pending (0 confirmed causal in the system's entire history). Fixed: `controllable_factor()` decomposes derived variables to their first controllable factor; `get_testable_hypotheses()` feeds only testable ones (incl. stuck-TESTING) to the EIG chooser; derived plans use scale semantics (halve/double the factor) resolved against the live action value; `_intervention_applicable()` skips factors the current classifier doesn't expose (would silently no-op → fake "no effect" trials). After: 1 trial per stage, statuses actually move (pending 37→31, correlational 1→10, refuted 0→1 in one 6-stage run).

### 15 — VersionStore torn-tail write breaks recovery (`cognition/version_store.py`)
A force-killed process (e.g. stopping the webui server) can leave NUL padding / a partial JSON line at the end of the newest event segment. `_read_last_seq` parsed the raw last line → `JSONDecodeError` → the entire store failed to initialize, and `events()` would poison replay. Fixed: `_repair_torn_tail()` truncates trailing garbage back to the last valid newline-terminated event on init (standard WAL recovery — valid history never touched), and both readers NUL-strip + skip unparsable lines.

### 16 — Champion resume returns a null model on a non-improving run (`core/optimizer.py`)
`DatasetChampionStore` restores the accuracy BAR (`self._best_accuracy = max(..., champion.best_accuracy)`) at step 0, but for classifiers whose model can't be JSON-persisted (XGBoost Booster), `champion["model"]` is absent, so `_best_model` stays None. The stage update only set `_best_model` when `best_action_score > self._best_accuracy`, so a run that merely TIED the champion returned `result["model"] is None` with a high accuracy — and `predict()` then raised. Surfaced by `test_v19_integration` TEST 9 (3-class XGBoost, routing on, resuming a champion). Fixed: capture the trained model whenever `_best_model is None and best_action_model is not None`, without lowering the accuracy bar. (Editability/HITL note: human edits must go through BOTH `record_cell` AND `apply_materialized_edit` — snapshots re-materialize from the library files, so an event-only edit is overwritten at the next snapshot. The webui `edit_cell` route does both; verified an edit survives replay + a fresh snapshot.)

### 17 — String thresholds crash numeric similarity checks (v22 routing rules)
`record_routing_rule` (M3) mints principles whose conditions carry **string** thresholds (`dataset.name ==`, `route.domain ==`), but three v19-era duplicates of the "conditions similar/overlap" logic all did numeric `abs()` math on thresholds: `uil_library._conditions_similar`, `editable_network._conditions_similar`, `journal_forgetting._conditions_overlap`. Any run after the first routing rule persisted → `TypeError: bad operand type for abs(): 'str'`. Never caught by M-tests because they use isolated `state_dir` (at most one routing rule per store). Fixed in all three: non-numeric thresholds compare by exact equality. If you add another similarity check, handle string thresholds.

### 18 — Knowledge-graph traversal explosion + unbounded growth (the "142-minute hang")
Three compounding problems, found via `faulthandler` stack dumps after a benchmark run hung for 142 min on UCI adult:
1. **`traverse_with_attention` frontier explosion** (`editable_network.py`): `queue.pop(0)` on a list (O(n) per pop), NO enqueue guard (weight/depth cutoffs checked only at pop, after the queue already exploded), and `strengthens` edges boost weights ×1.3 (can grow, not shrink). On a dense graph (~50k edges / 535 nodes ≈ avg degree 92) this enumerates weighted paths — effectively exponential. Fixed: `deque.popleft()`, pre-enqueue guard applying anticipated revisit-decay + depth cutoff, and a hard expansion cap `max(4000, 8·nodes)`.
2. **Working graph grows without bound in short runs**: `prune_working_graph` only ran at `stage % 15`, which 4–8-stage runs never reach; `discover_relationships` adds thousands of co-activation edges per run → the shared graph hit 72k edges across sessions. Fixed: load-time prune at `fit()` start when `edges > 6000`, plus the same size-triggered prune on the 5-stage cadence. Everything archived; nothing lost. Bonus: adult held-out acc went 0.771→0.815 after pruning — the mega-graph's noisy overrides were actively hurting.
3. **Bulk prune caused a VersionStore event storm**: pruning 68k edges emitted one event per edge (68k file appends, a full 33k-cell composite snapshot every 500th) — the prune hung inside logging, not pruning. Fixed: batches >200 emit ONE `_bulk_log` summary event; the KnowledgeArchive holds the full copies and the next periodic snapshot re-materializes canonical state.

(Related revision: E2's stagnation-triggered evolution is now **opt-in** via `cfg.enable_stagnation_evolution`, default off, with an `EVOLUTION_MIN_GAP=3` rate limiter when enabled — it was initially misdiagnosed as the hang's cause; the graph was the real culprit, but plateau-triggered every-stage births remain a runaway risk on their own.)

### 19 — `predict()` skipped normalization → train/serve mismatch (`core/optimizer.py`)
`fit()` fits the v22 §2.2 invertible normalization at step 0 and trains on `self._normalization.forward(X_train)`, but `predict()` passed **raw** X straight to `classifier.predict`. The model was therefore trained in normalized space and served out-of-distribution on every unseen row (bank_marketing: trained on mean 0.57 / std 3.11 / max 94.6 robust-median-MAD space, served mean 0.0 / std 1.0 / max 42.7). Silently corrupted **every held-out number** the benchmarks reported. Fixed: `predict()` applies `self._normalization.forward(X)`; `_normalization` initialized to None in `__init__` so predict is safe pre-fit. **Invariant: any new inference path must apply the same forward transform** — the normalization is only invertible/meaningful if both sides use it.

### 20 — Accuracy-maximizing threshold collapses on imbalanced data (`adapters/symbolic_formula_adapter.py`)
`_fit_threshold` maximized plain accuracy over a threshold grid. At a high base rate that is degenerate: on bank_marketing (88.3% majority) the best signal-using threshold scored train_acc 0.8815 while all-majority scored 0.8830 — so the optimum was to **throw the signal away and emit a single-class model** (AUC 0.60–0.65 but balanced acc exactly 0.500, `pred_classes=1`). Not a wiring bug: verified identical under uniform weights, PSO-like weights, and chained warm-starts, so neither PSO nor the optimizer caused it; only balanced datasets (sonar 54/46, breast_cancer 63/37) had hidden it. Fixed: threshold objective switches to **balanced accuracy** when the majority class ≥ `IMBALANCE_CUTOFF` (0.75). After: predicts both classes, balanced acc 0.500→0.521, at a ~0.002 accuracy cost — the honest trade, since a model that always answers "no" is useless whatever its accuracy.

### 21 — Router-selected mode silently overridden by PSO + champion (`adapters/symbolic_formula_adapter.py`, `search/principle_guided_search.py`)
Two independent paths were silently discarding an explicit `learning_mode`: (a) `principle_guided_search.py` proposed a new mode every stressed stage from a **hardcoded list** `["boosting", "bagging", "multi"]` that didn't include `sparse_linear`, so a router-selected sparse_linear survived one fit at most; (b) `DatasetChampionStore` restored `best_config` at fit() step 0 (optimizer.py:584), silently propagating a stale `learning_mode` from a prior run. Result: spambase kept scoring 0.685 (boosting fallback) despite `apply_config({"learning_mode":"sparse_linear"})` being called. Fixed: constructor & `apply_config` set `_mode_locked` on any explicit non-auto mode; `apply_config` skips the `learning_mode` field while locked; `get_controllable_params` only advertises `learning_mode` while it is `"auto"` (so PSO has nothing to override). Verified end-to-end: full optimizer on spambase now runs sparse_linear and hits **0.9486 (beats XGBoost 0.9435)**, up from 0.685. **Invariant: any architectural knob (mode, basis-family gates) set explicitly by router/operator must not be resettable by per-stage search or champion restore — those are hyperparameter channels, not architecture channels.**

### 25 — **The cognitive layer never reaches the model** (`adapters/symbolic_formula_adapter.py::_basis_matrix` / `_fit_sparse_linear`) — DIAGNOSED, NOT YET FIXED
**This explains the project's central negative result.** Measured 2026-07-23 with the codebase's own ablation switch: FULL cognition (principles+network+causal+formulas) **0.9561** vs ALL ABLATED (pure PSO) **0.9561** → **delta +0.0000**. And a HITL A/B (9 paired runs, breast_cancer/ionosphere/wine × 3 seeds, deterministic so any delta is causal) where a "human" deleted **8 of 9 principles** every run: **9/9 delta exactly +0.0000** (helped 0, hurt 0). Deleting the ENTIRE principle library is byte-identical too.

Full chain: principles/graph emit **only `max_penalty`** → `optimizer.py:2564` caps `action["penalty"]` → `principle_guided_search` shapes PSO **feature_weights** → `SymbolicFormula.fit_weighted(X, y, feature_weights)` → **dead end**, via two independent, separately-proven breaks:
1. **Dead variable.** `_basis_matrix` computes `X_eff = self._prepare_X(X, feature_weights)` and **never references it again** (exactly one occurrence = the assignment); the `raw`/`stump`/`rule` branches read the *unweighted* `X`. Proven: the basis is bit-identical for weights `[1]*6` vs `[1.0, 0.01, 0.01, 0.01, 0.01, 0.01]`.
2. **Standardization cancels scaling.** Even if `X_eff` were used, `_fit_sparse_linear` z-scores the basis (`Bz=(B−mu)/sd`); scaling a column by a constant then z-scoring is the identity map. Proven numerically.

So feature weights — the cognitive layer's ONLY channel into the learner — cannot influence the fit. Same bug CLASS as #1 (silent no-op), at the cognition→learner boundary.

**WHICH normalization is the blocker — isolated (breast_cancer, ablation delta):** `A` norm ON/fix OFF → **+0.0000** (dead); `B` **norm OFF**/fix OFF → **+0.0000** (STILL dead); `C` norm ON/**fix ON** → **+0.0175** (live, helps); `D` norm OFF/fix ON → **−0.0088** (live, hurts). So the **optimizer input normalization (`core/normalization.py`) is NOT the blocker** — the basis z-score in `_fit_sparse_linear` is. And **normalization is beneficial**: with the layer live, keeping it helps (+0.0175), removing it hurts (−0.0088). **Do NOT "strip normalization to let cognition through"** — the fix must make the weights ride INSIDE the z-score, never bypass it. Best config = norm ON + fix ON (0.9737).

**FIX IMPLEMENTED + VALIDATED — VERDICT: DO NOT SHIP AS DEFAULT (both flags OFF).** `adapters/symbolic_formula_adapter.py` gained `use_feature_weight_penalty` + `_basis_weight_vector()` (raw/stump take their own feature's weight, rule averages its conjunction, formula neutral; `WEIGHT_FLOOR=0.25` bounds it) folded into stored `sd` so `predict()` needs no change; `core/optimizer.py` gained `cfg.connect_cognition` (default False). **Gate 1 PASSED** — breast_cancer ablation delta 0.0000 → **+0.0175**; unit proof: good hint → 6 terms (drops a noise feature), bad hint → train acc 0.9225→0.8875, so the channel is live both ways. **Gate 2 FAILED** (3 seeds × 6 datasets): sonar **+0.0317** (0.754→0.786), breast_cancer +0.0058, spambase −0.0018, wine −0.0093, **digits −0.0111**, **ionosphere −0.0141**; mean **+0.0002** (improved 1 / unchanged 3 / regressed 2). **Pattern: cognition helps most where the learner is WEAKEST (sonar) and hurts where it's near ceiling (digits/ionosphere) ⇒ apply conditionally, never globally.** Measurement caveat: gate-1 ionosphere differed across two runs of identical code+seed (−0.1127 vs +0.0000) — unexplained cross-process variance; trust the 3-seed numbers.

**ROOT CAUSE OF THE BAD ADVICE (deeper than the wiring):** the entire cognitive apparatus emits exactly ONE key — `max_penalty` — from one principle (P-003) every stage (`uil_library.py` ~line 367 hardcodes it). 201 principles + causal hypotheses + graph edges all collapse into one scalar throttling the PSO feature penalty; "use fewer features" is precisely wrong on ionosphere/digits. The classifier already accepts `sparse_l1_c`, `use_stump_basis`, `use_rule_basis`, `boosting_stages`, `learning_mode` — the miner just never emits them.

**VERIFY-BEFORE-APPLY GATE: TRIED TWICE, FAILED, REVERTED (2026-07-23). Do not re-attempt without reading this.** Idea: re-score the winning action with cognition suppressed and keep the override only if it earns its place. (a) **v1** compared the winner (a MAX over N candidates, cognition applied) against a single plain evaluation *on the same selection split* — a winner's-curse comparison rigged toward keeping. Result: kept 36/36 on BOTH sonar (cognition helps) and ionosphere (cognition costs 0.014). Zero discrimination. (b) **v2** judged both variants on the independent cv resamples (`_score_extra_splits`, never used for selection). Still barely rejected (35/1) AND made things **worse than not gating** (ionosphere GATED 0.8732 < ON 0.8967) — because `_evaluate_one_action` is **NOT side-effect-free**: it spawns and fits a classifier that mutates shared adapter state (formula library writes, `_formulas_used_this_fit`, RNG). **A verification probe that mutates the run it measures is worthless.** (c) **Worse, the benefit it was built to capture is not real:** sonar's +0.0317 was measured at `cv_folds=1`; at `cv_folds=2` cognition has ZERO effect on sonar (OFF=ON=GATED=0.7540). So the single apparent win was a CV artifact.

**CONCLUSION: gating cannot rescue a one-knob vocabulary.** Even a clean, discriminating gate is choosing between two variants that differ only by `max_penalty` — too little signal for a noisy per-stage comparison. Fix the vocabulary FIRST (below); only then is a gate worth revisiting, and it must be built side-effect-free (score a cached model, never re-fit inside the loop).

**RECOMMENDED NEXT STEPS (in order):** (1) ~~verify-before-apply gate~~ **FAILED — see above** — keep a cognitive override only if it improves a held-out slice; converts the +0.0002 wash into "capture sonar's +0.032, reject the regressions" and makes the layer's contribution monotonic; (2) widen the mined-rule → action-override vocabulary beyond `max_penalty`; (3) re-measure with Gate 2 as the standing test.

**Superseded fix sketch:** using `X_eff` alone is insufficient (break #2 erases it). For an L1 model, scaling column *j* by *c* ≡ scaling its penalty by *1/c*, so apply weights **after** standardization (`Bz_j *= w_j`) and fold the factor into stored `sd` so `predict` stays consistent. This alters the learner's math on **every** `sparse_linear` fit — put it behind a flag and validate cross-seed on all six datasets before defaulting it on (see E6: an unvalidated "improvement" regressed ionosphere 0.916→0.868). **The ablation delta is now the instrument: it reads 0.0000 today; a real fix must move it off zero.**

### 24 — Multiclass one-vs-rest collapse (`adapters/symbolic_formula_adapter.py`)
`predict()` for `kind == "multiclass_ovr"` stacked **raw** per-head scores and argmaxed. Each OvR head is a different binary model on a different (1-vs-rest, heavily imbalanced) problem, and different `kind`s (sparse_linear/hybrid/boosting) emit scores on different scales — so raw scores are **not comparable across heads**, and argmax collapsed onto a few hot heads. Measured (full budget, held-out): digits emitted **3/10** classes → **0.214** acc; MNIST-784 4/10. Two-part fix: (1) **Platt-calibrate each head** at fit time — fit `sigmoid(a·score+b)` mapping the head's raw score → `P(class==c)` on the eval set (`_fit_platt`/`_apply_platt`), stored as `ovr_calibration` on the head; predict applies it before argmax so every class is reachable (digits 0.214→0.511, all 10 classes). (2) **Add a `sparse_linear` branch to the OvR loop** (`_fit_multiclass_ovr`) — it previously fell through to the weak single-formula `_fit_single`; now each head can be an L1 model over the composed basis (== multinomial-style logistic), lifting digits **0.511→0.947 (XGBoost 0.958)**. Old persisted models without `ovr_calibration` fall back to raw scores (back-compatible). Test: `tests/test_multiclass_ovr.py`. **Invariant: any OvR/argmax combiner over independently-fit heads must calibrate per-head scores to comparable probabilities first — raw scores from separate imbalanced fits are not comparable.**

### 23 — VersionStore reconstruct loaded EVERY snapshot (`cognition/version_store.py`)
`_load_snapshot_at_or_before` json.load()ed every snapshot file ≤ target instead of selecting the nearest first — with 6,935 snapshots (57 GB events dir) every `reconstruct()` call (webui `/state/cells`, `/state/replay`) read gigabytes and effectively hung. Same fix family: `events()` now skips whole segments outside `[start_seq, end_seq]` (segment n deterministically covers seqs `((n-1)·size, n·size]`). After: reconstruct at seq 3.47M takes 0.4s. Two companion webui bugs fixed the same day: `/state/cells` truncated rows BEFORE the cell_type filter (rare types like the single `goal` cell could never surface — filter the state dict first), and episode fingerprints carry NaN (`levene_f0` on constant features) which FastAPI's strict JSON encoder 500s on — `_json_safe()` now scrubs NaN/Inf in `routes/state.py`.

### 22 — Pruned formula tree crashes fit mid-stage (`adapters/symbolic_formula_adapter.py`)
`_pick_formula` selects a formula id from the library listing, then `_fit_single` fetches its tree — but between pick and fetch, the hybrid path fits branches in threads while pruning runs concurrently on the shared library, so `get_formula_tree` could return `None`. Crashed covertype mid-benchmark with `TypeError: Expected FormulaNode or dict tree, got NoneType`, killing the whole dataset. Fixed with `_tree_or_seed(formula_id, n_features)` fallback — a missing formula returns a seed instead of raising. A dropped tree is never worth aborting a run.

### 26 — Champion accuracy used as a GATE on the current run (`core/optimizer.py`)
`fit()` step 0 loaded the stored champion's accuracy straight into `self._best_accuracy` — the same variable that is the *save condition* (`improved = best_action_score > self._best_accuracy`). One variable, two jobs. A previous lucky run therefore locked the current one out: every stage did full work and was discarded, and `fit()` returned either the champion's model (when it serialized one) or **stage 1's** (via the `need_model` fallback added for bug #16) — neither of which is "the best model". Worse, the bar was measured on a **different random split**, so "can't beat 0.9123" often just meant "the old split was easier". Measured on breast_cancer, 3 sequential runs: OLD run3 reports **0.9123 == the stored bar exactly**, returning run2's model; NEW run3 reports its own **0.8772**, returns its own model, and declines to overwrite the champion. Fixed by splitting the variable: `_best_accuracy` is now this-run-only (starts at 0.0), the stored record lives in `_champion_bar`/`_champion_payload`, and the two are compared **once at the end of `fit()`**. Champion weights are still loaded as a warm start (that use was always correct). **Invariant: a persisted record must never be the gate on whether the current run may save its own work.** The comparison deliberately does NOT use `_X_heldout` — it is the one measurement nothing has selected against, and it stays that way. (`need_model` is now near-dead but kept for a run whose every stage scores exactly 0.0.)

### 27 — Champion written to the archive AFTER the archive's only flush (`core/optimizer.py`)
`self.knowledge_archive.save()` ran ~20 lines ABOVE `champion_store.save(...)`, and `DatasetChampionStore.save` only mutates the archive's in-memory `_data`. So a champion reached disk purely by accident — when some *later* component flushed the same singleton (`formula_library.prune_*`, `working_memory_pruner`, or the next `fit()` in a multi-dataset process). **A process that optimized one dataset and exited persisted no champion at all**, which is why champion-resume appeared to work in benchmarks and did nothing in single runs. Compounded by `formula_library.py:84` calling `KnowledgeArchive.shared()` with the DEFAULT path, which swaps the module singleton and makes a `state_dir` archive reload from disk. Fixed: the flush moved below the champion + conclusion writes. This bug was what made bug #26 invisible — the gate could not be observed because the champion never survived.

### 28 — JSON round-trip stringifies OvR class keys → mixed label types (`adapters/symbolic_formula_adapter.py`)
JSON object keys are always strings, so a `multiclass_ovr` champion comes back keyed `"0","1","2"`. `predict()` did `classes = sorted(model["classifiers"].keys())` then `np.array(classes)[argmax]`, returning `'2'` (dtype `<U1`) instead of `2` — and every downstream metric raised `ValueError: Mix of label input types (string and number)` at `f1_score(y_val, preds)`. **Latent for as long as bug #26 existed**: the champion gate meant the run never reached the `improved` branch where F1 is computed, so the crash only surfaced once #26 was fixed. Plain `sorted()` on string keys also misorders ≥10 classes (`'10' < '2'`), which would silently mislabel every prediction on digits/MNIST. Fixed in two places: `_repair_model_trees` restores integer keys at the designated post-load hook, and `predict()` sorts numerically and emits `int` labels regardless. **Invariant: anything persisted through JSON and read back must have its key/label dtypes restored explicitly — dict keys never survive.**

### 29 — Warm start SUBSTITUTED instead of resuming (`adapters/symbolic_formula_adapter.py`, `core/optimizer.py`)
`_apply_warm_start` returned a deepcopy of the prior model with new feature weights swapped in and **returned before any training**, for every mode except `sparse_linear` (which correctly opts out — and is the mode that benchmarks best; that is not a coincidence). Three consequences, all silent:
1. **Stale threshold.** Feature weights DO change scores (tree paths go through `_prepare_X`), but the copied decision threshold was calibrated against the old scores. Measured: refitting under new weights moved it **0.50 → 0.05**.
2. **Evolved formulas could not enter.** Formulas are chosen when a model is BUILT. Nothing rebuilt after the warm start, so everything the formula library discovered mid-run had nowhere to go — same dead-channel shape as #25.
3. **Mode changes silently dropped.** `mode = self._resolve_mode(...)` sat AFTER the warm-start return, so a warm-started fit kept the mode it was copied with. The search proposes boosting/bagging/multi under stress and got no answer and no error.

Fixed, in two parts with different ship status.

**Shipped ON (correctness, unconditional):** mode is resolved BEFORE the warm-start branch and a mode mismatch forces a real refit; `_refit_thresholds()` recalibrates (binary and per-OvR-head) against the scores the model will actually be judging. Measured: refitting under new weights moved the threshold **0.50 → 0.05**. There is no argument for keeping a stale ruler.

**Implemented, default OFF (`cfg.warm_start_first_fit_only`):** limiting warm start to one fit per run (deterministic — index 0 of the first evaluated stage, so replay is unaffected). The structural defect is real: distinct model STRUCTURES over 17 candidate fits on breast_cancer (feature weights stripped) went boosting **11 → 17**, multi **7 → 17**, i.e. **35–60% of candidate fits were duplicates the search was re-scoring**. But the 3-seed A/B did **not** support shipping it: breast_cancer **−0.0015**, wine **−0.0093**, digits **+0.0000**, mean **−0.0036** (two of three within noise). Per the E6 lesson, a structural argument does not override a failed cross-seed gate.

**Hypothesis for WHY recovering diversity did not help (testable, not established):** duplicate candidates were accidentally suppressing selection overfitting. Six identical models are one effective try against the fixed validation split; six distinct ones are six. So this fix increases the winner's-curse pressure documented in `architecture-race-selection-overfit` — which predicts it should stop regressing, and may turn positive, once the validation split rotates per stage and selection moves to `_extra_val_splits`. **Re-run this A/B after the split fix before concluding anything about warm start.** Do not flip the default on structural reasoning alone.

Caveat on the numbers above: that harness runs `learning_mode="auto"` at 5 stages, which routes digits to a weak multi-head mode (0.312, nowhere near the 0.947 that `sparse_linear` reaches at full budget — see bug #24). Both arms share the config, so the delta is valid; the absolute values are not production numbers.

(Note: "no learning at all after stage 1" would be an overstatement — `_apply_warm_start` already bailed on several conditions. The collapse was partial but large.) **Invariant: a warm start may seed a fit, never replace it; and any reused model must have its calibrated quantities refit against the data it will now be scoring.**

### 30 — The instrument was broken in two more ways (2026-07-23) — **read before trusting ANY past A/B**
Found while building `experiments/existence_audit.py`, whose control step ("do two identical runs agree?") failed immediately.

**30a — Determinism is MODE-SCOPED, not global. — SUPERSEDED 2026-08-05, see the determinism matrix below; this scoping no longer holds.** The documented invariant ("same seed → byte-identical predictions; verified") holds for **binary targets under `sparse_linear`** — the configuration `eval_crossseed` uses — and **fails elsewhere**. Measured, same seed, fresh `state_dir`, *separate processes*: binary + `sparse_linear` byte-identical; `auto` (→ hybrid/multi) diverges at 5 stages; **multiclass OvR** diverges too (wine, 6 stages: predictions AND `n_selected` 8 vs 9 differ between two identical runs). Consequence: **any A/B run at `auto` or on multiclass data was comparing partly-random arms** — that includes this session's warm-start A/B (−0.0036 at `auto`) and should be assumed of others unless their config is checked. Identity tests must pin binary + `sparse_linear` until the OvR/hybrid paths are made deterministic.

**30b — `state_dir` isolation is incomplete, and it leaks through DISK.** A run with `state_dir` set still writes six shared files under the default `results/`: `uil_fingerprints_v22.json`, `uil_routing_memory.json`, `uil_governance_manifest.json`, `uil_formula_changelog_v20.json`, `uil_graph_changelog_v20.json`, `uil_live_feed.jsonl`. Because the leak is on the filesystem, **process isolation does not fix it**. **FIXED (2026-07-24):** fingerprint / routing memory / governance / graph+formula changelogs / live_feed all co-locate under `state_dir` when set (`save_fingerprint(filepath=…)`, `RoutingMemory` on `_state_path`, `apply_bootstrap(state_dir=…)`, `scoped_*_changelog`, live_feed `filepath=`). `FormulaLibrary` already scoped its `KnowledgeArchive`. Tests: `tests/test_state_isolation_and_situation.py` I1–I3. Default (non-isolated) runs still write the shared `results/` paths so the web UI keeps working.

### 31 — Multiclass/hybrid nondeterminism: an unseeded `liblinear` (2026-07-23) — **FIXED, supersedes 30a**
Bug #30a recorded the SYMPTOM (identical multiclass runs disagree; wine `n_selected` 9 vs 8) and scoped determinism to "binary + sparse_linear only". The cause is now found and removed, and it is one line.

**`_fit_sparse_linear` built `LogisticRegression(penalty="l1", solver="liblinear", ...)` with no `random_state`.** `liblinear` is the one sklearn solver that consumes `random_state` for an otherwise deterministic-looking L1 fit — it randomly permutes its coordinate-descent order — and `random_state=None` means *draw that seed from the GLOBAL numpy RNG*, which is seeded from OS entropy at interpreter start. So the seed differed in every process. **Proven directly**: 8 fits of byte-identical data under 8 different global seeds produced **8 different coefficient vectors**. Multiclass amplified it because OvR gives the unstable L1 support one chance to flip *per head*.

Fixed: `random_state=self._sk_random_state()` (derived from the adapter's own seed), plus that helper as the single place any sklearn estimator in the adapter gets its seed. **Verified**: `existence_audit --dataset wine --stages 6` went from `CONTROL FAILED — n_selected: 9 vs 8` to `control: two identical runs agree exactly`. Regression test: `tests/test_vocabulary_and_determinism.py` D1–D3, which perturbs the global RNG between fits — the only way to catch this class.

**Second axis, same fix family:** `_fit_hybrid_ensemble` made branch results depend on the machine's thread count. Serially every branch drew from one shared advancing `self._rng`; in parallel each branch got a clone whose `_rng` restarted at `seed` — so a 2-core and an 8-core box produced different models, *and* all parallel branches drew the identical bootstrap stream (branches exist precisely to differ). Each branch now derives its stream from `(seed, branch index)`, making serial and parallel byte-identical and restoring between-branch diversity.

**Invariant: every sklearn estimator constructed anywhere in this codebase must be passed an explicit `random_state`.** `None` is not "no randomness", it is "randomness from a process-global source". Audited: all other call sites (`RandomForestClassifier`, `MLPClassifier`, `train_test_split`, `DecisionTreeRegressor`) were already seeded; `_fit_platt`'s lbfgs is deterministic.

### 32 — The readable law was not the model (`adapters/symbolic_formula_adapter.py`, `cognition/law_export.py`)
**The project's one surviving value claim had zero tests, and the renderer was wrong.** `readable_sparse_linear` printed **standardized-space coefficients against raw-space term names** (the model computes `sigmoid(Σ coef_j·(b_j−mu_j)/sd_j + intercept)`; the string showed `coef_j` next to `b_j`), **omitted the fitted threshold** (bug #20 makes it routinely ≠ 0.5 — measured 0.6 and 0.35 on test fits), **omitted `selected_features`** (so `x26` named a reduced-basis column, not the reader's column 26), omitted `positive_class`, and **silently truncated at `top=12`**. Read literally the printed law mislabelled **19/500 rows (3.8%)** on a clean 6-term model, and worse whenever truncation fired. Not a cosmetic issue: a human who acts on the string acts on a different function.

Fixed by `cognition/law_export.py`. The fold is exact, not approximate — `c_j = coef_j/sd_j`, `intercept' = intercept − Σ coef_j·mu_j/sd_j` — so the printed coefficient is the one that multiplies the printed term. The law also carries the threshold, the original feature indices, `positive_class`, and per-head Platt calibration for OvR; `evaluate_law()` is an independent pure-numpy evaluator that consumes ONLY the law. `readable_sparse_linear` now delegates to it (`top=None` for the complete law; truncation is labelled `[TRUNCATED: n of m terms hidden]`). The old renderer survives as `_readable_sparse_linear_legacy` solely so the test can demonstrate the defect.

**Honest scoping the export forced:** `raw`/`stump`/`rule`/`kan` terms are self-contained AND legible; `neural` terms are exactly specifiable but **not legible**; `formula` (evolved tree) terms are **not self-contained** — `evaluate_law` raises rather than guess, and needs a `tree_eval` callable. `law_is_self_contained()` / `law_is_legible()` report which, so "readable" is a measured property rather than a slogan.

Tests: `tests/test_law_fidelity.py` F1–F8 (8/8). F7 is the one that matters beyond inspection — deleting a term moves the logit by exactly the predicted amount (max error 7e-14), so the law is **operable**, not just printable. **Invariant: any new rendering or export path must round-trip — `evaluate(render(model)) == predict(model)` — and must never omit a quantity `predict()` uses. Truncation must announce itself.**

### 33 — Multiclass OvR collapse persists off the `sparse_linear` path (`adapters/symbolic_formula_adapter.py`) — OPEN
Bug #24 fixed the OvR collapse in two parts: Platt-calibrate each head, and give the OvR loop a `sparse_linear` branch. **Only the second part actually worked.** Measured 2026-08-05 on 3-class data (majority rate 0.515) with calibration confirmed present on every head: `single` **0.5150 / 1 class**, `bagging` **0.5150 / 1 class**, `multi` **0.5150 / 1 class**, `boosting` 0.6200 / 2 classes, vs `sparse_linear` **0.8450 / all 3**. Accuracy exactly equal to the base rate is the same signature as bugs #13 and #20 — the signal is being thrown away. The measured digits jump in bug #24's notes (0.214→0.511 from calibration, 0.511→0.947 from sparse_linear heads) is consistent: calibration alone never got those paths working, and nobody re-checked because production pins `sparse_linear` for multiclass. Pinned by `tests/test_determinism_matrix.py` M5, which asserts the *current broken* counts so a fix fails the test loudly. **The "use `learning_mode='sparse_linear'` for multiclass" directive is load-bearing, not a preference.**

### 34 — `situation_adaptive` silently overrides `explore_architecture`, and it disabled its own test (`core/optimizer.py`) — OPEN
`explore_architecture` is documented default **OFF** ("because the extra journal entries feed FUTURE mining, a cross-run side effect"). But `situation_adaptive` ships default **ON** (Category D Gate 2, 2026-07-24) and **auto-enables `explore_architecture` from the fingerprint** — on breast_cancer it does. So the documented default is not the effective default. Same bug CLASS as #21: an architectural knob set by one layer being silently replaced by another.

**It also switched off the test that guards step 2.** `test_vocabulary_and_determinism.py`'s A-block builds an OFF arm and an ON arm; with `situation_adaptive` left at its default both arms came out ON, so A1/A3 failed and **A2/A4 "passed" trivially — A4 in particular is guaranteed when the two runs are literally the same run.** Measured: default → OFF and ON both give 8 journal entries / `sparse_l1_c` 2 distinct; pinned False → OFF gives 4/1 and ON gives 8/2, the intended contrast. Fixed by pinning `Cfg.situation_adaptive = False` in `_run`, with A5 added to pin the override itself so it can never silently disable the test again.

### 35 — The architecture probe is only side-effect-free WITH a classifier factory (`core/optimizer.py`) — OPEN
Step 2's headline guarantee is that the probe is "winner's-curse-safe by construction — the probe NEVER competes for the stage winner", verified as byte-identical accuracy. **That holds only on the isolated-worker path.** With no `classifier_factory` the probe falls back to save/restore of config + `_shadowed_attrs`, which does not restore everything its fit mutates (formula-library writes, `_formulas_used_this_fit` / `_last_used_formula_id`, the adapter RNG). Measured, breast_cancer, 10 stages, `situation_adaptive` pinned off:

| path | explore OFF | explore ON | identical |
|---|---|---|---|
| with `classifier_factory` (isolated worker) | 0.9649122807017544 | 0.9649122807017544 | **yes** |
| no factory (save/restore fallback) | 0.9649122807017544 | **0.9736842105263158** | **no** |

The original 2026-07-24 "byte-identical" measurement was taken on the factory path, which is why this went unnoticed. This is exactly the hazard already documented for `_evaluate_one_action` in the failed verify-before-apply gate — *a verification probe that mutates the run it measures is worthless*. Pinned by A4 (guarantee holds with a worker) + A6 (defect asserted without one, so a fix fails loudly). **Until fixed, only enable `explore_architecture` when a `classifier_factory` is supplied.**

### Vocabulary widening — the cognitive layer can now say more than `max_penalty` (2026-07-23)
Bug #25's ROOT CAUSE, addressed. Everything except one function was already plumbed end to end:
- `uil_miner._build_matrix` already flattens `action.classifier_config` into `config.<name>` columns — the tree can split on `config.sparse_l1_c`;
- `ExecutionContext.to_flat_dict` (V20+.12) already carries `config.*` to the executor and the graph, so such a condition can MATCH;
- `optimizer._apply_overrides` already forwards `config.<name>` into `classifier.apply_config()`;
- `SymbolicFormulaAdapter.apply_config` already accepts every relevant knob.

**`uil_library._generate_override` recognised exactly one controllable thing — `penalty` — and silently dropped everything else.** 201 principles, the graph and the causal engine all collapsed into one scalar deciding where to cut the feature ranking. Now it emits constraints on any `config.*` the miner splits on.

**Bounds, not assignments.** A mined threshold is evidence about where regret rises, not a decree about where to sit; and a hard assignment would set every architecture-race arm identically, collapsing the race (E7's invariant). So it emits `config_max.<name>` / `config_min.<name>`, resolved against the live value in `optimizer._resolve_config_bounds`.

**Conditions are a CONJUNCTION — one root-to-leaf tree path — so they describe a REGION, not independent facts.** Inverting each side separately produced `config_min.boosting_stages: 20.51` *alongside* `config_max.boosting_stages: 16.5` — a floor above a ceiling, infeasible. Fixed by collecting the bad interval per feature and inverting once: one-sided → a bound; two-sided → `config_avoid.<name> = [lo, hi]`, which the actuator escapes by the **nearer** edge (smallest change satisfying the rule). Same fix applied to `penalty_avoid` — the old code emitted both penalty bounds and let statement order pin the result to the upper edge by accident.

`learning_mode` and `parallel_eval_workers` are excluded (`NON_CONSTRAINABLE_CONFIG`): the router owns mode (bug #21), and the other is a machine-resource knob that would make mined knowledge machine-specific.

**Derived features were posing as the knobs they are built from (pre-existing, found while measuring the above).** `_generate_override` tested `"penalty" in feature`, which also matches the miner's synthesized products — `derived.action.penalty*config.mutation_rounds_per_fit` — so a threshold on a PRODUCT became a threshold on penalty. **Measured on the live graph: 18 of 300 nodes carry `min_penalty: 1270.45` / `max_penalty: 0.0003`** for a knob proposed in 0.5–3.0 and clamped to [0,10]. Same bug family as #14 (derived variables leaking into the causal scheduler).

This was *harmless while bound aggregation summed* (a cap of 1223 caps nothing) and becomes **actively destructive once bounds intersect** — one bogus 0.0003 would pin penalty to zero for the whole stage. So the aggregation fix below could not ship without this one. Two-part fix: exact matching at generation (`PENALTY_FEATURES`), and `condition_match.sanitize_overrides` applied when principles and graph nodes LOAD — an override survives only if one of that rule's own conditions names the knob it constrains. **Provenance, not magnitude**: `0.0003` is a legal penalty cap and is only detectably wrong because the rule never mentions penalty. Measured on the real graph: **34 of 300 nodes repaired, penalty overrides 142→107, all 18 out-of-domain values gone, survivors now in [0, 1.78]**. A separate range guard in `_apply_overrides` drops bounds outside [0,10] (the half provenance cannot see: a rule may legitimately mention penalty and still carry a nonsense threshold).

**Graph aggregation fixed alongside.** Both traversals blended numeric overrides — one by weighted average, one by `overrides[key] += value * attention`. That is defensible for a VALUE and wrong for a BOUND: three nodes each saying "cap penalty at 2.0" accumulated to a cap of 6.0, i.e. **no cap at all** — the existing `max_penalty` channel was partly neutering itself. Constraints now intersect (lowest ceiling / highest floor) and are not scaled by attention; attention decides which nodes are REACHED, not how strong their constraint is. **This changes existing `max_penalty` behaviour and has not been through a cross-seed accuracy gate.**

**Verified end to end** with the real miner and library on planted data (regret rises with `boosting_stages`): held-out R² 0.978, 6 principles minted, **6/6 carrying architecture constraints**, 0 infeasible. Tests: `tests/test_vocabulary_and_determinism.py` V1–V7.

**BUT the emitter was not the only thing in the way. Two further blockers, both measured, both still open — this is why the audit still reads INERT.**

**(a) Promotion needs ~200 same-dataset stages.** `evidence_count` is a decision-tree LEAF SIZE, so it is bounded by the journal, and `MIN_EVIDENCE = 10` is absolute. Measured against a planted signal: largest leaf holds **4** rows at 16 journal entries, **6** at 45, **9** at 100, and only at ~**200** does anything clear 10. Every run with an isolated `state_dir` starts from an empty journal, so **no principle is ever mined in an experiment or an audit arm** — the library holds bootstrap seeds only, and those emit `max_penalty` alone. Confirmed on a real 16-stage breast_cancer run: 16 journal entries, miner held-out R² 0.99, **0 mined principles**. Opt-in scaling added (`cfg.adaptive_min_evidence`, default OFF, mirrors M0.2's tiny-mining shrinkage); it is an EPISTEMOLOGY change, not a wiring fix, so it needs a cross-seed gate.

**(b) The system never VARIES its architecture, so it has no architecture experience to mine.** Journal from that same run, per `classifier_config` column: `penalty` **16 distinct values**; `sparse_l1_c`, `use_rule_basis`, `use_stump_basis`, `use_neural_basis`, `mutation_rounds_per_fit` — **1 distinct value each**. A decision tree cannot split on a constant, so no rule can ever mention an architecture knob no matter how wide the emitter's vocabulary is. In `search_mode="classic"` the per-stage search only perturbs `penalty` (plus `boosting_stages`/`bagging_estimators`, which are inert under `sparse_linear`).

`search_mode="race"` **does** supply the variation — same run under race: `sparse_l1_c` 4 distinct, `use_rule_basis` 2, `use_stump_basis` 2. It still mined 0 architecture rules at 16 stages, because `penalty` alone already explains regret at R² 0.99 and the tree never needs another split. So the honest ordering is: **variation (race) → enough stages for promotion → then the vocabulary has something to say.** The emitter is no longer the binding constraint; it was simply the first of three.

**FIX for (b) in the CLASSIC path — architecture-exploration probe (2026-07-24, research step 2, opt-in).** The race supplies variation but is opt-in and comes with its whole successive-halving apparatus. For the default (classic) path, `cfg.explore_architecture` (default False) adds ONE architecture probe per stage: `optimizer._architecture_exploration_probe` takes the winning action, perturbs ONE axis from `get_architecture_axes()` round-robin (excludes `learning_mode` — bug #21), fits it, and JOURNALS the (config, outcome) pair. **Winner's-curse-safe by construction — the probe NEVER competes for the stage winner**, so it cannot ride the single-split judge and re-create the E7 regression (the research doc's explicit warning that variation must ship *with* the judge fix, not before it). Measured A/B (breast_cancer, isolated): accuracy **byte-identical** (0.9649122807017544 both ways — the non-competing guarantee, proven not asserted), journal 12→24 entries, and architecture variation created: `sparse_l1_c` 1→**4 distinct**, the three basis gates 1→**2 distinct** each — a decision tree can now split on them. Skipped in race mode (which has its own variation). Fit runs on an isolated worker (or save/restore of config + `_shadowed_attrs` when no factory), so it never corrupts the run it enriches. Default OFF because the extra journal entries feed FUTURE mining (a cross-run side effect), not because it risks accuracy. Tests: A1–A4 in `tests/test_vocabulary_and_determinism.py`. **This creates the evidence; the miner still needs step 3 (enough same-dataset stages / partial-pooled promotion) to turn it into architecture rules — variation was the second of three binding constraints, now cleared.**

**THE CHAIN NOW COMPLETES — first mined architecture rule (2026-07-24, research step 3).** Step 3's mechanism was already present (`cfg.adaptive_min_evidence`, `_promotion_min_evidence` = `max(3, min(10, n_entries//8))`, default OFF) — it lowers the promotion bar in the small-journal regime real runs live in (leaf sizes 4/6/9 at 16/45/100 entries vs the fixed bar of 10). With **step 2 + step 3 both on**, a 40-stage breast_cancer run mined its **first architecture-conditioned principle ever**: `P-011`, conditions `state.entropy>0.483 AND action.penalty<=1.546 AND config.bagging_estimators<=6.5`, override `{min_penalty: 1.556, config_min.bagging_estimators: 6.51}`. So variation → lowered promotion bar → the vocabulary emitter now runs end to end; the "inert layer" pipeline is connected for the first time. Accuracy unchanged (0.9737). Tests: P1–P3 in `tests/test_vocabulary_and_determinism.py` (fast, deterministic — plants an architecture→regret signal and checks the miner splits on `config.*`, the lowered bar admits an evidence=5 rule the default rejects).

**GATE 2 RESULT FOR STEPS 2+3 — FAILED (2026-07-24). `adaptive_min_evidence` stays OFF.** 40 stages (20 mines nothing, so a default-config A/B would be void), 4 datasets × 3 seeds, external 20% held-out test, cv_folds=3, ON = `explore_architecture` + `adaptive_min_evidence`:

| dataset | OFF | ON | Δ | ON mined | ON arch-override |
|---|---|---|---|---|---|
| breast_cancer | 0.9708 | 0.9708 | +0.0000 | 11 | 11 |
| ionosphere | 0.9155 | 0.9155 | +0.0000 | 25 | 24 |
| sonar | 0.7302 | 0.6984 | **−0.0317** | 26 | 20 |
| wine | 0.9537 | 0.9537 | +0.0000 | 7 | 4 |

Mean **−0.0079** (0 win / 3 tie / 1 lose). **69 principles mined, 59 carrying architecture overrides** — so this is NOT the "nothing mined, delta measures nothing" case; the chain genuinely worked and the rules were promoted and applied. They simply did not help.

**This is a STRONGER form of the project's central negative result than anything measured before.** Previously the cognitive layer was INERT, so "no effect" was explainable as a wiring bug — and this session fixed that wiring end to end. Now the layer varies its architecture, mines rules about it, promotes them, and applies them through a live actuator, and still returns **0 improvements out of 4 datasets**. Two mechanistic notes: (a) the three exact +0.0000 ties are all `sparse_linear`, meaning the architecture overrides existed but changed nothing about the output — the inert signature persisting even with rules present; (b) sonar (the `boosting`-mode dataset) is the one where rules moved the output, via the genuinely-live `boosting_stages` knob, and it moved DOWN. Attribution is clean: test A4 proved the probe alone is accuracy-neutral (byte-identical), so the regression comes from the RULES being applied, not the probe.

**HONEST CAVEAT (this is why step 3 stays opt-in):** the first rule minted is about `bagging_estimators`, which is **inert under `sparse_linear`** — a "true but useless" no-op override about a knob that doesn't affect the model. That is exactly the risk a lowered evidence bar carries: it admits weaker, possibly-spurious rules. The two flags being independent is the point — `explore_architecture` (step 2, evidence-gathering, no accuracy risk), `adaptive_min_evidence` (step 3, admits weaker rules, needs a cross-seed gate), and `search_mode="race"` (accuracy thread, judge fix) are **three orthogonal opt-ins**: "do both/all" is just enabling the flags you want, they compose. **INERT-KNOB FILTER — SHIPPED (2026-07-24), the refinement that makes step 3 safe.** `SymbolicFormulaAdapter.get_active_config_knobs()` reports which config keys can actually change the fit under the CURRENT learning mode; `optimizer._resolve_config_bounds` drops any `config_*` override naming a knob outside that set. So P-011's `config_min.bagging_estimators` under `sparse_linear` is now dropped instead of applied-and-journalled — verified: it resolves to `{}` while live knobs (`sparse_l1_c`, `use_rule_basis`) still apply normally. **Accuracy-neutral by construction**: `_fit_sparse_linear` provably never reads the dropped knobs (grep-verified: 0 references to `bagging_estimators`/`boosting_stages`/`hybrid_branch_count`), so the model cannot change — what the filter prevents is journalling a knob that did nothing, which would otherwise feed further spurious rules. **Mode-scoped, not a blocklist**: `boosting_stages` stays live under `boosting`, and `auto`/`multi`/`single` are deliberately PERMISSIVE (wrongly calling a knob inert would silently drop a real override — strictly worse than allowing a harmless no-op). Adapters without the method are unfiltered, so nothing else changes. A dropped override is counted in guard telemetry as `inert_config_override_dropped`, so "how often does the miner learn about knobs that cannot matter" is now measurable. Tests: P4–P5. (This change also correctly broke V6/V7, which had been using `boosting_stages` to test bound-resolution *math* under a sparse_linear adapter — they now use `tournament_k`, an int knob live in that mode.)

### Evolution was asleep, not disconnected — the stress gate was unreachable (2026-07-23)
`_want_new_structure` fired at `system_stress >= 0.6`. **Measured on the 1,299-entry journal covering 37 real runs: only TWO runs ever reached 0.6.** Run-max percentiles are p25 0.085, **p50 0.169**, p75 0.254, p90 0.431. The distribution is bimodal — a run either saturates at 1.0 or stays under ~0.43 — so 0.6 is not a conservative threshold, it is an **unreachable** one for ~95% of runs. That is why the existence audit read `formulas` INERT: the engine is correct and simply never ran.

Gate is now relative to the run's own stress — fire at a new stress peak, with `absolute` (default 0.6) as an upper bound so a saturating run behaves exactly as before, and a 0.05 floor so a flat calm run does not evolve on noise. `EVOLUTION_MIN_GAP` and E3's `prune_to_cap` still bound the birth rate, so this cannot reintroduce the runaway. `cfg.evolution_absolute_gate_only = True` restores the old behaviour exactly.

**MEASURED EFFECT (3 datasets × 3 seeds, 12 stages, OLD vs NEW gate): the gate now FIRES — classifier formula library 32 → 34 every single run — and accuracy is IDENTICAL in 9/9 (mean delta +0.0000, improved 0 / unchanged 9 / regressed 0).**

So the change is safe and it does what it says, but it also sharpens the negative result rather than overturning it: **evolution now runs and creates structures, and those structures still do not reach the model.** That is consistent with the audit's `formulas INERT` verdict and narrows the remaining question to selection — on the `sparse_linear` path a newly born formula only matters if it survives into `_build_basis_spec`'s symbolic family and then beats the other terms under L1. Waking the engine was necessary and is not sufficient.

**Research step 4 — utility-gated admission + newborn L1 sub-budget (2026-07-24), IMPLEMENTED, Gate 2 FAILED, default OFF.** Root cause of the selection deadlock: `_build_basis_spec` used only `get_active_formulas()` (active+seed), while mutations enter as `incubating` — so a newborn could never be L1-selected, never earn `record_result` credit under sparse_linear, and never promote. Fix (opt-in `cfg.utility_gated_formulas`): (1) DreamCoder-style held-out test (`utility_test_candidate`) — mint as `active` only if adding the term improves a held-out sparse_linear fit, else reject; (2) when the flag is on, incubating formulas also enter the basis tagged `newborn` with `NEWBORN_L1_BOOST=2.0` (E5 remedy for formula terms). Tests U1–U5. **Existence audit: formulas still INERT** (breast_cancer, 12 stages). **Gate 2 (6 datasets × 3 seeds): mean Δ −0.0013** (5 exact ties, sonar −0.0079). Stay default OFF. Logs: `results/gate2_utility/`.

**Research step 5 — EBM cyclic fit (2026-07-24), IMPLEMENTED, default OFF.** `use_ebm_fit` runs round-robin GLM steps over the same composed basis instead of one L1 solve; model shape stays `sparse_linear` (predict/readable shared). Exposed as a race axis. Unit-tested. Gate 2 pending (sonar first).

### Determinism matrix — the "mode-scoped" caveat is obsolete (2026-08-05)
`tests/test_determinism_matrix.py` extends bug #31's fix past what D1–D3 covered, and everything passes:
- **M1** — all 6 learning modes (`sparse_linear`/`single`/`boosting`/`bagging`/`multi`/`auto`) × binary and multiclass, each fit 3× under a **deliberately perturbed global numpy RNG**. That perturbation is the point: it is the only way to catch an estimator constructed with `random_state=None`, which is exactly bug #31's mechanism.
- **M2** — identical **across separate processes**, where the global RNG is seeded from OS entropy. An in-process test cannot see this failure class at all.
- **M3** — hybrid branches **serial vs 2 vs 4 workers** are byte-identical. This was bug #31's documented second axis and had **no test whatsoever** until now.
- **M4/M5** — non-degeneracy guards, because a model that always answers one class is also perfectly "stable". M5 is what surfaced bug #33.

**Consequence: bug #30a's scoping ("binary + `sparse_linear` only") is superseded.** A/Bs run at `auto` or on multiclass are no longer comparing partly-random arms, and the existence audit's `*_binary` dataset workaround is no longer required for determinism reasons.

### Ablation switches now cover nine subsystems, not four (2026-07-23)
`DEFAULT_ABLATION` gained `episodic`, `structural`, `meta_principles`, `safeguards`, `goal`. Each gates every call site of one subsystem — including `episodic`'s WRITE path, so an "off" arm cannot leave an episode behind for a later arm to retrieve. `structural` is a genuine prerequisite for `meta_principles` and `safeguards` (both consume `StructuralPrinciple` objects); the audit reports that dependency rather than pretending three independent switches exist. `experiments/existence_audit.py` now sweeps all nine.

### Triplicated condition matching unified (`cognition/condition_match.py`, 2026-07-23)
`uil_library._conditions_similar`, `editable_network._conditions_similar` and `journal_forgetting._conditions_overlap` were three copies of one comparison, and **all three carried bug #17 at the same time** — `abs()` arithmetic on the string thresholds that v22 routing rules mint. The copies were the hazard, not the arithmetic. All three now delegate. The two comparison MODES are preserved exactly and named rather than merged: `"reference"` (library/network — tolerance relative to the existing principle, which is the reference point) and `"max"` (forgetting — symmetric, since neither side is a reference). Merging them would silently change which principles are considered duplicates.

### Existence audit — 3 of 4 cognitive subsystems are INERT (2026-07-23)
`experiments/existence_audit.py`. Deliberately **not** an accuracy ablation (that answer is settled at zero). It asks: *remove this subsystem — does ANY bit of the output change?* Because the system is deterministic in the pinned config, this is an **identity test**, not a noisy comparison: no confidence interval, no winner's curse, and — importantly — **it does not need the validation-split fix first**. Each arm runs in its own process; a control step runs the baseline twice and aborts the audit if they disagree.

| subsystem | breast_cancer (5 stages) | wine_binary (6 stages) |
|---|---|---|
| principles | **INERT** | **INERT** |
| network | **INERT** | **INERT** |
| formulas | **INERT** | **INERT** |
| causal | CONNECTED (predictions, macro-F1) | **INERT** |

**STRENGTHENED AND GRADED (2026-08-05) — `experiments/existence_sweep.py`, `results/existence_sweep/`.** The two-cell base above is now four cells across two budgets, and the binary verdict is replaced by a graded one (share of probe predictions that moved, so "nudged one row" is distinguishable from "rewrote the model"): breast_cancer @ 5 and @ 20, wine_binary @ 5 and @ 20 — **all four 0/9 connected, max predictions moved exactly 0.0%, every control passing**. 36 subsystem-removals, not one moved a single predicted label. **Quadrupling the budget changed nothing**, which retires the "the subsystems can't act that early" defence (at 20 stages the prune/decay cadence, the ≥5-principle meta threshold, and the stress-peak evolution gate are all reachable). Caveat stated in `FINDINGS.md`: at default flags few or no principles are mined, so the `principles` row partly reflects an empty library — the complementary case (69 principles mined, 59 with architecture overrides, applied, 0 improvements) is the steps-2+3 Gate 2 below. Both halves are now measured.

**CORRECTED AND SHARPENED 2026-08-06 — `results/existence_mined/FINDINGS.md`.** The 0.0% headline is true **at default flags** and must always carry that qualifier. Re-run at **40 stages with the connect pack ON** (so principles are actually mined, promoted and applied), breast_cancer, control passing: **principles CONNECTED (2.0% of predictions moved, Δacc −0.0044), network CONNECTED (0.0% moved but accuracy/weights change, +0.0044), formulas CONNECTED (1.0% moved, −0.0088)**; the other six stay INERT. So the layer is **not structurally dead** — it is **connected and ineffective**: movement is 0–2%, accuracy deltas are ±0.009 and net negative, independently reproducing the steps-2+3 Gate 2 (mean −0.0079, 0 improvements). The correct statement of the central negative result is therefore: **the cognitive layer is inert by default (nothing is mined) and, when mining is enabled, it reaches the model but does not help.** That closes the last escape hatch — "it would work if the rules were really mined". They were, and it doesn't. **Second cell (wine_binary) is the cleanest illustration in the whole audit: removing `principles` moves 1.1% of predictions and removing `formulas` moves 0.6%, while accuracy is EXACTLY unchanged in both — the layer changes *which* rows are right, not *how many*. Pure churn.** Across both cells `principles`/`formulas` are CONNECTED in both, `network` in one, the other six INERT in both. (`network` moving 0.0% of probe predictions while changing accuracy is not a contradiction: the probe is `X[:200]`, accuracy is the internal val split — the graded metric is a lower bound on connectedness.)

So bug #25 was not a one-off: the principle library, the knowledge graph, and the formula engine all run, spend compute, and write state **without changing a single output bit** *at default flags*. Causal is the only one that ever moves the needle, and only on one of two datasets (it forces control/treatment actions, which changes what gets evaluated). **Not covered — no ablation switch exists:** episodic memory, meta-principles, structural encoding, abstraction safeguards, the goal cell. They need one before they can be audited.

### The search layer's three quiet defects (2026-07-23)
Found by tracing cognition → learner end to end. None crash; all three degrade or mislead.
- **Dead `X_eff` in `_basis_matrix`** — `X_eff = self._prepare_X(X, feature_weights)` was assigned and never read (every branch used unweighted `X`). Not merely wasted work: it *looked* like weights were applied, which is why the dead cognitive channel (#25) went unnoticed. Deleted, replaced with a comment recording that the obvious re-wiring is a **proven no-op** (z-scoring cancels column scaling — arm B, delta 0.0000) and that the live route is `_basis_weight_vector` folded into `sd`. `_prepare_X` is still correct and still used on the tree/gradient paths.
- **"Opposite of worst" was the opposite of something good** — the optimizer only reported the stage WINNER back to the search, so `_worst_weights` held "the winning vector of the lowest-scoring stage" (and on stage 1, literally `_best_weights`). One of 4–6 candidates per stage was the mirror of a decent solution, and the journal recorded that false label for the miner to mine. Fixed: the optimizer now passes the genuinely lowest-scoring candidate; the move became "step from best along the direction away from the loser" (a mirrored vector in a high-dim space is just another arbitrary point), renamed **"Away from worst candidate"**.
- **Feature-guidance ratchet (opt-in fix, `cfg.fix_guidance_ratchet`, default OFF)** — `safe_minmax` forces the minimum to exactly 0.0 on every update, and the next line zeroed anything `< 0.05`; with `effective_lr` up to 0.7/stage a feature under-credited 2–3 stages in a row is deleted permanently, and guidance is the ONLY input to candidate generation, so it can never be proposed again. Fix = `scale_by_max()` (new, `core/utils.py`; scales by max, never manufactures a zero) + a 0.02 floor instead of zeroing + halved learning rate. Proof (4 stages, one weak feature): OLD `[1.0, .887, .773, .660, 0.000]` vs NEW `[1.0, .889, .778, .667, 0.061]`. **Duplicated in BOTH `pso_search.py` and `principle_guided_search.py` — fix both or you fix the one you are not running.** Default OFF pending a cross-seed sweep, per the E6 lesson. Its falsifier: if this ratchet is what made `connect_cognition` regress on digits (−0.011) and ionosphere (−0.014), enabling it should reverse those two and nothing else.

**Test-suite status (2026-08-06).** The full sweep ran **34 files: 30 pass, 4 fail**. Of those 4, three are now resolved and none was a product defect. Four new suites were added this session (`test_law_fidelity`, `test_determinism_matrix`, `test_operator_guards`, `test_gplearn_adapter`), all passing.
- `test_comprehensive` **passes** — but only from the `figpso_uil/` cwd. It opens `results/uil_journal_v19.json` by **relative path**, so the documented invocation (`python figpso_uil/tests/test_comprehensive.py` from `Downloads/`) raises `FileNotFoundError`. Run it from `figpso_uil/` or make the path absolute; measured `ASSERTIONS PASSED (6 datasets validated)`, exit 0.
- `test_v20_integration` **fixed 2026-08-06** (7/7, verified repeatable across two consecutive runs) — see below.
- `test_v19_integration` remains known-stale (see the note after next).
- `test_stress_run` — **unresolved**. Produced a **zero-byte** log and exit code 4 on a background run: it died before printing its first line, so this is a crash/kill, not an assertion failure. Note its assertion block builds `UniversalOptimizer` **without** passing `state_dir=` (it sets `state_dir` on the cfg object instead, which the constructor does not read), so it writes to the shared default `results/` — worth ruling out state pollution when diagnosing.

**Known-stale test #2 — FIXED 2026-08-06: `test_v20_integration::test_v20_3_formula_library`.** Two independent defects, both pre-existing, and the second hides the first:
1. Line 270 asserts `SEED-logistic["status"] == "retired"` after 20 consecutive failures. Retirement is now a **two-step pipeline** — `record_result` sets `"retired"` (evidence ≥ 10 and confidence < 0.3), then E3's `prune_retired_to_archive` moves it to `"archived"` with a tombstone. Measured on a clean library: 5 failures → `seed`, 10+ → **`archived`**, never observed as `retired`. The assertion predates E3. Fix: accept `("retired", "archived")` — the intent is "leaves the working pool".
2. That failure skips the cleanup at line 273, which is the ONLY thing that deletes `%TEMP%/test_formula_lib.json` — a **fixed** path, not a per-run temp dir. The stale file then makes the *earlier* assertion at line 246 fail on every subsequent run (`best["evidence"] < MIN_EVIDENCE_BOOTSTRAP`, but seeds are already bootstrapped past the floor and F-0001 sits at evidence=10). **Proven**: delete the file → line 246 passes and line 270 fails; run again → line 246 fails. So the reported failure is not the real one. Fix: `mkdtemp()` per run plus `try/finally` cleanup.

**Both `test_v19_integration.py` and `test_v20_integration.py` under `tests/` are SYMLINKS into `figpso_uil_human/`** — the only two symlinked tests in the directory, and both are among the four suite failures. Editing them modifies that copy, so they were left untouched.

**Known-stale test:** `test_v19_integration` TEST 4 asserts miner R² > 0.3 on synthetic data whose planted pattern matches only **6 of 100 rows** — after bug #2's 75/25 held-out split, ~1 matching row lands in the test half, so held-out R² is legitimately 0.0. The test encodes the *pre-*bug-#2 in-sample behavior. The miner is fine (0.807 on the same generator with a denser pattern). Also: run this file with `PYTHONIOENCODING=utf-8` — 4 further "failures" are just cp1252 choking on `✓`.

---

## Enhancement (2026-07-17) — sparse linear over a symbolic basis

### E4 — SymbolicFormula gets a `sparse_linear` learning mode (`adapters/symbolic_formula_adapter.py`, `core/strategy_classifiers.py`)
Diagnosis: the boosting mode adds ONE library formula per stage, seeds reference ~1 feature each, so a 10-round fit touches ~6 of 57 columns. Measured on spambase, the readability–accuracy frontier for a 6-term model is **0.690** — and boosting scored **0.685**. The learner was **already at its ceiling FOR ITS SPARSITY**; the gap to XGBoost was capacity, not search. Reaching parity requires ~66 jointly-fitted terms (frontier: 6→0.690, 25→0.873, 43→0.921, 66→0.942).

New mode fits a **sparse linear model over a composed symbolic basis**: `sigmoid(Σ wⱼ · basisⱼ(x))`, where the basis is composed of three named families:
- **raw** (`x_j`) — raw features
- **formula** (`f(x)` from the evolved library) — the symbolic contribution
- **stump** (`1[x_k > θ]`) — axis-aligned splits as basis functions (user's KG+DT+NN "trees as basis" idea, cleanly integrated)

L1-regularized logistic regression fits *and* selects the terms jointly — this is what the user asked for by "select features according to accuracy, not correlation" (correlation was already the pre-hoc gate; L1 does it optimally, post-fit). Each surviving term stays a named, readable, editable cell; `readable_sparse_linear(model)` renders the fitted law as e.g. `sigmoid(-0.88·1[x26>-0.24] +0.59·1[x6>-0.28] ...) - 0.70`.

Basis families are individually gated (`use_raw_basis`, `use_symbolic_basis`, `use_stump_basis`) so the router can specialize per fingerprint — stumps helped spambase (+0.021) but hurt covertype (−0.030) in the research sweep. The domain router now maps its long-defined `sparse_linear` strategy (blueprint: "sparse linear over symbolic basis (readable)") to this mode, not XGBoost.

Measured (isolated, held-out): spambase 0.685→**0.949** (beats XGBoost 0.944). Related to the well-known RuleFit family; the novelty here is the composed basis (evolved formulas + stumps + raw) and full editability/provenance around each fitted term.

### E5 — `rule` basis family: readable feature INTERACTIONS (`adapters/symbolic_formula_adapter.py`)
Diagnosis (2026-07-18 audit): the E4 basis is purely **additive** (raw + stump + formula are summed), so it cannot represent feature INTERACTIONS. Proven the limit on sonar: every LINEAR model caps ~0.77 (LogReg 0.771, LinearSVC 0.767, UIL sparse_linear 0.746), while NON-LINEAR models reach ~0.88 (RandomForest 0.852, SVM-RBF 0.886). The ~0.13 gap is *entirely* interaction capacity — sonar's rock/mine boundary needs "band A high AND band B low," which a weighted sum can't express. This is the honest "price of readability," BUT it does NOT mean non-linear ⟹ unreadable: a decision rule `1[x3>0.2 & x7<=0.5]` is non-linear AND readable AND editable.

Added a fourth basis family, **rule** — RuleFit-style conjunctions mined as root→leaf paths of a shallow forest (`RULE_TREES=40`, `RULE_DEPTH=3`, `MAX_RULES=40`), each a `{feature,op,threshold}` AND-list, L1-selected jointly with the additive families. Rendered readably (`1[x47<=0.052 & x44<=0.28 & x20<=0.683]`); each rule stays a named, editable cell; `get_selected_feature_indices` credits every feature in a rule. Gated by `use_rule_basis` (**default OFF** — adds a forest-fit per fit and can overfit low-interaction data; the router enables it per fingerprint, like the stump gate).

**Key finding — rules need their own regularization budget.** Simply adding rules under the default strong L1 (`sparse_l1_c=0.1`) does NOTHING (sonar 0.748→0.748) — the 60 raw + 180 stump terms consume the whole L1 budget and starve the rules. With **weaker L1 (`sparse_l1_c≈0.5`)**: sonar **0.748→0.786** (+0.038, closing ~half the gap to RBF while staying readable). Decisive proof on a **pure-XOR** target (additive is provably at chance): additive **0.473 → +rules 0.616 (+0.143)**. So the router must set BOTH `use_rule_basis=True` AND a higher `sparse_l1_c` for interaction-dense fingerprints. Test: `tests/test_rule_basis.py`. **Invariant: any group of many correlated basis terms (rules) competing under a single L1 with a few strong terms needs a weaker penalty or a per-group budget, or it gets starved.**

### E6 — Interaction-aware routing (auto-enable rule basis) — REVERTED TO OPT-IN after cross-seed testing (`cognition/fingerprint.py`, `core/optimizer.py`, `cognition/domain_router.py`)
Intent: auto-enable the E5 rule basis when a dataset needs interactions. `fingerprint._axis4_dependence` adds a **non-linear probe** — a shallow RandomForest (60 trees, depth 8, subsampled ≤2000 rows) CV score — alongside the linear (LogReg) probe; `compute_fingerprint` surfaces `nonlinear_probe_acc` + `interaction_gap = nonlinear − linear`. In `fit()` step 0, when `interaction_gap ≥ 0.05` AND the classifier is a SymbolicFormula, the optimizer *would* `apply_config({use_rule_basis: True, sparse_l1_c: max(current, 0.5)})`.

**A 3-seed cross-seed sweep (2026-07-18) FALSIFIED the auto-enable — it caused NET REGRESSIONS, so `cfg.auto_rule_basis` now defaults to `False`.** ionosphere (gap ~0.06) dropped **0.916→0.868** (rules + weaker L1 overfit near-linear data; val-gap went +0.044), and sonar (gap ~0.07) — which the rules DO help in ISOLATION (bare adapter 0.748→0.786) — got **no** benefit through the full optimizer (**0.722**, worse than boosting's 0.746) with a +0.062 optimism gap. ionosphere's gap (0.051–0.066) and sonar's (0.072–0.077) are too close for any clean threshold, and neither wins through the pipeline. Walk-back verified full recovery: ionosphere→0.9155±0.030, sonar→0.746±0.030. **Lesson: single-split checks (which looked fine) hid this; only cross-seed exposed it — Finding #1 in action.** The E5 rule basis remains a proven, opt-in capability; `interaction_gap` stays as a diagnostic; re-enabling would need a higher gate (>~0.08) and validation on larger datasets where optimizer overhead doesn't dominate.

**Kept (correct + independent of the revert):** the `D4_structured` / `D4_moderate_linear` routing fix — linearly-separable classification now routes to **`sparse_linear`** (readable, ties/beats XGBoost on this data; blueprint's stated purpose) instead of black-box `gbm`. All M0–M5 + rule/OvR tests pass.

---

## E8 — `neural` basis family: the MLP as a basis, not as the model (2026-07-23)

**User's idea, and it holds.** Rather than choosing between the MLP and the symbolic model, use the MLP's hidden units AS a fifth basis family under the same L1 readout. A small MLP (`NEURAL_UNITS=16`, one hidden layer) is fit as a *feature builder*; each surviving unit becomes a named basis term with its weight vector stored on the model (plain lists, so it survives the champion store's JSON round-trip — bug #28). The top-level law stays a readable weighted sum, and families MIX freely:

`sigmoid(-1.551·x20 -1.260·x23 -0.934·1[x10>0.53] -0.802·relu(u15<-x7,x26,x20) ...)`

**Measured (single split, 25% test).** The XOR row is the decisive one — additive families are provably at chance there, so any gain is real interaction capacity:

| data | additive | +rules | +neural | +neural+rules | MLP (black box) |
|---|---|---|---|---|---|
| XOR (d=6) | 0.5133 | 0.8867 (39 terms) | **0.9133 (18 terms)** | **0.9400** | 0.9400 |
| interaction-dense (d=20) | 0.6320 | 0.5520 | **0.7920** | 0.7920 | 0.8320 |
| breast_cancer | **0.9720** | 0.9580 | 0.9650 | 0.9580 | 0.9161 |
| wine (binarized) | 0.9111 | 0.9333 | 0.9333 | 0.9333 | 0.9556 |

Three findings:
1. **On XOR the combined model EQUALS the black-box MLP (0.9400) while staying a readable sum** — and `+neural` alone beats `+rules` with less than half the terms (18 vs 39). Neural units are a far more compact way to buy interaction capacity than conjunctions.
2. **It is not a global win, and neither is the MLP.** On near-linear breast_cancer the plain additive model is best (0.9720) and the standalone **MLP is the WORST learner on the table (0.9161)**. Same shape as stumps (spambase +0.021 / covertype −0.030) and rules (sonar +0.038 / ionosphere −0.048): capacity helps only where interactions exist. So `use_neural_basis` defaults **OFF** and is exposed as a **race axis**, never a global switch — that global switch is exactly what E6 was, and E6 regressed.
3. Rules can actively HURT where neural helps (interaction-dense: rules 0.552 vs neural 0.792), so they are not interchangeable interaction mechanisms and both belong on the race.

**Caveat: these are single-split numbers on 4 datasets.** The XOR/interaction rows are synthetic with strong planted signal and are robust; the two real-dataset rows are NOT cross-seed validated and must not be quoted (Finding #1). Gate 2 before any default changes.

Also added: `adapters/mlp_adapter.py` (registered as `"MLP"`) — a standalone dense baseline / ceiling probe. `supports_warm_start` is deliberately `False` (warm-starting a net across candidates within a stage is bug #29's substitution trap), and `predict` re-applies the identical column scaling `fit` used (bug #19's invariant). It is NOT readable or editable and should not become the default learner; its value is as an honest ceiling, and the table above shows that ceiling is sometimes BELOW the symbolic model.

## Enhancements (2026-07-15) — high-dim, evolution cadence, pruning

### E1 — SymbolicFormula feature preselection (`adapters/symbolic_formula_adapter.py`)
On >`FEATURE_SELECT_CAP` (64) features the formula grammar could only reference ~16 columns (seed cap) → high-dim data collapsed to majority class while PSO/hybrid churned every column (MNIST-784: 7.2h, all-one-class). `fit_weighted` now preselects the top-K discriminative features (ANOVA F blended with PSO weights), runs the whole pipeline on that reduced basis, and stores `selected_features` on the model; `predict` and `get_selected_feature_indices` re-apply / invert it. **MNIST-784: 7.2h→0.2s, collapsed→0.934 (majority 0.887).** Selection is recorded (traceable/overridable).

### E2 — Evolution can fire on stagnation (OPT-IN) (`core/optimizer.py`)
Both formula-evolution blocks (loss + SymbolicFormula population) and the neural proposer were gated at `system_stress >= 0.6`, which short runs never reach → evolution stayed asleep. `_want_new_structure()` now also fires on stagnation (`_stages_since_improvement >= 2`) **but only when `cfg.enable_stagnation_evolution = True`** (default off — a plateaued run otherwise births every stage, a runaway risk), throttled to one birth per channel per `EVOLUTION_MIN_GAP` (3) stages. Default behavior = the original stable stress gate.

### E3 — Learning-preserving formula pruning (`cognition/formula_library.py`)
`prune_to_cap()` archives the weakest **active** mutants beyond `MAX_WORKING_ACTIVE` (60) to the queryable KnowledgeArchive, leaving tombstones (seeds/proven/incubating kept). Wired in `fit()` on a short cadence (every 5 stages, so short runs prune too) for BOTH libraries, alongside the existing `prune_retired_to_archive`. Verified: library 88→20 active / 33 archived-tombstones after an 8-stage run. This is the mitigation for E2's faster births — nothing is lost, working pool stays lean.

## E7 — Architecture Race replaces PSO surrogate search (2026-07-23) — BUILT, GATE FAILED, OPT-IN

**Diagnosis.** `pso_search.propose_actions` scores a candidate mask with
`all_fitness = Xp @ guidance - adaptive_penalty * count / n_features` — **linear and separable** in the mask, so the optimum is always `include j iff guidance[j] > penalty/n_features`. 20 particles × 30 iterations were re-deriving an argsort. **Proven, not asserted**: brute force over all 2¹⁰ masks matches the threshold rule 20/20 trials (`tests/test_architecture_race.py::test_r1`). Compounding it, the whole cognitive layer's only channel is `max_penalty` = *where that sorted list is cut*, so bug #25's ceiling is structural, not a wiring detail. Confirmed in the data: every principle in `bootstrap_knowledge.py` emits only `max_penalty`/`min_penalty`; **no principle anywhere emits a `config.*` override.**

**Built** (`search/architecture_race.py`, `cfg.search_mode="race"`, default `"classic"`):
1. `frontier_masks()` — the cutoff frontier in closed form (one argsort), replacing the swarm.
2. `ArchitectureRace.propose()` — one-axis-at-a-time candidates over `get_architecture_axes()` (new, opt-in adapter method: `sparse_l1_c`, `use_rule_basis`, `use_stump_basis`, `boosting_stages`) — the widened vocabulary bug #25 asks for. Kept **separate from `get_controllable_params`** so the classic per-stage search still cannot touch architecture (bug #21).
3. `rungs()` + `optimizer._race_actions()` — successive halving: cheap subsampled rungs, then one full-fidelity rung. ~7.2 full-fit equivalents for 12 arms at n=5000.
4. `ArmMemory` — per-routed-domain Beta posterior; arm budget 12→8→5→3→2 as evidence accumulates. Detachment becomes **measured**, not asserted (contrast bug #4).

**GATE 2 RESULT — FAILED. Do not enable by default.** (3 seeds × 6 datasets, `eval_crossseed --search-mode race`; classic arm reproduced the published numbers exactly, so the harness is sound.)

| dataset | n | classic | race | Δ | val-gap swing |
|---|---|---|---|---|---|
| wine | 178 | 0.9537 | 0.9537 | +0.0000 | +0.0000 |
| sonar | 208 | 0.7460 | 0.7302 | −0.0158 | +0.0289 |
| ionosphere | 351 | 0.9155 | 0.8592 | **−0.0563** | **+0.0722** |
| breast_cancer | 569 | 0.9737 | 0.9678 | −0.0059 | +0.0022 |
| digits | 1797 | 0.9472 | 0.9519 | **+0.0047** | +0.0058 |
| spambase | 4601 | 0.9439 | 0.9472 | **+0.0033** | −0.0018 |

Mean **−0.0117** (2 win / 1 tie / 3 lose).

**ROOT CAUSE IS DIAGNOSED, AND IT IS NOT THE VOCABULARY — IT IS SELECTION OVERFITTING.** `corr(Δaccuracy, val-gap swing) = **−0.977**`: essentially every point race loses is accounted for by increased validation optimism. The race picks best-of-9 arms on the **same single validation split** the classic path used for best-of-4, and that split is tiny on small data (ionosphere val ≈ 56 rows). `corr(Δ, log n) = +0.42` — it **wins on the two largest datasets and loses on the small ones**, exactly as winner's-curse predicts. The wider vocabulary works; the *judging* is under-powered.

**JUDGE FIX IMPLEMENTED + GATE-2 RE-RUN (2026-07-24) — NET POSITIVE, but keep opt-in.** The final rung is now selected on `self._extra_val_splits`: `_race_actions` scores every finalist on the extra resampled splits and attaches `select_score` (the multi-split mean); the stage loop's argmax reads `select_score` when present, falling back to the primary score — so the classic path and cv_folds=1 are byte-identical to before. The winner's M0.1 re-score reuses the cached extra scores (no duplicate fits). `intervention_results` still reads the primary score, keeping the causal control/treatment comparison single-split-internal by design. `cfg.race_select_on_extra_splits = False` restores the old (diagnosed-broken) selection for A/B. Test: R10 in `tests/test_architecture_race.py` — runs the race at cv_folds=3 against a pass-through recorder of the real `_race_actions`, verifies finalists carry `select_score`, the winner is its argmax, and two identical runs stay byte-identical. (R1–R9 all run at cv_folds=1 where this path cannot fire — they prove no regression, not the fix.)

**Gate 2 re-run — MATCHED same-session, same-code (both arms current), because this session's determinism fix (bug #31) moved the baseline: new-classic breast_cancer 0.9708 / digits 0.9500 vs the pre-session published 0.9737 / 0.9472, so the old E7 table above is NOT a valid comparator anymore.** Race − classic, stages 20, seeds 42/7/123, cv_folds 3:

| dataset | n | classic | race(judge) | Δ | val-gap swing |
|---|---|---|---|---|---|
| wine | 178 | 0.9537 | 0.9537 | +0.0000 | +0.0038 |
| sonar | 208 | 0.6984 | 0.7381 | **+0.0397** | −0.0233 |
| ionosphere | 351 | 0.9155 | 0.8685 | **−0.0470** | **+0.0807** |
| breast_cancer | 569 | 0.9708 | 0.9912 | **+0.0204** | −0.0242 |
| digits | 1797 | 0.9500 | 0.9806 | **+0.0306** | −0.0028 |
| spambase | 4601 | 0.9439 | 0.9482 | **+0.0043** | −0.0007 |

Mean **+0.0080** (4 win / 1 tie / 1 lose) — the race's standing vs classic went from **−0.0117 (E7 failure) → +0.0080**, and crossed zero. The judge fix is real.

**BUT the mechanism is only reduced, not solved.** `corr(Δacc, val-gap swing) = −0.932` — barely below the E7 failure's −0.977. The one regression (ionosphere −0.0470) has by far the largest optimism swing (+0.0807): the exact winner's-curse signature. On near-linear data the race's architecture arms (rules/stumps/neural) look better on the 3 internal folds and worse on the true external test, and 3-fold CV on n=351 (train split ~225 rows) doesn't catch it. This is the SAME data E6 documented (ionosphere 0.916→0.868 with rules). `corr(Δ, log n) = +0.11` — the old size-dependence is now weak (sonar n=208 gained +0.040 while ionosphere n=351 lost), so it is NOT a pure size story: it is *data linearity*, not just data size. **Do NOT flip `search_mode` to `"race"` by default** — a −0.047 regression on real data is exactly the E6 "unvalidated improvement" the project convention forbids shipping. Keep opt-in. **Next fix (interaction-gap gate) — IMPLEMENTED but MEASURED NOT to work; the signal is inadequate (2026-07-24).** `optimizer._gate_interaction_axes` drops the interaction-hungry race axes (`use_rule_basis`/`use_neural_basis`) when `fingerprint.interaction_gap < cfg.interaction_gate`, `cfg.race_all_axes=True` opts out, R11 unit-tests the logic, and it is harmless (opt-in, race-only, strictly narrows an overfitting search). **But a targeted A/B on the two discriminating datasets returned +0.0000 on both — the gate never fired.** Diagnosis: `interaction_gap` is NOISE-dominated at the ionosphere/sonar boundary. Standalone `compute_fingerprint` gives ionosphere 0.051–0.066 and sonar 0.072–0.077 (a clean-looking 0.006 separation), but the OPTIMIZER computes the probe on `X_train` (a different row subset) and gets ionosphere **0.0857** — above both thresholds AND above sonar's range. The same dataset's gap swings 0.05→0.086 depending on probe data, so **no threshold reliably separates "near-linear, rules overfit" (ionosphere) from "interaction-dense, rules help" (sonar)** — confirming E6's original conclusion. The mechanism is kept as an opt-in tool but it does NOT solve the ionosphere regression, which remains open. **The better-founded alternative — incumbent-margin rule, BUILT (2026-07-24), Gate 2 FAILED.** `optimizer._pick_winner_with_margin` centralizes stage-winner selection (one helper, used by both the stage loop and `_race_actions`' arm-memory credit — Category-C discipline). Default (`cfg.race_incumbent_margin == 0`) is a plain argmax on `select_score`, byte-identical to the prior inline loop. When set, a challenger arm displaces the incumbent (arm 0, carrying cognition's sparse_linear) only if its multi-split `select_score` exceeds the incumbent's by that margin. Unit-tested: R12 in `tests/test_architecture_race.py`. **Gate 2 (2026-07-24, same-session classic vs race@margin=0.01, stages 20, seeds 42/7/123):** mean **+0.0010** (3 win / 1 tie / 2 lose) — ionosphere still **−0.0470** (identical to race@margin=0), and breast_cancer flipped from the prior race win (+0.0204) to **−0.0030** (margin blocked helpful challengers). Digits still +0.0213 (dulled from +0.0306). **Do not enable; do not try 0.02** (0.01 already costs a win while leaving the regression untouched). `cfg.race_incumbent_margin` stays 0; `search_mode="race"` stays opt-in. Logs: `results/gate2_incumbent_margin/`.

**Invariants added:**
- `learning_mode` must **never** be a race axis — `apply_config` locks it permanently on first non-auto set, so a *losing* arm evaluated on the main classifier would pin the run's architecture. Router owns mode (M3).
- The **incumbent arm is never eliminated early** — it carries the cognitive layer's proposal; a 150-row subsample must not veto cognition before it is measured at full fidelity.
- Elimination rungs must be **side-effect free**: `_evaluate_one_action` mutates shared adapter bookkeeping, so rungs save/restore it (same attrs as `_score_extra_splits`). Verified by determinism (R7).
- **Causal control/treatment arms are never eliminated** — comparing them across different rungs makes the causal verdict an artifact of where each died.
- A `config.*` principle override must **not** overwrite the axis an arm is racing (it would set every arm identically and collapse the race). `_apply_overrides` skips `action["_race_axis"]` and now deep-copies `classifier_config` (it previously mutated the caller's nested dict despite its "Don't mutate original" comment).

Tests: `tests/test_architecture_race.py` (R1–R10, all passing; R10 exercises the judge fix at cv_folds=3). M0–M5 + multiclass OvR + rule basis all still pass.

---

## Key invariants (things that must stay true)

- `apply_config()` in XGBoost adapter uses `key in self.params` guard — all tunable params must be seeded in `_build_params()` or they silently do nothing.
- `tree.score()` after miner training must use a held-out split, not the training set.
- `CONFIRMED_CAUSAL` requires majority of the last 3 trials, not a single trial.
- `update_maturity()` must receive rolling accuracy, not the all-time peak.
- `discover_relationships()` must receive `base_rate` from the optimizer for lift-based classification.
- Cross-dataset rules must be penalised by 0.75× confidence before entering the library.
- `MetaPrincipleMerger(filepath=None)` for tests; `_state_path(...)` in optimizer.
- **The XGBoost custom (formula) objective is binary-only.** `supports_custom_objective` must stay `num_classes <= 2`; any new adapter claiming custom-objective support must shape-check multiclass.
- **All `VersionStore` state mutation goes through its `RLock`.** New methods that touch `_last_seq`, `_versions`, or segment files must take the lock.
- `LLMMutator` produces proposals only — it must never write knowledge state directly.
- Normalization transforms must remain invertible with stored provenance (symbolic laws are back-transformed to original units). **Every inference path must apply `self._normalization.forward(X)`** — `fit()` trains in normalized space, so serving raw X is a silent train/serve mismatch (bug #19).
- **Missing-data handling is the same train/serve chain:** `fit()` fits `MissingDataSpec` on the train fold (median / integer-mode fill + indicators only for columns that had train missing), applies it before fingerprint/normalization; `predict()` goes through `_perception_forward` (impute → normalize). Fully finite train data is a no-op (shape and values unchanged). Opt out with `cfg.handle_missing_data=False`.
- `state_dir` isolation: experiment/ablation runs must never share persisted state files with default runs.
- **Report macro-F1, not just accuracy or weighted-F1.** `fit()` returns `macro_f1` (unweighted mean over classes) in the result dict + stage snapshot. Accuracy and weighted-F1 both hide minority-class weakness on imbalanced data (adult: acc 0.82 but macro-F1 0.78). Any accuracy claim on imbalanced data must cite macro-F1.
- **Any readable/exported form of a model must round-trip**: `evaluate_law(export_law(model), X)` must equal `predict(model, X)` exactly, and no quantity `predict()` uses (standardization, threshold, `selected_features`, `positive_class`, per-head calibration) may be omitted from what a human is shown. Truncation must announce itself. This is bug #32; `tests/test_law_fidelity.py` enforces it.
- **A Gate 2 run must persist per-(dataset, seed) rows**, not just arm means — pass `--json` to `eval_crossseed`. Three past verdicts are permanently unreproducible for want of this, one of which flipped a default ON.
- **Never quote a single train/test split as "the" accuracy — it's a lucky/unlucky sample.** UIL is fully deterministic (same seed → byte-identical predictions; verified), so a single split is recorded+recreatable but NOT representative. Report cross-seed mean±std via `experiments/eval_crossseed.py`. (Single-split ionosphere 0.958 vs cross-seed 0.916±0.030.)
- **cv_folds auto-scales down by dataset size** (`optimizer.fit()`, `cfg.auto_cv_scale=True` default): n>20k → 1 fold, n>8k → ≤2. k-fold regret triples per-stage eval cost, impractical at scale (covertype 100k ran >40min CPU with cv_folds=3). Recorded in `_observability_meta["cv_folds_autoscaled"]`. Opt out with `cfg.auto_cv_scale=False`.

---

## Persistence files (in `results/`)

| File | Written by | Content |
|------|-----------|---------|
| `uil_journal_v19.json` | `uil_journal.py` | All stage entries (state, action, outcome, stress, regret) |
| `uil_network_v20.json` | `editable_network.py` | Nodes, edges, co-activation tally |
| `uil_principles_v19.json` | `uil_library.py` | All mined + seed principles with confidence/tier |
| `uil_causal_v20.json` | `causal_intervention.py` | Hypotheses, evidence_trials, causal status |
| `uil_episodes_v20.json` | `episodic_attention.py` | Dataset fingerprints + resolutions |
| `uil_formula_library.json` | `formula_library.py` | Loss-formula population |
| `uil_meta_principles.json` | `meta_principle_merger.py` | Merged meta-principles |
| `events/` (segments + snapshots) | `version_store.py` | Event-sourced knowledge history. **Event segments are uncapped/never deleted** (the lossless history). **Snapshots are compacted**: written gzip-compressed, and `compact()` prunes redundant intermediate ones (see "Snapshot compaction" below) |
| `uil_fingerprints_v22.json` | `fingerprint.py` | Five-axis dataset fingerprints |
| `uil_routing_memory.json` | `domain_router.py` | Router decisions + outcomes |
| `uil_knowledge_archive_v20.json` | `knowledge_archive.py` | Everything pruned from working pools |
| `uil_symbolic_classifier_library.json` | `symbolic_formula_adapter.py` | Formula-classifier population |
| `uil_graph_changelog_v20.json` / `uil_formula_changelog_v20.json` | `change_log.py` | Timestamped mutation logs |
| `uil_governance_manifest.json` | `bootstrap_knowledge.py` | Seed governance manifest |
| `uil_failures_v20.json` | `failure_point.py` | Captured failure diagnostics |
| `uil_live_feed.jsonl` | `live_feed.py` | Stage snapshots for the web UI |
| `uil_journal_v19_memory.json` | `uil_journal.py` | Per-dataset memory (best acc, stress override, preferred classifier) |
| `uil_guard_telemetry.json` | `guard_telemetry.py` | Observe-only bind-rate log for Category-A safety rails (audit Part E step 2): per guard `n_seen`/`n_bound`/min/max/mean/std of values seen. Accumulates across runs; scoped to `state_dir`. Never gates a decision. |

Note: default runs **share** persisted state across datasets and sessions (that's the design — accumulated knowledge). For isolated experiments pass `state_dir=` to `UniversalOptimizer` so arms can't contaminate each other.

### Snapshot compaction (`VersionStore.compact()`, 2026-07-18)

`results/events/` had grown to **60 GB** (6,900+ `snapshot_*.json` at ~8 MB each). Compaction:

```bash
python -m figpso_uil.cognition.version_store --compact           # run it
python -m figpso_uil.cognition.version_store --compact --dry-run  # preview plan + coarsening
# flags: --keep-recent N (default 200), --stride N (default 20), --no-gzip, --verify N, --dir PATH
```

What it does: (1) **never deletes an event** — segments are the lossless history; (2) **deletes redundant intermediate snapshots**, keeping the newest `keep_recent` + every `stride`-th older one (by seq, so it's idempotent) + the oldest; (3) **gzips the survivors** (and all *new* snapshots are now written `.json.gz` — `write_snapshot`/readers handle both). One run: **60.11 GB → 1.85 GB**. Re-running is a no-op.

**CRITICAL invariant — snapshots are NOT "pure acceleration" for this store.** The composite snapshot (`cell_store.build_composite_snapshot`) materializes from the LIVE graph + library files, and several mutations bypass the per-cell event stream: **edge/node bulk prune+decay emit ONE `GRAPH-BULK` summary event** (`editable_network._bulk_log`, bug #18), and **principle/routing_rule/formula/causal confidence decay + E3 tombstoning rewrite library files without per-cell events.** So `reconstruct(seq)` at a **deleted** snapshot's seq coarsens those fields to the nearest KEPT snapshot — measured up to ~11.6k edge cells differing across a stride-20 (10k-event) gap. This is by-design and was accepted for the space reclaim. What compaction **guarantees**: every event preserved; `reconstruct()` at every **kept** snapshot seq (incl. latest) byte-identical before/after (verified by hashing every bucket); trajectories & lab reports (which read events directly) unaffected; the M1 test (isolated per-cell store) still passes. **If you make bulk ops / decay emit lossless per-cell events, more buckets become event-reconstructible and compaction's coarsening shrinks accordingly.** Don't assume any bucket is losslessly replayable between two arbitrary snapshots without checking.

---

## "Ties XGBoost" has never been demonstrated (2026-08-05)

`results/compute_cost/HEAD_TO_HEAD.md`. `eval_crossseed` only ever ran the SymbolicFormula arm — the XGBoost side of the standing claim came from other harnesses on other splits, so the "tie" was never a paired result. `experiments/eval_head_to_head.py` runs UIL and every baseline on the **identical** split and pairs by (dataset, seed). 3 datasets × 3 seeds:

| baseline | mean Δ | 95% CI | W/T/L | cells needed for 0.01 | verdict |
|---|---:|---|---|---:|---|
| xgboost | −0.0046 | [−0.0289, +0.0197] | 2/4/3 | **79** | UNDERPOWERED |
| random_forest | −0.0097 | [−0.0314, +0.0120] | 3/2/4 | **63** | UNDERPOWERED |
| logreg | −0.0040 | [−0.0255, +0.0176] | 3/3/3 | **62** | UNDERPOWERED |
| symbolic_single_fit | −0.0056 | [−0.0175, +0.0064] | 2/3/4 | 19 | **NO MEANINGFUL BENEFIT** |

**Resolving ±0.01 against this variance needs ~79 paired cells; the project has run 9.** The tie may be true — it has never been shown at the precision it is stated with, and the same holds for RandomForest and logistic regression. The only adequately-powered comparison on the benefit side is against **one ordinary `fit_weighted` of the same learner**, where the CI upper bound (+0.0064) **excludes a 0.01 benefit** — reproducing `results/compute_cost/SUMMARY.md` on a different baseline set. Note also that a plain scaled **logistic regression wins 3 of 9 cells** and beats UIL by 0.026 on breast_cancer/42; framings that pit UIL against tree ensembles alone omit the cheapest baseline on the table.

## HITL is proven — on the LAW channel, not the knowledge-cell channel (2026-08-06)

`results/hitl_law_edit/SUMMARY.md`, `experiments/hitl_law_edit.py`. The last unfalsified value claim is now measured, with three powered arms and a control.

**Why every prior HITL attempt found nothing:** "human-editable knowledge" is two surfaces with opposite behaviour. **(a) knowledge cells** (principles/graph/formulas) — plumbing works, but **INERT**: 0.00% of predictions move. **(b) the fitted law** (sparse_linear terms) — **LIVE** since bug #32: deleting a term moves the logit by exactly the predicted amount. Every previous experiment, including `hitl_pilot_protocol.py`, aimed at (a) — a channel that provably carries nothing.

6 binary datasets × 30 seeds:

| arm | expert reasons from | cells | mean Δ | 95% CI | W/T/L | verdict |
|---|---|---:|---:|---|---|---|
| **iid** | the same training data | 180 | −0.0016 | [−0.0042, +0.0011] | 29/125/26 | **NULL CONFIRMED** (±0.01) |
| **shift** | knowledge the data lacks | 147 | **+0.3141** | **[+0.2984, +0.3298]** | **145/0/2** | **BENEFIT** |
| **shift, control** | the same training data | 180 | −0.0015 | [−0.0025, −0.0005] | 0/166/14 | negligible harm |

Channel liveness: **36.11% of predictions moved** (max 66.67%) vs **0.00%** for principle edits.

**The control is what makes it airtight.** On the *same shifted data*, the data-derived expert edited only **18/180** cells, moved **0.20%** of predictions, and never found the artifact — because **the artifact is the most stable term in training**. No resampling, CV or regularization can see it; only outside knowledge can. So the +0.31 is not automatable.

> **HITL's value is injecting knowledge the training data does not contain — not second-guessing a fit the model has already optimized.**

That explains the entire history of null HITL results: prior attempts tested experts reasoning from the same data, on iid splits, through an inert channel — three compounding reasons to measure zero. HITL is no longer "unfalsified": it is **falsified for iid/data-derived reasoning** and **confirmed for external knowledge under shift**.

**Caveats that travel with the number:** the expert is a program with oracle knowledge (proves the channel carries value *when the human is right*, not that humans are right — a real study still needs people); the artifact is synthetic and extreme, so +0.31 is not a general expectation; **33/180 shift cells excluded** where the law was *entirely* the artifact (sonar 30/30 — pure law-editing cannot repair a fully-captured model, that needs a refit); and this is the law channel, **not** the VersionStore/principle machinery, which remains inert.

**This was only possible after bug #32.** The expert points at `x37` and that really is column 37 — before the law-export fix, printed indices referred to a reduced basis and coefficients were in standardized space, so a human "correcting" the model would have edited a different term than the one they read. Readability is a *precondition* for editability, not a cosmetic nicety.

## gplearn registered as an optional symbolic learner (2026-08-05)

`adapters/gplearn_adapter.py`, registered as **`"GPLearn"`** in `classifier_registry` only when `pip install gplearn` is present (same optional pattern as TabPFN). Motivated by the extrapolation result above: it beat UIL's own symbolic regressor **17–4** with **0 blow-ups vs 19**.

**It does NOT replace `SymbolicFormula` on any routing strategy, and that is deliberate.** The evidence is about **regression / out-of-range recovery**; there is no benchmark of gplearn's *classifier* against SymbolicFormula on this project's datasets. Flipping a routing default on evidence from a different task is exactly what E6 did (auto-enable rule basis from an interaction-gap signal) and it regressed ionosphere 0.916→0.868. A `"symbolic_gp"` strategy key exists so the choice can be made and measured; pointing `"symbolic"` at it needs a Gate 2.

**The division of labour is the point.** gplearn programs are readable (`readable_program()`) but carry **no FormulaLibrary provenance, no VersionStore cells, and `cognition/law_export.py` does not apply to them** — so they sit outside the editable-knowledge machinery that is the project's other surviving claim. Use gplearn for out-of-range accuracy; use SymbolicFormula where the model must be inspectable, versioned and human-editable.

Tests: `tests/test_gplearn_adapter.py` GP1–GP4 (14 checks) — protocol conformance, determinism under a perturbed global RNG (bug #31's mechanism), OvR non-collapse (bugs #24/#33), and the feature-weight channel being live with `predict()` re-applying the identical transform (bug #19). **Note GP3: gplearn's OvR reaches all 3 classes on data where UIL's own `single`/`bagging`/`multi` modes collapse to one (bug #33).**

## Gate 2 re-scoring — most past verdicts are underpowered, not negative (2026-08-05)

`experiments/rescore_gates.py` re-reads the stored Gate 2 artifacts (**no new compute**), pairs by (dataset, seed), and reports CI + Wilcoxon + power. It reproduces every recorded mean exactly, so the re-analysis is validated against the originals.

| experiment | cells | mean | 95% CI | cells needed for 0.01 | re-verdict |
|---|---:|---:|---|---:|---|
| KAN basis | 18 | −0.0022 | [−0.0084, +0.0040] | 13 | **NULL CONFIRMED** (equivalent within ±0.01) |
| race @ incumbent margin 0.01 | 18 | +0.0010 | [−0.0154, +0.0174] | 86 | UNDERPOWERED |
| EBM cyclic (sonar) | 3 | +0.0079 | [−0.0582, +0.0741] | 60 | UNDERPOWERED |
| direct formula install | 4 | −0.0071 | [−0.0297, +0.0154] | 17 | UNDERPOWERED |

Three things follow.

1. **"FAILED" was frequently the wrong word.** KAN is the only adequately-powered experiment on the list, and what it establishes is *equivalence within ±0.01* — a real, quotable result, stronger than "it failed". The others could not have detected a 0.01 effect in either direction. In particular the **EBM "sonar +0.0080 win" quoted above is an order of magnitude below its own detection floor (MDE 0.083)** and should not be cited as a win.
2. **The pooled mean is the wrong summary when datasets disagree.** The race's per-dataset deltas span **0.0786** (ionosphere −0.047 vs sonar +0.032). The re-scorer flags heterogeneity and prints a dataset-level CI alongside the cell-level one; quote per-dataset.
3. **Three verdicts cannot be reproduced at all** — `try_uil_connect` (the **−0.0132** headline), `gate2_utility`, and `gate2_category_d` — because the artifact stored only `acc_mean`/`acc_std`. **Category D was flipped to default ON on that basis.** `eval_crossseed` now takes `--json` and writes per-(dataset, seed) rows including `fit_s`. **Never store only the mean again.**

## Ablation conditions

`experiments/ablation_harness.py` runs 5 conditions:

| Key | What it enables (baseline = all off) |
|-----|-----------------|
| `baseline` | Pure XGBoost + PSO |
| `formulas` | Formula engine |
| `network` | Knowledge graph |
| `principles` | Principle library |
| `causal` | Causal intervention engine |

Run: `python -m figpso_uil.experiments.ablation_harness`

**Do not re-run the accuracy ablations expecting a different answer** — measured repeatedly (2026-06/07): the cognitive layer ties plain search on accuracy, loses on compute. See "Honest empirical status" above.

---

## Pending work

- **Arm B MEASURED 2026-08-30 — `results/hitl_memory_transfer/SUMMARY.md`. The pre-registered falsifier PASSED; a check that was not pre-registered FAILED, and they must not be merged.** `experiments/hitl_memory_transfer.py`, 6 datasets × 10 seeds, leave-one-dataset-out. **B memory − C lookup = +0.2137, CI [+0.1678, +0.2596], 42W/11T/5L** — so "if B does not beat C, ship the lookup table" is not triggered. But **B fails specificity: it fires on 26.7% of healthy iid models, mean −0.0244, worst −0.2895**, so it is NOT shippable as-is. B also stays measurably below a present human (−0.0413, CI excludes 0). **Mechanism, threshold-independent: artifact terms score mean 0.744 vs 0.035 for healthy, but 9.8% of healthy terms score at or above the WEAKEST artifact term — no threshold separates.** All 12 harmful firings are breast_cancer, whose real features have standalone AUC ~0.97 against the artifact's ~0.98: by every training-data descriptor they are the same object. **So the memory INHERITS Arm A's identifiability limit rather than escaping it** — it transfers *what an artifact looks like*, never *this strong feature is real*. Converts "a human every time" into "a human whenever a genuine feature sits in the artifact's region": a real reduction in load, not autonomy. Next experiment is an **abstain on ambiguity** (not a higher threshold — separability rules that out). Secondary finding: the memory must record reviews that ended in NO edit, or it learns the rank reflex "delete the top term" (67% iid firing; `--no-negatives` reproduces it) — and the lookup table cannot use those reviews at all, which is a structural advantage for B independent of accuracy.
- **NEXT: cognitive-engine rescue — `claude_generated_files_and_plans/analysis/COGNITIVE_ENGINE_RESCUE.md`** (design, nothing measured). The engine's three killers are properties of the JOB (hyperparameter tuning on small saturated tabular data), not the code, and they invert when it is re-pointed at **human edits**: one row per edit instead of 8–40 per run, a discrete semantic action space instead of one scalar, and an exact judge instead of a 30-row val split. The shift control proves it can never *derive* human knowledge — only **remember and transfer** it. First experiment reuses `experiments/hitl_law_edit.py`: **A** no memory (+0.314 measured) / **B** memory, no human present / **C** lookup-table floor. **Falsification set in advance: if B does not beat C, ship the lookup table.** Cut the formula engine, stress/regret-as-search-signal, and the PSO loop; promote the **causal engine** to edit-locality (the open problem in Yao et al. 2023).
- **`claude_generated_files_and_plans/analysis/LITSURVEY_CORRECTIONS.md`** — the KDN literature survey's central Observed claim is attached to `NetworkNode`, which is measurably inert; it belongs to the fitted law. Exact before/after drafted for Sections 1, 2, 4, 9 + a proposed §5.4 (fidelity and edit-exactness acceptance tests). Not applied — it is an authored outward-facing document.
- **Bug #35 — probe leaks without a classifier factory.** Only enable `explore_architecture` alongside a `classifier_factory` until the save/restore fallback restores everything the probe's fit mutates.
- **Bug #34 — `situation_adaptive` overrides `explore_architecture`.** Decide whether that override is intended; if it is, the documented default should say so.
- **Bug #33 — OvR collapse off the `sparse_linear` path.** `single`/`bagging`/`multi` predict one class on multiclass data.
- **Underpowered gates.** Per `results/rescore_gates/REPORT.md`, the standing protocol (3 seeds × ~6 datasets) needs ~13–86 cells depending on variance. Decide a target effect and seed count *before* the next gate, and always pass `--json`.
- **HITL study** — the one unfalsified value claim (editability / trace / human correction between runs) needs a human-in-the-loop protocol, not another script. Draft in `experiments/hitl_pilot_protocol.py`. **Note the draft is not a study**: its "human" is `if r2_first < 0.9: n_terms = 10`, and on the logged run that branch never fired. The *mechanism* is now well tested (bug #32 round-trip + editability, M5/H1–H3); what remains is a real protocol — participants, paired tasks, a pre-registered primary metric, blinding on which arm produced which model.
- **Bug #35 — probe leaks without a classifier factory.** Only enable `explore_architecture` alongside a `classifier_factory` until the save/restore fallback restores everything the probe's fit mutates.
- **Bug #34 — `situation_adaptive` overrides `explore_architecture`.** Decide whether that override is intended; if it is, the documented default should say so.
- **Bug #33 — OvR collapse off the `sparse_linear` path.** `single`/`bagging`/`multi` predict one class on multiclass data.
- **Underpowered gates.** Per `results/rescore_gates/REPORT.md`, the standing protocol (3 seeds × ~6 datasets) needs ~13–86 cells depending on variance. Decide a target effect and seed count *before* the next gate, and always pass `--json`.
- ~~**`results/events/` retention**~~ — **DONE (2026-07-18)**. `VersionStore.compact()` + `python -m figpso_uil.cognition.version_store --compact` implement snapshot compaction; a one-time run took the dir from **60.11 GB → 1.85 GB**. New snapshots are now born gzip-compressed. Events are never touched. See "Snapshot compaction" below. (Remaining nice-to-have: a scheduled/periodic `--compact` so it doesn't need manual re-running as new snapshots accumulate.)
- ~~**Bug #30b state_dir disk leaks**~~ — **DONE (2026-07-24)**. Fingerprint/routing/governance/changelogs/live_feed scoped under `state_dir`.
- ~~**Category D Gate 2**~~ — **PASSED (2026-07-24)**. Soft maturity + stress adaptive blend default ON; `situation_adaptive` default ON (allow-list only — never Category A rails). See `results/gate2_category_d/SUMMARY.md`.
- ~~**EBM Gate 2** (`use_ebm_fit`)~~ — **DONE (2026-07-24)**. Sonar SL +0.0080; mean Δ +0.0014 across 6. **Default stays OFF** (race axis only — design invariant). See `results/gate2_ebm/SUMMARY.md`.
- ~~**Remaining Category-A telemetry**~~ — **DONE (2026-07-24)**. formula_engine GRAD_CLIP/MAX_DEPTH/MAX_NODES, feature_weight_floor, resource_budget clips now observe via threaded `GuardTelemetry` (no module singleton).
- ~~**Research steps 6–8**~~ — **DONE (2026-07-24).** (6) Thresholdout champion gate default ON; (7) TabPFN fingerprint probe + optional `D3_tiny_tabpfn` route; (8) `use_kan_basis` race axis default OFF. Tests: `tests/test_research_steps_6_8.py`. **KAN Gate 2 (2026-07-24): mean Δ −0.0022, sonar −0.0158 — default stays OFF** (`results/gate2_kan/`).
- ~~**KAN Gate 2**~~ — **DONE (2026-07-24), default stays OFF.** Mean Δ −0.0022; sonar −0.0158. Race axis / `--use-kan-basis` only. See `results/gate2_kan/SUMMARY.md`.
- ~~**Research F6 propensity footnote**~~ — **DONE (2026-07-24).** `logging_policy` on every mined rule/principle (HITL visibility only; no propensity reweight). Test: `test_f6_logging_policy_footnote`.
- **`optimizer.fit()` refactor** — the stage loop is ~1,400 lines in one method; each new subsystem raises the cost of the next. Wants extraction into named steps. Behavior-preserving only.
- **Exception hygiene** — broad `except Exception: pass` is pervasive (deliberate robustness, but it hid bug #10 for months). When touching a module, prefer `failure_point.py` capture over silent swallows.
- ~~**Formula evolution is stress-gated (≥0.6)**~~ — **FIXED (2026-07-23)**, see "Evolution was asleep, not disconnected" above. `_want_new_structure` now gates relative to the run's own stress peak (bounded above by 0.6), default ON. Measured: formula library 32→34 every run, accuracy identical 9/9. Remaining question is downstream of this fix, not the gate: evolved formulas still aren't selected into fits (see that section's closing paragraph).
- **Neural proposer retraining** — `results/uil_neural_proposer.json` is trained once via `python -m figpso_uil.experiments.train_neural_proposer`; it does not auto-retrain as the formula libraries grow. Re-run after substantial library growth to re-distill.
- ~~**Missing-data perception**~~ — **DONE (2026-07-24).** `core/missing_data.py`: train-only median/mode impute + missingness indicators; wired before fingerprint/normalization; `predict()` via `_perception_forward`. Default ON; clean data is identity. Tests: `tests/test_missing_data.py`.
- ~~**UIL master switch + connect pack**~~ — **DONE.** `uil_enabled=False` → plain learner. **Core wiring (translator) stays OPT-IN** via `uil_connect_pack` / explicit flags. **Gate 2 accuracy (2026-07-25):** mean Δacc **−0.0132** vs off. **Root cause (diagnose_wiring_drop):** drop is `connect_cognition` × nonuniform translator guidance (overfit — val−heldout gap 0.03→0.12 on wine/123); translator alone is neutral/helpful; utility innocent. See `results/diagnose_wiring_drop/SUMMARY.md`.

(The SymbolicFormula adapter previously listed here is **done** — `adapters/symbolic_formula_adapter.py`, registered in `classifier_registry.py`.)

---

## Other docs in this project

**Root tidy-up, 2026-08-30.** Loose documents were moved out of the project
root. Nothing was deleted and no Python module moved. Analysis/plan documents
are under `claude_generated_files_and_plans/` (see its `README.md` for the full
manifest and `UNDO_move.sh` to reverse the whole thing); the built
`01_`-`05_*.docx` now sit beside their `.md` sources in `academic_v2/`.
Some comments in `cognition/goal_cell.py`, `llm_mutator.py`, `neural_proposer.py`,
`formula_engine.py`, `situation_policy.py`, `core/optimizer.py` and
`experiments/eval_extrapolation.py` still cite these docs by bare filename --
they are prose citations, not file I/O, and were left untouched.


- `claude_generated_files_and_plans/analysis/RESEARCH_FAILURE_ANALYSIS.md` (2026-07-23) — maps every measured failure (inert layer, winner's curse, unreachable thresholds, no variation, unselected formulas, off-policy mining) to the published research that names it and its remedy (Cawley & Talbot nested CV, Dwork reusable holdout, PBT, DreamCoder utility-gated libraries, partial pooling, fANOVA, EBM, TabPFN v2, KANs), with an ordered 8-step program that preserves transparency/editability/reproducibility. **Read before designing the next fix — most of them are already designed in the literature.**
- `claude_generated_files_and_plans/analysis/HARDCODED_VALUE_AUDIT_QA.md` (2026-07-23) — full inventory of every hardcoded gating constant in the codebase: **~584 values across 54 files**, two independent sweeps converging on the same order of magnitude. Sorted into 6 categories (safety rails / already-fixed / duplicated-fix-now / hand-picked-formulas-need-data / semi-evidenced / arbitrary-candidates-for-evolution) rather than treated as one undifferentiated pile. Evidence-treated so far: `MIN_EVIDENCE`, evolution stress gate, **Category D maturity soft + stress adaptive (Gate 2 passed 2026-07-24, defaults ON)**, plus `situation_adaptive` allow-list. **Read before hand-tuning any constant, or before another "make everything dynamic" pass — Category A rails stay fixed.**

- `full.txt` — 14,465-line v19–v21 design specification.
- `UIL_V22_SCIENTIST_BLUEPRINT.md` — v22 design spec; the `§` references in module docstrings point here. Includes the do-not-do list (§8) — read it before proposing new subsystems.
- `RUN_UI.md` — web UI usage.
- `claude_generated_files_and_plans/unrelated_lca/agent_architecture.md` — a separate LCA (Layered Cognitive Agent) for UI automation. **Unrelated to UIL. Do not integrate.**
- `claude_generated_files_and_plans/unrelated_lca/lca_uil_analysis.md` — analysis concluding LCA and UIL should NOT be integrated.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

---

## Addendum (added 2026-08-12) — corrections and the transformer/KDN workstream

Addition only; nothing above this line has been edited. This section records
four stale claims still present above, the one open scope note, and the
pointer to a workstream that grew out of this document since it was last
touched.

### Four facts that changed since parts of this document were written

The "Honest empirical status" section at the top of this document, and
several later sections, state the following, which August 2026 measurement
overturned or narrowed. Each row: the claim as it still reads above, what was
actually measured, and the source of record.

| # | Claim as written above | What was measured | Source |
|---|---|---|---|
| A | "ties plain XGBoost+PSO on accuracy", "mean Δ ≈ +0.004" | **Never demonstrated at the stated precision.** Paired on identical splits: Δ −0.0046, CI [−0.0289, +0.0197] — 79 cells needed to resolve ±0.01, 9 run. A plain logistic regression wins 3 of those 9 cells. | `results/compute_cost/HEAD_TO_HEAD.md` |
| B | "10–100× the compute" (opening paragraph) | Wrong in both directions: the cognitive layer itself is **0.95×** (essentially free); the **search loop** costs **~3900×** one ordinary fit of the same learner. | `results/compute_cost/SUMMARY.md` |
| C | Extrapolation framed as "UIL's proven win" in some cross-references (see `claude_generated_files_and_plans/analysis/RESEARCH_FAILURE_ANALYSIS.md` F1 below, which this document is cited from) | Beating XGBoost out of range is a **theorem** (trees are piecewise-constant), not a finding. Against baselines that can actually extrapolate, UIL wins **0 of 21 cells**; against `gplearn` specifically, UIL loses **4–17** with **19 blow-ups vs 0**. `gplearn` itself beats every non-symbolic baseline, so the underlying thesis (symbolic methods are the right tool out-of-range) holds — UIL's own regressor is just a weak instance of it. | `results/extrapolation/SUMMARY.md` |
| D | "causal was the only subsystem ever measured CONNECTED" (in older existence-audit notes this document references) | Causal is **INERT** in all four `existence_sweep` cells and both 40-stage `existence_mined` cells. The subsystems actually connected are **principles** and **formulas** (both cells) and **network** (one cell) — and "connected" there means moving 0–2% of predictions with a net-negative accuracy effect, not a benefit. | `results/existence_sweep/FINDINGS.md`, `results/existence_mined/FINDINGS.md` |

**One addition, not a correction:** HITL is no longer unmeasured. A human edit
informed by knowledge the training data does not contain is worth **+0.3141**
accuracy under distribution shift (95% CI [+0.2984, +0.3298], 147 paired
cells), while the same expert reasoning from the training data alone is
**null** (−0.0016, CI [−0.0042, +0.0011], 180 cells). A control confirms the
gain is not automatable — a data-derived heuristic edited only 18/180 cells
and never found the planted artifact, because the artifact was the most
stable term in training. Two caveats travel with this number and should not be
dropped if it is cited elsewhere: the "expert" arm is a program with oracle
knowledge of the artifact, not a human study; and 33 of the 180 shift cells
were excluded because the law was entirely the artifact (pure law-editing
cannot repair a fully-captured model). Source: `results/hitl_law_edit/SUMMARY.md`.

### Known duplication in this document, not corrected here

Two blocks of text appear twice, verbatim, above: the three-command block
covering `existence_sweep` / `eval_extrapolation` / `compute_baseline` in *How
to run*, and the bug #35 / #34 / #33 / "Underpowered gates" block in *Pending
work*. Flagged here rather than fixed, per the addition-only scope of this
edit — deduplicating requires removing text, which this pass does not do.

### The transformer/KDN workstream this document does not yet mention

A concrete design for embedding KDN inside a small transformer's FFN
sublayer, and a v0 local-only transparency probe that precedes it, was
written up 2026-08-12 (plan file `claude_generated_files_and_plans/plans/synchronous-watching-seal.md`, "Workstream
2"). It is directly downstream of three things already in this document:

- **The "Ties XGBoost" / compute-cost sections above** are why the workstream
  does not attempt another accuracy claim — the swap experiment's acceptance
  bar is edit locality against a plain-distilled-MLP control, not raw
  accuracy.
- **The bug #31 determinism lesson** (every sklearn estimator needs an
  explicit `random_state`, or "randomness" silently means "seeded from a
  process-global source") is the exact failure mode the workstream's v0 Step
  0 checks for first: does the same input take the same path through the KDN
  every time.
- **`claude_generated_files_and_plans/analysis/COGNITIVE_ENGINE_RESCUE.md`**, listed in Pending work above as the
  **NEXT ACTION**, supplies the workstream's control-arm requirement: per its
  pre-registered falsification condition for the tabular HITL-memory
  experiment (if the memory arm does not beat a lookup-table floor, the
  lookup table ships), the transformer swap experiment is required to include
  a plain-distilled-MLP control on the same paired activations before any
  "KDN beats a plain net" claim is made.

See the addenda in `claude_generated_files_and_plans/analysis/DAPTIVE_KNOWLEDGE_NETWORK_DESIGN.md` §13 and
`academic_v2/06_TRANSPARENT_EDITABLE_NEURAL_SUBSTRATE_LITSURVEY.md` Appendix Z
for the full design; this note exists so a reader starting from this document
finds the pointer.
