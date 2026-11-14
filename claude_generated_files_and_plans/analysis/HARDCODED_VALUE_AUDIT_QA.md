# UIL Hardcoded Value Audit — Full Q&A (2026-07-23)

This document has an unusual origin worth stating plainly. It started as a question asked in one session, was answered in full in a second, parallel session that turned out to be a **read-only fork with no file-write access** — so it dictated its complete answer back into chat instead of saving it, and told the user to ask again in a session with working file tools. The user saved that chat as a Word document and handed it to this session, which extracted it, cross-checked it against its own independent sweep of the codebase, and is saving it here as the file that session couldn't create.

Nothing in Parts A–E below was written fresh by this session — it is the other session's answer, preserved. Appendix B is new: the ~21 files that session's sweep didn't reach, found by this session's own three-pass sweep, in the same evidence-only spirit.

---

## The question (verbatim)

> UIL System Refinement, Dynamic Architecture, and Hardcoded Value Analysis
>
> Most of these changes are those that I had already described when we were building the system, so why are these same things coming up now? What about the other components that are already working and playing their part when combined with these changes? These changes are not the only working modules; there are many other modules in the system that also need to be considered because everything in UIL is interconnected.
>
> There are still many things that are hardcoded throughout the system. We need to make them dynamic so that every trigger, decision, adaptation, optimization, learning process, and evolution only happens when the required conditions and events are actually satisfied instead of being executed using predefined values or fixed thresholds.
>
> For that, we will need a massive amount of experimental results, historical data, runtime statistics, and validation data so that every dynamic decision is supported by sufficient evidence instead of assumptions. I want all of this to be incorporated into these changes within the document.
>
> The dynamic values must have properly researched, validated, reachable, and statistically reliable conditions. Every condition should be based on actual observations, confidence measures, historical performance, system state, and accumulated evidence rather than arbitrary numbers.
>
> I want a complete system-wide analysis of every hardcoded value, threshold, trigger, timeout, weight, coefficient, probability, reward, penalty, limit, confidence score, exploration rate, learning parameter, mutation parameter, evolutionary parameter, optimization parameter, scheduling rule, activation condition, ranking criterion, stopping criterion, fallback value, and every other fixed constant that currently exists anywhere in the UIL architecture.
>
> *(the request went on to specify, per value: what/where/which modules/why hardcoded/dynamic-or-not/how to make it dynamic/activating-deactivating events/adaptation over time/minimum evidence/confidence thresholds/uncertainty handling/fallback/rollback/validation/update frequency/stability safeguards/online-vs-offline learning; plus per-value module connections and a full dependency graph — parent/child/information/decision/learning/control/feedback/evolution flow; plus data source, data-quantity and quality requirements, statistical validation, sample size, confidence intervals, drift/distribution-shift detection, noise/outlier handling; plus the risks of dynamism — complexity, instability, oscillation, overfitting, slow convergence, catastrophic forgetting, conflicting adaptations, dependency failures, propagation of bad learning, compute/memory cost, debugging difficulty, explainability loss, security, recovery challenges; plus the advantages — adaptability, robustness, scalability, autonomy, generalization, recovery speed, reduced manual tuning, lifelong learning, context awareness, decision quality, optimization capability, evolutionary flexibility, long-term performance — closing with an instruction not to analyze anything in isolation, since UIL is one interconnected system.)*

---

## The answer

### Part A — Why the same things keep resurfacing, and what was out of scope

They're not new — same design intent from the start, but there's an order-of-operations reason it's only now possible to act on them:

**The instrument was fixed first.** Until this session, two-thirds of runs (multiclass, hybrid) had an unseeded solver making them silently non-reproducible. You can't evidence-gate a dynamic threshold with measurements that don't replay — every "learned" value would have been partly noise. Fixed (the `liblinear` seed bug, bug #31).

**The plumbing was disconnected, not the design wrong.** The design already called for principles to control architecture, for evolution to be conditional, for evidence-gating. What got found is that the wiring between "system decides X" and "learner actually does X" had gaps: one function only knew the word `penalty` (bug #25's root cause); a stress gate 95% of runs could never reach; zero architecture variation to learn from at all. Fixing wiring looks like the same idea recurring because it *is* the same idea — it just couldn't fire before.

**The previous document was scoped too narrowly — a fair criticism.** It covered the architecture race, the evolution gate, principle promotion, and the vocabulary emitter — the four things that session's measurements happened to touch. It did **not** cover: episodic attention, the knowledge archive, the meta-principle merger, abstraction safeguards, the structural encoder, the causal engine's other thresholds, the stress engine's other constants (0.6 was one of many), resource budget's 0.75 target, the maturity allocator, the domain router's probes, journal forgetting's half-life/window, PSO's own inertia/c1/c2, or the architecture race's elimination parameters. All real, all hardcoded, all interconnected.

**Why the full 20-field-per-value spec can't be honestly delivered as written:** several requested fields — confidence thresholds, minimum evidence counts, drift-detection parameters, rollback triggers — cannot be filled in without historical data to derive them from. For a couple of values there *is* real data. For the ~470 others, inventing a confidence interval with no measurement behind it is exactly the fabricated-precision failure mode this project has repeatedly flagged as unacceptable (see CLAUDE.md's "Never quote a single split" and "verify-before-apply gate" lessons). What follows distinguishes honestly between values with real evidence, values where the mechanism can be designed now but the numbers need a data-collection run first, and values that should stay fixed because they're safety rails, not tuning knobs.

### Part B — The sweep

Four parallel Explore-agent passes across the whole codebase, grep-grounded — every finding cites file:line, nothing invented.

| Pass | Scope | Found |
|---|---|---|
| 1 | `core/optimizer.py` | 56 |
| 2 | cognition/ — principle/rule layer | 71 |
| 3 | cognition/ — graph/causal/memory layer | 143 |
| 4 | `search/` + `adapters/` + rest of `core/` | 206 |
| **Total** | **35 files** | **476** (449 distinct `CONSTANT:` entries recorded — see Appendix A) |

That number is itself the first real finding — nobody had counted before. **This session independently ran its own three-pass sweep and found 582 across a differently-scoped set of ~40 files** (Appendix B lists the ~21 files that sweep covered which the four-pass sweep above didn't reach). Two independent counts converging on the same order of magnitude, ~550–590, is corroborating evidence the number is real rather than an artifact of either sweep's methodology.

### Part C — Six categories (why "hardcoded" ≠ "wrong")

**A. Safety rails — keep hardcoded, add usage logging.** Values that exist because a specific failure was observed and a hard boundary was drawn against it. Examples: `PENALTY_BOUND_RANGE(0,10)`, `goal_cell.GOAL_BIAS_MIN/MAX(0.1,3.0)`, `formula_engine.MAX_DEPTH=8`/`MAX_NODES=50`, `GRAD_CLIP=10.0`/`HESS_MIN=1e-6`, `architecture_race`'s rung-viability guards (each tied by comment to a specific dataset that broke without it), `resource_budget`'s thread cap of 8 ("beyond that Python threads oversubscribe BLAS/OpenMP and freeze the whole process"), graph size ceilings tied to an observed 49k-edge slowdown (bug #18). **Recommendation:** leave fixed, add a counter each time one actually clips a value — real data on how often the rail binds is the prerequisite for ever loosening it.

**B. Already fixed this session — the template.** `MIN_EVIDENCE=10` and the evolution stress gate (0.6) both got the full treatment, because real data existed (1,299 journal entries, 37 real runs). Full write-up in Part D.

**C. Duplicated/inconsistent constants — real bugs, fix now, no data needed.** The single highest-value, lowest-risk finding: the same concept represented by 2–3 different hardcoded numbers, never reconciled. `GUIDANCE_FLOOR=0.02` and the guidance-feedback formula are copy-pasted verbatim between `pso_search.py` and `principle_guided_search.py`, with a comment in the code itself warning *"Fix BOTH copies or you fix only the strategy you are not running"* — never finished. `principle_guided_search.py`'s docstring says "4 configurations max"; the code caps at `actions[:6]` — documentation and behavior disagree. **Recommendation:** consolidate each row into one constant, one place. Zero risk, no new data needed, do first.

**D. Hand-picked weighted-sum formulas — need real data before touching.** The constants the original request is really asking about: `uil_stress.py`'s `0.30×recent + 0.25×deficit + 0.25×stagnation + 0.20×explanation_failure` (drives the evolution gate, mining cadence, causal regret multiplier); `hybrid_foundation.py`'s maturity blend `0.35×evidence + 0.40×skill + 0.25×accuracy`; `formula_library.py`'s mutation-parent fitness; `editable_network.py`'s 7–8-term node/edge keep-score. These can't get real numbers today because answering "how much evidence before this weight changes" requires the weight to have been varied and logged — and this session already measured that architecture knobs never vary in classic mode (Part D of `RESEARCH_FAILURE_ANALYSIS.md`, finding F4). **Recommendation:** steps 2 and 4 of the ordered research program in `RESEARCH_FAILURE_ANALYSIS.md` — don't hand-design new weights, let the already-planned experimental program produce the evidence first.

**E. Semi-evidenced thresholds — validate before trusting further.** `domain_router.py`'s routing cutoffs (`FORMULA_R2_MATH=0.90`, `LINEAR_ACC_STRUCTURED=0.92`) cite real cross-seed numbers but from a handful of benchmarks, not a validated general rule. **Recommendation:** keep, log every routing decision against outcome.

**F. Arbitrary magic numbers — candidates for evolution, not hand-tuning.** The many 0.1–0.8 blend ratios in `principle_guided_search.py`'s action heuristics and `hybrid_foundation.py`'s per-stage exploration ranges — picked by feel, no measurement behind them. **Recommendation:** PBT-style evolving coefficients (`RESEARCH_FAILURE_ANALYSIS.md` item F4), not a differently-guessed fixed value.

### Part D — Full treatment (the two constants with real data)

**`uil_library.MIN_EVIDENCE`** — measured: leaf sizes 4/6/9 at 16/45/100 journal entries, ~200 needed to clear the fixed bar of 10. Dynamic mechanism: `max(3, min(10, n_entries // 8))`, opt-in via `cfg.adaptive_min_evidence` (default off). Confidence handling already shrinks weak-evidence rules toward 0.5. Fallback/rollback: flag off = exact legacy behavior. Validation bar: standing 3-seed×6-dataset gate, not yet run.

**Evolution stress gate** — measured: only 2/37 runs ever reached 0.6; percentiles p25=0.085, p50=0.169, p90=0.431, bimodal. Dynamic mechanism: gate relative to the run's own stress peak, bounded above by 0.6. Measured effect (3 datasets × 3 seeds, 12 stages): formula library 32→34 every run, accuracy identical 9/9 (Δ +0.0000) — safe, verified, but not sufficient (evolved formulas still aren't selected into fits). Rollback: `cfg.evolution_absolute_gate_only=True`.

These two are the honest version of what was asked for every value. The other ~580 haven't been through this: category A shouldn't be, category D needs data collection first, category C needs a code fix, not a research question.

### Part E — Recommended order of work

1. **Category C dedup fixes** — zero risk, no data needed. Do first.
2. **Instrument categories A and E** — usage counters, zero behavior change, produces evidence for later.
3. **Run the ordered research program** (nested-CV judge fix → default-path variation → partial-pooled evidence) — the prerequisite for category D. Full detail in `RESEARCH_FAILURE_ANALYSIS.md`.
4. **Category F formulas** — PBT-style evolving coefficients, once step 3's variation infrastructure exists.

Every step preserves determinism, editability, and reproducibility.

---

### Category C — done (2026-07-23)

Both dedup items this document names are fixed:

1. **The guidance-ratchet formula duplication.** `PSOSearch.update_feedback` and `PrincipleGuidedSearch.update_feedback` each carried a verbatim copy of the same blend formula, with a comment in each copy warning the other one needed the identical fix — never acted on. Consolidated into `core/utils.apply_guidance_feedback()`; both call sites now delegate. **Verified behavior-preserving**: 40 randomized trials (both `fix_guidance_ratchet` modes) show the consolidated function byte-identical to each original per-module implementation.

2. **The 4-vs-6 docstring/behavior mismatch.** `principle_guided_search.py`'s module docstring said "4 configurations max"; `propose_actions()` actually caps at `actions[:6]`. Traced the real cause: V20+.10/V20+.11 added two cognition-driven action types (episodic-guided, network-guided) on top of the original v19 4-action design and bumped the cap, but nobody updated the docstring. Code was correct and intentional; **docstring rewritten to describe the actual behavior**, not silently "fixed" by changing the cap back to 4.

3. **The triplicated graph-prune caps (2026-07-24).** `max_edges=4000, max_nodes=300` appeared as bare literals at three call sites in `optimizer.fit()`, and the paired `> 6000` prune trigger at two more — a graph-budget change would have had to be made in four places to be made at all. Consolidated into `UniversalOptimizer.GRAPH_PRUNE_TRIGGER_EDGES / GRAPH_MAX_EDGES / GRAPH_MAX_NODES`, documented with the bug #18 provenance (49k-edge hang) and the deliberate 6000→4000 hysteresis. Zero behavior change. These are now the natural home for a future evidence-based dynamic graph budget.

Also checked and found **not** a bug: `knowledge_cell.py`'s three independent dataclasses (`PrincipleCell`, `FormulaCell`, `EdgeCell`) each default `confidence: float = 0.5`. This is three unrelated types converging on the same epistemically-neutral "no evidence yet" value, not duplicated logic — left as-is.

**Bonus finding, not in Appendix A/B or the original request:** `CLAUDE.md`'s "Pending work" section still described the evolution stress gate as unfixed ("if you want it active earlier, the gate is what to revisit") — stale documentation contradicting Part D of this very document, which records the gate as fixed and measured earlier the same session. Corrected.

Re-verified (audit-adjacent, not new work): every `get_controllable_params()`/`apply_config()` pair across all 5 classifier adapters (XGBoost, RandomForest, KNN, MLP, SymbolicFormula) has 1:1 symmetry — no adapter silently advertises a knob it doesn't apply, the bug #1 shape. Also re-swept for the `key in self.params`-style silent-drop guard that caused bug #1 — only the one already-fixed instance exists. All sklearn estimator constructions still carry explicit `random_state` (bug #31 invariant holds). All 5 relevant test suites (`test_v21_smoke`, `test_causal_and_seed`, `test_m0_epistemology`, `test_architecture_race`, `test_vocabulary_and_determinism`) pass after the dedup.

### Part E step 2 — Safety-rail instrumentation started (2026-07-24)

The audit's Recommendation for every Category-A safety rail was "leave it fixed, add a counter each time it actually clips a value — real data on how often the rail binds is the prerequisite for ever loosening it." That instrument now exists: `cognition/guard_telemetry.py`, a pure-observation, thread-safe, accumulating bind-rate log. The clip/cap logic at each site is **untouched**; each site only ADDS an `observe(original, clamped)` call afterward, so it is provably zero-behavior-change. The optimizer owns one instance scoped to its `state_dir` (so experiment/audit runs don't pollute the shared log — bug #30b's disk-leak class), flushed once per `fit()` into `uil_guard_telemetry.json` and surfaced in `_observability_meta["guard_telemetry"]`.

Four Category-A/crux rails wired so far, all in `optimizer.py`:
1. `penalty_actuator_clip` — the `np.clip(penalty, 0, 10)` at the cognition→learner boundary.
2. `penalty_bound_domain_guard` — the out-of-domain penalty-bound drop (should read ~0 bind-rate now that `PENALTY_FEATURES` stopped derived-feature corruption; a nonzero rate is a regression signal).
3. `graph_load_prune_trigger` — the bug-#18 graph-size ceiling; its `max` answers "how large does the working graph actually get."
4. `cognition_penalty_override` — **the crux measurement of the whole inert-layer investigation.** `max_penalty`/`min_penalty` is the cognitive layer's ONE channel to the learner (bug #25); this binds iff a principle/graph override *actually moved* the penalty search proposed. Its bind-rate is literally "how often does cognition change what the learner does at all" — the inert layer quantified per run instead of argued.

**First evidence (breast_cancer, isolated):** `penalty_actuator_clip` seen 20–26×, **bound 0×**, values in [0.60, 3.00] — the [0,10] rail never binds in normal operation, confirming it is a pathological-case guard, not a live knob. `graph_load_prune_trigger` bound 0× (isolated runs start empty). `cognition_penalty_override` **never fired** — no principles were mined in 8 isolated stages (the ~200-stage promotion finding), so cognition's one channel had zero opportunities to act: the inert layer, confirmed from a fresh angle. **Remaining Category-A rails** (resource_budget clips, formula_engine GRAD_CLIP/MAX_DEPTH/MAX_NODES, the feature-weight floor) live outside the optimizer and need the telemetry instance threaded in (a module-level singleton would reintroduce bug #30b's shared-file leak) — a follow-up.

### Part E step 2b — Outside-optimizer Category-A rails wired (2026-07-24)

Same observe-only discipline, no module singleton: `fit()` assigns `self.guard_telemetry` onto `formula_engine` / `symbolic_engine` / `classifier._guard_telemetry`, and passes it into `detect_resource_budget(..., telemetry=...)`.

| guard | where | what |
|---|---|---|
| `formula_grad_clip` | `formula_engine.compute_gradient` | max \|grad\| before vs after `GRAD_CLIP=10` |
| `formula_max_depth` / `formula_max_nodes` | mutate + crossover prune path | depth/nodes vs caps before prune |
| `feature_weight_floor` | `SymbolicFormulaAdapter._basis_weight_vector` | min raw minmax vs floored mult (`WEIGHT_FLOOR=0.25`) |
| `budget_n_particles` / `budget_pso_iters` / `budget_boosting_stages` / `budget_bagging` / `budget_tournament_k` | `resource_budget.np_clip_round` | scale suggestion vs [lo,hi] clip |

Zero behavior change — clip logic untouched.
### Part E — first Category-D evidence: two dead terms in the stress blend (2026-07-24)

The audit's whole point is making hand-tuned constants evidence-based. Category D is the weighted-sum formulas — the biggest being `uil_stress.py`'s `pressure = 0.30·recent + 0.25·deficit + 0.25·stagnation + 0.20·explanation_failure`, which drives the evolution gate, mining cadence, and causal regret multiplier. `guard_telemetry.record_signal` now logs the marginal distribution of each component every stage (via `StressEngine._last_pressure_components`), because the first question before touching any weight is "does this component even vary?"

**It does not, for two of the four.** Measured (breast_cancer, 8 stages, isolated):

| component | weight | mean | std | verdict |
|---|---|---|---|---|
| recent_stress | 0.30 | 0.058 | 0.028 | live |
| deficit | 0.25 | 0.000 | 0.000 | **DEAD** — acc >0.95 immediately, so `max(0,0.95−acc)/0.95` is constant 0 |
| stagnation | 0.25 | 0.438 | 0.286 | live |
| explanation_failure | 0.20 | 1.000 | 0.000 | **DEAD** — miner R²=0 in a short run, so it's a constant +0.20 offset |

So on this data the 4-term blend is effectively `0.30·recent + 0.25·stagnation + [fixed 0.20 baseline]`: `deficit`'s 0.25 weight contributes nothing on high-accuracy data, and `explanation_failure` just adds a constant 0.20 that shifts every stress value up — which partly explains why run stress hovers ~0.17 (the exact number behind the evolution-gate finding). This is a concrete, previously-unmeasured fact about a Category-D formula, produced automatically on the first run, no fabricated numbers. It is NOT yet licence to change the weights (one dataset, and both "dead" terms come alive on other regimes — `deficit` on a hard dataset, `explanation_failure` once mining succeeds); it is the first row of the evidence table those weights need.

**Second Category-D formula wired: the maturity blend (2026-07-24).** `hybrid_foundation.py`'s `update_maturity` decides when PSO detaches and reasoning takes over — via `0.35·evidence + 0.40·skill + 0.25·accuracy` (skill path) or a 6-term `legacy_blend` otherwise, all hand-weighted. Components now logged the same way (`_last_maturity_components` → `maturity_component.*`). Measured (breast_cancer, 10 stages, isolated), **four of six sub-scores are dead and the 0.40-weight skill path never activates**:

| component | weight | std | verdict |
|---|---|---|---|
| accuracy_score | 0.25 | 0.289 | live |
| formula_score | 0.10 | 0.050 | live |
| principle_score | **0.25** | 0.000 | **dead** — 9 bootstrap seeds / 50 = 0.18, never grows without mining |
| r2_score | **0.25** | 0.000 | **dead** — miner R² = 0 |
| episode_score | 0.10 | 0.000 | dead — no episodes retrieved |
| network_score | 0.15 | 0.000 | dead — fixed node count |
| skill_score | 0.40 | (NaN) | **never activated** — no principle outcomes, so the skill path never runs; maturity is always `legacy_blend` |

So the 6-term maturity blend collapses to ~2 varying terms (accuracy + a nearly-flat formula_score), and the entire skill-based detachment path is unreachable in isolated runs. The PSO-detachment decision is essentially accuracy-driven.

**Complementary evidence — the production regime (non-destructive read, 2026-07-24).** The obvious objection is "these terms are only dead because isolated runs accumulate nothing; they revive in production." Tested directly by reading the accumulated `results/` store (201 principles, 300 network nodes, 135 episodes, 32 formulas) and computing what each sub-score WOULD be:

| sub-score | normalization cap | production count | value |
|---|---|---|---|
| principle_score | / 50 | 201 | **1.0 (saturated)** |
| network_score | / 30 | 300 | **1.0 (saturated)** |
| episode_score | / 20 | 135 | **1.0 (saturated)** |
| formula_score | / 10 | 32 | **1.0 (saturated)** |

**So the knowledge-quantity terms are dead in BOTH regimes** — pinned at ~0 in isolated runs (nothing accumulated) and pinned at **1.0** in production (saturated past their caps). Either way std = 0, zero signal. The `/50, /30, /20, /10` caps were reasonable in early development but the system blew past all of them by 3–13×, so in any real deployment they saturate almost immediately. `min(count / cap, 1.0)` is **structurally guaranteed** to go dead in a lifelong-learning system — the one thing UIL claims to be. **Recommended fix (evidence-backed, but it changes the PSO-detachment decision so it needs a Gate-2, not a blind edit): replace fixed-cap count normalization with a RELATIVE measure** — a percentile/z-score against the running distribution, or a growth-RATE rather than an absolute count — so the term stays informative as the store grows. Same spirit as the partial-pooling recommendation in `RESEARCH_FAILURE_ANALYSIS.md` step 3.

**The two blends fail DIFFERENTLY — a nuance the evidence makes clear:**
- **maturity**: count-normalized terms **saturate** past mis-scaled fixed caps (fix = relative normalization).
- **stress**: its terms are already in [0,1] by construction (not count-normalized), so they don't saturate; they go **data-dependent constant** instead — `deficit` = 0 on easy data, `explanation_failure` = 1 without mining. Different mechanism, different fix (those weights want to be *conditional* on regime, or the signals want a floor/decay).

Both Category-D formulas now have evidence rows spanning both regimes, and the analysis names a specific, different remedy for each — which is exactly the output the "make hardcoded values dynamic with evidence" program is supposed to produce, instead of a blanket "make everything adaptive."

**Follow-ups (lower value-per-effort, deferred):** the remaining Category-A rails live outside the optimizer — `formula_engine`'s GRAD_CLIP/MAX_DEPTH/MAX_NODES (the engine is *cloned* inside the adapter, so telemetry would need threading through clones) and `editable_network`'s keep-score blend (only computed during a prune, which the graph_load telemetry just showed rarely fires in isolated runs — instrumenting it would log almost nothing). Both are wirable but neither is a clean single-instance target like the two blends above.

### Category E instrumented — are the routing cutoffs load-bearing? (2026-07-24)

`RoutingMemory` already records fingerprint → strategy → outcome, so *which* route won was never the gap. What was missing is evidence about the **cutoffs themselves** (`FORMULA_R2_MATH = 0.90`, `LINEAR_ACC_STRUCTURED = 0.92`), which cite real cross-seed numbers but from a handful of benchmarks. The open question is whether datasets ever land *near* them: if every dataset sits far from a cutoff, its exact value is irrelevant and it can stay fixed forever; if datasets cluster at the boundary, the cutoff is load-bearing and deserves to be learned. Now logged as `route_probe.formula_r2` / `route_probe.linear_acc` (distributions) plus `route_cutoff.*` (which side of the cutoff we fell on). Same machinery, no behavior change.

### Category D fix #1 implemented — maturity count normalization (2026-07-24, opt-in)

The saturation finding above is now actionable code. `KnowledgeFoundationBuilder._count_score` gains a `soft_normalization` mode (`cfg.maturity_soft_normalization`, **default OFF**) that swaps `min(1, count/cap)` for `count / (count + cap)`:

| term | production count | cap | legacy | soft |
|---|---|---|---|---|
| principles | 201 | 50 | **1.000** | 0.801 |
| network nodes | 300 | 30 | **1.000** | 0.909 |
| episodes | 135 | 20 | **1.000** | 0.871 |
| formulas | 32 | 10 | **1.000** | 0.762 |
| **std across terms** | | | **0.0000 (dead)** | **0.0576 (signal)** |

Same [0,1] range, same monotonicity, same "cap sets the scale" meaning (at `count == cap` it reads 0.5), but it approaches 1 **asymptotically and never saturates** — so the term keeps a nonzero derivative at any store size, which is the property a lifelong-learning system needs. Stateless and deterministic: no persisted running statistic, no effect on replay. At low counts it tracks the legacy form closely (isolated run: 0.153/0.231/0.000/0.286 vs 0.180/0.300/0.000/0.400), so it fixes the production regime without disturbing the small-data one.

**Opt-in and NOT yet gated.** It materially rescales maturity — at `count == cap` it reads 0.5 where the old form read 1.0 — and maturity gates PSO detachment, so it must clear a cross-seed gate before shipping on (the E6 lesson). Default is byte-identical to the historical formula. Tests: N1–N4.

### Category D fix #1 — SHIPPED default ON after Gate 2 (2026-07-24)

See Gate 2 table under "Category D Gate 2 — PASSED" below. `maturity_soft_normalization` now defaults **True**.

### Category D fix #2 — stress adaptive blend (2026-07-24, opt-in)

The stress blend's dead-term finding (deficit=0 on easy data; explanation_failure=1 when miner R²=0) now has a regime-conditional fix: `StressEngine(adaptive_blend=True)` / `cfg.stress_adaptive_blend` (default **OFF**). When on, any component that is uninformative *this stage* gets weight 0 and the survivors are renormalized to sum to 1 — no invented coefficients. Logged in `_last_pressure_components` as `w_*`. Tests in `tests/test_ebm_and_stress_adaptive.py`. Gate 2 required before default ON (stress drives evolution / mining cadence).

### Bug #30b disk leaks closed + situation_adaptive policy (2026-07-24)

Isolated `state_dir` runs no longer append into shared `results/` for fingerprints, routing memory, governance manifest, graph/formula changelogs, or live_feed (see CLAUDE bug #30b). 

`cognition/situation_policy.py` + `cfg.situation_adaptive` (default **OFF**) is the System Refinement answer to "make hardcoded switches dynamic per dataset" **without** dynamizing Category-A safety rails: when ON it may enable only the allow-listed knobs (`stress_adaptive_blend`, `maturity_soft_normalization`, `explore_architecture`), each logged with a reason in `_observability_meta["situation_policy"]`. Operator `_situation_lock` can force a flag OFF. **Do not default `situation_adaptive` ON until Category D Gate 2 passes** — that gate is `eval_crossseed --maturity-soft-normalization --stress-adaptive-blend` (logs under `results/gate2_category_d/`).

### Category D Gate 2 — PASSED; defaults flipped ON (2026-07-24)

`results/gate2_category_d/SUMMARY.md`: mean Δ **+0.0001** (5 exact ties, spambase +0.0004). No regressions. Shipped:

| flag | new default | rollback |
|---|---|---|
| `maturity_soft_normalization` | **True** | set False |
| `stress_adaptive_blend` | **True** | set False |
| `situation_adaptive` | **True** | set False (also skips `explore_architecture` auto) |

Category-A safety rails remain fixed + telemetry-only. Failed research flags (race, utility gate, adaptive_min_evidence, EBM) stay OFF the allow-list.

---

## Appendix A — Full per-constant inventory (449 entries, 33 files)

*This is the parallel session's raw sweep output, parsed verbatim from the saved chat transcript. Each entry cites file:line, the literal value, what it gates, and — where recorded — its surrounding context and its dependencies on other modules.*

### `adapters/mlp_adapter.py` — 11 values

**WEIGHT_FLOOR (module-level)** — `adapters/mlp_adapter.py:29`
- Value: `0.25`
- Gates: Minimum multiplier applied to a fully-deprioritized feature column before scaling X for the MLP fit.
- Context: "# matches the symbolic adapter's floor".
- Depends on / feeds into: _scale()'s `mult = WEIGHT_FLOOR + (1-WEIGHT_FLOOR)*safe_minmax(w)` at line 79.

**hidden_layer_sizes default** — `adapters/mlp_adapter.py:39`
- Value: `(64, 32)`
- Gates: Default MLP architecture (two hidden layers) unless the router/constructor overrides it.
- Context: constructor signature; comment elsewhere (line 182-184 in symbolic adapter's analog) treats architecture as not per-stage tunable.
- Depends on / feeds into: MLPClassifier(hidden_layer_sizes=...) in fit_weighted().

**max_iter default** — `adapters/mlp_adapter.py:40`
- Value: `400`
- Gates: Default training-iteration budget for the MLP before it's considered "fit" (non-convergence tolerated).
- Context: constructor signature.
- Depends on / feeds into: MLPClassifier(max_iter=...), also exposed as controllable param (line 160-162).

**alpha (L2 penalty) default** — `adapters/mlp_adapter.py:46`
- Value: `1e-4`
- Gates: Default L2 regularization strength for the MLP.
- Context: `self.alpha = 1e-4  # L2 penalty`.
- Depends on / feeds into: MLPClassifier(alpha=...), controllable range [1e-6,1e-1] at line 153-155.

**learning_rate_init default** — `adapters/mlp_adapter.py:47`
- Value: `1e-3`
- Gates: Default initial learning rate for the MLP optimizer.
- Context: field default.
- Depends on / feeds into: MLPClassifier(learning_rate_init=...), controllable range [1e-5,1e-1] at line 156-159.

**alpha controllable range** — `adapters/mlp_adapter.py:153-155`
- Value: `min 1e-6, max 1e-1`
- Gates: Bounds the per-stage search can move alpha within.
- Context: get_controllable_params().
- Depends on / feeds into: apply_config() clip at line 172.

**learning_rate_init controllable range** — `adapters/mlp_adapter.py:156-159`
- Value: `min 1e-5, max 1e-1`
- Gates: Bounds the per-stage search can move learning_rate_init within.
- Context: get_controllable_params().
- Depends on / feeds into: apply_config() clip at lines 174-176.

**max_iter controllable range** — `adapters/mlp_adapter.py:160-162`
- Value: `min 50, max 2000`
- Gates: Bounds the per-stage search can move max_iter within.
- Context: get_controllable_params().
- Depends on / feeds into: apply_config() clip at line 178.

**apply_config alpha clip** — `adapters/mlp_adapter.py:172`
- Value: `clip(config["alpha"], 1e-6, 1e-1)`
- Gates: Enforces the alpha bound whenever the search sets it.
- Context: apply_config().
- Depends on / feeds into: self.alpha.

**apply_config learning_rate_init clip** — `adapters/mlp_adapter.py:174-176`
- Value: `clip(config["learning_rate_init"], 1e-5, 1e-1)`
- Gates: Enforces the learning_rate_init bound whenever the search sets it.
- Context: apply_config().
- Depends on / feeds into: self.learning_rate_init.

**apply_config max_iter clip** — `adapters/mlp_adapter.py:178`
- Value: `clip(config["max_iter"], 50, 2000)`
- Gates: Enforces the max_iter bound whenever the search sets it.
- Context: apply_config().
- Depends on / feeds into: self.max_iter.


### `adapters/random_forest_adapter.py` — 10 values

**trees_per_stage default** — `adapters/random_forest_adapter.py:24`
- Value: `50`
- Gates: Default n_estimators added per fit call (and per warm-start increment).
- Context: constructor signature.
- Depends on / feeds into: fit_weighted()'s `self._total_estimators = self.trees_per_stage` and warm-start increment (lines 65-71).

**min_samples_split default** — `adapters/random_forest_adapter.py:33`
- Value: `2`
- Gates: Default sklearn min_samples_split for the forest.
- Context: "V20: additional controllable params".
- Depends on / feeds into: RandomForestClassifier(min_samples_split=...).

**max_features default** — `adapters/random_forest_adapter.py:34`
- Value: `"sqrt"`
- Gates: Default feature-subsampling strategy per split.
- Context: same block.
- Depends on / feeds into: RandomForestClassifier(max_features=...).

**criterion default** — `adapters/random_forest_adapter.py:35`
- Value: `"gini"`
- Gates: Default split-quality criterion.
- Context: same block.
- Depends on / feeds into: RandomForestClassifier(criterion=...).

**feature-weight floor for RF scaling** — `adapters/random_forest_adapter.py:61`
- Value: `clip(feature_weights, 1e-6, None)`
- Gates: Floor on feature weights before column-scaling X for RF fit — near-zero (1e-6) rather than the 0.25 floor used by the symbolic/MLP adapters, so a deprioritized feature can be almost entirely zeroed out.
- Context: "Train RandomForest with feature scaling to bias importance."
- Depends on / feeds into: X_weighted = X * weights[np.newaxis, :], directly shapes what RF ever sees of that column.

**n_estimators controllable range** — `adapters/random_forest_adapter.py:107-109`
- Value: `min 10, max 500`
- Gates: Bounds the per-stage search can move n_estimators within.
- Context: get_controllable_params().
- Depends on / feeds into: apply_config() clip at line 134.

**max_depth controllable range** — `adapters/random_forest_adapter.py:111-113`
- Value: `min 0, max 30 (0 = None/unlimited)`
- Gates: Bounds the per-stage search can move max_depth within, with 0 encoding "unlimited".
- Context: get_controllable_params().
- Depends on / feeds into: apply_config()'s `self.max_depth = val if val > 0 else None` at lines 136-137.

**min_samples_split controllable range** — `adapters/random_forest_adapter.py:115-117`
- Value: `min 2, max 50`
- Gates: Bounds the per-stage search can move min_samples_split within.
- Context: get_controllable_params().
- Depends on / feeds into: apply_config() clip at line 139.

**apply_config n_estimators clip** — `adapters/random_forest_adapter.py:134`
- Value: `clip(config["n_estimators"], 10, 500)`
- Gates: Enforces the n_estimators bound whenever the search sets it.
- Context: apply_config().
- Depends on / feeds into: self.trees_per_stage.

**apply_config min_samples_split clip** — `adapters/random_forest_adapter.py:139`
- Value: `clip(config["min_samples_split"], 2, 50)`
- Gates: Enforces the min_samples_split bound whenever the search sets it.
- Context: apply_config().
- Depends on / feeds into: self.min_samples_split.


### `adapters/symbolic_formula_adapter.py` — 45 values

**_THRESHOLD_GRID** — `adapters/symbolic_formula_adapter.py:26`
- Value: `np.linspace(0.05, 0.95, 19)`
- Gates: The 19-point grid searched for the optimal accuracy/balanced-accuracy decision threshold on model scores.
- Context: class-level constant used throughout _fit_threshold().
- Depends on / feeds into: _fit_threshold() loop at line 367.

**constructor defaults (boosting_rounds, bagging_estimators, learning_mode, learning_rate)** — `adapters/symbolic_formula_adapter.py:35-38`
- Value: `boosting_rounds=5, bagging_estimators=5, learning_mode="auto", learning_rate=0.35`
- Gates: Default boosting stage count, bagging estimator count, mode-selection policy, and per-stage gradient step size when not overridden by caller/router.
- Context: __init__ signature.
- Depends on / feeds into: self.boosting_stages/self.bagging_estimators/self.learning_mode/self.learning_rate, all mutable via apply_config().

**learning_rate clip range** — `adapters/symbolic_formula_adapter.py:51`
- Value: `clip(learning_rate, 0.05, 1.0)`
- Gates: Bounds the boosting step-size hyperparameter to [0.05, 1.0].
- Context: `self.learning_rate = float(np.clip(learning_rate, 0.05, 1.0))`.
- Depends on / feeds into: fit_boost_stage() calls throughout _fit_boosting().

**episodic confidence threshold (default)** — `adapters/symbolic_formula_adapter.py:94`
- Value: `0.5`
- Gates: Minimum retrieval_confidence required for set_episodic_hint() to accept a suggested formula.
- Context: field default, later lowered to 0.35 for champion-sourced hints (line 133).
- Depends on / feeds into: set_episodic_hint() comparison at line 120.

**champion-hint confidence override** — `adapters/symbolic_formula_adapter.py:133`
- Value: `0.35`
- Gates: Lowers the confidence bar specifically for a champion-sourced formula suggestion (more trusted than a generic episodic hint).
- Context: set_exploration_hint(), "champion" branch.
- Depends on / feeds into: self._episodic_confidence_threshold, reused by _pick_formula() gate.

**explored-low exploration penalty** — `adapters/symbolic_formula_adapter.py:140`
- Value: `-0.35`
- Gates: Score penalty applied to a formula the library has already flagged as low-accuracy-explored for this dataset.
- Context: _exploration_penalty(), first branch.
- Depends on / feeds into: added to tournament score in _select_formula()._score_entry (line 1838).

**avoid_formulas penalty** — `adapters/symbolic_formula_adapter.py:143`
- Value: `-0.25`
- Gates: Score penalty when a formula is on the abstract-memory "avoid_formulas" list.
- Context: _exploration_penalty(), second branch.
- Depends on / feeds into: same tournament-score consumer.

**known_good_formulas bonus** — `adapters/symbolic_formula_adapter.py:146`
- Value: `0.12`
- Gates: Score bonus when a formula is on the abstract-memory "known_good_formulas" list.
- Context: _exploration_penalty(), third branch.
- Depends on / feeds into: same tournament-score consumer.

**explored-formula skip evidence threshold** — `adapters/symbolic_formula_adapter.py:152`
- Value: `4 (entry.get("evidence", 0) < 4)`
- Gates: A formula needs ≥4 evidence points before it can be skipped as "explored low" for a dataset.
- Context: _should_skip_explored_formula().
- Depends on / feeds into: filters the candidate pool in _select_formula() (line 1804).

**graph-hint bonus coefficient** — `adapters/symbolic_formula_adapter.py:170`
- Value: `0.1`
- Gates: Weight of feature-emphasis alignment (from KG traversal) in a formula's tournament bonus.
- Context: _graph_formula_bonus().
- Depends on / feeds into: added to score in _score_entry (line 1837) / score_formula (line 1910).

**graph-hint tree-size bonus** — `adapters/symbolic_formula_adapter.py:171`
- Value: `0.03 * min(1.0, node_count / 24.0)`
- Gates: Small additional bonus for larger formula trees (up to node_count=24), on top of the emphasis-alignment bonus.
- Context: same _graph_formula_bonus() function.
- Depends on / feeds into: same consumers as above.

**_prepare_X weight floor** — `adapters/symbolic_formula_adapter.py:238`
- Value: `0.25 + 0.75 * w`
- Gates: Minimum multiplier (0.25) applied to a fully-deprioritized feature column on tree/gradient paths; matches WEIGHT_FLOOR elsewhere.
- Context: `return X * (0.25 + 0.75 * w)`.
- Depends on / feeds into: used by _raw_from_tree/_scores_from_tree, i.e. the single/boosting/bagging model paths.

**IMBALANCE_CUTOFF** — `adapters/symbolic_formula_adapter.py:334`
- Value: `0.75`
- Gates: Base-rate threshold above which threshold-fitting switches objective from plain accuracy to balanced accuracy.
- Context: long comment citing bank_marketing 88/12 example where plain-accuracy fitting collapsed to all-majority prediction.
- Depends on / feeds into: _fit_threshold()'s `imbalanced = base_rate >= self.IMBALANCE_CUTOFF`.

**FEATURE_SELECT_CAP** — `adapters/symbolic_formula_adapter.py:384`
- Value: `64`
- Gates: Feature-count threshold above which top-K ANOVA-F feature preselection kicks in before fitting.
- Context: comment on MNIST 784-dim collapse to majority class without this cap.
- Depends on / feeds into: fit_weighted()'s `if X.shape[1] > self.FEATURE_SELECT_CAP` gate (line 398).

**feature-selection score blend** — `adapters/symbolic_formula_adapter.py:476`
- Value: `0.7 * s + 0.3 * w`
- Gates: Blends ANOVA F-statistic (70%) with normalized PSO feature weights (30%) for the top-K feature selection ranking.
- Context: _select_features(), "so the search's own relevance signal is respected".
- Depends on / feeds into: idx = argsort(score) selecting FEATURE_SELECT_CAP columns.

**STUMPS_PER_FEATURE** — `adapters/symbolic_formula_adapter.py:616`
- Value: `3`
- Gates: Number of quantile-based stump thresholds generated per feature in the sparse_linear stump basis.
- Context: part of the "Basis families" comment block with measured spambase/covertype accuracy deltas.
- Depends on / feeds into: `qs = linspace(0.2, 0.8, STUMPS_PER_FEATURE)` at line 754.

**MAX_SYMBOLIC_BASIS** — `adapters/symbolic_formula_adapter.py:618`
- Value: `40`
- Gates: Caps how many active library formulas are included as basis terms in the sparse_linear model.
- Context: same basis-family comment block.
- Depends on / feeds into: `actives = ...get_active_formulas()[:self.MAX_SYMBOLIC_BASIS]` in _build_basis_spec (line 743).

**WEIGHT_FLOOR** — `adapters/symbolic_formula_adapter.py:623`
- Value: `0.25`
- Gates: Lowest effective multiplier a down-weighted basis column can be pushed to inside the L1 standardization channel.
- Context: "a down-weighted feature carries 4x the effective L1 penalty of a fully-weighted one".
- Depends on / feeds into: _basis_weight_vector()'s `mult = WEIGHT_FLOOR + (1-WEIGHT_FLOOR)*safe_minmax(w)` at line 850.

**MAX_RULES** — `adapters/symbolic_formula_adapter.py:627`
- Value: `40`
- Gates: Caps the number of RuleFit-style conjunctive rules mined and added as basis terms.
- Context: "cap so the L1 problem stays bounded/readable".
- Depends on / feeds into: _extract_rules(max_rules) and _build_basis_spec's use_rule_basis branch (line 759).

**stump quantile positions** — `adapters/symbolic_formula_adapter.py:754`
- Value: `linspace(0.2, 0.8, STUMPS_PER_FEATURE)`
- Gates: Which quantiles of each feature's distribution become stump split thresholds.
- Context: _build_basis_spec(), use_stump_basis branch.
- Depends on / feeds into: per-feature stump basis entries.

**sparse_linear L1 solve max_iter** — `adapters/symbolic_formula_adapter.py:914`
- Value: `3000`
- Gates: Iteration cap for the liblinear L1 logistic solve underlying sparse_linear mode.
- Context: within the long "Bug #31" comment about liblinear's random_state sensitivity.
- Depends on / feeds into: LogisticRegression(...) call in _fit_sparse_linear().

**fit_boost_stage inner refit rounds** — `adapters/symbolic_formula_adapter.py:1075`
- Value: `3`
- Gates: Literal "3" passed into fit_boost_stage() as an inner constant-refit parameter (e.g. Newton line-search rounds) for each boosting stage.
- Context: comment explains fit_boost_stage replaced a legacy constant-step formula that collapsed predictions to majority class.
- Depends on / feeds into: symbolic_gradient.fit_boost_stage(), which is outside the audited file set.

**gradient-variant weight clip** — `adapters/symbolic_formula_adapter.py:1106`
- Value: `clip(w, 1e-6, 0.25)`
- Gates: Bounds the logistic-variance weighting term used to compute the "gradient" boosting target.
- Context: _boosting_target(), gradient branch.
- Depends on / feeds into: `residual * clip(w, ...)`.

**newton-variant weight clip** — `adapters/symbolic_formula_adapter.py:1109`
- Value: `clip(w, 1e-3, 0.25)`
- Gates: Bounds the denominator weight used for the "newton" boosting-target variant, avoiding division blow-up.
- Context: _boosting_target(), newton branch.
- Depends on / feeds into: `residual / clip(w, ...)`.

**hybrid branch budget split** — `adapters/symbolic_formula_adapter.py:1114-1116`
- Value: `boost=max(2,boosting_stages//3), bag=max(2,bagging_estimators//3), mini_boost=max(2,min(4,boosting_stages//5))`
- Gates: Per-branch stage/estimator budgets when the hybrid ensemble splits the global boosting/bagging budget across branches.
- Context: "Split global budget across hybrid branches."
- Depends on / feeds into: _fit_hybrid_ensemble()'s per-branch fit calls (boosting/gradient/bagging/boosted_bag).

**hybrid branch-count clamp** — `adapters/symbolic_formula_adapter.py:1130`
- Value: `max(3, min(len(HYBRID_BRANCHES), hybrid_branch_count))`
- Gates: Number of hybrid branches actually trained is clamped to [3, len(HYBRID_BRANCHES)=5].
- Context: _fit_hybrid_ensemble() setup.
- Depends on / feeds into: branch_names list, drives the ThreadPoolExecutor / serial branch loop.

**hybrid blend random-search iterations** — `adapters/symbolic_formula_adapter.py:1339`
- Value: `120 (for _ in range(120))`
- Gates: Number of Dirichlet-random weight draws tried when optimizing the branch-blend weights.
- Context: _optimize_hybrid_blend().
- Depends on / feeds into: best_w / best_acc tracking, then refined by coordinate search.

**hybrid blend coordinate-refinement iterations** — `adapters/symbolic_formula_adapter.py:1348`
- Value: `len(names) * 8`
- Gates: Number of local coordinate-descent refinement steps applied after the random search.
- Context: "Coordinate refinement from best draw".
- Depends on / feeds into: best_w updates.

**hybrid blend perturbation delta range** — `adapters/symbolic_formula_adapter.py:1351`
- Value: `uniform(0.02, 0.15)`
- Gates: Size of the weight-mass shift between two branches tried during coordinate refinement.
- Context: same coordinate-refinement loop.
- Depends on / feeds into: w[i] -= delta; w[j] += delta.

**eval_holdout_split tiny-n threshold and split fraction** — `adapters/symbolic_formula_adapter.py:1777-1778`
- Value: `n < 12 → split = int(0.7 * n)`
- Gates: For very small datasets, a manual 70/30 split is used instead of sklearn's train_test_split.
- Context: _eval_holdout_split().
- Depends on / feeds into: X_tr/y_tr/X_val/y_val used by tournament and hybrid-blend evaluation.

**stratify eligibility threshold** — `adapters/symbolic_formula_adapter.py:1783`
- Value: `n >= 20 (and binary classes)`
- Gates: Whether train_test_split stratifies by label; below n=20 stratification is skipped.
- Context: same _eval_holdout_split().
- Depends on / feeds into: sklearn train_test_split stratify param.

**holdout test_size** — `adapters/symbolic_formula_adapter.py:1786`
- Value: `0.25`
- Gates: Fraction of data held out for validation in tournament formula selection and hybrid-branch blending.
- Context: same function.
- Depends on / feeds into: X_val/y_val sizes.

**tournament progress normalization** — `adapters/symbolic_formula_adapter.py:1810`
- Value: `sum(evidence) / max(1, len(pool) * 10)`
- Gates: Normalizes cumulative formula "evidence" into a [0,1] progress signal, treating 10 evidence points per pool formula as "full progress".
- Context: _select_formula(), used to call select_tournament_candidates(progress=...).
- Depends on / feeds into: FormulaLibrary.select_tournament_candidates (outside audited files).

**CONFIDENCE_BAND** — `adapters/symbolic_formula_adapter.py:1892`
- Value: `0.15`
- Gates: How far below the baseline formula's confidence a low-evidence formula may sit and still be included in the metadata-selection candidate pool.
- Context: _select_formula_metadata().
- Depends on / feeds into: `filtered = [...confidence >= baseline["confidence"] - CONFIDENCE_BAND]`.

**metadata fitness-score weights** — `adapters/symbolic_formula_adapter.py:1900-1908`
- Value: `confidence + 0.15*min(1,evidence/10) + 0.35*best_accuracy + 0.05 (method_ prefix bonus) + 0.25*usage-alignment`
- Gates: Composite score used to pick a formula when there's not enough data to run the accuracy tournament.
- Context: score_formula() inside _select_formula_metadata().
- Depends on / feeds into: max(filtered, key=score_formula) selection.

**mutation_rounds_per_fit controllable range** — `adapters/symbolic_formula_adapter.py:1933`
- Value: `min 0, max 3`
- Gates: Bounds exposed to the per-stage search for the mutation-rounds hyperparameter.
- Context: get_controllable_params().
- Depends on / feeds into: apply_config()'s clip at line 2035.

**boosting_stages controllable range** — `adapters/symbolic_formula_adapter.py:1937`
- Value: `min 1, max 50`
- Gates: Bounds exposed to the per-stage search for boosting_stages.
- Context: get_controllable_params().
- Depends on / feeds into: principle_guided_search._propose_config() perturbs within these bounds.

**bagging_estimators controllable range** — `adapters/symbolic_formula_adapter.py:1941`
- Value: `min 1, max 20`
- Gates: Bounds exposed to the per-stage search for bagging_estimators.
- Context: get_controllable_params().
- Depends on / feeds into: same as above.

**architecture-race sparse_l1_c values** — `adapters/symbolic_formula_adapter.py:1988`
- Value: `[0.05, 0.1, 0.5, 1.0]`
- Gates: The discrete L1-strength candidates the ArchitectureRace evaluates as one axis.
- Context: get_architecture_axes(), notes E5's finding that rule terms get starved under strong L1.
- Depends on / feeds into: ArchitectureRace.propose()'s "architecture arms" loop (architecture_race.py:404-420).

**architecture-race boosting_stages values** — `adapters/symbolic_formula_adapter.py:2018`
- Value: `[5, 10, 25]`
- Gates: The discrete boosting-stage candidates raced when learning_mode is boosting/multi/auto.
- Context: get_architecture_axes().
- Depends on / feeds into: same ArchitectureRace consumer.

**apply_config mutation_rounds_per_fit clip** — `adapters/symbolic_formula_adapter.py:2035`
- Value: `clip(value, 0, 3)`
- Gates: Enforces the same [0,3] bound at config-application time (belt-and-suspenders with get_controllable_params).
- Context: apply_config().
- Depends on / feeds into: self.mutation_rounds_per_fit.

**tournament_k apply-time floor** — `adapters/symbolic_formula_adapter.py:2056`
- Value: `max(3, int(config["tournament_k"]))`
- Gates: Enforces a hard floor of 3 candidates in any tournament regardless of what the search proposes (tournament_k=8 is the known default, this is a new floor-usage site).
- Context: apply_config().
- Depends on / feeds into: self.tournament_k, consumed in _select_formula() line 1809.

**hybrid_branch_count apply-time clamp** — `adapters/symbolic_formula_adapter.py:2058`
- Value: `max(3, min(5, int(config["hybrid_branch_count"])))`
- Gates: Clamps hybrid_branch_count to [3,5] whenever the search/config sets it directly (hybrid_branch_count=5 is the known default; this is a new clamp-usage site).
- Context: apply_config().
- Depends on / feeds into: self.hybrid_branch_count, consumed by _fit_hybrid_ensemble() line 1130.

**multi_boost_variants derived clamp** — `adapters/symbolic_formula_adapter.py:2060`
- Value: `max(3, min(5, int(config["multi_boost_variants"]) + 2))`
- Gates: Alternate path to set hybrid_branch_count via a "multi_boost_variants" key, offset by +2 then clamped to [3,5].
- Context: `elif "multi_boost_variants" in config:` branch, apply_config().
- Depends on / feeds into: self.hybrid_branch_count.

**sparse_l1_c apply-time clip range** — `adapters/symbolic_formula_adapter.py:2070`
- Value: `clip(value, 0.001, 10.0)`
- Gates: Global bounds on sparse_l1_c whenever set via config (wider than the race's discrete [0.05,1.0] menu).
- Context: apply_config().
- Depends on / feeds into: self.sparse_l1_c, consumed by _fit_sparse_linear()'s LogisticRegression(C=...).


### `cognition/abstraction_safeguards.py` — 6 values

**MIN_SUPPORT** — `cognition/abstraction_safeguards.py:28`
- Value: `3`
- Gates: Minimum number of children a cluster must have before a meta-principle can be promoted.
- Context: Listed under "# Gate thresholds" as the first of five promotion gates.
- Depends on / feeds into: Read in check_promotion() gate "min_support"; compared to meta_principle.support.

**MIN_CONFIDENCE** — `cognition/abstraction_safeguards.py:29`
- Value: `0.55`
- Gates: Minimum weighted-average confidence of a meta-principle's children required for promotion.
- Context: Part of the same "# Gate thresholds" block as MIN_SUPPORT.
- Depends on / feeds into: Read in check_promotion() gate "min_confidence" against meta_principle.confidence.

**MAX_CONTRADICTION_RATE** — `cognition/abstraction_safeguards.py:30`
- Value: `0.2`
- Gates: Maximum fraction of child-pairs allowed to contradict each other for a cluster to remain coherent.
- Context: Comment "At most 20% of pairs contradict".
- Depends on / feeds into: Compared against _compute_contradiction_rate() output in gate "max_contradictions".

**MIN_EVIDENCE_COUNT** — `cognition/abstraction_safeguards.py:31`
- Value: `10`
- Gates: Minimum total times_applied across children before evidence is considered sufficient.
- Context: Comment "Total times_applied across children".
- Depends on / feeds into: Gate "min_evidence"; also gates whether MIN_HELP_RATIO is enforced or relaxed (line 98).

**MIN_HELP_RATIO** — `cognition/abstraction_safeguards.py:32`
- Value: `0.5`
- Gates: Minimum fraction of applications that must have "helped" for promotion, once evidence is sufficient.
- Context: Comment "At least 50% of applications helped".
- Depends on / feeds into: Gate "min_help_ratio"; bypassed (auto-pass) when total_applied < MIN_EVIDENCE_COUNT.

**0.6 / 0.4 (monitor_meta_principle status bands)** — `cognition/abstraction_safeguards.py:166,168`
- Value: `0.6 (healthy cutoff), 0.4 (degrading cutoff)`
- Gates: Whether a promoted meta-principle is judged "healthy", "degrading", or "revoke" from recent application outcomes.
- Context: recent_ratio >= 0.6 -> healthy; elif >= 0.4 -> degrading; else -> revoke.
- Depends on / feeds into: Consumes recent_applications helped-ratio; feeds continuous-monitoring "status" used to decide revocation.


### `cognition/causal_intervention.py` — 11 values

**hypothesis dedupe tolerance** — `cognition/causal_intervention.py:191`
- Value: `0.15`
- Gates: Whether a new candidate hypothesis is "too similar" to an existing one and therefore skipped.
- Context: form_hypothesis(): `abs(h.threshold - threshold) / max(abs(threshold), 1e-6) < 0.15`.
- Depends on / feeds into: Prevents duplicate Hypothesis objects for near-identical thresholds.

**intervention scale factors (0.8/1.2/0.5/2.0)** — `cognition/causal_intervention.py:240-246`
- Value: `treatment_value=threshold*0.8 or *1.2; treatment_scale=0.5 or 2.0; control_value=threshold*1.2 or *0.8`
- Gates: How far the treatment arm is pushed from the threshold when designing an A/B intervention.
- Context: "20% below threshold"; scale-mode halves/doubles the controllable factor for derived (product) variables.
- Depends on / feeds into: design_intervention(); consumed by whatever runs the control/treatment trial.

**evidence-trial effect thresholds** — `cognition/causal_intervention.py:330-334`
- Value: `regret_delta>0.05 and accuracy_delta>0.01 (strong); regret_delta<-0.02 (refuted signal)`
- Gates: Which of the last-3 trials count as "strong" vs "refuted" evidence for causal status.
- Context: "CONFIRMED_CAUSAL now requires a majority of the last 3 trials to show a real effect".
- Depends on / feeds into: Computes _strong_count / _refuted_count, used in the status decision tree below.

**CONFIRMED_CAUSAL majority rule** — `cognition/causal_intervention.py:337,340`
- Value: `len(evidence_trials)>=3 and _strong_count>=2; causal_strength = min(0.95, 0.5 + avg_regret_delta*2)`
- Gates: When a hypothesis is promoted to STATUS_CONFIRMED_CAUSAL, and its resulting causal_strength score.
- Context: "A single lucky noisy trial can't declare causal status."
- Depends on / feeds into: hypothesis.status/causal_strength feed get_confidence_adjustment().

**single-trial CORRELATIONAL thresholds** — `cognition/causal_intervention.py:346,348,355,357`
- Value: `regret_delta>0.05 & accuracy_delta>0.01 -> causal_strength=min(0.5,0.3+regret_delta); regret_delta>0.02 -> weak, causal_strength=0.3+regret_delta`
- Gates: Correlational-vs-weak-correlational classification for a single trial pending replication.
- Context: evaluate_intervention().
- Depends on / feeds into: Sets hypothesis.status/causal_strength/conclusion text.

**REFUTED thresholds** — `cognition/causal_intervention.py:363`
- Value: `_refuted_count>=2 or regret_delta<-0.02`
- Gates: When a hypothesis is marked STATUS_REFUTED (treatment made things worse).
- Context: evaluate_intervention(), causal_strength forced to 0.0.
- Depends on / feeds into: Drives op="remove" in _log_cell (line 382).

**inconclusive causal_strength** — `cognition/causal_intervention.py:373`
- Value: `0.1`
- Gates: Default causal_strength assigned when no branch above matched ("no significant effect").
- Context: else branch of evaluate_intervention().
- Depends on / feeds into: Read later by get_confidence_adjustment() consumers.

**confidence-adjustment multipliers** — `cognition/causal_intervention.py:411,413,415`
- Value: `1.0 + causal_strength*0.5 (up to 1.475); 0.8 (correlational); 0.3 (refuted)`
- Gates: How much a principle's confidence is boosted/reduced in the Library based on causal-test outcome.
- Context: "Confirmed causal → boost confidence significantly … Refuted → sharply reduce confidence".
- Depends on / feeds into: get_confidence_adjustment(); return value multiplies principle confidence elsewhere (uil_library.py).

**hypothesis-worth-testing confidence band** — `cognition/causal_intervention.py:431`
- Value: `0.3 <= confidence <= 0.8`
- Gates: Which DT-mined rules are even converted into hypotheses (too-low = noise, too-high = already proven).
- Context: form_hypotheses_from_rules().
- Depends on / feeds into: Gates form_hypothesis() calls.

**goal-prior cutoff** — `cognition/causal_intervention.py:473-477`
- Value: `0.2`
- Gates: Minimum goal.hypothesis_score() for a hypothesis to remain testable/ranked when a GoalCell is active.
- Context: "Cutoff at 0.2 — a goal can push a hypothesis into the 'not now' bin (score ~0.15) but not deny that it exists."
- Depends on / feeds into: Reads GoalCell.hypothesis_score(); feeds get_testable_hypotheses() ranking.

**EIG scoring weights** — `cognition/causal_intervention.py:556-561`
- Value: `uncertainty*0.5 + regret_signal*0.35 + status_bonus(0.3); regret_signal=min(1.0, predicted_regret*2.0)`
- Gates: Which pending hypotheses are prioritized for the next intervention (expected-information-gain acquisition).
- Context: "Greedy expected-information-gain acquisition (v22 §5.1)".
- Depends on / feeds into: choose_intervention_eig(); consumes h.evidence_trials count and h.predicted_regret.


### `cognition/change_log.py` — 2 values

**_MAX_GRAPH_EVENTS** — `cognition/change_log.py:22`
- Value: `10_000`
- Gates: Cap on retained graph-changelog events (oldest truncated beyond this).
- Context: Module-level constant, used by GraphChangeLog.record().
- Depends on / feeds into: `if len(events) > _MAX_GRAPH_EVENTS: events[-_MAX_GRAPH_EVENTS:]`. Note: version_store.py's docstring explicitly calls out this changelog as "capped … cannot reconstruct past state" as the reason VersionStore was built.

**_MAX_FORMULA_EVENTS** — `cognition/change_log.py:23`
- Value: `10_000`
- Gates: Cap on retained formula-changelog events.
- Context: Same pattern as _MAX_GRAPH_EVENTS, used by FormulaChangeLog.record().
- Depends on / feeds into: Same truncation pattern.


### `cognition/condition_match.py` — 6 values

**DEFAULT_SIMILARITY_TOLERANCE** — `cognition/condition_match.py:36`
- Value: `0.10`
- Gates: Relative-threshold tolerance for judging two DT-rule conditions "the same rule" — directly gates node-merge behavior in editable_network.add_node().
- Context: "Relative tolerance used by the library/network 'similar' check."
- Depends on / feeds into: thresholds_match() / condition_sets_similar(), called from editable_network._find_similar_node() and (per the module docstring) uil_library.py and journal_forgetting.py.

**"max" mode zero-division floor** — `cognition/condition_match.py:66`
- Value: `1e-6`
- Gates: Floor on the denominator when comparing two thresholds symmetrically (ContradictionDetector's overlap check), preventing div-by-zero when both thresholds are ~0.
- Context: `denominator = max(abs(a), abs(b), 1e-6); return abs(a - b) / denominator < tolerance`.
- Depends on / feeds into: Used by thresholds_match(mode="max"), called from journal_forgetting.ContradictionDetector._conditions_overlap.

**"reference" mode near-zero cutoff** — `cognition/condition_match.py:72,74`
- Value: `1e-6`
- Gates: Threshold below which the reference value `a` is treated as "zero" (no scale to be relative to), switching to an absolute near-zero comparison for `b`.
- Context: "when the reference is ~0 there is no scale to be relative to, so only a `b` that is also ~0 counts as the same threshold" → `if abs(a) > 1e-6: return abs(a-b)/abs(a) <= tolerance; return abs(b) <= 1e-6`.
- Depends on / feeds into: Used by thresholds_match(mode="reference"), called from uil_library._conditions_similar (via condition_sets_similar) with DEFAULT_SIMILARITY_TOLERANCE=0.10 — this is a NEW usage site for the tolerance constant, gating principle merge/dedup.

**PENALTY_FEATURES** — `cognition/condition_match.py:79`
- Value: `frozenset({"penalty", "action.penalty"})`
- Gates: Exact feature names recognized as "the penalty knob"; only conditions naming these can be inverted into `max_penalty`/`min_penalty`/`penalty_avoid` overrides.
- Context: "The feature names that ARE the penalty knob (exact match, never a substring test -- see sanitize_overrides)."
- Depends on / feeds into: Imported and re-exposed as `UIL_Library.PENALTY_FEATURES` (uil_library.py:398, used in `_generate_override()` line 474); also referenced by `sanitize_overrides()`'s `has_penalty` check (line 106) here in condition_match.py.

**_PENALTY_OVERRIDE_KEYS** — `cognition/condition_match.py:81`
- Value: `("max_penalty", "min_penalty", "penalty_avoid")`
- Gates: Which override keys are subject to the "must have a penalty condition" provenance check in sanitize_overrides().
- Context: Used at line 111 `if key in _PENALTY_OVERRIDE_KEYS: if not has_penalty: drop.append(key)`.
- Depends on / feeds into: Drives which overrides get stripped from principles on load (uil_library._load → sanitize_overrides) and on generation.

**_CONFIG_OVERRIDE_PREFIXES** — `cognition/condition_match.py:82`
- Value: `("config_max.", "config_min.", "config_avoid.")`
- Gates: Which override key prefixes are subject to the "must have a matching config.<name> condition" provenance check.
- Context: Used at line 114-115 `elif key.startswith(_CONFIG_OVERRIDE_PREFIXES): if key.split(".",1)[1] not in configs: drop.append(key)`.
- Depends on / feeds into: Same sanitize_overrides() repair path as above.


### `cognition/domain_router.py` — 13 values

**TINY_N** — `cognition/domain_router.py:32`
- Value: `50`
- Gates: Sample-count cutoff below which a dataset is flagged "tiny" and routed to the few-shot/D3_tiny domain.
- Context: Module-level constant used by fingerprint() (fp["tiny"] = n < TINY_N) and matcher m_tiny (priority 100, highest).
- Depends on / feeds into: Read by fingerprint() and _default_registry().m_tiny; drives RoutedController.decide().

**FORMULA_R2_MATH** — `cognition/domain_router.py:33`
- Value: `0.90`
- Gates: Minimum symbolic-probe R² for a continuous-target dataset to be classified D2_math (a "law fits").
- Context: "continuous + probe above this => a law fits => D2".
- Depends on / feeds into: Compared against fp["formula_probe_r2"] in matcher m_math.

**LINEAR_ACC_STRUCTURED** — `cognition/domain_router.py:34`
- Value: `0.92`
- Gates: Minimum linear-probe accuracy for a classification dataset to be classified D4_structured (readable sparse-linear model).
- Context: "classification + linear probe above => D4".
- Depends on / feeds into: Used directly in m_structured and as the upper bound of m_moderate_linear's 0.75–0.92 band and m_hard_boundary's <0.75 test (lines 157,169 reuse it).

**n >= 40 / n >= 30 probe-eligibility gates** — `cognition/domain_router.py:58,65`
- Value: `40 (continuous probe), 30 (classification probe)`
- Gates: Minimum sample count required before the router bothers running the (expensive) symbolic/linear probes.
- Context: `if tt == "continuous" and n >= 40:` / `if tt in ("binary","multiclass") and n >= 30:`.
- Depends on / feeds into: Skips probe computation entirely below these sizes, leaving formula_probe_r2/linear_probe_acc as None.

**test_size / n_terms probe params** — `cognition/domain_router.py:61-62`
- Value: `test_size=0.3, n_terms=6`
- Gates: Train/test split fraction and formula complexity for the "cheap symbolic fit" probe used to decide routing.
- Context: Inside fingerprint(), guarded by `n >= 40`.
- Depends on / feeds into: Produces fp["formula_probe_r2"], gated further by FORMULA_R2_MATH.

**LogisticRegression max_iter / cv** — `cognition/domain_router.py:70-71`
- Value: `max_iter=500, cv=3`
- Gates: Convergence/robustness of the linear-probe accuracy used by the D4_structured gate.
- Context: Inside the `try` block computing `acc` via cross_val_score.
- Depends on / feeds into: Produces fp["linear_probe_acc"], consumed by LINEAR_ACC_STRUCTURED-based matchers.

**moderate-linear band 0.75–LINEAR_ACC_STRUCTURED** — `cognition/domain_router.py:152-158`
- Value: `0.75 (lower bound)`
- Gates: Whether a classification dataset is routed to D4_moderate_linear (near-linear, sparse_linear strategy).
- Context: `0.75 <= lin < LINEAR_ACC_STRUCTURED`.
- Depends on / feeds into: Domain priority 48, just below D4_structured's 50.

**hard-boundary cutoff 0.75** — `cognition/domain_router.py:163-169`
- Value: `0.75`
- Gates: Whether a classification dataset with a poor linear probe is routed to D1_hard_boundary (GBM).
- Context: `lin < 0.75`; confidence score computed as `0.75 - lin`.
- Depends on / feeds into: Domain priority 45.

**m_sequential confidence 0.85** — `cognition/domain_router.py:172`
- Value: `0.85`
- Gates: Fixed match-confidence assigned to the D5_sequential domain whenever fp["sequential"] is true.
- Context: `return bool(fp.get("sequential")), 0.85`.
- Depends on / feeds into: fp["sequential"] itself is computed in fingerprint.py (lag1/Ljung-Box gates below).

**Domain priority values** — `cognition/domain_router.py:175-192`
- Value: `D3_tiny=100, D5_sequential=90, D2_math=50, D4_structured=50, D4_moderate_linear=48, D1_hard_boundary=45, D1_noisy_regression=10, D1_noisy_tabular=10`
- Gates: Tie-breaking order when multiple domain matchers fire for the same fingerprint.
- Context: Comment explains D4_structured's sparse_linear choice with cross-seed accuracy figures (breast_cancer 0.974, digits 0.948, etc.).
- Depends on / feeds into: DomainRegistry.classify() picks `max((priority, confidence))`.

**RoutingMemory defaults (top_k, min_support, margin)** — `cognition/domain_router.py:243-246`
- Value: `top_k=6, min_support=2, margin=0.03`
- Gates: How many neighbor episodes are blended, minimum support before a recommendation is trusted, and the confidence margin required to override the heuristic strategy.
- Context: RoutingMemory.__init__ defaults.
- Depends on / feeds into: Used in recommend() (lines 284,294,296) which feeds RoutedController.decide().

**novelty_threshold** — `cognition/domain_router.py:331`
- Value: `0.9`
- Gates: Distance beyond which a fingerprint is judged "novel" (unlike anything routed before), and the radius used when minting a new discovered domain.
- Context: RoutedController.__init__ default; also default radius in discover_domain().
- Depends on / feeds into: Compared to RoutingMemory.novelty(fp) in decide(); reused as `radius` at line 385.

**discover_domain priority** — `cognition/domain_router.py:394`
- Value: `70`
- Gates: Priority assigned to a newly discovered domain — "above generic fallbacks, below hard rules like tiny".
- Context: Comment right above the registration call.
- Depends on / feeds into: Competes against the static priorities listed above in DomainRegistry.classify().


### `cognition/editable_network.py` — 26 values

**LOOP_DECAY_FACTOR** — `cognition/editable_network.py:54`
- Value: `0.5`
- Gates: Per-revisit weight decay during graph traversal (loop prevention): ω_new = ω_old * 0.5.
- Context: Module docstring: "Graph is constrained to prevent infinite loops."
- Depends on / feeds into: traverse() and traverse_with_attention() both apply this on revisit.

**LOOP_CUTOFF** — `cognition/editable_network.py:55`
- Value: `0.01`
- Gates: Weight floor below which traversal stops following a path.
- Context: Same docstring: "ω_new = ω_old * 0.5 per revisit, cutoff at 0.01".
- Depends on / feeds into: `if current_weight < LOOP_CUTOFF: continue` in both traversal methods; also used in the enqueue-guard anticipation check (line 1265).

**DEFAULT_EDGE_ARCHIVE_CUT** — `cognition/editable_network.py:59`
- Value: `0.22`
- Gates: Base edge-keep-score cutoff below which an edge becomes an archive candidate.
- Context: "Metric-based archive: scores below these thresholds are archive candidates. Thresholds are adjusted by system_stress at prune time (high stress → keep more)."
- Depends on / feeds into: _archive_thresholds() scales this by stress; consumed by prune_working_graph()/_edge_should_archive().

**DEFAULT_NODE_ARCHIVE_CUT** — `cognition/editable_network.py:60`
- Value: `0.24`
- Gates: Base node-keep-score cutoff below which a node becomes an archive candidate.
- Context: Same comment block as DEFAULT_EDGE_ARCHIVE_CUT.
- Depends on / feeds into: _archive_thresholds(); _node_should_archive().

**RECENCY_HALF_LIFE_DAYS** — `cognition/editable_network.py:61`
- Value: `14.0`
- Gates: Exponential-decay half-life for how "recently used" a node/edge is judged.
- Context: `_recency_score()`: "1.0 when just used; decays exponentially with idle time."
- Depends on / feeds into: Used in node_keep_score() and edge_keep_score() recency terms.

**dataset-relevance score bands** — `cognition/editable_network.py:132-140`
- Value: `0.55 (no dataset given), 0.35 (node has no domain tags), 1.0 (exact match), 0.12 (mismatch)`
- Gates: How strongly domain-mismatch penalizes a node/edge's keep score.
- Context: `_dataset_relevance_score()`.
- Depends on / feeds into: node_keep_score() (0.12 weight) and edge_keep_score() (0.08 weight).

**node_keep_score weights** — `cognition/editable_network.py:440-454`
- Value: `0.28 confidence + 0.12*min(1,evidence/25) + 0.18*min(1,activation/12) + 0.15*recency + 0.12*relevance + 0.10*regret-term (or 0.05 fallback) + 0.05*tier_boost`
- Gates: The composite "worth keeping" score for every KG node, driving working-memory pruning.
- Context: "Composite keep score in [0, 1]. Higher = more worth keeping in working memory."
- Depends on / feeds into: tier_boost = max(0, (6-min(tier,5))/5); pred_regret term = max(0, 1-min(1,regret*4)); feeds _node_should_archive() and prune_working_graph().

**edge_keep_score weights** — `cognition/editable_network.py:468-483`
- Value: `0.30 weight + 0.26 help_ratio + 0.12*min(1,evidence/15) + 0.10*min(1,traversed/25) + 0.10 recency + 0.08 relevance + 0.04+0.04 endpoint confidences, penalties -0.06 / -0.12`
- Gates: The composite "worth keeping" score for every edge.
- Context: "Composite keep score in [0, 1] using edge metrics + endpoint node quality."
- Depends on / feeds into: Penalty -0.06 if never traversed & evidence<=1; -0.12 if traversed>=5 & help_ratio<0.15. Feeds _edge_should_archive().

**stress-scaled archive thresholds** — `cognition/editable_network.py:492-493`
- Value: `edge_cut = DEFAULT_EDGE_ARCHIVE_CUT*(0.75+0.25*stress); node_cut analogous`
- Gates: How much system_stress relaxes the archive cutoffs at prune time (high stress -> keep more).
- Context: `_archive_thresholds()`.
- Depends on / feeds into: Reads system_stress (from uil_stress.py); output consumed by prune_working_graph().

**_edge_should_archive rules** — `cognition/editable_network.py:504-510`
- Value: `times_traversed>=3 -> archive; help_ratio<0.15 (traversed>0) -> archive; traversed==0 & evidence<=1 & weight<0.35 -> archive; else score<edge_cut*0.55`
- Gates: Fine-grained edge-archival decision beyond the raw score cutoff.
- Context: Called from prune_working_graph().
- Depends on / feeds into: Consumes edge_keep_score() and edge_cut.

**_node_should_archive rules** — `cognition/editable_network.py:521-535`
- Value: `score<node_cut*0.50 & activation_count==0 -> archive; score<node_cut*0.65 & activation_count<=1 & recency<0.05 -> archive`
- Gates: Fine-grained node-archival decision beyond the raw score cutoff.
- Context: Called from prune_working_graph().
- Depends on / feeds into: Consumes node_keep_score() and node_cut.

**node-merge confidence blend** — `cognition/editable_network.py:688`
- Value: `min(0.99, 0.6*existing.confidence + 0.4*new_confidence)`
- Gates: How confidence is blended when a near-duplicate node is merged (add_node()).
- Context: Also reused identically in consolidate_duplicate_nodes() (line ~1631-1634).
- Depends on / feeds into: Feeds node.confidence, which drives traversal weighting and keep-score.

**edge-strengthen increment** — `cognition/editable_network.py:818`
- Value: `weight = min(0.99, weight + 0.02)`
- Gates: How much an edge's weight grows each time it's re-observed (add_edge() de-dup path).
- Context: add_edge().
- Depends on / feeds into: Feeds edge.weight used throughout traversal/keep-score.

**override blend ratio (traverse())** — `cognition/editable_network.py:1034-1044`
- Value: `overrides[key] = overrides[key]*0.7 + value*0.3*effective_weight`
- Gates: How competing numeric overrides from different activated nodes are blended in plain traverse().
- Context: "# Weighted blend for numeric overrides".
- Depends on / feeds into: effective_weight = current_weight * node.confidence.

**max_expansions cap** — `cognition/editable_network.py:1162`
- Value: `max(4000, 8*len(nodes))`
- Gates: Hard ceiling on BFS expansions in traverse_with_attention() to prevent combinatorial blow-up on dense graphs.
- Context: "at ~50k edges the frontier explodes combinatorially (observed: a single traverse hung a stage for minutes)."
- Depends on / feeds into: `if expansions >= max_expansions: break`.

**CONTRADICTS suppression factor** — `cognition/editable_network.py:1231`
- Value: `gate_value *= 0.1 (90% suppression, not 100%)`
- Gates: How strongly a CONTRADICTS edge dampens attention flow (continuous, not hard-blocked).
- Context: "Continuous suppression: reduce weight instead of hard block."
- Depends on / feeds into: gate_value = edge.weight * causal_impact, then this multiplier.

**STRENGTHENS boost factor** — `cognition/editable_network.py:1238`
- Value: `gate_value *= 1.3 (30% boost)`
- Gates: How strongly a STRENGTHENS edge amplifies attention flow.
- Context: Same causal-gate block as CONTRADICTS suppression.
- Depends on / feeds into: Same gate_value pipeline, feeding propagated_weight.

**traversal-outcome feedback increments** — `cognition/editable_network.py:1344-1347`
- Value: `helped -> weight=min(0.99,weight+0.01); not helped -> weight=max(0.01,weight-0.02)`
- Gates: How edge weights are reinforced/punished after an outcome is observed.
- Context: record_traversal_outcome(): "Strengthens edges along successful paths, weakens failed ones."
- Depends on / feeds into: Directly mutates edge.weight, read by every downstream scoring/traversal function.

**apply_edge_decay default + help_ratio floor** — `cognition/editable_network.py:1353,1365`
- Value: `min_traversals=10 (default); keep iff times_traversed<min_traversals or help_ratio>=0.2`
- Gates: Which edges are decayed/removed for "consistently failing".
- Context: apply_edge_decay() docstring.
- Depends on / feeds into: Called with min_traversals=5 from prune_working_graph() (line 1429), a different value than the function's own default.

**prune_working_graph safety ceilings** — `cognition/editable_network.py:1402-1403`
- Value: `max_edges=4000, max_nodes=300`
- Gates: Hard cap applied only after metric-based pruning, as a last-resort safety trim.
- Context: "Hard caps apply only as a safety ceiling after metric pruning."
- Depends on / feeds into: Trims by edge_keep_score()/node_keep_score() rank when still over cap.

**discover_relationships minimum history** — `cognition/editable_network.py:1734`
- Value: `len(activated_history) < 5 -> return 0`
- Gates: Minimum traversal history required before co-activation edge discovery runs at all.
- Context: discover_relationships().
- Depends on / feeds into: Gates the entire co-activation tallying pipeline.

**stage_workers thread-pool cap** — `cognition/editable_network.py:1759`
- Value: `max(1, min(len(activated_history), cpu_count or 4))`
- Gates: Degree of parallelism for co-activation tallying (falls back to 4 if cpu_count unknown).
- Context: discover_relationships().
- Depends on / feeds into: Controls ThreadPoolExecutor(max_workers=...).

**co-activation minimum count** — `cognition/editable_network.py:1801`
- Value: `count >= 3`
- Gates: Minimum accumulated co-activation count (across calls/runs) before a pair is considered for an edge at all.
- Context: "# Minimum co-activation threshold".
- Depends on / feeds into: Reads self._co_activation_tally (persisted across runs).

**lift-based edge-type thresholds** — `cognition/editable_network.py:1806-1815`
- Value: `lift > 2.0 -> STRENGTHENS; lift < 0.5 -> CONTRADICTS; else LEADS_TO`
- Gates: Which relationship type is minted between two co-activating nodes, normalized against the dataset's base success rate.
- Context: "Lift thresholds: >2x base rate → STRENGTHENS… <0.5x → CONTRADICTS, else LEADS_TO."
- Depends on / feeds into: Used only when `_use_lift` (base_rate in (0,1)); weight = min(0.95, success_ratio) or 1.0-success_ratio.

**absolute-fallback edge-type thresholds** — `cognition/editable_network.py:1818-1826`
- Value: `success_ratio > 0.6 -> STRENGTHENS; < 0.3 -> CONTRADICTS; else LEADS_TO`
- Gates: Same decision as above but for the non-lift (balanced-dataset) fallback path.
- Context: "# Absolute fallback (original logic, for balanced datasets)".
- Depends on / feeds into: Same add_edge() call as the lift-based branch.

**get_co_activation_bonus defaults + penalty** — `cognition/editable_network.py:1846-1848,1878-1883`
- Value: `min_co_count=3, min_success_rate=0.6 (defaults); rate<0.3 -> penalty -0.2`
- Gates: Confidence bonus/penalty applied to a co-activation partner principle when the firing principle succeeds/fails historically.
- Context: "B gets a positive bonus … B gets a negative penalty (redundancy signal)."
- Depends on / feeds into: Reads _co_activation_tally; bonus magnitude = raw success rate when rate>=min_success_rate.


### `cognition/episodic_attention.py` — 4 values

**DEFAULT_BANDWIDTH** — `cognition/episodic_attention.py:34`
- Value: `0.3`
- Gates: Gaussian KDE bandwidth controlling how quickly retrieval weight falls off with distance between fingerprints.
- Context: "# KDE bandwidth".
- Depends on / feeds into: Used in retrieve(): `weights = np.exp(-distances**2 / (2*bandwidth**2))`.

**DEFAULT_TOP_K** — `cognition/episodic_attention.py:35`
- Value: `5`
- Gates: Number of nearest historical episodes blended for a warm-start recommendation.
- Context: "# Number of nearest episodes to blend".
- Depends on / feeds into: `k = min(self.top_k, len(self._episodes))` in retrieve().

**weight_sum floor** — `cognition/episodic_attention.py:150`
- Value: `1e-10`
- Gates: Whether a KDE retrieval is trusted at all (below this, retrieval returns None instead of a near-meaningless blend).
- Context: `if weight_sum < 1e-10: return None`.
- Depends on / feeds into: Consumes top_weights.sum(); gates whether _blend_resolutions() runs.

**recommend_strategy confidence margin** — `cognition/episodic_attention.py:371`
- Value: `0.03`
- Gates: Minimum score gap between the best and second-best routing strategy before a kNN recommendation is trusted.
- Context: `if len(means) >= 2 and means[0][1] - means[1][1] < 0.03: return None`.
- Depends on / feeds into: Feeds DomainRouter's episodic-routing path via RoutedController.decide().


### `cognition/fingerprint.py` — 11 values

**trim_mean proportiontocut** — `cognition/fingerprint.py:24`
- Value: `0.1`
- Gates: How much of each tail is trimmed when computing the robust "center" statistic.
- Context: `_axis1_center`, robust center measures.
- Depends on / feeds into: Feeds axes.center dict, part of the fingerprint hash.

**class-count / class-size gates (axis1)** — `cognition/fingerprint.py:26,29`
- Value: `len(unique)<=20, mask.sum()>=3`
- Gates: Whether per-class median features are computed at all.
- Context: `_axis1_center`.
- Depends on / feeds into: Populates class_{cls}_median_f0 fields.

**spread feature cap** — `cognition/fingerprint.py:37`
- Value: `min(X.shape[1], 8)`
- Gates: How many feature columns get MAD/dispersion statistics computed.
- Context: `_axis2_spread`.
- Depends on / feeds into: Bounds cost of the spread axis.

**Levene test group cap** — `cognition/fingerprint.py:43,46`
- Value: `len(groups)>=2 gate; groups[:5] cap`
- Gates: Whether/how many class groups feed the Levene variance-homogeneity statistic.
- Context: `_axis2_spread`.
- Depends on / feeds into: Produces axes.spread["levene_f0"].

**shape feature cap / const-column skip** — `cognition/fingerprint.py:56,58`
- Value: `min(X.shape[1], 5); np.std(col) < 1e-12 skip`
- Gates: How many features get skew/kurtosis/Shapiro tests, and skips constant columns entirely.
- Context: `_axis3_shape`.
- Depends on / feeds into: Populates axes.shape.

**Shapiro test sample gates** — `cognition/fingerprint.py:70,72`
- Value: `len(col)>=8 gate; col[:min(5000,len(col))] cap`
- Gates: Whether the Shapiro-Wilk normality statistic runs, and caps its sample size for cost.
- Context: `_axis3_shape`.
- Depends on / feeds into: Populates shapiro_w_f{j}.

**dependence axis feature/class gates** — `cognition/fingerprint.py:83-84,92,128,139`
- Value: `f>=2; min(f,10); len(u)<=20; len(u)<=10 and n>=30; len(unique(y))==2 and n>=40`
- Gates: Whether Spearman correlation, linear/nonlinear probes, Cramér's V, and the binary formula-R² probe are computed at all.
- Context: `_axis4_dependence`.
- Depends on / feeds into: Populates spearman_rho_*, linear_probe_acc, nonlinear_probe_acc, cramers_v, formula_probe_r2 — several of which feed domain_router.py's routing gates.

**LogisticRegression/RandomForest probe params** — `cognition/fingerprint.py:98,113-119`
- Value: `max_iter=500, cv=3; subsample n>2000 -> 2000; RandomForestClassifier n_estimators=60, max_depth=8, cv=3`
- Gates: Cost/robustness of the linear and nonlinear probes and the subsampling cutoff for the RF probe.
- Context: Comment: "A large gap over the linear probe means the class boundary NEEDS interactions (sonar: linear ~0.77, RF ~0.85 -> gap ~0.08)."
- Depends on / feeds into: interaction_gap = nonlinear_probe_acc - linear_probe_acc, read by the optimizer to enable the sparse_linear RULE basis (E5).

**sequence-axis minimum n** — `cognition/fingerprint.py:161,165,172`
- Value: `n<20 -> skip axis; n>2 for lag1; n<=lag+5 skip per-lag`
- Gates: Whether autocorrelation/Ljung-Box sequence signals are computed at all.
- Context: `_axis5_sequence`.
- Depends on / feeds into: Populates lag1_autocorr / ljung_box_lag{n}, which feed fp["sequential"].

**tiny flag threshold (fingerprint.py)** — `cognition/fingerprint.py:209`
- Value: `n < 50`
- Gates: The 5-axis fingerprint's own "tiny" flag (independent constant from domain_router.TINY_N, same value).
- Context: compute_fingerprint().
- Depends on / feeds into: Stored in fp["tiny"]; used by downstream routing/reporting.

**sequential decision (lag1 / Ljung-Box)** — `cognition/fingerprint.py:219-222`
- Value: `lag1 >= 0.35 or max(ljung_box) > 8.0`
- Gates: Whether the dataset is flagged "sequential" (order-dependent), routing it toward D5_sequential.
- Context: "Use lag-1 series autocorrelation only — diff-based proxies false-positive on IID data."
- Depends on / feeds into: fp["sequential"] read directly by domain_router.py's m_sequential matcher.


### `cognition/formula_engine.py` — 11 values

**MAX_DEPTH** — `cognition/formula_engine.py:260`
- Value: `8`
- Gates: Maximum allowed formula-tree depth; trees exceeding it are pruned or rejected by validate().
- Context: "# Safety limits".
- Depends on / feeds into: mutate(), crossover(), validate(), validate_classifier(), _prune().

**MAX_NODES** — `cognition/formula_engine.py:261`
- Value: `50`
- Gates: Maximum allowed node count in a formula tree.
- Context: Same "# Safety limits" block.
- Depends on / feeds into: Same call sites as MAX_DEPTH.

**GRAD_CLIP** — `cognition/formula_engine.py:262`
- Value: `10.0`
- Gates: Clipping bound on numerically-differentiated gradients used for XGBoost custom objectives.
- Context: compute_gradient(): `np.clip(grad, -GRAD_CLIP, GRAD_CLIP)`.
- Depends on / feeds into: Feeds the boosting objective's gradient.

**HESS_MIN** — `cognition/formula_engine.py:263`
- Value: `1e-6`
- Gates: Floor on the Hessian (must stay positive for XGBoost).
- Context: compute_hessian(): `hess = np.maximum(hess, HESS_MIN)`.
- Depends on / feeds into: Feeds the boosting objective's Hessian.

**MAX_SINGLE_FEATURE_SEEDS** — `cognition/formula_engine.py:289`
- Value: `12`
- Gates: Number of single-feature seed formulas minted for the direct-classifier population.
- Context: "N seeds therefore cost ~N * MIN_EVIDENCE_BOOTSTRAP stages of forced bootstrap… 30 seeds would consume ~90 stages alone."
- Depends on / feeds into: get_seed_classifier_formulas(); interacts with formula_library.MIN_EVIDENCE_BOOTSTRAP.

**MAX_PAIR_SEEDS** — `cognition/formula_engine.py:290`
- Value: `4`
- Gates: Number of two-feature linear seed formulas minted.
- Context: Same comment block as MAX_SINGLE_FEATURE_SEEDS.
- Depends on / feeds into: get_seed_classifier_formulas().

**gradient/hessian epsilon** — `cognition/formula_engine.py:425`
- Value: `1e-5`
- Gates: Step size for numerical differentiation of the formula (gradient and Hessian).
- Context: compute_gradient()/compute_hessian() default parameter.
- Depends on / feeds into: Central-difference formula.

**stress-tiered mutation bands** — `cognition/formula_engine.py:498,501,504`
- Value: `stress_level<0.3 (low), <0.6 (medium), <0.85 (high), else extreme`
- Gates: Which mutation operator pool is sampled (constant_tweak vs subtree_swap vs random_invention) based on stress.
- Context: "Aggressiveness scales with stress."
- Depends on / feeds into: Consumes stress_level (produced by uil_stress.py's StressEngine); drives mutate().

**constant-tweak perturbation** — `cognition/formula_engine.py:638-639`
- Value: `normal(0, 0.3) perturbation, clipped to [-10.0, 10.0]`
- Gates: Magnitude of Gaussian jitter applied to a formula's constant during "constant_tweak" mutation, and its hard bounds.
- Context: _mutate_constant().
- Depends on / feeds into: Mutates FormulaNode.const.

**term-addition scale range** — `cognition/formula_engine.py:662`
- Value: `uniform(0.01, 0.5)`
- Gates: Coefficient range for a newly added term during "term_addition" mutation.
- Context: _mutate_add_term().
- Depends on / feeds into: Wraps the original tree in an "add" with this scaled new term.

**random-subtree leaf/const probabilities** — `cognition/formula_engine.py:726,728,731`
- Value: `rng.random()<0.3 -> leaf; rng.random()<0.6 -> leaf-type var vs const; const range uniform(-2,2)`
- Gates: Structure/shape of randomly generated subtrees (used by term_addition, subtree_swap, random_invention).
- Context: _generate_random_subtree().
- Depends on / feeds into: Recursively controls tree growth; bounded ultimately by MAX_DEPTH/MAX_NODES via _prune().


### `cognition/formula_library.py` — 13 values

**MIN_EVIDENCE_BOOTSTRAP** — `cognition/formula_library.py:55`
- Value: `3`
- Gates: How many trials every new formula gets before confidence/UCB selection takes over (bootstrap phase).
- Context: "Bootstrap guarantees every formula gets tried before confidence matters."
- Depends on / feeds into: get_best_formula() under_tested filter; also gates _resolve_incubation() trigger (line 324).

**EXPLORATION_C** — `cognition/formula_library.py:56`
- Value: `0.3`
- Gates: Strength of the UCB1-style exploration bonus in get_best_formula()'s scoring function.
- Context: Same block as MIN_EVIDENCE_BOOTSTRAP.
- Depends on / feeds into: `exploration = EXPLORATION_C * sqrt(log(total_evidence+1)/(evidence+1))` (line ~517).

**DATASET_RELEVANCE_BONUS** — `cognition/formula_library.py:57`
- Value: `0.05`
- Gates: Extra score given to a formula already proven on the current dataset.
- Context: score() function inside get_best_formula().
- Depends on / feeds into: Added when `dataset_fingerprint in f["dataset_types"]`.

**MAX_INCUBATING** — `cognition/formula_library.py:73`
- Value: `5`
- Gates: Maximum number of formulas allowed mid-trial ("incubating") simultaneously — throttles the mutation birth rate.
- Context: "MAX_INCUBATING caps how many can be mid-trial at once, which is what actually throttles the birth rate (optimizer.py checks this before calling add_formula)."
- Depends on / feeds into: get_incubating_formulas() lets callers (optimizer) enforce this cap.

**MAX_ACTIVE_POPULATION** — `cognition/formula_library.py:74`
- Value: `20`
- Gates: Cap on the promoted/active formula population; a new promotion evicts the weakest incumbent above this cap.
- Context: Same comment block as MAX_INCUBATING.
- Depends on / feeds into: _resolve_incubation() (lines 400-425).

**PROMOTION_CONFIDENCE** — `cognition/formula_library.py:75`
- Value: `0.4`
- Gates: Minimum confidence for an incubated formula to be considered for promotion to "active" (else retired).
- Context: _resolve_incubation(): `if entry["confidence"] < self.PROMOTION_CONFIDENCE: entry["status"]="retired"`.
- Depends on / feeds into: Directly gates promotion/eviction.

**retirement threshold (evidence>=10, confidence<0.3)** — `cognition/formula_library.py:312`
- Value: `evidence >= 10 and confidence < 0.3`
- Gates: When a formula with a proven track record is auto-retired for consistently failing.
- Context: "# Retire formulas that consistently fail".
- Depends on / feeds into: record_result(); triggers _archive_retired_entry() with reason "low_confidence".

**MAX_WORKING_ACTIVE** — `cognition/formula_library.py:446`
- Value: `60`
- Gates: Working-pool cap on active/evolved formulas; weakest are archived beyond this.
- Context: "Working-pool cap for the mutated/evolved population… archive the weakest active mutants beyond the cap."
- Depends on / feeds into: prune_to_cap(); ranks by (confidence, evidence).

**PARENT_EVIDENCE_SATURATION** — `cognition/formula_library.py:554`
- Value: `10.0`
- Gates: Evidence count at which the "evidence_score" component of mutation-parent fitness saturates to 1.0.
- Context: Fitness formula comment: "0.3 * evidence_score (evidence, saturating at 10)".
- Depends on / feeds into: select_mutation_parent()'s fitness() function (line 637).

**MIN_STRATIFICATION_STRENGTH** — `cognition/formula_library.py:555`
- Value: `0.15`
- Gates: Floor on how much of a mutation tournament must be filled by lineage-stratified sampling (protects rare lineages) as `progress` -> 1.0.
- Context: "protection against population-frequency bias swamping fitness comparison while a lineage is still young."
- Depends on / feeds into: `strength = max(MIN_STRATIFICATION_STRENGTH, 1.0 - progress)` in select_mutation_parent() and select_tournament_candidates().

**default tournament sizes (k)** — `cognition/formula_library.py:578,649`
- Value: `k=4 (select_mutation_parent), k=6 (select_tournament_candidates)`
- Gates: Tournament size for mutation-parent selection vs. fit-time tournament evaluation.
- Context: Function default parameters.
- Depends on / feeds into: Both use MIN_STRATIFICATION_STRENGTH-derived `n_stratified` split.

**tournament fitness weights** — `cognition/formula_library.py:596-599 (comment), 637-639 (code)`
- Value: `fitness = 0.5*confidence + 0.3*evidence_score + 0.2*diversity_bonus`
- Gates: Which formula wins a mutation-parent tournament once the stratified candidate pool is assembled.
- Context: "Composite fitness (not confidence alone) decides the winner."
- Depends on / feeds into: diversity_bonus = 1/(1+times_used_as_parent); consumes PARENT_EVIDENCE_SATURATION.

**Wilson lower bound z** — `cognition/formula_library.py:706`
- Value: `1.96 (95% confidence)`
- Gates: How aggressively low-evidence success ratios are discounted toward 0.5 when computing formula confidence.
- Context: "3/3 -> ~0.44, not 1.0; 88/100 -> ~0.80, not 0.88".
- Depends on / feeds into: record_result() sets entry["confidence"] from this; feeds get_best_formula(), promotion, retirement.


### `cognition/goal_cell.py` — 6 values

**GOAL_BIAS_MAX** — `cognition/goal_cell.py:71`
- Value: `3.0`
- Gates: Maximum amplification a goal can apply to any KG node/edge attention multiplier.
- Context: "A goal SOFTENS the search, it does not replace it… the maximum wind-back the consumer applies before clamping."
- Depends on / feeds into: Clamp in node_relevance() (line 245); consumed by editable_network.traverse_with_attention().

**GOAL_BIAS_MIN** — `cognition/goal_cell.py:72`
- Value: `0.1`
- Gates: Minimum suppression a goal can apply — "a runaway goal never zeroes out a node".
- Context: Same comment block as GOAL_BIAS_MAX.
- Depends on / feeds into: Same clamp as above.

**focus/ignore bias magnitudes** — `cognition/goal_cell.py:240,242`
- Value: `focus boost = 1.0 + 0.5*confidence; ignore suppression = 1.0 - 0.6*confidence`
- Gates: Per-condition attention multiplier in node_relevance() for KG traversal.
- Context: node_relevance() docstring: "Attention multiplier for a KG node (consumed by editable_network)."
- Depends on / feeds into: Combined multiplicatively across matched conditions, then clamped to [GOAL_BIAS_MIN, GOAL_BIAS_MAX].

**relation_bias boost** — `cognition/goal_cell.py:261`
- Value: `1.0 + 0.3*confidence`
- Gates: Edge-relation multiplier applied when an edge's relation type is in prefer_relations.
- Context: "Cannot suppress: goals endorse relations, they do not censor them."
- Depends on / feeds into: Consumed by editable_network.traverse_with_attention() edge gating.

**hypothesis_score adjustments** — `cognition/goal_cell.py:282-287`
- Value: `focus +0.5*confidence; keyword +0.3*confidence; ignore -0.5*confidence; floor max(0.05, score)`
- Gates: Ranking score for causal hypotheses under an active goal.
- Context: hypothesis_score() docstring: "1.0 for neutral, higher for on-goal variables".
- Depends on / feeds into: Consumed by causal_intervention.get_testable_hypotheses()'s 0.2 cutoff.

**domain_bias boost** — `cognition/goal_cell.py:302`
- Value: `1.0 + 0.4*confidence`
- Gates: Router bias toward a candidate domain/strategy named in prefer_domains.
- Context: "the goal cannot forbid a domain, only endorse one (bug #21 invariant)."
- Depends on / feeds into: Intended consumer is domain_router.py (not currently wired in, per read).


### `cognition/journal_forgetting.py` — 9 values

**half_life floor** — `cognition/journal_forgetting.py:43`
- Value: `10`
- Gates: Minimum allowed half_life passed to JournalForgetting, regardless of caller input.
- Context: `self.half_life = max(10, half_life)`.
- Depends on / feeds into: Used to compute `self._decay_constant = np.log(2) / self.half_life`, which drives compute_weights().

**window_size floor** — `cognition/journal_forgetting.py:44`
- Value: `50`
- Gates: Minimum allowed sliding-window size, regardless of caller input.
- Context: `self.window_size = max(50, window_size)`.
- Depends on / feeds into: Used by apply_window() and weighted_sample()'s default `n_samples`.

**weighted_sample RNG seed** — `cognition/journal_forgetting.py:106`
- Value: `42`
- Gates: Default reproducibility seed for recency-weighted resampling of journal entries fed to the Miner.
- Context: `if rng is None: rng = np.random.default_rng(42)`.
- Depends on / feeds into: `rng.choice(..., p=probabilities)` — determines which entries (and duplicates) are sampled into the Miner's training set.

**PrincipleVersioning default confidence** — `cognition/journal_forgetting.py:183`
- Value: `0.5`
- Gates: Confidence recorded in a version snapshot when the principle dict lacks a confidence field.
- Context: `"confidence": principle.get("confidence", 0.5),` inside record_version().
- Depends on / feeds into: Used by has_regressed() to compare confidence across versions.

**ContradictionDetector.THRESHOLD_TOLERANCE** — `cognition/journal_forgetting.py:242`
- Value: `0.15 (15% relative difference)`
- Gates: How close two conditions' thresholds must be (relative, "max" mode) to count as the "same condition" for contradiction detection.
- Context: `THRESHOLD_TOLERANCE = 0.15  # 15% relative difference` — note this is a DIFFERENT constant from condition_match.DEFAULT_SIMILARITY_TOLERANCE (0.10) despite the similar name; ContradictionDetector uses a looser 15% band than the principle-merge 10% band.
- Depends on / feeds into: Passed into `thresholds_match(..., self.THRESHOLD_TOLERANCE, mode="max")` inside `_conditions_overlap()` (line 332-335), gating `detect_contradictions()`.

**conflict-check zero-division epsilon** — `cognition/journal_forgetting.py:297`
- Value: `1e-6`
- Gates: Minimum magnitude both override values must have before the ratio-based conflict check runs (avoids div-by-~0).
- Context: `elif abs(va) > 1e-6 and abs(vb) > 1e-6:`.
- Depends on / feeds into: Guards the 3.0-ratio check above.

**conflicting-action magnitude ratio** — `cognition/journal_forgetting.py:299`
- Value: `3.0`
- Gates: How large a magnitude ratio between two principles' same-key overrides must be to count as a conflict (when signs agree).
- Context: `ratio = max(va, vb) / min(abs(va), abs(vb)); if ratio > 3.0:  # 3x difference = conflict`.
- Depends on / feeds into: Appends to `conflicts` list in `_check_pair()`, which gates whether a contradiction record is emitted at all.

**contradiction severity weights** — `cognition/journal_forgetting.py:306`
- Value: `shared*0.4 + conflicts*0.6`
- Gates: Relative weighting of "how many conditions overlap" vs "how many actions conflict" in computing a contradiction's severity score.
- Context: `severity = len(shared) * 0.4 + len(conflicts) * 0.6` under "# Severity: more shared conditions + more conflicts = worse".
- Depends on / feeds into: Multiplied by the confidence sum below; final severity sorts `detect_contradictions()` output ("worst first").

**contradiction severity confidence multiplier** — `cognition/journal_forgetting.py:310`
- Value: ``severity *= (conf_a + conf_b)` (confidence defaults 0.5/0.5 at lines 307-308)`
- Gates: Amplifies severity when both contradicting principles are high-confidence.
- Context: "# Both high-confidence = more severe".
- Depends on / feeds into: Final `severity` field in the returned contradiction dict; `conf_a`/`conf_b` default to 0.5 via `.get("confidence", 0.5)` when missing.


### `cognition/knowledge_archive.py` — 8 values

**LOW_ACCURACY_THRESHOLD** — `cognition/knowledge_archive.py:27`
- Value: `0.62`
- Gates: Best-accuracy cutoff below which a well-tried formula is verdict "low" (avoid).
- Context: Class-level constant, used with MIN_TRIALS_FOR_AVOID.
- Depends on / feeds into: _formula_verdict(); flows into avoid_formulas list of the consciousness brief.

**MIN_TRIALS_FOR_AVOID** — `cognition/knowledge_archive.py:28`
- Value: `4`
- Gates: Minimum trial count before a formula's poor accuracy is trusted enough to mark it "low"/avoid.
- Context: Used alongside LOW_ACCURACY_THRESHOLD in _formula_verdict().
- Depends on / feeds into: `trials >= cls.MIN_TRIALS_FOR_AVOID and best < cls.LOW_ACCURACY_THRESHOLD`.

**graph_prune_events cap** — `cognition/knowledge_archive.py:112`
- Value: `40 (last 40 kept)`
- Gates: How much prune-event history is retained per dataset bucket.
- Context: `bucket["graph_prune_events"][-40:]`.
- Depends on / feeds into: record_graph_prune().

**journal archive cap** — `cognition/knowledge_archive.py:133-134`
- Value: `50_000`
- Gates: Maximum archived journal entries kept before the oldest are dropped.
- Context: `if len(store) > 50_000: self._data["journal"] = store[-50_000:]`.
- Depends on / feeds into: archive_journal_entries().

**cognition_prune_events cap** — `cognition/knowledge_archive.py:197`
- Value: `40`
- Gates: How much cognition-prune history (journal/principles/hypotheses) is retained per dataset.
- Context: record_cognition_prune(), mirrors graph_prune_events cap.
- Depends on / feeds into: Same pattern as line 112.

**verdict bands (strong/moderate)** — `cognition/knowledge_archive.py:264,266`
- Value: `0.78 (strong), 0.68 (moderate)`
- Gates: Formula-quality label used to decide inclusion in "known_good_formulas" of the consciousness brief.
- Context: `_formula_verdict`: also requires `trials < 2` -> "exploring" fallback.
- Depends on / feeds into: Read by get_consciousness_brief() to sort known_good vs avoid lists.

**conclusions cap** — `cognition/knowledge_archive.py:304`
- Value: `30`
- Gates: Maximum stored conclusions per dataset before oldest are dropped.
- Context: record_conclusion().
- Depends on / feeds into: get_consciousness_brief() further trims this to [-8:].

**consciousness-brief window sizes** — `cognition/knowledge_archive.py:335,366,368,375,376,377`
- Value: `conclusions[-8:], avoid_formulas[:6], known_good[:3], avoid_formulas[:12], known_good_formulas[:8], abstract_summary[-14:]`
- Gates: How much prior-knowledge context is surfaced to the optimizer/SymbolicFormula per dataset (the "consciousness brief").
- Context: get_consciousness_brief().
- Depends on / feeds into: Consumes formula_exploration + conclusions buckets; feeds optimizer prompts/state.


### `cognition/meta_principle_merger.py` — 5 values

**SIMILARITY_THRESHOLD** — `cognition/meta_principle_merger.py:100`
- Value: `0.6`
- Gates: Minimum structural similarity for two principles to form an edge in the community-detection similarity graph.
- Context: "Minimum similarity to form an edge in the similarity graph".
- Depends on / feeds into: Used in detect_communities() adjacency construction (sim_matrix[i,j] >= SIMILARITY_THRESHOLD); reads structural_similarity() from structural_representation.py.

**MIN_CLUSTER_SIZE** — `cognition/meta_principle_merger.py:102`
- Value: `3`
- Gates: Minimum number of principles required to form (and to even attempt) a cluster/meta-principle.
- Context: "Minimum cluster size to form a meta-principle"; also short-circuits detect_communities() entirely when n < MIN_CLUSTER_SIZE (line 127).
- Depends on / feeds into: Filters clusters_dict values before merge_cluster() is called.

**MAX_META_PRINCIPLES** — `cognition/meta_principle_merger.py:104`
- Value: `50`
- Gates: Cap on how many meta-principles can be created in a single merge pass.
- Context: "Maximum number of meta-principles to create per merge pass".
- Depends on / feeds into: Truncates the sorted `clusters` list (largest first) before merge().

**max_iterations** — `cognition/meta_principle_merger.py:151`
- Value: `20`
- Gates: Maximum number of label-propagation passes for greedy community detection.
- Context: Comment-free local variable inside detect_communities(); loop breaks early if no label changed.
- Depends on / feeds into: Bounds the `for _ in range(max_iterations)` loop that mutates `labels`.

**threshold = len(cluster_indices) * 0.5** — `cognition/meta_principle_merger.py:226`
- Value: `0.5 (50% of cluster size)`
- Gates: Which context labels are common enough to be included in the synthesized meta-principle's summary.
- Context: "Most common context labels (appearing in >50% of cluster)".
- Depends on / feeds into: Filters all_contexts dict into common_contexts, which becomes MetaPrinciple.context_labels.


### `cognition/principle_hierarchy.py` — 12 values

**PROMOTE_3_TO_2_MIN_DATASETS** — `cognition/principle_hierarchy.py:37`
- Value: `3`
- Gates: Minimum distinct datasets a Tier-3 (dataset-specific) principle must be confirmed on to promote to Tier-2 (category).
- Context: "To promote from Tier 3 → Tier 2: # Confirmed on ≥3 distinct datasets".
- Depends on / feeds into: Checked in evaluate_promotion() alongside PROMOTE_3_TO_2_MIN_CONFIDENCE; `n_datasets` comes from `_confirmed_datasets`, populated by record_application().

**PROMOTE_3_TO_2_MIN_CONFIDENCE** — `cognition/principle_hierarchy.py:38`
- Value: `0.80`
- Gates: Minimum confidence for Tier-3 → Tier-2 promotion.
- Context: Paired with PROMOTE_3_TO_2_MIN_DATASETS in evaluate_promotion() (line 146).
- Depends on / feeds into: `confidence` field maintained by uil_library.py's boosts/decays/clips.

**PROMOTE_2_TO_1_MIN_DATASETS** — `cognition/principle_hierarchy.py:41`
- Value: `10`
- Gates: Minimum distinct datasets for Tier-2 → Tier-1 (universal) promotion.
- Context: "Confirmed on ≥10 distinct datasets".
- Depends on / feeds into: evaluate_promotion() line ~153.

**PROMOTE_2_TO_1_MIN_CLASSIFIERS** — `cognition/principle_hierarchy.py:42`
- Value: `2`
- Gates: Minimum distinct classifiers a principle must be confirmed on for Tier-2 → Tier-1 promotion.
- Context: "Works on ≥2 different classifiers".
- Depends on / feeds into: `classifiers` from `principle.get("classifiers_confirmed", [])`, populated in uil_library.record_application().

**PROMOTE_2_TO_1_MIN_CONFIDENCE** — `cognition/principle_hierarchy.py:43`
- Value: `0.85`
- Gates: Minimum confidence for Tier-2 → Tier-1 promotion (highest confidence bar in the hierarchy).
- Context: Part of the four-way AND in evaluate_promotion() (line ~155).
- Depends on / feeds into: Same shared confidence field as all other confidence gates.

**PROMOTE_2_TO_1_MIN_EVIDENCE** — `cognition/principle_hierarchy.py:44`
- Value: `100`
- Gates: Minimum evidence_count for Tier-2 → Tier-1 promotion.
- Context: "≥100 evidence observations".
- Depends on / feeds into: `evidence = principle.get("evidence_count", 0)`, accumulated via `_merge_evidence` in uil_library.py.

**DEMOTE_MIN_APPLICATIONS** — `cognition/principle_hierarchy.py:47`
- Value: `15`
- Gates: Minimum times_applied before a principle above Tier-3 is eligible for demotion evaluation.
- Context: "Demotion: if help_ratio drops below threshold after enough applications".
- Depends on / feeds into: evaluate_demotion() line 179; note this is a DIFFERENT bar from uil_library's own DECAY_MIN_APPLICATIONS=20 — two independently-tuned "enough applications" constants for parallel decay/demotion mechanisms.

**DEMOTE_HELP_RATIO** — `cognition/principle_hierarchy.py:48`
- Value: `0.25`
- Gates: Help-ratio floor below which a principle demotes one tier.
- Context: "Must help >25% of the time or demote" — note this differs from uil_library's DECAY_HELP_RATIO=0.3, another parallel/independent bar.
- Depends on / feeds into: evaluate_demotion() line 181; demotion moves `min(current_tier + 1, TIER_DATASET_SPECIFIC)`.

**QUARANTINE_FAILURE_THRESHOLD** — `cognition/principle_hierarchy.py:51`
- Value: `5`
- Gates: Consecutive failures on one dataset before a principle is quarantined (excluded) for that dataset.
- Context: "≥5 consecutive failures on one dataset".
- Depends on / feeds into: check_quarantine() line 204; `consecutive_failures` tracked per-dataset in `_dataset_performance`, reset to 0 on any success (record_application line 119).

**default tier fallback (TIER_DATASET_SPECIFIC=3), repeated** — `cognition/principle_hierarchy.py:136, 170, 238, 353`
- Value: `3`
- Gates: Assumed tier for any principle dict missing a "tier" key, in promotion eval, demotion eval, filter_by_tier, and get_tier_summary respectively.
- Context: e.g. `current_tier = principle.get("tier", TIER_DATASET_SPECIFIC)`.
- Depends on / feeds into: Affects which promotion/demotion branch runs and how filter_by_tier sorts/includes principles with malformed data.

**exception-principle starting confidence** — `cognition/principle_hierarchy.py:333`
- Value: `0.5`
- Gates: Starting confidence for a newly created Tier-4 exception principle.
- Context: `exception["confidence"] = 0.5  # Start neutral`.
- Depends on / feeds into: create_exception_principle(), called from uil_library.create_exception(); exactly clears the executor's 0.50 fire-gate (0.5 < 0.50 is False → fires immediately).

**exception-principle starting evidence_count** — `cognition/principle_hierarchy.py:334`
- Value: `1`
- Gates: Starting evidence_count for a Tier-4 exception principle.
- Context: Immediately follows the confidence=0.5 line in create_exception_principle().
- Depends on / feeds into: Feeds subsequent decay/promotion evidence math.


### `cognition/structural_representation.py` — 2 values

**CONTEXT_VOCABULARY thresholds** — `cognition/structural_representation.py:23-46`
- Value: `tiny<50, small<200, medium<=1000, large>1000, low_dim<20, mid_dim<=100, high_dim>100, low_ratio<0.3, moderate_ratio<=1.0, high_ratio>1.0, extreme_ratio>5.0, dense<0.1, moderate_sparse<=0.5, highly_sparse>0.5, low_entropy<0.3, moderate_entropy<=0.7, high_entropy>0.7`
- Gates: The entire concept vocabulary used to translate raw DT conditions into structural context labels (e.g. "tiny", "high_dim").
- Context: "# Concept Vocabulary" — "Dataset size classes / Dimensionality / Feature ratio classes / Sparsity classes / Entropy classes".
- Depends on / feeds into: Consumed by StructuralEncoder._infer_context(), which is nearest-match (not exact) against condition thresholds; feeds MetaPrincipleMerger via structural_similarity().

**structural_similarity weighting** — `cognition/structural_representation.py:134`
- Value: `0.35 (context) + 0.35 (action) + 0.30 (conditions)`
- Gates: How two principles' structural similarity score is composed — directly feeds meta_principle_merger's SIMILARITY_THRESHOLD=0.6 gate.
- Context: "# Weighted combination".
- Depends on / feeds into: Called by MetaPrincipleMerger.detect_communities().


### `cognition/uil_executor.py` — 2 values

**executor sort-key defaults** — `cognition/uil_executor.py:73`
- Value: `tier default 3, confidence default 0`
- Gates: Fallback tier/confidence used to order principle evaluation when a principle dict is missing those fields (Tier-3-like, lowest-priority treatment).
- Context: `key=lambda p: (p.get("tier", 3), -p.get("confidence", 0))`.
- Depends on / feeds into: Determines override-merge precedence in `get_overrides()`'s first-wins loop.

**executor minimum firing confidence** — `cognition/uil_executor.py:78`
- Value: `0.50`
- Gates: The absolute floor confidence a principle must have to be evaluated/fire at all, regardless of tier.
- Context: `if confidence < 0.50: continue  # Skip low-confidence principles`.
- Depends on / feeds into: This is the single gate every principle's confidence (however computed — seed 0.5, decayed 0.5×, boosted +0.02/+0.03, clipped 0.05-0.99, etc.) must clear to ever produce an override; seed principles sit exactly at this boundary (0.5 fails `< 0.50`? note: 0.5 < 0.50 is False, so seeds DO pass and fire immediately).


### `cognition/uil_library.py` — 18 values

**max_tier default = TIER_DATASET_SPECIFIC (3)** — `cognition/uil_library.py:83`
- Value: `3`
- Gates: Default ceiling on which tier of principles get_principles() returns when no caller-supplied max_tier.
- Context: `max_tier: int = TIER_DATASET_SPECIFIC` in get_principles() signature.
- Depends on / feeds into: Passed to PrincipleHierarchy.filter_by_tier(); callers that omit max_tier silently get all tiers up to 3 (excludes only Tier-4 exceptions not matching dataset).

**routing-rule initial confidence** — `cognition/uil_library.py:141`
- Value: `0.55`
- Gates: Starting confidence assigned to a newly minted routing_rule cell (Category tier).
- Context: record_routing_rule() builds `rule = {..., "confidence": 0.55, "evidence_count": 1, ...}` with no comment explaining the specific value.
- Depends on / feeds into: Compared later against executor's fire threshold (0.50, see below) and against promotion thresholds in principle_hierarchy.py.

**routing-rule evidence-merge confidence bump** — `cognition/uil_library.py:154`
- Value: `cap 0.95, increment +0.03`
- Gates: How much confidence a routing rule gains each time an identical rule is re-observed, and its ceiling.
- Context: `p["confidence"] = min(0.95, p.get("confidence", 0.5) + 0.03)` inside record_routing_rule()'s merge branch.
- Depends on / feeds into: Feeds the same confidence field read by the executor's 0.50 fire gate and hierarchy promotion checks.

**min_evidence override floor** — `cognition/uil_library.py:187`
- Value: `3`
- Gates: Lower bound on the evidence-count threshold when a caller overrides MIN_EVIDENCE via `min_evidence` argument to update_principles().
- Context: `threshold = self.MIN_EVIDENCE if min_evidence is None else max(3, int(min_evidence))`; docstring calls the MIN_EVIDENCE=10 default an "unreachable threshold" requiring ~200 stages, and frames lowering it as an "EPISTEMOLOGY change... opt-in via cfg.adaptive_min_evidence."
- Depends on / feeds into: Gates promotion of mined rules into principles in the loop right below (line ~194).

**cross-classifier confidence boost per classifier** — `cognition/uil_library.py:237`
- Value: `0.02`
- Gates: Confidence increase per additional distinct classifier that confirms a principle.
- Context: `boost = 0.02 * (len(confirmed) - 1)` under comment "Cross-classifier boost: confirmed on multiple classifiers".
- Depends on / feeds into: Applied in record_application(); feeds principle["confidence"], capped by the constant below.

**cross-classifier confidence cap** — `cognition/uil_library.py:238`
- Value: `0.99`
- Gates: Ceiling on confidence after the cross-classifier boost.
- Context: `p["confidence"] = min(0.99, p["confidence"] + boost)`.
- Depends on / feeds into: Same confidence field read by executor fire-gate (0.50) and hierarchy promotion thresholds (0.80/0.85).

**causal-adjustment confidence clip bounds** — `cognition/uil_library.py:268`
- Value: `(0.05, 0.99)`
- Gates: Hard floor/ceiling on confidence after a causal-intervention multiplier is applied.
- Context: `p["confidence"] = float(np.clip(p.get("confidence", 0.5) * multiplier, 0.05, 0.99))` inside apply_causal_adjustment(); docstring: "Confirmed causal → multiplier > 1 boosts confidence; refuted → multiplier < 1 sharply cuts it."
- Depends on / feeds into: Called from CausalInterventionEngine.evaluate_intervention() (outside cognition/, per docstring); result feeds executor fire-gate and hierarchy tiers.

**all_principles_exhausted applied-count bar** — `cognition/uil_library.py:280`
- Value: `3`
- Gates: Minimum times_applied per principle for the library to be considered "exhausted" (all principles tried enough).
- Context: `return all(p.get("times_applied", 0) >= 3 for p in principles)`; comment: "Used by StressEngine to determine failure."
- Depends on / feeds into: Consumed by StressEngine (outside these files) as a failure signal.

**seed principle initial tier** — `cognition/uil_library.py:320`
- Value: `TIER_DATASET_SPECIFIC (3)`
- Gates: Starting tier for human-provided seed principles — forces them to earn promotion rather than start trusted.
- Context: `"tier": TIER_DATASET_SPECIFIC,  # Must earn promotion` inside accept_seed_principles().
- Depends on / feeds into: Read by PrincipleHierarchy.filter_by_tier / update_principle_tiers.

**seed principle initial confidence** — `cognition/uil_library.py:324`
- Value: `0.5`
- Gates: Starting confidence for seed principles ("neutral — not trusted yet").
- Context: `"confidence": 0.5,  # Neutral — not trusted yet`.
- Depends on / feeds into: Below the 0.50 executor fire-gate boundary condition (`< 0.50` excludes it, since equality passes — seed principles are right at the firing edge).

**seed principle initial evidence_count** — `cognition/uil_library.py:325`
- Value: `0`
- Gates: Starting evidence for seeds; they must accumulate real evidence before merge/promotion logic treats them as substantiated.
- Context: `"evidence_count": 0,` in the seed principle dict.
- Depends on / feeds into: Compared against MIN_EVIDENCE / hierarchy promotion evidence bars.

**NON_CONSTRAINABLE_CONFIG** — `cognition/uil_library.py:382-385`
- Value: `frozenset({"learning_mode", "parallel_eval_workers"})`
- Gates: Which classifier-config keys a mined principle is forbidden from ever overriding.
- Context: Comment explains `learning_mode` is architecture-locked (bug #21 invariant) and `parallel_eval_workers` is a "machine-resource knob with no bearing on model quality."
- Depends on / feeds into: Checked inside `_generate_override()` (line ~470) to skip emitting `config_max.<param>` for these keys.

**PENALTY_BOUND_RANGE** — `cognition/uil_library.py:404`
- Value: `(0.0, 10.0)`
- Gates: Legal domain for any penalty-related override value; values outside it are treated as corrupt, not as constraints.
- Context: "A penalty bound outside the knob's own domain is not a constraint, it is corrupt data... Applied on INGEST as well as on generation, because the graph already contains poisoned nodes."
- Depends on / feeds into: Read externally at core/optimizer.py:3117 (`lo_ok, hi_ok = UIL_Library.PENALTY_BOUND_RANGE`) — a cross-file usage site.

**threshold-inversion epsilon (lower edge)** — `cognition/uil_library.py:485`
- Value: `0.01`
- Gates: How ">=" is converted to an exclusive lower bound when building an override region from a mined condition.
- Context: `lo = threshold if op == ">" else threshold - 0.01`.
- Depends on / feeds into: Feeds `r["lo"]`, which becomes `config_max.<param>` / `max_penalty` overrides.

**threshold-inversion epsilon (upper edge)** — `cognition/uil_library.py:488`
- Value: `0.01`
- Gates: How "<=" is converted to an exclusive upper bound in the same region-building logic.
- Context: `hi = threshold if op == "<" else threshold + 0.01`.
- Depends on / feeds into: Feeds `r["hi"]`, becomes `config_min.<param>` / `min_penalty` overrides.

**merge-evidence weighted-average weights and cap** — `cognition/uil_library.py:539-541`
- Value: `old-weight 0.6, new-weight 0.4, cap 0.99, defaults 0.5/0.5`
- Gates: How confidence is recomputed when a new mined rule matches an existing principle (biased toward new evidence) and its ceiling.
- Context: "# Update confidence (weighted average, biased toward new evidence)" then `existing["confidence"] = min(0.99, 0.6 * old_conf + 0.4 * new_conf)`.
- Depends on / feeds into: Called from update_principles()'s `_merge_evidence()` path whenever `_find_similar` (condition_match.condition_sets_similar) matches.

**decay halving multiplier** — `cognition/uil_library.py:560`
- Value: `0.5`
- Gates: How much confidence is cut when a principle's help-ratio falls below DECAY_HELP_RATIO after DECAY_MIN_APPLICATIONS.
- Context: "If times_helped / times_applied < 0.3 after 20+ applications, halve the confidence." then `new_conf = old_conf * 0.5`.
- Depends on / feeds into: Feeds the prune-floor check immediately below (0.30) and is logged via `_log_cell(..., op="decay", cause=f"help_ratio=...")`.

**prune/removal confidence floor** — `cognition/uil_library.py:569`
- Value: `0.30`
- Gates: Confidence below which a principle is dropped from the library entirely (unless it's a Tier-4 exception).
- Context: "If confidence drops below 0.3, remove." then `if p.get("confidence", 0) >= 0.30 or p.get("tier") == TIER_EXCEPTION: surviving.append(p)`.
- Depends on / feeds into: Runs inside `_apply_decay()`, called every `update_principles()` cycle; survivors then pass through `hierarchy.update_principle_tiers()`.


### `cognition/uil_miner.py` — 20 values

**min_entries default (Miner constructor)** — `cognition/uil_miner.py:66`
- Value: `40`
- Gates: The full-mining row-count threshold; below it (but ≥ TINY_MIN_ENTRIES) tiny-data mode engages instead of normal mining.
- Context: `def __init__(self, min_entries: int = 40):`.
- Depends on / feeds into: Compared against `len(entries)` at line 110 to set `tiny_mode`; also gates `min_rows` at line 111.

**tiny-mode max_depth** — `cognition/uil_miner.py:127`
- Value: `2`
- Gates: Decision-tree depth used when mining in tiny-data mode.
- Context: "Tiny mode is pinned shallow: with <40 rows a depth-3+ tree is memorization, and there is no held-out split to catch it." → `max_depth = 2`.
- Depends on / feeds into: Passed to DecisionTreeRegressor(max_depth=...) at line 153-157.

**stress-scaled max_depth formula** — `cognition/uil_miner.py:129`
- Value: `base 3, stress multiplier 2 (range 3-5)`
- Gates: Non-tiny-mode tree depth, scaled by current stress level.
- Context: `max_depth = 3 + int(stress_level * 2)  # 3 calm → 5 max stress`.
- Depends on / feeds into: Passed into DecisionTreeRegressor; deeper trees emit more/finer conditions per rule.

**_MIN_FOR_HOLDOUT** — `cognition/uil_miner.py:136`
- Value: `40`
- Gates: Row-count threshold below which the held-out train/test split is skipped and full-data R² is used instead, and derived-feature synthesis is disabled.
- Context: "With <40 rows, the split would leave fewer than ~10 test rows — too noisy to be a meaningful signal... fall back to full-data R²... and skip synthesis."
- Depends on / feeds into: Controls `_has_holdout` flag, which gates the R²-threshold-triggered derived-feature synthesis at line 173.

**holdout split fraction** — `cognition/uil_miner.py:140`
- Value: `test_size=0.25, random_state=42`
- Gates: Fraction of rows held out to honestly score R² (and derived-feature acceptance) instead of scoring on training data.
- Context: `_tr_idx, _hld_idx = train_test_split(_all_idx, test_size=0.25, random_state=42)`.
- Depends on / feeds into: X_hld/y_hld feed `base_r2` (line 163), which gates DERIVED_FEATURE_R2_THRESHOLD and the +0.05 acceptance margin.

**min_samples_leaf formula (base tree)** — `cognition/uil_miner.py:152`
- Value: `tiny mode → 4; else max(5, len(X_tr) // 20)`
- Gates: Minimum samples per leaf for the base decision tree — indirectly the minimum evidence_count a mined rule can have.
- Context: `min_leaf = 4 if tiny_mode else max(5, len(X_tr) // 20)`.
- Depends on / feeds into: Passed to DecisionTreeRegressor and reused to compute rule-extraction `min_evidence` at line 204.

**DecisionTreeRegressor random_state (base tree)** — `cognition/uil_miner.py:153-156`
- Value: `random_state=42`
- Gates: Determinism of tree structure/split selection, hence which rules are mined.
- Context: `tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_leaf, random_state=42)`.
- Depends on / feeds into: Reused identically for the enhanced tree at line 178-182.

**base_r2 floor clamp** — `cognition/uil_miner.py:163`
- Value: `0.0`
- Gates: Prevents negative R² (worse than mean-predictor) from being reported/compared.
- Context: `base_r2 = max(0.0, float(tree.score(X_hld, y_hld)))`.
- Depends on / feeds into: Compared against DERIVED_FEATURE_R2_THRESHOLD (0.5) and reported to StressEngine via get_last_r2_score().

**enhanced tree min_samples_leaf** — `cognition/uil_miner.py:180`
- Value: `max(5, len(_tr_idx) // 20)`
- Gates: Same leaf-size floor as the base tree, applied to the derived-feature-enhanced tree.
- Context: Inside the `if base_r2 < DERIVED_FEATURE_R2_THRESHOLD and _has_holdout:` branch.
- Depends on / feeds into: Feeds `tree_enhanced`; used only if `enhanced_r2 > base_r2 + 0.05`.

**enhanced-model acceptance margin** — `cognition/uil_miner.py:190`
- Value: `0.05`
- Gates: How much better (in R²) the derived-feature model must be before it replaces the base model.
- Context: "# Accept enhanced model only if it genuinely generalises better." → `if enhanced_r2 > base_r2 + 0.05:`.
- Depends on / feeds into: Decides whether `_last_r2`, `_last_tree`, `X_meta`, `valid_features`, and `_useful_derived` are swapped to the enhanced versions.

**rule-extraction min_evidence** — `cognition/uil_miner.py:204`
- Value: `tiny mode → 2; else min(5, min_leaf)`
- Gates: Minimum leaf sample count for a tree leaf to be extracted as a candidate rule.
- Context: `min_evidence=2 if tiny_mode else min(5, min_leaf),` passed to `_extract_rules()`.
- Depends on / feeds into: Used at line 466 `n_samples >= min_evidence` inside the tree-walk leaf check.

**poisoned-value guard** — `cognition/uil_miner.py:315`
- Value: `1e30`
- Gates: Magnitude beyond which a journal feature value is treated as poisoned/invalid and the row is dropped from mining.
- Context: "M0 hardening: a poisoned entry (NaN/Inf or beyond float32 range) must not crash mining for everyone -- sklearn casts to float32 and rejects such values." → `if not np.isfinite(fval) or abs(fval) > 1e30: valid = False`.
- Depends on / feeds into: Determines which rows enter `X_meta`/`y_regret` in `_build_matrix()`.

**derived-feature "active column" variance threshold** — `cognition/uil_miner.py:363`
- Value: `1e-8`
- Gates: Whether a base feature column has enough variance to be used for derived-feature synthesis.
- Context: `active_cols = [i for i in range(n_base) if col_std[i] > 1e-8]`.
- Depends on / feeds into: Restricts which columns participate in the pairwise/nonlinear derivation loops below.

**max_derived cap** — `cognition/uil_miner.py:367`
- Value: `15`
- Gates: Hard cap on number of synthesized derived features (interactions + nonlinear transforms) per mining call.
- Context: "# Cap the number of interactions to prevent explosion / At most 15 derived features" → `max_derived = 15`.
- Depends on / feeds into: Checked repeatedly in the pairwise-interaction and nonlinear-transform loops (lines 371, 374, 386, 396, 405).

**derived pairwise-product variance threshold** — `cognition/uil_miner.py:378`
- Value: `1e-8`
- Gates: Whether a synthesized `a*b` interaction column is kept (rejected if constant).
- Context: `if np.std(product) > 1e-8: derived_cols.append(product)`.
- Depends on / feeds into: Determines contents of `X_enhanced`.

**derived log1p variance threshold** — `cognition/uil_miner.py:392`
- Value: `1e-8`
- Gates: Whether the `log1p(|x|)` transform column is kept.
- Context: `log_col = np.log1p(np.abs(col)); if np.std(log_col) > 1e-8:`.
- Depends on / feeds into: Determines contents of `X_enhanced`.

**derived square-transform variance threshold** — `cognition/uil_miner.py:401`
- Value: `1e-8`
- Gates: Whether the `x²` transform column is kept.
- Context: `sq_col = col ** 2; if np.std(sq_col) > 1e-8:`.
- Depends on / feeds into: Determines contents of `X_enhanced`.

**derived inverse-transform variance threshold** — `cognition/uil_miner.py:409-410`
- Value: `1e-8`
- Gates: Whether the bounded-inverse `1/(1+|x|)` transform column is kept.
- Context: `inv_col = 1.0 / (1.0 + np.abs(col)); if np.std(inv_col) > 1e-8:`.
- Depends on / feeds into: Determines contents of `X_enhanced`.

**tiny-mode confidence shrinkage constant** — `cognition/uil_miner.py:471`
- Value: `8.0`
- Gates: How aggressively a tiny-data rule's confidence is pulled toward 0.5 as a function of its sample count.
- Context: "confidence is shrunk toward 0.5 by an evidence-based factor n/(n+8) -- a cheap wide-confidence-interval stand-in" → `shrink = n_samples / (n_samples + 8.0)`.
- Depends on / feeds into: Applied at line 472 to the rule's `confidence` field, which later must clear MIN_CONFIDENCE (0.75) to be promoted.

**tiny-mode confidence shrink center** — `cognition/uil_miner.py:472`
- Value: `0.5`
- Gates: The neutral point rule confidence is shrunk toward in tiny-data mode.
- Context: `confidence = 0.5 + (confidence - 0.5) * shrink`.
- Depends on / feeds into: Same rule["confidence"] compared against MIN_CONFIDENCE in uil_library.update_principles().


### `cognition/uil_stress.py` — 12 values

**failure_cost scale factor** — `cognition/uil_stress.py:77`
- Value: `5.0 (scaled), clipped to 1.0`
- Gates: Converts a raw accuracy drop into the "failure cost" component of decision stress.
- Context: "# Scale: 0.1 accuracy drop → 0.5 failure cost".
- Depends on / feeds into: Feeds decision_stress weighted sum (line 101-105).

**stage_recovery / cpu_recovery denominators** — `cognition/uil_stress.py:81-82`
- Value: `stages_since_best/10.0; cpu_time_since_best / max(total_cpu_time*0.5, 1.0)`
- Gates: How quickly "recovery cost" saturates to 1.0 as stagnation or CPU-time-without-progress accumulates.
- Context: "V20: Dual recovery cost (stages AND cpu time)".
- Depends on / feeds into: recovery_cost = 0.6*stage_recovery + 0.4*cpu_recovery (line 83).

**recovery_cost blend weights** — `cognition/uil_stress.py:83`
- Value: `0.6 (stage) / 0.4 (cpu)`
- Gates: Relative weight of stagnation-stages vs CPU-time-lost in the recovery-cost signal.
- Context: Same block as above.
- Depends on / feeds into: Feeds decision_stress.

**decision_stress weighted sum** — `cognition/uil_stress.py:101-105`
- Value: `0.40*failure_cost + 0.35*recovery_cost + 0.25*time_lost`
- Gates: The composite per-action Decision Stress value.
- Context: "# Weighted combination".
- Depends on / feeds into: Appended to _decision_stress_history; feeds update_system_stress()'s recent_stress signal.

**recent_stress rolling window** — `cognition/uil_stress.py:156`
- Value: `5`
- Gates: Window size for the rolling-average "recent decision stress" pressure signal.
- Context: `recent_n = min(5, len(self._decision_stress_history))`.
- Depends on / feeds into: Feeds current_pressure's 0.30-weighted recent_stress term.

**accuracy deficit target** — `cognition/uil_stress.py:164`
- Value: `0.95`
- Gates: The "target accuracy" from which System Stress's deficit pressure signal is measured.
- Context: "# Signal 2: Accuracy deficit from 95% target".
- Depends on / feeds into: deficit = max(0, 0.95-accuracy)/0.95, one of four pressure signals.

**stagnation saturation** — `cognition/uil_stress.py:167`
- Value: `8.0`
- Gates: Number of stagnant stages at which the "stagnation" pressure signal saturates to 1.0.
- Context: "# Signal 3: Stagnation".
- Depends on / feeds into: One of four pressure signals in current_pressure.

**current_pressure weighted sum** — `cognition/uil_stress.py:173-178`
- Value: `0.30*recent_stress + 0.25*deficit + 0.25*stagnation + 0.20*explanation_failure`
- Gates: How the four pressure signals combine into a single "current pressure" value driving stress compounding.
- Context: "# Current pressure (all signals combined)".
- Depends on / feeds into: Used in the compounding-vs-relief branch below.

**improvement relief threshold and rate** — `cognition/uil_stress.py:181,184`
- Value: `improvement > 0.01 gate; relief_rate = min(0.20, 0.05*(improvement/0.05))`
- Gates: Whether system stress decays this stage, and how fast (capped at 20% relief per stage).
- Context: "Relief: stress decays, but SLOWLY. Need sustained improvement to bring stress down."
- Depends on / feeds into: self.system_stress *= (1 - relief_rate).

**compounding rate** — `cognition/uil_stress.py:189-190`
- Value: `compound_factor = current_pressure*(1.0+system_stress*0.5); system_stress += compound_factor*0.08`
- Gates: How fast stress grows when there's no improvement — "the MORE stressed, the FASTER it grows".
- Context: "# Compounding: stress grows on itself".
- Depends on / feeds into: Core of the "debt" stress ratchet described in the module docstring.

**regret stress multiplier** — `cognition/uil_stress.py:218`
- Value: `1.0 + system_stress*2.0`
- Gates: How much system stress amplifies backward-looking Regret.
- Context: "Higher stress → higher regret multiplier → more aggressive investigation."
- Depends on / feeds into: compute_regret(); feeds build_investigation() trigger logic.

**has_failed() thresholds** — `cognition/uil_stress.py:277,296-299`
- Value: `system_stress>0.75; stages_since_best_improved>=patience(default 15); miner_r2<0.3`
- Gates: Whether the system declares genuine, terminal failure (all four conditions, including principles_exhausted, must hold).
- Context: "NOT a threshold check — a logical conclusion from multiple evidence."
- Depends on / feeds into: Consumed by whatever caller decides to reset()/abort a run.


### `cognition/version_store.py` — 3 values

**segment_size** — `cognition/version_store.py:124`
- Value: `2000`
- Gates: How many events are stored per rotated segment file before a new segment starts.
- Context: VersionStore.__init__ default.
- Depends on / feeds into: _segment_for_append(), events() range-skip logic.

**snapshot_every** — `cognition/version_store.py:125`
- Value: `500`
- Gates: How often (in events) a full composite snapshot is automatically written.
- Context: "Snapshots are written automatically every snapshot_every events."
- Depends on / feeds into: append(): `if seq % self.snapshot_every == 0: self.write_snapshot(...)`.

**compact() keep_recent / stride / verify_samples defaults** — `cognition/version_store.py:558-559,562`
- Value: `keep_recent=200, stride=20, verify_samples=32`
- Gates: Snapshot retention policy — which historical snapshots survive compaction (newest 200, every 20th older one) and how many deleted seqs are sampled to measure coarsening.
- Context: "Retention is keyed on seq, not list position, so running compact() twice deletes nothing new."
- Depends on / feeds into: _compaction_plan(); also duplicated as CLI argparse defaults (lines 680-686).


### `core/failsafe.py` — 10 values

**diagnostic train/val split** — `core/failsafe.py:59`
- Value: `test_size=0.2`
- Gates: Fraction of data held out to evaluate every adapter during failsafe diagnosis.
- Context: diagnose(), "Split for evaluation".
- Depends on / feeds into: X_train/X_val/y_train/y_val used across all adapters' fit_weighted/predict calls.

**classifier-mismatch margin** — `core/failsafe.py:99`
- Value: `0.03`
- Gates: Minimum accuracy improvement an alternate classifier must show over the primary before the diagnosis is classified "classifier_mismatch" rather than "fundamental_problem" — this is the correlation/threshold gate for switching classifiers.
- Context: `if results[best_alt_name]["accuracy"] > primary_score + 0.03`.
- Depends on / feeds into: diagnosis dict's "diagnosis"/"recommendation" fields, which downstream logic (outside these files) presumably acts on.

**correlation-feature subsample cap** — `core/failsafe.py:157`
- Value: `200 (n_features > 200 → sample 200)`
- Gates: Whether feature-correlation computation is done on the full feature set or a random 200-feature subsample.
- Context: `_compute_data_profile()`, "Feature correlation (sample if too many features)".
- Depends on / feeds into: profile["feature_correlation_mean"/"_max"], consumed by both diagnosis reason-generators.

**RandomForest-favoring correlation threshold** — `core/failsafe.py:219`
- Value: `0.4 (corr > 0.4)`
- Gates: Whether "high feature correlation favors RandomForest" is added as a diagnosis reason.
- Context: _classifier_mismatch_diagnosis().
- Depends on / feeds into: reasons list in the returned diagnosis dict.

**KNN-favoring thresholds** — `core/failsafe.py:224`
- Value: `sample_feature_ratio > 5 and n_features < 50`
- Gates: Whether "low dimensionality favors distance-based methods" reason is added when KNN is the best alternative.
- Context: same diagnosis function.
- Depends on / feeds into: reasons list.

**TabPFN-favoring threshold** — `core/failsafe.py:229`
- Value: `sample_feature_ratio > 2`
- Gates: Whether a TabPFN-favoring reason is added.
- Context: same diagnosis function.
- Depends on / feeds into: reasons list.

**XGBoost-overfitting threshold** — `core/failsafe.py:234`
- Value: `sample_feature_ratio < 2`
- Gates: Whether "low sample-feature ratio makes gradient boosting prone to overfitting" reason is added when XGBoost is the primary.
- Context: same diagnosis function.
- Depends on / feeds into: reasons list.

**low-ratio fundamental-problem threshold** — `core/failsafe.py:266`
- Value: `sample_feature_ratio < 1.0`
- Gates: Whether "more features than samples" is cited as a fundamental data-limitation reason.
- Context: _fundamental_problem_diagnosis().
- Depends on / feeds into: reasons list.

**high-sparsity threshold** — `core/failsafe.py:271`
- Value: `sparsity > 0.7`
- Gates: Whether "high sparsity limits information content" is cited as a reason.
- Context: same function.
- Depends on / feeds into: reasons list.

**high-correlation redundancy threshold** — `core/failsafe.py:276`
- Value: `corr > 0.7`
- Gates: Whether "features may be redundant" is cited as a reason (distinct, higher threshold than the 0.4 RandomForest-favoring one).
- Context: same function.
- Depends on / feeds into: reasons list.


### `core/normalization.py` — 2 values

**skew/kurtosis robust-scaler thresholds** — `core/normalization.py:111`
- Value: `abs(skew) > 1.5 or abs(kurtosis) > 3.0`
- Gates: Decision boundary for choosing "robust" (median/MAD) scaling over zscore/minmax for a feature.
- Context: "Choose scaler from axis-3 shape hints."
- Depends on / feeds into: _pick_method(), consumed by fit_normalization() per-feature transform selection.

**minmax-detection range** — `core/normalization.py:114`
- Value: `col_min >= 0.0 and col_max <= 1.0`
- Gates: Whether a feature already in [0,1] gets classified as "minmax" (identity-like) rather than "zscore".
- Context: _pick_method(), fallback branch after shape-based robust check.
- Depends on / feeds into: same _pick_method() decision, feeding transform construction in fit_normalization().


### `core/optimizer.py` — 55 values

**0.15 (held-out split fraction)** — `core/optimizer.py:429`
- Value: `test_size=0.15`
- Gates: Fraction of data carved off as a never-touched held-out set when the EXPERIMENTAL held-out principle gate (UIL_HELDOUT_GATE=1) is on.
- Context: In fit(), step "Validation split" — held-out set used only to judge whether principles generalize.
- Depends on / feeds into: self._X_heldout/self._y_heldout, later used for heldout_score and principle "helped" credit.

**cv_folds default = 3** — `core/optimizer.py:442`
- Value: `getattr(self.cfg, "cv_folds", 3)`
- Gates: Default number of resampled train/val splits used for multi-split (k-fold-like) regret/mining scoring when cfg doesn't specify cv_folds.
- Context: fit(), M0.1 extra-split setup ("k-fold regret re-scores the winner on every extra split").
- Depends on / feeds into: self._extra_val_splits, auto_cv_scale logic below, _score_extra_splits().

**20000 (auto CV scale sample threshold)** — `core/optimizer.py:451`
- Value: `n_samples > 20000 -> cv_folds = 1`
- Gates: Disables extra CV splitting entirely for large datasets to bound per-stage evaluation cost.
- Context: fit(), "Auto-scale CV cost by dataset size" block; comment notes covertype (100k rows) took >40 min CPU with cv_folds=3.
- Depends on / feeds into: cv_folds, self._extra_val_splits, self._observability_meta["cv_folds_autoscaled"].

**8000 (auto CV scale mid-size threshold)** — `core/optimizer.py:453`
- Value: `n_samples > 8000 -> cv_folds = min(cv_folds, 2)`
- Gates: Caps extra CV splits to 2 for medium-sized datasets.
- Context: fit(), same auto_cv_scale block as above.
- Depends on / feeds into: cv_folds, self._extra_val_splits.

**0.2 (extra-split test_size)** — `core/optimizer.py:470, core/optimizer.py:478`
- Value: `test_size=0.2`
- Gates: Size of each extra resampled train/val split built for multi-split scoring (M0.1).
- Context: fit(), building self._extra_val_splits (both stratified and unstratified fallback paths).
- Depends on / feeds into: self._extra_val_splits -> _score_extra_splits() -> fold_scores -> best_action_score.

**seed + 101 * (i + 1)** — `core/optimizer.py:471, core/optimizer.py:479`
- Value: `random_state=self.seed + 101 * (_i + 1)`
- Gates: Deterministic-but-distinct RNG seed offset for each extra validation split so splits differ from each other and from the primary split.
- Context: fit(), extra-split construction loop (M0.1).
- Depends on / feeds into: reproducibility of self._extra_val_splits.

**boosting_rounds default = 30** — `core/optimizer.py:546, core/optimizer.py:559, core/optimizer.py:660`
- Value: ``
- Gates: Default number of boosting rounds passed to build_classifier_for_strategy / resource-budget base, when cfg.boosting_rounds is unset.
- Context: fit(), routed-classifier build and resource-budget detection.
- Depends on / feeds into: build_classifier_for_strategy(), detect_resource_budget() -> budget.boosting_stages.

**0.05 (auto_rule_basis interaction-gap threshold)** — `core/optimizer.py:598`
- Value: `gap >= 0.05`
- Gates: Whether the interaction-aware rule basis (E5) auto-enables based on the fingerprint's nonlinear-vs-linear probe gap.
- Context: fit(), guarded by opt-in cfg.auto_rule_basis; long comment explains a 3-seed sweep showed "No clean gap threshold separates 'helps' from 'hurts'" and suggests raising the gate "above ~0.08" before re-enabling.
- Depends on / feeds into: classifier.apply_config({"use_rule_basis": True, ...}), self._observability_meta["rule_basis_enabled"].

**0.5 (sparse_l1_c floor when rule basis enabled)** — `core/optimizer.py:604-606`
- Value: `sparse_l1_c=max(float(getattr(self.classifier, "sparse_l1_c", 0.1)), 0.5)`
- Gates: Minimum L1 regularization strength forced when auto rule-basis triggers.
- Context: fit(), same auto_rule_basis block as above; classifier default of 0.1 also hardcoded here.
- Depends on / feeds into: classifier.apply_config().

**n_particles/pso_iters resource-budget base = 15/15** — `core/optimizer.py:658-659`
- Value: `base_particles=getattr(self.cfg, "n_particles", 15), base_pso_iters=getattr(self.cfg, "pso_iters", 15)`
- Gates: Baseline swarm size/iteration count fed into resource-budget auto-scaling (before scaling to ~75% headroom).
- Context: fit(), "Scale workload to ~75% of machine CPU/RAM/GPU headroom" block.
- Depends on / feeds into: detect_resource_budget() -> self.cfg.n_particles / self.cfg.pso_iters.

**resource_fraction default = 0.75** — `core/optimizer.py:661`
- Value: `target_fraction=getattr(self.cfg, "resource_fraction", 0.75)`
- Gates: Target fraction of machine CPU/RAM/GPU the workload scales to use.
- Context: fit(), resource-budget detection block; comment "Scale workload to ~75% of machine CPU/RAM/GPU headroom".
- Depends on / feeds into: detect_resource_budget(), self._resource_budget.

**n_particles/pso_iters search-init defaults = 20/30** — `core/optimizer.py:691-692`
- Value: `n_particles=getattr(self.cfg, "n_particles", 20), pso_iters=getattr(self.cfg, "pso_iters", 30)`
- Gates: Default PSO swarm size and iteration count used to initialize the search strategy when not overridden by cfg or the resource budget.
- Context: fit(), self.search.initialize(...) call.
- Depends on / feeds into: search/pso_search.py (or PrincipleGuidedSearch) swarm behavior.

**seed + 7 (bootstrap-diversity RNG offset)** — `core/optimizer.py:698`
- Value: `np.random.default_rng(self.seed + 7)`
- Gates: Deterministic RNG stream used for search bootstrap diversity, offset from the base seed.
- Context: fit(), set_bootstrap_diversity() call right after search.initialize().
- Depends on / feeds into: search bootstrap-diversity mechanism.

**prior_stress > 0** — `core/optimizer.py:709`
- Value: `if prior_stress > 0`
- Gates: Whether the stress engine is seeded with the dataset's remembered prior stress at run start.
- Context: fit(), "Check dataset memory — start with prior stress if dataset was hard".
- Depends on / feeds into: self.journal.get_dataset_prior_stress(), self.stress_engine.set_initial_stress().

**6000 / 4000 / 300 (working-graph load-time prune thresholds)** — `core/optimizer.py:765-770`
- Value: `if len(self.network._edges) > 6000: prune_working_graph(max_edges=4000, max_nodes=300, ...)`
- Gates: Whether/how much to prune the shared knowledge-graph working set at load time before the run starts.
- Context: fit(), "Load-time guard" comment: "short runs never reach the stage%15 graph prune... observed 49k edges... make every traverse pathologically slow."
- Depends on / feeds into: self.network.prune_working_graph(), self.knowledge_archive.

**0.15 (co-activation confidence-bonus multiplier)** — `core/optimizer.py:922`
- Value: `p.get("confidence", 0.5) + _co_bonuses[...] * 0.15, capped at min(0.99, ...)`
- Gates: How much a principle's effective confidence is temporarily boosted this stage based on co-activation bonus with a principle that fired last stage.
- Context: fit() stage loop, "Part 16: Propagate co-activation attention from last stage's triggered principles."
- Depends on / feeds into: self.network.get_co_activation_bonus(), self.executor.get_overrides().

**0.99 (principle confidence cap)** — `core/optimizer.py:922`
- Value: `min(0.99, ...)`
- Gates: Upper cap on a principle's boosted confidence value before being passed to the executor.
- Context: fit() stage loop, same co-activation bonus block as above.
- Depends on / feeds into: executor.get_overrides() decisions based on confidence.

**0.5 (default principle confidence fallback)** — `core/optimizer.py:922`
- Value: `p.get("confidence", 0.5)`
- Gates: Fallback confidence value used if a principle dict lacks a "confidence" key, before the co-activation bonus is applied.
- Context: fit() stage loop, co-activation bonus block.
- Depends on / feeds into: same as above.

**1.0 (default/fallback action penalty)** — `core/optimizer.py:1028, core/optimizer.py:1044`
- Value: `"penalty": 1.0`
- Gates: Default penalty value used for the synthetic incumbent/fallback action when no real actions exist yet.
- Context: fit(), architecture-race incumbent fallback (1028) and "Fallback base guidance" when search proposes nothing (1044).
- Depends on / feeds into: action["penalty"] used by _evaluate_one_action / classifier fit_weighted.

**race_max_arms default = 12** — `core/optimizer.py:1037`
- Value: `max_arms=int(getattr(self.cfg, "race_max_arms", 12))`
- Gates: Maximum number of architecture-race arms proposed per stage when cfg.search_mode == "race".
- Context: fit(), architecture-race action proposal block (self._race.propose()).
- Depends on / feeds into: ArchitectureRace.propose(), downstream _race_actions() elimination rungs.

**0.3 (stress gate for custom-objective formula selection)** — `core/optimizer.py:1060`
- Value: `stress_level >= 0.3`
- Gates: Whether the optimizer bothers selecting/trying the formula library's best alternative loss function this stage.
- Context: fit() stage loop, comment: "Stress-governed like every other exploration knob here: a calm run keeps the classifier's default loss, only a stressed run spends effort trying the library's best alternative."
- Depends on / feeds into: self.formula_library.get_best_formula(), self._current_formula_id, classifier.set_custom_objective().

**0.5 (default causal-intervention scale factor)** — `core/optimizer.py:1126`
- Value: `scale = plan["treatment"].get("scale") or 0.5`
- Gates: Default multiplier applied to a derived variable's current live value to compute the treatment value when a causal intervention plan doesn't specify a scale.
- Context: fit(), V20+.5 causal intervention block, building control/treatment actions.
- Depends on / feeds into: self._set_intervention_value(), self.causal_engine.evaluate_intervention().

**0.5 (entropy-diversity normalization divisor)** — `core/optimizer.py:1545`
- Value: `entropy=np.clip(diversity / 0.5, 0.0, 1.0)`
- Gates: Normalizes the mean absolute deviation of candidate feature weights into a [0,1] "entropy" signal fed back to the search strategy.
- Context: fit() stage loop, step 11 "Update search strategy with feedback".
- Depends on / feeds into: self.search.update_feedback(entropy=...).

**5 (mining-interval base stage count)** — `core/optimizer.py:1551`
- Value: `mine_interval = max(1, int(5 * (1.0 - system_stress)))`
- Gates: How often (in stages) the journal is mined for new principles — mining frequency scales inversely with stress (base cadence of 5 stages when stress is 0).
- Context: fit() step 12, "Mine for principles (frequency scales with stress)".
- Depends on / feeds into: self.miner.mine_within_dataset()/mine_cross_dataset(), self.library.update_principles().

**50 (minimum journal entries for cross-dataset mining)** — `core/optimizer.py:1589`
- Value: `if len(mining_entries) >= 50`
- Gates: Whether cross-dataset (correlational) rule mining runs at all this stage.
- Context: fit(), "Cross-dataset mining (correlational, needs more evidence)".
- Depends on / feeds into: self.miner.mine_cross_dataset(), cross_rules -> self.library.update_principles().

**0.75 (cross-dataset rule confidence discount)** — `core/optimizer.py:1599`
- Value: `_r["confidence"] = round(_r.get("confidence", 0.5) * 0.75, 4)`
- Gates: Discounts confidence of cross-dataset-mined rules before they enter the principle library.
- Context: fit(), "Part 5: Cross-dataset rules represent weaker causal evidence — discount confidence before library update so they must prove themselves further to affect principle selection over within-dataset rules."
- Depends on / feeds into: self.library.update_principles(cross_rules, ...).

**0.5 (default confidence used in cross-rule discount)** — `core/optimizer.py:1599`
- Value: `_r.get("confidence", 0.5)`
- Gates: Fallback confidence for a cross-dataset rule lacking one, before the 0.75 discount is applied.
- Context: Same block as above.
- Depends on / feeds into: same as above.

**5 (activation-history minimum length for relationship discovery)** — `core/optimizer.py:1727`
- Value: `if self.ablation["network"] and len(_activation_history) >= 5`
- Gates: Whether the network attempts to discover relationships/edges from co-activation history this stage.
- Context: fit(), "V20+.3: Discover relationships from co-activation patterns."
- Depends on / feeds into: self.network.discover_relationships().

**1.0 (default regret fallback for base-rate calc)** — `core/optimizer.py:1749`
- Value: `e.get("regret", 1.0)`
- Gates: Fallback regret value (treated as "no improvement") for journal entries missing a regret field, used to compute the fold's success base rate.
- Context: fit(), base-rate computation for network edge-type thresholds ("lift-over-base-rate rather than absolute").
- Depends on / feeds into: self.network.discover_relationships(base_rate=...).

**-25 (activation-history window for relationship discovery)** — `core/optimizer.py:1756`
- Value: `_activation_history[-25:]`
- Gates: How many recent stages' activation history are passed into discover_relationships().
- Context: fit(), same relationship-discovery block.
- Depends on / feeds into: self.network.discover_relationships().

**10 (edge-decay/graph-prune cadence, every 10 stages)** — `core/optimizer.py:1793`
- Value: `if stage > 0 and stage % 10 == 0`
- Gates: Cadence (every 10 stages) at which network edge decay + prune runs.
- Context: fit(), "Periodic edge decay + working-graph prune (every 10 stages)".
- Depends on / feeds into: self.network.apply_edge_decay(min_traversals=5, ...).

**5 (min_traversals for edge decay)** — `core/optimizer.py:1795`
- Value: `min_traversals=5`
- Gates: Minimum traversal count an edge must have before it's eligible for decay pruning.
- Context: fit(), same periodic edge-decay block.
- Depends on / feeds into: self.network.apply_edge_decay().

**5 (formula/graph prune cadence, every 5 stages)** — `core/optimizer.py:1804, core/optimizer.py:1819`
- Value: `if stage > 0 and stage % 5 == 0`
- Gates: Cadence at which formula libraries get pruned-to-cap/retired-to-archive, and at which the working graph gets size-triggered pruning.
- Context: fit(), "Learning-preserving formula prune on a SHORT cadence so the accelerated evolution from stagnation-triggered births ... can't bloat the working pool"; and "Size-triggered graph prune ... on the same short cadence as the formula prune."
- Depends on / feeds into: formula_library/symbolic_library.prune_retired_to_archive()/prune_to_cap(), self.network.prune_working_graph().

**6000 / 4000 / 300 (stage-time size-triggered graph prune thresholds)** — `core/optimizer.py:1819, core/optimizer.py:1825-1826`
- Value: `len(self.network._edges) > 6000 -> prune_working_graph(max_edges=4000, max_nodes=300, ...)`
- Gates: Whether/how much the working knowledge graph is pruned mid-run when it grows oversized.
- Context: fit(), "Size-triggered graph prune" comment referencing the observed 49k-edge pathological slowdown.
- Depends on / feeds into: self.network.prune_working_graph(), self.knowledge_archive.

**-12 (activation-history window for "recent" node protection)** — `core/optimizer.py:1821, core/optimizer.py:1833`
- Value: `_activation_history[-12:]`
- Gates: Which recently-activated nodes are protected (exempted) from graph pruning.
- Context: fit(), size-triggered prune block (1821) and the stage%15 consolidate/prune block (1833).
- Depends on / feeds into: self.network.prune_working_graph(protect_nodes=...).

**15 (consolidation/full-prune cadence, every 15 stages)** — `core/optimizer.py:1831`
- Value: `if stage > 0 and stage % 15 == 0`
- Gates: Cadence at which duplicate-node consolidation, full graph prune, formula retirement archiving, and working-memory pruning all run together.
- Context: fit(), the "stage%15 graph prune" cadence referenced elsewhere in comments (line 761-762) as the baseline cadence that short runs never reach.
- Depends on / feeds into: self.network.consolidate_duplicate_nodes(), prune_working_graph(), self.working_memory_pruner.prune_all().

**6000 / 4000 / 300 (stage%15 prune thresholds, second call site)** — `core/optimizer.py:1838-1841`
- Value: `prune_working_graph(max_edges=4000, max_nodes=300, ...) — unconditional at this cadence (no size check)`
- Gates: Target cap for graph size at the 15-stage full-consolidation cadence.
- Context: fit(), same stage%15 block as above.
- Depends on / feeds into: self.network.prune_working_graph().

**5 (minimum principle count for post-run structural pipeline)** — `core/optimizer.py:2079`
- Value: `len(all_principles) >= 5`
- Gates: Whether the structural-encoding / meta-principle-merge / safeguards pipeline runs at all at the end of fit().
- Context: fit(), "V21 Pipeline: Structural → Meta → Safeguards (skip when target hit — post-run merge on a 5k-edge KG can block the worker for minutes)."
- Depends on / feeds into: self.structural_encoder.batch_encode(), self.meta_merger.merge(), self.safeguards.check_promotion().

**0.05 / 30th percentile (feature-selection threshold)** — `core/optimizer.py:2116, core/optimizer.py:2124`
- Value: `threshold = max(0.05, np.percentile(self._best_feature_weights, 30))`
- Gates: Which features are included in the final "selected_features" output — a feature must exceed the larger of a 0.05 floor or the 30th-percentile weight.
- Context: fit(), end-of-run "Feature selection: PSO weights + formula tree leaves" block (both the formula-adapter branch and the plain-PSO branch use the same threshold formula).
- Depends on / feeds into: result["selected_features"], result["n_selected"].

**0.5 (increment for KG-condition feature index match)** — `core/optimizer.py:2483`
- Value: `emphasis[idx] += 0.5`
- Gates: How strongly an activated knowledge-graph node's numeric-feature condition boosts that feature's tournament emphasis weight.
- Context: _graph_feature_emphasis(), mapping KG activations to per-feature tournament weights.
- Depends on / feeds into: classifier.set_graph_hint({"feature_emphasis": ...}).

**1e-9 (minimum-weight-signal threshold)** — `core/optimizer.py:2486`
- Value: `if w.size == n_features and np.max(w) > 1e-9`
- Gates: Whether best-known feature weights are considered "meaningful" enough to blend into the graph feature-emphasis vector.
- Context: _graph_feature_emphasis(), same feature-emphasis computation.
- Depends on / feeds into: emphasis vector fed to classifier.set_graph_hint().

**0.35 (feature-weight blend coefficient in graph emphasis)** — `core/optimizer.py:2487`
- Value: `emphasis += 0.35 * (w / (w.sum() + 1e-12))`
- Gates: How much the best-known PSO feature weights contribute (relative to KG-activation-derived emphasis) to the final per-feature tournament emphasis vector.
- Context: _graph_feature_emphasis(), same block.
- Depends on / feeds into: classifier.set_graph_hint({"feature_emphasis": ...}).

**0.2 (primary train/val split fraction)** — `core/optimizer.py:2496`
- Value: `test_size=0.2`
- Gates: Size of the primary validation split used across the whole run (or per-fold pool before rotation).
- Context: _split(), the base train/val split helper called at fit() step 0.
- Depends on / feeds into: X_train/X_val/y_train/y_val used throughout the stage loop.

**0.2 (rotating within-stage split fraction)** — `core/optimizer.py:2511, core/optimizer.py:2517`
- Value: `test_size=0.2`
- Gates: Size of the per-stage rotating validation split (cfg.rotate_val_split) drawn fresh each stage from seed+stage.
- Context: _stage_split(), "Row INDICES for this stage's train/val split (rotating)".
- Depends on / feeds into: X_train/X_val for that stage's action scoring.

**n_neighbors = 5 (mutual information estimator)** — `core/optimizer.py:2525`
- Value: `mutual_info_classif(X, y, random_state=self.seed, n_neighbors=5)`
- Gates: k-nearest-neighbor parameter shaping the base guidance (mutual-information feature relevance) that seeds the whole search.
- Context: _compute_mutual_info(), computed once at fit() start.
- Depends on / feeds into: self._base_guidance, used as the search's initial feature-weight guidance and PSO/race fallback actions.

**0.1 (selected-feature threshold for state's feat_ratio)** — `core/optimizer.py:2550`
- Value: `selected = np.sum(self._best_feature_weights > 0.1)`
- Gates: Which features count as "selected" when computing the feat_ratio component of the state vector fed to the Miner/search.
- Context: _build_state(), "Entropy from search diversity" section.
- Depends on / feeds into: state["feat_ratio"], later mined against by the Miner and read by search/executor.

**100 (feature correlation subsampling threshold/size)** — `core/optimizer.py:2560-2562`
- Value: `if n_features > 100: sample 100 features via rng.choice(n_features, 100, replace=False)`
- Gates: Whether/how the feature correlation matrix used in the state vector is computed on a subsample (for speed) instead of all features.
- Context: _build_state(), "Feature correlation (sample for speed)".
- Depends on / feeds into: state["feature_correlation_mean"].

**1e-6 (step-out-of-region epsilon)** — `core/optimizer.py:3014`
- Value: `eps = 1e-6`
- Gates: How far a continuous config knob is nudged past a mined "bad region" boundary when escaping it (float case; int case uses floor-1/ceil+1 instead).
- Context: _step_out_of_region(), "Move a knob just outside an interval a mined rule marked bad ... leave by the NEARER edge, which is the smallest change that satisfies the rule."
- Depends on / feeds into: _resolve_config_bounds() -> config_avoid.* overrides -> action["classifier_config"].

**3.0 (default penalty when applying max_penalty override)** — `core/optimizer.py:3130`
- Value: `action.get("penalty", 3.0)`
- Gates: Fallback penalty value used if the action has none, when capping penalty via a "max_penalty" principle override.
- Context: _apply_overrides(), "Cap penalty if principles say so".
- Depends on / feeds into: action["penalty"] -> classifier fit_weighted.

**0.0 (default penalty when applying min_penalty override)** — `core/optimizer.py:3134`
- Value: `action.get("penalty", 0.0)`
- Gates: Fallback penalty value used if the action has none, when flooring penalty via a "min_penalty" principle override.
- Context: _apply_overrides(), same penalty-cap block.
- Depends on / feeds into: action["penalty"].

**3.0 (default penalty for penalty_avoid override)** — `core/optimizer.py:3144`
- Value: `float(action.get("penalty", 3.0))`
- Gates: Fallback penalty value used when resolving a "penalty_avoid" bounded-region override via _step_out_of_region.
- Context: _apply_overrides(), "Same interval semantics as config_avoid.*: a tree path that bounds penalty on BOTH sides condemns a band, not a direction."
- Depends on / feeds into: action["penalty"].

**0.0 - 10.0 (penalty actuator clamp range)** — `core/optimizer.py:3154`
- Value: `action["penalty"] = float(np.clip(action["penalty"], 0.0, 10.0))`
- Gates: Hard bounds on the penalty knob actually applied to the classifier, regardless of what upstream cognition (network edges, overrides) proposed.
- Context: _apply_overrides(), "Actuator boundary: penalty is a bounded control knob (PSO and reasoning propose 0.5-3.0). Cognition upstream ... may propose anything; the actuator clamps."
- Depends on / feeds into: classifier.fit_weighted(), journal recording, principle mining of "action.penalty".

**6 (candidate mutation count for neural ranking)** — `core/optimizer.py:3218`
- Value: `for _ in range(6): candidates.append(engine.mutate(...))`
- Gates: How many grammar-safe candidate mutations are generated for the neural proposer to rank and pick from, when mutating a SymbolicFormula-classifier tree.
- Context: _neural_pick_classifier_mutation(), "Generates several grammar-safe candidate mutations and lets the offline own-transformer ... pick the highest-scored."
- Depends on / feeds into: neural.rank_candidates(), symbolic_library.add_formula().

**floor = 3, divisor = 8 (adaptive minimum-evidence formula)** — `core/optimizer.py:3263`
- Value: `return max(3, min(UIL_Library.MIN_EVIDENCE, n_entries // 8))`
- Gates: Minimum evidence-count (leaf size) a mined rule needs to be promoted to a principle, when cfg.adaptive_min_evidence is on — scales with journal size but is floored at 3 and never exceeds the fixed MIN_EVIDENCE bar.
- Context: _promotion_min_evidence(); long comment with measured leaf-size statistics ("largest leaf holds 4 rows at 16 journal entries, 9 at 100, and only at ~200 does anything clear 10").
- Depends on / feeds into: self.library.update_principles(min_evidence=...), gated behind cfg.adaptive_min_evidence (default off).

**2 (stagnation-fire threshold, stages since improvement)** — `core/optimizer.py:3316`
- Value: `stagnation_fire = getattr(self.cfg, "enable_stagnation_evolution", False) and self._stages_since_improvement >= 2`
- Gates: Whether stagnation (no accuracy improvement for >=2 stages) triggers formula/structure evolution, when cfg.enable_stagnation_evolution is on.
- Context: _want_new_structure(); comment: "Stagnation is powerful but destabilizing: a single birth can trigger a pathological slowdown ... It is therefore OPT-IN via cfg.enable_stagnation_evolution (default off)."
- Depends on / feeds into: self._stages_since_improvement, gated further by EVOLUTION_MIN_GAP rate limiter.


### `core/resource_budget.py` — 14 values

**TARGET_FRACTION (definition site)** — `core/resource_budget.py:10`
- Value: `0.75`
- Gates: Module-level default utilization target (known example — noted here as the definition site feeding every default below).
- Context: module docstring: "scale UIL workload to ~75% of available capacity."
- Depends on / feeds into: default `target_fraction` arg of detect_resource_budget() (line 122).

**base_particles / base_pso_iters / base_boosting defaults** — `core/resource_budget.py:119-121`
- Value: `base_particles=15, base_pso_iters=15, base_boosting=30`
- Gates: Baseline workload sizes before machine-based scaling is applied.
- Context: detect_resource_budget() signature.
- Depends on / feeds into: n_particles/pso_iters/boosting computed at lines 140-142 via np_clip_round(base*scale, ...).

**frac clip range** — `core/resource_budget.py:125`
- Value: `clip(target_fraction, 0.25, 0.95)`
- Gates: Bounds the utilization fraction the caller can request, regardless of target_fraction argument.
- Context: first line of detect_resource_budget() body.
- Depends on / feeds into: cpu_workers, ram_budget, gpu_budget_mb all derive from `frac`.

**mem_factor formula** — `core/resource_budget.py:135`
- Value: `min(4.0, ram_budget / 3072.0)`
- Gates: RAM-derived intensity multiplier, capped at 4x, using 3GB as the "baseline laptop" reference.
- Context: "Scale intensity with headroom (4 GB RAM + 4 cores = baseline laptop)."
- Depends on / feeds into: `scale` at line 138.

**cpu_factor formula** — `core/resource_budget.py:136`
- Value: `min(4.0, cpu_workers / 3.0)`
- Gates: CPU-derived intensity multiplier, capped at 4x, using 3 workers as baseline.
- Context: same comment block.
- Depends on / feeds into: `scale` at line 138.

**gpu_factor** — `core/resource_budget.py:137`
- Value: `1.25 if gpu_ok else 1.0`
- Gates: 25% workload boost when a GPU is detected.
- Context: same comment block.
- Depends on / feeds into: `scale` at line 138.

**scale cap** — `core/resource_budget.py:138`
- Value: `max(1.0, min(4.0, mem_factor*cpu_factor*gpu_factor))`
- Gates: Overall workload multiplier, floored at 1x and capped at 4x regardless of how much headroom is detected.
- Context: combines the three factors above.
- Depends on / feeds into: n_particles, pso_iters, boosting, bagging (lines 140-143).

**n_particles bounds** — `core/resource_budget.py:140`
- Value: `clip_round(base_particles*scale, 8, 120)`
- Gates: Final PSO swarm size, hard-bounded to [8,120] regardless of scale.
- Context: resource-scaled PSO knob.
- Depends on / feeds into: passed to PSOSearch.initialize(n_particles=...) elsewhere in the optimizer.

**pso_iters bounds** — `core/resource_budget.py:141`
- Value: `clip_round(base_pso_iters*scale, 10, 80)`
- Gates: Final PSO iteration count, bounded to [10,80].
- Context: same scaling block.
- Depends on / feeds into: PSOSearch.initialize(pso_iters=...).

**boosting_stages bounds** — `core/resource_budget.py:142`
- Value: `clip_round(base_boosting*scale, 5, 80)`
- Gates: Final boosting-stage count, bounded to [5,80].
- Context: same scaling block.
- Depends on / feeds into: apply_budget_to_classifier() → classifier.apply_config({"boosting_stages": ...}).

**bagging_estimators formula and bounds** — `core/resource_budget.py:143`
- Value: `clip_round(max(3, base_boosting // 6) * scale, 3, 24)`
- Gates: Final bagging-estimator count, derived as 1/6 of base_boosting then scaled, bounded to [3,24].
- Context: same scaling block.
- Depends on / feeds into: apply_budget_to_classifier() → classifier.apply_config({"bagging_estimators": ...}).

**tournament_k formula and bounds** — `core/resource_budget.py:144`
- Value: `clip_round(4 + cpu_workers, 4, 16)`
- Gates: Final tournament_k, derived from CPU worker count and bounded to [4,16] — a different derivation than the adapter's own default of 8.
- Context: same scaling block.
- Depends on / feeds into: apply_budget_to_classifier() → classifier.apply_config({"tournament_k": ...}) → SymbolicFormulaAdapter.tournament_k.

**multi_variants gate** — `core/resource_budget.py:145`
- Value: `5 if cpu_workers>=6 and ram_budget>=6144 else 4`
- Gates: hybrid_branch_count/multi_variants selection based on hardware headroom thresholds (6 workers, 6GB RAM).
- Context: sets both `multi_variants` and `hybrid_branch_count` return fields.
- Depends on / feeds into: apply_budget_to_classifier() → classifier.apply_config({"hybrid_branch_count": ...}).

**parallel_eval thread cap** — `core/resource_budget.py:150`
- Value: `min(8, cpu_workers - 1 if cpu_logical > 2 else cpu_workers)`
- Gates: Hard cap of 8 in-process worker threads to avoid BLAS/OpenMP oversubscription freezing the process.
- Context: "Cap at 8: beyond that Python threads oversubscribe BLAS/OpenMP and freeze the whole process (including the web UI)."
- Depends on / feeds into: parallel_eval_workers / intra_fit_threads, and apply_budget_to_classifier() → classifier.apply_config({"parallel_eval_workers": ...}).


### `core/strategy_classifiers.py` — 5 values

**fewshot boosting_rounds override** — `core/strategy_classifiers.py:32`
- Value: `6`
- Gates: Overrides boosting_rounds specifically for the "fewshot" strategy's SymbolicFormula instantiation.
- Context: STRATEGY_KWARGS dict.
- Depends on / feeds into: extra.get("boosting_rounds", boosting_rounds) at line 63.

**build_classifier_for_strategy boosting_rounds default** — `core/strategy_classifiers.py:43`
- Value: `30`
- Gates: Default boosting-round count for any strategy not overridden by STRATEGY_KWARGS.
- Context: function signature.
- Depends on / feeds into: passed into XGBoostAdapter/SymbolicFormulaAdapter constructors.

**XGBoost max_depth default** — `core/strategy_classifiers.py:52`
- Value: `6`
- Gates: Default XGBoost tree depth for any strategy lacking an explicit max_depth in STRATEGY_KWARGS.
- Context: `extra.get("max_depth", 6)` inside the XGBoost factory lambda.
- Depends on / feeds into: XGBoostAdapter(max_depth=...).

**XGBoost learning_rate default** — `core/strategy_classifiers.py:53`
- Value: `0.1`
- Gates: Default XGBoost learning rate for any strategy lacking an explicit override.
- Context: `extra.get("learning_rate", 0.1)`.
- Depends on / feeds into: XGBoostAdapter(learning_rate=...).

**KNN n_neighbors default** — `core/strategy_classifiers.py:57`
- Value: `5`
- Gates: Default number of neighbors for KNN when a strategy routes there without an explicit override.
- Context: `extra.get("n_neighbors", 5)`.
- Depends on / feeds into: KNNAdapter(n_neighbors=...).


### `search/architecture_race.py` — 13 values

**RACE_MIN_ROWS** — `search/architecture_race.py:59`
- Value: `300`
- Gates: Below this many training rows, rungs() returns [] (no elimination racing, single full-fidelity round).
- Context: "Below this many training rows a subsampled rung is pure noise (sonar is n=208)".
- Depends on / feeds into: guard in rungs() line 174.

**RUNG_MIN_ROWS** — `search/architecture_race.py:62`
- Value: `150`
- Gates: Floor on subsample size for any elimination rung.
- Context: "Never subsample below this many rows on an elimination rung."
- Depends on / feeds into: `rows = int(max(RUNG_MIN_ROWS, round(n_train*frac)))` in rungs() line 181.

**MAX_RUNG_FRACTION** — `search/architecture_race.py:68`
- Value: `0.6`
- Gates: A rung must be ≤60% of the training set or the schedule breaks (rung would cost as much as the full fit).
- Context: comment block explaining rung-viability guards, cites n=480 producing 150/192-row rungs as the motivating failure.
- Depends on / feeds into: `if rows > n_train * MAX_RUNG_FRACTION: break` in rungs() line 187.

**MIN_RUNG_GROWTH** — `search/architecture_race.py:69`
- Value: `1.5`
- Gates: Each successive rung must be ≥1.5x the row-count of the previous rung, or it's skipped.
- Context: same comment block, "genuinely MORE INFORMATIVE than the rung before it".
- Depends on / feeds into: `if prev_rows and rows < prev_rows * MIN_RUNG_GROWTH: continue` at line 189.

**DEFAULT_RUNGS schedule** — `search/architecture_race.py:73`
- Value: `((0.15, 0.5), (0.40, 0.5))`
- Gates: Data-fraction and survivor-fraction for each of the two elimination rungs (15%→keep 50%, 40%→keep 50%); final full-data rung is implicit.
- Context: "The final full-data rung is implicit and always runs."
- Depends on / feeds into: default `schedule` argument of rungs().

**MIN_ARMS_FOR_RACE** — `search/architecture_race.py:76`
- Value: `4`
- Gates: Below 4 candidate arms, rungs() returns [] — "nothing to eliminate."
- Context: comment on that line.
- Depends on / feeds into: guard in rungs() line 174.

**MIN_FINAL_ARMS** — `search/architecture_race.py:79`
- Value: `2`
- Gates: Floor on how many arms survive into the full-fidelity rung; racing stops once reached.
- Context: "Arms surviving into the final full-fidelity rung."
- Depends on / feeds into: `keep = int(max(MIN_FINAL_ARMS, ceil(survivors*keep_frac)))` line 191, and the `survivors <= MIN_FINAL_ARMS: break` at line 197.

**PHASEOUT_LADDER** — `search/architecture_race.py:84-90`
- Value: `((0,12), (20,8), (60,5), (150,3), (400,2))`
- Gates: How many arms to race (12→8→5→3→2) as a function of accumulated per-bucket trial evidence.
- Context: "As evidence accumulates the system proposes fewer candidates -- it 'becomes an expert' by needing fewer trials".
- Depends on / feeds into: ArmMemory.arm_budget() at lines 278-285.

**frontier_masks n_masks default** — `search/architecture_race.py:98`
- Value: `3`
- Gates: Default number of cutoff points sampled across the closed-form guidance frontier.
- Context: "We return n_masks of them spread across the frontier".
- Depends on / feeds into: ArchitectureRace.propose()'s n_frontier default (line 374) calls this.

**frontier keep-count starting point** — `search/architecture_race.py:139`
- Value: `max_keep // 4`
- Gates: The lower bound of the linspace sweep across cutoff keep-counts — never starts the frontier below a quarter of features.
- Context: "Keep-counts spread across the frontier, never emptying the mask."
- Depends on / feeds into: `np.linspace(max(1, max_keep // 4), max_keep, remaining + 1)`.

**ArmMemory Beta(1,1) prior** — `search/architecture_race.py:262`
- Value: `(wins + 1.0) / (trials + 2.0)`
- Gates: Posterior win-probability estimate for an unseen arm defaults to exactly 0.5 (no false confidence).
- Context: "Beta(1,1) prior => 0.5 for an unseen arm".
- Depends on / feeds into: used to rank/sort race arms in ArchitectureRace.propose() line 444.

**propose() max_arms default** — `search/architecture_race.py:373`
- Value: `12`
- Gates: Ceiling on candidate arms generated per stage before ArmMemory.arm_budget() further narrows it via PHASEOUT_LADDER.
- Context: function signature default.
- Depends on / feeds into: `budget = self.memory.arm_budget(bucket, max_arms)` line 384.

**propose() n_frontier default** — `search/architecture_race.py:374`
- Value: `3`
- Gates: How many closed-form cutoff arms are added to the race candidates per stage.
- Context: function signature default, passed to frontier_masks().
- Depends on / feeds into: frontier arm generation loop at line 424.


### `search/hybrid_foundation.py` — 33 values

**INFANT_THRESHOLD** — `search/hybrid_foundation.py:37`
- Value: `0.2`
- Gates: Maturity score boundary below which the "infant" stage (PSO dominates) is selected.
- Context: "# Maturity thresholds ... Below this: PSO dominates".
- Depends on / feeds into: allocate_actions() stage classification (line 205) and get_foundation_targets() branch.

**ADOLESCENT_THRESHOLD** — `search/hybrid_foundation.py:38`
- Value: `0.5`
- Gates: Maturity boundary between "infant" and "adolescent" stages ("PSO and reasoning share equally").
- Context: same threshold block.
- Depends on / feeds into: allocate_actions() line 207.

**MATURE_THRESHOLD** — `search/hybrid_foundation.py:39`
- Value: `0.8`
- Gates: Maturity boundary above which stage becomes "expert" (ZERO PSO); also caps maturity when skill evidence is insufficient/weak.
- Context: "Above MATURE: Expert — ZERO PSO, pure cognition".
- Depends on / feeds into: exploration_ratio property (line 179), allocate_actions() (line 209), maturity caps at lines 159/161.

**SKILL_WINDOW** — `search/hybrid_foundation.py:45`
- Value: `30`
- Gates: Rolling window size (in principle applications) used to compute principle_hit_rate.
- Context: "M0.4 ... detachment is earned by demonstrated prediction skill".
- Depends on / feeds into: record_principle_outcome() truncates _principle_outcomes to this length (line 66).

**MIN_SKILL_EVIDENCE** — `search/hybrid_foundation.py:46`
- Value: `10`
- Gates: Minimum number of principle-application outcomes needed before principle_hit_rate returns a value instead of None.
- Context: same M0.4 comment block.
- Depends on / feeds into: principle_hit_rate property (line 71), gates the skill-based vs legacy maturity blend (line 142).

**SKILL_GATE** — `search/hybrid_foundation.py:47`
- Value: `0.6`
- Gates: Hit-rate floor below which maturity can never reach MATURE_THRESHOLD, regardless of other evidence.
- Context: "hit rate below this can never reach MATURE".
- Depends on / feeds into: `if hit_rate < self.SKILL_GATE: maturity = min(maturity, MATURE_THRESHOLD - 0.05)` line 158-159.

**r2 score history cap** — `search/hybrid_foundation.py:110`
- Value: `20`
- Gates: Truncates _recent_r2_scores rolling history to the last 20 entries.
- Context: "Track miner R² history".
- Depends on / feeds into: not directly consumed elsewhere in this file (history retained but r2_score uses only latest miner_r2 argument).

**principle_score normalization** — `search/hybrid_foundation.py:114`
- Value: `min(1.0, n_principles / 50)`
- Gates: Caps the "principle count" maturity component at 50 principles = full score.
- Context: "Component scores (each in [0, 1])".
- Depends on / feeds into: legacy_blend (line 123) and evidence_score (line 147).

**episode_score normalization** — `search/hybrid_foundation.py:115`
- Value: `min(1.0, n_episodes / 20)`
- Gates: Caps the episode-count maturity component at 20 episodes = full score.
- Context: same component-score block.
- Depends on / feeds into: legacy_blend and evidence_score.

**formula_score normalization** — `search/hybrid_foundation.py:116`
- Value: `min(1.0, n_formulas / 10)`
- Gates: Caps the invented-formula-count maturity component at 10 formulas = full score.
- Context: same block.
- Depends on / feeds into: legacy_blend and evidence_score.

**network_score normalization** — `search/hybrid_foundation.py:118`
- Value: `min(1.0, network_nodes / 30)`
- Gates: Caps the reasoning-network-size maturity component at 30 nodes = full score.
- Context: same block.
- Depends on / feeds into: legacy_blend and evidence_score.

**legacy_blend weights** — `search/hybrid_foundation.py:122-128`
- Value: `0.25 principle + 0.10 episode + 0.10 formula + 0.25 r2 + 0.15 network + 0.15 accuracy`
- Gates: Weighted-sum maturity score used when no principle-hit-rate evidence exists (fallback path).
- Context: "Weighted combination (principle count and R² matter most)".
- Depends on / feeds into: used at line 161 when hit_rate is None.

**skill_score normalization** — `search/hybrid_foundation.py:144`
- Value: `clip((hit_rate - 0.5) / 0.4, 0.0, 1.0)`
- Gates: Maps hit_rate in [0.5,0.9] onto skill_score in [0,1]; below 0.5 hit rate contributes zero skill credit.
- Context: "With demonstrated skill, the skill term carries 40% weight".
- Depends on / feeds into: maturity formula at lines 153-157.

**evidence_score weights / renormalization** — `search/hybrid_foundation.py:146-152`
- Value: `(0.25 + 0.10 + 0.10 + 0.25 + 0.15) / 0.85`
- Gates: Quantity-evidence maturity component (accuracy factored out, renormalized so components sum to 1).
- Context: "Quantity evidence, accuracy factored out (renormalized 0.85)".
- Depends on / feeds into: maturity formula lines 153-157.

**maturity blend weights (skill path)** — `search/hybrid_foundation.py:153-157`
- Value: `0.35 * evidence_score + 0.40 * skill_score + 0.25 * accuracy_score`
- Gates: Final maturity score when principle-hit-rate evidence exists — skill dominates at 40%.
- Context: "detachment becomes REACHABLE with a half-built knowledge base".
- Depends on / feeds into: self._knowledge_maturity, drives exploration_ratio and allocate_actions().

**MATURE cap offset** — `search/hybrid_foundation.py:159`
- Value: `MATURE_THRESHOLD - 0.05 (i.e. 0.75)`
- Gates: When hit_rate < SKILL_GATE, maturity is capped just under MATURE_THRESHOLD so PSO never fully detaches.
- Context: point 1 of the numbered comment above.
- Depends on / feeds into: self._knowledge_maturity clip.

**legacy-path MATURE cap offset** — `search/hybrid_foundation.py:161`
- Value: `MATURE_THRESHOLD - 0.05`
- Gates: Same cap applied to legacy_blend when no skill evidence exists at all.
- Context: `maturity = min(legacy_blend, self.MATURE_THRESHOLD - 0.05)`.
- Depends on / feeds into: self._knowledge_maturity clip.

**exploration_ratio decay rate** — `search/hybrid_foundation.py:183`
- Value: `exp(-3.5 * m)`
- Gates: Smooth exponential decay of PSO exploration fraction as maturity m rises from 0→MATURE_THRESHOLD.
- Context: "Smooth exponential decay from 1.0 → 0.0".
- Depends on / feeds into: allocate_actions() pso/reasoning split at lines 228-241.

**allocate_actions() max_actions default** — `search/hybrid_foundation.py:193`
- Value: `6`
- Gates: Default total action budget split between PSO/reasoning/cognitive-novelty per stage.
- Context: function signature default.
- Depends on / feeds into: stage-specific pso/reasoning/cognitive splits (lines 214-241).

**expert-stage cognitive-novelty slot** — `search/hybrid_foundation.py:218`
- Value: `max_actions - 1 reasoning, 1 cognitive`
- Gates: At "expert" maturity, exactly one of the action slots is reserved for cognitive novelty generation.
- Context: "ZERO PSO. All cognitive... Leave 1 slot for cognitive novelty".
- Depends on / feeds into: get_cognitive_novelty_request() gating (line 355).

**mature-stage fixed PSO count** — `search/hybrid_foundation.py:223`
- Value: `1`
- Gates: At "mature" maturity, exactly 1 PSO action is generated regardless of exploration_ratio.
- Context: "Minimal PSO, mostly reasoning".
- Depends on / feeds into: get_foundation_targets() mature branch (line 326-334).

**infant reasoning-floor gate** — `search/hybrid_foundation.py:239`
- Value: `m > 0.05`
- Gates: Ensures at least 1 reasoning action is allocated once maturity exceeds 0.05, even in the infant stage.
- Context: "ensure at least 1 reasoning if we have ANY knowledge at all".
- Depends on / feeds into: pso/reasoning split for infant stage.

**infant warm-start noise std** — `search/hybrid_foundation.py:284`
- Value: `0.06`
- Gates: Std-dev of Gaussian noise added around warm-start weights for the first infant-stage PSO target.
- Context: inside get_foundation_targets(), infant branch, variant i==0.
- Depends on / feeds into: targets["feature_weights"] for that variant.

**sparse-variant percentile cutoff** — `search/hybrid_foundation.py:290`
- Value: `70th percentile`
- Gates: For infant variant i==1, only the top-30% of randomly-drawn feature weights survive (rest zeroed).
- Context: "Sparse: only keep top-30% features".
- Depends on / feeds into: w[w < threshold] = 0.0.

**dense-variant weight range** — `search/hybrid_foundation.py:294`
- Value: `uniform(0.3, 0.7)`
- Gates: For infant variant i==2, all feature weights drawn moderate (dense) rather than full [0,1] range.
- Context: "Dense: all features moderate".
- Depends on / feeds into: targets["feature_weights"].

**extreme-sparse feature fraction** — `search/hybrid_foundation.py:299`
- Value: `n_features // 5 (one fifth)`
- Gates: For infant variant i>=3 (extreme, sparse branch), how many features get nonzero weight.
- Context: "Extreme: very sparse or very dense".
- Depends on / feeds into: rng.choice(n_features, size=max(1, n_features//5)).

**extreme-variant weight ranges** — `search/hybrid_foundation.py:303, 305`
- Value: `uniform(0.5, 1.0)`
- Gates: Magnitude range for both the extreme-sparse active features and the extreme-dense full vector.
- Context: same "Extreme" branch.
- Depends on / feeds into: targets["feature_weights"].

**infant penalty range** — `search/hybrid_foundation.py:309`
- Value: `uniform(0.5, 3.0)`
- Gates: Penalty value attached to each infant-stage PSO exploration target.
- Context: end of infant for-loop.
- Depends on / feeds into: targets["penalty"].

**adolescent perturbation scale** — `search/hybrid_foundation.py:316`
- Value: `0.2 + 0.3 * rng.random()`
- Gates: Noise std-dev for adolescent-stage targeted perturbations (range 0.2-0.5).
- Context: "Targeted: explore with focused perturbations".
- Depends on / feeds into: noise = rng.normal(0, scale, n_features).

**adolescent base weight range** — `search/hybrid_foundation.py:318`
- Value: `uniform(0.2, 0.8)`
- Gates: Base feature-weight range before noise for adolescent-stage targets.
- Context: same adolescent branch.
- Depends on / feeds into: w = clip(base+noise, 0, 1).

**adolescent penalty range** — `search/hybrid_foundation.py:322`
- Value: `uniform(0.8, 2.5)`
- Gates: Penalty value for adolescent-stage exploration targets.
- Context: same adolescent branch.
- Depends on / feeds into: targets["penalty"].

**mature gap-filling weight range** — `search/hybrid_foundation.py:329`
- Value: `uniform(0.3, 0.7)`
- Gates: Feature weight range for the single mature-stage "last-mile gap filling" probe.
- Context: "Last-mile gap filling: single targeted probe".
- Depends on / feeds into: targets["feature_weights"].

**mature gap-filling penalty range** — `search/hybrid_foundation.py:332`
- Value: `uniform(1.0, 2.0)`
- Gates: Penalty value for the mature-stage gap-filling probe.
- Context: same mature branch.
- Depends on / feeds into: targets["penalty"].


### `search/principle_guided_search.py` — 33 values

**GUIDANCE_FLOOR** — `search/principle_guided_search.py:35`
- Value: `0.02`
- Gates: Same floor as pso_search.py's copy, applied to base_guidance under the ratchet-fix path.
- Context: Comment explicitly warns "Fix BOTH copies or you fix only the strategy you are not running" (referenced from pso_search.py).
- Depends on / feeds into: update_feedback() ratchet branch, line 288.

**_propagation_strength default** — `search/principle_guided_search.py:51`
- Value: `0.15`
- Gates: Default fraction of a feature-weight delta propagated to correlated features.
- Context: field default, later overridable via set_correlation_matrix(strength=0.15).
- Depends on / feeds into: _apply_correlation_propagation() multiplies delta by this.

**low-data threshold** — `search/principle_guided_search.py:80`
- Value: `80 (n < 80)`
- Gates: Whether bootstrap-diversified search actions are generated at all.
- Context: "Seed extra search actions from bootstrapped mini-batches (tiny data)".
- Depends on / feeds into: self._low_data flag, gates n_seeds computation and self._bootstrap_weights.

**bootstrap seed-count bounds** — `search/principle_guided_search.py:84`
- Value: `max(3, min(8, 80 // max(n,1)))`
- Gates: Number of bootstrap-diversity seeds (3 to 8) generated for low-data datasets.
- Context: directly follows the n<80 gate.
- Depends on / feeds into: bootstrap_guidance_weights() call; consumed in propose_actions() bootstrap loop (line 179).

**num_configs stress scaling** — `search/principle_guided_search.py:104`
- Value: `1 + int(stress_level * 3)`
- Gates: How many of the 4 core actions get generated (1 at calm, 4 at max stress).
- Context: comment "1 at calm → 4 at max".
- Depends on / feeds into: gates action 2/3/4 append blocks (num_configs >= 2/3/4).

**pert_scale stress scaling** — `search/principle_guided_search.py:107`
- Value: `0.05 + stress_level * 0.45`
- Gates: Perturbation magnitude for Action 1 (perturbation around best), 0.05→0.50.
- Context: comment "0.05 at calm → 0.50 at max".
- Depends on / feeds into: `_perturb(self._best_weights, pert_scale * 0.5)` at line 121.

**max_penalty default** — `search/principle_guided_search.py:110`
- Value: `3.0`
- Gates: Upper bound on penalty before principle overrides; identical semantics to pso_search.py's copy.
- Context: "Calculate penalty (may be overridden by principles)".
- Depends on / feeds into: base_penalty formula (line 117) and actions 3/4 penalties.

**base_penalty formula** — `search/principle_guided_search.py:117`
- Value: `1.0 + (1.0 - stress_level) * (max_penalty - 1.0)`
- Gates: Stage-level sparsity penalty; more aggressive (lower) when stressed.
- Context: "Base penalty scales with stress (more aggressive when stressed)".
- Depends on / feeds into: actions 1, 2 (×0.8), bootstrap (×0.9) penalties.

**Action-1 perturbation half-scale** — `search/principle_guided_search.py:121`
- Value: `0.5 (pert_scale * 0.5)`
- Gates: Actual noise scale for the "perturbation around best" action, half of pert_scale.
- Context: "Action 1: Perturbation around best known".
- Depends on / feeds into: _perturb() Gaussian noise.

**Action-2 blend coefficient** — `search/principle_guided_search.py:131`
- Value: `0.3 + stress_level * 0.4`
- Gates: How much weight base_guidance gets vs best_weights in the guidance-blended action.
- Context: "Action 2: Guidance-informed (if stress warrants)".
- Depends on / feeds into: `guided = (1-blend)*best + blend*base_guidance`.

**Action-2 penalty scale** — `search/principle_guided_search.py:136`
- Value: `0.8 (base_penalty * 0.8)`
- Gates: Penalty attached to the guidance-blended action.
- Context: same Action-2 block.
- Depends on / feeds into: appended action dict.

**direction-magnitude gate** — `search/principle_guided_search.py:154`
- Value: `1e-9`
- Gates: Whether "away from worst" direction vector is considered non-degenerate.
- Context: comment explains this replaced a flawed `1 - worst_weights` mirror-image approach.
- Depends on / feeds into: chooses between direction-based "away" vector and fallback to base_guidance.

**Action-3 step size** — `search/principle_guided_search.py:155`
- Value: `0.3 + 0.5 * stress_level`
- Gates: How far along the best-minus-worst direction the "away from worst" candidate moves.
- Context: "Step AWAY from the worst candidate (moderate+ stress)".
- Depends on / feeds into: `away = clip(best + step*direction, 0, 1)`.

**Action-3 penalty scale** — `search/principle_guided_search.py:161`
- Value: `0.7 (max_penalty * 0.7)`
- Gates: Penalty for the "away from worst" action.
- Context: same Action-3 block.
- Depends on / feeds into: appended action dict.

**Action-4 restart blend** — `search/principle_guided_search.py:169`
- Value: `0.3 * random_weights + 0.7 * base_guidance`
- Gates: "Aggressive random restart" candidate composition (high stress only).
- Context: "Action 4: Aggressive random restart (high stress only)".
- Depends on / feeds into: gated by num_configs >= 4 (i.e. stress_level near max).

**Action-4 penalty scale** — `search/principle_guided_search.py:173`
- Value: `0.5 (max_penalty * 0.5)`
- Gates: Penalty for the aggressive-restart action.
- Context: same Action-4 block.
- Depends on / feeds into: appended action dict.

**bootstrap blend coefficients** — `search/principle_guided_search.py:180`
- Value: `0.55 * bw + 0.45 * best_weights`
- Gates: How bootstrap-diversity seeds are blended with current best weights.
- Context: "Bootstrap-diversified actions (low-data datasets)".
- Depends on / feeds into: iterates self._bootstrap_weights[:4].

**bootstrap penalty scale** — `search/principle_guided_search.py:184`
- Value: `0.9 (base_penalty * 0.9)`
- Gates: Penalty for bootstrap-seeded actions.
- Context: same bootstrap block.
- Depends on / feeds into: appended action dict.

**episodic-hint blend** — `search/principle_guided_search.py:198`
- Value: `0.6 * hint + 0.4 * best_weights`
- Gates: How much the episodic warm-start hint dominates over current best.
- Context: "Blend with current best: 60% hint, 40% best".
- Depends on / feeds into: Action 5 ("Episodic-guided").

**max actions cap** — `search/principle_guided_search.py:239`
- Value: `6 (return actions[:6])`
- Gates: Hard ceiling on total actions proposed per stage regardless of how many were generated.
- Context: "# Max 6 configs"; note the class docstring elsewhere says "4 configurations max" — inconsistent with this 6 cap.
- Depends on / feeds into: final actions returned to optimizer.

**feedback effective_lr formula (duplicate)** — `search/principle_guided_search.py:272`
- Value: `0.1 + 0.6 * (1.0 - entropy)`
- Gates: Same formula/semantics as pso_search.py:158, applied here for PrincipleGuidedSearch's own guidance update.
- Context: "Update guidance with classifier importance".
- Depends on / feeds into: self._base_guidance blend (both ratchet and non-ratchet branches).

**ratchet lr halving (duplicate)** — `search/principle_guided_search.py:282`
- Value: `0.5 (effective_lr *= 0.5)`
- Gates: Same ratchet halving as pso_search.py:166, in this file's copy of the logic.
- Context: long comment block on why min-max rescaling manufactures zeros.
- Depends on / feeds into: scale_by_max()/GUIDANCE_FLOOR blend.

**guidance floor-noise threshold (duplicate)** — `search/principle_guided_search.py:295`
- Value: `0.05`
- Gates: Zeros base_guidance entries below 0.05 in the non-ratchet path (mirrors pso_search.py:179).
- Context: immediately followed by safe_minmax renormalization.
- Depends on / feeds into: self._base_guidance.

**config-mutation stress gate** — `search/principle_guided_search.py:311`
- Value: `0.2 (stress_level < 0.2 → return {})`
- Gates: Below this stress, classifier hyperparameters are never perturbed (empty config = defaults).
- Context: "Low stress: no changes (empty dict = use defaults)".
- Depends on / feeds into: _propose_config(), used by all 4 core actions' classifier_config.

**perturbation_scale formula** — `search/principle_guided_search.py:315`
- Value: `stress_level * 0.5`
- Gates: Overall magnitude of hyperparameter perturbation (0 at calm → 0.5 at max).
- Context: comment "0 at calm → 0.5 at max".
- Depends on / feeds into: noise draw at line 328.

**param-noise scale factor** — `search/principle_guided_search.py:328`
- Value: `0.3 (perturbation_scale * range_size * 0.3)`
- Gates: Std-dev of the Gaussian noise applied to a controllable numeric hyperparameter.
- Context: inside the int/float branch of _propose_config's per-param loop.
- Depends on / feeds into: new_val = current + noise, then clipped to [pmin,pmax].

**categorical-param mutation probability** — `search/principle_guided_search.py:348`
- Value: `stress_level > 0.5 and rng.random() < 0.3`
- Gates: Whether a categorical hyperparameter gets randomly changed (30% chance, only above stress 0.5).
- Context: categorical branch of _propose_config.
- Depends on / feeds into: config[param_name] set via rng.choice(options).

**learning_mode "multi" threshold** — `search/principle_guided_search.py:356`
- Value: `stress_level >= 0.55 and variant >= 2`
- Gates: Forces learning_mode to "multi" for higher-index action variants under high stress.
- Context: str-type branch specifically for "learning_mode" param.
- Depends on / feeds into: classifier_config["learning_mode"], consumed by SymbolicFormulaAdapter.apply_config.

**learning_mode random-choice threshold** — `search/principle_guided_search.py:358`
- Value: `stress_level >= 0.35 and variant == 1`
- Gates: Randomly picks among boosting/bagging/multi for variant 1 at moderate stress.
- Context: same learning_mode branch.
- Depends on / feeds into: config["learning_mode"].

**learning_mode low-stress random probability** — `search/principle_guided_search.py:360`
- Value: `stress_level >= 0.2 and rng.random() < 0.25`
- Gates: 25% chance of randomizing learning_mode at moderate-low stress.
- Context: same learning_mode branch, last fallback condition.
- Depends on / feeds into: config["learning_mode"].

**generic str-param mutation probability** — `search/principle_guided_search.py:362`
- Value: `stress_level > 0.5 and rng.random() < 0.3`
- Gates: 30% chance of mutating any non-learning_mode string hyperparameter.
- Context: else-branch of the str param_type handling.
- Depends on / feeds into: config[param_name].

**correlation-propagation strength default** — `search/principle_guided_search.py:397`
- Value: `0.15`
- Gates: Default propagation strength argument for set_correlation_matrix().
- Context: docstring "strength: How much to propagate (0 = none, 1 = full)".
- Depends on / feeds into: self._propagation_strength, clipped at line 412.

**propagation strength cap** — `search/principle_guided_search.py:412`
- Value: `clip(strength, 0.0, 0.5)`
- Gates: Hard ceiling of 0.5 on how much correlated-feature propagation can move a weight, regardless of caller input.
- Context: `self._propagation_strength = np.clip(strength, 0.0, 0.5)`.
- Depends on / feeds into: _apply_correlation_propagation() multiplier.


### `search/pso_search.py` — 18 values

**GUIDANCE_FLOOR** — `search/pso_search.py:22`
- Value: `0.02`
- Gates: Minimum value any feature's guidance weight can be pushed to when the ratchet-fix path rescales guidance by max.
- Context: Comment ties it to the min-max "manufactures a zero every stage" bug; used to keep a deprioritized feature reachable.
- Depends on / feeds into: read in update_feedback() when fix_guidance_ratchet is True (line 172); mirrors the same constant in principle_guided_search.py.

**_n_particles default** — `search/pso_search.py:34`
- Value: `20`
- Gates: Default swarm size if initialize() isn't given n_particles explicitly.
- Context: Constructor field default, overridden by initialize(n_particles=20) default at line 42.
- Depends on / feeds into: Xp/V array shapes in initialize(); resource_budget.py can override via detect_resource_budget().

**_pso_iters default** — `search/pso_search.py:35`
- Value: `30`
- Gates: Default number of PSO update iterations run per propose_actions() call.
- Context: Constructor field default, overridden by initialize(pso_iters=30) default at line 43.
- Depends on / feeds into: the `for _t in range(self._pso_iters)` loop at line 100.

**initialize() n_particles / pso_iters defaults** — `search/pso_search.py:42-43`
- Value: `n_particles=20, pso_iters=30`
- Gates: Swarm size and iteration budget used as decision boundaries whenever a caller doesn't pass explicit values.
- Context: Docstring says PSO is "FIXED internal parameters... No stress-scaling."
- Depends on / feeds into: sets self._n_particles/self._pso_iters, consumed throughout propose_actions().

**velocity init std** — `search/pso_search.py:53`
- Value: `0.2`
- Gates: Std-dev of the Gaussian used to initialize particle velocities V.
- Context: `self.V = self._rng.normal(0.0, 0.2, size=(p, n_features))`.
- Depends on / feeds into: velocity update loop at line 120-124.

**guidance-seeded particle fraction** — `search/pso_search.py:56`
- Value: `p // 3 (one third)`
- Gates: How many particles are initialized near base_guidance vs uniform-random.
- Context: `for i in range(p // 3): self.Xp[i] = clip(base_guidance + noise, 0, 1)`.
- Depends on / feeds into: initial Xp population, then pbest/gbest tracking.

**guidance-seed noise std** — `search/pso_search.py:58`
- Value: `0.1`
- Gates: Perturbation added to base_guidance when seeding the first third of particles.
- Context: `self._rng.normal(0, 0.1, n_features)` inside the seeding loop.
- Depends on / feeds into: initial Xp values.

**diversity normalization divisor** — `search/pso_search.py:88`
- Value: `0.5`
- Gates: Normalizes swarm diversity into [0,1] entropy proxy used to scale the sparsity penalty.
- Context: `normalized_entropy = clip(diversity / 0.5, 0.0, 1.0)`, "Calculate adaptive penalty" comment.
- Depends on / feeds into: adaptive_penalty formula at line 90.

**adaptive_penalty coefficients** — `search/pso_search.py:90`
- Value: `1.0 + 2.0 * (1.0 - normalized_entropy)`
- Gates: Base sparsity penalty applied to fitness before principle overrides, ranging [1.0, 3.0].
- Context: comment "Calculate adaptive penalty (capped by overrides from principles)".
- Depends on / feeds into: clipped against max_penalty (line 91/97) then used in fitness at line 103.

**max_penalty default** — `search/pso_search.py:91`
- Value: `3.0`
- Gates: Upper bound on adaptive_penalty unless a principle's action_override supplies a smaller cap.
- Context: "# Default, may be overridden".
- Depends on / feeds into: overridden by `p_dict["action_override"]["max_penalty"]` in the loop at lines 93-97; used in returned action penalty (line 138).

**distinctness bonus weight** — `search/pso_search.py:105`
- Value: `0.1`
- Gates: Weight of the diversity/distinctness term in the per-iteration PSO fitness function.
- Context: `fitness = relevance - sparsity + 0.1 * distinctness`.
- Depends on / feeds into: pbest/gbest updates each PSO iteration.

**inertia weight (w)** — `search/pso_search.py:121`
- Value: `0.7`
- Gates: How much prior velocity carries forward each PSO step.
- Context: "# Velocity and position update (fixed parameters)".
- Depends on / feeds into: self.V update, then Xp = clip(Xp+V, 0, 1).

**cognitive coefficient c1** — `search/pso_search.py:122`
- Value: `1.5`
- Gates: Pull strength toward each particle's personal best (pbest).
- Context: same "fixed parameters" velocity-update block.
- Depends on / feeds into: velocity update, combined with r1 random draw.

**social coefficient c2** — `search/pso_search.py:123`
- Value: `1.5`
- Gates: Pull strength toward the swarm's global best (gbest).
- Context: same block; standard PSO c1=c2=1.5, inertia 0.7 — classic fixed-parameter PSO, no velocity clamp.
- Depends on / feeds into: velocity update, combined with r2 random draw.

**top-N solution count** — `search/pso_search.py:131`
- Value: `4 (top_indices = argsort(...)[-4:][::-1])`
- Gates: How many candidate feature-weight vectors PSO returns per stage.
- Context: "Returns top 1-4 solutions from the swarm" per class docstring.
- Depends on / feeds into: `actions` list returned to the optimizer.

**feedback effective_lr formula** — `search/pso_search.py:158`
- Value: `0.1 + 0.6 * (1.0 - entropy)`
- Gates: Learning rate for blending classifier importance into base_guidance; ranges [0.1, 0.7].
- Context: "Feedback symbiosis: blend classifier importance into guidance".
- Depends on / feeds into: self._base_guidance update (both ratchet and non-ratchet branches).

**ratchet lr halving** — `search/pso_search.py:166`
- Value: `0.5 (effective_lr *= 0.5)`
- Gates: Halves the update rate when fix_guidance_ratchet is enabled, to avoid rewriting guidance every ~2 stages.
- Context: Comment: "halve the update rate (0.7/stage rewrites the vector every ~2 stages)".
- Depends on / feeds into: subsequent scale_by_max()/GUIDANCE_FLOOR blend.

**guidance floor-noise threshold** — `search/pso_search.py:179`
- Value: `0.05`
- Gates: Zeros any base_guidance entry below 0.05 in the non-ratchet feedback path.
- Context: "# Floor noise" before re-normalization via safe_minmax.
- Depends on / feeds into: self._base_guidance, later consumed by propose_actions() guidance vector.

---

## Appendix B — Files the parallel session's sweep didn't reach

The parallel session's four-pass sweep (Appendix A) covered 33 files and found 449 gating constants. This session ran its own independent three-pass sweep of the same codebase and covered several files the other pass didn't reach — 21 additional files, ~135 more values. They're listed here in the flatter `value at line N` format that sweep produced, **not yet given the FILE:LINE / WHAT-IT-GATES / DEPENDS-ON deep treatment** — that's real remaining work, not a gap in importance. A few (`bootstrap_knowledge.py`'s seed principles, `investigation_engine.py`'s stress bands) are directly relevant to Part C's categories and should get the deep treatment early.

### `cognition/router_tuning.py`
- `margin: float = 0.02` (line 26) — default margin subtracted from the fitted percentile when re-estimating router thresholds
- `max(0.5, min(0.98, formula_thr))` (line 46) — clamps the fitted D2_math routing threshold to [0.5, 0.98]
- `max(0.7, min(0.99, linear_thr))` (line 47) — clamps the fitted D4_structured routing threshold to [0.7, 0.99]

### `cognition/memory_reset.py`
- `FormulaEngine(seed=42)` (line 66) — fixed RNG seed used when re-bootstrapping the formula engine after a memory wipe

### `cognition/method_templates.py`
- `min(8, n_features)` (line 23) — caps anchor features sampled for seed-template generation
- `_c(2.0)` (line 46) — scale coefficient in the "stump" seed formula
- `_c(1e-3)` (line 63) — additive epsilon inside log() in the naive-bayes-style seed formula
- `_c(0.1)` (line 103) — additive regularizer in the inverse-distance seed formula's denominator

### `cognition/cell_store.py`
- `ep_list[-50:]` (line 32) — caps episodes carried into a composite VersionStore snapshot to the most recent 50
- `def cells_for_api(state, limit: int = 80)` (line 104) — default cap on HITL-reviewable rows returned
- `rows[:limit]` (line 124) — applies the row cap above

### `cognition/live_feed.py`
- `MAX_FEED_BYTES = 5 * 1024 * 1024` (line 22) — rotates the live feed file once it exceeds 5 MB
- `def read_since(line_offset=0, max_lines=200)` (line 70) — default page size for feed polling
- `def read_tail(max_lines=100)` (line 104) — default tail window returned on UI bootstrap

### `cognition/lab_report.py`
- `sorted(tested, ...)[:12]` (line 44) — caps hypotheses listed in the lab report's "Hypotheses & experiments" section
- `belief_changes[-20:]` (line 120) — caps recent belief-trajectory lines shown in the report

### `cognition/bootstrap_knowledge.py`
- `"threshold": 80` (line 45) — seed principle: n_samples < 80 gates the small-n guide
- `"max_penalty": 1.4` (line 46) — action-override coefficient for the small-n seed principle
- `"threshold": 0.5` (line 50) — seed condition: feat_ratio > 0.5 gates the high-dim guide
- `"max_penalty": 1.2` (line 51) — action-override coefficient for the high-dim seed principle
- `"threshold": 0.15` (line 55) — seed condition: entropy < 0.15 gates the low-entropy guide
- `"min_penalty": 0.8` (line 56) — action-override coefficient for the low-entropy seed principle
- `"threshold": 0.45` (line 60) — seed condition: decision_stress > 0.45 gates the stress guide
- `"max_penalty": 2.0` (line 61) — action-override coefficient for the stress seed principle
- `"threshold": 0.08` (line 65) — seed condition: regret > 0.08 gates the regret guide
- `"max_penalty": 1.5` (line 66) — action-override coefficient for the regret seed principle
- `"threshold": 2` (line 70) — seed condition: num_classes > 2 gates the multiclass guide
- `"max_penalty": 1.6` (line 71) — action-override coefficient for the multiclass seed principle
- `"threshold": 0.65` (line 75) — seed condition: outcome.accuracy < 0.65 gates the low-accuracy guide
- `"max_penalty": 2.2` (line 76) — action-override coefficient for the low-accuracy seed principle
- `"threshold": 0.6` (line 80) — seed condition: sparsity > 0.6 gates the sparse-data guide
- `"max_penalty": 1.3` (line 81) — action-override coefficient for the sparse-data seed principle
- `formula_count() > 40` (line 134) — threshold for resetting the symbolic library on forced bootstrap

*Note: these 7 seed principles are the system's entire starting knowledge before any mining happens. Every one of their `threshold`/`max_penalty`/`min_penalty` values was hand-picked at design time — they're prime Category D candidates (need real evidence before re-tuning) but are currently indistinguishable from mined, evidence-backed principles once loaded into the library.*

### `cognition/uil_journal.py`
- `def get_recent(self, n: int = 20)` (line 107) — default window size of recent entries returned
- `if max_stress_reached > 0.5` (line 159) — threshold for remembering a dataset as "hard"
- `min(0.5, max_stress_reached * 0.6)` (line 160) — cap and decay coefficient for carried-over initial stress

### `cognition/working_memory_pruner.py`
- `MAX_WORKING_JOURNAL = 4000` (line 29) — cap on live journal entries before archiving
- `JOURNAL_PROTECT_RECENT = 600` (line 30) — most-recent journal entries protected from archiving
- `MAX_WORKING_PRINCIPLES = 180` (line 31) — cap on live principle count
- `MAX_WORKING_HYPOTHESES = 100` (line 32) — cap on live hypothesis count
- `STALE_HYPOTHESIS_SEC = 7 * 24 * 3600` (line 33) — age (7 days) after which a pending hypothesis is archived
- `score *= 0.85` (line 93) — penalty multiplier for cross-dataset journal entries when ranking for archival
- `if acc < 0.55: score *= 0.7` (line 95) — accuracy threshold/penalty multiplier lowering archival priority
- `conf < 0.30` (line 146) — confidence threshold triggering "low_confidence_decay" archival
- `applied >= 20 and help_ratio < 0.25` (line 148) — thresholds for "low_help_ratio" archival
- `conf < 0.45 and applied >= 8` (line 150) — thresholds for "other_dataset_low_confidence" archival
- `conf < 0.50 and applied >= 10` (line 157) — thresholds for "quarantined_weak" archival
- `hyp.causal_strength < 0.15` (line 233) — threshold for archiving a weak correlational hypothesis

### `cognition/knowledge_cell.py`
- `confidence: float = 0.5` (line 75) — default/fallback confidence for a PrincipleCell
- `confidence: float = 0.5` (line 169) — default/fallback confidence for a FormulaCell
- `confidence: float = 0.5` (line 337) — default/fallback confidence for an EdgeCell

*Note: the same 0.5 default appears independently in three cell types. This is the same duplication pattern Part C flags for `GUIDANCE_FLOOR` — worth checking whether it should be one shared constant.*

### `cognition/investigation_engine.py`
- `STRESS_LOW = 0.3` (line 41) — stress threshold below which strategy C.1 is used
- `STRESS_MID = 0.6` (line 42) — stress threshold below which C.2/C.3 are used
- `STRESS_HIGH = 0.8` (line 43) — stress threshold above which C.4/C.5 are used
- `MAX_INVESTIGATION_STAGES = 5` (line 46) — hard cap on investigation stages
- `RETRY_STEP_DECAY = 0.5` (line 47) — step-size decay applied per controlled retry
- `investigation_count >= 3` (line 89) — escalation threshold to strategy C.5
- `investigation_count >= 2` (line 91) — escalation threshold to strategy C.4
- `rng = np.random.default_rng(42)` (line 366) — fixed seed for the orthogonal-exploration direction
- `random_dir * fail_norm * 0.3` (line 374) — step-size scaling coefficient for orthogonal exploration

*Note: this is the SAME 0.3/0.6/0.8-style stress-band pattern flagged in Part D for `uil_stress.py` — another hand-picked ladder gating a decision tree of five investigation strategies, with no cited measurement behind the specific cut points.*

### `cognition/llm_mutator.py`
- `max(n_proposals * 4, 8)` (line 88) — candidate-generation multiplier/floor for neural ranking
- `smoothness >= 0.7 and n_features <= 8` (line 172) — heuristic thresholds for the "prefer symbolic" goal
- `"confidence": 0.6` (line 185) — default confidence assigned to a heuristic-generated goal
- `sparsity >= 0.6 and n_features >= 32` (line 193) — heuristic thresholds for the "sparse" goal
- `"temperature": 0.2` (line 244) — LLM sampling temperature for goal-proposal calls
- `timeout=30` (line 255) — HTTP timeout (seconds) for the goal-proposal LLM call
- `cells[:8]` (line 278) — cap on cells included in the mutation-proposal LLM prompt
- `"temperature": 0.3` (line 292) — LLM sampling temperature for mutation-proposal calls
- `timeout=30` (line 303) — HTTP timeout (seconds) for the mutation-proposal LLM call

### `cognition/neural_proposer.py`
- `_MAX_LEN = 48` (line 48) — max token sequence length for the tiny transformer
- `c < 0.5` (line 74) — threshold bucketing a constant into the "c_small" token
- `c <= 2.0` (line 76) — threshold bucketing a constant into the "c_one" token
- `d: int = 16, h: int = 32` (line 93) — model embedding/hidden dimensionality defaults
- `epochs=120, lr=0.01, holdout=0.25` (line 176) — training hyperparameter defaults
- `m[k] = 0.9*m[k] + 0.1*g` (line 200) — Adam first-moment decay coefficient
- `v[k] = 0.999*v[k] + 0.001*g*g` (line 201) — Adam second-moment decay coefficient
- `1 - 0.9**t` / `1 - 0.999**t` (line 202) — Adam bias-correction constants
- `+ 1e-8` (line 204) — Adam epsilon
- `>= 0.5` (line 212) — decision threshold for binary training accuracy
- `(helped + 1.0) / (helped + hurt + 2.0)` (line 298) — Laplace smoothing for the quality label
- `0.5 * ratio + 0.5 * conf` (line 300) — weighting coefficients combining help-ratio and confidence
- `len(samples) < 20` (line 331) — minimum training samples before a model is trained

### `cognition/symbolic_regressor.py`
- `_EPS = 1e-12` (line 27) — numeric epsilon used throughout the module
- `n_terms: int = 10` (line 62) — default number of boosting terms
- `learning_rate: float = 0.6` (line 63) — default boosting learning rate
- `max_pair_features: int = 8` (line 64) — cap on features considered for pairwise interactions
- `g > best_gain + 1e-4` (line 116) — improvement margin for accepting a depth-extension
- `np.std(r) < 1e-9` (line 129) — residual-std stopping criterion
- `gain < 1e-3` (line 132) — minimum-gain stopping criterion
- `iters=12` (line 134) — constant-refit iteration count per boosting round
- `abs(a) < _EPS` (line 137) — degenerate-term stopping check

### `cognition/symbolic_gradient.py`
- `RAW_CLIP = 50.0` (line 36) — clip bound on raw formula outputs before standardization
- `_EPS = 1e-9` (line 37) — numeric epsilon used throughout the module
- `np.clip(z, -30.0, 30.0)` (line 41) — sigmoid clip bound
- `lr = 0.1` (line 114) — Adam learning rate for constant refit
- `b1, b2, eps = 0.9, 0.999, 1e-8` (line 117) — Adam decay/epsilon hyperparameters
- `h = 1e-4 * (1.0 + abs(theta[i]))` (line 122) — finite-difference step scaling
- `np.clip(theta, -1e3, 1e3)` (line 133) — constant bound during refit
- `n_iter: int = 8` (line 158) — Newton line-search iteration cap
- `abs(step) < 1e-4` (line 175) — convergence tolerance for line search
- `np.clip(g, 0.0, 10.0)` (line 179) — bound on the line-search multiplier
- `std if std > _EPS else 1.0` (line 209) — fallback std when the raw output is near-constant
- `np.clip(learning_rate, 0.01, 1.0)` (line 215) — learning-rate clamp bounds

### `core/small_data_diversify.py`
- `MIN_EFFECTIVE_ROWS = 15` (line 7) — minimum row count before bootstrap-jitter kicks in
- `noise_frac: float = 0.01` (line 15) — default jitter-noise fraction for row bootstrap
- `scale = np.where(scale > 1e-9, scale, 1.0)` (line 26) — degenerate std-dev floor
- `batch_size = ... min(n * 2, 32)` (line 42) — batch-size cap of 32
- `Xb += rng.normal(0.0, 0.008 * scale, ...)` (line 50) — batch jitter-noise scale
- `if diff.sum() < 1e-9` (line 70) — degenerate class-separation threshold

### `core/fig_xgboost.py`
- `getattr(self.cfg, "boosting_rounds_per_stage", 50)` (line 65) — default boosting-rounds-per-stage fallback
- `max_depth=6` (line 67) — fixed XGBoost tree depth for the backward-compat wrapper
- `learning_rate=0.1` (line 68) — fixed XGBoost learning rate for the backward-compat wrapper
- `(preds > 0.5).astype(int)` (line 146) — binary-classification decision threshold in raw fallback path

### `core/utils.py`
- `if hi - lo < 1e-12` (line 43) — degenerate-range floor in `safe_minmax`
- `if hi < 1e-12` (line 61) — degenerate-max floor in `scale_by_max`

### `adapters/xgboost_adapter.py`
- `boosting_rounds: int = 50` (line 29) — default/cap on number of boosting rounds
- `max_depth: int = 6` (line 30) — default max tree depth
- `learning_rate: float = 0.1` (line 31) — default learning rate
- `return self.num_classes <= 2` (line 58) — threshold gating whether custom objectives are supported (binary-only; **bug #10's fix**)
- `weights = np.clip(feature_weights, 1e-6, None)` (line 70) — epsilon floor preventing zero/negative column-sampling weights
- `(np.asarray(preds) > 0.5).astype(int)` (line 96) — decision threshold converting probability to binary class
- max_depth search bounds `[1, 15]`, default 6 (lines 123-124)
- learning_rate search bounds `[0.001, 1.0]`, default 0.1 (lines 127-128)
- subsample search bounds `[0.1, 1.0]`, default 0.8 (lines 131-132)
- colsample_bytree search bounds `[0.1, 1.0]`, default 1.0 (lines 135-136)
- colsample_bynode search bounds `[0.1, 1.0]`, default 0.2 (lines 139-140)
- min_child_weight search bounds `[0, 100]`, default 1 (lines 143-144)
- gamma search bounds `[0.0, 10.0]`, default 0 (lines 147-148) — **this is the bug #1 knob**, seeded correctly now
- reg_alpha search bounds `[0.0, 10.0]`, default 0 (lines 151-152) — **bug #1 knob**
- reg_lambda search bounds `[0.0, 10.0]`, default 1 (lines 155-156) — **bug #1 knob**
- boosting-round search bounds `[10, 500]` (lines 159-160)
- default sampling fractions `subsample=0.8, colsample_bytree=1.0, colsample_bynode=0.2` seeded into params (lines 224-226)
- default regularization values `gamma=0, reg_alpha=0, reg_lambda=1, min_child_weight=1` seeded into params (lines 233-236)

### `adapters/knn_adapter.py`
- `n_neighbors: int = 5` (line 25) — default k
- `self.p = 2` (line 36) — default Minkowski power
- `weights = np.clip(feature_weights, 1e-6, None)` (line 63) — epsilon floor on feature weights
- `n = min(500, len(self._last_y))` (line 106) — cap on sample count for permutation importance
- `n_repeats=5` (line 114) — repeat count for permutation-importance estimate
- neighbor-count search bounds `[1, 50]` (lines 134-136)
- Minkowski-power search bounds `[1, 5]` (lines 148-150)

### `adapters/tabpfn_adapter.py`
- `MAX_FEATURES = 500` (line 33) — cap on feature count TabPFN accepts
- `MAX_SAMPLES = 10_000` (line 34) — cap on sample count TabPFN accepts
- `threshold = max(0.05, np.percentile(feature_weights, 20))` (line 98) — fallback minimum feature-selection threshold and 20th-percentile cutoff
- `if mask.sum() < 2` (line 101) — minimum-count threshold ensuring at least 2 features are selected
- `top_indices = ...[-max(2, n_features // 5):]` (line 102) — fallback floor (2) and fraction (1/5) of features kept when selection collapses
- `n = min(300, len(self._last_y))` (line 139) — cap on sample count for permutation importance
- `n_repeats=3` (line 147) — repeat count for permutation-importance estimate

*Note: this is the adapter Part F/RESEARCH_FAILURE_ANALYSIS.md item 7 recommends activating as a router ceiling probe — it's currently registered nowhere (`classifier_registry.py` doesn't include it), so these constants govern code that never runs today.*

---

**Appendix B total: ~135 values across 21 files.**

Combined with Appendix A's 449, that's **~584 hardcoded gating constants** currently identified across the codebase — consistent with this session's own parallel count of 582 from a differently-scoped sweep. Two independent sweeps converging on the same order of magnitude (both ≈550–590) is itself evidence the count is real, not an artifact of how either sweep was run.
