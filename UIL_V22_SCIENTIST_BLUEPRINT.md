# UIL v22 — "The Editable Scientist" Blueprint

Unified design combining (a) the state-of-the-field roadmap (LLM-guided symbolic
regression, active causal experimentation, traceable reports) with (b) the four
user directives: orthogonal statistical fingerprinting, invertible normalization,
lossless versioning, and a smallest self-modifiable knowledge unit.

Date: 2026-07-07. Status: design. Build target: `figpso_uil_ai` working copy.

---

## 0. Ground-truth constraints (from completed experiments — do not relitigate)

1. **Falsified:** the cognitive layer does NOT beat plain PSO+classifier on
   accuracy or efficiency on saturated tabular data (4/4 configs ≤ 0, held-out).
2. **Proven win:** symbolic recovery + out-of-range extrapolation
   (R² 0.992 vs XGBoost −5.52). Classification version of the win is narrow:
   curved boundary tested OOD only.
3. **Surviving claim:** editability / traceability / human correction — untested,
   unrefuted. This is the differentiator to build toward, not raw accuracy.
4. Root defect found in the negative result: regret is computed against ONE
   fixed validation split; PSO detachment never fires; miner asleep below 40
   journal entries.

Therefore v22 optimizes for: **(i) equation/law discovery with extrapolation as
the headline benchmark, (ii) verifiable & editable knowledge as the product,
(iii) accuracy parity (not victory) on ordinary tabular via routing.**

---

## 1. The unifying primitive: the Knowledge Cell over an Event-Sourced Store

The single design decision that makes everything else compose.

### 1.1 KnowledgeCell (the "smallest unit")

One typed, versioned, discrete record that principles, formulas, graph edges,
routing rules, and causal hypotheses all conform to. NOT a transformer — a
transformer emulating a graph is unreadable and unedited-able, which would
destroy the one surviving claim. The unit is an *interface*; learnable/neural
members (e.g. a future KAN adapter) plug in as one cell type among many.

```python
class KnowledgeCell(Protocol):
    cell_id: str            # stable identity across versions
    cell_type: str          # principle | formula | edge | routing_rule | hypothesis
    version: int            # monotonically increasing
    provenance: Provenance  # evidence trail: journal entries, trials, human edits
    confidence: float       # current belief strength (history lives in the store)

    def evaluate(self, state: ExecutionContext) -> CellVerdict
    def mutate(self, proposal: MutationProposal) -> "KnowledgeCell"  # returns NEW version
    def compose(self, other: "KnowledgeCell") -> "KnowledgeCell | None"
    def to_readable(self) -> str          # human-readable rendering (HITL surface)
    def to_structural(self) -> StructuralForm  # Part 17 encoding (LLM/merger surface)
```

- `mutate()` never edits in place — it emits a new version + a `CellEvent`.
- The LLM (Section 5) is a *mutation-proposal generator*; the executor validates
  proposals; the store records them. Self-modification with a full audit trail.
- Existing classes adapt via thin wrappers first (no big-bang rewrite):
  `Principle`, `FormulaNode` tree, network edge, `Domain` routing rule,
  `CausalHypothesis`.

### 1.2 VersionStore (event sourcing — fixes "the network loses information")

Driving requirement (user, 2026-07-07): *"the versioning system is important —
as of right now the network loses at least some of the information."*
Confirmed against the code: three concrete loss channels exist today.

Current state: `change_log.py` (append-only but capped at 10,000 events, no
state reconstruction) + `knowledge_archive.py` (archives pruned items). Gaps:
caps truncate history; no replay to a past state; decay mutates confidence
in place, erasing belief trajectories.

Fix — `cognition/version_store.py`:
- **Append-only, uncapped event log** with segment files
  (`results/events/events_000001.jsonl`, …) — rotation instead of truncation.
- Event = `{seq, utc, cell_id, cell_type, from_version, to_version, op, payload, cause}`
  where `op ∈ {create, mutate, decay, promote, demote, quarantine, merge, prune,
  human_edit, human_reject}` and `cause` links to journal entries / trial ids /
  user id.
- **Decay becomes an event**, not an in-place write: a belief that fell
  0.9 → 0.3 is distinguishable from one that was always 0.3.
- **Snapshots** every N events (`results/snapshots/state_<seq>.json`);
  `reconstruct(seq_or_time)` = nearest snapshot + replay. Answers
  "what did the system believe at stage 340 and why did it change?"
- Current network/library/etc. become **materialized views** over the store;
  their existing JSON files remain as the latest snapshot for compatibility.
- `change_log.py` and `knowledge_archive.py` are subsumed: keep their APIs as
  facades over the store during migration.

Why this is load-bearing: state reconstruction is the mechanism behind the lab
reports (Section 6), HITL undo (Section 6), and honest belief-trajectory
evidence for promotions (Section 4).

---

## 2. Perception: orthogonal fingerprint + invertible normalization

### 2.1 Five-axis statistical fingerprint (`cognition/fingerprint.py`)

Principle: to profile data uniquely without testing the same characteristic
twice, use mutually exclusive, orthogonal measures — each isolating one
distinct property (center, spread, shape, dependence, sequence) without
overlapping the others. To build a comprehensive profile without redundancy,
**pick exactly one measure per category** as the default fingerprint dimension;
the rest of each category is the substitution menu (e.g. swap parametric →
non-parametric when axis 3 rejects normality).

**Store standardized effect sizes / statistics, never p-values** (p-values
conflate effect with sample size).

Full menu per axis (defaults in **bold**):

| Axis | Property isolated | Test/measure menu |
|---|---|---|
| 1 Center | where the middle resides (ignores spread & shape) | **trimmed mean / median (robust)**; one-sample t / Wilcoxon signed-rank (center = value); two-sample t / Mann-Whitney U (two groups); ANOVA / Kruskal-Wallis (3+ groups) |
| 2 Spread | dispersion only (ignores the center) | **MAD; Levene statistic across class groups**; F-test of equal variances (2 groups); Bartlett (3+ groups, normal); Ansari-Bradley (non-parametric scale) |
| 3 Shape | pattern, peaks, tails (regardless of center/spread) | **D'Agostino K² skewness; Anscombe-Glynn kurtosis; Shapiro-Wilk W** (normality distance); Kolmogorov-Smirnov (arbitrary reference shape) |
| 4 Dependence | how variables move together (ignores individual shapes) | **Spearman ρ matrix summary** (monotonic); Pearson r (strictly linear); **Cramér's V** (from χ² of independence, categorical); existing **linear/formula probe scores** |
| 5 Sequence | order of collection only (ignores the values' magnitudes) | **Ljung-Box Q at lags {1,5,10}** (multi-lag randomness); Wald-Wolfowitz runs-test Z (binary sequence randomness); Durbin-Watson (lag-1 autocorrelation) |

Honesty note: the categories are orthogonal as *descriptive targets*, not as
hypothesis tests (t-tests assume normality/equal variance, so assumptions
overlap). Used as fingerprint *features* — statistics, not accept/reject
decisions — the overlap is harmless.

- Axis 5 is a **new capability**: detects "this is a time series," enabling a
  new routed domain **D5_sequential** (currently invisible to the router).
- The existing probes (6-term symbolic probe R², logistic probe acc) stay —
  they are axis-4 evidence and were the router's best feature.
- Fingerprint output is itself versioned (a `dataset` cell) so routing decisions
  have reproducible inputs.

### 2.2 Invertible normalization (`core/normalization.py`)

Why normalization is mandatory here (three distinct failure modes it removes):
1. **Scale-driven bias** — a feature in the tens of thousands (income)
   dominates one in the tens (age) purely by unit magnitude; normalization
   makes every feature contribute by variance, not by unit.
2. **Optimization stability** — gradient-based fitting (formula constant
   refitting, any neural adapter) converges faster on similar scales and
   avoids the divergence/"NaN trap" the formula engine already guards against.
3. **Fair distances** — KNN adapter, episodic kNN retrieval, router
   fingerprint distance, and PSO neighborhoods all measure coordinate
   differences; unscaled features silently reweight them.

Method selection (per feature, decided by the axis-3 fingerprint):
- **Z-score standardization** `x' = (x − μ)/σ` — the default (unbounded,
  outlier-tolerant enough for near-normal features).
- **Min-max scaling** `x' = (x − min)/(max − min)` — only when a bounded
  [0,1] range is required (e.g. bounded activation inputs); brittle to
  outliers, so never the default.
- **Robust scaler** (median/MAD) — when axis 3 flags heavy tails or strong
  skew.

Rules:
- Fitted on **training folds only** — never full data (leakage = the
  single-split defect in miniature).
- The fitted transform is stored with provenance next to the fingerprint, and
  **every recovered symbolic law is back-transformed to original units before
  entering the library** — a scientist reports `y = 3.2·x²` in real units, not
  scaled ones. `to_readable()` on formula cells performs the inverse mapping.
- Router probes consume normalized features (fixes the known knife-edge
  threshold sensitivity).

---

## 3. Routing: the router becomes step 0 of `fit()`

- Move `domain_router.py` from the parallel store into `core/optimizer.py`
  step 0: fingerprint → route → select adapter + strategy.
- Routing rules become `routing_rule` cells in the principle hierarchy
  (Category tier), with provenance = the fingerprints + outcomes that minted
  them. `RoutingMemory` merges into `episodic_attention` (same kNN idea,
  one store).
- Domain registry stays OPEN (user requirement): `discover_domain()` mints new
  domains from novel fingerprint regions; minted domains are cells too —
  versioned, auditable, human-editable.
- Seed domains: D1_noisy_tabular→gbm(grad_fit), D2_math→symbolic,
  D3_tiny→fewshot/TabPFN, D4_structured→sparse_linear, **D5_sequential→(new)**.
- Known follow-ups honored: D3/D4 learners are named targets to implement;
  thresholds re-tuned after normalization lands.

---

## 4. Epistemology repair (prerequisite for everything above being true)

1. **Kill the single-split regret** (`core/optimizer.py`): regret and mining
   labels computed from K-fold / repeated-split validation; held-out data
   participates in mining, not only promotion. Every mined cell records the
   fold-spread in provenance.
2. **Make PSO detachment fire** (`search/hybrid_foundation.py`): detachment
   keyed to *cell-prediction accuracy* over the rolling window (do applied
   cells' predicted-better configs actually win?). At maturity, cells propose
   configs directly; PSO becomes fallback. This is the only path to an
   efficiency win — reasoning must replace search, not decorate it.
3. **Tiny-data mining mode** (`cognition/uil_miner.py`): below 40 entries,
   switch to shallow-rule mining with wide confidence intervals + seed-knowledge
   priors instead of sleeping. n<150 is where reasoning should shine; it is
   currently the regime where the layer is off.
4. **Bug gates:** guard `supports_custom_objective` for `num_classes > 2`
   (known crash); `MetaPrincipleMerger(filepath=None)` invariant preserved;
   all CLAUDE.md invariants remain in force.

---

## 5. The scientist core

### 5.1 Active experiment design (upgrade `causal_intervention.py`)

From passive (test hypotheses on whatever stages arrive) to active: maintain
competing hypothesis cells; choose the next intervention (config perturbation,
data slice, synthetic probe) by **expected information gain** across them
(Bayesian update over hypothesis posteriors; greedy EIG acquisition is enough —
no need for the full ACE/DPO machinery initially). Budgeted by
`core/resource_budget.py`. Majority-of-3-trials rule for CONFIRMED_CAUSAL stays.

### 5.2 LLM hypothesis generator (`cognition/llm_mutator.py`, optional dependency)

The LLM-SR / LaSR pattern, with UIL as the memory-and-verification substrate:
- Input: dataset fingerprint + top-k relevant cells (`to_structural()` /
  `to_readable()`) + current residuals.
- Output: `MutationProposal`s — new formula skeletons, principle edits,
  hypothesis candidates.
- **The LLM never writes state.** Executor evaluates proposals on validation
  folds; only survivors become new cell versions; every proposal (accepted or
  rejected) is an event with the LLM's rationale in `cause`.
- Degrades gracefully: with no API key, the existing 7-mutation formula engine
  is the proposal generator (system remains fully functional offline).
- Constant discovery: wire `symbolic_gradient.refit_constants` into the
  symbolic regressor so `sin(3x)` is reachable (currently unwired).

### 5.3 What "self-changing smallest unit" means here

The self-modification the user asked for = LLM/engine proposes → executor
validates → store versions. Discrete, typed, replayable. A learnable-unit
adapter (KAN: edge-wise learned activations prunable to formulas) may be added
later as `adapters/kan_adapter.py` — one more routed learner, **not** the
substrate.

### 5.4 Research track R1 — the uniform neural unit (the transformer cell)

User directive (2026-07-07): *"when shifting to LLMs etc. we will need to work
upon the smallest unit for our system that could change itself as per
parameters — likely a version of transformers that is able to replicate node,
neural network, knowledge graph and decision tree."*

This enters v22 as a **research track**, not the substrate, under one hard
safety condition that resolves the tension with the editability claim:

> **The Distillation Round-Trip condition.** A neural unit is admissible only
> if knowledge can flow BOTH ways: discrete cells → train/condition the neural
> unit (distill in), and neural unit → extracted rules / formulas / graph
> edges that re-enter the store as ordinary versioned cells (distill out).
> The discrete VersionStore remains the ground truth; the neural unit is a
> *computational organ*, never the *memory*.

Under that condition, one small transformer-style module (`cell_type:
"neural"`, implementing the full KnowledgeCell interface — so it is versioned,
provenance-tracked, and mutable-by-proposal like everything else) can
legitimately play each of the four roles the user named:

| Role the unit replicates | Concrete form | Distill-out path |
|---|---|---|
| Node | learned embedding of a cell (its structural form → vector) | nearest-cell attribution: which discrete cells explain the embedding |
| Neural network | fast approximate evaluator: state → predicted verdicts of the whole active cell set in one forward pass | disagreements with the true executor become mining targets |
| Knowledge graph | attention weights over cell embeddings ≈ soft edges | high-attention pairs proposed as candidate `edge` cells (executor-validated) |
| Decision tree | soft/differentiable decision tree head over fingerprint features | hardened into a readable routing_rule / principle cell |

Sequenced staging (cheapest first, each stage useful on its own):
1. **Learned router head** — soft tree over the 5-axis fingerprint, hardened
   back into routing_rule cells; compares directly against the heuristic
   registry (measurable on the 13-domain sweep).
2. **Cell-set evaluator** — one forward pass approximating `executor.evaluate`
   over all active cells; pure speed win; its errors are information (they mark
   under-determined regions → targets for active experimentation, §5.1).
3. **KAN formula learner** (`adapters/kan_adapter.py`) — learnable edge
   activations pruned to symbolic formulas; the closest existing artifact to
   "a neural unit that collapses into a formula."
4. **Full uniform unit** — only if 1–3 each pass their round-trip tests.

Gate: R1 starts only after M4 (the discrete scientist loop must work first,
because it is both the training-data generator and the verification harness
for the neural unit).

---

## 6. Communication & HITL (the differentiator)

1. **Lab report generator** (`cognition/lab_report.py`): compiled from the
   event stream per run — fingerprint, route chosen and why, hypotheses formed,
   experiments run with EIG justification, confirmed/refuted with evidence
   trials, recovered laws in original units, belief changes (with trajectories,
   thanks to decay-as-event). Markdown artifact in `results/reports/`.
2. **Principle review UI** (extend existing webui): list cells with
   `to_readable()` + provenance; accept / edit / reject / annotate between runs;
   every action is a `human_edit`/`human_reject` event. **Undo = replay to
   previous version** (free, given the store).
3. **The HITL study** — the first real test of the surviving claim: human+UIL
   vs UIL-alone on a discovery task; measure recovery rate, time-to-law, and
   error-correction latency.

---

## 7. Benchmarks & success criteria

North star: equation discovery with extrapolation splits — LLM-SRBench-style
tasks + Feynman subset with train-box/test-shell splits. Ordinary tabular is a
routing parity check, not a battleground.

| Milestone | Acceptance criterion |
|---|---|
| M0 epistemology | regret/mining on K-fold; miner produces >0 principles at n=60; multiclass runs green |
| M1 substrate | VersionStore replay reconstructs network state at arbitrary stage byte-identically; decay trajectories queryable |
| M2 perception | 5-axis fingerprint + invertible scaling; D5 detected on a synthetic AR(1) dataset; router thresholds re-tuned |
| M3 routing | router = fit() step 0; routing rules visible in principle hierarchy with provenance; parity (±0.01 held-out) with best single learner across the 13-domain sweep |
| M4 scientist | active EIG intervention beats passive on hypothesis-resolution trials-to-confirm; symbolic + refit_constants recovers `sin(3x)`-class laws; with LLM enabled, exact-recovery rate on Feynman-subset ≥ engine-only baseline |
| M5 HITL | lab report auto-generated per run; review UI round-trips edit→event→replay; HITL pilot protocol executed |
| R1 neural unit (post-M4, parallel to M5) | each stage passes the Distillation Round-Trip: learned router ≥ heuristic on 13-domain sweep AND hardened rules readable; evaluator disagreement rate < 5% with useful error-mining; KAN-extracted formula matches symbolic recovery on ≥1 Feynman task |

Order of work: **M0 → M1 → M2 → M3 → M4 → M5**, with **R1 gated behind M4.**
M0/M1 first — every later claim depends on honest validation and lossless
history.

## 8. Do-not-do list

- Do not re-run the falsified accuracy ablations (answer is stable).
- Do not make a neural unit the *memory substrate* (destroys editability — the
  surviving claim). Neural units are welcome as computational organs under the
  Distillation Round-Trip condition (§5.4) — ground truth stays in the
  discrete VersionStore.
- Do not store p-values in fingerprints; effect sizes only.
- Do not let any scaler/miner/promotion step see test folds.
- Do not let the LLM write state directly; proposals only, executor-validated.
- Build in `figpso_uil_ai`, not the user's copy.
