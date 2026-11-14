# Adaptive Knowledge Network — Design & Roadmap (2026-07-24)

**Status: DESIGN ONLY. Nothing here is built yet.** This document captures a
new model architecture (a network of decision trees linked by evidence-carrying
pointer-edges, laid out on the knowledge graph) and the sequencing that puts it
in context. It is deliberately forward-looking: the current codebase's
foundation is not yet measurement-clean, and this architecture is a **parallel
track** to be built *after* that foundation is finished — see §10.

Working name for the architecture: **KDN — Knowledge-routed Decision Network.**
The name is provisional; rename freely.

---

## 0. What this document is, and isn't

- It **is**: the full design of the KDN idea, an honest map of how much of it
  reuses machinery that already exists, the invariants it must preserve, the
  quality-control case for it, the hard problems that must be solved, the
  smallest version worth building first, and the roadmap that says when.
- It **isn't**: a claim that any of it works, or a green light to build it
  before the foundation is done. Every "will" below means "is designed to,"
  not "has been shown to."

---

## 1. The core idea

A traditional decision tree ends each path in a **leaf** that outputs a class.
In the KDN, a leaf does not have to be terminal — it can be a **pointer** (an
edge) to another node, and each node holds **its own decision tree**. The trees
chain together through pointer-edges into one network that is not bounded by any
single node: it can grow without a fixed size.

A data point flows in, routes down a node's tree, and at the leaf either emits a
prediction *or* follows a pointer to the next node's tree, and so on, until a
prediction falls out. The pointers are the knowledge graph's edges — directed,
typed, weighted, and carrying evidence (`leads_to` / strength / confidence /
times-used) — and they behave like the connections in a neural network (weighted
flow) while being maintained by the KG's own evidence rules rather than by
gradients.

In one sentence: **KDN is a decision graph whose nodes are decision trees and
whose routing edges are the knowledge graph's evidence-carrying, self-maintaining
pointers.**

---

## 2. How the pieces map to the three names the user has always used

| The user's term | In the KDN | In today's code |
|---|---|---|
| **KG** (knowledge graph) | the topology + the edge lifecycle (birth, reinforce, decay) | `cognition/editable_network.py` |
| **DT** (decision tree) | the payload *inside* each node — the node's memory, stored as a tree instead of flat fields | grown today by `uil_miner.py` (DecisionTreeRegressor) and the rule basis (`symbolic_formula_adapter.py`, RandomForest trees) |
| **NN** (neural network) | the *forward-flow* principle — weighted connections, activation along edges, a readout | the neural basis + the weighted attention traversal already in `editable_network.py` |

The KDN is the **synthesis**: KG gives the shape and the edge rules, DT is what
lives in each node, NN is how signal flows across the shape.

---

## 3. The architecture in detail

### 3.1 Node = a decision tree (the node's memory, stored differently)

Today a `NetworkNode` stores its knowledge as flat fields: `conditions`,
`action_override`, `confidence`, `tier`, `evidence_count`. In the KDN, a node
stores its knowledge **as a decision tree** — the same information, represented
as a tree so it fits the DT criteria and can route. The node's flat fields
become the tree's summary/metadata; the tree is the working memory.

Crucially, the tree in a node is **small and local** — it is not the whole
model. The whole model is the *graph of* trees. This is what "not constrained to
one node — it can be infinitely big" means: no single tree has to be large,
because depth comes from *chaining* trees through pointers, not from growing one
tree deep.

### 3.2 Edge = an evidence-carrying pointer

The KDN edge is the existing `NetworkEdge`, used as a pointer. It already
stores everything the user listed:

- **type** (`leads_to` / `strengthens` / `contradicts` / `depends_on`) — how the
  target relates to the source,
- **weight** — strength of the link, [0,1], starts 0.5,
- **times_traversed / times_helped** — "number of times used" and how often it
  helped,
- **evidence_count**,
- **formula_id** — an edge can already carry a *computation*, not just a
  pointer; this is the hook that becomes "this edge routes to the next tree."

And it is **already a pointer**: nodes are addressed by id, edges are
`(source_id → target_id)` held in adjacency lists (`_adj`, `_rev_adj`). The
pointer structure exists today; the KDN drives *routing* through it.

### 3.3 Forward pass (routing)

A data point `x` enters at one or more entry nodes. At each node:

1. the node's tree evaluates `x` (plus any signal arriving on incoming edges),
2. the tree's active leaf selects an **outgoing pointer** (or emits a
   prediction),
3. signal flows along the chosen pointer(s), scaled by the edge's
   `weight × help_ratio` (the "causal gate" that already exists in
   `traverse_with_attention`), and by edge type (STRENGTHENS boosts ×1.3,
   CONTRADICTS suppresses ×0.1 — both already implemented),
4. repeat at the next node.

The prediction is the accumulation of what the visited trees contributed, each
weighted by the strength of the path that reached it.

### 3.4 Termination (the KDN must halt)

Pointer-graphs have cycles, so routing must be stopped deliberately. **Reuse the
loop-decay that already exists** for exactly this reason:
`LOOP_DECAY_FACTOR = 0.5` (each revisit halves the carried weight) and
`LOOP_CUTOFF = 0.01` (flow below this stops). Plus a hard `max_depth`. A path's
influence decays each hop, so an infinite route contributes vanishing weight and
is cut — no special cycle-detection needed beyond what's there.

### 3.5 The open design decision: what flows on an edge

Two coherent choices, and they define the model's character:

- **Hard routing (pure decision-DAG):** exactly one path fires per input; the
  next tree sees the raw row. Most **readable** (a single legible trace),
  lower capacity.
- **Soft/accumulating (boosting/MoE hybrid):** multiple pointers fire with
  weights; each tree contributes to a running prediction (boosting-style on the
  residual). More **capacity**, still traceable but the trace is a weighted set
  of paths rather than one.

Recommendation for v1: **hard routing** — it maximizes readability (the
differentiator) and is the simpler thing to get correct and measurable. Revisit
soft routing only if v1's capacity is measurably short.

### 3.6 Growth — constructive, not gradient-based

Decision trees are **not differentiable**, so the KDN cannot be trained by
backprop. It grows the way the KG already grows — constructively and by
evidence:

- **Node/tree growth:** grow (or regrow) a node's tree on the data (or residual)
  that actually reaches it.
- **Edge (pointer) birth:** the existing `discover_relationships` — when a leaf
  of one node reliably co-activates with another node *and that co-activation
  helps* (lift > 2× base rate), a STRENGTHENS pointer is born; lift < 0.5× makes
  a CONTRADICTS pointer.
- **Reinforcement:** the existing `record_traversal_outcome` — a pointer on a
  helpful path gains weight (+0.01), on a harmful path loses it (−0.02, twice as
  fast: punishment outpaces reward, a deliberate stability choice).
- **Decay/prune:** the existing `apply_edge_decay` — a well-tried pointer
  (`times_traversed ≥ 10`) with `help_ratio < 0.2` is archived and removed.

This is greedy, evidence-driven growth. It is deterministic and readable — but,
honestly, greedy growth can get stuck in ways a gradient-trained net does not
(§8).

### 3.7 "Infinitely big" in principle, bounded by decay in practice

Unbounded growth is the aspiration; **grows-and-prunes** is the implementation.
Uncapped growth is precisely what has bitten this codebase before (the
142-minute KG traversal explosion, the 60 GB event store, the formula-birth
runaway). The KDN stays lifelong-open *because* it decays and caps, not despite
it. The existing size ceilings (≈4000 edges / 300 nodes in the working graph,
with everything else archived losslessly) are the safety rails that make
"infinite" survivable. Keep them.

---

## 4. What already exists (the reuse map)

This is closer to *wiring* than to a from-scratch build. Grounded in the code:

| KDN part | Already in the code? | Where |
|---|---|---|
| Pointer graph (nodes by id, `src→tgt` edges, adjacency) | **Yes** | `editable_network.py` — `_adj`, `_rebuild_adjacency` |
| Typed, weighted, evidence-carrying edges | **Yes** | `NetworkEdge` (type, weight, times_traversed, times_helped, formula_id) |
| Edge birth from co-activation + lift typing | **Yes** | `discover_relationships` |
| Outcome-based reinforce/punish | **Yes** | `record_traversal_outcome` (+0.01 / −0.02) |
| Decay + archive of useless edges | **Yes** | `apply_edge_decay` |
| Weighted routing with type-gates + loop-decay | **Yes** | `traverse_with_attention` (×1.3 / ×0.1, LOOP_DECAY/CUTOFF) |
| Size caps + lossless archive | **Yes** | working-graph caps, `KnowledgeArchive` |
| Decision trees to put in nodes | **Yes** | `uil_miner.py`, rule/stump basis in `symbolic_formula_adapter.py` |
| A linear readout to turn contributions into a prediction | **Yes** | `sparse_linear` L1 solve |

**The genuinely new glue (the ~30% that must be built):**

1. **DT-as-node-payload** — store a decision tree inside a `NetworkNode` as its
   memory, with (de)serialization that keeps replay byte-identical.
2. **DT-as-router** — make traversal driven by each node's tree (the leaf picks
   the outgoing pointer) instead of by fixed `conditions`.
3. **Prediction assembly** — collect the visited trees' contributions along the
   routed path(s) into a final prediction (hard-route: last/dominant tree;
   soft-route: weighted sum / boosting accumulation).

---

## 5. The three invariants — and how the KDN keeps them

These are non-negotiable for UIL. The KDN is designed to keep all three:

- **Transparency.** Every node is a readable hard-split tree; every pointer is a
  named, typed, weighted, evidence-tagged edge. The path a data point takes is a
  legible trace (see §6).
- **Editability.** A human can edit a node's tree, cut a pointer, or re-weight an
  edge by hand, and — because the graph is event-sourced (`VersionStore`) — the
  edit survives replay. (HITL plumbing is already verified end-to-end for the
  existing graph.)
- **Reproducibility.** Trees are deterministic given a seed; edge updates are
  deterministic; routing is deterministic. The whole forward pass and growth
  step replay byte-identically — *provided* the same seeding discipline that
  fixed bug #31 is applied to every tree the KDN grows (explicit `random_state`,
  never `None`).

---

## 6. Why this may help most for QC (the user's insight, developed)

The user's intuition is that this architecture helps quality-control the most.
It is one of the strongest arguments *for* the design. Concretely, a network of
readable trees linked by evidence-carrying pointers gives QC handles that a
black-box model cannot:

1. **Per-node testability.** Each node's tree is a small, self-contained unit
   you can unit-test in isolation — feed it inputs, assert its outputs — without
   running the whole network. You cannot meaningfully QC one neuron of an MLP;
   you can QC one tree of a KDN.
2. **Root-cause inference traces.** For any prediction, you get the exact path:
   which trees fired, which pointers routed, at what strength and with how much
   evidence. When the model is wrong, you can see *where* it went wrong — which
   node or which pointer — not merely *that* it was wrong. That is root-cause QC,
   not just error detection.
3. **Edge evidence as automatic QC flags.** Every pointer carries `help_ratio`
   and `times_traversed`. A pointer that is **hot but unreliable** (high use, low
   help) is a self-surfacing QC alert: a frequently-taken path that is often
   wrong. These can be listed for human review automatically.
4. **Coverage QC.** Because routing is explicit, you can measure which nodes and
   pointers are never or rarely exercised — "dead reasoning" detection, the model
   analogue of test coverage. Dead paths are either prunable or a sign the
   growth missed something.
5. **Correctable *and persistent*.** QC is not only detection here — a bad node
   or pointer can be edited/cut by hand, and the correction survives replay.
   Detection that leads to a durable fix.
6. **Reproducible findings.** Determinism means a QC finding replays exactly —
   no flaky "it was wrong once." Every defect is reproducible by construction,
   which is the precondition for fixing it with confidence.

This QC story is, arguably, a better headline than raw accuracy — it is the same
"durable, correctable, inspectable reasoning" that UIL's strongest standing
claims (extrapolation, verified HITL) are about.

---

## 7. Literature grounding (this is a real design space)

The KDN is a novel synthesis, but each ingredient is grounded — it is not
reinventing something badly:

- **Decision Jungles** (Shotton et al., 2013) — decision *DAGs* where branches
  merge and point onward instead of every leaf being terminal. This is the
  "leaves are pointers" idea, published.
- **Hierarchical Mixture of Experts** (Jordan & Jacobs, 1994) — a tree of
  routers sending inputs to expert sub-models.
- **Mixture-of-Experts routing** (the mechanism inside modern large models) — a
  learned router sends each input through a *subset* of the network. In the KDN,
  the DT splits are the router, the pointer-edges are the routing, the DT-nodes
  are the experts.
- **Deep Forest / gcForest** (Zhou & Feng, 2017) — cascades of tree ensembles
  that pass outputs forward like layers, no backprop.

What is genuinely original in the KDN: the routing edges are **not** a fixed
softmax gate — they are the KG's **evidence-driven, self-maintaining** edges
(co-activation birth, lift-typing, asymmetric reinforcement, help-ratio gating,
decay). Routing a tree-network with an evidence-graph is the new part.

---

## 8. Hard problems and honest risks

Named so they are solved deliberately, not discovered late:

1. **Non-differentiability.** No backprop through hard trees → growth is greedy
   and constructive. Greedy growth can get stuck in local structure a
   gradient-trained net would escape. Mitigation: the edge decay/rebirth cycle
   gives it a way to undo bad structure over time, but this is slower than
   gradient descent and must be measured, not assumed.
2. **Termination / cycles.** Handled by reused loop-decay + max_depth (§3.4), but
   must be tested against adversarial topologies (dense cycles).
3. **Unbounded growth eating itself.** The exact failure mode this codebase has
   hit before. Caps + decay are mandatory, not optional; "infinite" is always
   "bounded by decay."
4. **Determinism debt.** Every tree the KDN grows must get an explicit
   `random_state` (bug #31's lesson). One unseeded tree makes the whole network
   non-replayable and silently breaks QC (findings stop reproducing).
5. **Measurability.** This is a *new model*, not a probe. It is only trustworthy
   once it (a) beats `sparse_linear` on a real benchmark and (b) shows
   **CONNECTED** in the existence audit (which would also, finally, prove the
   knowledge graph reaches the model — today it is INERT). Building it on an
   ungraded foundation risks not being able to tell "good" from "unmeasurable."

---

## 9. The smallest measurable v1

Do **not** build the full vision first. The smallest thing that tests the core
claim:

- A **small KG subgraph** (a handful of nodes) as the topology.
- Each node holds **one shallow decision tree** (reuse the miner's / rule
  basis's tree-growing).
- **Hard routing** (§3.5): one path fires; a leaf either emits a class or follows
  one outgoing pointer, scaled by the existing `weight × help_ratio` gate.
- Pointer birth/reinforce/decay: reuse `discover_relationships`,
  `record_traversal_outcome`, `apply_edge_decay` unchanged.
- Wrap it as a `ClassifierAdapter` (same interface as the other classifiers) so
  it drops into the existing harness.

**Two measurements decide v1:**
1. Does the KDN classifier match or beat `sparse_linear` on the standard
   cross-seed benchmark (breast_cancer / wine / digits / spambase / sonar /
   ionosphere), quoted as mean ± std, never a single split?
2. Does the existence audit show the graph **CONNECTED** when the KDN uses it —
   i.e., does removing the graph change the output? (Today: INERT.)

If (2) is yes even where (1) is only a tie, that is still a landmark: it is the
first time the cognitive layer would provably reach the prediction.

---

## 10. Roadmap — sequencing (the user's plan, made explicit)

The agreed order. The KDN is a **parallel track**; the critical path is finishing
and grading the foundation first.

### Critical path — finish the current work, then test

- **Phase 0 — Grade the judge fix.** ~~Run the paired classic baseline~~
  **DONE (2026-07-24):** nested-CV race selection graded mean +0.0080 vs
  classic (stays opt-in; ionosphere −0.047 open). Follow-on incumbent-margin
  Gate 2 at 0.01 **FAILED** (ionosphere untouched, breast_cancer win dulled) —
  see `CLAUDE.md` E7 and `results/gate2_incumbent_margin/SUMMARY.md`.
  Phase 0 is clear; race stays opt-in with margin 0.
- **Phase 1 — Process every hardcoded value through its category, one by one.**
  Using `HARDCODED_VALUE_AUDIT_QA.md`'s six categories. **Important honest
  principle: "make all hardcoded values dynamic" does NOT mean dynamize all
  ~584.** Some are *safety rails* (Category A — the crash guards, size caps,
  numerical epsilons) and dynamizing them would reintroduce the exact failures
  they prevent. The real task is: take each value, decide its category, and then
  *dynamize the ones that should be, deliberately keep the ones that shouldn't
  (with usage logging), and gather data for the ones that need it first.*
  - Category C (duplicated/inconsistent) — **partly done** (guidance ratchet,
    docstring). Finish any remaining. Zero risk, no data needed.
  - Category A (safety rails) — keep fixed, add a counter each time one clips, so
    there is real data on how often each binds.
  - Category E (semi-evidenced thresholds) — keep, log every decision vs outcome.
  - Category D (hand-picked weighted-sum formulas) — needs the research program's
    variation+evidence first; do not hand-guess new weights.
  - Category F (arbitrary magic numbers) — candidates for PBT-style *evolving*
    coefficients once the variation infrastructure exists.
- **Phase 2 — Complete the research-program changes.** Steps 2–8 of
  `RESEARCH_FAILURE_ANALYSIS.md`: default-path architecture variation (PBT),
  partial-pooled principle promotion, utility-gated formula admission
  (DreamCoder), EBM-style cyclic fit, Thresholdout instrument, TabPFN ceiling
  probe, KAN basis. Each gated, each preserving the three invariants.
- **Phase 3 — Test the completed project.** Full cross-seed + existence-audit +
  regression suite on the finished foundation, so every number means something.

### Parallel track — the KDN

Designed here now; built alongside once Phase 0 is clear. The user drives this
in parallel. Its own gates are §9's two measurements. It is expected to be the
strongest QC vehicle (§6), and the first architecture that could move the
knowledge graph off INERT.

---

## 11. Open questions (to resolve before/while building)

1. Hard vs soft routing (§3.5) — start hard; measure whether capacity is short.
2. What a routed-to tree sees as input — raw row, or row + accumulated residual.
3. How a node decides *which* outgoing pointer at a leaf — by edge weight? by a
   small learned gate? by the tree's own leaf identity?
4. Entry-node selection — one fixed root, or fingerprint-chosen entry?
5. How KDN growth interacts with the existing per-stage optimizer loop — is the
   KDN a *classifier* the optimizer trains, or does it *replace* parts of the
   loop?
6. Does the KDN subsume the `sparse_linear` basis families, or sit beside them as
   another raced architecture?

---

## 12. Naming

Working name: **KDN — Knowledge-routed Decision Network.** Alternatives
considered: Pointer Forest, Decision Graph Network, Knowledge Tree Network. The
user's brainchild — rename as preferred.

---

*This is a living design document. Update it as decisions in §11 are made and as
the roadmap phases complete. Nothing in it is built; it is the plan and the
architecture, written down precisely so it can be built well rather than fast.*

---

## 13. Addendum (added 2026-08-12) — acceptance-gate update and the transformer track

Addition only; §1–§12 above are unchanged.

### 13.1 The §9 v1 acceptance gate needs a qualifier now

§9 asks two questions of the smallest measurable KDN: does it match/beat
`sparse_linear` on the standard benchmark, and does the existence audit show
the graph **CONNECTED** when KDN uses it, "i.e., does removing the graph
change the output? (Today: INERT.)"

That second question is no longer discriminating as written. Since this
document was written (2026-07-24):

- `results/existence_sweep/FINDINGS.md` (2026-08-05) confirms INERT at
  *default* flags across 36 subsystem-removals, 2 datasets, 2 budgets.
- `results/existence_mined/FINDINGS.md` (2026-08-06) shows that with mining
  forced on (`explore_architecture` + `adaptive_min_evidence` +
  `connect_cognition` + `utility_gated_formulas` + `signal_translator`), the
  graph is **CONNECTED** — 0–2% of predictions move — but the accompanying
  accuracy deltas are small and net negative (±0.009), matching the steps-2+3
  Gate 2 result (−0.0079, 0 improvements) by an independent instrument.

So "removing the graph changes the output" is now **achievable without KDN**,
by turning on flags that already exist. A KDN v1 that only clears that bar has
not shown anything new. §9's second acceptance test should read: **does
removing the graph change the output, AND does the KDN path improve on (or at
minimum not degrade) accuracy relative to the same flags without KDN** — i.e.,
connected is necessary, not sufficient.

### 13.2 §10 Phase 2 status, corrected

§10 lists research steps 2–8 from `RESEARCH_FAILURE_ANALYSIS.md` as pending
work on the critical path. As of the dates below, several are no longer
pending:

- Step 4 (utility-gated formula admission): implemented, Gate 2 **failed**
  (mean Δ −0.0013), stays default OFF. `results/gate2_utility/SUMMARY.md`.
- Step 5 (EBM cyclic fit): implemented, Gate 2 measured (mean Δ +0.0014, sonar
  +0.008), stays default OFF, race-axis only. `results/gate2_ebm/SUMMARY.md`.
- Step 6 (Thresholdout): implemented, default ON.
- Step 7 (TabPFN ceiling probe): implemented, registered behind an
  optional-dependency guard.
- Step 8 (KAN basis): implemented, Gate 2 **failed** (mean Δ −0.0022, sonar
  −0.0158), stays default OFF, race-axis only. `results/gate2_kan/SUMMARY.md`.

Steps 2 and 3 (default-path architecture exploration; partial-pooled
principle promotion) were also run together at 40 stages and **failed** Gate 2
(mean Δ −0.0079 across 4 datasets, 0 improvements) — see
`results/existence_mined/FINDINGS.md` and CLAUDE.md's "Connected layer still
no accuracy" entry. `adaptive_min_evidence` stays default OFF.

Net effect on §10's sequencing: the critical path described as "finish
research steps 2–8, then test" has been run. The honest status is that most of
Phase 2 is **done and did not help** — which is itself the finding that
motivated `COGNITIVE_ENGINE_RESCUE.md`'s re-aim (§13.4 below), not a reason to
keep iterating inside the old job.

### 13.3 The KDN name now has two referents — flagged, not resolved

This document's KDN ("Knowledge-**routed Decision** Network," §"Naming"
above) and the transformer-facing litsurvey's KDN
(`06_TRANSPARENT_EDITABLE_NEURAL_SUBSTRATE_LITSURVEY.md` / `.NEW.md`,
"Knowledge-**Directed** Network") are different architectures that share an
acronym and are cross-referenced from `COGNITIVE_ENGINE_RESCUE.md` and
`academic_v2/05_NOVELTY_IP_ANALYSIS_AND_FUTURE_RESEARCH_ROADMAP.md` Cluster 5
without disambiguation. This document is not renamed here — renaming is a
framing decision, out of scope for an addition-only edit — but readers
following both threads should treat "KDN" as ambiguous until one document
picks a different name or both are explicitly merged.

### 13.4 The transformer/KDN swap workstream, and how it reuses this design

A concrete plan for embedding KDN inside a small transformer's FFN sublayer
was written up 2026-08-12 as **Workstream 2** (plan file
`synchronous-watching-seal.md`). It is built directly on this document's own
reuse map (§4) and invariants (§5), so it is recorded here rather than only in
the litsurvey addendum.

**What carries over unchanged:**

- §4's reuse table — `NetworkNode`, `NetworkEdge`, `discover_relationships`,
  `record_traversal_outcome`, `apply_edge_decay`, `traverse_with_attention`,
  `KnowledgeArchive`, and the working-graph size caps — is reused as-is inside
  the transformer FFN slot. The three genuinely new pieces this document names
  (DT-as-node-payload, DT-as-router, prediction assembly) are exactly what the
  transformer workstream's "Step 0" and "Step 1" build; nothing additional is
  invented for the transformer case.
- §5's three invariants (transparency, editability, reproducibility) apply
  unchanged. Reproducibility in particular — "provided the same seeding
  discipline that fixed bug #31 is applied to every tree the KDN grows" — is
  the literal content of the transformer workstream's first measurement (does
  the same input take the same path every time, across repeated runs).
- §3.4's termination mechanism (`LOOP_DECAY_FACTOR`, `LOOP_CUTOFF`, hard
  `max_depth`) and §3.7's grows-and-prunes discipline are used unchanged for
  KDN nodes grown inside a transformer block.

**What is new, because a transformer block imposes constraints this document
does not consider:**

- **The math has to match the FFN it replaces.** A transformer FFN sublayer
  is a bounded, memoryless function `R^d → R^d`. Where this document treats a
  KDN node's tree as classifying/predicting from tabular rows, the
  transformer version fits activation vectors, and the readout at a terminal
  leaf is a `d`-dimensional contribution, not a scalar or class label.
  Training data for these node-trees comes from the transformer's own block
  activations (every token position, every layer, every training example) —
  which is orders of magnitude denser than the tabular case's per-run journal,
  a fact `COGNITIVE_ENGINE_RESCUE.md` names as inverting the sample-complexity
  problem that limits this document's tabular v1.
- **Hard vs soft routing is decided the same way §3.5 already recommends** —
  hard routing first, for the same readability reason — but the transformer
  case adds a specific reason to be careful about when to relax it: §5.2 of
  the litsurvey (not present in this document) calls soft routing "the
  scientific stress test of the thesis," because hard routing can manufacture
  attribution by architectural constraint rather than deliver it from the
  substrate. This document's "revisit soft routing only if v1's capacity is
  measurably short" (§3.5) becomes, in the transformer workstream, an
  explicit later step (Step 4) gated on the hard-routing step passing first.
- **A named, versioned identity per node**, already implied by this document's
  identity-persistence framing, becomes an explicit requirement for the
  transformer's live inspector: node ids must be stable strings assigned at
  construction (e.g. `block2_ffn_root`, `block2_ffn_R.left.left`) so a human
  reading a real-time trace can name what fired, not just see that something
  did.
- **§9's two-measurement acceptance test gets a transformer-scale analogue**
  before the full swap is attempted: a ~100-node KDN wired into one block of a
  tiny (~1–5M parameter) transformer, answering three questions locally —
  is the recorded path exactly what the forward pass computed (this
  document's transparency invariant, made falsifiable); is the path
  deterministic for a repeated identical input (this document's
  reproducibility invariant, made falsifiable); and, grouped by input class,
  is there a low-entropy dominant path that a human can name (a direct test of
  §6's QC claims 2 and 4 above — root-cause traces and coverage QC — at a
  scale small enough to check by hand). A live UI, reusing the project's
  existing webui service and design tokens rather than a new stack, is
  scoped as part of this step so the questions can be answered by watching the
  graph, not only by reading a log.
- **A control arm this document does not specify.** Per
  `COGNITIVE_ENGINE_RESCUE.md`'s pre-registered falsification condition for the
  tabular memory experiment ("if Arm B does not beat the lookup table, the
  cognitive engine is not earning its complexity, and the lookup table
  ships"), the transformer swap experiment is required to include a plain
  distilled-MLP control on the same paired activations. §9's "beats
  sparse_linear" bar becomes, for the transformer case, **"beats a plain
  distilled MLP on edit locality, not merely matches the FFN it replaced."**

§10's "Parallel track — the KDN" already frames this design as parallel to
the tabular foundation work and built alongside once Phase 0 was clear. The
transformer FFN slot is the second target substrate for that same parallel
track, not a new track — it inherits every invariant and reuse decision this
document already made, and adds only what a differentiable neighbor
(attention) requires.
