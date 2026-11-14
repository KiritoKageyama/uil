# Audit + correctness fixes: the academic_v2 master series and the five Cursor-touched docs

## Context

Two document sets assert claims that the August 2026 measurements overturned.

**Set A — `academic_v2/` Volumes I–V** (built 2026-08-01; the master doc, the
hypothesis/positioning doc, and the master plan/IP doc). These are the
**outward-facing** documents — abstracts, patent framing, publication roadmap —
and they are built on four claims that are now falsified or unsupported. None of
them received the correction that Volume VI (the litsurvey) got.

**Set B — the five docs Cursor's 2026-08-11 session touched** (`CLAUDE.md`,
`LITERATURE_SURVEY.md`, `RESEARCH_FAILURE_ANALYSIS.md`,
`DAPTIVE_KNOWLEDGE_NETWORK_DESIGN.md`, `HARDCODED_VALUE_AUDIT_QA.md`). Same four
facts, plus internal contradictions and duplicated text.

Scope: **correctness only.** Fix false and stale claims, internal contradictions,
and duplication. Do not restructure, re-argue theses, or change framing. Every
replacement claim must trace to a file in `results/`.

> **Before editing Set A:** `academic_v2/01`–`05` are **symlinks** into
> `C:\Users\arinb\OneDrive\Desktop\New folder (3)\academic_v2\`. Editing them
> writes to OneDrive and will sync. Volume 06 in that folder is a **separate,
> older copy** of the litsurvey — the corrected `.NEW.md` on the Desktop is not
> linked into the repo. Confirm this is intended before the first write.

---

## The four facts that changed

Fix these consistently across both sets — same wording, same cited source.

| # | Claim as written | What was measured | Source |
|---|---|---|---|
| **A** | "Matches / parity with tuned XGBoost", "94.9% vs 94.4%" | **Never demonstrated.** Paired, identical splits: Δ −0.0046, CI [−0.0289, +0.0197] — **79 cells needed, 9 run: UNDERPOWERED.** Plain logreg wins 3/9 cells | `results/compute_cost/HEAD_TO_HEAD.md` |
| **B** | "10× to 100× overhead" | Wrong in both directions: cognitive layer **0.95×** (free); search loop **~3900×** one plain fit | `results/compute_cost/SUMMARY.md` |
| **C** | "R² 0.992 vs XGBoost −5.52 — superior OOD extrapolation" | **Beating XGBoost out of range is a theorem, not a finding** (trees are piecewise-constant). vs *extrapolating* baselines UIL wins **0 of 21 cells**; vs gplearn **4–17**, with **19 blow-ups vs 0** | `results/extrapolation/SUMMARY.md` |
| **D** | "Causal is the only subsystem that changed predictions" / "nine modules inert" | Causal is **INERT** in all 4 sweep cells and both 40-stage mined cells. Connected: **principles + formulas** (both cells), **network** (one). Correct statement: *inert by default; connected but ineffective when mining is on* | `results/existence_{sweep,mined}/FINDINGS.md` |

**Addition, not correction:** HITL is no longer unmeasured
(`results/hitl_law_edit/SUMMARY.md`). Any new text must carry both source caveats:
the expert is a **program with oracle knowledge**, and **33 of 180 shift cells
were excluded** where the law was entirely the artifact.

---

## Set A — `academic_v2/` (priority; outward-facing)

### Volume III — `03_EMPIRICAL_EVALUATION_AND_FAILURE_MODE_ANALYSIS.md`
The empirical volume; most defects and they propagate to the others.

- **Abstract + §2.1**: "matches the accuracy of state-of-the-art gradient boosted
  trees" and the **Spambase 94.9% vs 94.4%** comparison → fact **A**. This single
  pair also violates the project's own hard invariant — *never quote a single
  split as "the" accuracy* — so it must go regardless of fact A.
- **Abstract + §2.2 + the ASCII flowchart (lines 86–119)**: R² 0.992 / −5.52 →
  fact **C**. Note the flowchart *explains the theorem* ("splits space into step
  bounds" → "constant step extrapolation") and then presents it as UIL's win. The
  explanation is correct; the conclusion drawn from it is not.
- **§3.1** "$10\times$ to $100\times$" → fact **B**.
- **§3.1 / Abstract** "Δ = −0.0132": per `results/rescore_gates/REPORT.md` this is
  one of **three verdicts that cannot be reproduced** — only `acc_mean`/`acc_std`
  were stored, no per-cell rows. It is the volume's headline number. Either mark
  it unreproducible or replace it with a figure that has per-cell backing.
- **§3.2 is internally inconsistent**: prose says "**nine** auxiliary modules were
  prediction-inert"; the table has **five** rows, and **two of the five are
  labelled not-inert** ("Rule Miner: Active", "Model Racing: Negative"). Rebuild
  the table from `existence_sweep`/`existence_mined` (9 subsystems × 4 cells at
  default, 3 of 9 connected with mining on).
- **§5 guideline 2** — *"Eliminate Inert Modules: run logging and graph updates
  asynchronously in background threads so they do not block core optimization."*
  Two problems: it does not follow from the finding (the modules don't change
  **predictions**; they were never measured as a bottleneck — and fact **B** shows
  cognition costs 0.95×, i.e. nothing), and **background threads would break both
  the determinism invariant (bug #31) and the VersionStore lock discipline
  (bug #11)**. This is the one recommendation in the series that is actively
  harmful if followed. Remove or rewrite.
- **§5 guideline 3** — "disable racing, use fixed CV" is superseded: the judge fix
  (select on `_extra_val_splits`) moved race from −0.0117 to **+0.0080**. The
  remedy is fixing the judge, not removing the race.
- **§4.3** attributes to Friedman & Popescu (2008) the claim that high-variance
  interaction terms are disproportionately penalised under uniform L1. Verify this
  is actually in the paper; UIL's E5 finding was measured empirically, not cited.
  If unverifiable, present it as UIL's own finding.
- **References**: Sutton's blog post is given a fabricated volume/issue
  ("*Incomplete Ideas (Blog)*, 13(1)"). Cite it as a blog post with URL and date.

### Volume I — `01_UIL_SYSTEM_MONOGRAPH.md` (the master doc)
- **Abstract**: "a carefully tuned readable learner can match the performance of
  complex ensemble models" → fact **A**; "superior prediction on structural tasks"
  → fact **C**.
- **Line 226** Spambase 94.9/94.4 → fact **A** (same edit as Vol III §2.1).
- **Line 227** "Superior Out-of-Distribution Extrapolation … R² ≈ 0.99 … trees fail
  completely" → fact **C**.
- **Line 231** "−0.0132" → same treatment as Vol III.

### Volume II — `02_THEORETICAL_FOUNDATIONS_AND_LITERATURE_SURVEY.md` (hypothesis/positioning)
- **Abstract**: "we establish why the system **matches** performance on readable
  classification tasks, **excels** in out-of-distribution physical extrapolation"
  → facts **A** and **C**. This sentence is the document's thesis, so the edit is
  load-bearing.
- **Line 42, diagram**: cell reads **"XGBoost: 94.9%"** — 94.9% is *UIL's* number;
  XGBoost's is 94.4%. The diagram attributes UIL's score to XGBoost. Fix
  regardless of fact A.
- **Line 49, Axis B** "Promising: Superior extrapolation (R² ≈ 0.99)" → fact **C**.
- **Line 50, Axis C** and **line 66** "measured parity with tree models validates…"
  → facts **A** and **D**.
- **Line 176** repeats the extrapolation claim in prose → fact **C**.

### Volume V — `05_NOVELTY_IP_ANALYSIS_AND_FUTURE_RESEARCH_ROADMAP.md` (master plan / IP)
- **Flowchart line 31** "Honest Parity with XGBoost" → fact **A**. The word
  *honest* is doing work here that the measurement does not support.
- **Clusters 4 and 5 are presented alongside built ones with no build-status
  marker.** Cluster 1–3 are built and tested; **Cluster 4 (4-Layer Memory) and
  Cluster 5 (KDN) are design-only** — `DAPTIVE_KNOWLEDGE_NETWORK_DESIGN.md` states
  "Status: DESIGN ONLY. Nothing here is built yet." For an IP document, reduction
  to practice is exactly the distinction that matters. Add a build-status column.
- **Cluster 2 contradicts the roadmap in the same document**: it claims
  "per-family regularization controls" as an existing mechanism, while **Stage 1 of
  the roadmap lists per-family multipliers as future work**. What exists is a
  single global `sparse_l1_c` relaxation (E5). Fix the cluster, keep the roadmap.
- **Cluster 4 "Prior Art Distance"** asserts that existing agent architectures use
  flat memory that suffers context contamination — unsourced. Either cite or soften.
- The **strongest available IP/novelty claim is missing**: the shift/iid/control
  editing protocol (`results/hitl_law_edit/`) and the fidelity + edit-exactness
  acceptance tests (`cognition/law_export.py`, `tests/test_law_fidelity.py`) are
  measured, unique, and absent from all five clusters. Adding them is *correctness*
  — they are built and tested, unlike Clusters 4–5.

### Volume IV — `04_HIERARCHICAL_MEMORY_AND_KNOWLEDGE_ROUTED_NETWORK_DESIGN.md`
- Clean. Contains none of the four stale facts; it is explicitly a design volume.
  Only change: note the **KDN acronym collision** (see Set B below).

---

## Set B — the five Cursor-touched docs

### `CLAUDE.md`
- Opening "Honest empirical status" asserts the XGBoost tie and "+0.004" while a
  later section correctly retires it → fact **A**; make the headline agree with the
  body.
- **Duplicated text, two places**: in *How to run*, the `existence_sweep` /
  `eval_extrapolation` / `compute_baseline` block appears twice; in *Pending work*,
  the bug #35 / #34 / #33 / "Underpowered gates" block appears twice. Delete the
  second copy of each.
- Old 2-cell existence table (causal CONNECTED) needs a superseded marker, matching
  how bug #30a is already handled.

### `LITERATURE_SURVEY.md` (dated 31 July — predates every August measurement)
- §0 axis A "**Strong** — primary positive result" → fact **A**; same edit in the
  §11 diagram, §11.4 thesis 1, §15 closing.
- §3.4 "R² ~0.99 vs XGBoost collapse — aligns" → fact **C**; also §11.4 thesis 2.
- §7.1 "causal was the subsystem that actually changed predictions" → fact **D**.
- §9.2 "Human benefit is **unmeasured**" → HITL result + both caveats. Keep the
  "still needs a study with participants" sentence — the source agrees.
- §13 gaps 2 and 6 are now **done** (identity tests → `existence_sweep`; symbolic
  OOD benchmark → `eval_extrapolation` vs gplearn). Record outcomes; four stay open.
- §14 table: add `COGNITIVE_ENGINE_RESCUE.md` and the `results/*/SUMMARY.md` set.
- **All ~30 citations verified correct — no citation changes.**

### `RESEARCH_FAILURE_ANALYSIS.md`
- F1: "10–100× compute" → fact **B**; "+0.004" → fact **A**; bullet 3 "already
  UIL's proven win: symbolic R² 0.99 vs XGBoost −5.5" → fact **C**.
- F6: "causal was the only subsystem ever measured CONNECTED" → fact **D**.
- **F9.1 contradicts the document's own program table**: F9 calls the TabPFN
  adapter "(unregistered)"; row 7 records that step 7 registered it. Verified at
  `core/classifier_registry.py:71-77`. Fix F9. Typo in the same line:
  "replacing/da complementing".
- **Program table row 1** is the only row with no recorded outcome — it was run
  (judge fix **+0.0080**, 4W/1T/1L, ionosphere −0.047 open; incumbent-margin Gate 2
  failed). Record it in the style of rows 4–8.
- Sources: the auto-sklearn entry links `arxiv.org/pdf/2202.10979`, which is not
  auto-sklearn. Replace with Feurer et al., NeurIPS 2015.
- F3 says "auto-sklearn 2.0 (2020)"; `LITERATURE_SURVEY.md` §4.1 says "JMLR 2022".
  Same work (arXiv 2020 / JMLR 2022) — use one form in both.

### `DAPTIVE_KNOWLEDGE_NETWORK_DESIGN.md`
- §8.5 and §9 gate 2 say the graph is "today INERT" → needs fact **D**'s qualifier
  (INERT at default flags; connected-but-useless with mining on). **This one changes
  a decision, not just a sentence**: without the qualifier the doc's own v1
  acceptance gate is already trivially passable and stops discriminating.
- §10 Phase 2 lists research steps 2–8 as pending; steps 4–8 are implemented with
  recorded gates. Sync with the program table in `RESEARCH_FAILURE_ANALYSIS.md`.
- **KDN acronym collision**: this doc = "Knowledge-**routed Decision** Network";
  Volume VI litsurvey = "Knowledge-**Directed** Network"; Volume V Cluster 5 uses
  the former. Different architectures, one acronym, cross-referenced from
  `COGNITIVE_ENGINE_RESCUE.md`. Add one disambiguating line to each doc. **Do not
  rename** — renaming is framing, out of scope.
- §7's novelty claim (evidence-driven self-maintaining routing edges vs a learned
  softmax gate, against Decision Jungles / gcForest / HME) is accurate. No change.

### `HARDCODED_VALUE_AUDIT_QA.md`
- **Headline counts verified correct** — Appendix A 449 values / 33 files + Appendix
  B ~135 / 21 files = **~584 across 54 files**, matching CLAUDE.md. No change.
- One stale claim near the Appendix B TabPFN entry: *"currently registered nowhere
  … these constants govern code that never runs today."* False since step 7 — same
  defect as F9; fix both together.
- **Scope note:** 222 KB / 3,200+ lines. I audited structure, headline counts,
  cross-references and the sections CLAUDE.md cites — **not** every constant entry.
  A line-by-line pass is a separate job; the plan should not imply otherwise.

---

## The six "what's done / what's planned" sections, reconciled

Six documents each carry a status-and-roadmap section, and **they disagree with
each other.** Applying your rule — *the most recent document is correct by data* —
gives this authority order, newest first:

| # | Document | Its status/plan section | Date | Authority |
|---|---|---|---|---|
| 1 | `results/*/SUMMARY.md`, `FINDINGS.md` | the measurements themselves | 08-05 / 08-06 | **ground truth** |
| 2 | `COGNITIVE_ENGINE_RESCUE.md` | "first experiment" A/B/C + "what to cut" / "what gets promoted" | 08-06 | **current plan of record** |
| 3 | `06_..._LITSURVEY.NEW.md` | §8 roadmap S0–S5 + §2 status table | 08-08 | current for the KC/KDN track |
| 4 | `CLAUDE.md` | "Pending work" + ✅ tables + bug list | rolling | current for engineering state |
| 5 | `academic_v2/05` | §3 four-stage roadmap + §4 publication strategy | 08-01 | **stale** |
| 6 | `RESEARCH_FAILURE_ANALYSIS.md` / `DAPTIVE_..._DESIGN.md` | 8-row program table / §10 Phases 0–3 | 07-23 / 07-24 | **stale** |

Three concrete conflicts follow, in descending order of consequence.

### 1. `academic_v2/05` Stage 2 proposes a fix that was tried and rejected

Stage 2 — *"Constrained Symbolic Extrapolation Bounds: introduce interval
arithmetic and boundary checks into the expression generator to prevent
division-by-zero errors and extreme output spikes"* — is **exactly** the
intervention that `results/extrapolation/GUARD_ABLATION.md` (2026-08-05)
measured and rejected. Verified in the artifact: reverting `MAX_MAGNITUDE` moves
gravity@1.5 from **−1.167 to 0.743**, while the tight clamp that halves blow-ups
(19→9) destroys those same working cases; **no setting of the four constants
dominates**, defaults were reverted, and the head-to-head against gplearn stays
**4–17 in every arm**.

The recorded diagnosis is that the gap is **term selection** — greedy residual
fitting picks terms that fit in-range with the wrong asymptotics — **not
arithmetic hygiene**. So the master plan's Stage 2 is not merely stale; it
schedules work that has been measured not to help. It should be replaced by the
selection problem (population search with parsimony pressure, as in gplearn), or
struck.

### 2. The current plan of record says to cut what the older roadmaps schedule

`COGNITIVE_ENGINE_RESCUE.md` (08-06) directs: **cut** the formula engine, cut
stress/regret as search signals, cut the PSO search loop; **promote** the causal
engine to edit-locality; and run the A/B/C memory experiment with a
pre-registered falsification condition (*if Arm B doesn't beat the lookup table,
ship the lookup table*).

`DAPTIVE_..._DESIGN.md` §10 (07-24) still schedules Phases 1–3 — processing all
~584 hardcoded values and completing research steps 2–8 — as the critical path
before anything else. `academic_v2/05` still schedules per-family regularization
and a KDN prototype. These are not reconcilable as written: one says finish the
foundation, the newest says most of that foundation is being cut.

Fix: mark Phases 1–3 and the four-stage roadmap as superseded, and point both at
`COGNITIVE_ENGINE_RESCUE.md` as the live plan. Do not delete them — they record
why the work was scheduled.

### 3. Two status claims are contradicted by the code

- **`06_..._LITSURVEY.NEW.md` §8 marks S1 (hard-routing TE-MLP) "In progress:
  designed, partially prototyped."** There is no TE-MLP code: zero hits for
  `te_mlp` / `TEMLP` / hard-routing across the package. The `KnowledgeCell` that
  exists (`cognition/knowledge_cell.py`) is the v22 knowledge-*governance*
  protocol — `CellVerdict`, `action_override`, `MutationProposal` — with no
  `forward()`, no `local_params`, no typed ports. On this document's own evidence
  ladder S1 is **Designed / Not started**.
- **`academic_v2/05` Cluster 2** claims per-family regularization controls exist,
  while **Stage 1 of the roadmap in the same document** lists them as future work.
  What exists is one global `sparse_l1_c`.

### What survives unchanged

- `academic_v2/05` **Stage 4** (build a KDN prototype) — consistent with the
  litsurvey S0–S5 and DAPTIVE's parallel track.
- `academic_v2/05` **§4 publication strategy** — targeting Volume III at a
  NeurIPS/ICML Datasets & Benchmarks track as a case study on saturated headroom
  and selection bias is the right venue for the negative results, **conditional on
  Volume III's numbers being corrected first**.
- `academic_v2/05` **Stage 3** (HITL user study) stays open, but its design must
  change: `results/hitl_law_edit/SUMMARY.md` already answers the channel question
  and shows iid/data-derived editing is null. A study should test **external
  knowledge under shift**, not iid second-guessing.

---

## Workstream 2 — Design the UIL/KDN transformer, then swap KDN in for the FFN of a small LLM

This is a **separate workstream from the correctness edits above**. It is design +
a proof-of-concept experiment, not a documentation fix. Keep the two workstreams
independent so a correctness pass can ship without waiting on the design.

### The shape of the problem, plainly stated

The litsurvey (`06_..._LITSURVEY.NEW.md` §6, T1–T8, §6.6) sketches what a
KC-augmented transformer should look like, but it does **not** design the
UIL/KDN version specifically. And the design that DAPTIVE gives — hard trees,
greedy constructive growth, non-differentiable — **cannot be dropped into a
transformer** as-is: a transformer FFN sublayer must be differentiable to be
trained end-to-end alongside attention. This is the acronym-collision problem
made concrete: DAPTIVE's KDN and the transformer FFN speak different math.

Two design choices resolve it, and only one is worth pursuing at PoC scale.

- **Option 1 — soft KDN.** Make routing edges differentiable weights (softmax
  gate over pointers, or straight-through estimator for the tree splits). Trains
  end-to-end with attention. **Loses** the interpretability handle DAPTIVE
  optimizes for — soft routing is exactly the "scientific stress test" the
  litsurvey names in §5.2, and it is where attribution can collapse.
- **Option 2 — frozen-KDN adapter.** Train the transformer with a standard FFN
  first. Then, per block, distill the FFN into a KDN using the transformer's
  own activations as (input, output) pairs, and swap the frozen KDN back in as
  the FFN replacement for that block. Attention is untouched. No gradient
  through the KDN. This is the cheapest, most honest first cut, and it is what
  the user described ("fitted as the replacement for the NN, for just checking
  and confirmation in small networks").

**Recommended v1: Option 2**, block-by-block, on a *tiny* transformer.

### Design targets — the transformer must be truly small

- **Parameter count**: ~1–5 M, not billions. `nanoGPT` scale (2 layers, 4 heads,
  d=128, ~2 M params) is the reference class. Both training and edit-locality
  measurement must be feasible on one machine; anything larger confounds the
  result with pretraining data leakage.
- **Task**: character-level or small-BPE next-token on a fully-controlled
  synthetic or small-natural corpus where the ground truth is inspectable
  (arithmetic, small grammar, TinyStories subset). No web-scraped data.
- **Two matched builds** on identical data / hyperparameters / compute:
  build A = standard FFN throughout; build B = KDN-replaced FFN in one or more
  blocks. This is the §6.6 protocol scoped down.

### KDN-as-FFN, the mechanics

An FFN sublayer is `y = W2 · σ(W1 · x + b1) + b2` with `x ∈ R^d`, hidden dim
`4d`. Its behaviour is a bounded, memoryless function `f: R^d → R^d`. The KDN
replacement must implement the same shape.

1. **Inputs**: block-input activations `x` from the trained FFN, harvested on
   the training corpus. This is the "data" the KDN is fit on — **abundant**
   (every token position × every layer × every training example), which
   directly answers the sample-complexity killer in `COGNITIVE_ENGINE_RESCUE.md`.
2. **Outputs**: the trained FFN's outputs on the same inputs. KDN is learning
   the FFN's function, not the language task.
3. **Node payload**: a shallow decision tree per KDN node routing on `x`
   coordinates or on projections of `x` (learned once, then frozen). Reuse the
   miner / rule-basis tree machinery.
4. **Pointer edges**: existing `NetworkEdge` with weight × help_ratio gating,
   already in `editable_network.py`. Termination via existing `LOOP_DECAY` /
   `LOOP_CUTOFF`.
5. **Readout**: at each terminal leaf, emit a `d`-vector — either a stored
   centroid (nearest-neighbour style) or a small L1-linear readout over `x`,
   consistent with the existing `sparse_linear` path. The output of the block
   is the accumulation of visited leaves' contributions, weighted by the
   evidence-carrying pointers, mirroring the additive readout that already
   works.

### What this experiment actually tests

Not "does KDN beat FFN." That is the wrong question at this scale, and it
recreates the accuracy trap the project has already lost.

The falsifiable claims are:

1. **Function replaceability.** After frozen-KDN swap-in, does the transformer's
   next-token perplexity on held-out data stay within `ε` of the original?
   Answer decides whether KDN can even *represent* an FFN's function to the
   fidelity needed. Metric borrowed from distillation literature.
2. **Attribution is real, not manufactured** (the §5.2 test). Under hard KDN
   routing, run the litsurvey §5.6 metric 1 (localization) on planted-fact
   probes. Report both hard-routing and softened-routing numbers; report the
   drop. A large drop means the interpretability came from the constraint, not
   from KDN.
3. **Edit locality against ROME/MEMIT** (§5.6 metric 2). A rank-one edit to a
   swapped KDN block vs the same edit on the original FFN, on the same fact
   probes. Fraction of target predictions changed correctly; fraction of
   unrelated predictions changed incorrectly. This is the one comparison the
   litsurvey names as decisive.
4. **Edit persistence and rollback** (§5.6 metrics 3, 4). VersionStore already
   supports both; the test is whether the KDN cell's identity survives a small
   amount of further training on top.

Explicitly **not** in scope for the PoC: catastrophic-interference-at-10k-edits
(§5.6 metric 5) — that is the S5 / LLM-scale test the litsurvey defers.

### Step 0 — v0 transparency probe: 100-node, local-only, three questions

Before the swap experiment, a much smaller probe that answers whether the
architecture actually delivers what it claims. **~100 KDN nodes total, runs on
one laptop, no training compute needed beyond the tiny transformer itself.**

The three questions it must answer, one metric each:

1. **Is the path truly transparent?** For an arbitrary input, can a human read
   off which nodes fired, in what order, at what pointer weight, and reproduce
   the prediction by hand from that trace alone? **Metric**: the fidelity round-
   trip from `tests/test_law_fidelity.py` F7, adapted to KDN — replay the trace,
   emit a prediction, compare byte-for-byte to the live forward pass. If they
   differ, the "transparent" label is manufactured.
2. **Is the path consistent for the same query?** Feed the *same input* through
   the model N times (N = 100). Does it take the same path every time? **Metric**:
   fraction of runs producing identical path signatures (the ordered list of
   `(node_id, pointer_id)` visited). Anything below 1.00 means there is hidden
   nondeterminism to find and kill — bug #31 was exactly this shape.
3. **Is there consistency *across* queries, and can we name the pattern?** Group
   inputs by their observable class or bucket; for each group, record the
   distribution of path signatures. **Metric**: entropy of paths within a group
   (low = one dominant path per group = a nameable rule), plus the top-3 paths
   per group listed in plain English (e.g., "node R → 4A → 12 fires for 92% of
   'greeting' inputs"). This is the pattern-identification bar the KDN
   interpretability claim ultimately depends on.

**What has to change in the transformer to make this readable:**

- **Node naming.** Every KDN node gets a stable, human-readable id at
  construction (`block2_ffn_root`, `block2_ffn_R.left.left`, …), persisted in
  the version store. Reuse the identity-persistence invariant §5.1 item 1
  already specifies.
- **Trace hooks in the block.** The block's forward pass emits, alongside its
  output tensor, an ordered `PathTrace = list[(node_id, pointer_id, weight)]`.
  Off by default in production, on for the probe. This is the transformer-side
  change the user is asking for; without it, the FFN swap is invisible to
  measurement.
- **A `PathLog`** collecting `(input_hash, path_signature)` per forward pass,
  reused for Q2 and Q3. Deterministic given the same seed.
- **`inspect(node_id)` on the KDN block**: return the node's local tree, its
  incoming/outgoing pointers with weights and evidence counts. Reuses the
  existing `NetworkNode.inspect()` shape.

**Scale and cost.** 100 nodes total (across however many blocks are swapped),
2–5 M parameter transformer, character-level task, single laptop, one afternoon
per pass. If any of the three questions fails at this scale, no larger version
of the experiment can rescue it — which is exactly why v0 goes first.

**Failure semantics.** Q1 failure → the trace is not the model; go back to the
substrate. Q2 failure → the architecture has hidden state or unseeded RNGs;
find and fix them before Q3 is even worth asking. Q3 all-uniform (paths spread
across many signatures per group) → there are no readable patterns to name, and
the interpretability claim ends here.

### Sequencing

- **Prereq**: Workstream 1 correctness pass ships first. The KDN acceptance gate
  in DAPTIVE §9 needs to be re-specified from "graph CONNECTED" (already true
  post-2026-08-06) to "graph CONNECTED and beneficial" before any KDN v1 build
  is meaningful. That is a documentation edit already in Workstream 1.
- **Step 0 (v0 probe, this section above)**: 100-node KDN in a tiny transformer
  block, answer the three questions, decide whether to proceed.
- **Step 1**: build the tabular KDN v1 that DAPTIVE §9 specifies. Same code
  reused in Step 3 as the frozen replacement.
- **Step 2**: train the tiny transformer (standard FFN, matched to build B).
- **Step 3**: harvest block activations, distill one FFN block into a KDN,
  swap it in, measure the four claims further up. Repeat block-by-block only
  if Step 3 passes on the first block.
- **Step 4** (defer until Step 3 passes): try Option 1 (soft KDN, trained
  end-to-end) as the scientific stress test — this is the §5.2 experiment.
- **Step 5** (defer until Step 3 passes; independent of Step 4): attention
  transparency (T7) — see its own section below. Does not touch the FFN/KDN
  work; scores the still-vanilla self-attention mechanism against the same
  fidelity and edit-locality bars Step 3 applies to KDN.

Order rationale: Step 0 is the cheapest way to falsify the whole plan. If the
architecture cannot pass 100-node transparency on one laptop, there is no
reason to build the larger swap experiment. Steps 4 and 5 both depend only on
Step 3, not on each other, and can run in either order or in parallel once
Step 3 passes.

### Step 0-UI — real-time KDN inspector, iterable, GUI-driven

Ships **with** Step 0, not after. The three questions above become vague words
without an interface that lets a human watch the answer form. The UI is not
decoration — it is the instrument.

**Deliverable**: a browser-native inspector for a running KDN, served locally
from the existing `webui/backend` FastAPI process. Reuses the site's design
system (`webui/shared/tokens.css`), the WebSocket hub (`webui/backend/ws_hub.py`)
and the existing graph engine (`webui/desktop/js/graph.js`). No new UI stack —
the pattern the project already uses for the desktop app extends cleanly.

**Panels, minimum set:**

1. **Graph view.** All ~100 KDN nodes drawn as a force-directed / hierarchical
   layout. Each node shows its stable human-readable name (Step 0 requirement).
   Edges drawn with width = current pointer weight, colour = edge type
   (STRENGTHENS green / CONTRADICTS red / plain grey), matching the existing
   `NetworkEdge` semantics.
2. **Live activation overlay.** As a forward pass runs, nodes that fire flash
   in sequence, edges pulse along the traversed pointers, and inactive nodes
   dim. Frame-rate throttled to human-perceivable (~15–30 Hz) — the raw
   trace can arrive faster and is coalesced client-side. Off by default in
   production; a toggle switches "record + stream" on.
3. **Path timeline.** Below the graph, a horizontal timeline of the ordered
   `PathTrace` for the current query: `node_A → (w=0.72) → node_B → …` with
   every hop clickable to pin it in the graph. This is what makes the trace
   *readable*, not just recordable.
4. **Query bar.** Type or paste an input (token string, feature vector, or a
   picker for a corpus example). Submit re-runs the forward pass and animates
   the new path. This is the "iterable" part — a human iterates queries and
   watches paths change, which is how Q3 patterns become visible in practice.
5. **Retrace / step-through.** A scrubber that steps the animation forward
   and backward hop by hop. Reverse-play uses the same `PathTrace` — no
   re-running the model. This is what "retracing … done by the UI" means:
   the model runs once, the trace is a replayable artifact.
6. **Node inspector**, opened by clicking any node: local decision tree,
   incoming/outgoing pointers with weights and evidence counts, and the
   node's edit history from the VersionStore. Reuses the existing
   `NetworkNode.inspect()` shape.
7. **Consistency dashboard**, answering Q2 and Q3 visually: "same input, N
   runs" showing path-identity fraction; "grouped inputs" showing the
   entropy-of-paths metric with the top-3 paths per group listed in prose.
   These are the same numbers the CI measurement produces; the UI just shows
   them live so a human can see the pattern (or its absence) as it forms.

**Data path.** The `PathLog` from Step 0 emits WebSocket events; the frontend
subscribes and replays them on the graph. Same pattern as the existing
`webui/frontend/js/decision-trace.js`. The trace itself is source-of-truth —
the UI never adds or interprets anything the trace does not already carry.

**Design-system discipline (per the user's brief on latest design and scientific
design systems):**

- **Reuse `webui/shared/tokens.css`** for spacing, type scale, colour tokens
  (dark/light aware). No inline colours.
- **Accessibility floor**: WCAG 2.1 AA — contrast, keyboard-navigable graph,
  motion-reduced mode that replaces flashes with stepped state changes so the
  live-activation panel is safe.
- **Motion**: use the existing motion primitives (`webui/shared/js/`) — subtle
  easing, no bounce, no attention-hijacking. This is a scientific instrument;
  a chart, not a demo reel.
- **Typography**: monospace only for node ids and trace tokens (they are
  identifiers); default sans for everything else. Matches the site's existing
  console-style panels.
- **Layout**: graph occupies the primary area; timeline pinned below; node
  inspector as a right-hand drawer; consistency dashboard as a switchable tab
  in the drawer. Two-pane layout collapses to tabs below 900 px.

**Non-goals for v0 UI.**
- **No editing from the UI in v0.** Inspection only. Edit-through-UI is Step 3
  scope (once the swap experiment can measure the effect of an edit) and
  belongs with the HITL law-edit machinery, not the probe.
- **No LLM-scale visualization.** 100 nodes is the target; graph performance
  above ~2 000 nodes is out of scope.
- **No remote hosting.** Runs on `localhost`, keeps traces in the existing
  `results/` layout, does not phone home. This matches the offline-first
  discipline the project already keeps.

**Falsifier for the UI itself.** If a human sitting in front of the UI cannot,
after five iterated queries, describe *in one sentence per input-group* the
rule the KDN appears to be following, then Q3 has already failed — and no
prettier version of the UI rescues it. The UI is the instrument for the
falsification, not a substitute for it.

### What could kill this cleanly, and how to know

- **Fidelity gap too large** (perplexity delta > ε on first block): frozen-KDN
  cannot represent even one FFN's function. Recorded finding; do not proceed.
- **Attribution gap under soft routing** (Step 4 metric 1 drops sharply vs
  Step 3): the KC/KDN thesis has been shown to be scoped by hard routing, not
  delivered by the substrate. Recorded finding; retire the transformer plan or
  redesign the routing.
- **Edit locality no better than ROME/MEMIT**: KDN buys interpretability but
  not edit-locality; keep it as an inspection substrate, drop the "better than
  ROME" framing.

Any of these is a **useful negative result**, and belongs in the same
NeurIPS/ICML Datasets & Benchmarks framing that Volume V §4 already targets for
Volume III.

### Connection to the three master-plan docs

The UIL project has **three** documents that together constitute its master
plan, each covering a different scope, each with its own future experiments.
Workstream 2 has to be threaded through all three so it reads as the concrete
next step for each of their theses, not as an unrelated fourth workstream.

Weighted by recency (most recent is most authoritative):

1. **`LITSURVEY_CORRECTIONS.md`** (2026-08-07 00:11) — targeted corrections to
   the litsurvey and one **proposed new §5 subsection** requiring causal
   attribution before any localized edit, plus a **proposed new §6** codifying
   the five acceptance metrics for any KC implementation.
2. **`academic_v2/06_...LITSURVEY.md`** (2026-08-02) — §6 T1–T8 transformer
   plan, G1–G6 GenAI plan, R1–R5 RL plan; §8 S0–S5 roadmap. The doc that
   literally maps future experiments for the KC / KDN track.
3. **`DAPTIVE_KNOWLEDGE_NETWORK_DESIGN.md`** (2026-07-24) — §10 Phase 0/1/2/3
   sequencing and the parallel KDN track; §9 v1 acceptance gates.

Each master doc contributes one non-negotiable requirement to Workstream 2.

#### From `LITSURVEY_CORRECTIONS.md` — the two acceptance tests and the causal-attribution requirement

The corrections doc's proposed **§5.4** ("Two acceptance tests any Knowledge
Cell implementation must pass") says: **fidelity** (`evaluate(render(cell)) ==
forward(cell)`, exactly), and **edit exactness** (a patch moves the output by
the amount the interface said it would). These are the same two tests the
fitted-law substrate already passes (`tests/test_law_fidelity.py` F7, max
error 7e-14) — and they *must* be passed by the KDN block before Step 3 is
allowed to measure anything. They become Workstream 2's Step 0 Q1 metric,
verbatim.

The proposed **§5.5** (causal-attribution requirement) is the harder demand:
*"a trace alone is not sufficient evidence — the trace-named cell has to
survive a control/treatment intervention before the `Apply patch` step is
permitted."* This is why the rescue doc's promoted **causal engine** becomes
the referee for Step 3 metric 2 (edit locality), and why the KDN inspector
UI's per-node "Apply Edit" flow (deferred to Step 3) must be gated by a
control/treatment probe, not by trace ranking alone.

The proposed **§5.6** (five metrics) supplies Workstream 2's *whole scoring
sheet* for the small-transformer swap: localization accuracy, edit locality,
edit persistence, rollback correctness, and catastrophic interference under
repeated editing. Adopt these five as the Step 3 report card — do not invent
a new metric suite when this one exists and is already argued for from the
literature.

#### From `academic_v2/06_..._LITSURVEY.md` — the small-transformer protocol, imported unchanged

**§6.6** ("The S4 protocol: small transformer first, LLM never before the
ablation") is *already* the protocol Workstream 2 needs. Copy it: two matched
transformers on identical data, ~millions of parameters, score on all five
§5.6 metrics, accuracy reported only as a secondary metric, no LLM jump until
the small-transformer ablation passes. The v0 100-node probe (Step 0 above)
plus the Step 3 frozen-swap experiment are precisely §6.6 scoped down to what
one laptop can measure.

**T1–T8** already name the transformer-side moves Workstream 2 needs, so it
does not need to reinvent them: T1 (name FFN cells), T3 (rank-one patch), T5
(per-cell versioning — the VersionStore already provides this), T7 (cell-level
attribution, TCAV-style — supplants raw attention weights per Jain & Wallace
P18). The two items to explicitly *defer past v0* are T4 (mass edits at scale)
and T8 (SAE decomposition) — both are for later transformer sizes; v0 does
not need them.

**S0–S5 mapping**. Workstream 2 Step 0 = S0's "cell substrate for a
transformer block"; Step 0-UI = the S2 inspector, brought forward because the
three v0 questions cannot be answered without it; Step 3 = S3 hard-routing
KDN-as-FFN inside a small transformer; Step 4 = S3's soft-routing scientific
stress test. **S4 (small-transformer KC-FFN ablation) is Workstream 2's Step 3
by another name — the two roadmaps are the same roadmap.** Do not run two.

#### From `DAPTIVE_KNOWLEDGE_NETWORK_DESIGN.md` — reuse, don't rebuild

The DAPTIVE doc's honest §4 reuse table says nine of twelve KDN parts already
exist in `editable_network.py`. Workstream 2 inherits that: **the KDN inside
the transformer FFN slot reuses `NetworkNode`, `NetworkEdge`,
`discover_relationships`, `record_traversal_outcome`, `apply_edge_decay`,
`traverse_with_attention`, `KnowledgeArchive`, and the working-graph caps.**
The three genuinely new pieces DAPTIVE names (DT-as-payload, DT-as-router,
prediction-assembly) are exactly the three pieces Step 0 has to build, and
nothing more.

DAPTIVE §5 gives three invariants for the tabular KDN (transparency,
editability, reproducibility). Workstream 2 preserves all three — reproducibility
in particular is the whole point of the v0 Q2 test, and the deterministic
seeding discipline DAPTIVE §5 calls out (bug #31's lesson) is copied over
verbatim to every tree the KDN grows inside a transformer block.

DAPTIVE §9's v1 acceptance gate ("does removing the graph change the output?")
is upgraded — as recorded in Workstream 1 — to "connected and beneficial",
and, further, per the rescue doc, to **"connected and beneficial for
edit-locality, beating the plain distilled-MLP control on the same paired
inputs."** This is the single number that decides all three master docs'
theses at once.

#### The unified table Workstream 2 answers to

| Master doc | Its ask | Workstream 2's answer |
|---|---|---|
| Rescue (08-06 — via `COGNITIVE_ENGINE_RESCUE.md`) | Institutional memory for edits; A/B/C with lookup-table falsifier; promote causal | Step 3 adds a plain-distilled-MLP control arm; causal engine is the edit-locality referee |
| Corrections (08-07) | Fidelity + edit-exactness tests; causal attribution gate; five §5.6 metrics | Step 0 Q1 = fidelity; Step 3 metrics = the five, verbatim; UI's Apply-Edit flow gated by control/treatment |
| Litsurvey (08-02) | S0–S5 roadmap; S4 small-transformer KC-FFN ablation; T1–T8 transformer moves | Step 0 = S0; Step 0-UI = S2 brought forward; Step 3 = S4; T1/T3/T5/T7 imported; T4/T8 deferred |
| DAPTIVE (07-24) | Reuse nine parts, build three; three invariants; v1 acceptance gate | KDN inside FFN reuses the nine parts; the three new pieces are Step 0's build; gate upgraded to "connected + beneficial + beats MLP-control" |

**Doc-level wiring when Workstream 1 ships.** Insert one paragraph in each
master doc pointing at this plan file's Workstream 2 as the concrete
embodiment:
- `COGNITIVE_ENGINE_RESCUE.md`'s closing "Why this matters beyond UIL"
  paragraph — extend it with the mapping table above.
- `LITSURVEY_CORRECTIONS.md`'s C6 (proposed §6 metric suite) — add a note
  that Workstream 2 Step 3 is the first planned evaluation harness for those
  metrics on a non-tabular substrate.
- `academic_v2/06_..._LITSURVEY.md`'s §8 roadmap — annotate S4 with a
  cross-reference so the two roadmaps stay one.
- `DAPTIVE_KNOWLEDGE_NETWORK_DESIGN.md` §10's "Parallel track — the KDN"
  paragraph — extend to say the transformer FFN slot is the second target
  substrate, sharing every DAPTIVE invariant.
- `CLAUDE.md`'s Pending-work entry for the rescue doc — add the four-row
  table so all threads read as one.

### Step 5 — attention transparency (T7), only after Step 3 passes

Steps 0–4 deliberately leave self-attention untouched. That is the correct
isolation for measuring whether KDN can replace an FFN sublayer at all — but
it means the resulting model is **not** a transparent LLM, only a model with
one transparent sublayer inside an otherwise-standard block. Self-attention is
mechanistically required for any sequence model (an FFN, KDN or otherwise, is
a per-token function and cannot see other positions — there is no version of
Workstream 2 that removes attention and still has a language model), so Step 5
is not "replace attention with KDN." It is "make attention inspectable using
the same rigor Steps 0–3 already apply to the FFN."

**Why this is a different, harder problem, not a bigger version of Step 3.**
KDN routes a fixed input through a fixed graph of nodes; the same node
identities exist call to call. Attention computes a **pairwise** score between
every query token and every key token in the *current* sequence — the
"pointer targets" are other token positions, which change every forward pass
depending on what the sequence contains. A per-token routing graph does not
have this shape. Whatever Step 5 builds has to route between positions, not
between fixed nodes, and no document in this project designs that today.

**Why raw attention weights are not the answer, even though they look free.**
Jain & Wallace (2019, P18 in the litsurvey) found that attention weights in
several NLP architectures correlate poorly with other importance measures and
can be adversarially perturbed while leaving the output nearly unchanged. So
exposing attention weights in a UI panel would not satisfy the same fidelity
bar Step 0 Q1 already enforces for KDN (`evaluate(render(cell)) ==
forward(cell)`, exact) — a rendered attention weight is not provably the
reason for the output the way a KDN path trace is. This is exactly what T7 in
`academic_v2/06_..._LITSURVEY.md` §6.1 already names as the fix: replace raw
attention-weight explanations with **cell-level attribution and concept-level
probes (TCAV, P19)**, not with the weights themselves.

**Step 5 scope, gated behind Step 3 passing:**

1. Build a concept/attribution probe over each attention head — TCAV-style
   (P19): does a head's output move in the direction of a human-labelled
   concept, measured by directional derivative, not by reading its weights.
   This reuses the existing `results/` -style A/B measurement discipline
   rather than trusting the weight matrix.
2. Extend the Step 0-UI's node inspector with a per-head attribution panel,
   fed by (1), following the same "trace is source of truth" rule the FFN
   panels already use — the UI must never display a number it cannot also
   prove drove the output.
3. Score against the same acceptance bar as the FFN swap: does head-level
   attribution survive the same localization-accuracy metric (litsurvey §5.6
   metric 1) that Step 3 already applies to KDN cells. If attention
   attribution scores no better than existing post-hoc methods (TCAV,
   integrated gradients) on a standard dense transformer, Step 5 has not
   bought anything beyond what already exists in the interpretability
   literature — record that as the outcome rather than as a failure to
   iterate past.

**Falsifier.** If, after (1)–(3), the per-head attribution panel cannot
predict which unrelated inputs a targeted change to one head will affect —
the same edit-locality bar Step 3 applies to KDN cells — then attention
remains the honest black-box half of the model, and the project's transparent-
LLM claim must be scoped to "the FFN sublayer only," not the whole block. That
scoping is a legitimate, publishable outcome on its own; it is not a reason to
avoid running Step 5, only a reason not to promise it will succeed in advance.

**Sequencing note.** Step 5 does not block Steps 0–4 and is not blocked by
Workstream 1. It is blocked only by Step 3 passing — there is no reason to
design attention transparency before the FFN-transparency result it is
modeled on has been measured to work.

### What is out of scope for this workstream

- LLM-scale training (S5 in the litsurvey). Do not touch until Step 3 and
  Step 4 both pass.
- Multi-modal or vision transformers. Text-only PoC.
- Novel attention *architectures* (replacing self-attention's mechanism).
  Attention's computation stays standard throughout Workstream 2 — Step 5 adds
  an inspection layer on top of it, it does not change how attention computes
  its output.
- Any change to KDN's tabular v1 gates. Those live in DAPTIVE §9 and are
  Workstream-1 work.

---

## Deliberately not doing

- Not touching framing, theses, or roadmaps — correctness-only per your choice.
- Not editing the Desktop `06_..._LITSURVEY.NEW.md` (audited last turn, outside
  both sets).
- Not renaming either KDN.
- Not re-running any experiment — every replacement number already exists in
  `results/`.

---

## Verification

1. **Retired-phrase sweep.** After editing, hits should appear only inside
   explicitly-marked "superseded/corrected" notes:

```bash
grep -rn "10–100×\|10-100x\|10\\\\times\|94\.9%\|0\.992\|-5\.52\|+0.004\|0\.0132\|only subsystem ever measured CONNECTED\|unregistered" CLAUDE.md LITERATURE_SURVEY.md RESEARCH_FAILURE_ANALYSIS.md DAPTIVE_KNOWLEDGE_NETWORK_DESIGN.md HARDCODED_VALUE_AUDIT_QA.md academic_v2/0*.md
```

2. **Every new number traces to a file** — each replacement cites one of
   `results/compute_cost/{SUMMARY,HEAD_TO_HEAD}.md`,
   `results/extrapolation/SUMMARY.md`, `results/existence_{sweep,mined}/FINDINGS.md`,
   `results/hitl_law_edit/SUMMARY.md`. Spot-check three at random against source.

3. **Internal consistency** — Vol III §3.2's module count matches its table; Vol II
   line 42 attributes 94.4% to XGBoost; Vol V Cluster 2 no longer claims what its
   own roadmap Stage 1 defers; `CLAUDE.md` has exactly one copy of each duplicated
   block.

4. **Code-backed claims re-checked against code, not memory:**

```bash
grep -n "TabPFN" core/classifier_registry.py
```

5. **No behaviour risk** — documentation edits only. Optional sanity run:

```bash
python figpso_uil/tests/test_law_fidelity.py
```
