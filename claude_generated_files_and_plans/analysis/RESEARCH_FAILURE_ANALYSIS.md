# Why the cognitive layer fails — mapped to published research (2026-07-23)

Every failure UIL has measured this month is a **known, named phenomenon in the
literature**, most with a published remedy. This document maps each measured
failure to that research and states the concrete UIL change it implies, under
the constraint that transparency, editability, and reproducibility are
non-negotiable. Nothing here is speculation about UIL's behavior — the UIL side
of every row is a measurement already recorded in CLAUDE.md.

---

## F1. "The cognitive layer ties plain PSO on accuracy" — the ceiling was never winnable

**UIL measurement:** FULL cognition vs ALL-ABLATED = +0.0000 (bug #25);
mean Δ ≈ +0.004 across benchmarks at 10–100× compute.

**Research:** Grinsztajn, Oyallon & Varoquaux (NeurIPS 2022), *Why do tree-based
models still outperform deep learning on tabular data?* — across 45 datasets,
tuned tree ensembles remain state of the art on medium tabular data, and the
residual headroom above a tuned GBDT is tiny. An entire decade of tabular deep
learning — thousands of papers of added machinery — also measured ≈ +0.000
against that baseline. Sutton's *Bitter Lesson* (2019) is the general form:
added human-designed reasoning machinery loses to plain search + compute
wherever compute is the binding constraint.

**Implication for UIL:** the accuracy-on-saturated-tabular claim was
structurally unwinnable, independent of any wiring bug. The field's answer is
not "reason harder," it is **change the axis of competition**:
- *interpretability at accuracy parity* (F8 — EBM),
- *small-sample priors* (F9 — TabPFN v2),
- *out-of-distribution extrapolation* (already UIL's proven win: symbolic
  R² 0.99 vs XGBoost −5.5).

UIL's own results are exactly what this literature predicts, including the
wins. Stop treating the tie as a failure to be fixed; treat it as the
literature-consistent boundary of the domain.

---

## F2. Selection overfitting killed the Architecture Race — the canonical model-selection failure

**UIL measurement:** race Gate 2 −0.0117 with corr(Δaccuracy, val-gap swing)
= **−0.977**; loses on small n, wins on the two largest datasets (E7).

**Research:** this is the textbook phenomenon of **Cawley & Talbot (JMLR
2010)**, *On Over-fitting in Model Selection and Subsequent Selection Bias in
Performance Evaluation*: the variance of the selection criterion matters as
much as its bias, and picking best-of-N on one small validation split
overfits the split — with degradation "comparable in magnitude to performance
differences between learning algorithms" (i.e., exactly big enough to erase a
real improvement, which is what Gate 2 saw). Their remedy is **nested
(double) cross-validation**: an inner loop for selection, an outer loop for
evaluation, never the same data for both.

**Dwork et al. (Science 2015)**, *The reusable holdout*, goes further: every
adaptive decision made against the same holdout leaks information into the
model regardless of sample size. Their **Thresholdout** mechanism answers
holdout queries through a noise-adding gate (differential-privacy mechanics),
which provably bounds the leakage while allowing the holdout to be reused.

**Implication for UIL (directly implementable, reproducibility-safe):**
1. Final-rung race selection on `self._extra_val_splits` (built, populated,
   never used for selection) = the inner/outer split Cawley & Talbot
   prescribe. This was already the planned next step; the research confirms
   it is *the* remedy, not one option among several.
2. For long-lived reuse — the champion bar, the standing Gate 2, HITL A/Bs —
   implement Thresholdout on the held-out slice: answer "did arm A beat arm
   B?" only through a seeded-noise threshold gate. Seeded noise keeps it
   byte-reproducible. This makes the *instrument* durable against the very
   pattern (repeated adaptive measurement) that this project's history is
   made of.
3. Cawley & Talbot's variance point also explains why sonar's +0.032
   "cognition win" evaporated at cv_folds=2: it was selection variance, not
   signal — the paper predicts precisely this class of mirage.

---

## F3. Principles need ~200 stages to exist — meta-learning sample complexity

**UIL measurement:** `evidence_count` is a tree leaf size; `MIN_EVIDENCE=10`
is first reachable ≈ 200 same-dataset stages; every isolated-`state_dir` run
mines **zero** principles.

**Research:** the field already priced this. **auto-sklearn** (Feurer et al.,
NeurIPS 2015) makes meta-learning work by training offline on **140 OpenML
datasets** and warm-starting new runs from the accumulated meta-knowledge;
auto-sklearn 2.0 (2020) doubles down with meta-learned policy selection.
Nobody in the AutoML literature attempts to induce transferable rules from
~16 observations of a single dataset — that is one to two orders of magnitude
below the sample sizes at which published meta-learning first works.

The statistics literature has the remedy for exactly the small-evidence
regime: **hierarchical Bayesian partial pooling** — estimate each rule's
effect with shrinkage toward a population prior learned across all rules /
datasets, so weak evidence yields a weak-but-nonzero principle instead of no
principle. Extreme estimates from tiny samples get pulled toward the
population mean; strong evidence escapes the pull. This replaces the *hard
threshold* (evidence ≥ 10: all-or-nothing) with a *soft weight* — which is
also more honest epistemology.

**Implication for UIL:**
1. UIL's default shared-state design **is** its OpenML: journals accumulate
   across datasets and sessions. The cross-dataset miner already exists (Part
   5). The research says lean INTO that — the per-dataset isolated-arm
   setting is the one configuration in which principle mining cannot work,
   for reasons of sample complexity, not wiring.
2. Replace `MIN_EVIDENCE` as a cliff with partial pooling: principle
   confidence = Wilson/Beta posterior shrunk toward the library-wide prior,
   influence proportional to posterior confidence. UIL already half-does this
   (M0.2 tiny-mining shrinks confidence toward 0.5) — the research-grade
   version shrinks toward a *learned population prior*, not a constant, and
   drops the threshold entirely. Fully transparent (a principle carries its
   posterior and its prior — both inspectable and editable) and reproducible
   (closed-form Beta updates, no sampling needed).

---

## F4. Nothing varies, so nothing can be learned — the exploration requirement

**UIL measurement:** in `classic` mode the journal contains 16 distinct
`penalty` values and **1** distinct value of every architecture knob. A
decision tree cannot split on a constant.

**Research:** **Population Based Training** (Jaderberg et al., 2017) is the
canonical demonstration that the fix is *online explore/exploit perturbation*:
maintain a small population, periodically copy the winner's weights
(exploit), perturb its hyperparameters (explore). The perturbation is not
overhead — it is what *generates the experience* from which schedules can be
learned at all, at roughly the cost of training the population once. fANOVA
(Hutter, Hoos & Leyton-Brown, ICML 2014) supplies the companion tool: fit a
random-forest surrogate over (config → performance) and decompose variance to
learn **which knobs matter and which interact** — from exactly the kind of
logged data the journal holds, once the knobs vary.

**Implication for UIL:**
1. UIL's stage loop is already a population (4–6 candidates/stage) and
   `ArchitectureRace.propose()` already perturbs one axis at a time. The
   PBT lesson: this variation must run in the DEFAULT path, not only in the
   opt-in race — a low-rate exploratory perturbation (one candidate per stage
   varies one architecture axis) is what feeds the journal. It composes with
   F2's fix: variation without honest selection re-creates the winner's
   curse, so ship them together.
2. Add fANOVA over the journal (a ~100-line dependency-free variant is
   feasible over the existing DT/RF machinery) as the *importance* signal:
   let it rank which axes deserve candidates, and let mined rules cite the
   fANOVA importance as provenance. It is itself a transparent artifact — a
   variance decomposition table is readable and auditable.

---

## F5. Evolution runs and changes nothing — library growth needs a utility gate

**UIL measurement:** with the reachable stress gate, the formula library grows
32 → 34 every run and accuracy is byte-identical 9/9. Newborn formulas are
never selected into a `sparse_linear` fit.

**Research:** **DreamCoder** (Ellis et al., PLDI 2021 / Phil. Trans. R. Soc.
2023) is the reference design for growing a symbolic library that stays
useful: in its sleep phase, a candidate abstraction enters the library **only
if it improves a compression objective over the existing corpus of
solutions** (MDL: the abstraction must pay for itself). Birth is
utility-gated, not event-gated. UIL births formulas on stress/stagnation
events with no requirement that the newborn ever help — the audit shows the
predictable result. The E5 finding (a correlated group under one L1 budget
gets starved) has the same shape as DreamCoder's need to protect new
abstractions: a newborn term competing against 200 entrenched terms under one
global penalty never gets picked, exactly as E5 measured for rules
(sonar 0.748→0.748 until the budget changed).

**Implication for UIL:**
1. Gate admission, not just birth: a mutant enters `active` status only if
   adding it to the basis improves a held-out fit (one cheap incremental
   L1 refit with the candidate term included — measurable, seeded,
   reproducible). This is DreamCoder's compression test translated into
   UIL's vocabulary, and it turns "the library grew" into "the library grew
   because it helped," which is also a *better provenance story* for HITL.
2. Give newborn symbolic terms their own L1 sub-budget for their first k
   fits (the E5 remedy, applied to formulas), or they can never displace
   incumbents regardless of quality.

---

## F6. Mining the journal is confounded — logged data is off-policy

**UIL measurement (older, still standing):** hypotheses born from mined rules
died untestable for the system's entire history until bug #14; mined
"penalty→regret" rules encode what the *search chose to try*, not the causal
effect of penalty.

**Research:** **Swaminathan & Joachims (ICML/JMLR 2015)**, *Counterfactual
Risk Minimization*: learning from logged interaction data requires correcting
for the logging policy (propensity weighting), or the learned rules inherit
its biases; naive empirical risk over logged data is inconsistent. The causal
inference consensus matches UIL's own architecture: **interventions beat
passive mining** — which is what the causal engine's control/treatment trials
do, and the existence audit agrees (causal was the only subsystem ever
measured CONNECTED).

**Implication for UIL:** prioritize the intervention path over the passive
miner as the source of *causal* claims (statuses stay as they are:
correlational from mining, causal only from trials). Where the miner's rules
are used anyway, record alongside each rule the action distribution it was
mined under (a propensity footnote — one dict per rule) so a human reviewing
a principle can see the confound. Cheap, transparent, and it makes the HITL
review meaningfully better.

**DONE (2026-07-24):** `UIL_Miner._logging_policy_footnote` attaches
`logging_policy` to every mined rule (penalty mean/std/n_distinct, config.*
summaries, classifiers, search_modes); `UIL_Library._create_principle`
persists it on the principle. Does **not** reweight / correct promotion —
HITL visibility only. Test: `test_f6_logging_policy_footnote` in
`tests/test_research_steps_6_8.py`.

---

## F7. Nondeterminism was an unseeded solver — reproducibility is a supply chain

**UIL measurement:** bug #31 — one `LogisticRegression(liblinear)` without
`random_state` explained the entire multiclass/hybrid nondeterminism; 8
identical fits under 8 global seeds → 8 different coefficient vectors.

**Research context:** this is the reproducibility literature's core finding in
miniature (Pineau et al., *Improving Reproducibility in ML Research*, JMLR
2021): nondeterminism is almost never "the algorithm," it is unpinned
randomness in the dependency chain. The standing invariant (every estimator
gets an explicit seed; identity tests as the control) is exactly the
recommended practice. Nothing further to adopt — UIL is now ahead of most
published code on this axis; the existence audit's two-identical-runs control
is precisely the "sanity replication" the reproducibility checklists ask for.

---

## F8. The strongest research validation of UIL's actual goal: EBM

**Research:** the **Explainable Boosting Machine** (Lou, Caruana et al.;
InterpretML) is a cyclic gradient-boosted **generalized additive model** with
automatic pairwise-interaction detection — a fully glass-box model with
accuracy *repeatedly measured at parity with XGBoost/Random Forest* on
tabular benchmarks. It is the existence proof that "readable at parity" — the
exact claim UIL's sparse_linear mode makes — is achievable and publishable.

**Structural comparison (UIL is close):**

| EBM | UIL sparse_linear |
|---|---|
| shape function per feature (boosted shallow trees) | raw + stump terms |
| pairwise interaction terms (GA2M), auto-detected | rule basis (E5), router-gated |
| cyclic boosting: one feature at a time, tiny LR, many rounds | one joint L1 fit |
| additive → per-feature plots, editable | additive → readable terms, editable cells |

The one mechanism UIL lacks is the **cyclic per-feature boosting fit**: EBM's
round-robin over features with a small learning rate is what lets thousands
of micro-steps build feature shapes that a single L1 pass cannot, while
staying strictly additive/readable. That is a *fit algorithm* upgrade, not an
architecture change — every UIL property (named terms, provenance, JSON
persistence, HITL edits) survives it.

**Implication for UIL:** implement an `ebm`-style fit path for the composed
basis (cyclic boosting over basis families instead of one L1 solve), as a
raced alternative to the L1 fit. It attacks exactly the gap UIL measured
(sonar: additive-under-L1 0.75 vs nonlinear 0.88) with the strongest
literature evidence available, and stays inside the transparency contract.

---

## F9. Free wins the research already handed over

1. **TabPFN v2** (Hollmann et al., **Nature, Jan 2025**): a tabular foundation
   model that "outperforms all previous methods on datasets up to 10,000
   samples," no tuning. UIL **already has a TabPFN adapter** (unregistered).
   It is not interpretable — but as (a) the router's honest *ceiling probe*
   (replacing/da complementing the RF probe in the fingerprint) and (b) a
   router strategy for small-n datasets where the user asks for accuracy over
   readability, it is the single cheapest accuracy improvement available to
   this codebase. The E8 table's lesson (the MLP ceiling is sometimes BELOW
   the symbolic model) makes an honest ceiling probe genuinely informative.
2. **Kolmogorov–Arnold Networks** (Liu et al. 2024; active 2024–25
   literature): learnable spline edge-functions, L1-sparsified, explicitly
   designed for **symbolic extraction** — each learned univariate function can
   be snapped to a closed-form operator. This is the research-grade version
   of E8's neural basis: instead of opaque `relu(u15<-x7,x26)` units, KAN-style
   univariate spline terms are *plottable* (EBM-like), *prunable*, and
   *convertible to symbolic form* — i.e., they end as formula cells. A
   `kan_unit` basis family would raise interaction capacity while *increasing*
   readability relative to the current MLP units.
3. **Nested CV / reusable holdout** (F2) and **partial pooling** (F3) are
   pure-math changes with no new dependencies — both are implementable
   deterministically and hence keep byte-replay.

---

## What this means as a program (ordered, each gated)

| # | Change | Research basis | Gate |
|---|---|---|---|
| 1 | Race final-rung selection on `_extra_val_splits` (nested CV) | Cawley & Talbot 2010 | re-run E7 Gate 2 |
| 2 | Default-path low-rate architecture exploration (1 candidate/stage varies 1 axis) | PBT, Jaderberg 2017 | existence audit: journal variation > 1 distinct value; then Gate 2 |
| 3 | Partial-pooling principle confidence (drop the MIN_EVIDENCE cliff) | hierarchical Bayes / shrinkage | audit: principles minted in isolated runs; then Gate 2 |
| 4 | Utility-gated formula admission + newborn L1 sub-budget | DreamCoder MDL gate; E5 finding | **IMPLEMENTED; Gate 2 FAILED (2026-07-24).** Existence audit INERT; accuracy mean Δ **−0.0013** (5 ties / sonar −0.0079). Stay default OFF. See `results/gate2_utility/SUMMARY.md`. |
| 5 | EBM-style cyclic boosting fit as raced alternative to L1 | InterpretML/EBM parity results | **IMPLEMENTED; Gate 2 DONE (2026-07-24).** Mean Δ **+0.0014** (sonar SL +0.008). Default stays OFF — race axis only. See `results/gate2_ebm/SUMMARY.md`. |
| 6 | Thresholdout for champion bar + standing gates | Dwork et al. 2015 | **IMPLEMENTED (2026-07-24).** `cognition/thresholdout.py`; champion bar uses seeded Laplace gate (`cfg.use_thresholdout`, default ON). Instrument tests T1–T3: deterministic; rejects null A/B; accepts clear wins. |
| 7 | TabPFN v2 as router ceiling probe / small-n strategy | Hollmann et al., Nature 2025 | **IMPLEMENTED (2026-07-24).** Fingerprint `tabpfn_probe_acc` / `tabpfn_gap` (soft-fail if unset); domain `D3_tiny_tabpfn` → strategy `tabpfn`; optional CLASSIFIERS entry. Never default when webui disables routing. |
| 8 | KAN-style spline basis family replacing/joining MLP units | KAN literature 2024–25 | **IMPLEMENTED opt-in (2026-07-24).** `use_kan_basis` — univariate truncated-power terms `(x_j-t)_+^d`, readable, raced. Default OFF. Unit K1–K2 + XOR exploratory. **Gate 2 FAILED for default ON (2026-07-24):** mean Δ **−0.0022** (sonar **−0.0158**; digits/spambase soft +0.001). Keep race axis only. Logs: `results/gate2_kan/`. |

Ordering rationale: 1 fixes the judge, 2 creates the evidence, 3 lets the
evidence become knowledge, 4 lets invented structure compete, 5–8 raise the
learner's ceiling. Running 3 before 2 is pointless (nothing to pool); running
2 before 1 re-creates the winner's curse the race already documented.

**Every row preserves the three constraints.** Nothing above introduces an
opaque component into the model itself (TabPFN is a probe/strategy, never the
default learner); every mechanism is deterministic given the seed; every new
artifact (posterior, variance table, admission test result, spline term) is a
readable, editable cell with provenance — several of them make the HITL story
stronger, not weaker.

---

## Sources

- Cawley & Talbot 2010, *On Over-fitting in Model Selection…* — https://www.jmlr.org/papers/v11/cawley10a.html
- Dwork et al. 2015, *The reusable holdout* (Science) — https://www.science.org/doi/10.1126/science.aaa9375
- Grinsztajn et al. 2022, *Why do tree-based models still outperform deep learning on tabular data?* — https://arxiv.org/abs/2207.08815
- Feurer et al. 2015 / auto-sklearn meta-learning — https://arxiv.org/pdf/2202.10979 (LDE overview); auto-sklearn docs
- Jaderberg et al. 2017, *Population Based Training* — https://arxiv.org/pdf/1711.09846
- Hutter, Hoos & Leyton-Brown 2014, *fANOVA* — http://proceedings.mlr.press/v32/hutter14.pdf
- Ellis et al., *DreamCoder* — https://arxiv.org/abs/2006.08381
- Swaminathan & Joachims 2015, *Counterfactual Risk Minimization* — https://proceedings.mlr.press/v37/swaminathan15.html
- InterpretML / EBM — https://interpret.ml/docs/ebm.html ; https://github.com/interpretml/interpret
- Hollmann et al., *TabPFN v2* (Nature 2025) — https://arxiv.org/pdf/2502.17361 (analysis); https://huggingface.co/Prior-Labs/TabPFN-v2-clf
- KAN survey 2024 — https://arxiv.org/pdf/2411.06078
- Hierarchical partial pooling notes — https://jrnold.github.io/bayesian_notes/shrinkage-and-hierarchical-models.html

---

## Addendum (added 2026-08-12) — corrections and the transformer/KDN workstream

Addition only; F1–F9 and the program table above are unchanged.

### Stale figures in F1, F6, and F9, corrected here

- **F1** above cites "10–100× compute" and "+0.004" as UIL's measured
  standing. Both are superseded: the cognitive layer itself measured **0.95×**
  (effectively free), while the **search loop** costs **~3900×** one ordinary
  fit of the same learner, and paired on identical splits the accuracy delta
  is −0.0046 (CI [−0.0289, +0.0197], 79 cells needed to resolve ±0.01, 9 run —
  underpowered, not zero). Sources: `results/compute_cost/SUMMARY.md`,
  `results/compute_cost/HEAD_TO_HEAD.md`.
- **F1**'s closing bullet cites "already UIL's proven win: symbolic R² 0.99
  vs XGBoost −5.5" as a settled positive result. Beating XGBoost out of range
  is a **theorem** (gradient-boosted trees are piecewise-constant and
  saturate outside the training hull by construction), not a finding. Against
  baselines that can actually extrapolate, UIL wins 0 of 21 cells; against
  `gplearn` specifically it loses 4–17 with 19 blow-ups vs 0.  `gplearn`
  itself beats every non-symbolic baseline, so F1's underlying point — that
  symbolic methods are the right tool for out-of-range recovery — still
  stands; UIL's own regressor is just measured to be a weak instance of it,
  not the exemplar. Source: `results/extrapolation/SUMMARY.md`.
- **F6** states "causal was the only subsystem ever measured CONNECTED." A
  later, stronger measurement (four cells across two budgets, then a
  mining-enabled follow-up) found causal **INERT** in every cell measured,
  and found **principles** and **formulas** connected instead (network
  connected in one of two cells). "Connected" in the later measurement means
  0–2% of predictions move with a net-negative accuracy effect — reaching the
  model, not helping it. Sources: `results/existence_sweep/FINDINGS.md`,
  `results/existence_mined/FINDINGS.md`.
- **F9.1** states the TabPFN adapter is "currently registered nowhere
  (`classifier_registry.py` doesn't include it)." This is stale: research
  step 7 (recorded in the program table above, row 7) registered it behind an
  optional-dependency guard, matching the `GPLearn` pattern. Verified at
  `core/classifier_registry.py:71-77`.

### Program table row 1, outcome recorded

Row 1 ("race final-rung selection on `_extra_val_splits`") is the only row in
the table above with no recorded outcome. It was run: the judge fix measured
**+0.0080** against classic search (4 win / 1 tie / 1 lose across 6 datasets;
ionosphere −0.047 remains an open regression). A follow-on
incumbent-margin gate at 0.01 subsequently **failed** (mean +0.0010,
3W/1T/2L; ionosphere untouched, breast_cancer's prior win reversed to a
small loss). `search_mode="race"` stays opt-in with margin 0.

### The transformer/KDN workstream this document's program did not anticipate

The 8-row program above (and its "ordering rationale") targets the tabular
`sparse_linear` learner. A separate track — embedding KDN inside a small
transformer's FFN sublayer — was designed 2026-08-12 (plan file
`synchronous-watching-seal.md`, "Workstream 2"), and it inherits this
document's research-to-remedy discipline rather than repeating it from
scratch:

- **F2's remedy (nested/double CV, selection on held-out splits never used for
  selection) is reused unchanged** as the reason the transformer swap
  experiment scores on the litsurvey's five acceptance metrics (localization,
  edit locality, edit persistence, rollback correctness, catastrophic
  interference) rather than on a single validation number — the same Cawley &
  Talbot lesson this document already applies to the Architecture Race.
- **F3's sample-complexity finding inverts, not repeats, at transformer
  scale.** F3 measures principle-mining as starved because a tabular run
  produces ~8–40 journal rows. The transformer workstream's KDN-as-FFN
  distillation step is fit on every token position × every layer × every
  training example of the transformer's own activations — orders of
  magnitude denser than the journal case, which is the same inversion
  `COGNITIVE_ENGINE_RESCUE.md` (2026-08-06) identifies for human-edit memory.
  F3's hierarchical-partial-pooling remedy is not needed there because the
  data scarcity it was designed to fix is absent.
- **F6's off-policy mining caution carries over as a design constraint, not a
  citation.** Where F6 recommends recording the logging policy alongside
  mined tabular rules, the transformer workstream's analogous discipline is
  requiring a control/treatment probe (the causal engine, re-aimed per
  `COGNITIVE_ENGINE_RESCUE.md`) before any KDN cell edit is applied inside a
  transformer block — attribution from a trace alone is treated as
  insufficient, the same way F6 treats naive mining as confounded.
- **F7's reproducibility discipline (explicit `random_state` everywhere,
  identity tests as the control) is the literal first measurement** in the
  transformer workstream's v0 probe: before anything else is measured, does
  the same input produce the same KDN path every time, across repeated runs.

None of the program table's 8 rows are rerun or extended by this addendum;
the transformer workstream is additive, using this document's already-proven
research-to-remedy method on a new substrate.
