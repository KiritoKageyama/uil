# UIL Version 2.0

## An Evidence-First Research Monograph for Professor Pachori

**Project:** Universal Intelligence Layer (UIL)  
**Prepared:** 26 July 2026  
**Purpose:** A supervisor-facing account of the research question, evidence, implementation, limitations, and next research decisions.  
**Status:** Consolidated from the repository, recorded Gate evaluations, the v22 blueprint, and the original project brief.

---

## Preface

UIL began as an attempt to place a persistent cognitive layer around a machine-learning system. The intuition was that a learner could record its experience, look for patterns in success and failure, preserve those patterns as editable knowledge, and use them in later decisions. That intuition produced a substantial software system: staged optimization, multiple learner families, dataset fingerprinting and routing, journals, principle mining, a knowledge graph, formula evolution, causal interventions, event-sourced history, a research interface, and a human-editable knowledge path.

It also produced an important correction. On the ordinary tabular classification settings examined so far, the cognitive shell has not produced a reliable accuracy gain over a carefully matched plain learner-plus-search baseline. In a direct cross-seed Gate 2 comparison, the default core wiring was worse. In short deterministic audits, several cognitive subsystems did not change predictions at all.

This is not a reason to hide the project or erase the work. It is the central scientific result that gives the next phase its focus. UIL is strongest when it is described as a transparent research instrument: a system for readable symbolic learning, out-of-distribution law recovery, reproducible evidence, and human-correctable knowledge. It is weakest when it is described as a universal cognitive layer that improves tabular accuracy.

This monograph makes that distinction explicit. It is written as a research design document, not as a feature list.

---

## 1. Purpose, audience, and how to read this document

### 1.1 Purpose

This document answers the questions a supervisor should be able to ask without first reading the code:

1. What scientific problem does UIL address?
2. What hypotheses were tested, and what happened?
3. What has actually been implemented?
4. Which claims are supported, unsupported, or still open?
5. What is the narrowest credible path to a thesis or publication?

It is not a software manual and does not claim state-of-the-art performance. Implementation detail appears only when it explains a research decision or enables independent review.

### 1.2 Intended audience

The primary reader is a faculty supervisor evaluating UIL as an M.Tech, early PhD, or sustained undergraduate research artifact. The document assumes familiarity with basic supervised learning, validation, and model selection, but does not assume prior familiarity with the repository.

### 1.3 Reading guide

For a 10-minute review, read the Executive Summary, Sections 3 and 8, and the explicit non-claims in Section 10.

For a 30-minute research discussion, add Sections 4 through 7. They explain the hypotheses, system design, and experimental safeguards.

For technical due diligence, read Sections 9 through 13 and the Evidence Register.

### 1.4 Evidence language

The system is large enough that code existence can easily be mistaken for scientific validation. This document therefore uses three terms consistently:

| Term | Meaning |
|---|---|
| **Observed evidence** | A result recorded by a controlled experiment, audit, unit or acceptance test, or inspectable repository artifact. |
| **Interpretation** | A reasoned explanation of evidence. It is not itself a measurement. |
| **Proposed work** | A future hypothesis, protocol, or design direction. It must not be described as achieved. |

An engineering test shows that a mechanism can execute correctly in its tested condition. It does not show that the mechanism improves learning, helps a human, or generalizes beyond that condition.

---

## 2. Executive summary

### The project in one paragraph

Universal Intelligence Layer is a research system that surrounds tabular learners with a persistent, inspectable learning record. A run can fingerprint a dataset, select or route to a learning strategy, search configurations, journal decisions and outcomes, mine tentative principles, store formulas and graph relations, run limited causal interventions, and preserve changes in replayable event history. Its distinctive engineering property is that knowledge objects are intended to remain inspectable, editable, versioned, and attributable to evidence rather than silently overwritten.

### The central question

The original broad question was whether a cognitive shell improves supervised learning. The work now separates this into two scientific questions:

1. Does UIL’s cognitive layer improve held-out accuracy or efficiency on ordinary tabular classification against a strong plain baseline?
2. If not, can UIL still provide value through readable symbolic models, out-of-distribution law recovery, and human-editable, replayable knowledge about learning?

The answer to the first question is negative in the tested regime. The answer to the second is partly supported in software and reported symbolic experiments, but it requires a real human study and broader benchmark reporting before becoming a general claim.

### Most important findings

The evidence does not support a claim that UIL cognition improves saturated tabular accuracy.

- A core-wiring Gate 2 evaluation across breast cancer, wine, and ionosphere reports a mean held-out accuracy difference of **−0.0132** for default cognitive wiring versus plain learner/search: zero wins, one tie, and two losses.
- Short deterministic existence audits on breast cancer and wine-binary found that removing each of nine cognitive subsystems left predictions unchanged. The mechanisms wrote state, but did not alter output bits in those runs.
- Utility-gated formula admission had mean change **−0.0013** across six datasets and remains default off.
- EBM-style cyclic fitting had mean change **+0.0014** across six datasets, with two wins, two ties, and two losses. It remains a routed or raced option, not a global default.
- A KAN-style basis had mean change **−0.0022** and remains default off.

The same record supports three narrower statements:

- A reported symbolic extrapolation result achieved **R² = 0.992** for symbolic recovery versus **R² = −5.52** for XGBoost. This supports a particular law-recovery and OOD direction, not general tabular classification superiority.
- SymbolicFormula implements a sparse named-basis classifier whose final decision can be expressed as a weighted combination of raw features, thresholds, rules, formula trees, and selected optional terms.
- The event/replay and human-edit path passes recorded engineering checks: an edit produces an event, materialized state can be changed, and replay reconstructs the target state. Whether that helps a human remains untested.

### Present contribution

UIL’s present contribution is not an accuracy-winning cognitive architecture. It is a carefully instrumented research platform that:

1. exposes the difference between a subsystem running and a subsystem mattering;
2. records a negative result for cognitive augmentation of ordinary tabular learning;
3. offers a readable symbolic learner and an extrapolation-oriented research path;
4. implements versioned, editable knowledge infrastructure suitable for a human-in-the-loop study; and
5. applies cross-seed gates before experimental features become defaults.

### Recommendation

The project should now be treated as three separable research threads:

1. **Symbolic readability and extrapolation:** when a transparent model can recover or communicate structure that a black-box model cannot extrapolate.
2. **Negative result and instrumentation:** why the cognitive shell was inert or harmful on saturated tabular classification, using existence audits and validation-hygiene analysis.
3. **Human-editable learning records:** a controlled pilot comparing human-plus-UIL with UIL-alone or a standard reporting baseline.

The next phase should not be another attempt to activate every cognitive module by default. It should be a smaller number of decisive experiments.

---

## 3. Research motivation and scientific framing

### 3.1 Why UIL exists

A conventional machine-learning run often leaves a thin trace: a score, a model file, and perhaps a parameter table. That may be adequate for deployment, but it is poor support for scientific inquiry. A researcher may need to know:

- Which alternatives were considered?
- Why was one configuration selected?
- Which observation motivated a rule?
- Was a pattern merely correlational or intervention-tested?
- What did the system believe at an earlier stage?
- If a human corrects a rule, what later conclusions depend on it?

UIL treats learning as a small laboratory. It retains decisions, outcomes, tentative explanations, and knowledge changes in a form intended for inspection and replay. This goal remains legitimate even if it adds no raw accuracy. A lab notebook is not valuable because it improves the accuracy of an experiment; it is valuable because it makes the experiment auditable, reproducible, and correctable.

The scientific question is whether that record leads to a meaningful downstream outcome: better law recovery, faster human correction, safer adaptation, improved reproducibility, or more trustworthy communication.

### 3.2 The original accuracy hypothesis

The original mechanism was conceptually simple:

    configure → train → validate → journal → mine patterns → store principles
        → apply selected principles to later configurations → repeat

If a system finds that some situations reliably precede high regret or poor validation outcomes, perhaps it can avoid them later. If it identifies a useful formula or route, perhaps it can apply that knowledge to similar tasks.

This only works if several conditions hold simultaneously:

1. the journal contains enough relevant variation;
2. outcomes are attributable rather than confounded by the search policy;
3. a learned rule reaches a future decision;
4. the resulting decision is evaluated without reusing the same selection-contaminated signal; and
5. any gain exceeds the compute and variance introduced by the machinery.

The measured record shows that these conditions were not jointly satisfied in the tested tabular regime.

### 3.3 Why ordinary tabular accuracy is a hard target

Strong tree ensembles are difficult to beat on many medium-sized tabular datasets. When a baseline is near the practical ceiling, a meta-layer has little remaining headroom and many ways to add variance. Configuration selection can overfit a validation signal even when every individual component seems reasonable.

The appropriate response is not to deny the result. It is to move the research target to settings where UIL’s distinctive properties are relevant:

- recovering compact laws;
- extrapolating outside the observed training region;
- exposing a model’s terms in a human-editable form;
- preserving the historical and causal context of decisions; and
- testing whether human review improves a correction or discovery task.

### 3.4 The research gap

AutoML systems search configurations. Explainable models expose feature effects. Symbolic regression searches formulas. Experiment trackers retain artifacts. Version control retains history. Human-in-the-loop systems solicit feedback.

UIL combines pieces of these ideas around one research substrate:

- a fingerprint can route a task to a learner or strategy;
- learning stages can be journaled;
- formulas, principles, graph edges, routing rules, and hypotheses can be represented as typed knowledge;
- those objects can carry provenance and versions;
- an executor can validate whether a proposal should affect later action; and
- a person can inspect, edit, reject, and replay stored knowledge.

The research gap is therefore not “no one has built an optimizer.” It is whether a transparent, typed, replayable knowledge substrate can make iterative learning more scientifically useful than disconnected logs, dashboards, and model artifacts.

---

## 4. Research questions and hypotheses

### 4.1 Research questions

**RQ1 — Accuracy and efficiency.** Does cognitive augmentation improve held-out predictive performance or reduce search cost relative to a plain search-plus-learner baseline on ordinary tabular problems?

**RQ2 — Causal relevance.** Do the cognitive mechanisms change actual decisions and predictions, rather than merely produce logs and auxiliary state?

**RQ3 — Readable predictive structure.** Can a sparse symbolic-basis learner be competitive on suitable tasks while producing an inspectable and editable decision representation?

**RQ4 — Out-of-distribution law recovery.** Can symbolic learning recover compact relations and extrapolate beyond the training region more reliably than a standard tree model on suitable tasks?

**RQ5 — Human correction.** Does editable, versioned learning knowledge improve human performance on a structured discovery or correction task?

**RQ6 — Reproducibility.** Can a reviewer reconstruct what the system believed at a past point, identify why it changed, and replay the relevant decision path?

### 4.2 Hypothesis status

| Hypothesis | Current status | Basis |
|---|---|---|
| **H1:** Cognition improves ordinary tabular accuracy. | **Falsified in the tested configuration and regime.** | Core-wiring Gate 2 is negative; earlier audits found no reliable influence. |
| **H2:** Cognitive modules materially alter predictions. | **Not supported at short audited budgets.** | Nine subsystems were prediction-inert in reported audits. |
| **H3:** Sparse readable bases can be competitive on suitable data. | **Partly supported, dataset-dependent.** | Mechanism is implemented; evidence is local rather than universal. |
| **H4:** Symbolic learning supports law recovery and OOD extrapolation. | **Supported for reported synthetic experiments.** | v22 record reports R² 0.992 versus −5.52 for XGBoost. |
| **H5:** Versioned human editing is technically feasible. | **Supported as engineering.** | Edit, report, and replay checks pass. |
| **H6:** Human editing improves a real task. | **Open.** | A protocol and simulation exist; no participant study is complete. |
| **H7:** Architecture racing is safe as a general default. | **Not supported.** | Mixed results and selection-overfitting signals. |

### 4.3 The precise null result

The evidence does not establish that every cognitive mechanism is useless. It establishes a narrower, more useful result:

> Under the recorded configurations, benchmarks, and validation protocol, adding UIL’s cognitive shell did not produce a robust held-out accuracy advantage over the baseline, and several subsystems were inert under short audited runs.

This wording avoids both overclaiming failure and evading it.

### 4.4 Why the hypotheses must remain separate

A working replay system does not prove human benefit. A synthetic extrapolation win does not prove general classification superiority. A readable model does not automatically establish causality. A thesis can contain all of these threads, but it must not use evidence from one as proof of another.

---

## 5. Design philosophy

### 5.1 Evidence before architecture

A component is not retained as a default merely because it is implemented or locally successful. It must pass a gate that asks whether it changes meaningful behavior and whether the change generalizes over a prescribed evaluation condition.

This is why several capabilities remain optional despite real implementation work: utility-gated formulas, direct formula installation, architecture racing, EBM-style cyclic fitting, KAN-style bases, and aggressive cognition-to-feature guidance.

### 5.2 Proposition, validation, and adoption

UIL should keep three roles distinct:

1. **Proposition:** a search routine, symbolic engine, graph mechanism, human, or optional language-model helper suggests a configuration, formula, rule, or hypothesis.
2. **Validation:** the executor evaluates that proposal under a stated protocol.
3. **Adoption:** only a proposal that satisfies a predefined gate changes active knowledge or a default.

A mined correlation is not ground truth. A human edit should be a versioned event rather than a silent overwrite. A language model, if used, should propose a mutation and never author system state without validation.

### 5.3 Readability is a model constraint

The readable model is not a black box followed by a post-hoc explanation. It constrains the predictive form:

    score(x) = b + Σᵢ wᵢ φᵢ(x)

For binary classification:

    P(y = 1 | x) = σ(score(x))

The basis terms φᵢ may be raw variables, thresholds, rules, formulas, or selected optional units. Sparse fitting retains only a subset. This makes the model inspectable, but not automatically causal or stable; those remain empirical questions.

### 5.4 Versioning is part of epistemology

If a research system silently overwrites its own beliefs, it cannot honestly answer “what did the system believe at stage 340, and why did that belief change?” VersionStore addresses this through append-only events, per-cell version counters, periodic snapshots, and reconstruction by replay.

A knowledge object is intended to preserve identity, type, version, provenance, confidence, and events such as create, mutate, promote, decay, human edit, and human reject. This supports reproducibility, explanation, and meaningful undo. It does not itself establish predictive value.

### 5.5 Safe defaults are scientific choices

A feature that helps one dataset is not automatically safe across datasets. The design therefore prefers routing or racing over universal activation, cross-seed means over lucky runs, isolated state for fair A/B comparisons, and explicit default-off verdicts when a gate fails.

---

## 6. System architecture

### 6.1 Architecture at a glance

    Dataset and task
        ↓
    Perception
    imputation, invertible normalization, fingerprint
        ↓
    Routing
    select a learner or strategy
        ↓
    Learning and search
    train candidates, validate, select a champion
        ↓
    Instrumentation
    journal scores, configurations, stress, regret, and provenance
        ↓
    Knowledge operations
    principles, formulas, graph relations, episodes, causal hypotheses
        ↓
    Validation and execution
    decide whether a proposal may influence later action
        ↓
    Versioned research record
    events, snapshots, reports, replay
        ↓
    Human review
    inspect, edit, reject, compare, and reconstruct

The left side is conventional machine learning. The right side is UIL’s research substrate: memory, provenance, controlled change, and human review.

### 6.2 Typical data flow

1. Data are prepared with train-consistent imputation and scaling.
2. A fingerprint records task properties and cheap measured probes.
3. The router selects a strategy or learner family.
4. The optimizer evaluates configurations through a learner adapter.
5. Each stage records outcomes and context in the journal.
6. Knowledge mechanisms may propose principles, formulas, graph hints, episodes, or interventions.
7. The executor validates whether a proposal may affect a later action.
8. Knowledge changes are preserved as events and materialized snapshots.
9. A report and interface expose the resulting record to a human reviewer.

### 6.3 Stress and regret

The recorded stress module uses a project-specific regret calculation:

    base_regret = max(0, best_proven_accuracy − current_accuracy)

    regret = base_regret × (1 + 2 × system_stress)

This is a control signal, not a universal learning objective. Its value depends on whether it leads to better, independently evaluated interventions. The v22 blueprint correctly identifies single-split grounding of regret as an epistemic weakness and proposes K-fold or repeated-split grounding for future use.

### 6.4 Perception and routing

The v22 fingerprint separates five descriptive targets:

| Axis | Question | Examples |
|---|---|---|
| Center | Where are values concentrated? | robust center statistics |
| Spread | How dispersed are they? | MAD and class-group spread |
| Shape | Are there skew, tails, or irregularity? | skewness, kurtosis, normality distance |
| Dependence | How do variables move together? | rank correlation, categorical dependence, probe scores |
| Sequence | Does collection order carry signal? | autocorrelation and run indicators |

The fingerprint is descriptive, not a causal test. Its purpose is to support auditing and routing.

The router uses task properties and cheap probes to select a plausible strategy: symbolic regression for a continuous task with strong symbolic probe, sparse readable models for structured classification, boosted or gradient methods for complex boundaries, and optional small-data strategies where installed. The router is itself a hypothesis to be evaluated on unseen tasks, not an oracle.

### 6.5 SymbolicFormula

SymbolicFormula is the core model-level artifact. It builds a candidate basis matrix from readable terms and fits a sparse predictor over it.

| Basis family | Example | Role | Maturity |
|---|---|---|---|
| Raw feature | x₇ | simple linear signal | active |
| Stump | 1[x₃ > 0.2] | threshold effect | active |
| Formula | sin(x₁), x₂ × x₅ | compact nonlinear structure | active, selection-dependent |
| Rule | 1[x₃ > 0.2 and x₇ ≤ 0.5] | readable interaction | routed/experimental |
| Neural basis | named hidden-unit response | nonlinear capacity with sparse readout | experimental |
| KAN-style term | (xⱼ − τ)₊ᵈ | plottable univariate shape | experimental |

For binary tasks, the sparse-linear path approximately solves:

    minimize over b,w:
      Σₙ log(1 + exp(−yₙ(b + wᵀΦ(xₙ)))) + λ||w||₁

Here Φ(x) is the named basis vector and the L1 term encourages a small number of retained terms. The representation is readable, but it can lack capacity on interaction-dense tasks. The gates around rule, neural, KAN, and cyclic options are therefore scientifically necessary rather than cosmetic.

### 6.6 Knowledge cells and VersionStore

The v22 abstraction is a typed KnowledgeCell: principle, formula, graph edge, routing rule, or causal hypothesis with identity, type, version, provenance, and confidence. The point is not to force all knowledge into one neural representation. It is to give all knowledge a compatible audit and mutation interface.

VersionStore persists append-only events with sequence number, timestamp, cell identity and type, source and target version, operation, payload, and cause. Snapshots reduce reconstruction cost; replay applies later events to a snapshot. Decay is an event rather than an invisible in-place change.

### 6.7 Causal interventions

Passive journal mining can find correlations generated by the optimizer’s own prior choices. UIL distinguishes this from a causal intervention engine that can force alternate actions and compare treatment and control. The recorded majority-of-three-trials confirmation logic is a pragmatic project gate, not a full causal-identification proof.

Active experiment design — choosing an intervention that best distinguishes competing hypotheses — is proposed future work.

---

## 7. Scientific methodology

### 7.1 Three levels of evaluation

UIL now requires three different tests:

1. **Engineering tests:** Does code execute, preserve state, and handle specified conditions?
2. **Existence audits:** Does removing a subsystem change output or relevant behavior?
3. **Cross-seed Gate 2 comparisons:** Does that change improve a fairly evaluated outcome across the prescribed suite?

Engineering tests can certify dead wiring. Existence tests can certify influence without benefit. Cross-seed tests without isolated state can measure contamination rather than treatment. The three levels must remain separate.

### 7.2 Baselines

The principal baseline is a matched plain learner plus search with cognitive influence disabled. This asks a specific causal question: what is the incremental effect of cognition, holding ordinary learning as comparable as possible?

Other tables compare narrower choices — cyclic fitting on/off, KAN on/off, race/classic, or symbolic versus XGBoost extrapolation. They must not be pooled into a generic “UIL wins” result.

### 7.3 Cross-seed reporting

Many Gate 2 runs use seeds 42, 7, and 123. A minimum report should state:

- datasets and selection rationale;
- seeds, stages, and folds;
- whether the learner was forced or routed;
- mean and dispersion per arm;
- mean difference and wins/ties/losses;
- runtime cost where relevant;
- state isolation; and
- a predeclared decision rule.

Three seeds improve on one seed but are not a definitive estimate of population uncertainty. Publication-quality evidence should expand tasks, use selection/evaluation separation where choices are adaptive, and include a task-level statistical analysis appropriate to the design.

### 7.4 Existence audits

An existence audit uses a deterministic repeated-run control, then removes a component and compares output. Byte-identical predictions mean the component was inert in that configuration and budget.

This does not prove it can never matter. It directs the next diagnosis: insufficient evidence, insufficient variation, failure to reach the decision path, or a proposal correctly rejected as unhelpful.

### 7.5 Selection overfitting

Architecture racing and aggressive feature guidance revealed a familiar risk: the system can choose a candidate that looks good on its internal validation signal while generalizing worse. The recorded wiring diagnosis shows that nonuniform feature guidance folded into the sparse L1 penalty could enlarge the validation-to-held-out gap.

The methodological answer is to separate selection from final evaluation, use extra or nested splits where practical, and treat long-lived adaptive reuse of a holdout as a limited resource.

### 7.6 Leakage and preprocessing safeguards

A credible evaluation must ensure that:

- imputers and scalers are fit on training data only;
- test folds are not visible to routing, mining, or promotion;
- A/B arms use isolated state directories;
- estimators receive explicit seeds;
- champion restoration preserves intended learner modes; and
- balanced accuracy or macro-aware metrics are used where raw accuracy is misleading.

### 7.7 Metrics

Raw accuracy is suitable for balanced classification with symmetric errors. On strongly imbalanced data it can conceal a majority-class predictor. The SymbolicFormula threshold path accordingly uses balanced accuracy under a recorded imbalance condition.

Symbolic research should report in-range error, OOD error, structural recovery, formula complexity, noise sensitivity, and seed stability. HITL work should report human outcomes such as correction accuracy, time to correct, confidence calibration, and reproducibility of the reviewed conclusion.

---

## 8. Implementation maturity

“Implemented” means code and recorded tests or artifacts exist. It does not mean the mechanism has proven scientific utility.

| Subsystem | Why it exists | How it works | Current maturity |
|---|---|---|---|
| UniversalOptimizer | Coordinate the staged experiment. | Connects preparation, routing, search, learning, cognition, persistence, and reports. | Implemented; large integration method remains a debt. |
| Learner adapters | Compare model families under one interface. | Shared fit, predict, configuration, and capability surface. | Implemented. |
| SymbolicFormula | Build readable predictive expressions. | Named basis matrix plus sparse or cyclic classifier. | Implemented; value is task-dependent. |
| Formula engine/library | Search and preserve symbolic structures. | Formula trees are evaluated, mutated, scored, and stored. | Implemented; default impact unproven. |
| Rule, neural, KAN bases | Add controlled capacity. | Readable rules or optional named nonlinear terms enter the basis. | Implemented, experimental/default off as applicable. |
| Search and PSO | Explore configuration space. | Stage-wise candidate evaluation. | Implemented; cognition replacing search remains unproven. |
| Architecture race | Choose among alternatives. | Evaluates axis variants and selects a champion. | Implemented; selection bias requires caution. |
| Perception | Make input preparation explicit. | Missing-data handling, normalization, fingerprint. | Implemented; clean-data impact neutral as designed. |
| Domain router | Avoid one learner everywhere. | Fingerprint and probes select a strategy. | Implemented; generalization is open. |
| Journal and miner | Retain and propose learning patterns. | Logs stages; derives tentative principles. | Implemented; short isolated runs often lack evidence. |
| Principle library/executor | Apply validated knowledge. | Principles can create limited overrides. | Implemented; inert in reported short audits. |
| Knowledge graph | Retain relations and contextual hints. | Co-activation relations and traversal. | Implemented; inert in reported short audits. |
| Causal engine | Test action rather than only observe. | Treatment/control intervention trials. | Implemented; relevant only if it alters evaluated action. |
| Episodic memory | Retrieve similar past experience. | Fingerprint-based hints and routing evidence. | Implemented; inert in reported short audits. |
| VersionStore | Preserve history and support replay. | Event log, snapshots, reconstruction. | Implemented; edit/replay checks pass. |
| Lab reports and HITL | Make evidence reviewable and correctable. | Report generation, human edit/reject events, replay. | Engineering path implemented; human benefit open. |
| Web and desktop UI | Give users access to research artifacts. | Research console and shell. | Implemented; usability unstudied. |

### Engineering strengths

The codebase is strong as a research platform. It emphasizes deterministic seeds, isolated state for comparisons, event history, recorded failure analysis, default-off decisions, and explicit identity checks.

### Known debts

The central optimizer is complex. Optional-feature interactions and historical broad exception handling add cognitive load. The recommended refactor is behavior-preserving: isolate preparation, routing, search, cognition, persistence, and reporting behind tested boundaries. Refactoring must not silently change scientific defaults.

---

## 9. Experimental results

### 9.1 Core cognitive wiring versus plain learning

The decisive direct comparison used 12 stages and seeds 42, 7, and 123.

| Dataset | UIL off | Default cognitive wiring | Difference | Connected configuration | Difference |
|---|---:|---:|---:|---:|---:|
| Breast cancer | 0.9708 ± 0.015 | 0.9591 ± 0.015 | −0.0117 | 0.9591 ± 0.015 | −0.0117 |
| Wine | 0.9537 ± 0.013 | 0.9259 ± 0.047 | −0.0278 | 0.9259 ± 0.047 | −0.0278 |
| Ionosphere | 0.9202 ± 0.037 | 0.9202 ± 0.040 | −0.0000 | 0.9202 ± 0.040 | −0.0000 |

**Observed evidence:** Mean default-versus-off difference is **−0.0132**, with zero wins, one tie, and two losses.

**Interpretation:** The cognitive connection was not merely neutral. In this configuration, it made performance worse. A wiring diagnosis points to nonuniform feature guidance being folded into the sparse L1 penalty, which enlarged validation-to-held-out gaps.

**Decision:** Do not describe default cognition as an accuracy improvement. Keep it disabled or constrained until a new mechanism clears a fair gate.

### 9.2 Existence audit: running versus mattering

On breast cancer at five stages and wine-binary at six stages, each of the following was prediction-inert under the reported deterministic audit:

| Subsystem | Verdict |
|---|---|
| Principles | Inert |
| Knowledge network | Inert |
| Formulas | Inert |
| Causal mechanism | Inert |
| Episodic memory | Inert |
| Structural representation | Inert |
| Meta-principles | Inert |
| Safeguards | Inert |
| Goal cell | Inert |

**Observed evidence:** Removing all nine subsystems produced byte-identical predictions in the specified short runs.

**Lesson:** A subsystem must be evaluated for both existence and outcome. Logs, state changes, and software objects do not establish influence on the learner.

### 9.3 Utility-gated formulas

| Dataset | Classic | Gate on | Difference |
|---|---:|---:|---:|
| Wine | 0.9537 | 0.9537 | +0.0000 |
| Sonar | 0.7063 | 0.6984 | −0.0079 |
| Ionosphere | 0.9155 | 0.9155 | +0.0000 |
| Breast cancer | 0.9708 | 0.9708 | +0.0000 |
| Digits | 0.9500 | 0.9500 | +0.0000 |
| Spambase | 0.9439 | 0.9439 | +0.0000 |

**Observed evidence:** Mean change **−0.0013**, with zero wins, five ties, and one loss. A formula-only audit was also inert on breast cancer at 12 stages.

**Decision:** Default off. Formula admission is conceptually better when utility-gated, but its current Gate result is negative.

### 9.4 EBM-style cyclic fitting

| Dataset | Classic | EBM-style fit on | Difference |
|---|---:|---:|---:|
| Wine | 0.9537 | 0.9537 | +0.0000 |
| Sonar, forced sparse-linear | 0.7698 | 0.7778 | +0.0080 |
| Ionosphere | 0.9155 | 0.9202 | +0.0047 |
| Breast cancer | 0.9708 | 0.9678 | −0.0030 |
| Digits | 0.9500 | 0.9500 | +0.0000 |
| Spambase | 0.9439 | 0.9428 | −0.0011 |

**Observed evidence:** Mean change **+0.0014**, with two wins, two ties, and two losses.

**Decision:** Retain as a race or routed option. It is not justified as a universal default.

### 9.5 KAN-style basis

| Dataset | KAN off | KAN on | Difference |
|---|---:|---:|---:|
| Wine | 0.9537 | 0.9537 | +0.0000 |
| Sonar | 0.7698 | 0.7540 | −0.0158 |
| Ionosphere | 0.9155 | 0.9155 | +0.0000 |
| Breast cancer | 0.9708 | 0.9708 | +0.0000 |
| Digits | 0.9500 | 0.9509 | +0.0009 |
| Spambase | 0.9439 | 0.9457 | +0.0018 |

**Observed evidence:** Mean change **−0.0022**.

**Interpretation:** Univariate nonlinear terms do not solve an interaction-capacity problem. The sonar loss is consistent with that limitation.

**Decision:** Default off; retain only as an experimental race axis.

### 9.6 Architecture race

A race with incumbent margin 0.01 produced mean difference **+0.0010**, but this hides meaningful regressions: sonar improved by +0.0318 and digits by +0.0213, while ionosphere lost −0.0470 and breast cancer lost −0.0030.

**Decision:** The gate failed. A mean near zero cannot justify a global default when it contains unacceptable failures. Future races require clearer separation of selection from evaluation.

### 9.7 Missing-data perception and safe defaults

The clean-data impact check reports exactly zero change across wine, sonar, ionosphere, breast cancer, digits, and spambase. This is an engineering identity result: adding the missing-data path does not perturb clean input as designed. It is not evidence of robustness to real missingness.

The Category D combination of maturity-soft normalization, stress-adaptive blending, and situation adaptation reports mean **+0.0001**, with one small win and five ties. It passes as safe under the project gate, not as a material predictive advance.

### 9.8 Symbolic recovery and extrapolation

The v22 ground-truth constraints record symbolic recovery at **R² = 0.992** versus **R² = −5.52** for XGBoost on an out-of-range task.

This is the clearest positive direction because the evaluation matches the model’s structural advantage. A tree may fit locally but extrapolate poorly; a recovered functional law can generalize outside the observed domain when the expression class is appropriate.

The limitation is equally important: a single reported contrast does not establish superiority over modern symbolic regression systems, noisy physical-law benchmarks, or arbitrary OOD classification tasks. A proper benchmark is the next experiment.

### 9.9 Editable knowledge and HITL engineering checks

The component/HITL record reports passing checks for human-edit events, lab reports, replay reconstruction, materialized edit plus VersionStore replay, human rejection, and a simulated symbolic sin-recovery pilot. The simulation reports R² = 0.9647 alone and no correction because the simulated first pass was already above threshold.

**Interpretation:** The software path is ready for study. A simulated user is a test fixture, not evidence that a person benefits.

---

## 10. Lessons learned and current direction

### 10.1 Failure has several meanings

The project’s negative results include distinct cases:

1. **Inert:** a component did not alter predictions in an audited run.
2. **Connected but harmful:** a component altered the decision path and overfit an internal validation signal.
3. **Under-evidenced:** the journal lacked enough varied examples to support reliable meta-learning.
4. **Locally useful but globally unsafe:** a feature helped one dataset and hurt another.

These require different remedies. Inert wiring needs diagnosis. Harmful guidance needs better validation and regularization. Under-evidenced mining needs a cross-task evidence design. Mixed features need routing or default-off restraint.

### 10.2 Saturated baseline headroom

When tree baselines already perform strongly on a tabular task, a cognitive shell has little headroom and several new variance sources. UIL should not compete where its special capabilities are irrelevant. Its distinctive value appears when the task requires readable structure, extrapolation, historical accountability, or human correction.

### 10.3 Selection bias is central

Every adaptive decision spends validation information. Architecture races and feature guidance made this visible. A candidate that wins a frequently reused validation signal may be a validation-specific winner.

Future work should use distinct selection and final-evaluation data, or a carefully controlled reusable-holdout mechanism. This is particularly important for a system whose purpose is to make many meta-decisions.

### 10.4 Meta-learning needs evidence

A principle miner cannot learn the effect of an architecture knob if the knob never changes. It also cannot learn reliable transferable structure from a tiny journal on one dataset. The project identifies a real tension:

- fair A/B comparisons need isolated state;
- meta-learning needs a predeclared cross-task corpus and held-out transfer evaluation.

These must be treated as separate protocols.

### 10.5 Passive logs are confounded

The journal reflects what the optimizer chose to try. A correlation between a configuration and regret can encode search policy rather than causal effect. UIL’s appropriate distinction is: passive mining yields correlational candidates; interventions are needed for causal claims.

### 10.6 The defensible thesis now

> UIL is a transparent research substrate for learning systems. It makes hypotheses, model terms, routes, and human corrections inspectable and replayable; it demonstrates a negative result for generic cognitive augmentation on saturated tabular classification; and it identifies symbolic extrapolation and human-correctable knowledge as the appropriate next evaluation targets.

### 10.7 Explicit non-claims

UIL should not currently claim:

- that it beats XGBoost or strong tree ensembles on standard tabular benchmarks through cognition;
- that a knowledge graph or principle library automatically improves learning;
- that versioning has demonstrated human benefit;
- that a mined rule is causal without intervention evidence;
- that a symbolic OOD result proves broad real-world superiority; or
- that implementation or unit tests establish scientific utility.

---

## 11. Future research programme

### 11.1 Priority A: symbolic extrapolation benchmark

Build a frozen evaluation suite containing controlled synthetic functions, selected Feynman-style tasks, and train-box/test-shell extrapolation splits. Compare:

- UIL symbolic regressor;
- a strong tree baseline;
- a neural or kernel baseline where appropriate; and
- an established symbolic-regression baseline where feasible.

Report in-range fit, OOD error, structural recovery, formula complexity, runtime, noise sensitivity, and seed stability. The goal is not universal victory; it is to identify the regime in which structural recovery matters.

### 11.2 Priority B: human-in-the-loop pilot

The first real HITL study must test a human outcome, not an API path. Compare a participant using a standard model report with one using UIL lab reports and editable knowledge cells. If possible, include a UIL interface with edit/replay disabled to isolate the feature.

Tasks can include detecting an intentionally seeded wrong rule, correcting a formula under changed units, or resolving conflicting run histories. Predeclare correction accuracy, time to correct, confidence calibration, and reproducibility of the final conclusion.

### 11.3 Priority C: negative result and instrumentation paper

The contribution should not be “we tried cognition and it did not work.” It should be the method:

- matched baselines;
- identity-based existence audits;
- explicit wiring diagnosis;
- selection-bias analysis;
- default-off gates;
- reproducibility safeguards; and
- transparent reporting of failed hypotheses.

### 11.4 Priority D: reduce to a testable core

The clearest primary pipeline is:

    fingerprint → learner or SymbolicFormula → VersionStore → lab report → human review

The broader cognitive architecture can remain available as a research laboratory, but it should not obscure the central thesis.

### 11.5 Medium-term questions

Only after the immediate studies are complete should UIL pursue questions such as whether a learned router outperforms its heuristic version, whether cross-task partial pooling yields calibrated principles, whether active interventions resolve hypotheses faster than passive mining, and whether neural components can satisfy a strict distillation round-trip from discrete knowledge to readable validated knowledge.

---

## 12. Codebase guide

### 12.1 Repository map

| Area | Role |
|---|---|
| core/ | Optimizer, preprocessing, normalization, and run coordination. |
| adapters/ | Learner interfaces and implementations, including SymbolicFormula. |
| cognition/ | Journal, mining, principles, graphs, formulas, routing, causality, VersionStore, and reports. |
| search/ | PSO and hybrid search behavior. |
| experiments/ | Evaluation harnesses and study protocols. |
| tests/ | Unit, milestone, reproducibility, and acceptance tests. |
| results/ | Gate summaries, reports, events, snapshots, and experiment artifacts. |
| webui/ | Research console backend and frontend. |

### 12.2 Recommended code review order

1. Read the Gate summaries in results/.
2. Read the cross-seed harness in experiments/.
3. Review adapters/symbolic_formula_adapter.py for the readable learner.
4. Review cognition/version_store.py and cognition/lab_report.py for the audit/replay thesis.
5. Review cognition/domain_router.py and cognition/fingerprint.py for routing.
6. Review core/optimizer.py last, because it is the complex integration point.

### 12.3 Reproducible entry points

From the parent directory of the repository, the original brief identifies:

    python figpso_uil/tests/test_v21_smoke.py

    python figpso_uil/tests/test_m0_epistemology.py
    python figpso_uil/tests/test_m1_version_store.py
    python figpso_uil/tests/test_m2_perception.py
    python figpso_uil/tests/test_m3_routing.py
    python figpso_uil/tests/test_m4_scientist.py
    python figpso_uil/tests/test_m5_hitl.py

    python -m figpso_uil.experiments.eval_crossseed

    python -m uvicorn figpso_uil.webui.backend.main:app --port 8000

Before quoting accuracy in a thesis or paper, rerun the sanctioned evaluation with an isolated state directory and archive the revision, configuration, seeds, and raw results.

### 12.4 Default-change checklist

Any change to a default should include:

1. intended mechanism and decision path;
2. existence audit;
3. matched baseline;
4. cross-seed, multi-dataset result;
5. regressions as well as means;
6. runtime impact;
7. rollback switch;
8. state-isolation and leakage statement; and
9. a note on adaptive selection and holdout reuse.

---

## 13. Professor FAQ

### Is UIL AutoML?

Partly. It searches configurations and routes strategies, but its central aspiration is persistent, editable, evidence-linked knowledge about learning rather than opaque hyperparameter racing.

### Does UIL beat XGBoost?

Not as a general claim. The cognitive-layer comparisons do not support it. UIL’s credible lead is symbolic extrapolation, where a structural law may generalize outside the training range in ways a tree does not.

### What is the one-sentence thesis?

UIL is an editable learning laboratory that measured a null result for cognitive augmentation of saturated tabular classification and now focuses on transparent symbolic discovery, replayable knowledge, and human correction.

### Why keep modules that did not improve accuracy?

Some are useful instruments for understanding why a system does or does not change outcomes. Keeping them as opt-in research mechanisms does not mean claiming they improve the default learner.

### What is an existence audit?

A deterministic ablation: remove one subsystem and ask whether the output changes. It separates a mechanism that runs from one that matters.

### What is SymbolicFormula?

A classifier whose score is a sparse weighted combination of named terms such as features, thresholds, rules, and formula trees.

### Is a symbolic model automatically causal?

No. Readability makes a relationship inspectable; it does not identify a causal relationship.

### What is VersionStore for?

It records knowledge changes as append-only events and reconstructs prior states. This makes provenance, replay, and edits auditable.

### Does VersionStore prove human usefulness?

No. It proves technical capability. Human benefit requires a controlled study.

### What is a principle?

An evidence-linked stored rule or recommendation. Unless intervention-tested, it should be read as correlational.

### Why are mined principles not automatically causal?

The journal reflects the optimizer’s own choices, not a randomized observational process.

### Why not turn every feature on?

Several recorded features are neutral or harmful as global defaults, and feature interactions can create failures that individual tests miss.

### What is Gate 2?

The cross-seed, multi-dataset project gate for changing a default. A unit test pass is not a Gate 2 pass.

### What should a supervisor ask for next?

A real HITL pilot with predeclared human endpoints and a broader symbolic extrapolation benchmark.

### Is UIL thesis-ready?

Yes, if it is focused. A credible thesis can present the negative tabular result, the instrumentation methodology, the symbolic/OOD direction, and one completed human or reproducibility study. It should not make every module an independent thesis claim.

### What should not be said?

Do not say “UIL learns how to learn and beats XGBoost.” Say what was built, what was tested, what failed, and why the next claim is narrower and better supported.

---

## 14. Evidence register

| Repository artifact | Contribution |
|---|---|
| PROFESSOR_PROJECT_BRIEF.md | Original concise project summary and operational guide. |
| RESEARCH_FAILURE_ANALYSIS.md | Measured failures, methodological interpretation, and proposed remedies. |
| UIL_V22_SCIENTIST_BLUEPRINT.md | v22 design constraints, KnowledgeCell/VersionStore design, and milestones. |
| results/try_uil_connect/SUMMARY.md | Core-wiring versus off comparison; mean −0.0132. |
| results/component_hitl/SUMMARY.md | Nine-subsystem existence audit and HITL engineering checks. |
| results/gate2_utility/SUMMARY.md | Utility-gated formula Gate 2; mean −0.0013. |
| results/gate2_utility/EXISTENCE_SUMMARY.md | Formula existence audit. |
| results/gate2_ebm/SUMMARY.md | EBM-style fitting; mean +0.0014. |
| results/gate2_kan/SUMMARY.md | KAN-style basis; mean −0.0022. |
| results/gate2_incumbent_margin/SUMMARY.md | Architecture race evaluation and failed default gate. |
| results/gate2_category_d/SUMMARY.md | Safe-default Category D result; mean +0.0001. |
| results/impact_missing_data/SUMMARY.md | Clean-data identity check for missing-data perception. |
| results/diagnose_wiring_drop/SUMMARY.md | Feature-guidance overfitting diagnosis. |
| adapters/symbolic_formula_adapter.py | Readable named-basis learner and optional capacities. |
| cognition/version_store.py | Event sourcing, snapshots, and replay. |
| cognition/domain_router.py | Routing registry, probes, and memory. |
| cognition/uil_stress.py | Stress and regret calculation. |

The repository’s failure analysis situates these findings alongside research on tabular baselines, model-selection bias, adaptive holdouts, meta-learning, counterfactual learning from logged data, symbolic library learning, explainable boosting, foundation tabular models, and KAN-style functions. Those references frame UIL’s hypotheses; they do not substitute for reproducing UIL’s own measurements.

---

## 15. Glossary

**Ablation:** Removing a component to estimate its incremental effect.

**Architecture race:** A search procedure that compares configuration or basis alternatives and selects a candidate.

**Basis term:** A named function of inputs used as a design-matrix column.

**Cognitive shell:** UIL’s journal, mining, knowledge, graph, causal, memory, and replay mechanisms surrounding a learner.

**Connected:** A subsystem changes relevant downstream behavior under a specified audit. Connected does not mean beneficial.

**Cross-seed evaluation:** Repeating evaluation under multiple random seeds.

**Default off:** Available but not active in ordinary runs because safety or benefit evidence is insufficient.

**Existence audit:** An identity-focused ablation that asks whether removal changes output.

**Fingerprint:** A compact descriptive summary of a dataset used for auditing and routing.

**Gate 2:** The project’s cross-seed, multi-dataset decision rule for changing a default.

**HITL:** Human in the loop; here, inspecting, editing, rejecting, or replaying stored knowledge.

**Inert:** Removing a subsystem leaves the audited output unchanged. This is specific to the tested condition.

**KnowledgeCell:** A typed, versioned, provenance-carrying unit such as a principle, formula, edge, route, or hypothesis.

**OOD extrapolation:** Prediction beyond the input range covered by training data.

**Principle:** A stored evidence-linked rule; without intervention it is correlational.

**Replay:** Reconstructing prior state from snapshots and events.

**Routing:** Selecting a learner or strategy from measured task characteristics.

**Sparse model:** A model retaining relatively few nonzero basis weights.

**Symbolic recovery:** Recovering a compact functional expression from data.

**VersionStore:** UIL’s append-only event and snapshot mechanism for preserving knowledge history.

---

## Closing statement

UIL is worth supervising because it makes its own claims falsifiable and preserves the record when they fail.

The project contains more than a typical student prototype: a genuine experimental substrate, readable model family, replayable knowledge infrastructure, recorded gates, and a history of diagnosed correctness failures. Its principal challenge is focus, not ambition.

The strongest next step is not another architectural layer. It is a small number of decisive studies:

1. establish the scope of symbolic extrapolation with a rigorous benchmark;
2. establish whether editable history helps a person perform a real correction or discovery task; and
3. publish the tabular null result as a lesson in instrumentation, selection bias, and scientific restraint.

That programme can turn UIL into a coherent thesis: not a claim that cognition automatically improves every learner, but a rigorous account of how transparent, editable, evidence-preserving learning systems should be designed and tested.


---

## 16. Complete experimental protocol

This section turns the evaluation principles above into an operational protocol. Its purpose is to prevent a future experiment from being interpreted more broadly than it was designed to support.

### 16.1 Start with a claim, not a feature

Every experiment should begin with one primary claim written before the result is known. Examples:

- “Adding EBM-style cyclic fitting improves OOD accuracy of the readable learner on the specified symbolic benchmark.”
- “Access to replayable knowledge cells improves a participant’s ability to detect and correct a seeded model error.”
- “A router trained on the specified meta-training tasks improves mean rank on held-out tasks over the fixed heuristic router.”
- “The cognitive layer changes selected configurations and improves final held-out score under a selection protocol separated from final evaluation.”

A feature such as “use KAN basis” is not a claim. It is an intervention. Naming the outcome, comparison, task population, and decision rule first makes the result intelligible.

### 16.2 Define the unit of analysis

UIL has several possible units of analysis, and mixing them creates ambiguity:

| Study type | Correct unit of analysis | Typical outcome |
|---|---|---|
| Learner comparison | Dataset-task × seed × split | held-out score, rank, runtime |
| Symbolic recovery | Equation/problem × seed × train region | OOD error, exact recovery, complexity |
| Routing evaluation | Held-out task | strategy rank or regret against best available strategy |
| Existence audit | Controlled run | changed predictions, changed selected configuration, changed event path |
| HITL study | Participant × task | correction accuracy, time, confidence, explanation quality |
| VersionStore test | Event sequence | exact reconstruction, event completeness, replay time |

A participant trial should not be analyzed as a model seed. A formula-recovery problem should not be treated as one row in an ordinary classification table without reporting its structural properties.

### 16.3 Pre-register the comparison arms

For any substantive Gate 2 or publication experiment, record the following before execution:

1. **Control arm.** Define the baseline and list every feature that is disabled.
2. **Treatment arm.** Define exactly one intended change, unless an interaction is itself the predeclared object of study.
3. **Shared conditions.** Use the same preprocessing, folds, learner family, resource budget, seed set, and stopping rule wherever possible.
4. **Allowed deviations.** State whether routing can choose different learners, whether a mode is forced, and which optional dependencies are allowed.
5. **State isolation.** Give each arm a separate state directory unless shared memory is the experimental intervention.
6. **Success criterion.** State what outcome would justify a default change or paper claim.
7. **Failure criterion.** State which regressions, costs, or uncertainty conditions force default off.

The current repository’s Gate summaries model part of this discipline. The next generation should save a machine-readable experiment manifest with every run.

### 16.4 Data splitting and adaptive decisions

The central data-splitting rule is:

> Data used to choose an option should not be the only data used to certify that the option works.

A practical three-level structure is:

    Training folds
        fit models and transformations
    Inner validation or selection folds
        choose configurations, routes, formula candidates, and race winners
    Outer test or final evaluation folds
        compare the selected procedure against the baseline

For expensive experiments, a complete nested scheme may be impractical. In that case, the document should state the compromise rather than imply full independence. A reusable standing holdout is particularly vulnerable when many choices are made after inspecting it.

### 16.5 Cross-seed and cross-task evaluation

The recorded seed set of 42, 7, and 123 is a useful deterministic start. A stronger study should specify whether seeds vary:

- learner initialization;
- split construction;
- search trajectory;
- formula mutation;
- architecture-race ordering; and
- any optional proposal generator.

For each task, retain individual seed outcomes. Then report both:

1. the within-task distribution over seeds; and
2. the across-task distribution of per-task differences.

A mean pooled over all task-seed rows can obscure a feature that is harmless on many easy tasks and sharply harmful on one important class of task. The project’s architecture-race result demonstrates precisely this risk.

### 16.6 Reporting a default-change decision

A decision table should include more than a mean difference.

| Field | Why it is required |
|---|---|
| Task name and family | Reveals whether a result is limited to one problem type. |
| Baseline and treatment configuration | Makes the intervention identifiable. |
| Mean and dispersion | Shows both central tendency and seed sensitivity. |
| Difference and direction | Prevents readers from inferring a gain from two raw scores. |
| Wins, ties, and losses | Reveals uneven safety across tasks. |
| Runtime, memory, or stage cost | Tests any efficiency claim. |
| Selection protocol | Reveals adaptive-evaluation risk. |
| State directory and persistence policy | Prevents hidden transfer or contamination. |
| Verdict and rollback | Documents the actual engineering decision. |

If a feature is retained as a race axis after failing as a global default, the report should say so explicitly. “Useful in some settings” and “safe globally” are different conclusions.

### 16.7 Existence-audit protocol

A rigorous existence audit should have the following steps:

1. Freeze the dataset, split, seed, stage budget, learner mode, and state directory.
2. Run the full configuration twice. Confirm deterministic equality of the chosen audit outputs.
3. Disable exactly one subsystem while preserving all unrelated behavior.
4. Compare prediction arrays, selected configurations, key events, model terms, and relevant persisted state.
5. Classify the result as inert, connected-but-neutral, connected-beneficial, connected-harmful, or indeterminate.
6. If inert, trace the expected decision path rather than immediately re-enabling the feature.
7. If connected, promote the comparison to a cross-seed Gate before altering a default.

Predictions are the strictest initial output. For components intended to change a route, a formula library, or a human report rather than predictions, the audit should also report the intended artifact and its downstream consequence.

### 16.8 Causal-intervention protocol

A causal label should require more than a mined association. A minimal protocol is:

1. State the hypothesis, intervention, expected direction, and competing explanations.
2. Define control and treatment configurations that differ only in the intervention.
3. Repeat under the prescribed seeds or trial count.
4. Measure the specified outcome and any relevant side effects.
5. Retain all outcomes, including failed trials.
6. Mark the conclusion as supported, refuted, or unresolved rather than forcing a binary causal verdict.

The recorded majority-of-three rule is useful as an engineering guardrail, but a publication should report the actual effect sizes and trial-level results. “Confirmed causal” should be reserved for the scope the intervention truly identifies.

### 16.9 Reproducibility packet

Every publication-quality experiment should archive:

- repository revision or immutable source snapshot;
- dependency and hardware environment;
- command line and machine-readable configuration;
- dataset provenance and split indices;
- seed list;
- state-directory policy;
- raw per-run metrics;
- raw event and report artifacts where ethically shareable;
- generated tables and plotting code; and
- a short statement of deviations from the planned protocol.

The point is not bureaucratic completeness. It is to make a negative or mixed result just as reproducible as a positive one.

---

## 17. Scientific claim matrix

A claim matrix prevents architectural enthusiasm from silently changing the meaning of evidence.

| Candidate claim | Evidence needed | Current evidence | Status | Next decisive test |
|---|---|---|---|---|
| UIL cognition improves ordinary tabular accuracy. | Matched, cross-task, cross-seed held-out gain and acceptable cost. | Direct core-wiring Gate is negative; short audits inert. | Refuted for tested default regime. | Do not prioritize unless a materially different, preregistered mechanism is proposed. |
| UIL cognition improves efficiency. | Same or better accuracy at lower search cost under a fair resource budget. | No such supported result; extra machinery has overhead. | Unsupported. | Test only if cognition replaces search decisions rather than decorates them. |
| A cognitive subsystem changes the decision path. | Existence audit and traceable causal path. | Nine short-run subsystems inert; some intervention paths can be connected. | Mixed and configuration-specific. | Audit at a budget where the intended preconditions are met. |
| SymbolicFormula is readable. | Inspectable small named basis with reproducible rendering. | Implemented in code and reports. | Supported as representation. | Study whether readers understand and use the representation correctly. |
| SymbolicFormula is competitive on selected tasks. | Cross-seed, fair model-family comparison by task family. | Local parity and mixed gate evidence. | Partly supported. | Focused readable-model benchmark. |
| Symbolic learning extrapolates better than a tree on suitable laws. | Multi-problem OOD benchmark with structural and predictive metrics. | One reported strong contrast, R² 0.992 versus −5.52. | Promising but incomplete. | Frozen symbolic/OOD benchmark. |
| Routing selects a better strategy than one fixed learner. | Held-out task-family comparison against heuristic and fixed baselines. | Router is implemented; no broad held-out task study recorded here. | Open. | Meta-train/meta-test routing benchmark. |
| VersionStore preserves reconstructable history. | Event, snapshot, and replay identity tests. | Acceptance checks recorded as passing. | Supported as engineering. | Scale and fault-injection study if reliability is a publication target. |
| Human editing improves a task. | Participant study with predeclared endpoints. | Software path and simulation only. | Open. | Human pilot. |
| Mined principles support causal claims. | Intervention study with explicit assumptions and effect report. | Logging-policy annotation and causal engine exist. | Not established broadly. | Controlled intervention experiments. |
| Race-based architecture selection is safe by default. | Robust no-regression result with selection/evaluation separation. | Mixed results and clear losses. | Unsupported. | Nested selection study, if strategically valuable. |

### 17.1 What this matrix changes in practice

The matrix changes presentation. Rather than saying “the system has routing, causal reasoning, and HITL,” a faculty presentation should say:

- routing is implemented and awaiting a held-out task-family study;
- causal interventions are implemented but causal effectiveness remains an experimental question;
- HITL editing is implemented, while human benefit remains the central open claim.

This is not a weakness in presentation. It is the correct grammar of research maturity.

### 17.2 Evidence, interpretation, and future work

A useful rule for writing about UIL is:

| Statement type | Example | Permitted wording |
|---|---|---|
| Evidence | A Gate table reports −0.0132. | “The recorded Gate found…” |
| Interpretation | Guidance overfit the internal selection signal. | “The pattern is consistent with…” |
| Future work | Nested evaluation may reduce selection bias. | “We will test whether…” |

Avoid upgrading an interpretation into evidence or a proposal into an achievement. Doing so is particularly important for terms such as intelligent, adaptive, causal, self-improving, and editable scientist.

---

## 18. Component-interaction trace

The following trace explains how the system’s components are intended to interact and where the evidence currently shows weak links.

### 18.1 Stage-level trace

1. **Dataset arrives.** The run receives features, target, task metadata, and an isolated state location.
2. **Perception executes.** Missing-data handling and normalization prepare train-consistent input. Fingerprinting derives descriptive and probe signals.
3. **Router chooses a strategy.** It may select symbolic, sparse-readable, gradient/boosted, tiny-data, or another registered strategy. The rationale should be retained in the run record.
4. **Optimizer proposes candidates.** Search evaluates configurations, learner modes, and optionally architecture axes.
5. **Learner fits.** For SymbolicFormula, the adapter composes a basis matrix and chooses a sparse or optional cyclic fit.
6. **Validation produces evidence.** Scores, selected configurations, model terms, and resource use are available for the journal.
7. **Stress and regret update.** The system converts current performance and history into control signals.
8. **Knowledge modules observe or propose.** The miner may propose principles; the formula engine may propose terms; graph, episodic, and routing memory may provide context; causal logic may request a controlled action.
9. **Executor applies only admissible influence.** A proposal may adjust a configuration, feature emphasis, or candidate set if it reaches the relevant decision path.
10. **VersionStore records changes.** Knowledge mutation, promotion, decay, human editing, and rejection become events.
11. **Report renders evidence.** The lab report connects route, outcomes, hypotheses, changes, and human actions.
12. **Human reviews.** A reviewer can accept, edit, reject, or replay knowledge.

### 18.2 Weak links identified by evidence

The recorded results identify four important weak links.

**Journal to principle.** A miner cannot induce reliable principles without diverse evidence. Short isolated runs often produce no reachable principles, so later executor behavior cannot depend on them.

**Principle to action.** A stored principle must reach a configuration or learner decision. In the short existence audits, removing principles did not alter predictions.

**Action to fair outcome.** Even when cognition influences action, the selected action can overfit internal validation. Aggressive feature guidance illustrates this failure mode.

**Artifact to human benefit.** An editable report or replay system can work technically without improving a person’s reasoning. This is the open HITL gap.

### 18.3 Why the trace is valuable

A large project often presents a component diagram that makes every arrow appear equally real. The trace above asks a stronger question at each arrow:

> What observation would show that this transition happened, and what downstream outcome would show that it mattered?

This question should become a standard requirement for every future subsystem.

### 18.4 Example: formula proposal path

A formula proposal is scientifically meaningful only if the following chain can be demonstrated:

    mutation event
        → candidate formula evaluated
        → held-out utility check
        → admission to active basis
        → nonzero selected coefficient or selected stage
        → changed model prediction or OOD error
        → cross-seed comparison
        → readable provenance for review

The utility-gated formula work improved the first half of this chain by preventing indiscriminate admission. Its Gate result shows that the later half was not achieved on the recorded suite.

### 18.5 Example: human edit path

A human edit is meaningful only if the chain is explicit:

    reviewer sees a readable cell and provenance
        → reviewer edits or rejects it
        → VersionStore logs the event
        → materialized state updates
        → replay reproduces the changed state
        → a later task reveals whether the correction improved a defined outcome

The current system supports the first five links as engineering. The last link is the purpose of the human study.

---

## 19. Threats to validity

A professional research document should state how it could be wrong.

### 19.1 Internal validity

**Shared state contamination.** Persistent memory can leak information across experimental arms. This is beneficial only when transfer is the treatment. Otherwise, state directories must be isolated.

**Preprocessing leakage.** Fitting scaling, imputation, fingerprint thresholds, or promotion rules on test data can inflate reported performance. Train-only fitting and auditable transformations are mandatory.

**Uncontrolled feature interactions.** A change to cognition can alter search, basis selection, routing, or persistence. One change at a time is preferable for diagnosis; intentional interactions require factorial design.

**Nondeterminism.** An unseeded dependency can invalidate identity checks. Explicit seeds and repeated controls are essential.

### 19.2 Construct validity

**Accuracy as the wrong outcome.** If UIL’s distinct value is readability or correction, raw accuracy alone is an incomplete endpoint.

**Readability as an assertion.** A named formula is not automatically understandable to a target user. Human comprehension should be measured.

**Replay as usefulness.** State reconstruction shows technical fidelity, not that a reviewer can identify a better scientific conclusion.

**Causal language.** An intervention engine does not make every stored principle causal. Claims must match the intervention and its assumptions.

### 19.3 Statistical conclusion validity

**Small seed counts.** Three seeds offer a basic stability check but can miss variability.

**Dataset selection.** A small set of familiar UCI-style tasks may not represent the conditions where UIL’s special affordances matter.

**Multiple adaptive comparisons.** Repeatedly inspecting Gate results increases false-discovery risk. Selection and certification must be separated.

**Mean masking.** Average improvements can conceal severe regressions. Wins, ties, losses, and task-family stratification should accompany means.

### 19.4 External validity

**Synthetic to real transfer.** A symbolic OOD success on a clean generated law may not transfer to noisy, misspecified, or multivariate scientific data.

**Developer to user transfer.** An interface that works for its builder may not help a new student, analyst, or domain expert.

**Repository to deployment.** UIL is a research system. Claims about production robustness require separate reliability, security, and performance evaluation.

### 19.5 Ethical and human-subject validity

A HITL study should avoid misleading participants about what the system knows. It should distinguish assistance from authority, protect participant data, and measure whether the system creates overconfidence as well as whether it improves correctness. If domain experts are involved, their time and expertise should be respected as research contributions rather than treated as a free oracle.

### 19.6 Mitigation summary

| Threat | Primary mitigation |
|---|---|
| Leakage | Train-only transforms, isolated state, frozen splits. |
| Nondeterminism | Explicit seeds, identity controls, environment capture. |
| Selection bias | Nested or extra evaluation, limited adaptive holdout reuse. |
| Mean masking | Per-task reporting, wins/ties/losses, dispersion. |
| Scope inflation | Claim matrix and evidence labels. |
| Human overclaim | Participant study with objective endpoints and calibrated confidence measures. |
| Synthetic-only evidence | Benchmark expansion to varied equations and realistic noise. |

---

## 20. Human-in-the-loop study design

### 20.1 Purpose

The HITL study is not intended to prove that humans can edit a JSON cell. The system already demonstrates the technical path. The study asks whether editable provenance improves a person’s ability to make a correct, reproducible scientific correction.

### 20.2 Candidate research question

> Does access to a readable, versioned UIL lab record improve a participant’s ability to identify and correct a model or knowledge error compared with a conventional model report?

### 20.3 Candidate participants

A pilot could recruit advanced undergraduate, M.Tech, or early PhD students with basic machine-learning literacy. A later study could include domain experts if tasks are grounded in a real scientific application. Mixing these populations should be avoided unless expertise is explicitly modeled.

### 20.4 Study conditions

| Condition | Materials |
|---|---|
| Baseline report | Final score, standard feature summary, and ordinary experiment metadata. |
| UIL read-only | Baseline materials plus lab report, provenance, and state history, but no editing. |
| UIL editable | UIL read-only materials plus inspect/edit/reject/replay tools. |

The read-only condition is valuable because it separates the effect of extra explanation from the effect of actual editability.

### 20.5 Candidate tasks

1. Identify a formula or rule whose provenance conflicts with a held-out result.
2. Correct a unit-conversion or scaling error and determine whether the recovered law changes.
3. Detect that a principle was mined from insufficient or confounded evidence.
4. Choose between competing routes using the recorded fingerprint and outcomes.
5. Reconstruct the correct historical state after a deliberately injected bad edit.

Tasks should have objectively defensible answers, a known difficulty range, and no requirement that participants reverse-engineer code.

### 20.6 Outcomes

Primary outcomes can be:

- correct final decision;
- time to correct decision;
- ability to cite the evidence supporting the decision;
- ability to reproduce the decision after a delay.

Secondary outcomes can include:

- confidence and calibration;
- perceived workload;
- number and quality of edits;
- reliance on incorrect system suggestions; and
- disagreement between participants.

A system that makes participants faster but more confidently wrong has not succeeded.

### 20.7 Analysis

For a pilot, report individual task outcomes and uncertainty transparently rather than overinterpreting a small p-value. A within-subject crossover can improve power, but task order must be counterbalanced and conditions must use comparable tasks. Predefine exclusion criteria, timing rules, and how partial corrections are scored.

### 20.8 What would count as a useful result

A useful pilot need not show a dramatic gain. It could establish that participants can navigate provenance, make coherent edits, and identify errors without unjustified overconfidence. A negative pilot would also be valuable: it may reveal that the vocabulary, cognitive load, or interface design is the limiting factor rather than the underlying replay mechanism.

---

## 21. Symbolic extrapolation benchmark design

### 21.1 Purpose

The symbolic thread should test a claim aligned with the learner’s inductive bias:

> When data are generated by compact functional structure and evaluation requires extrapolation, can a readable symbolic learner recover a useful law more reliably than a local-fit baseline?

This is different from asking whether it wins every ordinary in-distribution tabular classification problem.

### 21.2 Problem families

The benchmark should include several families:

| Family | Example properties | Why include it |
|---|---|---|
| Elementary analytic laws | polynomial, trigonometric, exponential, rational | Tests basic operator recovery. |
| Multivariate compositional laws | products, nested functions, coupled variables | Tests compositional structure. |
| Piecewise or threshold laws | regimes and discontinuities | Tests limits of smooth symbolic assumptions. |
| Noisy laws | controlled noise and outliers | Tests robustness and complexity control. |
| Distractor features | irrelevant variables and correlated proxies | Tests sparsity and false-discovery behavior. |
| Unit-transformed laws | scaled and shifted variables | Tests invertible normalization and readable back-transformation. |

### 21.3 Split design

For each equation, define a training box and at least one test shell outside it. For example, if training uses x in [−1, 1], evaluate both in-range and on shells such as [−2, −1) and (1, 2]. In multivariate cases, define extrapolation along individual axes and jointly.

Avoid a split that is merely a random train/test partition. That tests interpolation, not extrapolation.

### 21.4 Baselines

At minimum, compare UIL symbolic recovery with:

- a tree ensemble;
- a neural or kernel regressor appropriate to the scale;
- a linear or generalized additive baseline; and
- an established symbolic-regression implementation if the experimental environment supports it.

The point is to locate the tradeoff: fit, OOD behavior, formula size, runtime, and recoverability.

### 21.5 Metrics

Use:

- in-range R² or error;
- OOD R² or error;
- exact structural recovery where the generator is known;
- simplified formula equivalence where algebraic rearrangement exists;
- number of active terms and expression depth;
- runtime and evaluation cost;
- stability across noise levels and seeds; and
- error under feature rescaling.

A model with slightly worse in-range error but much better OOD behavior and a compact correct formula may be scientifically preferable for this task family. Report the tradeoff rather than collapsing everything into one score.

### 21.6 Failure analysis

For every symbolic failure, classify whether it arose from:

- insufficient operator vocabulary;
- constants that could not be fitted;
- search budget;
- noisy or underidentified data;
- incorrect model-selection criterion;
- scaling or numerical instability;
- a simpler rival formula with equivalent fit; or
- an inherently non-extrapolable task.

This turns a benchmark into a research programme rather than a win/loss table.

---

## 22. Architecture evolution and milestone timeline

### 22.1 Generation 1: v19 through v21

The first generation built a broad learning laboratory: staged optimization, journals, mining, hierarchy, graph machinery, causal checks, episodic memory, maturity and detachment concepts, safeguards, and optimizer integration.

Its achievement was breadth and operational infrastructure. Its limitation was that breadth made it easy to confuse component presence with component influence.

### 22.2 Generation 2: v22 “Editable Scientist”

The v22 blueprint reorganized the work around four load-bearing ideas:

1. typed KnowledgeCell objects;
2. lossless, replayable event history;
3. statistically grounded perception with invertible normalization; and
4. a focused scientific target of law discovery, editability, and accountability.

It also wrote an important do-not-do list: do not relitigate the falsified generic-accuracy claim, do not make a neural component the opaque memory substrate, do not leak test information into preprocessing or promotion, and do not let an LLM write state directly.

### 22.3 Milestone interpretation

| Milestone | Intended outcome | Current meaning |
|---|---|---|
| M0: epistemology | Better regret/mining safeguards, tiny-data and multiclass handling. | Engineering milestone; statistical sufficiency still requires task-level evidence. |
| M1: substrate | VersionStore replayable knowledge. | Implemented; replay/edit acceptance path passes. |
| M2: perception | Fingerprint and invertible normalization. | Implemented; clean-data identity verified. |
| M3: routing | Fingerprint-guided strategy selection. | Implemented; held-out routing value remains open. |
| M4: scientist core | Experiments, symbolic recovery, proposal/validation separation. | Research surface largely implemented; active-experiment efficacy open. |
| M5: HITL | Editing, reports, replay, and protocol. | Engineering path exists; real participant study open. |

“Complete” in a milestone log means an architectural acceptance target was met. It does not mean every associated scientific hypothesis has been proved.

### 22.4 Recommended next six months

1. Freeze and run the symbolic extrapolation benchmark.
2. Complete a small but real HITL pilot.
3. Refactor the optimizer without changing behavior.
4. Package a reproducibility artifact for the central negative result.
5. Remove or clearly quarantine any default path that has failed a Gate.
6. Prepare one focused manuscript rather than one omnibus system paper.

---

## 23. Maintenance rules for this monograph

This document should remain the primary faculty reference only if it is updated with the same discipline it advocates.

When a new experiment is run:

1. add the raw result to results/ with a manifest;
2. update the claim matrix before updating rhetorical summary;
3. state whether the new result is evidence, interpretation, or proposal;
4. retain the previous result if the new one conflicts;
5. explain changed protocols or data; and
6. do not change a conclusion merely because a single rerun is favorable.

A project brief becomes trustworthy when it preserves the path by which the project learned, not only the most favorable endpoint.


---

## 24. Major contributions, stated at the correct level

A project of this scale should not list every class or milestone as a separate contribution. The following are the contributions that can be defended at present, separated into engineering, methodological, and scientific levels.

### 24.1 Engineering contribution: an auditable learning substrate

UIL implements a learning environment in which model-facing knowledge is intended to remain inspectable and historically recoverable. The engineering contribution is the integration of:

- typed knowledge objects;
- append-only event history;
- snapshots and reconstruction;
- materialized current views for ordinary system operation;
- human edit and rejection events;
- lab-report generation; and
- research interfaces exposing those artifacts.

The contribution is not merely that the project stores logs. Many systems store logs. It is that the project attempts to preserve the semantic history of knowledge changes: a formula, principle, route, or hypothesis can be tied to a version, cause, and prior state.

The correct evidence level is engineering. Replay and edit tests show that the plumbing exists. A stronger systems contribution would require fault-injection, scale, storage-cost, concurrency, and usability results; those are not yet the primary claims.

### 24.2 Methodological contribution: existence before effect

UIL makes “does this component matter?” a first-class empirical question. This may be the most portable lesson from the project.

A large research architecture can appear active because each module produces data, files, or visualizations. The existence audit asks something harder: if the module is removed, do the relevant outputs change? In the reported short-run conditions, the answer was no for nine modules. That is a useful result because it prevents a false story of integrated cognition.

The method has three advantages:

1. it detects silent no-ops and disconnected wiring;
2. it distinguishes a computation from a causal influence on the learner; and
3. it provides a necessary bridge between unit testing and accuracy evaluation.

The method should become a standard appendix in any future UIL paper. For every claimed mechanism, include a path diagram, existence result, and outcome result.

### 24.3 Methodological contribution: gates as epistemic controls

The project’s default-on/default-off policy is an important engineering-scientific contribution. A feature can be implemented and even locally useful without being globally safe. The Gate 2 record documents this explicitly for formula admission, EBM-style fitting, KAN-style terms, architecture racing, and safe-neutral adaptive settings.

The contribution is a decision process:

    idea
      → implementation
      → deterministic check
      → existence audit
      → cross-seed Gate
      → default, race axis, or default off

This is a better discipline than feature accumulation. It is particularly appropriate in a system whose optional subsystems can interact in complex, non-obvious ways.

### 24.4 Scientific contribution: a negative result with a mechanism

The core negative result is meaningful because the project did more than report a tie. It investigated why:

- strong tabular baselines leave little headroom;
- a journal may contain insufficient varied evidence for meta-learning;
- passive logs are confounded by the search policy;
- architecture selection can overfit validation;
- formulas can be generated without being selected;
- aggressive feature guidance can increase a validation-to-held-out gap; and
- reproducibility defects can make an apparent result unreliable.

This constellation forms a credible research story: a cognitive wrapper does not become useful merely because its components are individually sensible. Its data-generating process, influence path, and evaluation protocol matter.

The contribution should not be exaggerated into a universal claim about cognition. It is a careful boundary result for a concrete architecture and evaluation regime.

### 24.5 Scientific contribution: symbolic structure as a distinct performance axis

UIL’s symbolic direction changes the objective. Instead of maximizing only in-distribution classification accuracy, it asks whether a learner can recover a compact relationship, render it in human terms, and continue it outside the training range.

The reported extrapolation contrast is a compelling motivation for this programme. The research contribution will become stronger when a benchmark establishes where the benefit occurs, where it does not, and what complexity or noise conditions defeat it.

### 24.6 Proposed contribution: human-correctable learning knowledge

The final potential contribution is human correction. UIL is unusually well positioned to study it because its versioned design can retain what a person saw, what they changed, why they changed it, and what happened afterward.

At present this is a proposed scientific contribution, not a completed one. That distinction should be kept in every abstract, slide deck, and thesis chapter.

---

## 25. Detailed subsystem dossiers

The following dossiers explain why the major components exist, how they are intended to work, and what claims their current maturity can carry.

### 25.1 UniversalOptimizer and the staged loop

**Why it exists.** A research system needs a place where learning, search, memory, validation, and persistence meet. The optimizer provides that integration point.

**How it works.** A stage selects or receives a configuration, fits a learner, measures outcome, updates run state, writes journal evidence, and makes selected cognitive hooks available for later stages. The optimizer is also responsible for preserving choices such as learner mode when a search or champion restore occurs.

**Current maturity.** The optimizer is implemented and operational, but its breadth is a maintenance risk. It is a central integration artifact, not a scientific contribution by itself. Its large size makes it a priority for behavior-preserving refactoring.

**Main research risk.** When many mechanisms meet in one method, it becomes difficult to determine which one caused an outcome. This is why matched comparisons and component-level existence audits are indispensable.

### 25.2 Search, PSO, and the detachment aspiration

**Why it exists.** Search explores configurations when the best configuration is unknown. Particle-swarm and related guided procedures offer a way to generate varied candidate actions.

**How it works.** The search layer proposes candidate settings over successive stages. It can optionally incorporate hints from prior knowledge or architecture axes.

**Original aspiration.** The architecture includes a detachment idea: as stored knowledge becomes reliable, it might propose configurations directly and reduce reliance on search. This is the only plausible route by which a cognitive layer could create an efficiency gain rather than merely add overhead.

**Current maturity.** Search is implemented. Detachment as an efficiency result is not established. The v22 blueprint identifies that the old trigger did not fire in a meaningful way and proposes tying maturity to the predictive accuracy of applied knowledge.

**Research implication.** A future efficiency study must measure not only accuracy but number of candidate evaluations, wall time, and the accuracy of knowledge-led proposals against search-led alternatives.

### 25.3 Adapters and model-family pluralism

**Why it exists.** No single learner is appropriate for every task. The adapter layer lets the optimizer compare or route among model families without hard-coding all behavior into one learner.

**How it works.** An adapter exposes fitting, prediction, configuration, capability, and sometimes warm-start interfaces. It allows the routing layer to select a strategy while preserving an evaluator and persistence contract.

**Current maturity.** Implemented. The main scientific value is experimental fairness: a task can be compared under comparable orchestration rather than under unrelated scripts.

**Research risk.** Multiple adapters make it easy to compare incomparable budgets. Every model-family comparison should state resource limits, preprocessing, tuning scope, and selection protocol.

### 25.4 SymbolicFormula and sparse selection

**Why it exists.** A standard black-box model may be accurate but difficult to inspect or modify. SymbolicFormula is intended to expose the final predictor as a small collection of named basis functions.

**How it works.** The adapter creates a candidate design matrix. Raw features, stumps, formulas, and optional rule or nonlinear terms become columns. An L1-regularized logistic readout chooses a sparse subset. In a binary setting, the selected terms and coefficients define a score and probability.

**Current maturity.** This is a mature implementation artifact and the clearest model-level research focus. It supports readout inspection, formula provenance, and optional capacity experiments.

**Research risk.** Sparse selection can be unstable under correlated bases, and a readable expression can still be misleading if the data are insufficient. Stability selection, coefficient variation across seeds, and human comprehension are natural future evaluations.

### 25.5 Formula generation and utility admission

**Why it exists.** Formula generation seeks candidate mathematical structure not available in raw or stump bases. It is especially relevant to law recovery and extrapolation.

**How it works.** A formula engine builds and mutates tree-like symbolic expressions. The library retains candidates and associated metadata. A utility-gated option attempts to admit a new formula only when it improves an evaluation criterion, and can give newborn formulas a temporary selection opportunity.

**Current maturity.** Generation and persistence are implemented. The utility-gated default did not clear its recorded Gate and is correctly off by default.

**Research risk.** Generating many expressions can become artifact inflation. The appropriate measure is not library size but the number of formulas that survive validation, enter a model, remain stable, and improve an external outcome.

### 25.6 Rule basis and interaction capacity

**Why it exists.** Purely additive features and stumps cannot represent interactions such as “feature A is high only when feature B is low.” Readable conjunction rules introduce this capacity while retaining a human-renderable term.

**How it works.** A rule is a conjunction of simple conditions, represented as an indicator basis feature. Sparse selection can keep or discard it with other terms.

**Current maturity.** Implemented and intended to be enabled selectively by routing or architecture choice rather than globally.

**Research risk.** Rules can expand the basis rapidly and overfit, especially under one shared L1 penalty. They should be evaluated with explicit complexity budgets, held-out utility, and a readability criterion rather than only training fit.

### 25.7 EBM-style cyclic fitting

**Why it exists.** A single sparse joint fit can underfit structured nonlinear relationships. A cyclic additive procedure can build shape functions gradually while preserving a readable decomposition.

**How it works.** The implementation applies repeated small updates across basis families or components rather than solving only one joint sparse fit. It is conceptually related to generalized additive modeling.

**Current maturity.** Implemented as an option. The Gate record is modestly positive on average but mixed, so it remains a race axis rather than a default.

**Research opportunity.** This is one of the highest-value model improvements because it targets a capacity limitation without abandoning the readability contract. A focused comparison against standard GAM/EBM implementations would be more informative than another broad cognition ablation.

### 25.8 Neural and KAN-style basis terms

**Why they exist.** Neural or spline-like terms seek nonlinear capacity that ordinary symbolic or additive bases may lack.

**How they work.** Neural units can be introduced as named candidate basis functions under a sparse top-level readout. KAN-style terms are represented as univariate truncated-power functions that remain plottable and potentially prunable.

**Current maturity.** Implemented as experimental options. The KAN gate is negative as a global default.

**Research risk.** Greater capacity does not automatically yield useful interaction structure. If a term is not readable enough for the human interface, it weakens the core claim. If it is readable but cannot represent cross-feature structure, it may simply add variance.

### 25.9 Perception, missing data, and normalization

**Why they exist.** Data preprocessing determines whether later distances, gradients, probe thresholds, and formula constants are meaningful. It is part of the experiment, not a hidden preliminary step.

**How they work.** The perception path handles missing values, may create indicators, normalizes features with a fitted transform, and computes a fingerprint. The intended transform is train-only and invertible so a discovered law can be returned to original units.

**Current maturity.** Implemented. The clean-data identity check reports no performance change, which is the correct result for a path designed not to perturb clean inputs.

**Research opportunity.** Dedicated missingness experiments should vary missing-at-random and informative-missingness regimes, imputation mechanisms, and indicator usefulness. A clean-data identity check cannot answer those questions.

### 25.10 Fingerprinting and routing memory

**Why it exists.** A learner should not be selected solely from feature count or intuition. Fingerprinting provides a measured description; routing memory retains which strategies succeeded in similar regions.

**How it works.** The router combines target type, data size, distributional descriptors, and cheap probe scores. A registry maps the fingerprint to seed domains, while memory can recommend a strategy when support and margin are sufficient.

**Current maturity.** Implemented. The open domain registry is a useful design choice because it allows new task regions to be named rather than forced into a rigid taxonomy.

**Research risk.** A router can itself overfit to a small collection of tasks. Its validation must use held-out task families and must distinguish meta-training tasks from meta-test tasks.

### 25.11 Journal, miner, and principle library

**Why they exist.** These components embody the “learn from learning” hypothesis. The journal retains experiences, the miner proposes patterns, and the library preserves tentative principles that might guide future action.

**How they work.** Stage records are transformed into candidate rules or principles with evidence and provenance. The executor may convert selected principles into limited overrides such as configuration constraints or maximum values.

**Current maturity.** Implemented. Short isolated runs often fail to produce enough evidence, and short-run existence audits found principle removal inert.

**Research implication.** The next credible miner study is cross-task meta-learning: define a source corpus, generate controlled variation, train or pool principle confidence, and test transfer on held-out tasks. It is not a repeated same-dataset run until a rule appears.

### 25.12 Knowledge graph and structural abstraction

**Why it exists.** Graph structure can represent co-activation, relations between knowledge items, and paths through related concepts. Structural encodings can support grouping or later proposal mechanisms.

**How it works.** Nodes and edges store relations; traversal or attention-like mechanisms can produce contextual hints such as feature emphasis or relevant formulas.

**Current maturity.** Implemented, but inert in the reported short audits.

**Research risk.** A graph can become a visually impressive database with no causal effect on decisions. Every edge type should answer: what downstream action can it influence, under what evidence, and how will that influence be audited?

### 25.13 Causal interventions

**Why it exists.** Passive observations from an optimizer are confounded. Intervention is required if the project wants to distinguish “the search often chose this setting” from “this setting caused the outcome.”

**How it works.** The engine creates hypotheses and control/treatment trials. A result can change action only if the intervention path actually runs and is measured.

**Current maturity.** Implemented as a research mechanism. It was inert at the short audit budget because no intervention changed the decisive fingerprint there.

**Research opportunity.** An active-intervention study can compare passive hypothesis resolution with deliberately informative trials. It should report trials-to-resolution, error rate, and cost, not only whether an internal flag changed.

### 25.14 Episodic memory and meta-principles

**Why they exist.** Similar tasks may share useful experience, and related principles may be grouped into higher-level abstractions.

**How they work.** Episodic retrieval uses fingerprint similarity; meta-principles cluster or generalize stored rules.

**Current maturity.** Implemented. Both were inert in the reported short-run audits.

**Research risk.** Similarity in a compact fingerprint may not correspond to transferability of an action. Any success claim must compare retrieval against random, heuristic, and no-memory baselines on held-out tasks.

### 25.15 Safeguards and promotion

**Why they exist.** A system that creates abstractions needs rules governing when they become active. Safeguards prevent premature promotion.

**How they work.** Promotion gates inspect support, confidence, and related criteria before a principle or abstraction becomes influential.

**Current maturity.** Implemented, but inactive in reported short prediction audits.

**Interpretation.** An inactive safeguard may be functioning correctly if no candidate has earned promotion. The audit must therefore inspect both output and whether the expected preconditions occurred.

### 25.16 VersionStore, reports, and interface

**Why they exist.** These components are the foundation of the editability thesis. Without history, a correction cannot be attributed, reversed, or compared to prior belief.

**How they work.** VersionStore writes append-only events and snapshots. Reports compile run evidence into a human-facing narrative. The interface exposes cells, provenance, edits, rejections, and reconstruction.

**Current maturity.** The technical path is implemented and has passing checks. It is ready for usability and human-effect studies.

**Research risk.** The interface can overwhelm a user with terminology and component names. The human study should measure cognitive load and error, not only user satisfaction.

---

## 26. Publication and thesis framing

### 26.1 A coherent thesis title and claim

A credible thesis title would foreground the surviving contribution rather than generic intelligence. For example:

> Transparent and Editable Learning Systems: Symbolic Discovery, Evidence Replay, and the Limits of Cognitive Augmentation

A corresponding thesis claim would be:

> This thesis develops and evaluates an evidence-preserving learning substrate. It shows that a cognitive shell does not reliably improve saturated tabular classification under the evaluated protocol, and investigates symbolic extrapolation and human-correctable knowledge as more appropriate objectives.

This framing is ambitious enough for real research and narrow enough to defend.

### 26.2 Suggested chapter structure for a thesis

1. Introduction and research problem.
2. Related work: tabular learning, symbolic regression, explainability, experiment tracking, adaptive evaluation, HITL.
3. UIL architecture and evidence model.
4. Negative result: cognitive augmentation on tabular classification.
5. Readable symbolic modeling and extrapolation benchmark.
6. Versioned knowledge and human-in-the-loop study.
7. Discussion, limitations, and future work.

This structure gives negative results and engineering infrastructure a legitimate place without allowing the architecture chapter to overwhelm the scientific chapters.

### 26.3 Three paper-sized units

**Paper A — Negative result and instrumentation.**  
Question: why did a rich cognitive shell fail to improve a strong tabular baseline?  
Contribution: existence audits, wiring diagnosis, gate discipline, adaptive-selection analysis.  
Risk: must include enough task diversity and crisp methodology to be more than an anecdote.

**Paper B — Symbolic extrapolation.**  
Question: when does a readable symbolic learner extrapolate better than local-fit models?  
Contribution: controlled benchmark and structural evaluation.  
Risk: must compare against established symbolic baselines and avoid relying on one demo.

**Paper C — Human-editable learning records.**  
Question: can replayable provenance and editable cells improve correction or discovery?  
Contribution: system plus controlled user study.  
Risk: tasks must be realistic enough to matter and objective enough to score.

### 26.4 What not to publish first

Do not begin with an omnibus paper claiming a universal architecture that combines AutoML, causal reasoning, knowledge graphs, symbolic regression, LLMs, HITL, and self-modification. That framing invites a reviewer to demand depth of validation for every subsystem and obscures the project’s strongest result: it learned where its original hypothesis failed.

### 26.5 Oral presentation sequence

A faculty meeting or defense should follow this order:

1. Start with the problem: learning systems leave poor scientific records.
2. State the initial cognitive-accuracy hypothesis.
3. Show the negative Gate result early.
4. Explain the audit methodology that made the result trustworthy.
5. Show the symbolic OOD direction and readable-model design.
6. Show the edit/replay path and distinguish engineering evidence from human evidence.
7. End with the next two decisive experiments.

Leading with the negative result is counterintuitive but credible. It tells a supervisor that the project can be trusted with harder research questions.

---

## 27. Supervisor meeting brief

If this monograph is reviewed in a first supervision meeting, the most productive questions are:

1. Which of the three research threads should become the primary thesis contribution?
2. What task family best represents a meaningful symbolic extrapolation problem in the intended discipline?
3. What participant population is realistic for a pilot HITL study?
4. What evaluation budget and benchmark breadth are expected before a negative tabular result is publication-ready?
5. Should the project pursue a thin editable system or retain a broad laboratory as its primary artifact?
6. Which optional subsystems should be frozen rather than extended until the core studies are complete?

The decisions requested from a supervisor are not implementation preferences. They are scope decisions that determine what evidence the project needs next.


---

## 28. Technical notes on objectives, complexity, and provenance

This appendix collects the mathematical and computational ideas that a faculty reader may want available without interrupting the main narrative. The notation clarifies the implementation intent; it does not imply that every equation has already received an independent theoretical analysis.

### 28.1 Train-only transformation and inverse rendering

Let X denote an input matrix, and let X_train be the training portion of a split. For a standardization transform, the system should estimate:

    μ_j = mean of feature j on X_train
    σ_j = standard deviation of feature j on X_train

and apply:

    x'_j = (x_j − μ_j) / σ_j

to training, validation, and test observations using the same μ_j and σ_j. The test set must not influence these statistics.

If a symbolic model is fitted in normalized coordinates, readable reporting should apply the inverse transform before presenting a formula. For a linear term a x'_j, the original-unit equivalent is:

    a (x_j − μ_j) / σ_j
      = (a / σ_j) x_j − a μ_j / σ_j

For nonlinear formulas, the substitution should be preserved symbolically where possible rather than hidden behind normalized variable names. This is essential if a human is meant to verify units or edit a law.

### 28.2 Sparse named-basis classifier

Given m candidate basis terms φ₁ through φ_m, define:

    Φ(x) = [φ₁(x), φ₂(x), …, φ_m(x)]ᵀ

For a binary label y in {0,1}, an L1-regularized logistic model uses:

    p(y = 1 | x) = 1 / (1 + exp(−(b + wᵀΦ(x))))

and estimates b and w by minimizing:

    L(b,w) =
      − Σₙ [yₙ log pₙ + (1 − yₙ) log(1 − pₙ)]
      + λ Σᵢ |wᵢ|

The L1 term encourages a small active subset of named terms. This is the mathematical basis for calling the model sparse and readable.

Several cautions follow immediately:

- Correlated basis terms can make selected coefficients unstable.
- A nonzero weight is not a causal effect.
- A sparse representation can still be long or semantically opaque if terms are not simplified.
- The value of λ is itself a model-selection decision and must be selected without contaminating final evaluation.
- If the basis includes many candidates generated adaptively, the validation protocol must account for the search process.

### 28.3 Rule and interaction terms

A stump term has the form:

    φ(x) = 1[x_j > τ]

A conjunction rule can have the form:

    φ(x) = 1[x_j > τ_j and x_k ≤ τ_k]

These terms make the prediction surface piecewise constant or piecewise additive in understandable regions. Their attraction is that a reviewer can read a selected rule. Their risk is combinatorial growth: even a modest number of features and thresholds can generate many possible interactions.

A responsible rule-basis experiment should report:

- maximum rule length;
- number of candidate rules;
- number of selected rules;
- validation criterion for admission;
- stability of selected rules across seeds; and
- whether a human can accurately interpret the final rule set.

### 28.4 Utility-gated formula admission

A formula candidate f should not be active merely because it was generated. A generic utility gate can be written as:

    U(f) = Score(validation; basis ∪ {f}) − Score(validation; basis)

Admit f only when:

    U(f) > ε

for a predefined tolerance ε, potentially combined with a complexity penalty. A more conservative form uses a held-out selection split that is distinct from the final evaluation split.

The current implementation’s Gate result shows why this logical improvement is not yet a predictive success: a gate can avoid admitting bad candidates without producing candidates that are good enough to alter final external outcomes. The correct conclusion is not that utility gates are useless; it is that their present use has not justified default activation.

### 28.5 Cyclic additive fitting

An EBM-style or cyclic fitting procedure replaces one global fit with repeated small updates. In abstract form, maintain a score function:

    F^(t)(x) = b + Σᵢ gᵢ^(t)(x)

At each iteration, choose a component or basis family i and make a small update:

    gᵢ^(t+1) = gᵢ^(t) + η hᵢ^(t)

where hᵢ is fitted to the residual signal and η is a small learning rate. The appeal is that complex shapes can be accumulated while preserving a decomposed representation.

The current Gate result is mixed. This is exactly the kind of result for which architecture racing or routing is preferable to a global default. A future study should compare the project implementation with an established generalized additive benchmark under matched preprocessing and resource budgets.

### 28.6 Regret, stress, and control

UIL’s regret signal is not a statistical loss in the usual optimization sense. It is a control concept:

    r_t = max(0, best_proven_accuracy − current_accuracy_t)
          × (1 + 2 × stress_t)

The intended semantics are “how far current performance has fallen below a previously demonstrated level, amplified by accumulated stress.” This can be useful for initiating an investigation, but it should not be treated as an unbiased estimate of generalization loss.

For the control signal to support a scientific claim, a study must show:

1. the signal predicts a useful need for intervention;
2. the intervention selected because of the signal is better than an appropriate control;
3. the result survives independent evaluation; and
4. the cost of monitoring and intervening is justified.

### 28.7 Routing as a decision rule

A router maps a fingerprint z(D) of dataset D to a strategy s:

    s = π(z(D))

The heuristic router currently includes measurable probes, such as linear-probe accuracy or formula-probe R², in addition to basic task properties. A memory-augmented router can estimate strategy value from nearby fingerprints:

    V(s | z) = average score of strategy s among similar stored fingerprints

and make a recommendation only when the best strategy has sufficient support and margin.

This is a valid design pattern, but it is vulnerable to meta-overfitting. The stored tasks used to estimate V must be distinct from the tasks used to evaluate the router. Otherwise, the router’s apparent improvement can simply reproduce known outcomes.

### 28.8 Passive mining and propensity

Suppose the optimizer selects action a under context x according to logging policy π₀(a | x) and observes reward r. A naive miner learns patterns from observed pairs (x,a,r). But some actions may appear good because the policy chose them only in favorable contexts.

A conceptual counterfactual correction would weight an observation by the inverse propensity of the chosen action:

    weighted contribution ∝ r / π₀(a | x)

The project does not claim to implement a complete counterfactual estimator. Its documented logging-policy footnote is a transparency step: it records what variety of actions and configurations produced a mined rule so a reviewer can see whether the evidence is narrow or policy-driven.

The important scientific distinction remains: passive mining suggests hypotheses; interventions test them.

### 28.9 Event-sourced provenance

For a cell c with stable identity, let e₁ through e_T be its ordered events. A versioned state can be written conceptually as:

    state_t(c) = apply(e_t, apply(e_(t−1), … apply(e₁, state_0(c)) …))

Snapshots provide an optimization:

    state_t(c) = replay events after the latest snapshot at or before t

The scientific relevance of this representation is provenance. An event can state not only that a confidence changed, but why: an experiment, a source journal entry, a promotion rule, or a human edit. This enables later distinction between “the system always believed this” and “the system adopted this after a particular piece of evidence.”

### 28.10 Computational cost

No universal asymptotic cost claim is made for UIL because cost depends on learner family, number of stages, candidate count, basis size, event volume, and enabled subsystems. The practical cost drivers are:

| Driver | Cost effect |
|---|---|
| More search candidates or stages | More model fits and validation evaluations. |
| Larger basis library | Larger design matrices and selection problems. |
| Rule interactions | Potentially rapid growth in candidate terms. |
| Architecture race | Multiple competing runs or selection evaluations. |
| Cross-seed Gate | Cost scales with number of seeds and tasks. |
| Event history | Storage and replay work, mitigated by snapshots and compression. |
| Human reports | Usually minor compared with model fitting, but dependent on history size. |

Any future efficiency claim should state the budget explicitly. A system that finds a slightly better score after ten times as many model fits is not necessarily more efficient.

---

## 29. Reporting templates

The templates below are intended to make future results comparable and to prevent a polished narrative from outrunning its evidence.

### 29.1 Gate report template

**Experiment name:**  
**Date and repository revision:**  
**Primary claim:**  
**Control configuration:**  
**Treatment configuration:**  
**Datasets and rationale:**  
**Seeds:**  
**Stages, folds, and selection protocol:**  
**State-directory policy:**  
**Primary metric:**  
**Secondary metrics:**  
**Resource budget:**  
**Predeclared pass/fail rule:**  

| Dataset | Control mean ± spread | Treatment mean ± spread | Difference | Runtime ratio | Notes |
|---|---:|---:|---:|---:|---|

**Aggregate result:**  
**Wins / ties / losses:**  
**Regressions requiring explanation:**  
**Existence-audit result:**  
**Verdict:** default on / race axis / default off / inconclusive  
**Limitations:**  
**Raw artifacts:**  

### 29.2 Symbolic benchmark report template

**Problem family:**  
**Generating law or task description:**  
**Variables and units:**  
**Training domain:**  
**Interpolation test domain:**  
**Extrapolation test domain:**  
**Noise/outlier process:**  
**Candidate operator vocabulary:**  
**Baselines and budgets:**  
**Seed list:**  

| Model | In-range error | OOD error | Structural recovery | Formula/parameter complexity | Runtime |
|---|---:|---:|---|---:|---:|

**Selected formula(s):**  
**Original-unit rendering:**  
**Equivalent-form simplification procedure:**  
**Failure classification:**  
**Interpretation limited to:**  

### 29.3 HITL pilot report template

**Research question:**  
**Participant population:**  
**Ethics/consent status:**  
**Conditions:**  
**Tasks and answer key:**  
**Counterbalancing:**  
**Training or onboarding:**  
**Primary endpoint:**  
**Secondary endpoints:**  
**Stopping/exclusion rules:**  

| Participant | Condition | Task | Correct | Time | Confidence | Evidence cited | Notes |
|---|---|---|---:|---:|---:|---|---|

**Observed result:**  
**Calibration result:**  
**Cognitive-load observations:**  
**Unexpected failure modes:**  
**What the study does not establish:**  

### 29.4 Claim-update template

When any new result is obtained, update the monograph using this compact record:

| Field | Entry |
|---|---|
| Existing claim | |
| New evidence | |
| Protocol change from earlier evidence | |
| Does it replicate, qualify, or contradict earlier evidence? | |
| Revised status | supported / refuted / mixed / open |
| Required change to defaults | |
| Required change to this document | |

This preserves scientific memory. It prevents the current best-looking run from deleting the history that made the project credible.

---

## 30. Final research stance

The central lesson of UIL is not that more architecture makes a learner more intelligent. It is that learning systems need evidence-preserving design if their own adaptations are to be believed.

UIL has already done the difficult and valuable part: it built a large architecture, tested its central accuracy promise, found a negative result, and retained that result in the project record. The right response is not to make the story smaller by hiding the failure. It is to make the research sharper.

The sharp version of UIL is:

- a readable model family for structure-sensitive tasks;
- a transparent event and provenance substrate;
- disciplined experiments that distinguish influence from benefit;
- a human-editable interface whose value is tested rather than assumed; and
- a clear refusal to claim more than the data support.

That is a credible foundation for graduate research. It is also a more enduring contribution than a one-off benchmark win.



