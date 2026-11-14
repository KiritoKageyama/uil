# UIL — Project Overview and Results Report

**Universal Intelligence Layer** (`figpso_uil`)  
**Date:** 26 July 2026  
**Audience:** researchers, reviewers, and technical collaborators  
**Purpose:** Explain what UIL is, how its parts are structured, what role each part was designed to play, how accuracy is produced, what experiments show, and what impact the work has delivered.

---

## 1. Executive summary

UIL is a research system for tabular machine learning. It combines ordinary learners (and especially a **readable symbolic classifier**) with an optional **cognitive layer**: a journal of training decisions, mined principles, a knowledge graph, causal checks, and a versioned record that humans can inspect and edit.

**How accuracy is mainly produced today.**  
Most predictive performance comes from the **learner**, **search**, and **data preparation** — above all SymbolicFormula in `sparse_linear` mode — not from “the system thinking about itself.”

**What experiments show about cognition.**  
Turning the cognitive shell on aggressively does **not** reliably improve held-out accuracy on ordinary tabular problems. In a cross-seed Gate 2 study, default core wiring scored a mean accuracy change of **−0.0132** versus plain search. Some modules were **inert** on short default runs (they ran and wrote files, but did not change predictions). After later wiring fixes, many modules **can** change predictions; that does not mean they help. Connected guidance can still **hurt** held-out scores.

**Where the project is strong.**  
Readable models at roughly **XGBoost parity** on several datasets; large gains from learner and scale fixes (spambase, digits, MNIST, adult); symbolic out-of-range recovery; inspectable history and edit tooling; careful ablation and gating.

**Where it remains weak.**  
High compute for little or negative cognitive accuracy gain; some parts still rarely reach the fitted model; human benefit of editing knowledge is not yet studied with real users.

**Impact in one line.**  
UIL delivers a working, inspectable learning stack and an honest negative result on cognitive accuracy — see Section 7.

---

## 2. System structure at a glance

```
Dataset (X, y)
    │
    ▼
Perception          impute missing values → normalize features → fingerprint the data
    │
    ▼
Domain router       choose a learning strategy / mode (optional; UI can lock the classifier)
    │
    ▼
UniversalOptimizer  multi-stage loop (the central controller)
    ├── Search            propose configurations (weights, penalty, optional architecture)
    ├── Classifier        train and score (SymbolicFormula, XGBoost, …)
    ├── Cognition         journal → mine → principles / graph / formulas / causal trials
    └── VersionStore      append events, snapshots, lab reports, human edits
    │
    ▼
Predict             apply the same perception transform used in training
```

Two design generations share one codebase:

| Generation | Focus |
|------------|--------|
| **v19–v21** | Full cognitive loop (search → journal → miner → library → executor) plus supporting extensions |
| **v22** | Editable Scientist: fingerprints, routing, event history, symbolic scientist tools, human-in-the-loop path |

---

## 3. Components: structure, designed role, and role today

This section is the heart of the document. For each major part it states:

- **Structure** — what it is made of  
- **Designed role** — what it was meant to do in the system  
- **Role today** — how it actually behaves in current defaults and tests  

### 3.1 UniversalOptimizer (`core/optimizer.py`)

**Structure.** The central orchestrator. It owns the stage loop inside `fit()`: prepare data, call search, evaluate actions with a classifier, record outcomes, invoke cognition, update champions, and return the best model.

**Designed role.** Be the single place where learning, search, and cognition meet — a controlled multi-stage experiment rather than a one-shot fit.

**Role today.** Still the hub of every run. It is large and complex, but it is what makes perception, routing, search, classifiers, and optional cognition operate as one system.

---

### 3.2 Perception — missing data and normalization

**Structure.**  
- `MissingDataSpec`: train-only median/mode fill and missingness indicators.  
- `NormalizationSpec`: invertible per-feature transforms with stored provenance.

**Designed role.** Make input preparation explicit, consistent between train and predict, and reversible so symbolic laws can be read in original units.

**Role today.** Active and important for correctness. Prediction applies the same forward path used in training. On clean data the missing-data path is a no-op (identity), as designed.

---

### 3.3 Fingerprint and domain router

**Structure.**  
- Fingerprint: five statistical axes (center, spread, shape, dependence, sequence), plus optional probes.  
- Router: maps fingerprint (and memory) to a domain/strategy, then to a classifier configuration.

**Designed role.** Avoid “one learner for every dataset.” Profile the task cheaply and route it to a suitable strategy (for example sparse readable models on structured data).

**Role today.** Implemented and used at the start of `fit()`. It can select modes such as `sparse_linear`. A broad held-out study of whether routing beats a fixed learner across many task families is still open. The web UI can disable silent swaps so a user’s chosen classifier is respected.

---

### 3.4 Classifier adapters

**Structure.** A common adapter interface (`fit_weighted`, `predict`, `apply_config`, …) with implementations for SymbolicFormula, XGBoost, Random Forest, k-NN, MLP, and TabPFN (optional).

**Designed role.** Let the optimizer treat different learners uniformly, while each adapter owns its own training details.

**Role today.** This is the **main accuracy surface**. SymbolicFormula is the flagship readable learner; tree boosters remain strong baselines and routing targets.

---

### 3.5 SymbolicFormula (`adapters/symbolic_formula_adapter.py`)

**Structure.** Builds a named **basis** (raw features, stumps, library formulas, optional rules, neural units, KAN-style terms), then fits a sparse predictor — typically L1 logistic regression (`sparse_linear`), with optional boosting, bagging, hybrid, multiclass one-vs-rest, or EBM-style cyclic fitting.

**Designed role.** Deliver competitive accuracy in a form that stays **readable and editable**: a weighted sum of named terms, not a black box.

**Role today.** The primary path for honest accuracy claims. Documented wins include spambase rising from about **0.685 to ~0.949**, digits recovering from collapse (**~0.214 to ~0.947**), and MNIST-784 becoming usable after feature preselection. Extra bases (rules, neural, KAN) and EBM fitting remain **optional** because cross-seed gates did not justify turning them on everywhere.

---

### 3.6 Search — PSO and principle-guided search

**Structure.** Particle-style or guided search proposes actions: feature weights, penalties, and limited hyperparameters. Principle-guided search can bias proposals using cognitive overrides.

**Designed role.** Explore configurations across stages and improve the champion without hand-tuning every knob.

**Role today.** The default optimization engine for classic runs. It works. Alone, it mostly varies penalty and weights; it does not invent a full architecture unless race mode or exploration probes are enabled.

---

### 3.7 Architecture race (`search/architecture_race.py`)

**Structure.** Instead of a large PSO swarm over masks, it evaluates a frontier of feature cutoffs and races one architecture axis at a time (for example rule basis on/off, L1 strength), with successive halving and optional multi-split final selection.

**Designed role.** Search architecture honestly and efficiently, and let cognition ride on an incumbent arm that is not eliminated early.

**Role today.** Optional (`search_mode="race"`). It can improve some datasets (for example sonar, digits) and **seriously regress** others (ionosphere about **−0.047**). It stays opt-in.

---

### 3.8 Journal and stress / regret

**Structure.**  
- Journal: stage records of state, action, outcome, stress, and regret.  
- Stress engine: control signals from performance shortfall and system stress.

**Designed role.** Create the evidence trail from which principles are mined, and signal when the system should investigate or evolve.

**Role today.** Always central to logging and audits. Regret can be grounded with extra validation folds. Stress once used an unreachable threshold that kept evolution “asleep”; that gate is now relative to the run, so evolution can fire — but new formulas still often fail to enter the final sparse model.

---

### 3.9 Miner, principle library, and executor

**Structure.**  
- Miner: decision-tree mining over journal matrices (including config columns), with held-out R² checks.  
- Library: stores principles with confidence, tiers, and overrides.  
- Executor: matches principles to the current state and emits overrides.

**Designed role.** Learn reusable rules such as “in situation S, avoid action A,” then apply them on later stages or datasets — the core meta-learning idea.

**Role today.** Fully built. On short isolated runs it is often **inert** (too little journal variation or evidence). With architecture exploration and a lower evidence bar it can mint real architecture rules — but that combination failed an accuracy Gate (mean about **−0.0079**). Aggressive connection into the learner also failed Gate 2 (mean **−0.0132**). So the designed role is clear; the default accuracy payoff is not.

---

### 3.10 Editable knowledge graph

**Structure.** Nodes and typed edges (supports, contradicts, co-activation, and related relations), attention-based traversal, decay, prune, and archive.

**Designed role.** Hold relational context between principles and situations, and supply soft overrides or feature emphasis during a run.

**Role today.** Active at scale: an unbounded graph caused a long hang on adult data and noisy overrides; pruning raised adult held-out accuracy from **0.771 to 0.815**. On short default audits it is often inert. Caps and archives keep it usable.

---

### 3.11 Formula engine and formula library

**Structure.** Symbolic expression trees, mutation operators, tournament libraries for loss formulas and classifier formulas, with prune-to-archive.

**Designed role.** Invent new symbolic structure that the learner can use — especially inside SymbolicFormula’s basis.

**Role today.** Evolution can run after the stress-gate fix. New formulas are frequently born but **not selected** into sparse_linear fits, so accuracy often stays unchanged even when the library grows. Utility-gated admission was tried and did not pass Gate 2 as a default.

---

### 3.12 Causal intervention engine

**Structure.** Hypotheses, scheduled control/treatment trials on controllable factors, majority-of-three confirmation before strong causal status.

**Designed role.** Move beyond correlational mining: actively test whether changing an action changes outcomes.

**Role today.** Can be **connected** when it forces alternate evaluated actions. On short budgets it may not fire. It is a scientific instrument more than a default accuracy booster.

---

### 3.13 Episodic memory, structural encoding, meta-principles, safeguards, goals

**Structure.** Separate modules for case retrieval, structural principle encoding, clustering into meta-principles, promotion guards, and optional goal cells.

**Designed role.** Support transfer, abstraction control, and optional goal bias — the “full cognitive architecture.”

**Role today.** Implemented and ablatable. On short identity audits they are often inert. Their practical role is mostly **support and research completeness**, not day-to-day accuracy.

---

### 3.14 Signal translator and cognition connection

**Structure.** Translates cognitive signals into learner directives (feature guidance, preferred formulas, config hints). `connect_cognition` can fold feature weights into the sparse L1 penalty.

**Designed role.** Close the loop so cognition actually reaches the model (fixing an earlier dead channel).

**Role today.** The channel is **live**. Hard nonuniform guidance often **overfits** the internal validation split and hurts held-out accuracy. Softer settings are mixed by dataset. Defaults stay conservative.

---

### 3.15 VersionStore, lab reports, and human-in-the-loop

**Structure.** Append-only events, periodic snapshots, reconstruct/replay, lab reports, and edit/reject operations that update both the event log and materialized files.

**Designed role.** Make knowledge inspectable, attributable, correctable, and replayable — the “Editable Scientist” product thesis.

**Role today.** Engineering path works (acceptance and HITL path tests pass). Snapshot compaction reduced large snapshot stores from tens of gigabytes to about **1.9 GB** without deleting events. A real human study of whether edits help users is still ahead.

---

### 3.16 Champions, archives, UI, and safety switches

**Structure.** Per-dataset champion store; knowledge archive for pruned items; web and desktop research consoles; `uil_enabled=False` master switch; guard telemetry.

**Designed role.** Persist progress, keep nothing silently lost, expose the lab to users, and allow a plain baseline for fair comparison.

**Role today.** All in use. Champion logic was corrected so an old record cannot block the current run from saving its own work. The UI streams stages and exposes knowledge for inspection.

---

## 4. How accuracy is produced

Accuracy follows this chain:

1. Prepare **X** and **y** with train-consistent imputation and normalization.  
2. Optionally fingerprint and route to a strategy.  
3. Across stages, **search** proposes configurations.  
4. The **classifier** trains and is scored on validation (and optional extra folds).  
5. The best model is kept; at the end it may be compared with a stored champion.  
6. **Predict** applies the same perception transform used in training.

For SymbolicFormula `sparse_linear`:

1. Build named basis terms Φ(x).  
2. Fit a sparse linear model (usually L1).  
3. Predict with a sigmoid of the weighted sum.  
4. For multiclass, use calibrated one-vs-rest heads, then choose the best class.

Cognition may change penalties, config bounds, or feature weights **when enabled**. That path is optional because experiments show it is easy to change the model without improving held-out accuracy.

**Practical guidance.** For best tabular accuracy now, use a strong mode (`sparse_linear` or a routed tree learner), solid perception, and classic search. Treat full cognition as an experiment, not a free upgrade.

---

## 5. Experimental results

Prefer cross-seed means and held-out evaluation. Re-run `eval_crossseed` before citing numbers in a paper.

### 5.1 Does cognitive wiring help accuracy?

Stages 12 · seeds 42 / 7 / 123 · `results/try_uil_connect/SUMMARY.md`

| Dataset | Off | Default wiring | Change | Connect pack | Change |
|---------|----:|---------------:|-------:|-------------:|-------:|
| breast_cancer | 0.9708±0.015 | 0.9591±0.015 | **−0.0117** | 0.9591±0.015 | −0.0117 |
| wine | 0.9537±0.013 | 0.9259±0.047 | **−0.0278** | 0.9259±0.047 | −0.0278 |
| ionosphere | 0.9202±0.037 | 0.9202±0.040 | 0.0000 | 0.9202±0.040 | 0.0000 |

Mean change (default − off): **−0.0132** (no wins, one tie, two losses).  
Diagnosis: hard connection of nonuniform guidance into L1 overfits the selection split.

### 5.2 Parity with XGBoost (readable UIL arms)

From `results/soft_wiring_sota/NOTES.md`:

| Dataset | Best UIL arm in that study | XGBoost | Change |
|---------|---------------------------:|--------:|-------:|
| breast_cancer | off 0.9708 | 0.9649 | **+0.006** |
| wine | soft / no-weights 0.9630 | 0.9722 | −0.009 |
| ionosphere | off 0.9202 | 0.8967 | **+0.024** |

Stronger black-box methods (HistGBM, SVM-RBF, MLP) still win raw accuracy on some sets and finish much faster. UIL’s distinctive claim is **readable and editable** models near XGBoost parity, not victory over every baseline.

### 5.3 Large and stressful tests

| Stress | What happened | Outcome |
|--------|---------------|---------|
| UCI adult, huge graph | Hang (~142 minutes); noisy overrides | Prune + archive; held-out **0.771 → 0.815** |
| Covertype, large *n* | Mid-run crash; CV too costly | Safer formula fetch; CV auto-scale |
| MNIST-784 | Hours of work; majority collapse | Preselect features → **~0.2 s**, acc **~0.934** |
| Digits / MNIST OvR | Few classes predicted (**0.214**) | Calibration + sparse OvR → digits **~0.947** |
| Spambase | Mode overwritten → **0.685** | Mode lock → **~0.949** |
| Imbalanced data | Always-majority thresholds | Balanced-accuracy threshold when needed |
| Train/serve bug | Train normalized, predict raw | Same forward transform everywhere |

### 5.4 Ablations and existence audits

An **existence audit** asks: if this module is removed, do predictions change at all?

| Setting | Typical finding |
|---------|-----------------|
| Short default runs (breast / wine) | Many cognitive modules **inert** (state written, predictions unchanged) |
| Principles wiped (HITL check) | Identical predictions, accuracy change **0** |
| Causal interventions when scheduled | Sometimes **connected** |
| Hard `connect_cognition` | **Connected**, often **harmful** |
| Soft wiring | Mixed by dataset |
| Evolution after stress-gate fix | Library grows; accuracy often unchanged |

Historical full-versus-ablated cognition on saturated tabular data: accuracy change about **zero** at much higher compute.

### 5.5 Gate 2 board (should a feature become default?)

| Experiment | Mean accuracy change | Default decision |
|------------|---------------------:|------------------|
| Utility-gated formulas | −0.0013 | Off |
| EBM cyclic fit | +0.0014 | Off (race option) |
| KAN basis | −0.0022 | Off |
| Race with margin 0.01 | +0.0010 (ionosphere −0.047) | Off |
| Category D soft maturity / stress blend | +0.0001 | **On** (safe) |
| Missing-data path on clean data | 0.0000 | **On** |
| Explore architecture + adaptive evidence | −0.0079 | Adaptive evidence off |

### 5.6 Reference classic accuracies (SymbolicFormula band)

| Dataset | Representative classic accuracy |
|---------|--------------------------------:|
| wine | 0.9537 |
| sonar | ~0.70–0.77 (depends on mode) |
| ionosphere | 0.9155 |
| breast_cancer | 0.9708 |
| digits | 0.9500 |
| spambase | 0.9439 |

Symbolic out-of-range example (reported): R² **0.992** for symbolic recovery versus **−5.52** for XGBoost on that extrapolation task.

---

## 6. Component scorecard (quick view)

| Component | Designed for | Status in the current system |
|-----------|--------------|------------------------------|
| SymbolicFormula | Accuracy + readability | **Primary working path** |
| Perception | Correct train/serve prep | **Working** |
| Router / fingerprint | Strategy selection | **Working; broad meta-proof open** |
| Classic search | Configuration search | **Working** |
| Architecture race | Architecture search | **Optional; mixed results** |
| Journal / stress | Evidence and control | **Working** |
| Miner / library / executor | Meta-rules → overrides | **Built; often unused at short budget; risky if hard-wired** |
| Knowledge graph | Relational context | **Working with caps; matters at scale** |
| Formula evolution | New symbolic terms | **Partial (births ≠ selection)** |
| Causal engine | Active tests | **Partial; budget-dependent** |
| Translator / connect | Cognition → model | **Live channel; hard mode unsafe as default** |
| VersionStore / HITL / UI | Transparency and editability | **Software complete; human study open** |
| Episodic / meta / safeguards / goal | Transfer and control | **Support roles; often inert in short audits** |

---

## 7. Impact

Impact means measured or delivered outcomes.

### 7.1 Accuracy impact (learner and correctness)

| Outcome | Evidence | Why it matters |
|---------|----------|----------------|
| Readable model competes with XGBoost | Spambase ~**0.685 → 0.949** | Glass-box path is viable |
| Multiclass becomes valid | Digits ~**0.214 → 0.947** | Collapsed heads were invalid science |
| High-dim becomes practical | MNIST-784 hours → **~0.2 s**, ~**0.934** | Scale inside the same framework |
| Graph hygiene helps | Adult **0.771 → 0.815** | Unbounded memory can hurt |
| Train/serve consistency | Shared forward transform | Held-out scores become trustworthy |
| Optional interaction capacity | Sonar rules uplift in isolation | Capacity where needed, not everywhere |

**Net:** strong positive impact on **inspectable accuracy** when the learner path is used well.

### 7.2 Cognitive-shell impact (honest)

| Outcome | Evidence | Interpretation |
|---------|----------|----------------|
| Hard wiring hurts | Mean **−0.0132** | Do not default it on |
| Soft wiring mixed | Wine up, breast down | Not a universal fix |
| Race mixed | Wins and sharp losses | Opt-in only |
| Short-run inert modules | Existence audits | Built ≠ load-bearing |
| Ablation ≈ tie | Historical Δ ≈ 0 | Meta-layer ≠ free accuracy |

**Net:** high value as an **instrument and negative result**; low value as a default accuracy booster.

### 7.3 Scale, science, and transparency impact

- Brutal tests forced real engineering (adult hang, covertype, MNIST, multiclass).  
- Existence audits and Gate 2 created a durable evaluation culture.  
- Failures were mapped to published literature, not only local anecdotes.  
- VersionStore, lab reports, and HITL tooling make learning knowledge auditable.  
- Human benefit of editing remains an **open** impact claim.

### 7.4 Impact by purpose

| Purpose | Impact level |
|---------|--------------|
| Learner accuracy and readability | **High** |
| Search / race / routing | **Medium / mixed** |
| Support, scale, correctness | **High** |
| Editability and transparency (software) | **High** |
| Editability (proven human benefit) | **Open** |
| Cognitive meta-accuracy as default | **Low / negative** |

### 7.5 What not to claim

- That UIL beats every black-box tabular method on every dataset.  
- That enabling all cognition improves accuracy by default.  
- That every module matters on every run.  
- That human editing is already proven to help users.

### 7.6 Impact paragraph

UIL’s main positive impact is a **working, inspectable learning stack** that can reach **XGBoost-like accuracy with readable models**, survive **large and multiclass stresses**, and keep a **versioned research record**. Its main scientific impact is showing, with careful tests, that a rich cognitive shell **does not automatically improve** ordinary tabular accuracy and can hurt when hard-wired. The impact still to earn is **human-in-the-loop value** and broader symbolic out-of-distribution benchmarks.

---

## 8. Current direction and next steps

**Current direction**

1. Keep defaults conservative.  
2. Treat SymbolicFormula, perception, and routing as the accuracy product.  
3. Keep cognition optional and instrumented.  
4. Complete a real human HITL pilot.  
5. Write papers that match the evidence (readable methods, symbolic OOD, negative/instrumentation result).  
6. Refactor the large `fit()` method without changing behavior.

**Priority next work**

| Priority | Work |
|----------|------|
| 1 | Human HITL study |
| 2 | Frozen symbolic / OOD benchmark suite |
| 3 | Readable-model benchmark versus EBM / RuleFit / XGBoost |
| 4 | Safer cognition only if a new Gate 2 passes |
| 5 | Held-out routing meta-test |
| 6 | Decide: thin product stack versus full research instrument |

---

## 9. How to reproduce

```bash
# From the parent directory of figpso_uil/

python figpso_uil/tests/test_v21_smoke.py
python figpso_uil/tests/test_m0_epistemology.py   # through test_m5_hitl.py

python -m figpso_uil.experiments.eval_crossseed
python -m figpso_uil.experiments.existence_audit
python -m figpso_uil.experiments.ablation_harness

python -m uvicorn figpso_uil.webui.backend.main:app --port 8000
```

Use an isolated `state_dir` for clean A/B tests. Gate summaries live under `results/gate2_*/`, `results/try_uil_connect/`, `results/soft_wiring_sota/`, and `results/component_hitl/`.

---

## 10. Closing takeaway

UIL is a complete research platform: learners, search, cognitive memory, and editable history.  

**Accuracy** comes mainly from the readable (or routed) learner and from correctness at scale.  

**Cognition** was designed to improve later decisions from experience; in current defaults it is often unused, sometimes connected, and unsafe as a hard global accuracy switch.  

**Impact** is strongest in readable accuracy, scale survival, transparent records, and honest negative results — not in a claim that “more cognition beats XGBoost.”  

**Next** is human evidence for editability, plus papers that stay inside what the measurements support.

---

*Figures are taken from in-repository Gate and ablation summaries and from documented stress fixes. Regenerate cross-seed evaluation before formal publication.*
