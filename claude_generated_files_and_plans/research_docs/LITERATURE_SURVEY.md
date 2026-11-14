# Literature Survey — Universal Intelligence Layer (UIL)

**Audience:** researchers, advisors, and collaborators reading `PROFESSOR_PROJECT_BRIEF.md`  
**Scope:** place UIL in the published literature across interpretable tabular learning, symbolic regression, AutoML / meta-learning, search & selection, causal intervention, cognitive rule systems, and human-in-the-loop editability  
**Companion:** measured failure→literature map in `RESEARCH_FAILURE_ANALYSIS.md`  
**Paper catalog (table form):** `LITERATURE_SURVEY_PAPERS.md`  
**Date:** 31 July 2026  

---

## 0. How to read this survey

UIL is not one algorithm. It is a **stack** with three claim axes that the literature treats differently:

| Claim axis | What UIL means by it | Literature verdict (field consensus) | UIL empirical status |
|------------|----------------------|--------------------------------------|----------------------|
| **A. Readable tabular accuracy** | Sparse linear / RuleFit-like model near GBDT | Achievable (EBM, RuleFit, GA²M) | **Strong** — primary positive result |
| **B. Symbolic / OOD science** | Recover closed-form laws; extrapolate | Strong for physics-like targets; weak for arbitrary tabular | **Promising / under-benchmarked** |
| **C. Cognitive meta-layer** | Journal → mine → graph → causal → edit | Sample-hungry; often inert or harmful if miswired | **Honest negative / instrument** |

This survey is organized by research tradition, then closed with a **positioning map** (what UIL inherits, what it adds, what it should *not* claim) and a **citation shortlist** for papers.

For a flat catalog of papers with fixed columns (title, authors, venue, description, impact, uniqueness, UIL ground), see **`LITERATURE_SURVEY_PAPERS.md`**.

---

## 1. Tabular machine learning — the accuracy ceiling

### 1.1 Tree ensembles remain the default SOTA

**XGBoost** (Chen & Guestrin, KDD 2016) and gradient-boosted trees more broadly set the practical accuracy bar on medium tabular data. Deep tabular learning repeatedly tried to displace them; large systematic evidence says the residual headroom is small.

**Key paper:** Grinsztajn, Oyallon & Varoquaux (NeurIPS 2022 Datasets & Benchmarks), *Why do tree-based models still outperform deep learning on tabular data?*  
Benchmark of 45 datasets; tuned trees beat NNs on medium tabular (~10k samples). Inductive biases: trees handle irregular targets, uninformative features, and non–rotation-invariant feature semantics better than typical NNs.

**Implication for UIL:** competing on raw accuracy against tuned GBDT with *extra cognitive machinery* was always a weak pitch. UIL’s measured cognitive Δ ≈ 0 (or slightly negative when hard-wired) is **literature-consistent**, not a local failure mode. The defendable accuracy product is *parity with a strong baseline under a readability constraint*, not beating every black box.

### 1.2 Foundation models for small tabular data

**TabPFN / TabPFN v2** (Hollmann et al.; Nature 2025 for v2): prior-data fitted networks that act as Bayesian inference over a synthetic prior of tabular tasks. Strong on small-to-medium *n* with little tuning; **not** glass-box.

**Implication for UIL:** correct role is **ceiling probe / small-*n* router strategy** (already implemented as optional fingerprint probe + `D3_tiny_tabpfn`), never the default “Editable Scientist” learner. Matches the brief’s honesty: accuracy ceilings and readability claims must stay separate.

### 1.3 Classical baselines UIL already wraps

- Random forests (Breiman 2001)
- *k*-NN
- Logistic / L1 logistic (Tibshirani Lasso 1996; Friedman–Hastie–Tibshirani elastic net lineage)
- MLP

These matter as **adapters and ablations**, not as the research thesis.

---

## 2. Interpretable (“glass-box”) tabular models — UIL’s strongest home

This is the tradition where UIL’s SymbolicFormula `sparse_linear` path belongs.

### 2.1 Generalized additive models and intelligibility

- **Hastie & Tibshirani (1986/1990)** — GAMs: \(g(\mathbb{E}[y]) = \beta_0 + \sum_j f_j(x_j)\). Additivity buys intelligibility (per-feature shape plots) at the cost of interaction capacity.
- **Lou, Caruana, Gehrke et al.** — *Accurate intelligible models with pairwise interactions* (KDD 2013, GA²M): add selected pairwise terms \(f_{ij}(x_i,x_j)\) while keeping modular explanations.
- **Caruana et al. (KDD 2015)** — *Intelligible models for healthcare*: editable GAMs in high-stakes settings; humans can inspect and fix harmful patterns (e.g., pneumonia risk). This is the closest classical precedent for UIL’s **HITL edit** thesis.
- **Nori, Jenkins, Koch & Caruana (2019)** — *InterpretML*; **Explainable Boosting Machine (EBM)**: cyclic gradient boosting of shallow trees per feature (and optional pairs), accuracy often near RF/XGBoost while remaining glass-box.

| EBM | UIL `sparse_linear` |
|-----|---------------------|
| Shape function per feature | Raw + stump basis terms |
| Pairwise interactions (GA²M) | Rule basis (RuleFit-style conjunctions) |
| Cyclic boosting fit | Joint L1 (optional EBM-style cyclic fit, Gate 2: small +; default OFF) |
| Editable shape plots | Named editable cells + VersionStore |

**Takeaway:** “Readable at GBDT-ish parity” is a **published, repeated** result in this line. UIL should cite EBM/GA²M as existence proof and position SymbolicFormula as a **composed symbolic basis + L1 readout** variant with stronger provenance / event history, not as inventing glass-box tabular learning.

### 2.2 Rule ensembles and RuleFit

- **Friedman & Popescu (2008)**, *Predictive learning via rule ensembles* (RuleFit): extract rules from tree ensembles; fit sparse linear model over **original features + rules** with L1. Standard reference for “interaction as readable conjunction + sparse linear.”
- **Fokkema & colleagues** — R package `pre` (JSS): modern PRE implementations; accuracy competitive with forests at higher sparsity.

**UIL connection (E4/E5):**

- Raw + stump + formula ≈ additive RuleFit family.
- Rule basis ≈ RuleFit’s interaction terms.
- Measured E5 lesson (rules starve under strong L1 until `sparse_l1_c` is relaxed) is exactly the group-sparsity / budget problem RuleFit practitioners know: many correlated binary rules compete with a few strong linear terms under one global penalty.

### 2.3 Other glass-box / rule-list lines (for related work)

- **Node Harvest** (Meinshausen 2010) — soft weighting of tree nodes.
- **SLIPPER / rule lists / Falling Rule Lists** — compact logical models; often trade accuracy harder.
- **CORELS / Certifiably Optimal RulE ListS** (Angelino et al.) — optimality under interpretability constraints.
- **Molnar, *Interpretable Machine Learning*** (book) — survey framing RuleFit, PDP, LIME, SHAP.

UIL is closer to **RuleFit + EBM** than to pure rule lists: the final object is still a weighted sum of named terms.

### 2.4 Post-hoc XAI vs inherent interpretability

**LIME** (Ribeiro et al. 2016), **SHAP** (Lundberg & Lee 2017) explain *black boxes*. UIL’s product thesis is **inherent** interpretability (glass-box model + editable knowledge), in the Caruana/InterpretML tradition — not “XGBoost + SHAP.” Related-work sections should state this contrast explicitly.

---

## 3. Symbolic regression and scientific discovery

UIL’s “scientist” milestone (M4) and OOD symbolic wins sit here.

### 3.1 Classical symbolic regression

- **Koza** — genetic programming for symbolic expressions.
- **Schmidt & Lipson (Science 2009)** — data-driven discovery of natural laws (Eureqa lineage).
- **Udrescu & Tegmark (Science Advances 2020)** — *AI Feynman*: physics-inspired recursive symbolic regression; solves Feynman lecture equations at high rates.
- **Udrescu et al. (NeurIPS 2020)** — *AI Feynman 2.0*: Pareto complexity–accuracy, modularity from NN gradients.
- **Cranmer et al. / PySR** — modern, practical symbolic regression stacks used widely in scientific ML.

### 3.2 Program synthesis libraries that stay useful

**DreamCoder** (Ellis et al., PLDI 2021 / Phil. Trans. R. Soc.): wake–sleep; abstractions enter the library only if they **pay for themselves** (MDL / compression). Direct analogue of UIL’s attempted **utility-gated formula admission** (implemented; Gate 2 failed as default — still the right *research* framing).

### 3.3 Kolmogorov–Arnold Networks

**Liu et al. (2024+)** — KANs: learnable univariate edge functions, sparsifiable, often marketed for symbolic extraction. UIL’s `use_kan_basis` is an opt-in truncated-power spline family inspired by this line; Gate 2 did **not** support default ON.

### 3.4 Positioning UIL’s symbolic claim

| Setting | Literature expectation | UIL |
|---------|------------------------|-----|
| Physics-like / compositional laws | Symbolic methods dominate GBDT OOD | Reported R² ~0.99 vs XGBoost collapse — **aligns** |
| Saturated classification tables | Trees / EBM / RuleFit; formulas rarely win accuracy | Cognitive formulas often **inert** in `sparse_linear` selection — **aligns** |
| “Discover new science from UCI adult” | Unrealistic | Do not claim |

**Paper angle:** frozen symbolic / OOD suite (brief §8 priority 2), not “formulas improve breast_cancer accuracy.”

---

## 4. AutoML, meta-learning, and dataset-adaptive routing

### 4.1 Combined algorithm + hyperparameter search

- **Auto-WEKA** (Thornton et al. 2013), **hyperopt-sklearn**, **TPOT** (Olson et al.), **Auto-sklearn** (Feurer et al., NeurIPS 2015): CASH — Combined Algorithm Selection and Hyperparameter optimization — with Bayesian optimization / evolutionary search.
- **Auto-sklearn 2.0** (Feurer et al., JMLR 2022): meta-learned portfolios, successive-halving budgets, policy selection over AutoML itself.

**Critical sample-complexity point for UIL:** published meta-learning works from **tens to hundreds of datasets** (OpenML-scale), not ~16 journal rows on one dataset. UIL’s failure to mint principles under isolated `state_dir` short runs is predicted by this literature (see also `RESEARCH_FAILURE_ANALYSIS.md` F3).

### 4.2 Meta-features and routing

OpenML / AutoML meta-features (simple statistics, landmarkers, information-theoretic features) → warm-start or portfolio selection.

**UIL fingerprint + domain router (v22 M2/M3)** is the same *family of idea* (cheap dataset profile → strategy), implemented as a five-axis fingerprint and domain map rather than a full Auto-sklearn portfolio. Broad held-out proof that routing beats a fixed learner across many task families remains **open** in the project brief — that is also the standard bar AutoML papers hold themselves to.

### 4.3 Learning to learn / meta-learning theory (context only)

MAML, Reptile, metric meta-learning (Matching Networks, Prototypical Nets) are mostly **few-shot deep learning**, not tabular AutoML. Cite lightly as “meta-learning” umbrella; do not over-claim kinship. UIL’s meta-layer is closer to **AutoML experience + rule mining** than to gradient-based few-shot meta-learning.

---

## 5. Search, populations, and architecture selection

### 5.1 PSO and swarm search

**Kennedy & Eberhart (1995)** — Particle Swarm Optimization. UIL’s classic path uses PSO-like / principle-guided search over feature weights and penalties. Important honesty from the codebase: linear separable surrogates make large swarms **re-derive an argsort** — a complexity pathology, not a swarm failure. Related literature: surrogate-assisted optimization; “when the surrogate is separable, closed form beats population search.”

### 5.2 Successive Halving and Hyperband

- **Jamieson & Talwalkar** — Successive Halving for non-stochastic bandits / HPO.
- **Li, Jamieson, DeSalvo, Rostamizadeh & Talwalkar (JMLR 2018)** — *Hyperband*: brackets of successive halving.

**UIL Architecture Race** is this family: cheap rungs → full fidelity; incumbent protected. Literature predicts **winner’s curse** if the selection metric is a high-variance single split — exactly Gate 2’s corr(Δacc, val-gap) ≈ −0.98 finding (Cawley & Talbot, next section).

### 5.3 Population Based Training

**Jaderberg et al. (2017)** — PBT: explore (perturb hyperparameters) + exploit (copy winners) online. Motivates UIL’s need for **architecture variation in the default path** so the journal contains learnable experience (F4).

### 5.4 fANOVA and configuration importance

**Hutter, Hoos & Leyton-Brown (ICML 2014)** — fANOVA: variance decomposition of performance over hyperparameters. Natural analysis tool over UIL journals once knobs vary.

---

## 6. Model selection, adaptive analysis, and evaluation hygiene

This section is as important as the model literature for UIL’s **Gate 2 culture**.

### 6.1 Overfitting the selection criterion

**Cawley & Talbot (JMLR 2010)**, *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*: variance of the selection score matters; best-of-*N* on a small validation set can erase real algorithm differences. Remedy: nested / double CV; keep evaluation data out of selection.

**UIL mapping:** Architecture Race regression on small *n*; sonar “cognition win” vanishing at `cv_folds=2`; judge fix via `_extra_val_splits`.

### 6.2 Reusable holdout / Thresholdout

**Dwork, Feldman, Hardt, Pitassi, Reingold & Roth (Science 2015)** — *The reusable holdout*; Thresholdout uses DP-style noise to answer adaptive queries without destroying holdout validity.

**UIL mapping:** champion bar + standing A/B gates (`cognition/thresholdout.py`, default ON). Correct citation when explaining why repeated Gate 2 queries need an instrument, not just “another split.”

### 6.3 Cross-seed reporting

UIL’s standing rule — never quote a single split; report mean±std via `eval_crossseed` — matches reproducibility checklists (**Pineau et al., JMLR 2021**, *Improving Reproducibility in Machine Learning Research*) and the bitter lesson that single-split luck is not a result.

### 6.4 Sutton’s Bitter Lesson

**Sutton (2019)** — general methods that leverage computation beat hand-crafted knowledge over time. Cognitive shells that add reasoning without adding effective search/compute often lose. Supports framing UIL’s cognitive negative result as **field-expected**, while still allowing specialized wins (interpretability, symbolic OOD) where the metric is not “more accuracy per FLOP on IID tabular.”

---

## 7. Causal inference and off-policy learning from logs

### 7.1 Interventions vs associations

- **Pearl** — *Causality*; do-calculus; interventions \(P(y \mid do(x))\) ≠ conditioning.
- Textbooks: Spirtes–Glymour–Scheines; Imbens & Rubin (potential outcomes).

**UIL causal engine:** control/treatment trials on controllable factors; majority-of-3 confirmation before `CONFIRMED_CAUSAL`. Aligns with the principle that **active intervention**, not correlational mining, grounds causal status (existence audits: causal was the subsystem that actually changed predictions when scheduled).

### 7.2 Learning from logged bandit / search data

**Swaminathan & Joachims (ICML/JMLR 2015)** — Counterfactual Risk Minimization: logged data inherits the logging policy; naive ERM is biased; propensity weighting / CRM needed.

**UIL mapping:** journal mining without propensity is **confounded** by what PSO chose to try. Propensity footnote on principles (F6) is HITL visibility; full CRM reweighting remains research-grade future work.

---

## 8. Rule-based cognitive systems and knowledge graphs

### 8.1 Learning Classifier Systems

- **Holland** — Cognitive System One / LCS.
- **Wilson XCS** — accuracy-based LCS with genetic rule discovery. Condition→action rules, fitness from experience, population of classifiers.

UIL’s principle library (conditions → overrides, confidence, tiers) is spiritually closer to **LCS / production rules** than to deep meta-learning. Differences: UIL mines from a supervised regret journal with DT rules; XCS typically online RL credit assignment. Cite as **architectural ancestry**, not algorithmic identity.

### 8.2 Cognitive architectures (broad)

ACT-R, Soar, Sigma — symbolic/hybrid cognitive architectures with memories, production rules, and learning. UIL’s “20-part architecture” rhetoric sits in this cultural lineage; reviewers will expect either a careful cognitive-science claim or a clear statement that UIL is an **engineering meta-learning shell**, not a model of human cognition. Prefer the latter unless a psych study is run.

### 8.3 Knowledge graphs in ML systems

Industrial/academic KGs for recommendations, reasoning, and MLOps provenance. UIL’s editable network (typed edges, attention traversal, prune-to-archive) is a **run-internal epistemic graph**, not a world KG. Bug #18 (traversal explosion) echoes known issues with unbounded graph expansion in production systems — cite systems lessons (caps, archives, bulk ops) rather than Semantic Web theory.

### 8.4 Event sourcing / reproducible science artifacts

Event-sourced systems (Vernon / Fowler lineage in software engineering); computational notebook reproducibility; **MLflow / DVC**-style experiment tracking. UIL VersionStore (M1/M5) is closest to **append-only scientific lab notebook + reconstructible state**. Novelty for ML papers is usually the *coupling* to editable knowledge cells, not event sourcing per se.

---

## 9. Human-in-the-loop and interactive machine learning

### 9.1 Interactive ML

- **Amershi et al. (AI Magazine / CHI lines)** — interactive machine learning: tight loops where humans correct models.
- **Fails & Olsen; Settles (active learning survey)** — query strategies for labels.
- **Caruana et al. 2015** — editing intelligible models in medicine (again: strongest precedent for *editing the model*, not just labeling data).

### 9.2 What UIL still owes the literature

Software path (edit/reject, replay, lab reports) is built. **Human benefit is unmeasured** — the brief correctly lists HITL pilot as priority 1. Without a user study, papers should claim *capability* and *process*, not *proven usability gains*.

Suggested study designs from IML literature:

- within-subject edit vs no-edit on fidelity / trust / time-to-correction;
- experts fixing a planted harmful principle or basis term;
- comparison to SHAP-assisted black-box debugging.

---

## 10. Sparse linear models, basis expansion, and calibration

Technical underpinnings of SymbolicFormula (for methods sections).

| Topic | Canonical refs | UIL use |
|-------|----------------|---------|
| L1 / Lasso | Tibshirani 1996 | Sparse term selection |
| Basis expansion / splines | Hastie–Tibshirani–Friedman ESL | Stumps, KAN-style powers |
| Platt scaling | Platt 1999 | OvR head calibration (bug #24) |
| One-vs-rest multiclass | Rifkin & Klautau 2004 debate; practical OvR | Multiclass SymbolicFormula |
| Imbalanced metrics | Balanced accuracy; macro-F1 | Threshold objective under imbalance |
| Missing data | Little & Rubin; median/mode + missingness indicators | Perception path |

---

## 11. Synthesis — where UIL sits

```
                    Accuracy on IID tabular
                              ▲
                              │  TabPFN (small n)
                              │  XGBoost / HistGBM
                    EBM ●─────● RuleFit / PRE
                         \   /
                          \ /
                     UIL SymbolicFormula ←── primary defendable product
                          |
                          │ (optional shell)
                          ▼
              AutoML meta-learning / LCS-like rules
              Pearl-style interventions
              DreamCoder library growth
              Versioned HITL edits
                              │
                              ▼
                    Scientific metric shifts:
                    readability, editability, OOD symbolic,
                    negative result on cognitive accuracy
```

### 11.1 What UIL inherits

1. Glass-box tabular learning (GAM / GA²M / EBM / RuleFit).
2. AutoML routing + CASH / successive-halving search.
3. Symbolic regression for scientific targets.
4. Causal intervention epistemology.
5. Selection-hygiene and reusable-holdout methodology.
6. Interactive editing of intelligible models.

### 11.2 What UIL attempts to combine (integration claim)

A **single orchestrated loop**: perception → route → multi-stage search → learner → journal → mine → graph / causal / formulas → versioned HITL. Few published systems integrate *all* of these with existence audits and Gate 2 defaults. Integration is a systems contribution; it is **not** automatically an accuracy contribution.

### 11.3 What not to claim (literature-aligned)

- That cognitive meta-reasoning beats tuned GBDT on saturated tabular data.
- That short single-dataset journals yield transferable principles (AutoML sample complexity).
- That post-hoc XAI equivalence (SHAP) is the product.
- That HITL value is proven before a user study.
- That every module is load-bearing on every run (existence audits).

### 11.4 Strong paper theses consistent with both literature and measurements

1. **Readable sparse-symbolic classifiers at near-XGBoost parity**, with RuleFit/EBM baselines, cross-seed protocol.
2. **Symbolic OOD / Feynman-style recovery** where trees fail extrapolation.
3. **Systems + negative result:** a full cognitive shell can be wired end-to-end and still fail to improve IID accuracy — with ablations mapping to Cawley/Dwork/DreamCoder/AutoML lessons.
4. **Editable event-sourced knowledge** as an IML artifact (claim software + protocol; defer efficacy).

---

## 12. Annotated shortlist (start here)

### Must-cite for a UIL methods / related-work section

| Paper | Why |
|-------|-----|
| Friedman & Popescu 2008 (RuleFit) | Closest classical learner shape |
| Lou et al. 2013 (GA²M); Nori et al. 2019 / EBM docs | Glass-box parity precedent |
| Caruana et al. 2015 | Editable intelligible models |
| Chen & Guestrin 2016 (XGBoost) | Accuracy baseline |
| Grinsztajn et al. 2022 | Why cognitive accuracy gains are unlikely |
| Feurer et al. 2015 / Auto-sklearn 2.0 2022 | Meta-learning sample needs + routing |
| Cawley & Talbot 2010 | Selection overfitting (race / gates) |
| Dwork et al. 2015 | Thresholdout / reusable holdout |
| Udrescu & Tegmark 2020 (AI Feynman) | Symbolic science claim |
| Ellis et al. DreamCoder | Utility-gated libraries |
| Pearl (Causality) + Swaminathan & Joachims 2015 | Interventions vs logged mining |
| Hollmann et al. TabPFN v2 (Nature 2025) | Small-*n* ceiling / probe |
| Li et al. 2018 Hyperband; Jaderberg et al. 2017 PBT | Race / exploration design |
| Pineau et al. 2021 | Reproducibility / reporting |

### Secondary / context

- Hastie & Tibshirani — GAM
- Tibshirani — Lasso
- Schmidt & Lipson 2009
- Wilson — XCS
- Amershi — IML
- Sutton — Bitter Lesson
- Liu et al. — KAN
- Meinshausen — Node Harvest
- Lundberg — SHAP (contrast)

---

## 13. Gaps in the literature UIL could fill (research opportunities)

1. **Cross-seed readable-model bakeoff** — SymbolicFormula vs EBM vs RuleFit/`pre` vs XGBoost on a frozen suite (brief priority 3). Still surprisingly thin as a *single* reproducible package with edit provenance.
2. **Cognitive shells with identity tests** — “does this subsystem change any bit?” as a standard ablation, not only accuracy Δ.
3. **HITL edit of symbolic basis terms + principles** — user study with planted errors (open).
4. **Partial pooling across datasets for mined rules** — hierarchical Bayes between AutoML portfolios and single-run DT mining.
5. **Propensity-aware principle mining** — move F6 footnote to CRM-style correction.
6. **Symbolic OOD benchmark** — beyond one extrapolation anecdote; compare AI Feynman / PySR / UIL regressor.

---

## 14. Relation to in-repo documents

| Document | Role vs this survey |
|----------|---------------------|
| `PROFESSOR_PROJECT_BRIEF.md` | What was built and measured |
| `RESEARCH_FAILURE_ANALYSIS.md` | Failure → specific paper → remedy program |
| `UIL_V22_SCIENTIST_BLUEPRINT.md` | Design intent (v22) |
| `HARDCODED_VALUE_AUDIT_QA.md` | Constants taxonomy (not lit review) |
| `LITERATURE_SURVEY_PAPERS.md` | Flat paper catalog (7-column table form) |
| **This file** | Narrative field context, claim boundaries |

---

## 15. Closing

The literature supports UIL most where the brief is already strongest: **glass-box tabular models near tree-ensemble accuracy**, **symbolic methods on scientific/OOD targets**, and **rigorous selection/evaluation methodology**. It predicts UIL’s weakest headline — **cognitive meta-accuracy on ordinary tables** — with unusual clarity (Grinsztajn; Bitter Lesson; AutoML sample complexity; Cawley selection variance; DreamCoder utility gating).

A professor-facing narrative that stays inside both the measurements and the literature is therefore:

> UIL is a readable, editable tabular learning stack with a full experimental cognitive shell. The stack’s accuracy story is SymbolicFormula and correctness engineering; the shell’s scientific story is instrumentation, negative results, and human editability still to be validated with users.

---

*Primary sources checked via publisher/arXiv pages (RuleFit, EBM/InterpretML, Grinsztajn 2022, Cawley & Talbot 2010, Dwork et al. 2015, Auto-sklearn, AI Feynman, Hyperband, PBT, DreamCoder). Align citation keys with your bibliography style before submission.*
