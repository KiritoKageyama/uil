# Performance Tests, System Diagnostics, and Failure Analysis

**Project:** Universal Intelligence Layer (`figpso_uil`)  
**Document Series:** Technical Report — Volume III  
**Subject Area:** Performance Diagnostics, Failure Mode Analysis, and Model Selection Verification  

---

## Abstract

This volume presents the performance testing and diagnostic failure analysis of the Universal Intelligence Layer (UIL). Scientific clarity requires that negative test outcomes be investigated with the same care as positive results. We evaluate UIL across standard tabular classification benchmarks and scientific law discovery tasks. Our test measurements reveal two distinct performance regimes. On one hand, UIL's sparse basis classifier matches the accuracy of state-of-the-art gradient boosted trees while maintaining full human interpretability, and its symbolic engine achieves near-perfect out-of-distribution law extrapolation ($R^2 \approx 0.99$). On the other hand, aggressive deployment of continuous meta-reasoning layers yields a slightly negative average accuracy impact ($\Delta = -0.0132$) and introduces operational inertia. We analyze these test failures through established literature, mapping them to saturated tabular headroom, validation selection bias, and regularization penalty imbalance.

---

## 1. Testing Methodology and Evaluation Setup

To ensure fair comparisons, UIL's performance was evaluated under a strict cross-validation setup using standardized tabular datasets and synthetic physical benchmarks.

### Flowchart: Testing & Gating Evaluation Protocol

```
+-------------------------------------------------------------------+
|                        RAW TABULAR DATASET                        |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|               TRAIN / VALIDATION / TEST DATA SPLIT                |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|             TRAIN-ONLY IMPUTATION & NORMALIZATION                 |
|            (Median/Mode Imputer + Invertible Scaler)              |
+---------------------------------+---------------------------------+
                                  |
        +-------------------------+-------------------------+
        |                                                   |
        v                                                   v
+-----------------------------------+   +-----------------------------------+
|   FIT PLAIN SEARCH + BASELINE     |   |   FIT COGNITIVE SHELL AUGMENTED   |
|   LEARNER (XGBoost / PSO)         |   |   LEARNER (Full Meta-Layer)       |
+-----------------+-----------------+   +-----------------+-----------------+
                  |                                       |
                  +-------------------+-------------------+
                                      |
                                      v
+-------------------------------------------------------------------+
|            EVALUATE ON ISOLATED HELD-OUT TEST FOLD                |
+---------------------------------+---------------------------------+
                                  |
                 +----------------+----------------+
                 |                                 |
                 v                                 v
+----------------------------------+  +-------------------------------------+
| ACCURACY PARITY / OOD LAW WIN    |  | ACCURACY DELTA <= 0 / INERT MODULES |
| (Record Glass-Box Readability)   |  | (Audit Inert Software & Overfit)    |
+----------------------------------+  +-------------------------------------+
```

### 1.1 Baseline Learners and Configuration Controls
All tests compared UIL against established algorithms operating under identical data splits:
- **XGBoost (Gradient Boosted Decision Trees):** Tuned over standard hyperparameter grids (tree depth, learning rate, subsample ratio).
- **Random Forest:** Standard ensemble baseline using axis-aligned decision trees.
- **Sparse Logistic Regression (Lasso / ElasticNet):** Standard linear baselines.
- **Plain Particle Swarm Optimization (PSO):** Hyperparameter search operating directly over learner configurations without cognitive metadata processing.

### 1.2 Evaluation Metrics and Testing Rules
- **Held-Out Predictive Accuracy & AUC:** Evaluated on test splits that were isolated during perception, feature selection, and model tuning.
- **Out-of-Distribution (OOD) Extrapolation $R^2$:** Measured on synthetic physical datasets where test evaluation features fell outside the numerical range of training samples.
- **Module Prediction Impact:** Tracked whether enabling an auxiliary cognitive module altered final prediction labels or remained prediction-inert.

---

## 2. Positive Test Results: Interpretability and Extrapolation

### 2.1 Readable Classification Matching Gradient Boosting
On standard tabular classification tasks, UIL's sparse basis classifier—combining raw features, evolved formulas, decision stumps, and conjunction rules under L1 regularization—demonstrated that interpretability does not require sacrificing predictive accuracy.

For example, on the standard Spambase benchmark:
- **Tuned XGBoost (Black Box):** Achieved **94.4%** test accuracy.
- **UIL Sparse Basis Classifier (Glass Box):** Achieved **94.9%** test accuracy.

Crucially, UIL's final model consisted of a compact additive equation containing readable rules and linear terms, allowing human reviewers to inspect exactly which feature combinations triggered a classification decision.

### Flowchart: Out-of-Distribution Extrapolation Comparison

```
+-------------------------------------------------------------------+
|     IN-DOMAIN TRAINING REGION:  x in [x_min, x_max]               |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|     OUT-OF-DOMAIN TEST EVALUATION REGION:  x > x_max              |
+---------------------------------+---------------------------------+
                                  |
         +------------------------+------------------------+
         |                                                 |
         v                                                 v
+----------------------------------+   +------------------------------------+
|  DECISION TREE / XGBOOST MODEL   |   |   UIL SYMBOLIC FORMULA ENGINE      |
|  (Splits Space into Step Bounds) |   |   (Discovers Closed-Form Expression) |
+----------------+-----------------+   +-----------------+------------------+
                 |                                       |
                 v                                       v
+----------------------------------+   +------------------------------------+
|  CONSTANT STEP EXTRAPOLATION     |   |  EXACT PHYSICAL LAW EXTRAPOLATION  |
|  FAILED PREDICTION: R^2 = -5.52  |   |  HIGH ACCURACY WIN: R^2 = 0.992    |
+----------------------------------+   +------------------------------------+
```

### 2.2 Out-of-Distribution Law Recovery
In scientific discovery tests, models were trained on physical data (e.g., synthetic kinetic energy datasets) within a bounded feature range $[x_{\min}, x_{\max}]$ and tested on values $x > x_{\max}$.

- **XGBoost:** Produced an out-of-distribution $R^2$ score of **$-5.52$**. Because decision trees split space into bounded step functions, XGBoost output constant predictions for all points outside the training boundary.
- **UIL Symbolic Basis Engine:** Successfully discovered the true underlying mathematical relationship, achieving an out-of-distribution $R^2$ score of **$0.992$**.

This result highlights UIL's core strength: when data follows analytical physical laws, symbolic basis evolution outperforms tree ensembles under extrapolation.

---

## 3. Diagnostic Audits of Cognitive Meta-Layer Processing

To test whether UIL's cognitive shell—comprising continuous journal logging, rule mining, knowledge graph traversals, and causal checks—improved model optimization, we executed systematic cross-seed gating evaluations.

### 3.1 Gated Accuracy Comparison
In a controlled cross-seed study (Gate 2 evaluation), system performance was measured with the cognitive shell enabled versus a plain PSO baseline searching over the same learner configurations.

- **Mean Accuracy Change ($\Delta \text{Accuracy}$):** Enabling the default cognitive shell resulted in a mean change of **$-0.0132$** relative to plain search.
- **Execution Overhead:** Running cognitive processing increased optimization time by $10\times$ to $100\times$ without delivering systematic accuracy gains.

### 3.2 Subsystem Inertia Audits
In short optimization runs, diagnostic tests revealed that nine auxiliary cognitive modules were **prediction-inert**. These modules executed successfully, generated log entries, and wrote state files to disk, but had zero impact on final model predictions.

| Cognitive Subsystem | Operational Activity | Prediction Impact | Status |
|---|---|---|---|
| **Journaling & Logging** | Recorded every training iteration to disk. | None (Passive observation). | Passive Log |
| **Knowledge Graph Traversal** | Built node-edge connections between rules. | Inert on short runs. | Passive Graph |
| **Rule Miner (DT-Miner)** | Mined decision tree paths into rule candidates. | Active when wired to basis. | Predictive Component |
| **Causal Hypothesis Checker** | Evaluated feature intervention hypotheses. | Inert without manual trigger. | Diagnostic Tool |
| **Automated Model Racing** | Swapped learner families based on validation splits. | Negative ($\Delta < 0$). | Overfitting Risk |

---

## 4. Literature Analysis of Measured Failure Modes

Rather than viewing negative results as unexplained software bugs, UIL maps every measured failure to established principles in computer science literature.

### Flowchart: Failure Mode Diagnostic Mapping

```
+-------------------------------------------------------------------+
|                    MEASURED SYSTEM FAILURE                        |
|                                                                   |
| [F1: Cognitive Shell Delta ~ 0]   [F2: Model Racing Overfitting]  |
|               [F3: Complex Conjunction Rules Zeroed Out]          |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                    LITERATURE ROOT CAUSE                          |
|                                                                   |
| [C1: Saturated Tabular Headroom (Grinsztajn 2022 / Sutton 2019)]  |
| [C2: Validation Selection Bias (Cawley & Talbot 2010)]            |
| [C3: Disproportionate Penalty on High-Variance Rules (Friedman)]  |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                    PRACTICAL ARCHITECTURAL FIX                    |
|                                                                   |
| [R1: Shift Goal to Readability]   [R2: Use Fixed CV Not Racing]   |
|            [R3: Per-Family Regularization Budgets]                |
+-------------------------------------------------------------------+
```

### 4.1 Failure Mode 1: Saturated Tabular Headroom (F1)
- **Observed Behavior:** Adding cognitive reasoning machinery yielded $\Delta \text{Accuracy} \approx 0.000$ over tuned tree baselines across standard tabular benchmarks.
- **Literature Mapping:** Grinsztajn, Oyallon, and Varoquaux (NeurIPS 2022) demonstrated that tuned gradient boosted trees sit near the practical performance ceiling for medium tabular datasets. Headroom above a tuned tree ensemble is negligible. Furthermore, Rich Sutton's *Bitter Lesson* (2019) notes that hand-crafted reasoning machinery loses to direct search and compute when compute is the binding constraint.
- **Scientific Conclusion:** Competing on raw tabular accuracy by adding cognitive machinery was structurally unfeasible. UIL's tie with tuned baselines represents the literature-consistent boundary of the tabular domain.

### 4.2 Failure Mode 2: Validation Selection Overfitting (F2)
- **Observed Behavior:** Automated algorithm racing produced a strong negative correlation ($r = -0.977$) between validation gap expansion and test accuracy degradation. The system selected candidate models that overfit short validation splits.
- **Literature Mapping:** Cawley and Talbot (JMLR 2010) proved that optimizing hyperparameter or model choices over small validation sets introduces heavy selection bias. The variance of the selection criterion degrades test performance by an amount comparable to choosing a weaker algorithm.
- **Scientific Conclusion:** Continuous automated racing based on short validation split feedback must be disabled in favor of fixed, cross-validated learner paths.

### 4.3 Failure Mode 3: Regularization Starvation of Complex Rules (F3)
- **Observed Behavior:** When complex conjunctive rules (e.g., multi-condition logical statements) were combined with simple linear features under a single uniform L1 penalty, the linear features dominated, leaving complex rules with zero weights.
- **Literature Mapping:** Friedman and Popescu (2008) noted that complex interaction terms naturally possess larger variance than simple attributes. Under uniform L1 regularization, high-variance interaction terms are penalized disproportionately and starved of weight budget.
- **Scientific Conclusion:** Multi-family basis models require separate regularization multipliers for each basis family to allow complex rules to compete fairly against raw attributes.

---

## 5. Guidelines for Auditable Learning Systems

Based on these test findings, future engineering of transparent tabular learning systems should follow four practical guidelines:

1. **Separate Readability from Meta-Reasoning:** Evaluate interpretable models on their clarity and performance parity rather than expecting meta-reasoning shells to boost accuracy.
2. **Eliminate Inert Modules:** Passive logging and graph updates should run asynchronously in background threads so they do not block core optimization paths.
3. **Use Fixed Validation Schemes:** Avoid continuous algorithm selection over volatile validation splits; use static cross-validation to prevent selection overfitting.
4. **Implement Family-Specific Regularization Multipliers:** Apply distinct regularization budgets to raw features, decision stumps, conjunction rules, and symbolic formulas.

---

## References

- Cawley, G. C., & Talbot, N. L. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research*, 11, 2079–2107.
- Friedman, J. H., & Popescu, B. E. (2008). Predictive learning via rule ensembles. *The Annals of Applied Statistics*, 2(3), 916–954.
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on tabular data? *NeurIPS*, 35, 507–520.
- Sutton, R. (2019). The bitter lesson. *Incomplete Ideas (Blog)*, 13(1).
