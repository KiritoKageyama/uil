# Corrections to `06_TRANSPARENT_EDITABLE_NEURAL_SUBSTRATE_LITSURVEY.md`

Drafted 2026-08-06 against measurements from that day. **Not applied** — the
litsurvey is an authored, outward-facing document (Arin Bansal, 2 Aug 2026) on
OneDrive/Desktop, so these are exact before/after replacements for review.

**Why these matter:** the document's central Observed claim is attached to the
wrong unit. As written it overclaims for `NetworkNode` and *under*claims for
what was actually demonstrated. Both are now measured.

Evidence: `results/existence_sweep/FINDINGS.md`,
`results/existence_mined/FINDINGS.md`, `results/hitl_law_edit/SUMMARY.md`,
`tests/test_law_fidelity.py`.

---

## C1 — Section 1, final paragraph (the core Observed claim)

**BEFORE**

> The current Unified Intelligence Layer (UIL) proof-of-concept exists to
> establish the smallest possible version of this claim: that in a tabular
> AutoML setting, individual computational nodes (NetworkNode instances) can be
> inspected, that errors or outdated knowledge inside them can be identified,
> and that they can be corrected in real time, with the correction taking
> effect in the live forward pass rather than requiring a full retrain.

**AFTER**

> The current Unified Intelligence Layer (UIL) proof-of-concept exists to
> establish the smallest possible version of this claim, and measurement has
> since located it precisely. The claim holds for **the terms of the fitted
> law** — the named basis functions of the sparse-linear model — which can be
> inspected, attributed, and corrected in real time, with the correction taking
> effect in the live forward pass and moving the output by exactly the
> predicted amount (maximum error 7e-14). It does **not** hold for
> `NetworkNode` graph instances: those can be inspected, edited, versioned and
> replayed, but an identity test across 36 subsystem-removals showed graph
> edits move 0.0% of predictions at default settings, and 0–2% with rule mining
> forced on, with no accuracy effect. The editability property is real and is a
> property of the **model's own law**, not of the knowledge graph above it.

---

## C2 — Section 2, table row for NetworkNode

**BEFORE**

> | NetworkNode (graph substrate) | High | Low to none (symbolic/graph-native) | Built and tested | Typed nodes, typed edges, traversal, and per-node inspection and correction are implemented and have been exercised in the PoC. |

**AFTER**

> | NetworkNode (graph substrate) | High | Low to none (symbolic/graph-native) | Built and tested; **edits do not reach the forward pass** | Typed nodes, typed edges, traversal, per-node inspection, edit, versioning and replay are implemented and exercised. Measured: removing the substrate entirely changes 0.0% of predictions (default) / 0–2% (mining on), with no accuracy effect. Inspectable and editable, but not causally connected to the output. |
> | **Fitted law (sparse-linear terms)** | **High** | **Medium (linear readout over a composed basis)** | **Built and tested; edits ARE live** | **The unit the editability claim actually holds for. Exact round-trip render/evaluate; term deletion moves the logit by exactly the predicted amount; human edit under distribution shift measured at +0.3141, 95% CI [+0.2984, +0.3298], 147 cells.** |

---

## C3 — Section 2, "central honest statement"

**BEFORE**

> The claim under test is narrower and, if true, more structurally
> significant: that individual computational nodes in this substrate can be
> inspected, that the source of a specific error or piece of outdated knowledge
> inside a node can be attributed, and that a correction can be applied
> directly to that node and take effect in the live forward pass, without a
> full retrain.

**AFTER**

> The claim under test is narrower and, if true, more structurally
> significant: that individual computational units in this substrate can be
> inspected, that a specific error or piece of outdated knowledge can be
> attributed to a unit, and that a correction applied to that unit takes effect
> in the live forward pass without a full retrain. That claim is now
> **Observed** for the fitted law's terms and **falsified** for graph nodes.
> Two further results scope it. First, the correction must come from knowledge
> the training data does not contain: an expert reasoning from the same data is
> worth **−0.0016, 95% CI [−0.0042, +0.0011]** over 180 cells — equivalent to
> no edit at all — while an expert supplying outside knowledge under
> distribution shift is worth **+0.3141**. Second, a control confirms the gain
> is not automatable: on the same shifted data a data-derived heuristic edited
> 18 of 180 cells, moved 0.20% of predictions, and never found the artifact,
> because the artifact was the most stable term in training.

---

## C4 — Section 4, white-space table, final column

**BEFORE** — every cell reads `Proposed: yes…`

**AFTER** — split the column into what is now measured vs still proposed:

> | Capability | … | This work, **Observed** (fitted law, tabular) | This work, **Proposed** (Knowledge Cell / KDN) |
> |---|---|---|---|
> | Structural node identity persists across training | … | Yes — named basis terms with stable identity | Proposed at cell granularity inside NN/transformer |
> | Single-unit surgical edit without full retrain | … | Yes — exact, error ≤ 7e-14 | Proposed |
> | Human-readable versioned edit history | … | Yes — event-sourced, byte-identical replay | Proposed |
> | Forward-flow computation preserved after edit | … | Yes — round-trip tested | Proposed |
> | Unified graph-plus-neural substrate | … | **No — the graph half is inert** | Proposed |
> | **Measured benefit from a human edit** | … | **Yes: +0.314 under shift, with control** | Proposed |

Add after the table:

> The bottom row is the one no surveyed paper reports. ROME, MEMIT, MEND and
> SERAC (P8, P9, P11, P12) demonstrate that an edit *can be applied*; none
> demonstrates, with a control, that a human edit *is worth something a
> data-derived method could not have found*. That control is the contribution
> most directly transferable from this PoC.

---

## C5 — Section 9, first non-claim (upgrade from hedge to result)

**BEFORE**

> - We do not claim state-of-the-art tabular prediction accuracy. TabPFN (P15)
>   and a tuned dense MLP baseline can match or exceed the current PoC on raw
>   accuracy; accuracy was never the metric under test (Section 2).

**AFTER**

> - We do not claim state-of-the-art tabular prediction accuracy, and this is
>   now a measured position rather than a hedge. Paired against one ordinary
>   fit of the same learner on identical splits, the full search loop returns
>   **−0.0056, 95% CI [−0.0175, +0.0064]** — a benefit of 0.01 is *excluded* —
>   at roughly 3900× the compute. The cognitive layer above it moves 0–2% of
>   predictions with no accuracy effect. Accuracy is not merely "not the metric
>   under test"; it has been tested thoroughly and is zero. The editability
>   result (+0.314, Section 2) is what the project claims instead.

Add two further non-claims:

> - We do not claim the knowledge-graph substrate participates in the forward
>   pass. It is inspectable, editable, versioned and replayable, and measurably
>   does not change the output. Invariant 4 of Section 5.1 ("inspectable
>   activation trace") is therefore satisfied by the fitted law, not by
>   NetworkNode.
> - We do not claim an LLM narration of a trace constitutes an explanation. Per
>   Jain and Wallace (P18), any natural-language account must be regenerable
>   from a faithful trace and must never be the source of the explanation. A
>   rendered view that does not round-trip is worse than opacity, because it is
>   trusted: this project's own renderer printed standardized coefficients
>   against raw-space term names and mislabelled 3.8% of rows before the defect
>   was found.

---

## C6 — new subsection for Section 5 (proposed addition)

> ### 5.4 Two acceptance tests any Knowledge Cell implementation must pass
>
> Both were learned the hard way in the PoC and neither is standard practice
> in the editing literature.
>
> **Fidelity.** `evaluate(render(cell)) == forward(cell)`, exactly, for every
> quantity the forward pass uses. A rendered view that silently omits a
> standardization, a threshold, or an index mapping is not an inspection
> surface; it is a plausible-looking wrong answer. Measured in this PoC before
> the fix: 3.8% of rows mislabelled by a renderer nobody suspected.
>
> **Edit exactness.** Applying a patch must move the output by the amount the
> interface said it would. "The edit was accepted" is not the property that
> matters; "the edit did what it claimed" is. Measured after the fix: maximum
> error 7e-14.
