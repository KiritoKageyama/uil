# Plan — Update `06_TRANSPARENT_EDITABLE_NEURAL_SUBSTRATE_LITSURVEY.md`

**Target file (single):**
`C:\Users\arinb\OneDrive\Desktop\06_TRANSPARENT_EDITABLE_NEURAL_SUBSTRATE_LITSURVEY.md`
(This document is both the "master document" and the "hypothesis document" —
it states the scientific thesis.)

Also modifies its identical mirror in
`C:\Users\arinb\OneDrive\Desktop\New folder (3)\academic_v2\06_...` so the two
copies do not drift.

---

## Context

Two independent inputs are asking for the same edits, from opposite directions.

1. **The external review** you pasted concludes: strong document, but the
   scientific novelty is unproven, the thesis is too broad ("make NNs
   interpretable"), the white space ≠ novelty, localization must become
   *causal*, and the paper lacks the two things a reviewer will demand — a
   concrete metric suite (localization / edit locality / edit persistence /
   rollback correctness / catastrophic interference) and an explicit list of
   what would falsify the Knowledge Cell hypothesis. It also warns that hard
   routing can *manufacture* interpretability and that S4 must be a small
   transformer with a full ablation, not an LLM jump.

2. **Yesterday's measurements** — `results/hitl_law_edit/SUMMARY.md`,
   `results/existence_sweep/FINDINGS.md`, `results/existence_mined/FINDINGS.md`,
   `results/compute_cost/SUMMARY.md`, `tests/test_law_fidelity.py` — resolve
   exactly the problem the reviewer flagged: the paper's central **Observed**
   claim is attached to `NetworkNode`, and `NetworkNode` was measured to be
   inspectable, editable, versioned, replayable, **and inert** (0.0% of
   predictions moved by removing it entirely at default flags; 0–2% with mining
   forced on; accuracy unchanged). The unit the claim actually holds for is
   **the fitted law** — where a single-term edit moves the logit by exactly the
   predicted amount (error 7e-14), and where a human edit under distribution
   shift is worth **+0.3141**, CI [+0.298, +0.330], with a control showing the
   gain is not automatable.

The two inputs converge. The reviewer's structural changes turn the document
from a research proposal into a research **program** with a metric suite and a
falsifier list. My C1–C6 changes (drafted in `figpso_uil/LITSURVEY_CORRECTIONS.md`)
move the central Observed claim to the unit it actually holds for and upgrade
Section 9 from a hedge into a measured position. Applied together, the
document becomes both more honest and more defensible.

Intended outcome: the document reads as *"a well-formulated research program
with one measured positive result (editability, +0.314 under shift, with
control), a precise metric suite, and an explicit list of failure modes"* —
which is the reviewer's own recommended framing and matches what is now on
disk in `results/`.

---

## What changes (grouped so the diff is reviewable in three passes)

### Group A — Corrections from measurement (already drafted C1–C6)

Apply the six replacements from `figpso_uil/LITSURVEY_CORRECTIONS.md` exactly
as drafted. They are already worded to preserve the Observed / Designed /
Proposed framework of the document (Section 0). Summary:

- **C1** — Section 1 final paragraph: relocate the core Observed claim from
  NetworkNode to the fitted law; state both the +7e-14 exactness and the 0.0%
  NetworkNode identity-test result.
- **C2** — Section 2 table: mark NetworkNode "edits do not reach the forward
  pass" with the numbers, and add a new row for the fitted law with the +0.3141
  shift result and CI.
- **C3** — Section 2 "central honest statement": say the claim is Observed for
  the fitted law and falsified for graph nodes; add the shift and control
  results.
- **C4** — Section 4 white-space table: split final column into `Observed
  (fitted law, tabular)` and `Proposed (Knowledge Cell / KDN)`; add a
  "Measured benefit from a human edit" row; add the paragraph noting no
  surveyed paper reports this result with a control.
- **C5** — Section 9: upgrade the accuracy non-claim from hedge to measured
  position (search loop −0.0056, CI upper +0.0064 excludes a 0.01 benefit);
  add two new non-claims (graph substrate does not participate; LLM narration
  is not explanation).
- **C6** — New §5.4: two acceptance tests (fidelity round-trip, edit
  exactness) with the measured numbers.

### Group B — Structural changes from the review

These are new content, added or reframed in place. They do not conflict with
Group A; they build on it.

- **B1 — Reframe the thesis (Section 1, opening).**
  Move from *"KG + NN mechanics combined in a Knowledge Cell produces an
  inspectable, editable, versioned, forward-flow substrate"* (a design claim)
  to the reviewer's operational form:
  > *"Persistent, typed computational cells can enable reliable localized
  > intervention in a learned model while preserving unrelated behaviour
  > better than existing model-editing approaches."*
  Keep the four properties list, but present them as *ingredients* of the
  above claim, not as the claim itself. This directly answers the reviewer's
  "why do these four properties need to be unified?" — because *reliable
  localized intervention* is the capability the combination is meant to buy,
  and existing editing methods (P8, P9, P11, P12) provide editing without
  persistent identity or versioning and therefore degrade after many edits
  (P22).

- **B2 — New §5.5: Localization must be causal, not just logging.**
  Add a subsection following C6's §5.4. State the trap the reviewer raised
  ("Cell B might be wrong only because Cell A changed") explicitly, connect it
  to `results/existence_mined/FINDINGS.md` (removing a subsystem moves
  predictions in both cells — that is a *correlational* attribution, not
  causal), and adopt the position that Knowledge Cell attribution must be
  validated by control/treatment interventions (which is what the current
  codebase's `cognition/causal_intervention.py` already provides the machinery
  for, and what `COGNITIVE_ENGINE_RESCUE.md` re-aims it toward: *"does this
  edit change behaviour on inputs it shouldn't?"* — the edit-locality
  question).

- **B3 — New §5.6: Metric suite (five measured metrics).**
  Adopt the reviewer's five metrics as first-class evaluation criteria. Each
  is one paragraph, in the reviewer's order:
  1. **Localization accuracy** — top-k precision/recall of "cell identified"
     vs "cell truly responsible" (needs a synthetic ground truth or a plant).
  2. **Edit locality** — the ROME/MEMIT/Yao 2023 axis: target changes,
     non-targets stay. Ties directly to `cognition/causal_intervention.py`
     control/treatment framing.
  3. **Edit persistence** — cell identity survives further training and
     further edits. The reviewer correctly names this as the property
     dense NNs do not naturally have.
  4. **Rollback correctness** — restoring v1 restores *behaviour*, not merely
     parameter values. The VersionStore already provides byte-identical
     parameter replay (`cognition/version_store.py`); the behavioural
     equivalent is untested and must be measured.
  5. **Catastrophic interference under repeated editing** — 1 → 10 → 100 →
     1000 → 10000 edits, measuring locality drift. This is the reviewer's
     "killer experiment" and the direct MEMIT (P9) comparison.
  Where a metric is already measured today for the fitted law, cite the
  number (fidelity + edit-exactness for #4 in the small case). Otherwise mark
  Designed/Proposed per Section 0.

- **B4 — New Section 9.5: What would falsify the Knowledge Cell hypothesis?**
  Insert the reviewer's eight failure modes verbatim in intent (paraphrased
  for the document's voice). This is the sharpest single change the reviewer
  asked for and the paper cannot ship at the level the reviewer suggests
  without it. Cross-reference each item to the metric in §5.6 that would
  detect it.

- **B5 — S3 stress-test framing (Section 5.2 + Section 8 row S3).**
  Add one paragraph acknowledging the reviewer's caution: hard routing can
  manufacture interpretability by construction. State that S3 (soft routing)
  is therefore not just an optimization stage but a **scientific stress test
  of the thesis** — the point at which one either shows Knowledge Cell
  attribution survives blended contributions, or admits the thesis was
  scoped by architectural constraint. Add the same note to the S3 row in the
  Section 8 roadmap.

- **B6 — S4 ablation protocol (Section 8 row S4 + a new §6.6).**
  Replace the current S4 language ("Replace or augment FFN sublayers with
  inspectable Knowledge Cell banks in a small transformer") with a proper
  ablation protocol: **small transformer, KC-FFN vs standard FFN, trained on
  identical data, compared on all five §5.6 metrics before any scale-up**.
  Explicitly state "do not jump to an LLM" as a scoped decision. Add a §6.6
  giving the ablation table shape.

- **B7 — Closing statement (Section 11) update.**
  Two-sentence delta: what was previously "a first, small piece of evidence
  for this at the NetworkNode level" is now "a first, small piece of evidence
  for this at the *fitted-law* level, with a measured shift-condition benefit
  of +0.314 and a control ruling out data-derived automation of the same
  gain; the metric suite in §5.6 and the failure-mode list in §9.5 make the
  Knowledge Cell hypothesis falsifiable."

### What deliberately does NOT change

- The overall structure (Sections 0–11) and every paper entry in Section 3.
- The Observed / Designed / Proposed tagging system (Section 0) — the review
  praised it and it is doing exactly what it was built to do.
- Authorship, date, project fields.
- The mermaid diagrams in Sections 5.1, 5.2, 6.4, 6.5, 7.1–7.3 — they are
  still accurate; only their surrounding claims change.
- Section 3's paper entries and Section 10 references.
- The Gantt chart in Section 8 — dates are illustrative and I am not
  modifying the schedule.

---

## Critical files

- **`C:\Users\arinb\OneDrive\Desktop\06_TRANSPARENT_EDITABLE_NEURAL_SUBSTRATE_LITSURVEY.md`**
  — primary target.
- **`C:\Users\arinb\OneDrive\Desktop\New folder (3)\academic_v2\06_TRANSPARENT_EDITABLE_NEURAL_SUBSTRATE_LITSURVEY.md`**
  — identical mirror per `Glob`; apply the same edits so the two copies do
  not drift.
- Sources being cited into the document (READ ONLY, no edits):
  `figpso_uil/LITSURVEY_CORRECTIONS.md` (drafted C1–C6),
  `figpso_uil/results/hitl_law_edit/SUMMARY.md`,
  `figpso_uil/results/existence_sweep/FINDINGS.md`,
  `figpso_uil/results/existence_mined/FINDINGS.md`,
  `figpso_uil/results/compute_cost/SUMMARY.md`,
  `figpso_uil/COGNITIVE_ENGINE_RESCUE.md`,
  `figpso_uil/tests/test_law_fidelity.py`.
- **Not modified:** the other five academic docs (01–05). If any of them
  restate the same NetworkNode-Observed claim they will drift out of sync
  with 06; that is a follow-up, deliberately out of scope for this plan.

---

## Reuse — existing artifacts this plan builds on rather than duplicating

- **C1–C6 replacements** are already fully drafted in
  `figpso_uil/LITSURVEY_CORRECTIONS.md` — Group A is a mechanical apply of
  that file. I do not re-derive the corrections.
- The **causal-attribution machinery** for §5.5 already exists in
  `figpso_uil/cognition/causal_intervention.py` (control/treatment
  interventions with a `_strong_count >= 2` majority-of-3 gate). §5.5 cites
  it as the implementation vehicle rather than proposing something new.
- The **rollback / provenance** claims in §5.6 metric 4 point at
  `figpso_uil/cognition/version_store.py` (event-sourced, byte-identical
  parameter replay, verified by M1 in `tests/test_m1_version_store.py`).
- The **fidelity round-trip** and **edit exactness** numbers used in §5.6
  and C6 come from `figpso_uil/tests/test_law_fidelity.py` (F1–F8, 8/8
  passing) — cited, not re-run.

---

## Verification

After applying the edits, verify end-to-end without executing any code
against the litsurvey itself (it is a text document):

1. **Both copies match.** `diff` the two 06_...md files (main copy on Desktop
   and mirror under `academic_v2/`). They must be byte-identical after the
   edit.
2. **No Observed claim is orphaned.** Grep the modified document for
   `Observed` and confirm every occurrence is either (a) about the fitted
   law and cites `results/hitl_law_edit/` or `tests/test_law_fidelity.py`,
   (b) about a cited paper's own reported result, or (c) explicitly narrated
   as "was Observed pre-2026-08-05, now falsified for NetworkNode". No
   `Observed` about `NetworkNode` invariant 4 (forward-flow participation)
   may remain.
3. **Every numeric result carries a source path.** Every quantity newly
   introduced (+0.3141, [+0.298, +0.330], 7e-14, 0.0% / 0–2%, −0.0056,
   +0.0064, 3900×, 3.8%) is followed by a `results/…` or `tests/…` reference.
4. **Section 0's tagging convention still works.** Every Group B addition
   (thesis reframe, causal §5.5, metrics §5.6, falsifiers §9.5, S3/S4
   language) uses `Observed / Designed / Proposed` phrasing as Section 0
   specifies.
5. **Reviewer round-trip.** Re-read the reviewer's message with the modified
   document open, item by item (1 reframe, 2 white-space, 3 causal, 4
   metrics, 5 S3 stress-test, 6 S4 ablation, 7 falsifier list, 8 program
   framing). Each of the eight points must have a locatable, cited paragraph
   in the modified document.

If any of the five checks fail, the plan is not fully executed; fix and
re-verify before considering the change shipped.
