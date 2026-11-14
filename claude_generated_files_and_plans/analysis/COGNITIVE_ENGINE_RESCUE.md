# The cognitive engine: re-aim, don't discard

**Status: DESIGN. Nothing here is measured yet. Written 2026-08-06, end of the
session that falsified the accuracy claims and proved the HITL law channel.**

Read `results/hitl_law_edit/SUMMARY.md` and
`results/existence_mined/FINDINGS.md` first — this document only makes sense
against those numbers.

---

## The thesis

The machinery is fine. The job is wrong.

The cognitive engine was built as a **hyperparameter tuner** and measured, at
every level, to be worth nothing in that role. But look at what it actually
*is*, ignoring what it currently does:

- a journal of (situation → action → outcome)
- a miner that extracts conditional rules from that journal
- a library with confidence, evidence, promotion, retirement
- a matcher that fires rules in the right context
- provenance and replay on everything

That is not a tuner. **That is an institutional memory for corrections.**

---

## The three killers, and why they are properties of the JOB

| killer | measured | why it is the job, not the code |
|---|---|---|
| sample complexity | miner needs ~200 journal rows, gets 8–40 | one run produces a handful of records |
| narrow action space | output was one scalar; widened to arbitrary `config.*` and still nothing | "nudge a hyperparameter" is a low-information decision |
| unreliable judge | `corr(Δacc, val-gap swing) = −0.98` | a ~30-row validation split cannot rank candidates |

All three are properties of the task *"choose hyperparameters for a fit on
small saturated tabular data."* That task has no data, no headroom, and no
referee. A case-based reasoning system was pointed at the one problem where
case-based reasoning cannot win.

## Re-point it at human edits and all three invert

| | tuning hyperparameters | learning from human edits |
|---|---|---|
| **data** | 8–40 rows per run | one row per edit — and law edits are ~1s, so thousands are cheap |
| **action space** | scalar nudge | "delete term j", "feature k is an artifact", "this rule fails for segment S" — discrete, semantic, high-information |
| **judge** | noisy 30-row val split | did the edit hold on held-out data? did unrelated predictions break? measurable on every input, exactly |

---

## The constraint that makes this more than a reshuffle

The shift control (`--mode shift_stability`, 180 cells) is unambiguous: the
data-derived expert edited **18/180** cells, moved **0.20%** of predictions,
and **never located the artifact** — because the artifact was the *most stable
term in training*.

**So the cognitive engine can never DERIVE the human's knowledge.** That door
is closed and we have the number.

But it can do the other thing: **remember it and transfer it.** *"A human
previously flagged features with this signature as logging artifacts — flag it
again here."* That is not derivation from data; it is **amortizing human
judgement across datasets**. And that is precisely what a library with
confidence, evidence, promotion and condition-matching is built to do.

> **The cognitive engine cannot learn to tune a model better. It may be able
> to learn to reuse human corrections. Nothing else in ML does that, because
> ordinary ML can only learn from the data — which is exactly where this
> information is not.**

---

## The first experiment (cheap, decisive, powerable from day one)

Reuse the `experiments/hitl_law_edit.py` harness. Law edits are ~1 second, so
this can be run at 180+ cells like the HITL arms were.

- **Arm A — no memory.** Each dataset gets its own human edit. This is the
  already-measured **+0.3141**, CI [+0.2984, +0.3298].
- **Arm B — memory.** The engine observes edits on datasets 1…k−1, then
  proposes edits on dataset k **with no human present**. Measure held-out
  accuracy under shift.
- **Arm C — floor.** A lookup table: "apply the last edit that worked." No
  graph, no confidence, no attention, no promotion.

**The question:** what fraction of A's +0.314 does B recover without a human,
and **does B beat C**?

### Falsification condition — set NOW, before running

> **If Arm B does not beat Arm C, the cognitive engine is not earning its
> complexity.** The library, graph, attention and confidence machinery are
> ceremony over a lookup table, and the lookup table ships.

This is the test the earlier Gate 2s never had. It is what makes this a real
rescue attempt rather than a hope. Per `results/rescore_gates/REPORT.md`,
decide the target effect and cell count *before* running, and pass `--json`.

---

## What to cut regardless

- **Formula engine** (evolving loss functions) — no obvious job in this
  framing; measured connected-but-useless.
- **Stress / regret as search signals** — repurpose the signal to
  edit-induced regression, or drop it.
- **PSO / the search loop** — separate matter, already settled: ~3900× one
  plain fit, a +0.01 benefit statistically excluded.

## What gets promoted

**The causal engine.** Currently the deadest component in the system (INERT in
both 40-stage cells) because it asks "does penalty cause regret" from ~40
trials.

Re-aimed it asks: **does this edit change behaviour on inputs it shouldn't?**
That is **locality** — the named open problem in the model-editing literature
(Yao et al. 2023, the reason ROME degrades after many sequential edits).
Control/treatment is exactly the right instrument, and every input is a free
trial.

It goes from the deadest component to the one addressing a problem the field
has not solved.

---

## Why this matters beyond UIL

This is the component that carries into KDN / Knowledge Cells. See
`LITSURVEY_CORRECTIONS.md`. In a deployed editable substrate the memory
argument gets *stronger*, not weaker: many edits over months is exactly the
data regime the miner never had, and edit locality is a question with abundant
trials and real stakes.
