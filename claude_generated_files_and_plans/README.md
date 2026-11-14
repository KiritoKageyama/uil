# claude_generated_files_and_plans

Created 2026-08-30 while tidying the project root. Everything here was
previously loose at `figpso_uil/`. **Nothing was deleted and no Python
module was moved** — this folder holds documents only.

Some of these are load-bearing working documents, not archive. In
particular `analysis/COGNITIVE_ENGINE_RESCUE.md` is the project's
current NEXT ACTION, and `analysis/RESEARCH_FAILURE_ANALYSIS.md` +
`analysis/HARDCODED_VALUE_AUDIT_QA.md` are both "read before designing
the next fix" per CLAUDE.md. CLAUDE.md's pointers were updated to these
paths in the same pass.

## Layout

### `analysis/` — AI-authored analysis and audit documents
| file | what it is |
|---|---|
| `COGNITIVE_ENGINE_RESCUE.md` | **NEXT ACTION.** Design for re-pointing the engine at human edits. Nothing measured yet. |
| `RESEARCH_FAILURE_ANALYSIS.md` | Maps every measured UIL failure to the published research that names it + its remedy. |
| `HARDCODED_VALUE_AUDIT_QA.md` | Inventory of ~584 hardcoded constants across 54 files, sorted into 6 categories. |
| `LITSURVEY_CORRECTIONS.md` | Drafted corrections to the KDN literature survey (not applied). |
| `DAPTIVE_KNOWLEDGE_NETWORK_DESIGN.md` | KDN design doc; §13 carries the transformer/KDN addendum. |

### `research_docs/` — surveys, briefs, strategy
| file | note |
|---|---|
| `LITERATURE_SURVEY.md` | Cited by `experiments/eval_extrapolation.py` comments (line 375 reference). |
| `LITERATURE_SURVEY_PAPERS.md` | Paper list. |
| `PROFESSOR_PROJECT_BRIEF.md` | |
| `PROFESSOR_PROJECT_BRIEF_V2.md` | symlink → `Documents/Codex/2026-07-26/.../outputs/` |
| `uil_ip_strategy.md` | **NOT the original — a recovered outline.** The real file was destroyed 2026-08-02 (see below). Headings recovered from the graphify index; prose is gone. |

### `documents/` — standalone document exports
`Universal Intelligence Layer.docx`, `UIL System Refinement.docx`,
`DAPTIVE.docx` (symlink → OneDrive Desktop).

### `plans/` — **copies** of Claude Code plan files for this project
Originals remain in `~/.claude/plans/` and were not touched.
| file | original slug |
|---|---|
| `synchronous-watching-seal.md` | `synchronous-watching-seal.md` — the academic_v2 audit + the "Workstream 2" transformer/KDN design CLAUDE.md's addendum points at. |
| `06-transparent-substrate-litsurvey-plan.md` | `c-users-arinb-onedrive-desktop-06-trans-enumerated-sphinx.md` — plan for `academic_v2/06_TRANSPARENT_EDITABLE_NEURAL_SUBSTRATE_LITSURVEY.md`. |

The other four plans in `~/.claude/plans/` belong to different projects
(Research Agent, ECIS, the Zotero plugin) and were left alone.

### `unrelated_lca/` — the LCA documents
`agent_architecture.md`, `lca_uil_analysis.md`. CLAUDE.md states these
describe a separate Layered Cognitive Agent for UI automation:
*"Unrelated to UIL. Do not integrate."* They are still cited **by name**
in comments in `cognition/goal_cell.py`, `llm_mutator.py`,
`neural_proposer.py` — comments only, no file I/O, so nothing broke.

### `scratch/` — throwaway leftovers
`new.text` (an AutoML-vs-UIL critique transcript), plus three test
artifacts pulled out of `academic_v2/`: `_wtest.md` (`# hi`),
`_build_doc.py` (a stub that only prints a path), `_parts_t1.md`
(was `_parts/t1.md`, "Hello world test markdown").

## Moves that did NOT land here

| what | where it went | why |
|---|---|---|
| `01_…docx` … `05_…docx` | `academic_v2/` | 1:1 name match with the `.md` files already there — they are the built versions. |
| `_recover_site.py` | `_recovery/` | It only ever read/wrote `_recovery/`. Its `OUT = ROOT / "_recovery"` was changed to `OUT = ROOT` so it still writes to the same place from its new location. |
| `tests/uil_formula_changelog_v20.json` | `results/_leaked_from_tests/` | State leaked into `tests/` by a run with no `state_dir` (bug #30b class). `results/uil_formula_changelog_v20.json` already exists and is newer, so it was quarantined rather than merged. |

## Undo

`bash claude_generated_files_and_plans/UNDO_move.sh` restores every path
exactly, including the `_recover_site.py` edit.

## Symlink dereference, 2026-08-30

Three files in this project were symlinks into ephemeral caches, created by the
same 2026-08-02 duplicate-file pass that scattered the rest. All three are fixed
and **`UNDO_move.sh` deliberately does not restore them** -- restoring a link
into an editor undo buffer would be reintroducing the bug.

| file | had pointed at | outcome |
|---|---|---|
| `cognition/investigation_engine.py` | Cursor's *undo-history snapshot of that same file* (`User/History/2b8e0118/Duj5.py`) | Now a real 18,630-byte file, sha256 verified identical. It is imported at `core/optimizer.py:37`; `tests/test_investigation_engine.py` passes. |
| `full.txt` | Antigravity IDE's history (`User/History/77c7b38c/emdX.txt`) | Now a real 226,234-byte file (14,464 lines), sha256 verified identical. |
| `uil_ip_strategy.md` | A WhatsApp Desktop transfers folder, since cleared | **Unrecoverable.** Both known copies were links to the same dead path. Replaced with a recovered outline (18 headings + line numbers) reconstructed from `graphify-out/`, which indexed the file on 2026-07-24. |

The 23 remaining symlinks all point at real, durable locations: 16 into the
sibling `figpso_uil_ai/` and `figpso_uil_human/` workspaces, 6 into OneDrive,
1 into `Documents/Codex/`. Those were left alone.

Still dangling **outside** this project: `C:/Users/arinb/Downloads/uil_ip_strategy.md`,
a second dead link to the same WhatsApp path. Not touched -- it is outside the
project root.
