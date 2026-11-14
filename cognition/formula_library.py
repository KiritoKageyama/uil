"""UIL Formula Library — Stores, promotes, decays, and retrieves formulas.

Like uil_library.py but for formulas. Tracks which formulas worked on which
dataset types, promotes successful ones, decays failed ones, and provides
the best formula for a new dataset based on similar past datasets.

V20.3: Companion to formula_engine.py.
"""

from __future__ import annotations
import json
import math
import os
import re
from typing import Any

import numpy as np

from figpso_uil.cognition.formula_engine import FormulaNode
from figpso_uil.cognition.paths import results_path
from figpso_uil.cognition.knowledge_archive import KnowledgeArchive
from figpso_uil.cognition.change_log import scoped_formula_changelog, utc_now
from figpso_uil.cognition.version_store import VersionStore


class FormulaLibrary:
    """Persistent storage for discovered formulas.
    
    Each formula entry:
    {
        "formula_id": "F-0012",
        "name": "focal_variant_1",
        "tree": {...},  # serialized FormulaNode
        "origin": "mutation_of_F-0003",
        "mutation_type": "constant_tweak",
        "confidence": 0.75,       # how reliably it works
        "evidence": 8,            # times tested
        "times_helped": 6,        # times it improved accuracy
        "times_hurt": 2,          # times it decreased accuracy
        "best_accuracy": 0.85,    # best accuracy achieved with this formula
        "dataset_types": [...],   # dataset fingerprints it worked on
        "status": "active",       # active, retired, seed
    }
    """
    
    # V20+.13: exploration/exploitation tuning for get_best_formula().
    # Without these, a fresh mutation (confidence=0.5, evidence=0) can
    # never out-rank an incumbent that's already accumulated evidence --
    # it would need to already be selected to prove itself, and would
    # need to already have proven itself to be selected. Bootstrap
    # guarantees every formula gets tried before confidence matters;
    # the UCB term then keeps low-evidence formulas competitive for a
    # while even past the bootstrap floor, instead of falling straight
    # back into permanent exploitation.
    MIN_EVIDENCE_BOOTSTRAP = 3
    EXPLORATION_C = 0.3
    DATASET_RELEVANCE_BONUS = 0.05

    # V20+.14: bootstrap-forced trials are real training runs under a
    # possibly-bad objective -- an unbounded mutation population means
    # an unbounded amount of forced exploration, and a feedback loop
    # (bad formula -> worse accuracy -> higher stress -> more mutations
    # -> more forced exploration) that can swamp a whole run. New
    # mutations enter as "incubating" (selectable/testable, like any
    # other under-tested formula, but not yet a permanent population
    # member) and are promoted into "active" or discarded once they've
    # cleared MIN_EVIDENCE_BOOTSTRAP -- never lingering in limbo.
    # MAX_INCUBATING caps how many can be mid-trial at once, which is
    # what actually throttles the birth rate (optimizer.py checks this
    # before calling add_formula); MAX_ACTIVE_POPULATION caps the
    # promoted pool itself, evicting the weakest active formula when a
    # promoted newcomer is better than it.
    MAX_INCUBATING = 5
    MAX_ACTIVE_POPULATION = 20
    PROMOTION_CONFIDENCE = 0.4

    def __init__(
        self,
        filepath: str = results_path("uil_formula_library.json"),
        version_store: VersionStore | None = None,
    ):
        self.filepath = filepath
        self._formulas: dict[str, dict] = {}
        # Scope the archive to THIS library's directory. Calling
        # KnowledgeArchive.shared() with no path pinned the module singleton
        # to the default results/ archive, so a library living under an
        # experiment's state_dir still archived into (and read from) the
        # shared one. Two supposedly isolated arms then contaminated each
        # other via prune/decay, which fire on a 5-stage cadence -- measured:
        # two identical isolated runs are byte-identical at 4 stages and
        # DIVERGE at 5. That silently coupled every ablation/A-B arm that ran
        # >= 5 stages. Invariant restored: state_dir isolation must be total.
        self._archive = KnowledgeArchive.shared(
            os.path.join(os.path.dirname(os.path.abspath(filepath)),
                         "uil_knowledge_archive_v20.json")
            if filepath else None
        )
        # Co-locate changelog with this library (bug #30b). Isolated
        # state_dir arms get a private instance; default results/ still
        # uses the process singleton so the web UI can read it.
        self._changelog = scoped_formula_changelog(filepath)
        self._version_store = version_store
        self._load()

    def _cell_store(self) -> VersionStore | None:
        if self._version_store is not None:
            return self._version_store
        try:
            return VersionStore.shared()
        except Exception:
            return None

    def _log_cell(self, entry: dict, op: str, cause: str | None = None) -> None:
        store = self._cell_store()
        if store is None:
            return
        store.record_cell(
            cell_id=entry.get("formula_id", "?"),
            cell_type="formula",
            op=op,
            post_state=dict(entry),
            cause=cause,
        )

    def canonical_state(self) -> dict[str, dict]:
        return {fid: dict(entry) for fid, entry in self._formulas.items()}

    def _touch_entry(self, entry: dict, change: str) -> None:
        now = utc_now()
        entry.setdefault("created_at", now)
        entry["updated_at"] = now
        entry["last_change"] = change

    def _log_formula(
        self,
        action: str,
        formula_id: str,
        *,
        dataset: str | None = None,
        details: dict | None = None,
        snapshot: dict | None = None,
    ) -> None:
        self._changelog.record(
            action,
            formula_id,
            dataset=dataset,
            details=details,
            snapshot=snapshot,
        )
    
    def seed_formulas(self, seeds: dict[str, FormulaNode]) -> None:
        """Add seed formulas (logistic, focal, etc.) if not already present."""
        for name, tree in seeds.items():
            seed_id = f"SEED-{name}"
            if seed_id not in self._formulas:
                now = utc_now()
                self._formulas[seed_id] = {
                    "formula_id": seed_id,
                    "name": name,
                    "tree": tree.to_dict(),
                    "origin": "seed",
                    "mutation_type": "none",
                    "confidence": 0.5,   # neutral — prove yourself
                    "evidence": 0,
                    "times_helped": 0,
                    "times_hurt": 0,
                    "best_accuracy": 0.0,
                    "dataset_types": [],
                    "status": "seed",
                    "times_used_as_parent": 0,
                    "created_at_stage": -1,  # present before the loop starts
                    "created_at": now,
                    "updated_at": now,
                    "last_change": "seeded",
                }
                self._log_formula("formula_seeded", seed_id, details={"name": name})
        self._save()

    def _next_formula_id(self) -> str:
        """Derive the next ID from the max ID already in THIS persisted
        library, rather than a per-instance counter. A fresh FormulaEngine
        is constructed for every optimizer run (core/optimizer.py), each
        starting its own counter at 0 -- two runs sharing the same library
        file would both mint "F-0001", and whichever saved last would
        silently overwrite the other's entry (confirmed empirically: the
        same library file held two different trees under "F-0001" after
        two separate runs). Scanning the loaded library itself is
        immune to that because it reflects every entry actually on disk,
        regardless of which process/instance created them."""
        existing = [
            int(m.group(1))
            for fid in self._formulas
            if (m := re.match(r"F-(\d+)$", fid))
        ]
        next_n = max(existing, default=0) + 1
        return f"F-{next_n:04d}"

    def add_formula(
        self,
        name: str,
        tree: FormulaNode,
        origin: str,
        mutation_type: str,
        created_at_stage: int | None = None,
        status: str = "incubating",
    ) -> str:
        """Add a newly discovered formula to the library.

        Default status is "incubating" (see MAX_INCUBATING) -- it must
        prove itself via record_result() before becoming a permanent
        "active" population member. Research step 4 may pass
        status="active" when a held-out utility test already admitted
        the formula (DreamCoder-style gate), skipping the deadlocked
        incubating→never-selected path under sparse_linear.

        Returns the newly assigned formula_id.

        created_at_stage: the optimizer stage this mutation was born on
        (optimizer.py passes the loop's `stage`) -- lets diagnostics
        measure a lineage's true stage-lifespan, not just mutation-order.
        None for callers that don't track stages (e.g. unit tests)."""
        if status not in ("incubating", "active", "seed"):
            status = "incubating"
        formula_id = self._next_formula_id()
        now = utc_now()
        self._formulas[formula_id] = {
            "formula_id": formula_id,
            "name": name,
            "tree": tree.to_dict(),
            "origin": origin,
            "mutation_type": mutation_type,
            "confidence": 0.5 if status != "active" else 0.55,
            "evidence": 0 if status != "active" else 1,
            "times_helped": 0,
            "times_hurt": 0,
            "best_accuracy": 0.0,
            "dataset_types": [],
            "status": status,
            "times_used_as_parent": 0,
            "created_at_stage": created_at_stage,
            "created_at": now,
            "updated_at": now,
            "last_change": "added" if status != "active" else "utility_admitted",
        }
        self._log_formula(
            "formula_added",
            formula_id,
            details={
                "name": name,
                "origin": origin,
                "mutation_type": mutation_type,
                "created_at_stage": created_at_stage,
                "status": status,
            },
        )
        self._log_cell(self._formulas[formula_id], "create", cause="add_formula")
        self._save()
        return formula_id

    def get_incubating_formulas(self) -> list[dict]:
        """Formulas still in their mandatory trial period (see
        MAX_INCUBATING). Callers use this to throttle how many new
        mutations get created -- the birth rate, not just the population
        cap, is what stops bootstrap-forced exploration from growing
        without bound."""
        return [
            f for f in self._formulas.values() if f["status"] == "incubating"
        ]
    
    def record_result(
        self,
        formula_id: str,
        accuracy: float,
        helped: bool,
        dataset_fingerprint: str | None = None,
    ) -> None:
        """Record the result of using a formula."""
        if formula_id not in self._formulas:
            return
        
        entry = self._formulas[formula_id]
        # Heal legacy/partial entries: formulas persisted before these
        # counters existed (or minted by an external path) may lack them.
        entry.setdefault("evidence", 0)
        entry.setdefault("times_helped", 0)
        entry.setdefault("times_hurt", 0)
        entry.setdefault("best_accuracy", 0.0)
        entry.setdefault("dataset_types", [])
        prev_status = entry.get("status")
        entry["evidence"] += 1

        if helped:
            entry["times_helped"] += 1
        else:
            entry["times_hurt"] += 1

        if accuracy > entry["best_accuracy"]:
            entry["best_accuracy"] = accuracy

        if dataset_fingerprint and dataset_fingerprint not in entry["dataset_types"]:
            entry["dataset_types"].append(dataset_fingerprint)

        if dataset_fingerprint:
            self._archive.record_formula_trial(
                dataset_fingerprint,
                formula_id,
                accuracy,
                helped,
                formula_name=entry.get("name"),
            )
        
        # Update confidence. Raw times_helped/total treats 3/3 as a
        # stronger signal than 88/100 -- confirmed harmful empirically
        # (it let a formula with 3 trials permanently outrank one with
        # far more evidence). The Wilson score lower bound discounts
        # low-evidence success rates toward 0.5 instead of letting them
        # sit at the raw (often 1.0) ratio.
        total = entry["times_helped"] + entry["times_hurt"]
        if total > 0:
            entry["confidence"] = self._wilson_lower_bound(
                entry["times_helped"], total
            )

        # Retire formulas that consistently fail
        if entry["evidence"] >= 10 and entry["confidence"] < 0.3:
            entry["status"] = "retired"
            self._touch_entry(entry, "retired_low_confidence")
            self._archive_retired_entry(entry, dataset_fingerprint, "low_confidence")

        # V20+.14: incubation is over once it's cleared its mandatory
        # trial period -- promote (possibly evicting the weakest active
        # formula) or discard. Never left to linger as "incubating"
        # forever, which would otherwise keep forcing it through
        # bootstrap selection indefinitely.
        if (
            entry["status"] == "incubating"
            and entry["evidence"] >= self.MIN_EVIDENCE_BOOTSTRAP
        ):
            self._resolve_incubation(entry)

        self._touch_entry(entry, "result_recorded")
        self._log_formula(
            "formula_result",
            formula_id,
            dataset=dataset_fingerprint,
            details={
                "accuracy": accuracy,
                "helped": helped,
                "evidence": entry["evidence"],
                "confidence": entry.get("confidence"),
                "status": entry.get("status"),
                "status_changed": prev_status != entry.get("status"),
            },
        )
        op = "decay" if entry.get("status") == "retired" else "mutate"
        self._log_cell(entry, op, cause="record_result")
        self._save()

    def _archive_retired_entry(
        self,
        entry: dict,
        dataset_fingerprint: str | None,
        reason: str,
    ) -> None:
        """Move full formula to archive; leave a tombstone in working library."""
        if entry.get("status") == "archived":
            return
        full = dict(entry)
        self._archive.archive_formula(
            full,
            reason=reason,
            dataset=dataset_fingerprint,
        )
        fid = full["formula_id"]
        summary = ""
        if dataset_fingerprint:
            rec = self._archive._dataset_bucket(dataset_fingerprint).get(
                "formula_exploration", {},
            ).get(fid, {})
            summary = rec.get("summary", "")
        tombstone = {
            "formula_id": fid,
            "name": full.get("name", fid),
            "status": "archived",
            "best_accuracy": full.get("best_accuracy", 0.0),
            "confidence": full.get("confidence", 0.0),
            "evidence": full.get("evidence", 0),
            "archive_summary": summary,
            "origin": full.get("origin", "unknown"),
            "created_at": full.get("created_at", utc_now()),
            "updated_at": utc_now(),
            "last_change": f"archived:{reason}",
        }
        self._formulas[fid] = tombstone
        self._log_formula(
            "formula_archived",
            fid,
            dataset=dataset_fingerprint,
            details={"reason": reason},
            snapshot=full,
        )

    def _resolve_incubation(self, entry: dict) -> None:
        """Promote an incubated formula into the active population, or
        discard it -- see MAX_ACTIVE_POPULATION/PROMOTION_CONFIDENCE."""
        dataset = entry.get("dataset_types", [None])[-1] if entry.get("dataset_types") else None
        if entry["confidence"] < self.PROMOTION_CONFIDENCE:
            entry["status"] = "retired"
            self._touch_entry(entry, "incubation_failed")
            self._archive_retired_entry(entry, dataset, "incubation_failed")
            return

        active_pop = [
            f for f in self._formulas.values() if f["status"] == "active"
        ]
        if len(active_pop) < self.MAX_ACTIVE_POPULATION:
            entry["status"] = "active"
            self._touch_entry(entry, "promoted")
            self._log_formula("formula_promoted", entry["formula_id"], dataset=dataset)
            return

        weakest = min(active_pop, key=lambda f: f["confidence"])
        if entry["confidence"] > weakest["confidence"]:
            weakest["status"] = "retired"
            self._touch_entry(weakest, "evicted")
            self._archive_retired_entry(weakest, dataset, "evicted_for_promotion")
            entry["status"] = "active"
            self._touch_entry(entry, "promoted")
            self._log_formula(
                "formula_promoted",
                entry["formula_id"],
                dataset=dataset,
                details={"evicted": weakest["formula_id"]},
            )
        else:
            entry["status"] = "retired"
            self._touch_entry(entry, "incubation_outcompeted")
            self._archive_retired_entry(entry, dataset, "incubation_outcompeted")

    def prune_retired_to_archive(self, dataset_fingerprint: str | None = None) -> int:
        """Ensure retired formulas are archived and stripped from working pool."""
        pruned = 0
        for entry in list(self._formulas.values()):
            if entry.get("status") == "retired" and "tree" in entry:
                self._archive_retired_entry(entry, dataset_fingerprint, "prune_pass")
                pruned += 1
        if pruned:
            self._archive.save()
            self._save()
            self._changelog.save()
        return pruned

    # Working-pool cap for the mutated/evolved population. Keep the LEARNING
    # (seeds, proven high-confidence/evidence formulas, incubating ones still
    # under trial); archive the weakest active mutants beyond the cap. The
    # full formula + its outcome summary live on in the KnowledgeArchive
    # (queryable), and a tombstone stays in the working pool -- nothing is
    # lost, the working set just stays lean as evolution accelerates (b).
    MAX_WORKING_ACTIVE = 60

    def prune_to_cap(
        self,
        dataset_fingerprint: str | None = None,
        max_active: int | None = None,
    ) -> int:
        """Archive the weakest active mutants beyond the working cap."""
        cap = self.MAX_WORKING_ACTIVE if max_active is None else int(max_active)
        # Only ordinary evolved/active formulas are eligible for eviction.
        active = [
            f for f in self._formulas.values()
            if f.get("status") == "active" and "tree" in f
        ]
        if len(active) <= cap:
            return 0
        # Rank by proven value: confidence, then evidence. Weakest first.
        active.sort(key=lambda f: (
            float(f.get("confidence", 0.0)),
            int(f.get("evidence", 0)),
        ))
        drop = active[: len(active) - cap]
        pruned = 0
        for entry in drop:
            self._archive_retired_entry(entry, dataset_fingerprint, "working_cap")
            pruned += 1
        if pruned:
            self._archive.save()
            self._save()
            self._changelog.save()
        return pruned
    
    def get_best_formula(
        self,
        dataset_fingerprint: str | None = None,
    ) -> dict | None:
        """Get the best formula to try next.

        Two phases, evaluated over the WHOLE active/seed pool (not just
        ones already tied to this dataset -- a never-tried formula by
        definition has no dataset_types yet, so restricting to a
        dataset-matched subset first would make it permanently
        invisible the moment ANY formula records a result for this
        dataset):

        1. Bootstrap: any formula with < MIN_EVIDENCE_BOOTSTRAP evidence
           is tried regardless of confidence, least-tested first. This
           is what makes formula evolution an actual select step instead
           of a leaderboard a fresh mutation can never join.
        2. Mature: rank by confidence + a UCB1-style exploration bonus
           (large when evidence is low relative to total library
           evidence, shrinking as it accumulates) + a small relevance
           bonus for formulas already proven on THIS dataset.
        """
        active = [
            f for f in self._formulas.values()
            if f["status"] in ("active", "seed", "incubating")
        ]
        if not active:
            return None

        under_tested = [
            f for f in active if f["evidence"] < self.MIN_EVIDENCE_BOOTSTRAP
        ]
        if under_tested:
            under_tested.sort(key=lambda f: (f["evidence"], f["formula_id"]))
            return under_tested[0]

        total_evidence = sum(f["evidence"] for f in active) or 1

        def score(f: dict) -> float:
            exploration = self.EXPLORATION_C * math.sqrt(
                math.log(total_evidence + 1) / (f["evidence"] + 1)
            )
            relevance = (
                self.DATASET_RELEVANCE_BONUS
                if dataset_fingerprint
                and dataset_fingerprint in f.get("dataset_types", [])
                else 0.0
            )
            return f["confidence"] + exploration + relevance

        return max(active, key=score)

    # V20+.15: mutation-parent selection. get_best_formula() above always
    # returns ONE winner -- using it as the mutation base too means every
    # mutation extends the literal champion's lineage forever (confirmed
    # empirically: 49 of 53 formulas in one run traced back through a
    # single chain off SEED-squared; the other 3 seeds were never once
    # selected as a parent). Tournament selection samples a random subset
    # and mutates only the fittest of that subset, so every formula in
    # the pool gets an occasional chance to become a parent without
    # abandoning selection pressure toward quality.
    #
    # V20+.17: tournament selection ALONE didn't fix it -- confirmed by
    # the Phase A diagnostics (experiments/formula_diagnostics.py): once
    # one lineage outnumbers the others (e.g. 92 hinge vs 1 logistic),
    # an individual-uniform draw of k=4 almost never even contains the
    # rare lineage (P(miss) = (99/100)^4 ~= 96% in that example), so its
    # fitness is never compared. The fix is WHERE candidates come from,
    # not the fitness math: stratify by root lineage so every lineage
    # gets a fair shot at entering the tournament regardless of how many
    # descendants it currently has, then let fitness decide the winner
    # as before. Stratification strength decays with `progress` (0 at
    # the start of a run, 1 at the end) so a lineage that's genuinely
    # bad can still go extinct given enough evidence -- this isn't a
    # permanent quota, just protection against population-frequency
    # bias swamping fitness comparison while a lineage is still young.
    PARENT_EVIDENCE_SATURATION = 10.0
    MIN_STRATIFICATION_STRENGTH = 0.15

    def _root_lineage(self, formula_id: str) -> str:
        """Walk the origin chain back to the seed formula this formula_id
        ultimately descends from. Walks the FULL registry (not just the
        active pool) since an ancestor may have since retired."""
        seen: set[str] = set()
        current = formula_id
        while current not in seen:
            seen.add(current)
            entry = self._formulas.get(current)
            if entry is None:
                return current
            origin = entry.get("origin", "seed")
            if origin == "seed" or not origin.startswith("mutation_of_"):
                return current
            parent_id = origin[len("mutation_of_"):]
            if parent_id not in self._formulas:
                return current
            current = parent_id
        return current  # cycle guard

    def select_mutation_parent(
        self, rng: np.random.Generator, k: int = 4, progress: float = 0.0
    ) -> dict | None:
        """Stratified-tournament-select a mutation parent from the
        active/seed/incubating pool.

        Candidate assembly (stratified by lineage, NOT individual-uniform
        -- see V20+.17 comment above): `round(k * strength)` slots are
        filled with one representative drawn from each of that many
        distinct random lineages; any remaining slots are filled by
        ordinary individual-uniform sampling from the whole pool (which
        is naturally population-weighted, giving dominant/proven
        lineages extra weight as `progress` grows and `strength` decays).

        Composite fitness (not confidence alone) decides the winner once
        the tournament is assembled, exactly as before -- a formula
        that's reliable but rarely chosen, or one with deep evidence but
        only middling confidence, still has a real shot:

            fitness = 0.5 * confidence
                    + 0.3 * evidence_score   (evidence, saturating at 10)
                    + 0.2 * diversity_bonus  (decays the more often this
                                               formula has already been
                                               picked as a parent)

        progress: 0.0 (start of run) to 1.0 (end of run). Controls how
        strongly lineage representation is protected vs. left to pure
        population-weighted sampling.
        """
        pool = [
            f for f in self._formulas.values()
            if f["status"] in ("active", "seed", "incubating")
        ]
        if not pool:
            return None

        k = min(k, len(pool))
        strength = max(self.MIN_STRATIFICATION_STRENGTH, 1.0 - np.clip(progress, 0.0, 1.0))
        n_stratified = round(k * strength)

        lineages: dict[str, list[dict]] = {}
        for f in pool:
            lineages.setdefault(self._root_lineage(f["formula_id"]), []).append(f)
        lineage_keys = list(lineages.keys())

        tournament: list[dict] = []
        if n_stratified > 0 and lineage_keys:
            order = rng.permutation(len(lineage_keys))
            for i in range(n_stratified):
                root = lineage_keys[order[i % len(lineage_keys)]]
                candidates = lineages[root]
                rep = candidates[rng.integers(len(candidates))]
                tournament.append(rep)

        remaining = k - len(tournament)
        if remaining > 0:
            for i in rng.choice(len(pool), size=remaining, replace=False):
                tournament.append(pool[i])

        def fitness(f: dict) -> float:
            evidence_score = min(1.0, f["evidence"] / self.PARENT_EVIDENCE_SATURATION)
            diversity_bonus = 1.0 / (1.0 + f.get("times_used_as_parent", 0))
            return 0.5 * f["confidence"] + 0.3 * evidence_score + 0.2 * diversity_bonus

        winner = max(tournament, key=fitness)
        winner["times_used_as_parent"] = winner.get("times_used_as_parent", 0) + 1
        self._save()
        return winner

    def select_tournament_candidates(
        self,
        rng: np.random.Generator,
        k: int = 6,
        progress: float = 0.0,
    ) -> list[dict]:
        """Stratified sample of formulas for fit-time tournament evaluation.

        Same lineage-fair candidate assembly as select_mutation_parent(), but
        returns the full tournament roster instead of picking one winner by
        library metadata alone.
        """
        pool = [
            f for f in self._formulas.values()
            if f["status"] in ("active", "seed", "incubating")
        ]
        if not pool:
            return []

        k = min(k, len(pool))
        strength = max(self.MIN_STRATIFICATION_STRENGTH, 1.0 - np.clip(progress, 0.0, 1.0))
        n_stratified = round(k * strength)

        lineages: dict[str, list[dict]] = {}
        for f in pool:
            lineages.setdefault(self._root_lineage(f["formula_id"]), []).append(f)
        lineage_keys = list(lineages.keys())

        tournament: list[dict] = []
        seen_ids: set[str] = set()

        def _add(entry: dict) -> None:
            fid = entry["formula_id"]
            if fid not in seen_ids:
                seen_ids.add(fid)
                tournament.append(entry)

        if n_stratified > 0 and lineage_keys:
            order = rng.permutation(len(lineage_keys))
            for i in range(n_stratified):
                root = lineage_keys[order[i % len(lineage_keys)]]
                candidates = lineages[root]
                _add(candidates[rng.integers(len(candidates))])

        remaining = k - len(tournament)
        if remaining > 0:
            available = [f for f in pool if f["formula_id"] not in seen_ids]
            if not available:
                available = pool
            picks = rng.choice(
                len(available),
                size=min(remaining, len(available)),
                replace=False,
            )
            for i in picks:
                _add(available[int(i)])

        return tournament

    @staticmethod
    def _wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
        """95%-confidence lower bound on the true success rate. Discounts
        low-evidence ratios toward 0.5 instead of trusting them at face
        value (3/3 -> ~0.44, not 1.0; 88/100 -> ~0.80, not 0.88)."""
        if total == 0:
            return 0.0
        phat = successes / total
        denom = 1.0 + z * z / total
        center = phat + z * z / (2 * total)
        margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
        return (center - margin) / denom

    def get_formula_tree(self, formula_id: str) -> FormulaNode | None:
        """Deserialize and return a formula tree by ID."""
        entry = self._formulas.get(formula_id)
        if entry is None or entry.get("status") == "archived":
            return None
        tree_data = entry.get("tree")
        if not tree_data:
            return None
        return FormulaNode.from_dict(tree_data)

    def get_archived_summary(self, formula_id: str) -> str:
        entry = self._formulas.get(formula_id) or self._archive._data.get(
            "formulas", {},
        ).get(formula_id, {})
        return entry.get("archive_summary") or entry.get("_archive_reason", "")

    def is_explored_low(self, formula_id: str, dataset: str) -> bool:
        explor = self._archive._dataset_bucket(dataset).get("formula_exploration", {})
        rec = explor.get(formula_id)
        return bool(rec and rec.get("verdict") == "low")
    
    def get_active_formulas(self) -> list[dict]:
        """Get all active formulas (for mutation source selection)."""
        return [
            f for f in self._formulas.values()
            if f["status"] in ("active", "seed")
        ]
    
    def get_all(self) -> list[dict]:
        """Get all formulas."""
        return list(self._formulas.values())
    
    def formula_count(self) -> int:
        return len(self._formulas)
    
    # ── Internal ─────────────────────────────────────────────────────
    
    def _load(self) -> None:
        """Load formula library from disk."""
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
                self._formulas = {e["formula_id"]: e for e in data}
                for entry in self._formulas.values():
                    if "created_at" not in entry:
                        entry["created_at"] = utc_now()
                    if "updated_at" not in entry:
                        entry["updated_at"] = entry.get("created_at", utc_now())
        except (FileNotFoundError, json.JSONDecodeError):
            self._formulas = {}

    def _save(self) -> None:
        """Save formula library to disk."""
        dirpath = os.path.dirname(self.filepath)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        with open(self.filepath, "w") as f:
            json.dump(list(self._formulas.values()), f, indent=2, default=str)
        self._changelog.save()
