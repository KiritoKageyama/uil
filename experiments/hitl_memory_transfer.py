"""Arms B and C of the cognitive-engine rescue: can the engine REMEMBER a
human correction and apply it to a dataset it has never seen?

WHY THIS EXPERIMENT EXISTS
--------------------------
`hitl_law_edit.py` established Arm A: a human who supplies knowledge the
training data does not contain is worth +0.3141 accuracy under distribution
shift (95% CI [+0.2984, +0.3298], 145W/0T/2L), while the same expert
reasoning from the training data alone is NULL (-0.0016, 180 cells). Its
control settled the mechanism: on the SAME shifted data a data-derived expert
edited 18/180 cells and found the artifact ZERO times, because the artifact is
the most stable term in training. No resampling, CV or regularization can see
it.

That leaves the question the rescue document (COGNITIVE_ENGINE_RESCUE.md) is
built on. The engine cannot DERIVE human knowledge -- that is settled and it
is not what is being tested here. It might be able to REMEMBER and TRANSFER
it. So:

  A  human present every time                (measured: +0.3141)
  B  memory of past corrections, NO human    (this file)
  C  exact-match lookup table, NO human      (this file -- the floor)

FALSIFICATION, SET IN ADVANCE (rescue doc, and unchanged here):
    If B does not beat C, ship the lookup table.

WHAT MAKES B A REAL TEST AND NOT A RIGGED ONE
---------------------------------------------
The memory is evaluated LEAVE-ONE-DATASET-OUT. To be scored on `sonar`, the
memory may only have seen corrections made on the other five datasets. If it
were allowed to remember `sonar` itself it would be a lookup table with extra
steps, which is precisely arm C.

Under LODO the lookup table can never fire, so C degenerates to "no edit".
That is not C being strawmanned -- it is the lookup table's actual limitation,
and it is the whole reason to ask whether B is worth building. It also means
B is NOT scored against a pushover: B must beat DOING NOTHING on data it has
never seen, and a memory that misfires is strictly worse than no memory. C+
below reports what the lookup table achieves when its assumption holds.

THE CONFOUND THIS FILE MUST RULE OUT
------------------------------------
There is a degenerate strategy that would ace the shift arm and be worthless:
"always delete the single most dominant term". It would find the planted
artifact every time and would also gut every healthy model. So B is scored on
BOTH shifted and iid cells. On iid data there is no artifact and the correct
behaviour is to do nothing; a B that edits there is a memory that has learned
a reflex, not a correction. Reported as B's specificity, and a B that fails it
should not be shipped whatever its shift number says.

NEGATIVE EXAMPLES -- the finding that decided the design
--------------------------------------------------------
The first build trained the memory only on corrections, i.e. only on laws
that DID contain an artifact. It scored +0.2292 against the lookup table and
FAILED specificity: it fired on 67% of healthy iid models and cost -0.0534
accuracy there. The coefficient dump said why -- `dominance_ratio` +1.58 and
`is_top` +0.86 swamped `standalone_auc` +0.29. Having never seen a healthy
law, it could not learn an absolute threshold, so it learned the rank reflex
"delete the top term", which is right whenever an artifact is present and
wrong the rest of the time.

The memory therefore also records reviews that ended in NO edit. That is not
a concession to make the numbers work: a deployed system has far more of
those than of corrections, and the lookup table cannot use them at all --
there is no feature key to store when nothing was deleted. Run with
`--no-negatives` to reproduce the reflex.

SCOPE, STATED NOT HIDDEN
------------------------
* The "human" in the memory's training data is still the oracle program from
  Arm A. This measures whether corrections TRANSFER, never whether real humans
  produce good ones. That still needs people.
* The planted artifact is synthetic and deliberately extreme (its standalone
  train AUC is ~0.98, far above any genuine column in these datasets). So B's
  margin here reflects how separable THIS artifact is. The per-term descriptor
  table is printed precisely so that margin is visible rather than implied.
* Binary datasets only, self-contained/legible law only -- same scope as the
  parent file, for the same reasons.

Run:
    python -m figpso_uil.experiments.hitl_memory_transfer
    python -m figpso_uil.experiments.hitl_memory_transfer --seeds 30 --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from figpso_uil.adapters.symbolic_formula_adapter import SymbolicFormulaAdapter
from figpso_uil.cognition.law_export import _term_values, evaluate_law, export_law
from figpso_uil.experiments.hitl_law_edit import (
    DATASETS, N_BOOTSTRAP, _design_matrix, _load, apply_edit,
    expert_drop_feature,
)
from figpso_uil.experiments.rescore_gates import analyze

#: Memory fires on a term when its predicted "a human would delete this"
#: probability clears this. Deliberately high: a false deletion costs real
#: accuracy, and the iid specificity check below is what this protects.
FIRE_THRESHOLD = 0.90

DESCRIPTOR_NAMES = (
    "logit_share",       # share of the law's total logit spread this term carries
    "dominance_ratio",   # its share / the next largest share
    "is_top",            # 1.0 if it is the largest
    "standalone_auc",    # train AUC of this term ALONE
    "auc_gap",           # standalone_auc minus the mean of the other terms
    "sign_stability",    # bootstrap sign agreement (the iid expert's signal)
    "n_terms_scaled",    # law size, context for the above
)


# --------------------------------------------------------------------------
# the cell: fit once, share across every arm so the comparison is exactly paired
# --------------------------------------------------------------------------

def build_cell(name: str, seed: int, shifted: bool) -> dict | None:
    """Fit the model and export its law. Deterministic for (name, seed).

    Every arm scores the SAME law on the SAME split, so arm differences are
    the edit and nothing else.
    """
    X, y = _load(name)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)

    spurious_idx = None
    if shifted:
        # Strongly predictive in training, pure noise at test time -- a leaked
        # label, a logging artifact, a sensor due for recalibration. Identical
        # construction to hitl_law_edit.one_cell so the arms stay comparable.
        rng = np.random.default_rng(seed + 9001)
        s_tr = (ytr * 2.0 - 1.0) + rng.normal(0, 0.35, size=len(ytr))
        s_te = rng.normal(0, 1.0, size=len(yte))
        Xtr = np.column_stack([Xtr, s_tr])
        Xte = np.column_stack([Xte, s_te])
        spurious_idx = Xtr.shape[1] - 1

    sd = tempfile.mkdtemp(prefix="hitlmem_")
    clf = SymbolicFormulaAdapter(
        seed=seed, num_classes=2, learning_mode="sparse_linear",
        formula_library_path=os.path.join(sd, "lib.json"))
    clf.use_symbolic_basis = False   # self-contained + legible law only
    model = clf.fit_weighted(Xtr, ytr, np.ones(Xtr.shape[1]))
    try:
        law = export_law(model)
    except Exception:
        return None
    if law["kind"] != "binary":
        return None

    base_pred = evaluate_law(law, Xte)
    return {
        "dataset": name, "seed": seed, "shifted": bool(shifted),
        "law": law, "Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte,
        "spurious_idx": spurious_idx,
        "base_pred": base_pred,
        "base_acc": float(np.mean(base_pred == yte)),
        "oracle_drop": (expert_drop_feature(law, spurious_idx)
                        if spurious_idx is not None else []),
    }


# --------------------------------------------------------------------------
# what a memory is allowed to see about a term
# --------------------------------------------------------------------------

def term_descriptors(law: dict, Xtr: np.ndarray, ytr: np.ndarray,
                     seed: int) -> np.ndarray:
    """One descriptor row per term in the law.

    TRAIN DATA AND THE LAW ONLY. Nothing here touches the held-out split --
    that is the same discipline bug #19 enforces for normalization, and it is
    what makes the result mean anything.

    These are the properties a correction can generalize OVER. A memory keyed
    on "column 30 of breast_cancer" transfers nowhere; a memory keyed on "a
    term carrying 80% of the logit whose standalone AUC is 0.98" can fire on a
    dataset it has never seen.
    """
    B = _design_matrix(law, Xtr)          # n x k, the law's own terms
    terms = law["heads"][0]["terms"]
    k = B.shape[1]
    if k == 0:
        return np.zeros((0, len(DESCRIPTOR_NAMES)))

    coefs = np.array([float(t.get("coef", 0.0)) for t in terms])
    # Contribution spread: |coef| * std(term). A term with a big coefficient on
    # a constant column moves nothing, so std is the honest weighting.
    spread = np.abs(coefs) * (B.std(axis=0) + 1e-12)
    total = float(spread.sum()) + 1e-12
    share = spread / total

    order = np.argsort(share)[::-1]
    second = float(share[order[1]]) if k > 1 else 1e-9
    top_idx = int(order[0])

    # Standalone train AUC of each term on its own.
    y_bin = (np.asarray(ytr) == law["heads"][0].get("positive_class", 1)).astype(int)
    aucs = np.array([_auc(B[:, j], y_bin) for j in range(k)])
    # "How much better than its peers" -- leave-one-out mean, so a lone strong
    # term is not compared against itself.
    peer_mean = np.array([
        float(np.mean(np.delete(aucs, j))) if k > 1 else 0.5 for j in range(k)])
    auc_gap = aucs - peer_mean

    stability = _sign_stability(B, ytr, seed)

    rows = np.column_stack([
        share,
        share / max(second, 1e-9),
        (np.arange(k) == top_idx).astype(float),
        aucs,
        auc_gap,
        stability,
        np.full(k, min(k, 40) / 40.0),
    ])
    return rows


def _auc(col: np.ndarray, y_bin: np.ndarray) -> float:
    """Rank AUC, direction-free (|auc - .5| folded back up).

    Direction-free because the law's own coefficient already carries the sign;
    what the memory needs to know is how much SIGNAL the term has, not which
    way it points.
    """
    pos, neg = col[y_bin == 1], col[y_bin == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    ranks = np.argsort(np.argsort(col)) + 1.0
    r_pos = ranks[y_bin == 1].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(max(auc, 1.0 - auc))


def _sign_stability(B: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Bootstrap sign agreement per term -- the signal the iid expert uses.

    Included precisely because it is the one the shift control PROVED cannot
    find the artifact (the artifact is maximally stable). If the memory ends
    up leaning on this descriptor, that is visible in the coefficient dump and
    would be evidence the memory is re-deriving rather than remembering.
    """
    n, k = B.shape
    if n < 20 or k == 0:
        return np.full(k, 0.5)
    rng = np.random.default_rng(seed)
    signs = np.zeros((N_BOOTSTRAP, k))
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        lr = LogisticRegression(max_iter=2000, random_state=seed + b)
        lr.fit(B[idx], yb)
        signs[b] = np.sign(lr.coef_.ravel())
    return np.maximum(np.mean(signs > 0, axis=0), np.mean(signs < 0, axis=0))


# --------------------------------------------------------------------------
# ARM B -- memory of corrections, generalizing over descriptors
# --------------------------------------------------------------------------

class CorrectionMemory:
    """Remembers which TERM DESCRIPTIONS past humans chose to delete.

    This is the engine re-pointed, exactly as the rescue document argues: one
    clean row per edit instead of 8-40 noisy rows per run, a discrete semantic
    action ("delete this term") instead of one scalar, and an exact judge
    (law_export: deleting a term moves the logit by the predicted amount to
    7e-14) instead of a 30-row validation split.

    It is deliberately a small, inspectable model -- a logistic regression
    over seven named descriptors, whose coefficients are printed. If the
    memory works, you can read WHY.
    """

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self._X: list[np.ndarray] = []
        self._y: list[int] = []
        self._model: LogisticRegression | None = None

    def record(self, descriptors: np.ndarray, dropped: list[int]) -> None:
        """Log one past correction: every term, labelled kept or deleted."""
        drop = set(int(d) for d in dropped)
        for j in range(descriptors.shape[0]):
            self._X.append(descriptors[j])
            self._y.append(1 if j in drop else 0)

    def fit(self) -> bool:
        if len(set(self._y)) < 2:
            return False
        # bug #31 invariant: every sklearn estimator gets an explicit seed.
        self._model = LogisticRegression(
            max_iter=3000, random_state=self.seed, class_weight="balanced")
        self._model.fit(np.asarray(self._X), np.asarray(self._y))
        return True

    def propose(self, descriptors: np.ndarray) -> list[int]:
        """Which terms would past humans have deleted here? No human asked."""
        if self._model is None or descriptors.shape[0] == 0:
            return []
        p = self._model.predict_proba(descriptors)[:, 1]
        return [int(j) for j in np.where(p >= FIRE_THRESHOLD)[0]]

    def scores(self, descriptors: np.ndarray) -> np.ndarray:
        """Raw per-term 'a human would delete this' probabilities.

        Exposed so separability can be reported independently of
        FIRE_THRESHOLD. If artifact terms and healthy terms overlap here, no
        choice of threshold rescues the memory, and tuning the threshold to
        make the arm pass would be measuring the tuner rather than the memory.
        """
        if self._model is None or descriptors.shape[0] == 0:
            return np.zeros(descriptors.shape[0])
        return self._model.predict_proba(descriptors)[:, 1]

    def explain(self) -> str:
        if self._model is None:
            return "  (memory not fitted -- no corrections recorded)"
        c = self._model.coef_.ravel()
        parts = [f"    {n:<16} {v:+.3f}" for n, v in zip(DESCRIPTOR_NAMES, c)]
        return (f"  memory: {len(self._y)} term-observations, "
                f"{sum(self._y)} deletions\n" + "\n".join(parts))


class LookupTable:
    """ARM C -- the floor. Exact dataset-key recall, nothing more.

    Remembers "on dataset D the human deleted terms on feature F" and replays
    it when it sees dataset D again. Correct and cheap when the same dataset
    recurs; it has no opinion at all about a dataset it has not seen, which is
    the entire question.
    """

    def __init__(self) -> None:
        self._by_dataset: dict[str, set[int]] = {}

    def record(self, dataset: str, law: dict, dropped: list[int]) -> None:
        terms = law["heads"][0]["terms"]
        feats = {int(terms[j]["feature"]) for j in dropped
                 if terms[j].get("feature") is not None}
        if feats:
            self._by_dataset.setdefault(dataset, set()).update(feats)

    def propose(self, dataset: str, law: dict) -> list[int]:
        feats = self._by_dataset.get(dataset)
        if not feats:
            return []
        out: list[int] = []
        for f in sorted(feats):
            out.extend(expert_drop_feature(law, f))
        return sorted(set(out))


# --------------------------------------------------------------------------
# scoring one arm on one cell
# --------------------------------------------------------------------------

def score_edit(cell: dict, drop: list[int]) -> dict:
    """Apply an edit to the cell's law and score it on the held-out split."""
    law, Xtr, ytr, Xte, yte = (cell["law"], cell["Xtr"], cell["ytr"],
                               cell["Xte"], cell["yte"])
    if not drop:
        return {"acc": cell["base_acc"], "delta": 0.0, "n_dropped": 0,
                "applied": True, "moved": 0.0, "fired": False}
    edited, applied = apply_edit(law, drop, Xtr, ytr)
    pred = evaluate_law(edited, Xte)
    acc = float(np.mean(pred == yte))
    return {
        "acc": acc, "delta": acc - cell["base_acc"], "n_dropped": len(drop),
        "applied": bool(applied),
        "moved": float(np.mean(pred != cell["base_pred"])),
        "fired": True,
    }


# --------------------------------------------------------------------------
# experiment
# --------------------------------------------------------------------------

def run(names: list[str], seeds: list[int], verbose: bool = True,
        negative_examples: bool = True) -> dict:
    """Leave-one-dataset-out over the shifted cells, plus an iid specificity
    check for arm B.

    `negative_examples` controls what the memory is allowed to remember, and
    it turned out to be the whole experiment. See NEGATIVE EXAMPLES in the
    module docstring: with it off, the memory only ever sees laws that DO
    contain an artifact, cannot learn an absolute threshold, and settles for
    the rank reflex "delete the top term" -- which passes the shift arm and
    fails specificity. Pass False to reproduce that.
    """
    if verbose:
        print("building cells (shift + iid, shared by every arm)...", flush=True)
    shift_cells: dict[str, list[dict]] = {n: [] for n in names}
    iid_cells: dict[str, list[dict]] = {n: [] for n in names}
    for n in names:
        for s in seeds:
            for shifted, bucket in ((True, shift_cells), (False, iid_cells)):
                try:
                    c = build_cell(n, s, shifted=shifted)
                except Exception as exc:
                    if verbose:
                        print(f"  {n}/{s}/{'shift' if shifted else 'iid'}: "
                              f"ERROR {type(exc).__name__}: {exc}", flush=True)
                    continue
                if c is not None:
                    bucket[n].append(c)
        if verbose:
            print(f"  {n:<16} shift {len(shift_cells[n]):>3} | "
                  f"iid {len(iid_cells[n]):>3}", flush=True)

    # descriptors are expensive (bootstrap refits) -- compute once per cell
    for bucket in (shift_cells, iid_cells):
        for n in names:
            for c in bucket[n]:
                c["desc"] = term_descriptors(c["law"], c["Xtr"], c["ytr"],
                                             c["seed"])

    rows: list[dict] = []
    memory_reports: list[str] = []

    for held_out in names:
        train_names = [n for n in names if n != held_out]

        mem = CorrectionMemory(seed=0)
        lut_unseen = LookupTable()      # arm C: LODO, so it never sees held_out
        lut_seen = LookupTable()        # arm C+: diagnostic, allowed to see it
        for n in train_names:
            for c in shift_cells[n]:
                mem.record(c["desc"], c["oracle_drop"])
                lut_unseen.record(n, c["law"], c["oracle_drop"])
                lut_seen.record(n, c["law"], c["oracle_drop"])
            if negative_examples:
                # Reviews that ended in NO edit. A deployed system has these in
                # abundance -- most reviews find nothing -- and without them the
                # memory has no example of a healthy law and can only rank.
                # The lookup table gets nothing from these (there is no feature
                # key to store), which is part of why it is the weaker design.
                for c in iid_cells[n]:
                    mem.record(c["desc"], [])
        fitted = mem.fit()
        memory_reports.append(f"[held out: {held_out}]\n{mem.explain()}")

        for c in shift_cells[held_out]:
            # C+ is allowed to have seen THIS dataset from OTHER seeds. That is
            # the lookup table at its best, and it is reported as a diagnostic,
            # not as the falsifier.
            lut_plus = LookupTable()
            lut_plus._by_dataset = {k: set(v) for k, v in lut_seen._by_dataset.items()}
            for other in shift_cells[held_out]:
                if other["seed"] != c["seed"]:
                    lut_plus.record(held_out, other["law"], other["oracle_drop"])

            arms = {
                "A_human":   score_edit(c, c["oracle_drop"]),
                "B_memory":  score_edit(c, mem.propose(c["desc"]) if fitted else []),
                "C_lookup":  score_edit(c, lut_unseen.propose(held_out, c["law"])),
                "C_plus":    score_edit(c, lut_plus.propose(held_out, c["law"])),
            }
            # separability: what probability does the memory put on the
            # terms the human actually deleted, vs everything else?
            pr = mem.scores(c["desc"]) if fitted else np.zeros(c["desc"].shape[0])
            drop_set = set(int(j) for j in c["oracle_drop"])
            p_art = [float(pr[j]) for j in range(len(pr)) if j in drop_set]
            p_ok = [float(pr[j]) for j in range(len(pr)) if j not in drop_set]
            row = {"dataset": held_out, "seed": c["seed"], "mode": "shift",
                   "base_acc": c["base_acc"], "n_terms": c["desc"].shape[0],
                   "oracle_n_drop": len(c["oracle_drop"]),
                   "p_artifact": p_art, "p_healthy": p_ok}
            for k, v in arms.items():
                row[f"{k}_acc"] = v["acc"]
                row[f"{k}_delta"] = v["delta"]
                row[f"{k}_fired"] = v["fired"]
                row[f"{k}_applied"] = v["applied"]
                row[f"{k}_moved"] = v["moved"]
            rows.append(row)

        # specificity: same memory, iid cells, where the right answer is "do
        # nothing". A memory that has learned a reflex is exposed here.
        for c in iid_cells[held_out]:
            b = score_edit(c, mem.propose(c["desc"]) if fitted else [])
            rows.append({
                "dataset": held_out, "seed": c["seed"], "mode": "iid",
                "base_acc": c["base_acc"], "n_terms": c["desc"].shape[0],
                "oracle_n_drop": 0,
                "p_artifact": [], "p_healthy": [float(v) for v in
                    (mem.scores(c["desc"]) if fitted else [])],
                "B_memory_acc": b["acc"], "B_memory_delta": b["delta"],
                "B_memory_fired": b["fired"], "B_memory_applied": b["applied"],
                "B_memory_moved": b["moved"],
            })

    return {"rows": rows, "memory_reports": memory_reports}


def _paired(rows: list[dict], arm_a: str, arm_b: str, label: str, note: str):
    """Pair two arms cell-by-cell and hand them to the standard analyzer.

    Cells where either arm's edit could not be applied (it would have deleted
    every term) are EXCLUDED, matching hitl_law_edit's rule: including them
    answers "what happens when you ask for an impossible edit".
    """
    a_map, b_map = {}, {}
    for r in rows:
        if f"{arm_a}_acc" not in r or f"{arm_b}_acc" not in r:
            continue
        if not (r.get(f"{arm_a}_applied", True) and r.get(f"{arm_b}_applied", True)):
            continue
        key = (r["dataset"], r["seed"])
        a_map[key] = r[f"{arm_a}_acc"]
        b_map[key] = r[f"{arm_b}_acc"]
    if not a_map:
        return None
    return analyze(label, a_map, b_map, note)


def _print_report(rep, title: str) -> None:
    if rep is None:
        print(f"\n{title}: no usable cells")
        return
    w, t, l = rep["win_tie_lose"]
    print(f"\n{'-' * 74}")
    print(title)
    print(f"{'-' * 74}")
    print(f"  cells          : {rep['n_cells']}")
    print(f"  mean delta     : {rep['mean_delta']:+.4f}")
    print(f"  95% CI         : [{rep['ci95'][0]:+.4f}, {rep['ci95'][1]:+.4f}]")
    print(f"  win/tie/lose   : {w}/{t}/{l}")
    print(f"  cells for 0.01 : {rep['n_needed_for_0.01']} (ran {rep['n_cells']})")
    print(f"  verdict        : {rep['verdict']}")
    for d, v in sorted(rep["per_dataset"].items()):
        print(f"    {d:<16} {v:+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-negatives", action="store_true",
                    help="Train the memory ONLY on laws that contained an "
                         "artifact. Reproduces the rank-reflex failure "
                         "documented in the module docstring.")
    args = ap.parse_args()

    names = [d.strip() for d in args.datasets.split(",") if d.strip() in DATASETS]
    seeds = list(range(args.seeds))
    out = run(names, seeds, verbose=not args.quiet,
              negative_examples=not args.no_negatives)
    rows = out["rows"]
    if not rows:
        print("no cells produced")
        return 1

    shift = [r for r in rows if r["mode"] == "shift"]
    iid = [r for r in rows if r["mode"] == "iid"]

    print(f"\n{'=' * 74}")
    print("WHAT THE MEMORY LEARNED (leave-one-dataset-out)")
    print(f"{'=' * 74}")
    for rep in out["memory_reports"]:
        print(rep)

    print(f"\n{'=' * 74}")
    print("DID EACH ARM FIRE? (shifted cells)")
    print(f"{'=' * 74}")
    print(f"  {'arm':<12} {'fired':>8} {'/cells':>8}   {'mean moved':>12}")
    for arm in ("A_human", "B_memory", "C_lookup", "C_plus"):
        fired = sum(1 for r in shift if r.get(f"{arm}_fired"))
        moved = float(np.mean([r.get(f"{arm}_moved", 0.0) for r in shift]))
        print(f"  {arm:<12} {fired:>8} {len(shift):>8}   {100 * moved:>11.2f}%")

    print(f"\n{'=' * 74}")
    print("SEPARABILITY: can the descriptors tell an artifact term from a")
    print("healthy one AT ALL? (threshold-independent -- if these overlap,")
    print("no FIRE_THRESHOLD rescues the memory)")
    print(f"{'=' * 74}")
    art = np.array([v for r in shift for v in r.get("p_artifact", [])])
    ok_s = np.array([v for r in shift for v in r.get("p_healthy", [])])
    ok_i = np.array([v for r in iid for v in r.get("p_healthy", [])])
    for lbl, arr in (("artifact terms (shift)", art),
                     ("healthy  terms (shift)", ok_s),
                     ("healthy  terms (iid)  ", ok_i)):
        if arr.size:
            print(f"  {lbl}: n={arr.size:>5}  mean {arr.mean():.3f}  "
                  f"p10 {np.percentile(arr, 10):.3f}  "
                  f"p90 {np.percentile(arr, 90):.3f}  max {arr.max():.3f}")
    if art.size and ok_i.size:
        # A healthy term scoring above the WEAKEST artifact term is a false
        # positive that no choice of threshold can avoid.
        overlap = float(np.mean(ok_i >= art.min()))
        print(f"\nhealthy iid terms scoring >= the WEAKEST artifact term: "
              f"{100 * overlap:.1f}%   (0% = a clean threshold exists)")
        print(f"  current FIRE_THRESHOLD = {FIRE_THRESHOLD}")

    print(f"\n{'=' * 74}")
    print("THE FALSIFIER: does memory (B) beat the lookup table (C)?")
    print("Pre-registered: if B does not beat C, ship the lookup table.")
    print(f"{'=' * 74}")
    _print_report(_paired(shift, "C_lookup", "B_memory",
                          "B memory - C lookup",
                          "positive = the memory beat the lookup table"),
                  "B vs C  (the decision)")
    _print_report(_paired(shift, "A_human", "B_memory",
                          "B memory - A human",
                          "negative = the memory is worse than a present human"),
                  "B vs A  (how much of the human is recovered)")
    _print_report(_paired(shift, "C_lookup", "C_plus",
                          "C+ seen - C unseen",
                          "positive = the lookup table works when it HAS seen "
                          "the dataset"),
                  "C+ vs C (diagnostic: the lookup table at its best)")

    print(f"\n{'=' * 74}")
    print("SPECIFICITY: does B stay quiet on iid data, where the right")
    print("answer is to do nothing? (guards the 'always delete the top")
    print("term' degenerate strategy, which would ace the shift arm)")
    print(f"{'=' * 74}")
    if iid:
        fired = sum(1 for r in iid if r.get("B_memory_fired"))
        d = np.array([r["B_memory_delta"] for r in iid])
        print(f"  iid cells                 : {len(iid)}")
        print(f"  cells where B fired       : {fired}  "
              f"({100 * fired / len(iid):.1f}%)  <- should be near 0")
        print(f"  mean delta on iid         : {float(d.mean()):+.4f}  "
              f"<- should be ~0; negative means the memory is damaging "
              f"healthy models")
        print(f"  worst single iid delta    : {float(d.min()):+.4f}")
        by: dict[str, list[float]] = {}
        for r in iid:
            by.setdefault(r["dataset"], []).append(r["B_memory_delta"])
        for k in sorted(by):
            print(f"    {k:<16} {float(np.mean(by[k])):+.4f}")
    else:
        print("  no iid cells")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows,
                       "memory_reports": out["memory_reports"]},
                      fh, indent=2, default=float)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
