"""HITL, measured on the channel that is actually live.

THE PROBLEM WITH EVERY PRIOR HITL RESULT
----------------------------------------
"Human-editable knowledge" has two completely different surfaces in this
codebase, and they have opposite measured behaviour:

  (a) EDITING KNOWLEDGE CELLS -- principles, graph edges, formulas. The
      plumbing works (M5/H1-H3: edit -> event -> materialized file ->
      replay). But it is INERT: deleting 8 of 9 principles gives
      byte-identical predictions, and the graded sweep moved 0.0% of
      predictions across 36 subsystem-removals
      (results/existence_sweep/FINDINGS.md). A human editing principles
      changes nothing about the model. No study design rescues that.

  (b) EDITING THE FITTED LAW -- the sparse_linear terms themselves. Bug #32
      made this exact and testable: `cognition/law_export.py` renders the
      model as a law whose printed coefficients are the ones that multiply
      the printed terms, and F7 showed that deleting a term moves the logit
      by EXACTLY the predicted amount (max error 7e-14).

Every previous HITL experiment aimed at (a) and therefore could not have
found an effect. This one aims at (b).

WHAT THIS DOES AND DOES NOT SHOW
--------------------------------
The "expert" here is a program, so this can only establish that the CHANNEL
can carry value -- never that real humans supply it. That distinction is
exactly what `hitl_pilot_protocol.py` blurred (its "human" was
`if r2 < 0.9: n_terms = 10`, and on the logged run it never fired). Stated
plainly here so the result is not over-read.

The edit is constrained to be honest:
  * computed from TRAINING DATA ONLY -- never the held-out split;
  * a heuristic a human could actually apply by reading the law ("I don't
    trust a term whose sign is not reproducible");
  * followed by a threshold recalibration, because a law with terms removed
    is scoring on a different scale and its old threshold is a stale ruler
    (the invariant from bug #29).

Law-level edits are ~1s, not ~60s like a full optimizer run, so unlike every
prior Gate 2 this experiment can afford enough cells to be POWERED.

Run:
    python -m figpso_uil.experiments.hitl_law_edit
    python -m figpso_uil.experiments.hitl_law_edit --seeds 10 --json out.json
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from figpso_uil.adapters.symbolic_formula_adapter import SymbolicFormulaAdapter
from figpso_uil.cognition.law_export import (
    _term_values, evaluate_law, export_law, head_scores,
)
from figpso_uil.experiments.rescore_gates import analyze

#: Sign-agreement below which an expert would not trust a term.
STABILITY_THRESHOLD = 0.70
N_BOOTSTRAP = 25
#: Matches the adapter: above this majority rate, accuracy is degenerate and
#: balanced accuracy is the honest objective (bug #20).
IMBALANCE_CUTOFF = 0.75


def _load(name: str):
    """Binary datasets only: the OvR path needs per-head Platt recalibration
    after an edit, which is a different experiment. Scope stated, not hidden."""
    from sklearn import datasets as d
    if name == "breast_cancer":
        X, y = d.load_breast_cancer(return_X_y=True)
    elif name == "wine_binary":
        X, y = d.load_wine(return_X_y=True); y = (y == 0).astype(int)
    elif name == "digits_binary":
        X, y = d.load_digits(return_X_y=True); y = (y % 2 == 0).astype(int)
    elif name in ("ionosphere", "sonar", "spambase"):
        import pandas as pd
        from sklearn.datasets import fetch_openml
        from sklearn.preprocessing import LabelEncoder
        ids = {"ionosphere": 59, "spambase": 44}
        b = (fetch_openml(name="sonar", version=1, as_frame=True, parser="auto")
             if name == "sonar" else
             fetch_openml(data_id=ids[name], as_frame=True, parser="auto"))
        df = b.frame.copy()
        tcol = b.target_names[0] if b.target_names else df.columns[-1]
        y = LabelEncoder().fit_transform(df[tcol].astype(str))
        X = pd.get_dummies(df.drop(columns=[tcol]), drop_first=False).to_numpy(float)
        m = np.isfinite(X).all(axis=1)
        X, y = X[m], np.asarray(y, np.int64)[m]
    else:
        raise ValueError(name)
    return np.asarray(X, float), np.asarray(y, int)


DATASETS = ("breast_cancer", "ionosphere", "sonar", "wine_binary",
            "digits_binary", "spambase")


# --------------------------------------------------------------------------
# the "expert"
# --------------------------------------------------------------------------

def _design_matrix(law: dict, X: np.ndarray) -> np.ndarray:
    """Evaluate the law's OWN terms into a design matrix.

    Uses the exported law, not adapter internals -- so this is exactly what a
    human reading the law has access to.
    """
    sel = law.get("selected_features")
    X_red = X[:, np.asarray(sel, dtype=int)] if sel is not None else X
    terms = law["heads"][0]["terms"]
    cols = [_term_values(t, X, X_red, None) for t in terms]
    return np.column_stack(cols) if cols else np.zeros((X.shape[0], 1))


def expert_flag_unstable(law: dict, X_tr: np.ndarray, y_tr: np.ndarray,
                         seed: int) -> list[int]:
    """Return indices of terms an expert would delete.

    Heuristic: refit the law's coefficients on bootstrap resamples of the
    TRAINING data with the basis held FIXED, and flag any term whose
    coefficient sign is not reproducible. "I do not trust a term that flips
    sign when you resample the data" is a judgement a human can make from the
    law alone, and it targets the documented failure mode of this project --
    selection overfitting on small validation splits.
    """
    B = _design_matrix(law, X_tr)
    n, k = B.shape
    if k == 0 or n < 20:
        return []
    rng = np.random.default_rng(seed)
    signs = np.zeros((N_BOOTSTRAP, k))
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        yb = y_tr[idx]
        if len(np.unique(yb)) < 2:
            continue
        lr = LogisticRegression(max_iter=2000, random_state=seed + b)
        lr.fit(B[idx], yb)
        signs[b] = np.sign(lr.coef_.ravel())
    pos = np.mean(signs > 0, axis=0)
    neg = np.mean(signs < 0, axis=0)
    stability = np.maximum(pos, neg)
    return [int(j) for j in np.where(stability < STABILITY_THRESHOLD)[0]]


def expert_drop_feature(law: dict, feature_idx: int) -> list[int]:
    """Return indices of terms that reference a feature the expert knows is bad.

    This is the OTHER kind of human input, and the interesting one. Stability
    pruning (above) reasons about the training data -- but the model already
    fit that data optimally, so on an iid split a human can only add noise.
    A human's real advantage is knowledge the training data DOES NOT CONTAIN:
    "column k is a logging artifact / leaked label / sensor that will be
    recalibrated next month". No amount of resampling reveals that.

    Term feature indices are in ORIGINAL column space (law_export maps them
    back through selected_features), so a human reading `x37` really is
    pointing at column 37 of their data -- which is only true because bug #32
    was fixed.
    """
    drop = []
    for j, t in enumerate(law["heads"][0]["terms"]):
        if t.get("feature") == feature_idx:
            drop.append(j)
        elif t["type"] == "rule" and any(
                c["feature"] == feature_idx for c in t.get("conditions", [])):
            drop.append(j)
        elif t["type"] == "neural" and str(feature_idx) in t.get("weights", {}):
            drop.append(j)
    return drop


def _refit_threshold(law: dict, X: np.ndarray, y: np.ndarray) -> float:
    """Recalibrate the decision threshold after an edit.

    A law with terms removed scores on a different scale, so its old
    threshold is a stale ruler -- the invariant from bug #29. Balanced
    accuracy once the majority class is large (bug #20).
    """
    head = law["heads"][0]
    sel = law.get("selected_features")
    X_red = X[:, np.asarray(sel, dtype=int)] if sel is not None else X
    scores = head_scores(head, X, X_red, None)
    pos = head.get("positive_class", 1)
    y_bin = (np.asarray(y) == pos).astype(int)
    base = max(float(np.mean(y_bin == 0)), float(np.mean(y_bin == 1)))
    imbalanced = base >= IMBALANCE_CUTOFF

    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        pred = (scores >= t).astype(int)
        if imbalanced:
            recalls = [float(np.mean(pred[y_bin == c] == c))
                       for c in (0, 1) if np.any(y_bin == c)]
            s = float(np.mean(recalls)) if recalls else 0.0
        else:
            s = float(np.mean(pred == y_bin))
        if s > best_s:
            best_t, best_s = float(t), s
    return best_t


def apply_edit(law: dict, drop: list[int], X_tr, y_tr) -> tuple[dict, bool]:
    """Delete terms and recalibrate. Pure law-level surgery.

    Returns (law, applied). `applied` is False when the edit would delete
    EVERY term -- the model is then entirely the thing the expert wants gone,
    and pure law-editing cannot repair it: the honest human action there is
    "drop the feature and refit", a strictly more powerful intervention than
    this experiment measures. Those cells must be reported separately, not
    silently counted as +0.0000 ties, which would drag the mean toward zero
    and misstate the result.
    """
    edited = json.loads(json.dumps(law))  # deep copy, JSON-round-trip safe
    keep = [t for j, t in enumerate(edited["heads"][0]["terms"]) if j not in set(drop)]
    if not keep:
        return law, False
    edited["heads"][0]["terms"] = keep
    edited["heads"][0]["threshold"] = _refit_threshold(edited, X_tr, y_tr)
    return edited, True


# --------------------------------------------------------------------------
# experiment
# --------------------------------------------------------------------------

def one_cell(name: str, seed: int, mode: str = "iid") -> dict | None:
    X, y = _load(name)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)

    spurious_idx = None
    if mode in ("shift", "shift_stability"):
        # A spurious column that is STRONGLY predictive in training and pure
        # noise at test time -- a leaked label, a logging artifact, a sensor
        # that gets recalibrated. The training data cannot reveal that it will
        # stop working; only outside knowledge can. This is the situation
        # where a human should beat the fit, and if HITL has no value HERE it
        # has none anywhere.
        rng = np.random.default_rng(seed + 9001)
        s_tr = (ytr * 2.0 - 1.0) + rng.normal(0, 0.35, size=len(ytr))
        s_te = rng.normal(0, 1.0, size=len(yte))
        Xtr = np.column_stack([Xtr, s_tr])
        Xte = np.column_stack([Xte, s_te])
        spurious_idx = Xtr.shape[1] - 1

    import os, tempfile
    sd = tempfile.mkdtemp(prefix="hitl_")
    clf = SymbolicFormulaAdapter(
        seed=seed, num_classes=2, learning_mode="sparse_linear",
        formula_library_path=os.path.join(sd, "lib.json"))
    # SCOPE, stated not hidden: the evolved-formula basis family is switched
    # off so the law is fully SELF-CONTAINED and LEGIBLE (raw + stump terms).
    # That is the right scope for a human-edit experiment: bug #32 established
    # that formula-tree terms are neither self-contained nor legible, so a
    # human could not audit or judge them anyway, and `evaluate_law` correctly
    # refuses to guess at them. Including them would measure "can a human edit
    # something they cannot read", which is not the claim.
    clf.use_symbolic_basis = False
    model = clf.fit_weighted(Xtr, ytr, np.ones(Xtr.shape[1]))
    try:
        law = export_law(model)
    except Exception:
        return None
    if law["kind"] != "binary":
        return None

    base_pred = evaluate_law(law, Xte)
    base_acc = float(np.mean(base_pred == yte))

    if mode == "shift":
        drop = expert_drop_feature(law, spurious_idx)
    else:
        # `shift_stability` is the CONTROL for the shift result: same shifted
        # data, but the expert reasons only from the training data instead of
        # supplying outside knowledge. If a data-derived heuristic could spot
        # the artifact, the shift gain would be automatable and not a human
        # contribution at all. (It cannot: the artifact is the MOST stable
        # term in training -- that is what makes it dangerous.)
        drop = expert_flag_unstable(law, Xtr, ytr, seed)
    edited, applied = apply_edit(law, drop, Xtr, ytr)
    edit_pred = evaluate_law(edited, Xte)
    edit_acc = float(np.mean(edit_pred == yte))

    n_terms = len(law["heads"][0]["terms"])
    moved = float(np.mean(base_pred != edit_pred))
    return {
        "dataset": name, "seed": seed,
        "base_acc": base_acc, "edit_acc": edit_acc,
        "delta": edit_acc - base_acc,
        "n_terms": n_terms, "n_dropped": len(drop),
        "edit_applied": bool(applied),
        "frac_predictions_moved": moved,
        "threshold_before": law["heads"][0]["threshold"],
        "threshold_after": edited["heads"][0]["threshold"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--mode", default="iid",
                    choices=["iid", "shift", "shift_stability"],
                    help="iid: expert prunes unstable terms (data-derived). "
                         "shift: expert deletes a term on a feature it knows "
                         "is a training artifact (knowledge the data lacks).")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    names = [d.strip() for d in args.datasets.split(",") if d.strip() in DATASETS]
    seeds = list(range(args.seeds))

    rows = []
    for name in names:
        print(f"\n=== {name} [{args.mode}] ===", flush=True)
        for s in seeds:
            try:
                r = one_cell(name, s, mode=args.mode)
            except Exception as exc:
                print(f"  seed {s}: ERROR {type(exc).__name__}: {exc}", flush=True)
                continue
            if r is None:
                continue
            rows.append(r)
            print(f"  seed {s:>2}: {r['n_terms']:>3} terms, dropped "
                  f"{r['n_dropped']:>2} | base {r['base_acc']:.4f} -> "
                  f"edit {r['edit_acc']:.4f} ({r['delta']:+.4f}) | "
                  f"{100 * r['frac_predictions_moved']:.1f}% preds moved",
                  flush=True)

    if not rows:
        print("no cells produced")
        return 1

    # --- claim 1: is the channel LIVE? -----------------------------------
    moved = np.array([r["frac_predictions_moved"] for r in rows])
    dropped = np.array([r["n_dropped"] for r in rows])
    live = int(np.sum(moved > 0))
    print(f"\n{'=' * 74}")
    print("CLAIM 1 — is the edit channel live?")
    print(f"{'=' * 74}")
    print(f"  cells                        : {len(rows)}")
    print(f"  cells where an edit was made : {int(np.sum(dropped > 0))}")
    print(f"  cells where predictions MOVED: {live}")
    print(f"  mean share of predictions moved: {100 * float(moved.mean()):.2f}%")
    print(f"  max                            : {100 * float(moved.max()):.2f}%")
    print("  (contrast: editing PRINCIPLES moves 0.00% — "
          "results/existence_sweep/FINDINGS.md)")

    # --- claim 2: does it IMPROVE? ---------------------------------------
    # Cells where the edit could not be applied (it would have deleted every
    # term) are EXCLUDED, not counted as ties. Including them would answer
    # "what happens when you ask for an impossible edit", not "does the edit
    # help". They are reported separately below because how often the model is
    # ENTIRELY the thing the expert wants gone is itself a finding.
    blocked = [r for r in rows if not r.get("edit_applied", True)]
    usable = [r for r in rows if r.get("edit_applied", True)]
    if blocked:
        by_ds: dict[str, int] = {}
        for r in blocked:
            by_ds[r["dataset"]] = by_ds.get(r["dataset"], 0) + 1
        print(f"\n  NOTE: {len(blocked)}/{len(rows)} cells EXCLUDED — the edit "
              f"would have deleted every term")
        print(f"        (the law was entirely the flagged feature; pure "
              f"law-editing cannot repair that, a refit is needed)")
        for d, c in sorted(by_ds.items()):
            print(f"          {d}: {c}")
    if not usable:
        print("  no usable cells for claim 2")
        return 0
    base_map = {(r["dataset"], r["seed"]): r["base_acc"] for r in usable}
    edit_map = {(r["dataset"], r["seed"]): r["edit_acc"] for r in usable}
    rep = analyze("expert law edit - no edit", base_map, edit_map,
                  "positive = the human edit helped held-out accuracy")
    print(f"\n{'=' * 74}")
    print("CLAIM 2 — does the edit improve held-out accuracy?")
    print(f"{'=' * 74}")
    w, t, l = rep["win_tie_lose"]
    print(f"  cells            : {rep['n_cells']}")
    print(f"  mean delta       : {rep['mean_delta']:+.4f}")
    print(f"  95% CI           : [{rep['ci95'][0]:+.4f}, {rep['ci95'][1]:+.4f}]")
    print(f"  win/tie/lose     : {w}/{t}/{l}")
    print(f"  cells for 0.01   : {rep['n_needed_for_0.01']} (ran {rep['n_cells']})")
    print(f"  verdict          : {rep['verdict']}")
    for d, v in sorted(rep["per_dataset"].items()):
        print(f"    {d:<16} {v:+.4f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows, "report": rep}, fh, indent=2, default=float)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
