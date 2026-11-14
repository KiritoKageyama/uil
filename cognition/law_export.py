"""Faithful export of a fitted model as a standalone, executable law.

WHY THIS EXISTS
---------------
The project's one surviving value claim is that its models are *inspectable
and human-editable*. That claim was never tested. `readable_sparse_linear`
renders a pretty string, and nothing anywhere checked that the string IS the
model. Inspection showed it is not, in five separate ways:

  1. It truncates to `top=12` terms and prints "(+N more terms)".
  2. It omits the standardization (`mu`/`sd`). The printed coefficients live
     in STANDARDIZED space while the printed term names live in RAW space, so
     the string read literally computes a different function than the model.
  3. It omits the decision threshold. `predict()` uses `score >= threshold`,
     and the threshold is fitted (bug #20) -- it is frequently not 0.5.
  4. It omits `selected_features`. After E1 preselection, `x26` in the string
     means "column 26 of the REDUCED basis", not column 26 of the user's data.
  5. It omits `positive_class`, so the label mapping is invisible.

None of that is a rendering nicety: a reader who acts on the string acts on
the wrong function. This module makes the readability claim FALSIFIABLE by
producing a law that carries every quantity `predict()` uses, and an
evaluator that consumes ONLY the law.

THE ALGEBRA
-----------
The fitted score is

    z = sum_j coef_j * (b_j(x) - mu_j) / sd_j + intercept

which is exactly linear in the basis values, so it folds into raw basis space
with no approximation:

    c_j       = coef_j / sd_j
    intercept'= intercept - sum_j coef_j * mu_j / sd_j
    z         = sum_j c_j * b_j(x) + intercept'

The exported law stores `c_j` and `intercept'`, so the printed coefficient is
the one that actually multiplies the printed term. That is the whole point.

OPAQUE TERMS
------------
Two basis families are not self-describing:

  * `neural` -- exactly specifiable (relu/tanh of an affine map) and exported
    in full, but a human cannot read a 30-input dot product at a glance. It is
    exported as EXACT but flagged `legible=False`.
  * `formula` -- an evolved expression tree. Exact evaluation needs the
    formula engine, so `evaluate_law` requires a `tree_eval` callable for
    these. Flagged `legible=False` and `self_contained=False`.

`law_is_self_contained(law)` reports whether the law can be evaluated with no
help from the codebase at all. That distinction is the honest version of
"readable": a law over raw/stump/rule/kan terms genuinely is; one leaning on
neural units or formula trees is inspectable but not self-contained.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

LAW_FORMAT = "uil.law/1"

#: Basis families that a human can read directly off the page.
LEGIBLE_TYPES = frozenset({"raw", "stump", "rule", "kan"})
#: Basis families evaluable from the law alone (no codebase required).
SELF_CONTAINED_TYPES = frozenset({"raw", "stump", "rule", "kan", "neural"})


class LawExportError(ValueError):
    """Raised when a model cannot be represented as a law."""


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def _orig_index(j: int, sel: list[int] | None) -> int:
    """Map a reduced-basis column index back to the caller's column index."""
    if sel is None:
        return int(j)
    return int(sel[j]) if 0 <= j < len(sel) else int(j)


def _export_term(entry: dict, coef_raw: float, sel: list[int] | None) -> dict:
    kind = entry.get("type", "formula")
    term: dict = {"type": kind, "coef": float(coef_raw)}

    if kind == "raw":
        term["feature"] = _orig_index(entry["feature"], sel)
        term["expr"] = f"x{term['feature']}"
    elif kind == "stump":
        term["feature"] = _orig_index(entry["feature"], sel)
        term["threshold"] = float(entry["threshold"])
        term["expr"] = f"1[x{term['feature']} > {term['threshold']:.6g}]"
    elif kind == "rule":
        conds = [
            {
                "feature": _orig_index(c["feature"], sel),
                "op": c["op"],
                "threshold": float(c["threshold"]),
            }
            for c in entry.get("conditions", [])
        ]
        term["conditions"] = conds
        inner = " & ".join(
            f"x{c['feature']} {c['op']} {c['threshold']:.6g}" for c in conds
        )
        term["expr"] = f"1[{inner}]"
    elif kind == "kan":
        term["feature"] = _orig_index(entry["feature"], sel)
        term["knot"] = float(entry.get("knot", 0.0))
        term["degree"] = int(entry.get("degree", 2))
        term["expr"] = (
            f"max(x{term['feature']} - {term['knot']:.6g}, 0)^{term['degree']}"
        )
    elif kind == "neural":
        w = np.asarray(entry.get("w", []), dtype=float)
        term["weights"] = {
            str(_orig_index(k, sel)): float(v)
            for k, v in enumerate(w)
            if abs(float(v)) > 0.0
        }
        term["bias"] = float(entry.get("b", 0.0))
        term["activation"] = str(entry.get("act", "relu"))
        term["unit"] = entry.get("unit")
        top = np.argsort(-np.abs(w))[:3] if w.size else []
        lean = ",".join(f"x{_orig_index(int(k), sel)}" for k in top)
        term["expr"] = f"{term['activation']}(u{term['unit']} <- {lean}, ...)"
    else:  # formula
        term["type"] = "formula"
        term["formula_id"] = entry.get("formula_id")
        term["tree"] = entry.get("tree")
        term["expr"] = f"f[{entry.get('formula_id', '?')}](x)"

    term["legible"] = kind in LEGIBLE_TYPES
    term["self_contained"] = kind in SELF_CONTAINED_TYPES
    return term


def _export_head(model: dict, sel: list[int] | None) -> dict:
    """Fold standardization into raw-space coefficients for one binary head."""
    basis = model.get("basis") or []
    coef = np.asarray(model.get("coef", []), dtype=float)
    mu = np.asarray(model.get("mu", []), dtype=float)
    sd = np.asarray(model.get("sd", []), dtype=float)

    if not (len(basis) == coef.size == mu.size == sd.size):
        raise LawExportError(
            f"ragged sparse_linear model: {len(basis)} basis / {coef.size} coef "
            f"/ {mu.size} mu / {sd.size} sd"
        )

    sd_safe = np.where(np.abs(sd) > 1e-9, sd, 1.0)
    coef_raw = coef / sd_safe
    intercept_raw = float(model.get("intercept", 0.0)) - float(
        np.sum(coef * mu / sd_safe)
    )

    return {
        "terms": [
            _export_term(basis[i], coef_raw[i], sel) for i in range(len(basis))
        ],
        "intercept": intercept_raw,
        "threshold": float(model.get("threshold", 0.5)),
        "positive_class": model.get("positive_class", 1),
        "fit": model.get("fit", "l1"),
    }


def export_law(model: dict) -> dict:
    """Convert a fitted model dict into a standalone executable law.

    Supports `sparse_linear` (including the EBM fit variant, which shares the
    model shape) and `multiclass_ovr` stacks of them. Other kinds -- boosting,
    bagging, hybrid -- are NOT linear laws and raise; that is a true statement
    about them, not a gap to paper over.
    """
    if not isinstance(model, dict):
        raise LawExportError(f"expected a model dict, got {type(model).__name__}")

    sel_raw = model.get("selected_features")
    sel = [int(v) for v in np.asarray(sel_raw).ravel()] if sel_raw is not None else None
    fw = model.get("feature_weights")

    kind = model.get("kind")
    if kind == "sparse_linear":
        law = {
            "format": LAW_FORMAT,
            "kind": "binary",
            "selected_features": sel,
            "feature_weights": list(np.asarray(fw, dtype=float).ravel())
            if fw is not None else None,
            "heads": [_export_head(model, sel)],
            "decision": "predict positive_class iff sigmoid(z) >= threshold",
        }
        return law

    if kind == "multiclass_ovr":
        heads = model.get("classifiers", {})
        labels = sorted(heads.keys(), key=lambda c: float(c))
        out_heads = []
        for c in labels:
            head_model = heads[c]
            if head_model.get("kind") != "sparse_linear":
                raise LawExportError(
                    f"OvR head {c!r} is kind={head_model.get('kind')!r}, "
                    "not a linear law"
                )
            head = _export_head(head_model, sel)
            head["label"] = int(float(c))
            head["calibration"] = head_model.get("ovr_calibration")
            out_heads.append(head)
        return {
            "format": LAW_FORMAT,
            "kind": "multiclass_ovr",
            "selected_features": sel,
            "feature_weights": list(np.asarray(fw, dtype=float).ravel())
            if fw is not None else None,
            "heads": out_heads,
            "decision": "argmax over Platt-calibrated head probabilities",
        }

    raise LawExportError(
        f"model kind {kind!r} is not a linear law (only sparse_linear and "
        "multiclass_ovr stacks of it can be rendered as one)"
    )


# --------------------------------------------------------------------------
# evaluate -- consumes ONLY the law
# --------------------------------------------------------------------------

def _term_values(term: dict, X: np.ndarray, X_red: np.ndarray,
                 tree_eval: Callable | None) -> np.ndarray:
    n = X.shape[0]
    kind = term["type"]

    if kind == "raw":
        j = term["feature"]
        return X[:, j] if j < X.shape[1] else np.zeros(n)

    if kind == "stump":
        j = term["feature"]
        if j >= X.shape[1]:
            return np.zeros(n)
        return (X[:, j] > term["threshold"]).astype(float)

    if kind == "rule":
        mask = np.ones(n, dtype=float)
        for c in term["conditions"]:
            j = c["feature"]
            if j >= X.shape[1]:
                return np.zeros(n)
            if c["op"] == ">":
                mask *= (X[:, j] > c["threshold"]).astype(float)
            else:
                mask *= (X[:, j] <= c["threshold"]).astype(float)
        return mask

    if kind == "kan":
        j = term["feature"]
        if j >= X.shape[1]:
            return np.zeros(n)
        return np.power(np.maximum(X[:, j] - term["knot"], 0.0), term["degree"])

    if kind == "neural":
        z = np.full(n, term["bias"], dtype=float)
        for k, v in term["weights"].items():
            j = int(k)
            if j < X.shape[1]:
                z = z + v * X[:, j]
        return np.maximum(z, 0.0) if term["activation"] == "relu" else np.tanh(z)

    if kind == "formula":
        if tree_eval is None:
            raise LawExportError(
                f"term {term.get('formula_id')!r} is a formula tree; pass "
                "tree_eval= to evaluate it (the law is not self-contained)"
            )
        v = np.asarray(tree_eval(term["tree"], X_red), dtype=float).ravel()
        if v.shape[0] != n or not np.isfinite(v).all():
            return np.zeros(n)
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    raise LawExportError(f"unknown term type {kind!r}")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def head_scores(head: dict, X: np.ndarray, X_red: np.ndarray,
                tree_eval: Callable | None) -> np.ndarray:
    z = np.full(X.shape[0], head["intercept"], dtype=float)
    for term in head["terms"]:
        z = z + term["coef"] * _term_values(term, X, X_red, tree_eval)
    return _sigmoid(z)


def evaluate_law(law: dict, X: np.ndarray,
                 tree_eval: Callable | None = None) -> np.ndarray:
    """Predict labels from the law alone. X is in the CALLER's column space."""
    X = np.asarray(X, dtype=float)
    sel = law.get("selected_features")
    X_red = X[:, np.asarray(sel, dtype=int)] if sel is not None else X

    if law["kind"] == "binary":
        head = law["heads"][0]
        scores = head_scores(head, X, X_red, tree_eval)
        preds = (scores >= head["threshold"]).astype(int)
        pos = head.get("positive_class", 1)
        return np.where(preds == 1, pos, 0) if pos != 1 else preds

    mat = []
    for head in law["heads"]:
        s = head_scores(head, X, X_red, tree_eval)
        cal = head.get("calibration")
        if cal:
            s = _sigmoid(float(cal.get("a", 1.0)) * s + float(cal.get("b", 0.0)))
        mat.append(s)
    labels = np.array([h["label"] for h in law["heads"]])
    return labels[np.argmax(np.column_stack(mat), axis=1)]


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------

def law_is_self_contained(law: dict) -> bool:
    """True when the law can be evaluated with no help from this codebase."""
    return all(
        t.get("self_contained", False)
        for head in law["heads"] for t in head["terms"]
    )


def law_is_legible(law: dict) -> bool:
    """True when every term is readable off the page by a human."""
    return all(
        t.get("legible", False)
        for head in law["heads"] for t in head["terms"]
    )


def _render_head(head: dict, top: int | None) -> str:
    terms = sorted(head["terms"], key=lambda t: -abs(t["coef"]))
    shown = terms if top is None else terms[:top]
    body = " ".join(f"{t['coef']:+.6g}*{t['expr']}" for t in shown)
    tail = ""
    if top is not None and len(terms) > len(shown):
        tail = f"  [TRUNCATED: {len(terms) - len(shown)} of {len(terms)} terms hidden]"
    law_str = f"z = {body} {head['intercept']:+.6g}"
    pos = head.get("positive_class", 1)
    rule = (
        f"predict {pos} iff sigmoid(z) >= {head['threshold']:.6g}, else 0"
    )
    return f"{law_str}\n  {rule}{tail}"


def render_law(law: dict, top: int | None = None) -> str:
    """Render the law in full. `top` truncates and SAYS SO -- silent
    truncation is what made the old renderer unfaithful."""
    lines = []
    sel = law.get("selected_features")
    if sel is not None:
        lines.append(
            f"# feature indices are ORIGINAL columns "
            f"(model used {len(sel)} preselected of them)"
        )
    if not law_is_self_contained(law):
        lines.append("# NOT self-contained: contains formula-tree terms")
    elif not law_is_legible(law):
        lines.append("# exact but not fully legible: contains neural units")

    if law["kind"] == "binary":
        lines.append(_render_head(law["heads"][0], top))
    else:
        for head in law["heads"]:
            lines.append(f"class {head['label']}:")
            lines.append("  " + _render_head(head, top).replace("\n", "\n  "))
        lines.append("predict argmax over calibrated class scores")
    return "\n".join(lines)
