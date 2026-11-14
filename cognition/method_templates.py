"""Simplified formula templates extracted from well-known ML methods."""

from __future__ import annotations

import numpy as np

from figpso_uil.cognition.formula_engine import FormulaNode


def _f(idx: int) -> FormulaNode:
    return FormulaNode(feature=idx)


def _c(val: float) -> FormulaNode:
    return FormulaNode(const=float(val))


def method_template_seeds(n_features: int) -> dict[str, FormulaNode]:
    """Nearest symbolic forms of classic ML methods — UIL mutates from here."""
    if n_features <= 0:
        return {}

    anchors = sorted(set(int(i) for i in np.linspace(0, n_features - 1, min(8, n_features))))
    seeds: dict[str, FormulaNode] = {}

    for idx in anchors[:4]:
        seeds[f"method_logistic_f{idx}"] = FormulaNode(
            op="sigmoid",
            children=[
                FormulaNode(
                    op="add",
                    children=[
                        FormulaNode(op="mul", children=[_c(1.0), _f(idx)]),
                        _c(0.0),
                    ],
                )
            ],
        )

    for idx in anchors[:3]:
        seeds[f"method_stump_f{idx}"] = FormulaNode(
            op="sigmoid",
            children=[
                FormulaNode(
                    op="mul",
                    children=[_c(2.0), FormulaNode(op="sub", children=[_f(idx), _c(0.0)])],
                )
            ],
        )

    for idx in anchors[:2]:
        seeds[f"method_hinge_f{idx}"] = FormulaNode(
            op="relu",
            children=[FormulaNode(op="mul", children=[_c(1.0), _f(idx)])],
        )

    idx = anchors[0]
    seeds["method_nb_log_f"] = FormulaNode(
        op="log",
        children=[
            FormulaNode(
                op="add",
                children=[FormulaNode(op="abs", children=[_f(idx)]), _c(1e-3)],
            )
        ],
    )

    ridx = anchors[min(1, len(anchors) - 1)]
    seeds[f"method_rbf_f{ridx}"] = FormulaNode(
        op="exp",
        children=[
            FormulaNode(
                op="neg",
                children=[FormulaNode(op="mul", children=[_f(ridx), _f(ridx)])],
            )
        ],
    )

    if n_features >= 2:
        i, j = anchors[0], anchors[min(1, len(anchors) - 1)]
        seeds[f"method_maxout_f{i}_f{j}"] = FormulaNode(op="max", children=[_f(i), _f(j)])
        seeds[f"method_minpool_f{i}_f{j}"] = FormulaNode(op="min", children=[_f(i), _f(j)])

    if len(anchors) >= 3:
        a, b, c_idx = anchors[0], anchors[1], anchors[2]
        seeds["method_linear_3feat"] = FormulaNode(
            op="add",
            children=[FormulaNode(op="add", children=[_f(a), _f(b)]), _f(c_idx)],
        )

    lidx = anchors[-1]
    seeds[f"method_inv_dist_f{lidx}"] = FormulaNode(
        op="div",
        children=[
            _c(1.0),
            FormulaNode(
                op="add",
                children=[
                    FormulaNode(
                        op="abs",
                        children=[FormulaNode(op="sub", children=[_f(lidx), _c(0.0)])],
                    ),
                    _c(0.1),
                ],
            ),
        ],
    )

    if n_features >= 2:
        i, j = anchors[0], anchors[min(1, len(anchors) - 1)]
        seeds[f"method_interaction_f{i}_f{j}"] = FormulaNode(
            op="mul",
            children=[
                FormulaNode(op="sigmoid", children=[_f(i)]),
                FormulaNode(op="sigmoid", children=[_f(j)]),
            ],
        )

    return seeds
