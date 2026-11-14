"""Accuracy: soft translator+connect vs hard wiring vs SOTA baselines.

Arms
----
UIL (SymbolicFormula sparse_linear, 12 stages, classic search):
  uil_off              — cognition off
  hard_wiring          — translator + connect (failed Gate 2 combo)
  soft_wiring          — translator + connect, but guidance = |corr| only
                         (no concentrate / hard top-k / l1 bump)
  translator_no_weights— translator + connect, feature_guidance=None
                         (config/formulas only; weights not published)

SOTA / strong ML (same split, no UIL loop — one fit on train):
  logreg_l2, random_forest, hist_gbm, xgboost (if installed),
  svm_rbf (n<1000), mlp

Usage (from Downloads/):
    python -m figpso_uil.experiments.compare_soft_wiring_sota
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from datetime import datetime, timezone

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from figpso_uil.experiments.eval_crossseed import one_split, DATASETS
from figpso_uil.cognition.paths import results_path
from figpso_uil.cognition import signal_translator as st
import figpso_uil.core.optimizer as optmod

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


_ORIG = st.translate


def _soft_translate(**kw):
    """Corr prior only — no concentrate / hard top-k / sparse_l1 bump."""
    d = st.LearnerDirective()
    d.feature_guidance = st._safe_corr_guidance(
        kw.get("X_train"), kw.get("y_train"), kw["n_features"],
    )
    d.sources = ["dataset_correlation", "soft_wiring"]
    d.use_symbolic_basis = True
    # Still prefer active formulas (helpful channel without weight skew)
    lib = kw.get("formula_library")
    if lib is not None and (kw.get("ablation") or {}).get("formulas", True):
        try:
            for f in lib.get_active_formulas()[:12]:
                fid = f.get("formula_id")
                if fid:
                    d.prefer_formula_ids.append(fid)
            d.sources.append("formulas")
        except Exception:
            pass
    return d


def _no_weight_translate(**kw):
    d = _ORIG(**kw)
    d.feature_guidance = None  # config + formulas only
    d.sources = list(d.sources) + ["no_feature_guidance"]
    return d


def _uil_cell(dataset, seed, stages, arm: str) -> dict:
    loader, mode = DATASETS[dataset]
    X, y = loader()
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    n_classes = int(len(np.unique(y)))

    if arm == "uil_off":
        optmod.translate_signals = _ORIG
        kw = dict(uil_enabled=False)
    elif arm == "hard_wiring":
        optmod.translate_signals = _ORIG
        kw = dict(
            uil_enabled=True, signal_translator=True,
            connect_cognition=True, utility_gated_formulas=False,
        )
    elif arm == "soft_wiring":
        optmod.translate_signals = _soft_translate
        kw = dict(
            uil_enabled=True, signal_translator=True,
            connect_cognition=True, utility_gated_formulas=False,
        )
    elif arm == "translator_no_weights":
        optmod.translate_signals = _no_weight_translate
        kw = dict(
            uil_enabled=True, signal_translator=True,
            connect_cognition=True, utility_gated_formulas=False,
        )
    else:
        raise ValueError(arm)

    t0 = time.perf_counter()
    try:
        r = one_split(
            X, y, mode, seed, stages, 10, 10, 40, n_classes,
            search_mode="classic", **kw,
        )
    finally:
        optmod.translate_signals = _ORIG
    return {
        "acc": float(r["acc"]),
        "macro_f1": float(r["macro_f1"]),
        "gap": float(r["internal_val"] - r["acc"]),
        "wall_s": round(time.perf_counter() - t0, 2),
    }


def _sota_cell(dataset, seed, name: str) -> dict:
    loader, _ = DATASETS[dataset]
    X, y = loader()
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )
    n = len(y)
    rs = seed & 0x7FFFFFFF
    t0 = time.perf_counter()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "logreg_l2":
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, random_state=rs),
            )
        elif name == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=200, random_state=rs, n_jobs=-1,
            )
        elif name == "hist_gbm":
            clf = HistGradientBoostingClassifier(random_state=rs, max_iter=200)
        elif name == "xgboost":
            if not _HAS_XGB:
                return {"acc": float("nan"), "macro_f1": float("nan"),
                        "wall_s": 0.0, "skipped": True}
            objective = "multi:softmax" if len(np.unique(y)) > 2 else "binary:logistic"
            params = {
                "objective": objective,
                "max_depth": 6,
                "eta": 0.1,
                "seed": rs,
                "nthread": -1,
            }
            if len(np.unique(y)) > 2:
                params["num_class"] = int(len(np.unique(y)))
            dtr = xgb.DMatrix(Xtr, label=ytr)
            dte = xgb.DMatrix(Xte)
            model = xgb.train(params, dtr, num_boost_round=100)
            pred = model.predict(dte)
            if len(np.unique(y)) <= 2:
                pred = (np.asarray(pred) > 0.5).astype(int)
            else:
                pred = np.asarray(pred, dtype=int)
            return {
                "acc": float(accuracy_score(yte, pred)),
                "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
                "wall_s": round(time.perf_counter() - t0, 2),
            }
        elif name == "svm_rbf":
            if n > 2000:
                return {"acc": float("nan"), "macro_f1": float("nan"),
                        "wall_s": 0.0, "skipped": True}
            clf = make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", C=1.0, random_state=rs),
            )
        elif name == "mlp":
            clf = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(64, 32), max_iter=400,
                    random_state=rs,
                ),
            )
        else:
            raise ValueError(name)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)

    return {
        "acc": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
        "wall_s": round(time.perf_counter() - t0, 2),
    }


UIL_ARMS = (
    "uil_off", "hard_wiring", "soft_wiring", "translator_no_weights",
)
SOTA_ARMS = (
    "logreg_l2", "random_forest", "hist_gbm", "xgboost", "svm_rbf", "mlp",
)

# What each SOTA method is doing (for NOTES.md)
ALGO_NOTES = {
    "uil_off": (
        "UIL SymbolicFormula sparse_linear + classic search, cognition wiring OFF. "
        "Readable L1 over raw/stump/formula basis; search only tweaks penalty/weights "
        "that sparse_linear largely ignores without connect."
    ),
    "hard_wiring": (
        "Translator (concentrate + hard top-k) + connect_cognition. "
        "Aggressive feature skew enters L1 — the combo that failed Gate 2."
    ),
    "soft_wiring": (
        "Translator publishes |corr(x,y)| guidance only + prefer formulas; "
        "connect folds those soft weights into L1. Hypothesis: keeps channel "
        "live without overfit from hard top-k."
    ),
    "translator_no_weights": (
        "Full translator for config/formula prefer, but feature_guidance=None "
        "so connect has nothing nonuniform to apply. Measured helpful on wine/123."
    ),
    "logreg_l2": (
        "L2-regularized logistic regression on standardized features. "
        "Linear decision boundary; strong on near-linear tabular data; "
        "cannot model interactions without feature engineering."
    ),
    "random_forest": (
        "Bag of shallow decision trees on bootstrap samples + random feature "
        "subsets; averages votes. Captures interactions; less readable; "
        "strong default tabular baseline."
    ),
    "hist_gbm": (
        "Sklearn HistGradientBoosting — modern GBM with histogram binning "
        "(LightGBM-like). Sequential trees fit residuals; usually near SOTA "
        "on medium tabular without heavy tuning."
    ),
    "xgboost": (
        "Gradient-boosted trees (Chen & Guestrin). Industry tabular workhorse; "
        "column/row sampling, second-order approx. Opaque; strong accuracy ceiling."
    ),
    "svm_rbf": (
        "RBF-kernel SVM on standardized features. Implicit high-dim feature map; "
        "great on small interaction-dense sets (e.g. sonar); scales poorly in n."
    ),
    "mlp": (
        "Small multilayer perceptron (64–32 ReLU) on standardized inputs. "
        "Dense nonlinear capacity; not readable; can underperform trees on "
        "small tabular."
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="breast_cancer,wine,ionosphere")
    ap.add_argument("--seeds", default="42,7,123")
    ap.add_argument("--stages", type=int, default=12)
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip() in DATASETS]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stages": args.stages,
        "seeds": seeds,
        "algo_notes": ALGO_NOTES,
        "datasets": {},
    }

    print(f"=== soft wiring vs SOTA  stages={args.stages} seeds={seeds} ===\n")
    for ds in datasets:
        print(f"\n######## {ds} ########", flush=True)
        block = {"uil": {}, "sota": {}}
        for arm in UIL_ARMS:
            print(f"\n--- UIL {arm} ---", flush=True)
            accs, mf1s, walls, gaps = [], [], [], []
            for seed in seeds:
                r = _uil_cell(ds, seed, args.stages, arm)
                accs.append(r["acc"]); mf1s.append(r["macro_f1"])
                walls.append(r["wall_s"]); gaps.append(r["gap"])
                print(
                    f"  seed {seed:>4}: acc={r['acc']:.4f}  "
                    f"mf1={r['macro_f1']:.4f}  gap={r['gap']:+.4f}  "
                    f"wall={r['wall_s']:.1f}s",
                    flush=True,
                )
            block["uil"][arm] = {
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs)),
                "mf1_mean": float(np.mean(mf1s)),
                "gap_mean": float(np.mean(gaps)),
                "wall_s": float(np.sum(walls)),
            }
            print(
                f"  >> {arm}: {block['uil'][arm]['acc_mean']:.4f}"
                f"+/-{block['uil'][arm]['acc_std']:.4f}",
                flush=True,
            )

        for arm in SOTA_ARMS:
            print(f"\n--- SOTA {arm} ---", flush=True)
            accs, mf1s, walls = [], [], []
            skipped = False
            for seed in seeds:
                r = _sota_cell(ds, seed, arm)
                if r.get("skipped"):
                    skipped = True
                    break
                accs.append(r["acc"]); mf1s.append(r["macro_f1"])
                walls.append(r["wall_s"])
                print(
                    f"  seed {seed:>4}: acc={r['acc']:.4f}  "
                    f"mf1={r['macro_f1']:.4f}  wall={r['wall_s']:.2f}s",
                    flush=True,
                )
            if skipped or not accs:
                block["sota"][arm] = {"skipped": True}
                print(f"  >> {arm}: SKIPPED", flush=True)
            else:
                block["sota"][arm] = {
                    "acc_mean": float(np.mean(accs)),
                    "acc_std": float(np.std(accs)),
                    "mf1_mean": float(np.mean(mf1s)),
                    "wall_s": float(np.sum(walls)),
                }
                print(
                    f"  >> {arm}: {block['sota'][arm]['acc_mean']:.4f}"
                    f"+/-{block['sota'][arm]['acc_std']:.4f}",
                    flush=True,
                )
        report["datasets"][ds] = block

    out_dir = results_path("soft_wiring_sota")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    _write_notes(report, out_dir)
    print(f"\nWrote {out_dir}/report.json and NOTES.md", flush=True)
    return 0


def _write_notes(report: dict, out_dir: str) -> None:
    lines = [
        "# Soft wiring vs SOTA — notes",
        "",
        f"Generated: {report['generated_at']}",
        f"Stages (UIL arms): {report['stages']}  |  Seeds: {report['seeds']}",
        "",
        "## What we tested",
        "",
        "Question: if we keep translator + connect + preprocessing, but drop "
        "the *aggressive* guidance (hard top-k / concentrate), how does accuracy "
        "compare to (a) hard wiring, (b) UIL off, (c) current strong ML baselines?",
        "",
        "## What each method is doing",
        "",
    ]
    for k, note in report["algo_notes"].items():
        lines.append(f"### `{k}`")
        lines.append("")
        lines.append(note)
        lines.append("")

    lines.extend([
        "## Results (held-out accuracy mean±std)",
        "",
    ])
    for ds, block in report["datasets"].items():
        lines.append(f"### {ds}")
        lines.append("")
        lines.append("| method | acc | macro-F1 | notes |")
        lines.append("|---|---:|---:|---|")
        for arm, r in block["uil"].items():
            lines.append(
                f"| **{arm}** | {r['acc_mean']:.4f}±{r['acc_std']:.3f} | "
                f"{r['mf1_mean']:.4f} | gap={r['gap_mean']:+.3f} |"
            )
        for arm, r in block["sota"].items():
            if r.get("skipped"):
                lines.append(f"| {arm} | — | — | skipped |")
            else:
                lines.append(
                    f"| {arm} | {r['acc_mean']:.4f}±{r['acc_std']:.3f} | "
                    f"{r['mf1_mean']:.4f} | single fit |"
                )
        # Deltas vs off and vs best SOTA
        off = block["uil"]["uil_off"]["acc_mean"]
        soft = block["uil"]["soft_wiring"]["acc_mean"]
        hard = block["uil"]["hard_wiring"]["acc_mean"]
        now = block["uil"]["translator_no_weights"]["acc_mean"]
        sota_accs = [
            (a, r["acc_mean"]) for a, r in block["sota"].items()
            if not r.get("skipped")
        ]
        best_sota = max(sota_accs, key=lambda x: x[1]) if sota_accs else ("?", float("nan"))
        lines.append("")
        lines.append(
            f"- soft − off = **{soft - off:+.4f}**; "
            f"hard − off = **{hard - off:+.4f}**; "
            f"no_weights − off = **{now - off:+.4f}**"
        )
        lines.append(
            f"- best SOTA = `{best_sota[0]}` at {best_sota[1]:.4f}; "
            f"soft − best_sota = **{soft - best_sota[1]:+.4f}**"
        )
        lines.append("")

    lines.extend([
        "## Takeaways (fill after run)",
        "",
        "- Soft wiring keeps connect live with mild |corr| priors instead of hard top-k.",
        "- SOTA GBMs (hist_gbm / xgboost) are the accuracy ceiling on saturated tabular; "
        "UIL's defensible edge remains readability / editability, not beating GBM everywhere.",
        "- Compare val−heldout **gap** on UIL arms: large positive gap = selection overfit "
        "(the hard-wiring failure mode).",
        "",
        "Raw: `report.json`",
        "",
    ])
    with open(os.path.join(out_dir, "NOTES.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
