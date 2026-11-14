"""Canonical cross-seed evaluator — the ONLY sanctioned way to quote a
UIL+SymbolicFormula accuracy.

Finding (2026-07-18 production audit): single train/test splits OVERSTATE
(or understate) — ionosphere read 0.958 on one split but 0.916 +/- 0.030
across three. A single split is a lucky/unlucky sample of the evaluation,
fully determined by its seed, and must never be reported as "the" number.

This tool runs N seeds, each a deterministic (recorded, reproducible)
split, and reports mean +/- std of accuracy AND macro-F1 (macro-F1
surfaces class imbalance that accuracy hides). It also reports the
self-reported (internal val) vs held-out gap so optimism is visible.

SymbolicFormula only (standing directive). Isolated state_dir per run so
runs never share memory. No XGBoost.

Usage (from Downloads/, parent of figpso_uil):
    python -m figpso_uil.experiments.eval_crossseed
    python -m figpso_uil.experiments.eval_crossseed --datasets breast_cancer,digits --seeds 42,7,123
    python -m figpso_uil.experiments.eval_crossseed --stages 25
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

# Sklearn 1.8+ FutureWarnings (penalty/l1_ratio) and fingerprint-probe
# ConvergenceWarnings drown Gate-2 logs; they are not actionable here.
warnings.filterwarnings(
    "ignore", category=FutureWarning, module=r"sklearn\.linear_model\._logistic")
warnings.filterwarnings(
    "ignore", category=UserWarning, module=r"sklearn\.linear_model\._logistic")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from figpso_uil.core.optimizer import UniversalOptimizer
from figpso_uil.core.classifier_registry import CLASSIFIERS
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch


class _Cfg:
    def __init__(self, stages, particles, pso_iters, boosting,
                 search_mode="classic", race_incumbent_margin=0.0,
                 utility_gated_formulas=False,
                 maturity_soft_normalization=False,
                 stress_adaptive_blend=False,
                 situation_adaptive=False,
                 use_ebm_fit=False,
                 use_kan_basis=False,
                 uil_enabled=True,
                 uil_connect_pack=False,
                 signal_translator=False,
                 connect_cognition=False):
        # "classic" = PSO/principle-guided surrogate search (default).
        # "race"    = architecture race (search/architecture_race.py).
        self.search_mode = search_mode
        # Challenger must beat incumbent by this margin on select_score
        # (race only). 0 = plain argmax. Gate 2 candidate: 0.01–0.02.
        self.race_incumbent_margin = float(race_incumbent_margin)
        self.utility_gated_formulas = bool(utility_gated_formulas)
        self.maturity_soft_normalization = bool(maturity_soft_normalization)
        self.stress_adaptive_blend = bool(stress_adaptive_blend)
        # Default ON after Category D Gate 2 (2026-07-24). Eval harness
        # still defaults False so classic A/B stays comparable; pass
        # --situation-adaptive to exercise the fingerprint policy.
        self.situation_adaptive = bool(situation_adaptive)
        # Research step 5 — EBM cyclic fit. Default OFF pending Gate 2.
        self.use_ebm_fit = bool(use_ebm_fit)
        # Research step 8 — KAN truncated-power basis. Default OFF pending Gate 2.
        self.use_kan_basis = bool(use_kan_basis)
        # Master UIL switch + experimental connect pack (core/uil_mode.py).
        self.uil_enabled = bool(uil_enabled)
        self.uil_connect_pack = bool(uil_connect_pack)
        # Individual core-wiring knobs (for diagnosis / ablation). Pack
        # forces these ON; setting them here arms them without the pack.
        self.signal_translator = bool(signal_translator)
        self.connect_cognition = bool(connect_cognition)
        self.explore_architecture = False
        self.adaptive_min_evidence = False
        self.stages = stages
        self.n_particles = particles
        self.pso_iters = pso_iters
        self.particles = particles
        self.pso_iters_per_stage = pso_iters
        self.boosting_rounds = boosting
        self.cv_folds = 3            # auto-scales down on large data (optimizer)
        self.use_resource_budget = False
        self.disable_routing = True  # keep SymbolicFormula
        self.enable_stagnation_evolution = True
        self.target_accuracy = None
        self.max_stages = stages


# dataset -> (loader, recommended symbolic mode)
def _load_breast():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(); return np.asarray(d.data, float), np.asarray(d.target, int)
def _load_wine():
    from sklearn.datasets import load_wine
    d = load_wine(); return np.asarray(d.data, float), np.asarray(d.target, int)
def _load_digits():
    from sklearn.datasets import load_digits
    d = load_digits(); return np.asarray(d.data, float), np.asarray(d.target, int)
def _openml(name=None, data_id=None):
    import pandas as pd
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder
    b = (fetch_openml(name=name, version=1, as_frame=True, parser="auto") if name
         else fetch_openml(data_id=data_id, as_frame=True, parser="auto"))
    df = b.frame.copy()
    tcol = b.target_names[0] if b.target_names else df.columns[-1]
    y = LabelEncoder().fit_transform(df[tcol].astype(str))
    X = pd.get_dummies(df.drop(columns=[tcol]), drop_first=False).to_numpy(dtype=np.float64)
    m = np.isfinite(X).all(axis=1)
    return X[m], np.asarray(y, dtype=np.int64)[m]

DATASETS = {
    "breast_cancer": (_load_breast, "sparse_linear"),
    "ionosphere":    (lambda: _openml(data_id=59), "sparse_linear"),
    "sonar":         (lambda: _openml(name="sonar"), "boosting"),  # boosting 0.746 > sparse_linear+rules 0.722 through the full optimizer
    "wine":          (_load_wine, "sparse_linear"),
    "digits":        (_load_digits, "sparse_linear"),
    "spambase":      (lambda: _openml(data_id=44), "sparse_linear"),
}


def one_split(X, y, mode, seed, stages, particles, pso_iters, boosting, n_classes,
              search_mode="classic", race_incumbent_margin=0.0,
              utility_gated_formulas=False, maturity_soft_normalization=False,
              stress_adaptive_blend=False, situation_adaptive=False,
              use_ebm_fit=False, use_kan_basis=False,
              uil_enabled=True, uil_connect_pack=False,
              signal_translator=False, connect_cognition=False):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    sd = tempfile.mkdtemp(prefix=f"eval_{seed}_")
    clf = CLASSIFIERS["SymbolicFormula"](seed, n_classes, boosting, sd)
    if hasattr(clf, "apply_config"):
        clf.apply_config({"learning_mode": mode})
    opt = UniversalOptimizer(
        cfg=_Cfg(stages, particles, pso_iters, boosting, search_mode,
                 race_incumbent_margin=race_incumbent_margin,
                 utility_gated_formulas=utility_gated_formulas,
                 maturity_soft_normalization=maturity_soft_normalization,
                 stress_adaptive_blend=stress_adaptive_blend,
                 situation_adaptive=situation_adaptive,
                 use_ebm_fit=use_ebm_fit,
                 use_kan_basis=use_kan_basis,
                 uil_enabled=uil_enabled,
                 uil_connect_pack=uil_connect_pack,
                 signal_translator=signal_translator,
                 connect_cognition=connect_cognition),
        seed=seed,
        dataset_name=f"eval_{seed}", fold=0, start_fold=0, classifier=clf,
        search=PrincipleGuidedSearch(),
        classifier_factory=CLASSIFIERS["SymbolicFormula"],
        classifier_build_args=(seed, n_classes, boosting, sd), state_dir=sd)
    opt._observability_meta["source"] = "terminal"
    t0 = time.perf_counter()
    result = opt.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    yp = opt.predict(Xte)
    return {
        "acc": accuracy_score(yte, yp),
        "macro_f1": f1_score(yte, yp, average="macro", zero_division=0),
        "internal_val": float(result.get("accuracy") or 0.0),
        "fit_s": fit_s,
        "observability": dict(opt._observability_meta or {}),
    }


def evaluate(name, seeds, stages, particles, pso_iters, boosting,
             search_mode="classic", race_incumbent_margin=0.0,
             utility_gated_formulas=False, maturity_soft_normalization=False,
             stress_adaptive_blend=False, situation_adaptive=False,
             use_ebm_fit=False, use_kan_basis=False,
             uil_enabled=True, uil_connect_pack=False, force_mode=None,
             signal_translator=False, connect_cognition=False):
    loader, mode = DATASETS[name]
    if force_mode:
        mode = force_mode
    X, y = loader(); X = np.asarray(X, float); y = np.asarray(y, int)
    n_classes = int(len(np.unique(y)))
    accs, mf1s, gaps, walls, per_seed = [], [], [], [], []
    for s in seeds:
        r = one_split(X, y, mode, s, stages, particles, pso_iters, boosting,
                      n_classes, search_mode, race_incumbent_margin,
                      utility_gated_formulas, maturity_soft_normalization,
                      stress_adaptive_blend, situation_adaptive,
                      use_ebm_fit, use_kan_basis,
                      uil_enabled, uil_connect_pack,
                      signal_translator, connect_cognition)
        accs.append(r["acc"]); mf1s.append(r["macro_f1"])
        gaps.append(r["internal_val"] - r["acc"]); walls.append(r["fit_s"])
        # Per-(dataset, seed) rows are written to --json. Gate 2 artifacts
        # that stored only arm MEANS cannot be re-analysed later: three past
        # verdicts are unreproducible for exactly this reason (see
        # experiments/rescore_gates.py). Never store only the mean again.
        per_seed.append({
            "dataset": name, "seed": s, "acc": float(r["acc"]),
            "macro_f1": float(r["macro_f1"]),
            "internal_val": float(r["internal_val"]),
            "val_gap": float(r["internal_val"] - r["acc"]),
            "fit_s": float(r["fit_s"]),
        })
        print(f"    seed {s:>4}: acc={r['acc']:.4f}  macroF1={r['macro_f1']:.4f}"
              f"  fit={r['fit_s']:.1f}s", flush=True)
    accs, mf1s = np.array(accs), np.array(mf1s)
    return {
        "name": name, "mode": mode, "n_classes": n_classes,
        "shape": f"{X.shape[0]}x{X.shape[1]}",
        "acc_mean": accs.mean(), "acc_std": accs.std(),
        "mf1_mean": mf1s.mean(), "mf1_std": mf1s.std(),
        "gap": float(np.mean(gaps)),
        "fit_s_mean": float(np.mean(walls)), "fit_s_total": float(np.sum(walls)),
        "per_seed": per_seed,
    }


def main():
    ap = argparse.ArgumentParser(description="Cross-seed UIL+SymbolicFormula evaluator")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", default="42,7,123")
    ap.add_argument("--stages", type=int, default=20)
    ap.add_argument("--particles", type=int, default=10)
    ap.add_argument("--pso-iters", type=int, default=10)
    ap.add_argument("--boosting", type=int, default=40)
    ap.add_argument("--search-mode", default="classic",
                    choices=["classic", "race"],
                    help="classic = PSO surrogate search; race = architecture race")
    ap.add_argument("--race-incumbent-margin", type=float, default=0.0,
                    help="race only: challenger must beat incumbent by this "
                         "select_score margin (0 = plain argmax)")
    ap.add_argument("--utility-gated-formulas", action="store_true",
                    help="research step 4: DreamCoder utility gate + newborn L1")
    ap.add_argument("--maturity-soft-normalization", action="store_true",
                    help="Category D: soft count normalization for maturity blend")
    ap.add_argument("--stress-adaptive-blend", action="store_true",
                    help="Category D: drop dead stress terms and renormalize")
    ap.add_argument("--situation-adaptive", action="store_true",
                    help="auto-enable Category D + explore_architecture from fingerprint")
    ap.add_argument("--use-ebm-fit", action="store_true",
                    help="research step 5: EBM cyclic fit (sparse_linear path only)")
    ap.add_argument("--use-kan-basis", action="store_true",
                    help="research step 8: KAN truncated-power basis (sparse_linear)")
    ap.add_argument("--uil-off", action="store_true",
                    help="master switch: disable all UIL cognition (learner+search only)")
    ap.add_argument("--uil-connect", action="store_true",
                    help="experimental connect pack: connect_cognition + explore_architecture "
                         "+ adaptive_min_evidence + utility_gated_formulas + signal_translator")
    ap.add_argument("--json", default=None,
                    help="write per-(dataset, seed) rows here so the run can "
                         "be re-analysed later (paired stats, power)")
    ap.add_argument("--force-mode", default=None,
                    choices=["sparse_linear", "boosting", "bagging", "multi", "single", "auto"],
                    help="override per-dataset learning_mode (needed for EBM/KAN on sonar)")
    args = ap.parse_args()

    names = [d.strip() for d in args.datasets.split(",") if d.strip() in DATASETS]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    uil_enabled = not bool(args.uil_off)
    uil_connect = bool(args.uil_connect) and uil_enabled

    rows = []
    for name in names:
        print(f"\n=== {name} ({len(seeds)} seeds) ===", flush=True)
        try:
            rows.append(evaluate(name, seeds, args.stages, args.particles,
                                  args.pso_iters, args.boosting,
                                  args.search_mode,
                                  args.race_incumbent_margin,
                                  args.utility_gated_formulas,
                                  args.maturity_soft_normalization,
                                  args.stress_adaptive_blend,
                                  args.situation_adaptive,
                                  args.use_ebm_fit,
                                  args.use_kan_basis,
                                  uil_enabled,
                                  uil_connect,
                                  args.force_mode))
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", flush=True)
            rows.append({"name": name, "error": str(e)})

    extras = []
    if args.search_mode == "race" and args.race_incumbent_margin:
        extras.append(f"race_incumbent_margin={args.race_incumbent_margin}")
    if args.utility_gated_formulas:
        extras.append("utility_gated_formulas=ON")
    if args.maturity_soft_normalization:
        extras.append("maturity_soft_normalization=ON")
    if args.stress_adaptive_blend:
        extras.append("stress_adaptive_blend=ON")
    if args.situation_adaptive:
        extras.append("situation_adaptive=ON")
    if args.use_ebm_fit:
        extras.append("use_ebm_fit=ON")
    if args.use_kan_basis:
        extras.append("use_kan_basis=ON")
    if args.uil_off:
        extras.append("uil_enabled=OFF")
    if uil_connect:
        extras.append("uil_connect_pack=ON")
    if args.force_mode:
        extras.append(f"force_mode={args.force_mode}")
    extra_note = (", " + ", ".join(extras)) if extras else ""
    print(f"\n{'='*80}\nCROSS-SEED RESULTS (mean +/- std over {len(seeds)} seeds) — "
          f"SymbolicFormula only, search_mode={args.search_mode}{extra_note}"
          f"\n{'='*80}",
          flush=True)
    hdr = (f"{'dataset':<15}{'shape':>12}{'cls':>4}{'mode':>14}{'accuracy':>18}"
           f"{'macroF1':>16}{'val-gap':>9}{'fit_s':>9}")
    print(hdr, flush=True); print("-"*len(hdr), flush=True)
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<15}  ERROR — {r['error']}", flush=True); continue
        acc = f"{r['acc_mean']:.4f}+/-{r['acc_std']:.4f}"
        mf1 = f"{r['mf1_mean']:.4f}+/-{r['mf1_std']:.4f}"
        print(f"{r['name']:<15}{r['shape']:>12}{r['n_classes']:>4}{r['mode']:>14}"
              f"{acc:>18}{mf1:>16}{r['gap']:>+9.4f}{r['fit_s_mean']:>9.1f}", flush=True)
    total = sum(r.get("fit_s_total", 0.0) for r in rows)
    print(f"\nTotal fit wall-clock: {total:.1f}s", flush=True)
    print("Note: quote the mean+/-std, never a single split. A positive val-gap "
          "means the run's self-reported accuracy is OPTIMISTIC vs held-out.", flush=True)

    if args.json:
        payload = {
            "args": vars(args),
            "seeds": seeds,
            "rows": [
                {k: v for k, v in r.items() if k != "per_seed"} for r in rows
            ],
            "per_seed": [p for r in rows for p in r.get("per_seed", [])],
            "total_fit_s": total,
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=float)
        print(f"Wrote per-seed rows to {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
