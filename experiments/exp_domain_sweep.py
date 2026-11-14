"""Cross-domain 'reasoned vs bulldozer' sweep: XGBoost vs SymbolicFormula,
held-out accuracy, across real data of different TYPES plus mathematics.

Real domains (offline, from the webui registry -- UCI-derived):
  medical    real_breast_cancer, real_diabetes_risk
  chemistry  real_wine
  image      real_digits
  text       real_spambase       (email word/char frequencies -> spam)
  financial  real_credit_g
  signal     real_ionosphere, real_sonar

Mathematics (generated from real functions -- no fetch needed) is run TWICE:
  interp  random split           (test drawn from the training range)
  extrap  out-of-range split     (train on |x|<1, test on |x|>1)
Extrapolation is the regime where a tree-based bulldozer structurally
cannot win (it predicts a constant past its last split) and a symbolic
formula can -- the domain where 'reasoned' should categorically beat
'bulldozer'.

Cognitive layer is OFF (pure classifier comparison) so this isolates the
CLASSIFIER x DOMAIN effect. Metric: held-out accuracy (honest).

Usage:
    python -m figpso_uil.experiments.exp_domain_sweep --stages 60 --seeds 1,2
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import json
import statistics
import tempfile
import time
import shutil
from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from figpso_uil.webui.backend import datasets as dataset_registry
from figpso_uil.core.classifier_registry import CLASSIFIERS
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch
from figpso_uil.core.optimizer import UniversalOptimizer
from figpso_uil.cognition.paths import results_path

ABLATION_OFF = {"principles": False, "network": False, "formulas": False, "causal": False}
CLASSIFIERS_TO_TEST = ["XGBoost", "SymbolicFormula"]

# (domain-label, registry-name)
REAL_DOMAINS = [
    ("medical:breast_cancer", "real_breast_cancer"),
    ("medical:diabetes",      "real_diabetes_risk"),
    ("chemistry:wine",        "real_wine"),
    ("image:digits",          "real_digits"),
    ("text:spambase",         "real_spambase"),
    ("financial:credit_g",    "real_credit_g"),
    ("signal:sonar",          "real_sonar"),
]


class _Cfg:
    def __init__(self, stages, n_particles, pso_iters):
        self.stages = stages
        self.n_particles = n_particles
        self.pso_iters = pso_iters
        self.particles = n_particles
        self.pso_iters_per_stage = pso_iters
        self.use_resource_budget = False


# ── Mathematics datasets (real functions) ────────────────────────────
def _math_dataset(rule: str, n: int, seed: int, extrap: bool):
    """Binary classification by a real mathematical rule. When extrap,
    train points come from the inner range |x|<1 and test points from the
    outer shell 1<|x|<1.8 -- a genuine out-of-distribution split."""
    rng = np.random.default_rng(seed)

    def label(X):
        x1, x2 = X[:, 0], X[:, 1]
        # All three stay ~balanced in the outer shell (no degenerate
        # single-class extrapolation test).
        if rule == "xor_sign":   # sign(x1*x2) -- quadrant rule, axis-aligned
            return (x1 * x2 > 0).astype(int)
        if rule == "diagonal":   # x2 > x1 -- rotated linear boundary
            return (x2 > x1).astype(int)
        if rule == "wave":       # x2 > 0.5*sin(3*x1) -- curved, tree-hard OOD
            return (x2 > 0.5 * np.sin(3.0 * x1)).astype(int)
        raise ValueError(rule)

    if not extrap:
        X = rng.uniform(-1.0, 1.0, size=(n, 2))
        y = label(X)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    else:
        Xtr = rng.uniform(-1.0, 1.0, size=(n, 2))
        ytr = label(Xtr)
        # outer shell, radius in (1.0, 1.8): strictly outside training box corners too
        ang = rng.uniform(0, 2 * np.pi, size=n // 2)
        rad = rng.uniform(1.0, 1.8, size=n // 2)
        Xte = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
        yte = label(Xte)
    return Xtr, ytr, Xte, yte


def _load_domain(label, ref, seed):
    """Return (X_train, y_train, X_heldout, y_heldout) for one domain."""
    if isinstance(ref, tuple) and ref[0] == "math":
        _, rule, extrap = ref
        return _math_dataset(rule, n=400, seed=seed, extrap=extrap)
    X, y = dataset_registry.load_dataset(ref, seed=seed)
    Xtr, Xho, ytr, yho = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    return Xtr, ytr, Xho, yho


def _run(label, ref, classifier_name, seed, stages, n_particles, pso_iters, boosting):
    Xtr, ytr, Xho, yho = _load_domain(label, ref, seed)
    n_classes = len(np.unique(ytr))
    sd = tempfile.mkdtemp(prefix="uil_dom_")
    clf = CLASSIFIERS[classifier_name](seed, n_classes, boosting, state_dir=sd)
    if hasattr(clf, "params") and "n_jobs" in clf.params:
        clf.params["n_jobs"] = 1
    try:
        opt = UniversalOptimizer(cfg=_Cfg(stages, n_particles, pso_iters), seed=seed,
                                 dataset_name=label, fold=0, start_fold=0, classifier=clf,
                                 search=PrincipleGuidedSearch(), ablation=ABLATION_OFF, state_dir=sd)
        opt.fit(Xtr, ytr)
        return float(accuracy_score(yho, opt.predict(Xho)))
    finally:
        shutil.rmtree(sd, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", type=int, default=60)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--n-particles", type=int, default=8)
    ap.add_argument("--pso-iters", type=int, default=8)
    ap.add_argument("--boosting", type=int, default=30)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    domains = list(REAL_DOMAINS)
    for rule in ("xor_sign", "diagonal", "wave"):
        domains.append((f"math:{rule}:interp", ("math", rule, False)))
        domains.append((f"math:{rule}:extrap", ("math", rule, True)))

    print(f"Domain sweep  (held-out accuracy, cognitive layer OFF)  "
          f"stages={args.stages} seeds={seeds}")
    print(f"{'domain':28s} {'XGBoost':>10s} {'Symbolic':>10s} {'winner':>10s}  gap")
    print("-" * 74)

    report = {"generated_at": datetime.now().isoformat(), "stages": args.stages,
              "seeds": seeds, "results": {}}
    t0 = time.perf_counter()
    for label, ref in domains:
        row = {}
        for cname in CLASSIFIERS_TO_TEST:
            accs = [_run(label, ref, cname, s, args.stages, args.n_particles,
                         args.pso_iters, args.boosting) for s in seeds]
            row[cname] = statistics.fmean(accs)
        xg, sf = row["XGBoost"], row["SymbolicFormula"]
        gap = sf - xg
        winner = "Symbolic" if gap > 0.02 else ("XGBoost" if gap < -0.02 else "~tie")
        print(f"{label:28s} {xg:10.3f} {sf:10.3f} {winner:>10s}  {gap:+.3f}")
        report["results"][label] = {"XGBoost": xg, "SymbolicFormula": sf, "gap": gap, "winner": winner}

    report["wall_clock_seconds"] = time.perf_counter() - t0
    out = results_path(f"ablation_reports/domain_sweep_{args.stages}s.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nwall clock: {report['wall_clock_seconds']:.0f}s   saved: {out}")


if __name__ == "__main__":
    main()
