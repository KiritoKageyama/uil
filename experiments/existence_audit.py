"""Existence audit — is each cognitive subsystem CONNECTED at all?

This is deliberately NOT an accuracy ablation. That question is settled and
the answer is zero (see "Honest empirical status" in CLAUDE.md). This asks
something blunter and much harder to argue with:

    If I remove this subsystem, does ANY bit of the output change?

Why this works, and why it does not need the validation-split fix first:
UIL is deterministic (same seed -> byte-identical predictions, verified).
So this is an IDENTITY test, not a comparison of two noisy scores. There
is no confidence interval to argue about and no winner's curse to correct
for. Either the output moved or it did not.

    output identical  -> INERT. The subsystem ran, spent compute, wrote
                         state, and reached the model with exactly nothing.
    output differs    -> CONNECTED. It influences the result. Whether it
                         influences it for the BETTER is a separate
                         question this audit deliberately does not ask.

Bug #25 established that the principle library is inert on the sparse_linear
path. This checks whether it is the only one.

Run:
    python -m figpso_uil.experiments.existence_audit
    python -m figpso_uil.experiments.existence_audit --dataset wine --stages 6

Isolation note: each arm runs in its OWN PROCESS. `state_dir` alone is not
enough -- several components reach module-level singletons (`KnowledgeArchive
.shared()`, `FormulaChangeLog.shared()`, the `VersionStore` fallback in
`_cell_store`) that ignore it, so two supposedly isolated in-process runs are
byte-identical at 4 stages and DIVERGE at 5, when the prune/decay cadence
first touches shared state. A separate process is the only isolation that
currently holds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from figpso_uil.adapters.symbolic_formula_adapter import SymbolicFormulaAdapter
from figpso_uil.core.optimizer import UniversalOptimizer
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch

# The subsystems the optimizer can gate. The last five were added
# 2026-07-23; before that they ran on every fit with no way to remove
# them, so their connectedness was simply unknown.
#
# `structural` is a prerequisite for `meta_principles` and `safeguards`
# (both consume StructuralPrinciple objects), so removing it necessarily
# removes their input as well. That is reported as-is rather than papered
# over -- a switch that only half-removes a subsystem is worse than none.
SUBSYSTEMS = (
    "principles",
    "network",
    "formulas",
    "causal",
    "episodic",
    "structural",
    "meta_principles",
    "safeguards",
    "goal",
)


class Cfg:
    stages = 5
    n_particles = 8
    pso_iters = 8
    use_resource_budget = False
    disable_routing = True
    cv_folds = 2
    # Research step 4 — DreamCoder utility gate. Default off so the
    # historical audit (formulas INERT) still reproduces; pass
    # --utility-gated to measure whether the gate flips CONNECTED.
    utility_gated_formulas = False
    uil_enabled = True
    uil_connect_pack = False


def _load(name: str):
    from sklearn import datasets as d
    # MEASURED SCOPE: determinism (and therefore this audit) currently holds
    # only for BINARY targets under sparse_linear. The multiclass OvR path is
    # nondeterministic -- wine (3-class) fails the control at 6 stages, with
    # both the predictions AND the selected-feature count moving between two
    # identical runs. So the *_binary variants exist to keep the identity
    # test valid on data that is otherwise multiclass.
    loaders = {
        "breast_cancer": d.load_breast_cancer,
        "wine": d.load_wine,
        "digits": d.load_digits,
        "iris": d.load_iris,
    }
    if name.endswith("_binary"):
        base = name[: -len("_binary")]
        if base not in loaders:
            raise SystemExit(f"unknown dataset {name!r}")
        X, y = loaders[base](return_X_y=True)
        return X, (np.asarray(y) == 0).astype(int)
    if name not in loaders:
        raise SystemExit(f"unknown dataset {name!r}; try {list(loaders)}")
    return loaders[name](return_X_y=True)


def _fingerprint(opt: UniversalOptimizer, result: dict, X_probe: np.ndarray) -> dict:
    """A hashable summary of everything the run produced.

    Carries the raw probe predictions as well as their hash. The hash answers
    "did anything change" (INERT vs CONNECTED); the raw vector answers "HOW
    MUCH changed", which is the difference between a subsystem that moves one
    borderline row and one that rewrites the model. A binary verdict cannot
    tell those apart, and for this project's central negative result the
    distinction is the whole question.
    """
    model = result.get("model")
    pred_list = None
    try:
        preds = opt.predict(X_probe)
        pred_hash = hashlib.sha256(
            np.asarray(preds).astype(str).tobytes()
        ).hexdigest()[:16]
        pred_list = [str(v) for v in np.asarray(preds).ravel().tolist()]
    except Exception as exc:  # a run that cannot predict is itself a finding
        pred_hash = f"ERR:{type(exc).__name__}"
    weights = result.get("feature_weights")
    return {
        "preds": pred_list,
        "predictions": pred_hash,
        "accuracy": round(float(result.get("accuracy", 0.0)), 6),
        "macro_f1": round(float(result.get("macro_f1") or 0.0), 6),
        "n_selected": int(result.get("n_selected") or 0),
        "weights": hashlib.sha256(
            np.asarray(weights, dtype=float).round(9).tobytes()
        ).hexdigest()[:16] if weights is not None else None,
        "config": json.dumps(result.get("best_config") or {}, sort_keys=True,
                             default=str)[:400],
        "model_kind": (model or {}).get("kind") if isinstance(model, dict) else None,
    }


#: Fingerprint keys that are graded numerically rather than by equality.
_GRADED = ("preds", "accuracy", "macro_f1", "n_selected")


def grade(base: dict, off: dict) -> dict:
    """How MUCH did removing the subsystem move the output?

    `changed` keeps the old binary semantics (which fingerprint fields
    differ). `pred_frac_changed` is the graded one: the share of probe rows
    whose predicted label moved. INERT is exactly pred_frac_changed == 0 AND
    no other field moved.
    """
    changed = [k for k in base if k != "preds" and base[k] != off[k]]
    bp, op = base.get("preds"), off.get("preds")
    if bp and op and len(bp) == len(op):
        frac = sum(1 for a, b in zip(bp, op) if a != b) / len(bp)
    else:
        frac = None
    return {
        "changed": changed,
        "pred_frac_changed": frac,
        "d_accuracy": round(float(off.get("accuracy", 0.0))
                            - float(base.get("accuracy", 0.0)), 6),
        "d_macro_f1": round(float(off.get("macro_f1") or 0.0)
                            - float(base.get("macro_f1") or 0.0), 6),
        "d_n_selected": int(off.get("n_selected") or 0)
                        - int(base.get("n_selected") or 0),
        "verdict": "INERT" if (not changed and not frac) else "CONNECTED",
    }


def _pinned_classifier(seed: int, n_classes: int, state_dir: str):
    """SymbolicFormula pinned to sparse_linear.

    MEASURED: the identity test is only valid in a deterministic
    configuration, and determinism here is MODE-DEPENDENT. Two identical
    5-stage runs -- same seed, fresh state_dir, separate processes -- are
    byte-identical under `sparse_linear` and DIVERGE under `auto` (which
    resolves to the hybrid/multi path). So `auto` cannot be used for this
    audit, and any A/B run at `auto` is comparing partly-random arms.
    """
    clf = SymbolicFormulaAdapter(
        seed=seed, num_classes=n_classes,
        formula_library_path=os.path.join(
            state_dir, "uil_symbolic_classifier_library.json"),
    )
    clf.apply_config({"learning_mode": "sparse_linear"})
    return clf


def _run_arm_inprocess(dataset: str, seed: int, ablation: dict, stages: int,
                       utility_gated: bool = False,
                       uil_connect_pack: bool = False) -> dict:
    X, y = _load(dataset)
    cfg = Cfg()
    cfg.stages = stages
    cfg.utility_gated_formulas = bool(utility_gated)
    cfg.uil_connect_pack = bool(uil_connect_pack)
    state_dir = tempfile.mkdtemp(prefix="uil_audit_")
    opt = UniversalOptimizer(
        cfg=cfg, seed=seed, dataset_name="audit",
        fold=0, start_fold=0,
        classifier=_pinned_classifier(seed, len(np.unique(y)), state_dir),
        search=PrincipleGuidedSearch(),
        ablation=ablation,
        state_dir=state_dir,
    )
    result = opt.fit(X, y)
    return _fingerprint(opt, result, X[:200])


def run_arm(dataset: str, seed: int, ablation: dict, stages: int, tag: str,
            utility_gated: bool = False,
            uil_connect_pack: bool = False) -> dict:
    """Run one arm in a FRESH PROCESS and return its fingerprint.

    Same dataset_name for every arm, so nothing keys off the label; the only
    difference between arms is the ablation dict.
    """
    payload = json.dumps({
        "dataset": dataset, "seed": seed,
        "ablation": ablation, "stages": stages,
        "utility_gated": bool(utility_gated),
        "uil_connect_pack": bool(uil_connect_pack),
    })
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "-m", "figpso_uil.experiments.existence_audit",
         "--arm", payload],
        capture_output=True, text=True, env=env,
    )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__ARM__"):
            return json.loads(line[len("__ARM__"):])
    raise RuntimeError(
        f"arm {tag} produced no result\nstdout:\n{proc.stdout[-2000:]}"
        f"\nstderr:\n{proc.stderr[-2000:]}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="breast_cancer")
    ap.add_argument("--stages", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arm", default=None, help="internal: run one arm")
    ap.add_argument("--utility-gated", action="store_true",
                    help="enable cfg.utility_gated_formulas (research step 4)")
    ap.add_argument("--uil-connect", action="store_true",
                    help="enable experimental uil_connect_pack (learner channels)")
    ap.add_argument("--only", default=None,
                    help="comma-separated subsystem names to audit "
                         "(default: all)")
    ap.add_argument("--json", default=None, help="write graded results here")
    args = ap.parse_args()

    if args.arm:  # child process: run a single arm, print its fingerprint
        spec = json.loads(args.arm)
        fp = _run_arm_inprocess(
            spec["dataset"], spec["seed"], spec["ablation"], spec["stages"],
            utility_gated=bool(spec.get("utility_gated", False)),
            uil_connect_pack=bool(spec.get("uil_connect_pack", False)),
        )
        print("__ARM__" + json.dumps(fp))
        return 0

    all_on = {k: True for k in SUBSYSTEMS}
    only = ([s.strip() for s in args.only.split(",") if s.strip()]
            if args.only else list(SUBSYSTEMS))
    for s in only:
        if s not in SUBSYSTEMS:
            raise SystemExit(f"unknown subsystem {s!r}; try {list(SUBSYSTEMS)}")

    connect = bool(args.uil_connect)
    gated_note = ""
    if args.utility_gated:
        gated_note += " utility_gated=ON"
    if connect:
        gated_note += " uil_connect_pack=ON"
    print(f"=== Existence audit: {args.dataset}, {args.stages} stages, "
          f"seed {args.seed}{gated_note} ===\n")

    # CONTROL. If two identical runs disagree, the whole audit is void --
    # every "difference" below would be indistinguishable from run noise.
    base = run_arm(args.dataset, args.seed, dict(all_on), args.stages, "base",
                   utility_gated=args.utility_gated,
                   uil_connect_pack=connect)
    base2 = run_arm(args.dataset, args.seed, dict(all_on), args.stages, "base2",
                    utility_gated=args.utility_gated,
                    uil_connect_pack=connect)
    if base != base2:
        print("CONTROL FAILED — two identical runs produced different output.")
        print("  The audit cannot distinguish 'inert' from 'nondeterministic'.")
        for k in base:
            if base[k] != base2[k]:
                print(f"    {k}: {base[k]!r} vs {base2[k]!r}")
        return 2
    print("control: two identical runs agree exactly -> determinism holds\n")

    rows = []
    for sub in only:
        ablation = dict(all_on)
        ablation[sub] = False
        off = run_arm(args.dataset, args.seed, ablation, args.stages, f"no_{sub}",
                      utility_gated=args.utility_gated,
                      uil_connect_pack=connect)
        rows.append((sub, grade(base, off), off))

    width = max(len(s) for s in only)
    print(f"{'subsystem':>{width}}  {'verdict':<10}{'preds moved':>12}"
          f"{'d_acc':>9}  what changed")
    print("-" * 84)
    inert = []
    for sub, g, off in rows:
        frac = g["pred_frac_changed"]
        fstr = "—" if frac is None else f"{100.0 * frac:.1f}%"
        if g["verdict"] == "INERT":
            inert.append(sub)
            print(f"{sub:>{width}}  {'INERT':<10}{fstr:>12}{0.0:>+9.4f}  "
                  f"nothing — output byte-identical")
        else:
            print(f"{sub:>{width}}  {'CONNECTED':<10}{fstr:>12}"
                  f"{g['d_accuracy']:>+9.4f}  {', '.join(g['changed'])}")
    print("-" * 84)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({
                "dataset": args.dataset, "stages": args.stages,
                "seed": args.seed, "base": {k: v for k, v in base.items()
                                            if k != "preds"},
                "rows": [{"subsystem": s, **g} for s, g, _ in rows],
            }, fh, indent=2)
        print(f"wrote {args.json}")
    if inert:
        print(f"\n{len(inert)}/{len(only)} subsystems are INERT: {', '.join(inert)}")
        print("They consume compute and write state without reaching the model.")
    else:
        print("\nAll gated subsystems are connected (this says nothing about "
              "whether they HELP).")
    print("\nNote: `structural` is a prerequisite for `meta_principles` and "
          "`safeguards`\n(both consume StructuralPrinciple objects), so a "
          "CONNECTED verdict on structural\nmay be inherited from either of "
          "them. Read those three rows together.")
    print("A subsystem with no active input this run (e.g. `goal` with no "
          "goal set, or\n`meta_principles` under 5 principles) reads INERT "
          "for that reason, not because\nits wiring is broken — check the run "
          "before drawing a conclusion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
