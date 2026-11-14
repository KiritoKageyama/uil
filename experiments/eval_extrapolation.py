"""Cross-seed out-of-range evaluation for the symbolic regressor.

WHY THIS EXISTS
---------------
"Symbolic categorically beats XGBoost out-of-range (R2 0.99 vs -5.5)" is the
project's one claimed POSITIVE result -- the domain where the reasoned learner
is supposed to win. Its evidence base is the weakest of any claim here:

  * single runs, never passed through the cross-seed protocol that every
    accuracy claim is required to use;
  * benchmarked against XGBoost alone.

The second point is the fatal one. Gradient-boosted trees predict a constant
outside the training range -- every leaf saturates -- so a large negative R2
out of range is a THEOREM about trees, not a discovery about UIL. Beating
XGBoost at extrapolation demonstrates nothing.

The claim only survives against baselines that CAN extrapolate:

  * `linear`     -- ordinary least squares. Extrapolates perfectly on any
                    linear law, and many physical laws are linear in some
                    variable. If linear ties the symbolic regressor, the
                    "symbolic wins" story is really "trees lose".
  * `poly2`      -- degree-2 ridge. A closed-form interpretable model with
                    genuine curvature.
  * `mlp`        -- a small neural net: non-linear, unbounded, and able to
                    extrapolate (badly, but not constantly like a tree).
  * `xgboost`    -- kept only to QUANTIFY the strawman, not as a rival.

PROTOCOL
--------
A genuine range split, not a random one: fit on the inner region
(|x| <= inner quantile), test on the outer shell only. Reported as mean +/-
std of out-of-range R2 over N seeds, plus in-range R2 so interpolation and
extrapolation are never conflated.

Run:
    python -m figpso_uil.experiments.eval_extrapolation
    python -m figpso_uil.experiments.eval_extrapolation --seeds 42,7,123,5,99 \
        --json results/extrapolation/report.json
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from figpso_uil.cognition.symbolic_regressor import SymbolicRegressor


# --------------------------------------------------------------------------
# tasks — physics-style laws (the SRBench/Feynman convention), plus controls
# --------------------------------------------------------------------------

def _law_linear(X):        return 3.0 * X[:, 0] - 2.0 * X[:, 1] + 1.5
def _law_poly(X):          return 2.0 * X[:, 0] ** 2 - 1.2 * X[:, 1] + 0.4
def _law_product(X):       return X[:, 0] * X[:, 1]
def _law_ratio(X):         return X[:, 0] / (1.0 + X[:, 1] ** 2)
def _law_kinetic(X):       return 0.5 * X[:, 0] * X[:, 1] ** 2          # E = 1/2 m v^2
def _law_gravity(X):       return X[:, 0] * X[:, 1] / (X[:, 2] ** 2 + 0.5)
def _law_harmonic(X):      return np.sin(1.5 * X[:, 0]) + 0.5 * X[:, 1]

TASKS = {
    "linear":   (_law_linear, 2),
    "poly":     (_law_poly, 2),
    "product":  (_law_product, 2),
    "ratio":    (_law_ratio, 2),
    "kinetic":  (_law_kinetic, 2),
    "gravity":  (_law_gravity, 3),
    "harmonic": (_law_harmonic, 2),
}


def make_split(task: str, seed: int, n: int = 600, noise: float = 0.02,
               inner: float = 1.0, outer: float = 2.0):
    """Train INSIDE the inner box, test in the OUTER SHELL only.

    `outer/inner` is the extrapolation REACH, and it is the axis that
    matters. The project's original claim (R2 0.992 vs XGBoost -5.52,
    exp_feynman_discovery.py) was measured at a reach of roughly 1.1-1.5 in
    ONE or TWO dimensions, on ONE side of the training range, at ONE seed.
    Sweeping the reach shows where recovery stops being recovery.
    """
    fn, d = TASKS[task]
    rng = np.random.default_rng(seed)
    # Inner: uniform in [-inner, inner]^d.
    Xtr = rng.uniform(-inner, inner, size=(4 * n, d))
    # Outer shell: uniform in [-outer, outer]^d, keep only points that fall
    # OUTSIDE the training box on at least one axis. That is the whole test.
    cand = rng.uniform(-outer, outer, size=(20 * n, d))
    shell = cand[np.max(np.abs(cand), axis=1) > inner]
    Xte_out = shell[:n]
    Xte_in = rng.uniform(-inner, inner, size=(n, d))

    scale = float(np.std(fn(Xtr))) or 1.0
    ytr = fn(Xtr) + rng.normal(0, noise * scale, len(Xtr))
    return Xtr[:n], ytr[:n], Xte_in, fn(Xte_in), Xte_out, fn(Xte_out)


# --------------------------------------------------------------------------
# learners
# --------------------------------------------------------------------------

def _fit_predict(name: str, Xtr, ytr, Xa, Xb, seed: int):
    if name == "uil_symbolic":
        m = SymbolicRegressor(seed=seed, n_terms=10).fit(Xtr, ytr)
        return m.predict(Xa), m.predict(Xb), m.formula_str()[:160]
    if name == "linear":
        from sklearn.linear_model import LinearRegression
        m = LinearRegression().fit(Xtr, ytr)
        return m.predict(Xa), m.predict(Xb), "OLS"
    if name == "poly2":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures
        m = make_pipeline(PolynomialFeatures(2), Ridge(alpha=1e-3)).fit(Xtr, ytr)
        return m.predict(Xa), m.predict(Xb), "degree-2 ridge"
    if name == "mlp":
        from sklearn.neural_network import MLPRegressor
        m = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000,
                         random_state=seed).fit(Xtr, ytr)
        return m.predict(Xa), m.predict(Xb), "MLP(64,64)"
    if name == "gplearn":
        # The genuine symbolic rival. Same family as UIL's regressor (evolved
        # expression trees), so this separates "symbolic regression is the
        # wrong tool here" from "this implementation is weak" -- the question
        # LITERATURE_SURVEY.md line 375 asks for.
        # `div` is protected in gplearn (returns 1.0 for |denom| < 0.001),
        # which is exactly the guard UIL's regressor lacks; keeping the
        # default function set makes that difference part of the measurement.
        from gplearn.genetic import SymbolicRegressor as GPSymbolic
        m = GPSymbolic(population_size=1000, generations=20,
                       function_set=("add", "sub", "mul", "div", "sqrt",
                                     "log", "sin", "cos"),
                       parsimony_coefficient=0.001, random_state=seed,
                       n_jobs=1, verbose=0).fit(Xtr, ytr)
        return m.predict(Xa), m.predict(Xb), str(m._program)[:160]
    if name == "xgboost":
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.1,
                         random_state=seed, verbosity=0).fit(Xtr, ytr)
        return m.predict(Xa), m.predict(Xb), "XGBRegressor"
    raise ValueError(name)


LEARNERS = ("uil_symbolic", "gplearn", "linear", "poly2", "mlp", "xgboost")


def r2(y, p) -> float:
    y = np.asarray(y, float); p = np.asarray(p, float)
    if not np.isfinite(p).all():
        return float("nan")
    ss = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((y - p) ** 2) / ss) if ss > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--seeds", default="42,7,123,5,99")
    ap.add_argument("--learners", default=",".join(LEARNERS))
    ap.add_argument("--reach", default="1.2,1.5,2.0",
                    help="outer/inner extrapolation reach values to sweep")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip() in TASKS]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    learners = [l.strip() for l in args.learners.split(",") if l.strip() in LEARNERS]
    reaches = [float(r) for r in args.reach.split(",") if r.strip()]

    rows, per_run = [], []
    for reach in reaches:
      for task in tasks:
        print(f"\n=== {task} @ reach {reach}x ({len(seeds)} seeds) ===", flush=True)
        acc: dict[str, dict[str, list]] = {
            l: {"out": [], "in": [], "t": [], "blow": 0} for l in learners}
        for seed in seeds:
            Xtr, ytr, Xin, yin, Xout, yout = make_split(
                task, seed, inner=1.0, outer=reach)
            yscale = float(np.std(ytr)) or 1.0
            for l in learners:
                t0 = time.perf_counter()
                try:
                    p_in, p_out, desc = _fit_predict(l, Xtr, ytr, Xin, Xout, seed)
                    ri, ro = r2(yin, p_in), r2(yout, p_out)
                except Exception as exc:
                    print(f"    {l}: ERROR {type(exc).__name__}: {exc}", flush=True)
                    continue
                el = time.perf_counter() - t0
                # A "blowup" is the failure mode that makes a MEAN R2
                # meaningless: one seed emitting 1e17 dominates every average.
                # Count it separately rather than averaging it in.
                blow = bool(np.max(np.abs(np.asarray(p_out, float)))
                            > 1e3 * (abs(float(np.mean(ytr))) + yscale))
                acc[l]["blow"] += int(blow)
                acc[l]["out"].append(ro); acc[l]["in"].append(ri)
                acc[l]["t"].append(el)
                per_run.append({"task": task, "reach": reach, "seed": seed,
                                "learner": l, "r2_in": ri, "r2_out": ro,
                                "blowup": blow, "fit_s": el})
        for l in learners:
            o, i = np.array(acc[l]["out"]), np.array(acc[l]["in"])
            if not o.size:
                continue
            rows.append({
                "task": task, "reach": reach, "learner": l,
                # MEDIAN is the headline: with blowups present the mean is not
                # a summary of anything.
                "r2_out_median": float(np.nanmedian(o)),
                "r2_out_mean": float(np.nanmean(o)),
                "r2_in_median": float(np.nanmedian(i)),
                "blowups": int(acc[l]["blow"]), "n_seeds": int(o.size),
                "fit_s": float(np.mean(acc[l]["t"])),
            })

    def _get(task, reach, l):
        return next((x for x in rows if x["task"] == task
                     and x["reach"] == reach and x["learner"] == l), None)

    def _fmt(v):
        if v is None:
            return "-"
        return f"{v:.3f}" if v > -10 else "<-10"

    for reach in reaches:
        print(f"\n{'=' * 86}")
        print(f"OUT-OF-RANGE R2 (MEDIAN over {len(seeds)} seeds) at reach {reach}x")
        print(f"{'=' * 86}")
        hdr = f"{'task':<10}" + "".join(f"{l:>15}" for l in learners)
        print(hdr); print("-" * len(hdr))
        for task in tasks:
            line = f"{task:<10}"
            for l in learners:
                r = _get(task, reach, l)
                cell = _fmt(r["r2_out_median"]) if r else "-"
                if r and r["blowups"]:
                    cell += f"!{r['blowups']}"
                line += f"{cell:>15}"
            print(line)
        print("  (! = seeds whose predictions exploded past 1000x the training scale)")

    print("\n--- does UIL beat baselines that CAN extrapolate? ---")
    rivals = [l for l in learners if l not in ("uil_symbolic", "xgboost")]
    total = wins = 0
    for reach in reaches:
        for task in tasks:
            u = _get(task, reach, "uil_symbolic")
            if not u:
                continue
            best = max(((x["learner"], x["r2_out_median"]) for x in rows
                        if x["task"] == task and x["reach"] == reach
                        and x["learner"] in rivals),
                       key=lambda kv: kv[1], default=(None, float("-inf")))
            total += 1
            if u["r2_out_median"] > best[1]:
                wins += 1
                print(f"  WIN  {task}@{reach}x  uil={_fmt(u['r2_out_median'])} "
                      f"> {best[0]}={_fmt(best[1])}")
    print(f"\nUIL beats the best extrapolating baseline on {wins}/{total} "
          f"(task x reach) cells.")

    # Symbolic vs symbolic. This is the comparison that separates "symbolic
    # regression is the wrong tool for out-of-range prediction" from "UIL's
    # implementation of it is weak" -- same model family, same data, same
    # seeds. LITERATURE_SURVEY.md line 375 asks for exactly this.
    if "gplearn" in learners:
        u_w = g_w = tie = u_blow = g_blow = n = 0
        for reach in reaches:
            for task in tasks:
                u, g = _get(task, reach, "uil_symbolic"), _get(task, reach, "gplearn")
                if not (u and g):
                    continue
                n += 1
                u_blow += u["blowups"]; g_blow += g["blowups"]
                du = u["r2_out_median"] - g["r2_out_median"]
                if abs(du) < 1e-6:
                    tie += 1
                elif du > 0:
                    u_w += 1
                else:
                    g_w += 1
        print(f"\n--- UIL vs gplearn (symbolic vs symbolic, {n} cells) ---")
        print(f"  UIL better: {u_w}   gplearn better: {g_w}   tie: {tie}")
        print(f"  seeds that exploded past 1000x training scale: "
              f"UIL {u_blow}, gplearn {g_blow}")
    print("Beating xgboost is NOT evidence: gradient-boosted trees are constant")
    print("outside the training range by construction, so a negative R2 there is")
    print("a theorem about trees, not a finding about UIL.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows, "per_run": per_run, "seeds": seeds,
                       "tasks": tasks, "reaches": reaches}, fh, indent=2)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
