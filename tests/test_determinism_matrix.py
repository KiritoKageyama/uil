"""Determinism matrix — the instrument every A/B in this project depends on.

Bug #31 fixed one unseeded `liblinear`, and D1-D3 in
test_vocabulary_and_determinism.py pin that fix for `sparse_linear`. But the
scope note left behind (bug #30a) said determinism was "mode-scoped", and
nothing ever re-checked the other modes after the fix, nor the two axes that
are known to have carried nondeterminism independently:

  M1  every learning mode, binary and multiclass, under a PERTURBED GLOBAL
      numpy RNG -- the only way to catch an estimator that draws its seed
      from the process-global source (that IS bug #31's mechanism).
  M2  across separate PROCESSES, where the global RNG is seeded from OS
      entropy. An in-process test cannot see this class at all.
  M3  serial vs PARALLEL hybrid branches. Bug #31's second axis: branch
      results depended on the machine's thread count. There was no test.
  M4  a non-degeneracy guard, so none of the above can pass by predicting
      one class forever.

Why hash predictions + selected features: those are the two quantities the
historical failures actually surfaced as ("predictions AND n_selected 8 vs 9
differ between two identical runs"). A model whose internals wobble but whose
behaviour does not is not what broke the A/Bs.

Run:  python figpso_uil/tests/test_determinism_matrix.py
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(os.path.dirname(_HERE))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from figpso_uil.adapters.symbolic_formula_adapter import SymbolicFormulaAdapter

MODES = ("sparse_linear", "single", "boosting", "bagging", "multi", "auto")
_FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        _FAILURES.append(name)


def make_data(n_classes: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 6))
    if n_classes == 2:
        y = (1.2 * X[:, 0] - 0.8 * X[:, 2] + rng.normal(scale=0.3, size=200) > 0)
        return X, y.astype(int)
    z = np.column_stack([X[:, 0], X[:, 1] - X[:, 3], -X[:, 2]])
    return X, np.argmax(z + rng.normal(scale=0.25, size=z.shape), axis=1)


def fit_digest(mode: str, n_classes: int, *, workers: int = 1,
               branch_parallel: bool = False, seed: int = 42) -> str:
    """Fit once and digest the BEHAVIOUR: predictions + selected features."""
    X, y = make_data(n_classes)
    state = tempfile.mkdtemp(prefix="uil_detmx_")
    clf = SymbolicFormulaAdapter(
        seed=seed, num_classes=n_classes, learning_mode=mode,
        boosting_rounds=3, bagging_estimators=3,
        formula_library_path=os.path.join(state, "lib.json"),
    )
    clf.parallel_eval_workers = workers
    clf._branch_parallel = branch_parallel
    model = clf.fit_weighted(X, y, np.ones(X.shape[1]))
    preds = np.asarray(clf.predict(model, X)).ravel()
    sel = np.asarray(clf.get_selected_feature_indices(model)).ravel()
    payload = json.dumps(
        {"preds": preds.tolist(), "selected": sorted(int(v) for v in sel)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def n_predicted_classes(mode: str, n_classes: int) -> int:
    X, y = make_data(n_classes)
    state = tempfile.mkdtemp(prefix="uil_detnd_")
    clf = SymbolicFormulaAdapter(
        seed=42, num_classes=n_classes, learning_mode=mode,
        boosting_rounds=3, bagging_estimators=3,
        formula_library_path=os.path.join(state, "lib.json"),
    )
    model = clf.fit_weighted(X, y, np.ones(X.shape[1]))
    return len(np.unique(np.asarray(clf.predict(model, X)).ravel()))


# --------------------------------------------------------------------------
# M1 — every mode, under a perturbed global RNG
# --------------------------------------------------------------------------

def test_m1_modes_under_global_rng_perturbation() -> None:
    print("\nM1. Every learning mode is stable under global-RNG perturbation")
    for n_classes, tag in ((2, "binary"), (3, "multiclass")):
        for mode in MODES:
            digests = set()
            for g in range(3):
                # This is the whole point: an estimator constructed with
                # random_state=None inherits THIS, and would differ per call.
                np.random.seed(g * 7919 + 13)
                digests.add(fit_digest(mode, n_classes))
            check(f"M1 {mode}/{tag}", len(digests) == 1,
                  f"{len(digests)} distinct outputs over 3 global seeds")


# --------------------------------------------------------------------------
# M2 — across processes
# --------------------------------------------------------------------------

def test_m2_cross_process() -> None:
    print("\nM2. Identical across separate processes (OS-entropy global seed)")
    env = dict(os.environ)
    env["PYTHONPATH"] = _PARENT + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONHASHSEED"] = "random"  # must NOT matter; if it does, that is a bug
    for n_classes, tag in ((2, "binary"), (3, "multiclass")):
        for mode in ("sparse_linear", "multi"):
            outs = []
            for _ in range(2):
                proc = subprocess.run(
                    [sys.executable, os.path.abspath(__file__),
                     "--worker", mode, str(n_classes)],
                    capture_output=True, text=True, env=env, timeout=900,
                )
                if proc.returncode != 0:
                    check(f"M2 {mode}/{tag}", False,
                          f"worker failed: {proc.stderr.strip()[-200:]}")
                    outs = []
                    break
                outs.append(proc.stdout.strip().splitlines()[-1])
            if outs:
                check(f"M2 {mode}/{tag}", len(set(outs)) == 1,
                      f"{outs[0]} vs {outs[-1]}")


# --------------------------------------------------------------------------
# M3 — serial vs parallel branches (bug #31's second axis, untested until now)
# --------------------------------------------------------------------------

def test_m3_serial_vs_parallel_branches() -> None:
    print("\nM3. Hybrid branches: serial == parallel (thread count must not matter)")
    for n_classes, tag in ((2, "binary"), (3, "multiclass")):
        serial = fit_digest("multi", n_classes, workers=1, branch_parallel=False)
        par2 = fit_digest("multi", n_classes, workers=2, branch_parallel=True)
        par4 = fit_digest("multi", n_classes, workers=4, branch_parallel=True)
        check(f"M3 multi/{tag} serial == 2 workers == 4 workers",
              serial == par2 == par4,
              f"{serial} / {par2} / {par4}")


# --------------------------------------------------------------------------
# M4 — the guard against passing trivially
# --------------------------------------------------------------------------

def test_m4_not_trivially_stable() -> None:
    """A model that always answers the same class is perfectly 'stable', so
    M1-M3 would pass on a corpse. Require a live fit wherever the project
    actually claims one."""
    print("\nM4. Fits are non-degenerate (a constant model is also 'stable')")
    for mode, n_classes, tag in (
        ("sparse_linear", 2, "binary"),
        ("sparse_linear", 3, "multiclass"),
        ("multi", 2, "binary"),
    ):
        k = n_predicted_classes(mode, n_classes)
        check(f"M4 {mode}/{tag} predicts >1 class", k > 1,
              f"{k} distinct predicted classes")


def test_m5_known_defect_ovr_collapse_off_sparse_linear() -> None:
    """KNOWN OPEN DEFECT, pinned so it stays visible.

    Bug #24 fixed the multiclass OvR collapse in two parts: Platt-calibrate
    each head, and give the OvR loop a `sparse_linear` branch. Only the
    second part actually rescued accuracy (digits 0.511 -> 0.947). Every
    OTHER head kind was left at the post-calibration level and never
    re-checked, because production pins `learning_mode="sparse_linear"` for
    multiclass.

    Measured here (3 classes, majority rate 0.515), calibration present on
    every head: `single`, `bagging` and `multi` predict exactly ONE class and
    score exactly the majority rate; `boosting` reaches 2 of 3. That is the
    same signature bugs #13 and #20 document -- accuracy == base rate means
    the signal was thrown away.

    This test asserts the DEFECT. If someone fixes those paths it fails, and
    the fix should be recorded and this test promoted to a real assertion.
    """
    print("\nM5. Known defect: OvR heads other than sparse_linear collapse")
    healthy = n_predicted_classes("sparse_linear", 3)
    check("M5 sparse_linear heads are healthy (the supported path)",
          healthy == 3, f"{healthy}/3 classes reachable")
    for mode, expected in (("single", 1), ("bagging", 1), ("multi", 1),
                           ("boosting", 2)):
        k = n_predicted_classes(mode, 3)
        check(f"M5 {mode} still collapses to {expected} class(es) [KNOWN BUG]",
              k == expected,
              f"{k} classes reachable — if this went UP, the bug is fixed: "
              f"record it and promote this assertion")


def _worker() -> None:
    mode, n_classes = sys.argv[2], int(sys.argv[3])
    print(fit_digest(mode, n_classes))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _worker()
        sys.exit(0)
    test_m1_modes_under_global_rng_perturbation()
    test_m3_serial_vs_parallel_branches()
    test_m4_not_trivially_stable()
    test_m5_known_defect_ovr_collapse_off_sparse_linear()
    test_m2_cross_process()
    print(f"\n{'FAILURES: ' + ', '.join(_FAILURES) if _FAILURES else 'All checks passed'}")
    sys.exit(1 if _FAILURES else 0)
