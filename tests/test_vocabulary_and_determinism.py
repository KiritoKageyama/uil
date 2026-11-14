"""Bug #31 (determinism) + bug #25 root cause (one-word vocabulary).

Two independent defects, tested together because the first one is what
made the second one unmeasurable.

V1-V6  the cognitive layer can say more than `max_penalty`
D1-D3  identical inputs produce identical models

Run:
    python figpso_uil/tests/test_vocabulary_and_determinism.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from figpso_uil.adapters.symbolic_formula_adapter import SymbolicFormulaAdapter
from figpso_uil.cognition.condition_match import sanitize_overrides
from figpso_uil.cognition.uil_library import UIL_Library
from figpso_uil.core.optimizer import UniversalOptimizer
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch

_FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        _FAILURES.append(name)


def _override(conditions: list[dict]) -> dict:
    """Call the real generator without touching the principle file."""
    return UIL_Library._generate_override(
        UIL_Library.__new__(UIL_Library), conditions,
    )


def _optimizer(state_dir: str, **cfg_kw):
    class Cfg:
        stages = 2
        n_particles = 4
        pso_iters = 4
        use_resource_budget = False
        disable_routing = True
        cv_folds = 1

    cfg = Cfg()
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    clf = SymbolicFormulaAdapter(
        seed=42, num_classes=2,
        formula_library_path=os.path.join(state_dir, "lib.json"),
    )
    clf.apply_config({"learning_mode": "sparse_linear"})
    return UniversalOptimizer(
        cfg=cfg, seed=42, dataset_name="t", fold=0, start_fold=0,
        classifier=clf, search=PrincipleGuidedSearch(), state_dir=state_dir,
    ), clf


# ── V: vocabulary ────────────────────────────────────────────────────

def test_vocabulary() -> None:
    print("\nV. Mined rules can constrain the learner's architecture")

    # V1: the legacy penalty path is untouched.
    check(
        "V1 penalty override unchanged",
        _override([{"feature": "action.penalty", "op": ">", "threshold": 2.14}])
        == {"max_penalty": 2.14},
    )

    # V2: THE fix. Before 2026-07-23 this returned {}.
    got = _override(
        [{"feature": "config.sparse_l1_c", "op": ">", "threshold": 0.3}]
    )
    check("V2 config rule emits a bound", got == {"config_max.sparse_l1_c": 0.3}, str(got))

    # V3: architecture the router owns must stay un-constrainable (bug #21).
    check(
        "V3 learning_mode not constrainable",
        _override([{"feature": "config.learning_mode", "op": ">", "threshold": 1.0}]) == {},
    )

    # V4: un-actuatable and non-orderable conditions stay context-only.
    check(
        "V4 state/string conditions ignored",
        _override([
            {"feature": "state.entropy", "op": "<=", "threshold": 0.08},
            {"feature": "dataset.name", "op": "==", "threshold": "wine"},
        ]) == {},
    )

    # V5: two same-side splits on one feature NARROW the bad region
    # (`x > 0.5 AND x > 0.2` is just `x > 0.5`), so the resulting bound
    # must be the one the evidence actually supports -- 0.5. Capping at
    # 0.2 would forbid a range the tree never accused of anything.
    got = _override([
        {"feature": "config.sparse_l1_c", "op": ">", "threshold": 0.5},
        {"feature": "config.sparse_l1_c", "op": ">", "threshold": 0.2},
    ])
    check("V5 same-side splits narrow the bad region",
          got == {"config_max.sparse_l1_c": 0.5}, str(got))

    # V5b: a rule's conditions are a CONJUNCTION -- one tree path. A path
    # bounded on both sides condemns a REGION, and inverting each side
    # independently produced a floor ABOVE the ceiling: a constraint no
    # value can satisfy.
    got = _override([
        {"feature": "config.boosting_stages", "op": ">", "threshold": 14.5},
        {"feature": "config.boosting_stages", "op": "<=", "threshold": 19.5},
    ])
    check("V5b two-sided path emits a region, not two bounds",
          got == {"config_avoid.boosting_stages": [14.5, 19.51]}, str(got))

    # V5c: DERIVED features must not pose as the knob they are built
    # from. `"penalty" in feature` matched
    # `derived.action.penalty*config.mutation_rounds_per_fit` and turned
    # a threshold on a PRODUCT into a penalty bound -- 18 of 300 live
    # graph nodes carry the result (min_penalty 1270.45, max_penalty
    # 0.0003) for a knob whose domain is [0, 10].
    check(
        "V5c derived features do not become penalty bounds",
        _override([{
            "feature": "derived.action.penalty*config.mutation_rounds_per_fit",
            "op": "<=", "threshold": 0.0003,
        }]) == {}
        and _override([{"feature": "action.penalty", "op": ">",
                        "threshold": 2.14}]) == {"max_penalty": 2.14},
    )

    # V6: bounds resolve against a REAL adapter's live config.
    state_dir = tempfile.mkdtemp(prefix="uil_vocab_")
    opt, clf = _optimizer(state_dir)
    base = {"penalty": 3.0, "feature_weights": np.ones(4)}

    def cfg_of(ov: dict) -> dict:
        return opt._apply_overrides(base, ov).get("classifier_config", {})

    cases = [
        ("float ceiling binds", {"config_max.sparse_l1_c": 0.03},
         {"sparse_l1_c": 0.03}),
        ("float floor binds", {"config_min.sparse_l1_c": 0.5},
         {"sparse_l1_c": 0.5}),
        ("satisfied ceiling is a no-op", {"config_max.sparse_l1_c": 9.0},
         {"sparse_l1_c": clf.sparse_l1_c}),
        # A boolean gate journals as 0/1 and must come back a real bool --
        # `apply_config` does bool(value), so a stray 0.5 would read True.
        ("bool floor turns gate ON", {"config_min.use_rule_basis": 0.51},
         {"use_rule_basis": True}),
        ("bool ceiling turns gate OFF", {"config_max.use_rule_basis": 0.5},
         {"use_rule_basis": False}),
        # Rounding must land INSIDE the feasible region, not outside it.
        # NB: uses `tournament_k`, an int knob that is LIVE under
        # sparse_linear. `boosting_stages` would be dropped by the
        # inert-knob filter here (P4) -- correctly, but that would test the
        # filter rather than the rounding math this case is about.
        ("int floor rounds up", {"config_min.tournament_k": 12.4},
         {"tournament_k": 13}),
        ("unknown knob is inapplicable", {"config_max.nonexistent": 1.0}, {}),
    ]
    ok = True
    for label, ov, expect in cases:
        got = cfg_of(ov)
        if got != expect:
            ok = False
            print(f"        {label}: expected {expect}, got {got}")
    check("V6 bounds resolve against a live adapter", ok)

    # V7: escaping a condemned region leaves by the NEARER edge, and a
    # knob already outside it is left alone. Uses `tournament_k` for the
    # same reason as the rounding case above -- it is live under
    # sparse_linear, so this tests the escape math, not the P4 filter.
    clf.apply_config({"tournament_k": 16})
    near_low = cfg_of({"config_avoid.tournament_k": [14.5, 25.0]})
    clf.apply_config({"tournament_k": 24})
    near_high = cfg_of({"config_avoid.tournament_k": [14.5, 25.0]})
    clf.apply_config({"tournament_k": 5})
    outside = cfg_of({"config_avoid.tournament_k": [14.5, 25.0]})
    check(
        "V7 region escape picks the nearer edge",
        near_low == {"tournament_k": 14}
        and near_high == {"tournament_k": 26}
        and outside == {},
        f"low={near_low} high={near_high} outside={outside}",
    )
    clf.apply_config({"tournament_k": 8})

    # V8: repair by PROVENANCE, at load time, where the conditions still
    # exist. A magnitude check cannot do this job -- 0.0003 is a
    # perfectly legal penalty cap and only looks wrong once you see the
    # rule never mentions penalty. This is the exact shape of node P-061
    # in the live graph.
    poisoned = {
        "principle_id": "P-061",
        "conditions": [
            {"feature": "derived.action.penalty*config.mutation_rounds_per_fit",
             "op": "<=", "threshold": 0.0003},
            {"feature": "state.feat_ratio", "op": "<=", "threshold": 0.4035},
        ],
        "action_override": {"min_penalty": 18.3932, "max_penalty": 0.0003},
    }
    healthy = {
        "conditions": [{"feature": "action.penalty", "op": ">", "threshold": 2.14}],
        "action_override": {"max_penalty": 2.14},
    }
    removed = sanitize_overrides(poisoned)
    untouched = sanitize_overrides(healthy)
    check(
        "V8 unsupported overrides repaired, supported ones kept",
        removed and poisoned["action_override"] == {}
        and not untouched and healthy["action_override"] == {"max_penalty": 2.14},
        f"poisoned->{poisoned['action_override']} healthy->{healthy['action_override']}",
    )

    # V8b: the range guard still drops bounds outside penalty's domain,
    # which is the half of the problem provenance cannot see (a rule can
    # legitimately mention penalty and still carry a nonsense threshold).
    base3 = {"penalty": 2.0, "feature_weights": np.ones(4)}
    out_of_range = opt._apply_overrides(base3, {"min_penalty": 1270.45})["penalty"]
    in_range = opt._apply_overrides(base3, {"max_penalty": 1.25})["penalty"]
    check("V8b out-of-domain penalty bounds ignored",
          out_of_range == 2.0 and in_range == 1.25,
          f"out_of_range->{out_of_range} in_range->{in_range}")

    # The boolean must be a genuine bool, not a truthy float.
    val = cfg_of({"config_min.use_rule_basis": 0.51})["use_rule_basis"]
    check("V6b boolean gate stays a bool", isinstance(val, bool), f"type={type(val).__name__}")


# ── D: determinism ───────────────────────────────────────────────────

def _coef_hash(seed: int, y: np.ndarray) -> str:
    """Fit the real sparse_linear path and hash its coefficients."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(160, 8))
    state_dir = tempfile.mkdtemp(prefix="uil_det_")
    clf = SymbolicFormulaAdapter(
        seed=seed, num_classes=len(np.unique(y)),
        formula_library_path=os.path.join(state_dir, "lib.json"),
    )
    clf.apply_config({"learning_mode": "sparse_linear"})
    model = clf.fit_weighted(X, y, np.ones(8))
    preds = clf.predict(model, X)
    return hashlib.sha256(
        np.asarray(preds).astype(str).tobytes()
    ).hexdigest()[:16]


def test_determinism() -> None:
    print("\nD. Identical inputs produce identical models")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(160, 8))
    y_bin = (X[:, 0] + 0.5 * X[:, 3] > 0).astype(int)
    # 3 classes: the arity that actually broke (wine, n_selected 9 vs 8).
    y_multi = np.digitize(X[:, 0] + 0.5 * X[:, 3], [-0.6, 0.6])

    # D1/D2: the failure mode was a seed drawn from the GLOBAL numpy RNG,
    # so perturbing that RNG between fits is what reproduces it.
    for label, y, tag in (("D1 binary", y_bin, "binary"),
                          ("D2 multiclass OvR", y_multi, "multiclass")):
        hashes = set()
        for g in range(3):
            np.random.seed(g * 977 + 1)
            hashes.add(_coef_hash(42, y))
        check(f"{label} stable under global-RNG perturbation",
              len(hashes) == 1, f"{len(hashes)} distinct outputs ({tag})")

    # D3: guard against D1/D2 passing trivially. A model that always
    # emitted one class would also be perfectly "stable", so require the
    # fit to be non-degenerate: it must use several basis terms and
    # predict more than one class.
    #
    # Note the seed deliberately is NOT varied here. On sparse_linear the
    # L1 solution is unique given the data, so an honest fit IS seed-
    # independent -- the seed only ever set liblinear's coordinate order,
    # which is precisely the nondeterminism bug #31 removed. Asserting
    # "different seeds differ" would be asserting the bug.
    state_dir = tempfile.mkdtemp(prefix="uil_det3_")
    clf = SymbolicFormulaAdapter(
        seed=42, num_classes=3,
        formula_library_path=os.path.join(state_dir, "lib.json"),
    )
    clf.apply_config({"learning_mode": "sparse_linear"})
    model = clf.fit_weighted(X, y_multi, np.ones(8))
    preds = clf.predict(model, X)
    n_terms = sum(
        len(h.get("coef", [])) for h in model.get("classifiers", {}).values()
    )
    check("D3 fit is non-degenerate (not trivially stable)",
          n_terms > 1 and len(np.unique(preds)) > 1,
          f"{n_terms} basis terms, {len(np.unique(preds))} predicted classes")


# ── G: guard telemetry (audit Part E step 2) ─────────────────────────

def test_guard_telemetry() -> None:
    print("\nG. Safety-rail bind telemetry (observe-only)")
    from figpso_uil.cognition.guard_telemetry import GuardTelemetry

    # G1: a guard that never binds records seen>0, bound=0, and the true
    # range of originals seen -- the evidence "this rail is never hit".
    gt = GuardTelemetry(filepath=None)
    for v in (0.6, 1.0, 3.0, 2.0):
        gt.observe("penalty", v, min(max(v, 0.0), 10.0))  # all in-domain
    snap = gt.snapshot()["penalty"]
    check("G1 non-binding rail: seen counted, bind_rate 0, range exact",
          snap["n_seen"] == 4 and snap["n_bound"] == 0
          and snap["bind_rate"] == 0.0
          and abs(snap["min"] - 0.6) < 1e-9 and abs(snap["max"] - 3.0) < 1e-9,
          str(snap))

    # G2: a binding event is counted iff original != clamped.
    gt.observe("penalty", 1270.45, 10.0)   # out of domain -> binds
    gt.observe("penalty", 5.0, 5.0)         # in domain    -> no bind
    snap = gt.snapshot()["penalty"]
    check("G2 bind counted only when the value actually changed",
          snap["n_seen"] == 6 and snap["n_bound"] == 1,
          f"seen={snap['n_seen']} bound={snap['n_bound']}")

    # G3: mean/std track the ORIGINAL magnitudes via Welford, matching numpy.
    gt2 = GuardTelemetry(filepath=None)
    vals = [0.6, 1.0, 3.0, 2.0, 1270.45, 5.0]
    for v in vals:
        gt2.observe("p", v, v)
    s = gt2.snapshot()["p"]
    check("G3 online mean/std match numpy",
          abs(s["mean"] - float(np.mean(vals))) < 1e-6
          and abs(s["std"] - float(np.std(vals))) < 1e-6,
          f"mean {s['mean']} vs {np.mean(vals):.4f}, std {s['std']} vs {np.std(vals):.4f}")

    # G4: persistence round-trips and ACCUMULATES across instances (the
    # cross-run evidence property the whole audit-Part-E design rests on).
    import tempfile as _tf
    path = os.path.join(_tf.mkdtemp(prefix="uil_guard_"), "g.json")
    a = GuardTelemetry(filepath=path)
    a.observe("x", 1.0, 1.0); a.flush()
    b = GuardTelemetry(filepath=path)      # fresh instance, same file
    b.observe("x", 2.0, 2.0); b.flush()
    reloaded = GuardTelemetry(filepath=path).snapshot()["x"]
    check("G4 telemetry persists and accumulates across runs",
          reloaded["n_seen"] == 2, str(reloaded))

    # G5: array guard -- bound iff ANY element changed; never raises.
    gt3 = GuardTelemetry(filepath=None)
    gt3.observe_array("w", [0.1, 0.5, 1.0], [0.25, 0.5, 1.0])  # floor bound elem 0
    gt3.observe_array("w", [0.3, 0.5, 1.0], [0.3, 0.5, 1.0])   # nothing changed
    s = gt3.snapshot()["w"]
    check("G5 array guard: per-array bind flag",
          s["n_seen"] == 2 and s["n_bound"] == 1, str(s))


# ── A: default-path architecture exploration (research step 2) ───────

def test_architecture_exploration() -> None:
    print("\nA. Default-path architecture variation (PBT-style probe)")
    import json
    from figpso_uil.core.optimizer import UniversalOptimizer
    from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch

    def _run(explore: bool, situation_adaptive: bool = False,
             use_factory: bool = False):
        class Cfg:
            stages = 10
            n_particles = 5
            pso_iters = 5
            use_resource_budget = False
            disable_routing = True
            cv_folds = 1
        Cfg.explore_architecture = explore
        # MUST be pinned. `situation_adaptive` ships default ON (Category D,
        # 2026-07-24) and AUTO-ENABLES `explore_architecture` from the
        # fingerprint -- on breast_cancer it does. Leaving it at the default
        # therefore set explore_architecture=True in BOTH arms, so A1-A4 were
        # silently comparing one arm with itself: A1/A3 failed and A2/A4
        # "passed" trivially (A4 in particular is guaranteed when the two runs
        # are literally identical). Measured: default -> OFF and ON both give
        # 8 journal entries / sparse_l1_c 2 distinct; pinned False -> OFF
        # gives 4 / 1 and ON gives 8 / 2, which is the intended contrast.
        Cfg.situation_adaptive = situation_adaptive
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        sd = tempfile.mkdtemp(prefix="uil_probe_")
        clf = SymbolicFormulaAdapter(
            seed=42, num_classes=2,
            formula_library_path=os.path.join(sd, "lib.json"),
            learning_mode="sparse_linear",
        )
        kw = {}
        if use_factory:
            from figpso_uil.core.classifier_registry import CLASSIFIERS
            kw = dict(classifier_factory=CLASSIFIERS["SymbolicFormula"],
                      classifier_build_args=(42, 2, 40, sd))
        opt = UniversalOptimizer(
            cfg=Cfg(), seed=42, dataset_name="p", fold=0, start_fold=0,
            classifier=clf, search=PrincipleGuidedSearch(), state_dir=sd, **kw,
        )
        r = opt.fit(X, y)
        j = json.load(open(os.path.join(sd, "uil_journal_v19.json")))
        cfgs = [e["action"].get("classifier_config", {}) for e in j]
        distinct = {
            a: len({str(c.get(a)) for c in cfgs if a in c})
            for a in ("sparse_l1_c", "use_rule_basis", "use_stump_basis")
        }
        return r["accuracy"], len(j), distinct, getattr(
            opt.cfg, "explore_architecture", None)

    acc_off, n_off, d_off, _ = _run(False)
    acc_on, n_on, d_on, _ = _run(True)

    # A1: OFF is the F4 root cause -- every architecture axis is constant.
    check("A1 baseline: architecture knobs are constant (the F4 defect)",
          all(v == 1 for v in d_off.values()), str(d_off))

    # A2: ON creates real variation the miner can split on.
    check("A2 probe creates architecture variation in the journal",
          d_on["sparse_l1_c"] > 1 and any(v > 1 for v in d_on.values()),
          str(d_on))

    # A3: the probe adds journal entries (one per stage) ...
    check("A3 probe journals its evidence (extra entries)",
          n_on > n_off, f"{n_off} -> {n_on} entries")

    # A4: ... but NEVER competes for the stage winner, so accuracy is
    # byte-identical. This is the winner's-curse-safety guarantee: if the
    # probe could win, this would drift (and re-create the E7 regression).
    # A4: the winner's-curse-safety guarantee -- the probe never competes for
    # the stage winner, so accuracy must be byte-identical. This holds ONLY on
    # the isolated-worker path (a classifier_factory is supplied), which is
    # how the original 2026-07-24 measurement was taken.
    f_off, _, _, _ = _run(False, use_factory=True)
    f_on, _, _, _ = _run(True, use_factory=True)
    check("A4 probe does not change accuracy with an isolated worker",
          f_off == f_on, f"off {f_off} vs on {f_on}")

    # A6: KNOWN DEFECT -- the same guarantee FAILS without a factory, where
    # the probe falls back to save/restore of config + _shadowed_attrs. That
    # fallback does not restore everything the probe's fit mutates (formula
    # library writes, _formulas_used_this_fit / _last_used_formula_id, the
    # adapter RNG), so the probe leaks into the run it is supposed to be
    # merely observing. Measured on breast_cancer, 10 stages:
    #     with factory : 0.9649122807017544 both ways (identical)
    #     no factory   : off 0.9649122807017544 -> on 0.9736842105263158
    # This is the same hazard already documented for _evaluate_one_action in
    # the failed verify-before-apply gate: "a verification probe that mutates
    # the run it measures is worthless". Asserted as broken so a fix fails
    # loudly and the guarantee can be promoted to unconditional.
    check("A6 probe LEAKS without an isolated worker [KNOWN BUG]",
          acc_off != acc_on,
          f"off {acc_off} vs on {acc_on} — if these are now equal the "
          f"save/restore path was fixed: record it and merge A6 into A4")

    # A5: pin the interaction that broke A1-A4 above, so it can never again
    # disable this test silently. `explore_architecture` is DOCUMENTED default
    # OFF, but `situation_adaptive` (default ON) overrides it from the
    # fingerprint -- so the documented default is not the effective one.
    # Same bug CLASS as #21: an architectural knob set by one layer being
    # silently replaced by another.
    _, _, _, eff = _run(False, situation_adaptive=True)
    check("A5 situation_adaptive auto-enables explore_architecture "
          "(documented default OFF is not the effective default)",
          eff is True,
          f"cfg.explore_architecture={eff} after fit with explore=False")


# ── P: promotion chain (research step 3) ─────────────────────────────

def test_promotion_chain() -> None:
    print("\nP. Step-2+3 chain: architecture variation -> mined arch rule")
    from figpso_uil.cognition.uil_miner import UIL_Miner
    from figpso_uil.cognition.uil_library import UIL_Library

    # Planted journal: regret depends on an ARCHITECTURE knob
    # (config.sparse_l1_c), which only exists because step-2 variation puts
    # several distinct values in the journal. This is the fast, deterministic
    # version of the 40-stage integration run — it isolates the step-3 chain
    # (miner splits on config.* -> library promotes at a LOWERED bar ->
    # emitter turns it into a config bound) from the slow optimizer loop.
    rng = np.random.default_rng(0)
    entries = []
    for i in range(120):
        l1 = float(rng.choice([0.05, 0.1, 0.5, 1.0]))   # step-2 variation
        # planted: high L1 (>0.3) causes high regret
        regret = 0.05 + (0.15 if l1 > 0.3 else 0.0) + rng.normal(0, 0.005)
        entries.append({
            "dataset": "d", "fold": 0, "stage": i, "classifier": "SymbolicFormula",
            "state": {"n_samples": 500, "n_features": 30, "num_classes": 2,
                      "entropy": 0.9, "feat_ratio": 0.06},
            "action": {"penalty": 2.0, "classifier_config": {"sparse_l1_c": l1}},
            "outcome": {}, "regret": float(regret),
        })

    rules = UIL_Miner().mine_within_dataset(entries, "d", 0.3)
    arch_rules = [
        r for r in rules
        if any(str(c.get("feature", "")).startswith("config.sparse_l1_c")
               for c in r.get("conditions", []))
    ]
    check("P1 miner splits on the architecture knob (needs step-2 variation)",
          len(arch_rules) > 0, f"{len(rules)} rules, {len(arch_rules)} on config.sparse_l1_c")

    # Promote at a LOWERED bar (step 3). Default MIN_EVIDENCE=10 would reject
    # these small leaves; min_evidence=3 (what adaptive_min_evidence yields
    # for a small journal) admits them.
    import tempfile as _tf
    lib = UIL_Library(library_path=os.path.join(
        _tf.mkdtemp(prefix="uil_promo_"), "p.json"))
    lib.update_principles(rules, min_evidence=3)
    ps = lib.get_principles()
    arch_overrides = [
        p for p in ps
        if any(k.startswith("config_") for k in (p.get("action_override") or {}))
    ]
    check("P2 lowered bar promotes it into an architecture override",
          len(arch_overrides) > 0,
          f"{len(ps)} principles, {len(arch_overrides)} carry a config_* override")

    # P3: step 3's ACTUAL contract, tested directly and deterministically
    # (not via leaf-size luck). A rule with evidence_count between the
    # lowered bar and the fixed one is admitted by the lowered bar and
    # rejected by the default — which is the whole point of step 3 in the
    # small-journal regime real runs actually live in.
    small_rule = {
        "conditions": [
            {"feature": "config.sparse_l1_c", "op": ">", "threshold": 0.3},
        ],
        "confidence": 0.9, "evidence_count": 5,   # between 3 and 10
        "predicted_regret": 0.2, "mining_level": "within_dataset", "dataset": "d",
    }
    lib_low = UIL_Library(library_path=os.path.join(
        _tf.mkdtemp(prefix="uil_lo_"), "p.json"))
    lib_low.update_principles([small_rule], min_evidence=3)
    lib_def = UIL_Library(library_path=os.path.join(
        _tf.mkdtemp(prefix="uil_def_"), "p.json"))
    lib_def.update_principles([small_rule])  # default MIN_EVIDENCE = 10
    admitted_low = any(
        k.startswith("config_") for p in lib_low.get_principles()
        for k in (p.get("action_override") or {})
    )
    admitted_def = any(
        k.startswith("config_") for p in lib_def.get_principles()
        for k in (p.get("action_override") or {})
    )
    check("P3 lowered bar admits an evidence=5 rule the default (10) rejects",
          admitted_low and not admitted_def,
          f"lowered admitted={admitted_low}, default admitted={admitted_def}")

    # P4: the inert-knob filter. Step 3's lowered bar admits weaker rules,
    # and the FIRST one it ever minted was `config_min.bagging_estimators`
    # under sparse_linear -- provably a no-op. The adapter reports which
    # knobs are live for the active mode; inert ones are dropped instead of
    # being applied and journalled as if they mattered.
    sd_f = tempfile.mkdtemp(prefix="uil_inert_")
    opt_f, clf_f = _optimizer(sd_f)          # pinned to sparse_linear
    base_f = {"penalty": 3.0, "feature_weights": np.ones(4)}

    def _cfg(ov):
        return opt_f._apply_overrides(base_f, ov).get("classifier_config", {})

    inert_bound = _cfg({"config_min.bagging_estimators": 6.51})   # P-011's real override
    inert_region = _cfg({"config_avoid.boosting_stages": [4.0, 9.0]})
    live_bound = _cfg({"config_max.sparse_l1_c": 0.03})
    live_gate = _cfg({"config_min.use_rule_basis": 0.51})
    check("P4 inert-under-mode overrides dropped, live ones still applied",
          inert_bound == {} and inert_region == {}
          and live_bound == {"sparse_l1_c": 0.03}
          and live_gate == {"use_rule_basis": True},
          f"inert={inert_bound}/{inert_region} live={live_bound}/{live_gate}")

    # P5: the filter is MODE-SCOPED, not a blanket blocklist — the same
    # knob must stay live in the mode that actually uses it, or a real
    # override would be silently dropped (worse than allowing a no-op).
    from figpso_uil.adapters.symbolic_formula_adapter import (
        SymbolicFormulaAdapter as _SFA,
    )
    clf_boost = _SFA(seed=42, num_classes=2,
                     formula_library_path=os.path.join(sd_f, "b.json"))
    clf_boost.apply_config({"learning_mode": "boosting"})
    check("P5 filter is mode-scoped (boosting_stages live under boosting)",
          "boosting_stages" in clf_boost.get_active_config_knobs()
          and "boosting_stages" not in clf_f.get_active_config_knobs(),
          "live under boosting, inert under sparse_linear")


# ── N: maturity count normalization (audit Category D) ───────────────

def test_maturity_normalization() -> None:
    print("\nN. Maturity count normalization (saturation fix, opt-in)")
    from figpso_uil.search.hybrid_foundation import KnowledgeFoundationBuilder

    # Real production counts, measured from results/ on 2026-07-24.
    prod = [("principles", 201, 50), ("network", 300, 30),
            ("episodes", 135, 20), ("formulas", 32, 10)]

    legacy = KnowledgeFoundationBuilder()
    soft = KnowledgeFoundationBuilder()
    soft.soft_normalization = True

    # N1: default is byte-identical to the historical formula.
    check("N1 default normalization unchanged (min(1, n/cap))",
          all(legacy._count_score(c, cap) == min(1.0, c / cap)
              for _, c, cap in prod))

    # N2: the MEASURED defect -- every production term saturates to
    # exactly 1.0, so the blend's four quantity terms carry zero signal.
    legacy_vals = [legacy._count_score(c, cap) for _, c, cap in prod]
    check("N2 legacy saturates on real production counts (the defect)",
          all(v == 1.0 for v in legacy_vals), str(legacy_vals))

    # N3: the fix restores variation at the same counts -- and must never
    # reach exactly 1.0, which is the property that makes it survive a
    # lifelong-growing store.
    soft_vals = [soft._count_score(c, cap) for _, c, cap in prod]
    check("N3 soft normalization restores signal and never saturates",
          len(set(round(v, 6) for v in soft_vals)) > 1
          and all(0.0 < v < 1.0 for v in soft_vals),
          str([round(v, 3) for v in soft_vals]))

    # N4: monotonic and bounded -- more knowledge never scores lower.
    seq = [soft._count_score(n, 50) for n in (0, 1, 10, 50, 200, 5000)]
    check("N4 soft normalization is monotonic and bounded in [0,1)",
          all(a <= b for a, b in zip(seq, seq[1:])) and seq[0] == 0.0 and seq[-1] < 1.0,
          str([round(v, 3) for v in seq]))


def test_utility_gate() -> None:
    """Research step 4 — DreamCoder-style utility-gated formula admission."""
    import tempfile
    from figpso_uil.cognition.formula_engine import FormulaNode

    print("\n--- U: utility-gated formula admission (research step 4) ---")
    rng = np.random.default_rng(0)
    n, d = 200, 4
    X = rng.normal(size=(n, d))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)

    with tempfile.TemporaryDirectory() as td:
        lib_path = os.path.join(td, "lib.json")
        clf = SymbolicFormulaAdapter(
            seed=0, num_classes=2, formula_library_path=lib_path,
        )
        clf.apply_config({
            "learning_mode": "sparse_linear",
            "utility_gated_formulas": True,
            "use_raw_basis": True,
            "use_symbolic_basis": True,
            "use_stump_basis": False,
            "use_rule_basis": False,
            "use_neural_basis": False,
        })
        useful = FormulaNode(feature=0)

        # U1: default OFF leaves incubating formulas OUT of the basis.
        clf.utility_gated_formulas = False
        fid = clf.formula_library.add_formula(
            name="incub_probe", tree=useful, origin="test",
            mutation_type="test", status="incubating",
        )
        spec_off = clf._build_basis_spec(X, y, d)
        fids_off = {e.get("formula_id") for e in spec_off if e.get("type") == "formula"}
        check("U1 incubating excluded from basis when gate OFF",
              fid not in fids_off, str(fids_off))

        # U2: gate ON includes incubating as newborn.
        clf.utility_gated_formulas = True
        spec_on = clf._build_basis_spec(X, y, d)
        newborns = [e for e in spec_on
                    if e.get("type") == "formula" and e.get("newborn")]
        check("U2 incubating enters basis as newborn when gate ON",
              any(e.get("formula_id") == fid for e in newborns),
              str([(e.get("formula_id"), e.get("newborn")) for e in newborns]))

        # U3: newborn L1 boost is applied in the weight vector.
        w = clf._basis_weight_vector(spec_on, None)
        nb_idx = [i for i, e in enumerate(spec_on)
                  if e.get("formula_id") == fid]
        check("U3 newborn column gets NEWBORN_L1_BOOST",
              bool(nb_idx) and abs(w[nb_idx[0]] - clf.NEWBORN_L1_BOOST) < 1e-9,
              f"idx={nb_idx} w={w[nb_idx] if nb_idx else None}")

        # U4: utility_test_candidate is deterministic and returns a bool/float.
        a1, d1 = clf.utility_test_candidate(useful, X, y)
        a2, d2 = clf.utility_test_candidate(useful, X, y)
        check("U4 utility_test_candidate is deterministic",
              a1 == a2 and abs(d1 - d2) < 1e-12,
              f"{(a1, d1)} vs {(a2, d2)}")

        # U5: add_formula(status='active') lands in get_active_formulas.
        fid2 = clf.formula_library.add_formula(
            name="admitted", tree=useful, origin="test",
            mutation_type="utility", status="active",
        )
        actives = {f["formula_id"] for f in clf.formula_library.get_active_formulas()}
        check("U5 utility-admitted formula is active immediately",
              fid2 in actives, str(actives))


def main() -> int:
    print("=" * 70)
    print("Vocabulary widening (bug #25 root cause) + determinism (bug #31)")
    print("=" * 70)
    test_vocabulary()
    test_determinism()
    test_guard_telemetry()
    test_architecture_exploration()
    test_promotion_chain()
    test_maturity_normalization()
    test_utility_gate()
    print("\n" + "=" * 70)
    if _FAILURES:
        print(f"FAILED ({len(_FAILURES)}): {', '.join(_FAILURES)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
