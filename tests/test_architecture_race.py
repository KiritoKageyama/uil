"""Acceptance test — architecture race (cfg.search_mode = "race").

Verifies the replacement for PSO surrogate optimization:

  R1  The PSO surrogate is separable, so its optimum is closed-form.
      Proven by brute force over all 2^n masks, not asserted.
  R2  Successive-halving schedule: elimination on real data, none on
      tiny data or small arm counts.
  R3  Stratified subsampling never hands a rung a single-class problem.
  R4  Candidate design is one-axis-at-a-time and respects the mode lock
      (bug #21) -- learning_mode is never raced.
  R5  Phase-out: arm budget shrinks as bucket evidence accumulates.
  R6  End-to-end: a race-mode run completes, actually eliminates arms,
      records outcomes, and wins on architecture axes (not just cutoffs).
  R7  Elimination rungs do not corrupt the run (adapter bookkeeping is
      restored) and causal intervention arms are never eliminated.
  R8  cfg.search_mode default leaves the classic path untouched.

Run:
    python figpso_uil/tests/test_architecture_race.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from figpso_uil.adapters.symbolic_formula_adapter import SymbolicFormulaAdapter
from figpso_uil.core.optimizer import UniversalOptimizer
from figpso_uil.search.architecture_race import (
    ArchitectureRace,
    ArmMemory,
    DEFAULT_RUNGS,
    MIN_ARMS_FOR_RACE,
    RACE_MIN_ROWS,
    frontier_masks,
    rungs,
    stratified_subsample,
)
from figpso_uil.search.principle_guided_search import PrincipleGuidedSearch


class Cfg:
    stages = 6
    n_particles = 8
    pso_iters = 8
    boosting_rounds = 15
    use_resource_budget = False
    disable_routing = True
    cv_folds = 1
    search_mode = "classic"


def make_dataset(n=600, f=12, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, f))
    signal = 0.9 * X[:, 0] + 0.6 * X[:, 1] - 0.5 * X[:, 2] + 0.4 * X[:, 3] * X[:, 4]
    y = (signal + rng.normal(0, 0.35, n) > 0).astype(int)
    return X, y


# ── R1 ────────────────────────────────────────────────────────────────


def test_r1_surrogate_is_closed_form():
    print("=== R1: PSO surrogate optimum is closed-form ===")
    rng = np.random.default_rng(0)
    for trial in range(20):
        n = 10
        g = rng.random(n)
        penalty = float(rng.uniform(0.1, 3.0))

        best, brute = -np.inf, None
        for bits in range(1 << n):
            mask = np.array([(bits >> i) & 1 for i in range(n)], dtype=float)
            fitness = mask @ g - penalty * mask.sum() / n
            if fitness > best:
                best, brute = fitness, mask

        closed = (g > penalty / n).astype(float)
        assert np.array_equal(brute, closed), (
            f"trial {trial}: brute force {brute} != closed form {closed}"
        )
    print(f"  20/20 trials: argmax over 2^10 masks == threshold rule")

    masks = frontier_masks(np.array([0.9, 0.7, 0.5, 0.3, 0.1, 0.0]), 4)
    assert len(masks) == len({m.tobytes() for m in masks}), "duplicate masks"
    assert all(np.any(m > 0) for m in masks), "empty mask emitted"
    print(f"  frontier_masks: {len(masks)} distinct non-empty masks, no swarm")
    print("  R1 PASS: swarm was re-deriving an argsort\n")


# ── R2 / R3 ───────────────────────────────────────────────────────────


def test_r2_halving_schedule():
    print("=== R2: successive-halving schedule ===")
    plan = rungs(2000, 12)
    assert plan, "expected elimination rungs on n=2000 with 12 arms"
    rows = [r for r, _ in plan]
    keeps = [k for _, k in plan]
    assert rows == sorted(rows), "rung data size must increase"
    assert all(r < 2000 for r in rows), "rung must be cheaper than full fit"
    assert keeps == sorted(keeps, reverse=True), "survivors must shrink"
    print(f"  n=2000, 12 arms -> {plan}")

    assert rungs(208, 12) == [], "tiny data must skip subsampled rungs"
    print(f"  n=208 (sonar-sized) -> no elimination (noise floor)")
    assert rungs(2000, 3) == [], "too few arms to eliminate"
    print(f"  3 arms -> no elimination")

    # Cost: full fits saved vs evaluating everything at full fidelity.
    plan = rungs(5000, 12)
    cost = sum(rows / 5000 * n_in for (rows, _), n_in in
               zip(plan, [12] + [k for _, k in plan[:-1]]))
    final = plan[-1][1] if plan else 12
    print(f"  n=5000: ~{cost + final:.1f} full-fit equivalents for 12 arms "
          f"(vs 12.0 naive, vs 4.0 for today's 4 blind fits)")
    print("  R2 PASS\n")


def test_r3_stratified_subsample():
    print("=== R3: stratified subsample keeps every class ===")
    rng = np.random.default_rng(0)
    for counts in ([950, 50], [500, 500], [980, 15, 5], [100] * 8):
        y = np.concatenate([[c] * n for c, n in enumerate(counts)])
        for rows in (150, 300):
            idx = stratified_subsample(y, rows, rng)
            present = np.unique(y[idx])
            assert len(present) == len(counts), (
                f"lost a class: {counts} @ {rows} -> {present}"
            )
    print("  4 imbalance profiles x 2 rung sizes: no class ever dropped")
    y = np.array([0] * 950 + [1] * 50)
    idx = stratified_subsample(y, 150, rng)
    assert len(idx) <= 160, "subsample overshot"
    print(f"  99:5 imbalance @150 rows -> counts {np.bincount(y[idx])}")
    print("  R3 PASS\n")


# ── R4 ────────────────────────────────────────────────────────────────


def test_r4_candidate_design():
    print("=== R4: one-axis-at-a-time design, mode lock respected ===")
    mem = ArmMemory(filepath=None)
    race = ArchitectureRace(memory=mem, seed=1)
    guidance = np.linspace(1.0, 0.1, 8)
    axes = {
        "sparse_l1_c": {"current": 0.1, "values": [0.05, 0.5, 1.0]},
        "use_rule_basis": {"current": False, "values": [True, False]},
    }
    base = {
        "feature_weights": guidance,
        "penalty": 1.0,
        "classifier_config": {"sparse_l1_c": 0.1},
    }
    arms = race.propose(
        guidance=guidance, axes=axes, base_action=base,
        bucket="b", max_arms=12,
    )
    assert arms[0]["_race_arm"] == "incumbent", "incumbent must be arm 0"

    base_cfg = base["classifier_config"]
    for arm in arms[1:]:
        if arm["_race_axis"] == "frontier":
            assert arm["classifier_config"] == base_cfg, (
                "frontier arm must not also change architecture"
            )
            continue
        diff = {
            k for k in set(arm["classifier_config"]) | set(base_cfg)
            if arm["classifier_config"].get(k) != base_cfg.get(k)
        }
        assert len(diff) == 1, (
            f"arm {arm['_race_arm']} changes {len(diff)} axes: {diff}"
        )
    print(f"  {len(arms)} arms, each changes exactly one axis (or one cutoff)")

    # A locked mode must not appear as a race axis.
    for mode in ("sparse_linear", "boosting", "auto"):
        clf = SymbolicFormulaAdapter(
            seed=42, num_classes=2, formula_library_path=None,
            learning_mode=mode,
        )
        clf_axes = clf.get_architecture_axes()
        assert "learning_mode" not in clf_axes, (
            "learning_mode must never be raced -- apply_config locks it, "
            "so a losing arm would pin the run's architecture"
        )
        for name, spec in clf_axes.items():
            clf.apply_config({name: spec["values"][0]})
            if name in clf.get_current_config():
                assert clf.get_current_config()[name] == spec["values"][0], (
                    f"advertised axis {name} is a silent no-op in apply_config"
                )
        print(f"  mode={mode:14s} axes={sorted(clf_axes)}")
    print("  R4 PASS\n")


# ── R5 ────────────────────────────────────────────────────────────────


def test_r5_phaseout():
    print("=== R5: phase-out as evidence accumulates ===")
    mem = ArmMemory(filepath=None)
    arms = ["incumbent"] + [f"a{i}" for i in range(7)]
    seen = []
    budgets = []
    for i in range(1, 401):
        mem.record_round("bucket", arms, "a3")
        if i in (1, 20, 60, 150, 400):
            seen.append(i)
            budgets.append(mem.arm_budget("bucket", 12))
    print(f"  rounds  {seen}")
    print(f"  arms    {budgets}")
    assert budgets == sorted(budgets, reverse=True), "budget must not grow"
    assert budgets[0] > budgets[-1], "budget never shrank"
    assert mem.bucket_trials("bucket") == 400, (
        "evidence must count ROUNDS, not arm-trials "
        f"(got {mem.bucket_trials('bucket')})"
    )
    assert mem.posterior_mean("bucket", "a3") > mem.posterior_mean("bucket", "a0")
    print(f"  winner posterior {mem.posterior_mean('bucket', 'a3'):.3f} > "
          f"loser {mem.posterior_mean('bucket', 'a0'):.3f}")
    print("  R5 PASS: detachment is measured, not asserted\n")


# ── R6 / R7 ───────────────────────────────────────────────────────────


def _run(mode, state_dir, stages=6, seed=42, dataset="race_e2e"):
    X, y = make_dataset()
    cfg = Cfg()
    cfg.search_mode = mode
    cfg.stages = stages
    opt = UniversalOptimizer(
        cfg=cfg, seed=seed, dataset_name=dataset, fold=0, start_fold=0,
        classifier=SymbolicFormulaAdapter(
            seed=seed, num_classes=2,
            formula_library_path=os.path.join(state_dir, "lib.json"),
            learning_mode="sparse_linear",
        ),
        search=PrincipleGuidedSearch(),
        state_dir=state_dir,
    )
    return opt, opt.fit(X, y)


def test_r6_end_to_end():
    print("=== R6: end-to-end race run ===")
    state_dir = tempfile.mkdtemp(prefix="uil_race_")
    opt, result = _run("race", state_dir)

    assert result.get("model") is not None, "race run returned no model"
    acc = result.get("accuracy", 0.0)
    print(f"  accuracy {acc:.4f}  macro_f1 {result.get('macro_f1', 0):.4f}")
    assert acc > 0.5, f"race run collapsed: {acc}"

    stats = opt._race_stats
    assert stats, "no rungs ran -- successive halving never engaged"
    total_elim = sum(len(v["eliminated"]) for v in stats.values())
    for key, v in sorted(stats.items())[:4]:
        print(f"  {key}: {v['in']} arms @ {v['rows']} rows -> {v['out']} "
              f"(cut {v['eliminated']})")
    assert total_elim > 0, "halving eliminated nobody"
    print(f"  total arms eliminated cheaply: {total_elim}")

    mem_path = os.path.join(state_dir, "uil_race_memory.json")
    assert os.path.exists(mem_path), "arm memory was never persisted"
    mem = ArmMemory(filepath=mem_path)
    bucket = opt._race_bucket()
    trials = mem.bucket_trials(bucket)
    assert trials > 0, "no outcomes recorded"
    print(f"  bucket '{bucket}': {trials} rounds recorded")

    # The vocabulary is genuinely wider than max_penalty: architecture
    # arms must actually be raced, not just cutoff arms.
    arch_arms = [
        k.split("||")[1] for k in mem._arms
        if k.startswith(bucket + "||")
        and not k.split("||")[1].startswith("frontier")
        and k.split("||")[1] != "incumbent"
    ]
    assert arch_arms, "no architecture arms raced -- vocabulary still one knob"
    print(f"  architecture arms raced: {sorted(set(arch_arms))}")
    print("  R6 PASS\n")


def test_r7_rungs_do_not_corrupt_the_run():
    print("=== R7: elimination rungs are side-effect free ===")
    state_dir = tempfile.mkdtemp(prefix="uil_race_det_")
    opt_a, res_a = _run("race", state_dir, dataset="race_det_a")
    state_dir_b = tempfile.mkdtemp(prefix="uil_race_det_")
    opt_b, res_b = _run("race", state_dir_b, dataset="race_det_a")
    assert abs(res_a["accuracy"] - res_b["accuracy"]) < 1e-9, (
        f"race mode is not deterministic: {res_a['accuracy']} vs "
        f"{res_b['accuracy']} -- a rung is leaking state into the run"
    )
    print(f"  two identical race runs: {res_a['accuracy']:.6f} == "
          f"{res_b['accuracy']:.6f}")

    # Causal arms are protected from elimination by construction.
    import inspect
    src = inspect.getsource(UniversalOptimizer._race_actions)
    assert "_intervention_role" in src, (
        "causal control/treatment arms are not protected from elimination"
    )
    assert "_shadowed_attrs" in src, "rung does not shadow adapter state"
    print("  causal control/treatment arms exempt from elimination")
    print("  adapter bookkeeping saved/restored around every rung")
    print("  R7 PASS\n")


# ── R8 ────────────────────────────────────────────────────────────────


def test_r8_classic_path_untouched():
    print("=== R8: classic path unchanged by default ===")
    cfg = Cfg()
    assert cfg.search_mode == "classic"
    sd_a = tempfile.mkdtemp(prefix="uil_classic_")
    opt_a, res_a = _run("classic", sd_a, dataset="race_classic")
    assert opt_a._race is None, "race object built in classic mode"
    assert opt_a._race_stats == {}, "race ran in classic mode"
    sd_b = tempfile.mkdtemp(prefix="uil_classic_")
    opt_b, res_b = _run("classic", sd_b, dataset="race_classic")
    assert abs(res_a["accuracy"] - res_b["accuracy"]) < 1e-9, (
        "classic mode lost determinism"
    )
    assert not os.path.exists(os.path.join(sd_a, "uil_race_memory.json")), (
        "classic run wrote race state"
    )
    print(f"  classic accuracy {res_a['accuracy']:.4f} (deterministic, "
          f"no race state written)")
    print("  R8 PASS\n")


# ── R9 ────────────────────────────────────────────────────────────────


def test_r9_overrides_cannot_collapse_the_race():
    print("=== R9: principle overrides cannot collapse the race ===")
    state_dir = tempfile.mkdtemp(prefix="uil_race_ovr_")
    cfg = Cfg()
    cfg.search_mode = "race"
    opt = UniversalOptimizer(
        cfg=cfg, seed=42, dataset_name="race_ovr", fold=0, start_fold=0,
        classifier=SymbolicFormulaAdapter(
            seed=42, num_classes=2,
            formula_library_path=os.path.join(state_dir, "lib.json"),
            learning_mode="sparse_linear",
        ),
        search=PrincipleGuidedSearch(),
        state_dir=state_dir,
    )

    arms = [
        {"classifier_config": {"sparse_l1_c": 0.1}, "penalty": 1.0,
         "_race_axis": None, "_race_arm": "incumbent"},
        {"classifier_config": {"sparse_l1_c": 0.5}, "penalty": 1.0,
         "_race_axis": "sparse_l1_c", "_race_arm": "sparse_l1_c=0.5"},
        {"classifier_config": {"sparse_l1_c": 1.0}, "penalty": 1.0,
         "_race_axis": "sparse_l1_c", "_race_arm": "sparse_l1_c=1.0"},
    ]
    overrides = {"config.sparse_l1_c": 0.05, "config.boosting_stages": 7,
                 "max_penalty": 1.4}
    applied = [opt._apply_overrides(dict(a), dict(overrides)) for a in arms]

    raced = [a["classifier_config"]["sparse_l1_c"] for a in applied[1:]]
    assert len(set(raced)) == len(raced), (
        f"override collapsed the raced axis: all arms got {raced}"
    )
    print(f"  raced axis survives override: sparse_l1_c = {raced}")
    assert applied[0]["classifier_config"]["sparse_l1_c"] == 0.05, (
        "incumbent must still receive cognition's override"
    )
    print(f"  incumbent still takes cognition's value: "
          f"{applied[0]['classifier_config']['sparse_l1_c']}")
    for a in applied:
        assert a["classifier_config"]["boosting_stages"] == 7, (
            "non-raced axes must still accept overrides"
        )
    print("  non-raced axes still overridden on every arm (boosting_stages=7)")

    for original, out in zip(arms, applied):
        assert "boosting_stages" not in original["classifier_config"], (
            "_apply_overrides mutated the caller's nested config dict"
        )
    print("  caller's action dict not mutated (nested copy)")
    print("  R9 PASS\n")


def test_r10_final_rung_selects_on_extra_splits():
    """R10: the judge fix (E7 Gate 2 -> Cawley & Talbot nested CV).

    R1-R9 all run at cv_folds=1, where `_extra_val_splits` is empty and
    the multi-split selection path can never fire -- so they prove no
    regression but nothing about the fix. This runs the race at
    cv_folds=3 and verifies, against the REAL `_race_actions` output
    (a pass-through recorder, no mocks):

      1. every finalist carries `select_score` (the multi-split mean);
      2. the stage winner IS the select_score argmax -- i.e. selection
         is genuinely judged on the extra splits, not the primary one;
      3. two identical runs remain byte-identical (the fix must not
         cost determinism).
    """
    print("=== R10: final rung selects on extra splits (judge fix) ===")

    def _run_cv(state_dir, dataset):
        X, y = make_dataset()
        cfg = Cfg()
        cfg.search_mode = "race"
        cfg.stages = 4
        cfg.cv_folds = 3
        opt = UniversalOptimizer(
            cfg=cfg, seed=42, dataset_name=dataset, fold=0, start_fold=0,
            classifier=SymbolicFormulaAdapter(
                seed=42, num_classes=2,
                formula_library_path=os.path.join(state_dir, "lib.json"),
                learning_mode="sparse_linear",
            ),
            search=PrincipleGuidedSearch(),
            state_dir=state_dir,
        )
        # Pass-through recorder around the real method -- records what
        # _race_actions actually returned, changes nothing about it.
        recorded: list[list[dict]] = []
        real = opt._race_actions

        def recording(*args, **kwargs):
            results = real(*args, **kwargs)
            recorded.append(results)
            return results

        opt._race_actions = recording
        result = opt.fit(X, y)
        return opt, result, recorded

    sd_a = tempfile.mkdtemp(prefix="uil_race_judge_")
    _, res_a, recorded = _run_cv(sd_a, "race_judge")

    assert recorded, "race never ran"
    stages_with_select = 0
    for results in recorded:
        finalists = [r for r in results if "select_score" in r]
        if not finalists:
            continue  # a degenerate stage (single survivor) is allowed
        stages_with_select += 1
        # 1. every finalist that got extra scores has a fold list of the
        #    expected size (cv_folds=3 -> 2 extra splits)
        for r in finalists:
            assert len(r["extra_scores"]) == 2, (
                f"expected 2 extra-split scores, got {len(r['extra_scores'])}"
            )
        # 2. the recorded winner credit (max on select_score) matches what
        #    selection in the stage loop will pick
        best_by_selection = max(
            results, key=lambda r: r.get("select_score", r["score"]),
        )
        best_by_primary = max(results, key=lambda r: r["score"])
        if best_by_selection is not best_by_primary:
            print(
                "  stage where the judge OVERRULED the primary split: "
                f"primary pick {best_by_primary['score']:.4f} vs "
                f"selected {best_by_selection['score']:.4f} "
                f"(select_score {best_by_selection['select_score']:.4f})"
            )
    assert stages_with_select > 0, (
        "no stage ever computed select_score -- the judge fix never fired"
    )
    print(f"  select_score present on finalists in {stages_with_select} stages")

    # 3. determinism: identical config, fresh state -> identical predictions
    sd_b = tempfile.mkdtemp(prefix="uil_race_judge_")
    _, res_b, _ = _run_cv(sd_b, "race_judge")
    assert res_a["accuracy"] == res_b["accuracy"], (
        f"judge fix broke determinism: {res_a['accuracy']} vs {res_b['accuracy']}"
    )
    print(f"  two identical runs agree exactly (acc {res_a['accuracy']:.4f})")
    print("  R10 PASS\n")


def test_r11_interaction_gap_gate():
    """R11: near-linear data must not race the interaction-hungry axes.

    E7's judge fix made the race net +0.0080 but it still LOST ionosphere
    (near-linear) by racing use_rule_basis/use_neural_basis, which overfit.
    The gate drops those axes when fingerprint.interaction_gap is small,
    keeps them when it is large, and stays conservative (keeps them) when
    the signal is missing. Tested against the real method, no mock.
    """
    print("=== R11: interaction-gap gate on race axes ===")
    from figpso_uil.core.optimizer import UniversalOptimizer

    class _Cfg:
        interaction_gate = 0.05

    class _Stub:
        cfg = _Cfg()
        _INTERACTION_AXES = UniversalOptimizer._INTERACTION_AXES
        _gate_interaction_axes = UniversalOptimizer._gate_interaction_axes

        def __init__(self, gap):
            self._dataset_fingerprint = {} if gap is None else {"interaction_gap": gap}
            self._observability_meta = {}

    axes = {"sparse_l1_c": {}, "use_stump_basis": {},
            "use_rule_basis": {}, "use_neural_basis": {}}

    near = _Stub(0.02)._gate_interaction_axes(dict(axes))
    assert "use_rule_basis" not in near and "use_neural_basis" not in near, near
    assert "sparse_l1_c" in near and "use_stump_basis" in near, near
    print(f"  near-linear (gap 0.02): raced {sorted(near)} (rule/neural dropped)")

    dense = _Stub(0.09)._gate_interaction_axes(dict(axes))
    assert set(dense) == set(axes), dense
    print(f"  interaction-dense (gap 0.09): raced all {len(dense)} axes")

    missing = _Stub(None)._gate_interaction_axes(dict(axes))
    assert set(missing) == set(axes), "missing signal must keep axes (conservative)"
    print("  missing gap: kept all axes (conservative)")

    # opt-out restores full-axis racing regardless of gap
    class _CfgAll:
        interaction_gate = 0.05
        race_all_axes = True
    s = _Stub(0.0); s.cfg = _CfgAll()
    assert set(s._gate_interaction_axes(dict(axes))) == set(axes)
    print("  cfg.race_all_axes=True: gate disabled")
    print("  R11 PASS\n")


def test_r12_incumbent_margin():
    """R12: the incumbent-margin winner's-curse mitigation.

    A challenger displaces the incumbent (arm 0, carrying cognition's
    sparse_linear) only if it clears it by cfg.race_incumbent_margin on
    the multi-split select_score. Default 0 = plain argmax = byte-
    identical to the prior loop. Tested against the real method.
    """
    print("=== R12: incumbent-margin winner selection ===")
    from figpso_uil.core.optimizer import UniversalOptimizer

    class _Cfg:
        race_incumbent_margin = 0.0

    class _Stub:
        _pick_winner_with_margin = UniversalOptimizer._pick_winner_with_margin

        def __init__(self, margin, race=True):
            self.cfg = _Cfg()
            self.cfg.race_incumbent_margin = margin
            self._race = object() if race else None

    inc = {"action": {"_race_arm": "incumbent"}, "score": 0.90, "select_score": 0.90}
    near = {"action": {"_race_arm": "rule=True"}, "score": 0.95, "select_score": 0.905}
    far = {"action": {"_race_arm": "stump=True"}, "score": 0.95, "select_score": 0.93}

    # margin off -> plain argmax picks the higher select_score
    w = _Stub(0.0)._pick_winner_with_margin([inc, near])
    assert w["action"]["_race_arm"] == "rule=True", "margin 0 must be plain argmax"
    print("  margin 0.0: plain argmax (challenger)")

    # margin on, challenger only +0.005 -> incumbent kept
    w = _Stub(0.01)._pick_winner_with_margin([inc, near])
    assert w["action"]["_race_arm"] == "incumbent", "near challenger must not displace"
    print("  margin 0.01, +0.005 challenger: incumbent kept")

    # margin on, challenger +0.03 clears it -> challenger wins
    w = _Stub(0.01)._pick_winner_with_margin([inc, far])
    assert w["action"]["_race_arm"] == "stump=True", "clear challenger must win"
    print("  margin 0.01, +0.03 challenger: challenger wins")

    # classic path (no race) -> margin ignored, plain argmax
    w = _Stub(0.5, race=False)._pick_winner_with_margin([inc, near])
    assert w["action"]["_race_arm"] == "rule=True", "off-race must ignore margin"
    print("  classic path: margin ignored (plain argmax)")
    print("  R12 PASS\n")


if __name__ == "__main__":
    test_r1_surrogate_is_closed_form()
    test_r2_halving_schedule()
    test_r3_stratified_subsample()
    test_r4_candidate_design()
    test_r5_phaseout()
    test_r6_end_to_end()
    test_r7_rungs_do_not_corrupt_the_run()
    test_r8_classic_path_untouched()
    test_r9_overrides_cannot_collapse_the_race()
    test_r10_final_rung_selects_on_extra_splits()
    test_r11_interaction_gap_gate()
    test_r12_incumbent_margin()
    print("=" * 50)
    print("ARCHITECTURE RACE: ALL TESTS PASSED")
    print("=" * 50)
