"""Unified SymbolicFormula classifier — standalone with optional boosting/bagging/hybrid.

Each row: weighted features → formula tree(s) → score → class.
Modes: single | boosting | bagging | multi (hybrid stack) | auto.
Method templates (logistic, stump, RBF, …) seed the formula library.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from figpso_uil.cognition.formula_engine import FormulaEngine, FormulaNode
from figpso_uil.cognition.formula_library import FormulaLibrary
from figpso_uil.cognition.paths import results_path
from figpso_uil.cognition.symbolic_gradient import fit_boost_stage, stage_contribution
from figpso_uil.core.small_data_diversify import ensure_min_rows
from figpso_uil.core.utils import blas_thread_scope, safe_minmax, coerce_float_array


class SymbolicFormulaAdapter:
    """Formula-tree classifier with internal boosting/bagging."""

    _THRESHOLD_GRID = np.linspace(0.05, 0.95, 19)
    LEARNING_MODES = ("auto", "single", "boosting", "bagging", "multi", "sparse_linear")
    HYBRID_BRANCHES = ("single", "boosting", "gradient", "bagging", "boosted_bag")

    def __init__(
        self,
        seed: int,
        num_classes: int,
        formula_library_path: str | None = None,
        boosting_rounds: int = 5,
        bagging_estimators: int = 5,
        learning_mode: str = "auto",
        learning_rate: float = 0.35,
    ):
        self.seed = seed
        self.num_classes = num_classes
        self.boosting_stages = max(1, int(boosting_rounds))
        self.bagging_estimators = max(1, int(bagging_estimators))
        self.learning_mode = learning_mode if learning_mode in self.LEARNING_MODES else "auto"
        # A mode chosen by the router/operator is an architectural decision
        # and must survive BOTH the PSO action search and champion-config
        # restore (both silently overrode it before: PSO from a hardcoded
        # list lacking sparse_linear, the champion from a stale prior run).
        # "auto" = explore freely; explicit = locked.
        self._mode_locked = self.learning_mode != "auto"
        self.learning_rate = float(np.clip(learning_rate, 0.05, 1.0))
        self.formula_engine = FormulaEngine(seed=seed)
        self.formula_library = FormulaLibrary(
            filepath=formula_library_path
            or results_path("uil_symbolic_classifier_library.json")
        )
        self.mutation_rounds_per_fit = 0
        # sparse_linear basis-family gates: the router/fingerprint decides
        # which families to enable (stumps help spambase-like data, hurt
        # covertype-like). All three on by default.
        self.use_raw_basis = True
        self.use_symbolic_basis = True
        self.use_stump_basis = True
        # Rule family (RuleFit-style conjunctions of stump conditions, e.g.
        # 1[x3>0.2 & x7<=0.5]) captures FEATURE INTERACTIONS that the purely
        # additive raw/stump/formula basis cannot — the capacity gap on
        # interaction-dense data like sonar (measured: additive ~0.75,
        # readable rules ~0.81, black-box RBF ~0.89). Each rule stays a
        # named, readable, editable term. OFF by default (adds compute +
        # can overfit low-interaction data); the router enables it for
        # fingerprints that need interactions, like the stump gate.
        self.use_rule_basis = False
        # Neural family: hidden units of a small MLP, used as BASIS
        # FUNCTIONS rather than as the model. The MLP supplies the dense
        # non-linear capacity that the additive families provably lack
        # (sonar: additive ~0.75 vs MLP/RBF ~0.88); the L1 readout still
        # decides which of its units survive, jointly with raw/stump/rule/
        # formula terms, so the TOP-LEVEL law stays a readable weighted sum
        # of named terms. Each unit is a named, inspectable, editable cell
        # whose weight vector is stored on the model. OFF by default.
        self.use_neural_basis = False
        # Research step 8 — KAN-style univariate truncated-power splines.
        # Plottable / prunable / readable as (x_j-t)_+^d. Default OFF;
        # raced like neural/rules. Pure numpy (no pykan dependency).
        self.use_kan_basis = False
        self.KAN_FEATURES = 8
        self.KAN_KNOTS = 5
        self.KAN_DEGREE = 2
        # Bug #25 fix (OFF by default until cross-seed validated): let the
        # search's feature weights actually REACH the sparse_linear model.
        # Today they cannot — _basis_matrix ignores them for raw/stump/rule
        # terms, and standardization would cancel any column scaling anyway,
        # so the whole cognitive layer is a no-op (ablation delta 0.0000).
        self.use_feature_weight_penalty = False
        # Research step 4 (DreamCoder-style utility gate). Default OFF —
        # Gate 2 required before shipping. When on: incubating formulas
        # enter the sparse_linear basis with a newborn L1 boost, and
        # birth is gated on a held-out improvement test.
        self.utility_gated_formulas = False
        # Direct influence (opt-in): (1) grow formulas on this dataset's
        # residual/target, (2) install them as a forced symbolic stage
        # that does NOT compete under the same L1 as 200 stumps. Default
        # OFF — needs existence CONNECTED + Gate 2 before shipping.
        self.direct_formula_install = False
        self._residual_install_ids: list[str] = []
        # In-memory forced trees for this fit (avoid minting a new library
        # entry on every candidate evaluation — that flooded the library).
        self._forced_residual_trees: list = []
        self.RESIDUAL_INSTALL_TERMS = 3
        self.MAX_FORCED_FORMULAS = 6
        self.RESIDUAL_OFFLINE_TRIALS = 48
        # Signal translator directive (opt-in): prefer formulas, strip
        # symbolic basis when formulas ablated, boost preferred terms.
        self._learner_directive = None
        self.PREFER_L1_BOOST = 2.5
        self.sparse_l1_c = self.SPARSE_L1_C
        # Research step 5 — EBM-style cyclic fit over the composed basis.
        # Opt-in; raced via get_architecture_axes when learning_mode is
        # sparse_linear/auto. Default OFF pending Gate 2.
        self.use_ebm_fit = False
        self.ebm_rounds = 40
        self.ebm_learning_rate = 0.08
        self._last_used_formula_id: str | None = None
        self._formulas_used_this_fit: list[str] = []
        self._last_tournament: list[dict] = []
        self._last_hybrid_blend: dict[str, float] | None = None
        self._episodic_suggestion: str | None = None
        self._episodic_confidence_threshold = 0.5
        self._used_episodic_suggestion = False
        self._rng = np.random.default_rng(seed)
        self.tournament_k = 8
        self.hybrid_branch_count = 5
        self.parallel_eval_workers = 1
        self._branch_parallel = True
        self._train_augmented = False
        self._graph_hint: dict | None = None
        self._exploration_hint: dict | None = None

    @property
    def name(self) -> str:
        return "SymbolicFormula"

    @property
    def supports_warm_start(self) -> bool:
        return True

    @property
    def supports_custom_objective(self) -> bool:
        return False

    def set_learner_directive(self, directive) -> None:
        """Honor a stage LearnerDirective from cognition.signal_translator."""
        self._learner_directive = directive
        if directive is None:
            return
        if getattr(directive, "use_symbolic_basis", None) is not None:
            self.use_symbolic_basis = bool(directive.use_symbolic_basis)
        cfg = getattr(directive, "config", None) or {}
        if cfg:
            self.apply_config(cfg)

    def set_episodic_hint(self, hint: dict) -> None:
        confidence = hint.get("retrieval_confidence", 0.0)
        suggestion = hint.get("suggested_symbolic_formula")
        if suggestion and confidence >= self._episodic_confidence_threshold:
            self._episodic_suggestion = suggestion

    def set_graph_hint(self, hint: dict) -> None:
        """KG traversal hint: activated nodes + per-feature emphasis for tournament."""
        self._graph_hint = hint

    def set_exploration_hint(self, hint: dict | None) -> None:
        """Abstract memory: prior conclusions, low-accuracy formulas, champion."""
        self._exploration_hint = hint or {}
        champion = self._exploration_hint.get("champion")
        if champion and champion.get("formula_id"):
            self._episodic_suggestion = champion["formula_id"]
            self._episodic_confidence_threshold = 0.35

    def _exploration_penalty(self, formula_id: str) -> float:
        hint = self._exploration_hint or {}
        dataset = hint.get("dataset", "")
        if dataset and hasattr(self.formula_library, "is_explored_low"):
            if self.formula_library.is_explored_low(formula_id, dataset):
                return -0.35
        for item in hint.get("avoid_formulas", []):
            if item.get("formula_id") == formula_id:
                return -0.25
        for item in hint.get("known_good_formulas", []):
            if item.get("formula_id") == formula_id:
                return 0.12
        return 0.0

    def _should_skip_explored_formula(self, entry: dict) -> bool:
        hint = self._exploration_hint or {}
        dataset = hint.get("dataset")
        if not dataset or entry.get("evidence", 0) < 4:
            return False
        fid = entry.get("formula_id", "")
        if hasattr(self.formula_library, "is_explored_low"):
            return self.formula_library.is_explored_low(fid, dataset)
        return False

    def _graph_formula_bonus(self, tree, n_features: int) -> float:
        hint = self._graph_hint or {}
        emphasis = hint.get("feature_emphasis")
        if emphasis is None:
            return 0.0
        emphasis = np.asarray(emphasis, dtype=float)
        if emphasis.size != n_features or emphasis.sum() <= 0:
            return 0.0
        usage = self._feature_usage_counts(tree, n_features)
        if usage.sum() <= 0:
            return 0.0
        bonus = 0.1 * float(np.dot(usage / usage.sum(), emphasis / emphasis.sum()))
        bonus += 0.03 * min(1.0, tree.node_count() / 24.0)
        return bonus

    def _sk_random_state(self) -> int:
        """A stable, non-negative seed for sklearn estimators.

        Every sklearn estimator this adapter builds must be handed one.
        `random_state=None` means "draw from the global numpy RNG", which
        is process-entropy-seeded -- see bug #31 at the liblinear call.
        """
        return int(getattr(self, "seed", 0)) & 0x7FFFFFFF

    def clone_for_parallel_eval(
        self,
        *,
        branch_parallel: bool = False,
        parallel_workers: int | None = None,
    ) -> "SymbolicFormulaAdapter":
        """Fresh adapter sharing the formula library (read-mostly during fit)."""
        worker = SymbolicFormulaAdapter(
            seed=self.seed,
            num_classes=self.num_classes,
            formula_library_path=self.formula_library.filepath,
            boosting_rounds=self.boosting_stages,
            bagging_estimators=self.bagging_estimators,
            learning_mode=self.learning_mode,
            learning_rate=self.learning_rate,
        )
        worker.apply_config(self.get_current_config())
        worker._branch_parallel = branch_parallel
        if parallel_workers is not None:
            worker.parallel_eval_workers = max(1, int(parallel_workers))
        if self._graph_hint is not None:
            worker.set_graph_hint(self._graph_hint)
        if self._episodic_suggestion is not None:
            worker._episodic_suggestion = self._episodic_suggestion
            worker._episodic_confidence_threshold = self._episodic_confidence_threshold
        if self._exploration_hint is not None:
            worker.set_exploration_hint(self._exploration_hint)
        if self._learner_directive is not None:
            worker.set_learner_directive(self._learner_directive)
        return worker

    def _resolve_mode(self, n_features: int, n_samples: int) -> str:
        if self.learning_mode != "auto":
            return self.learning_mode
        if n_samples < 150 and n_features >= 12:
            return "boosting"
        if self.hybrid_branch_count >= 3 and n_samples >= 200:
            return "multi"
        if n_features >= 20 or self.boosting_stages >= 8:
            return "boosting"
        if n_features >= 8 and self.bagging_estimators >= 3:
            return "bagging"
        return "single"

    @staticmethod
    def _prepare_X(X: np.ndarray, feature_weights: np.ndarray | None) -> np.ndarray:
        if feature_weights is None:
            return X
        w = coerce_float_array(feature_weights)
        if w is None:
            return X
        w = np.asarray(w, dtype=float)
        if w.size != X.shape[1]:
            return X
        w = safe_minmax(w)
        if np.max(w) < 1e-9:
            return X
        return X * (0.25 + 0.75 * w)

    def _coerce_tree(self, tree: object, formula_id: str | None = None) -> FormulaNode:
        """Models loaded from JSON/champion store trees as dicts, not FormulaNode."""
        if isinstance(tree, FormulaNode):
            return tree
        if isinstance(tree, dict):
            return FormulaNode.from_dict(tree)
        if isinstance(tree, str):
            fid = formula_id if (formula_id and not tree.startswith("(")) else None
            if fid or tree.startswith("F-") or tree.startswith("SEED-"):
                resolved = self.formula_library.get_formula_tree(fid or tree)
                if resolved is not None:
                    return resolved
        if formula_id:
            resolved = self.formula_library.get_formula_tree(formula_id)
            if resolved is not None:
                return resolved
        raise TypeError(
            f"Expected FormulaNode or dict tree, got {type(tree).__name__}"
        )

    def _repair_model_trees(self, model: dict) -> None:
        """Fix champion JSON where trees were stored as repr strings."""
        kind = model.get("kind")
        if kind == "hybrid":
            for branch in model.get("branches", {}).values():
                self._repair_model_trees(branch)
        elif kind == "boosted_bag":
            for e in model.get("estimators", []):
                if "submodel" in e:
                    self._repair_model_trees(e["submodel"])
        elif kind == "boosting":
            for stage in model.get("stages", []):
                if isinstance(stage.get("tree"), str) and stage.get("formula_id"):
                    t = self.formula_library.get_formula_tree(stage["formula_id"])
                    if t is not None:
                        stage["tree"] = t
        elif kind == "bagging":
            for e in model.get("estimators", []):
                if isinstance(e.get("tree"), str) and e.get("formula_id"):
                    t = self.formula_library.get_formula_tree(e["formula_id"])
                    if t is not None:
                        e["tree"] = t
        elif kind == "multiclass_ovr":
            # Bug #28: JSON object keys are ALWAYS strings, so a champion
            # round-tripped through the archive comes back keyed "0","1","2"
            # instead of 0,1,2. predict() then argmaxes into np.array(keys)
            # and returns '2' (dtype <U1) instead of 2 -- every downstream
            # metric raises "Mix of label input types (string and number)".
            # Restore the integer keys here, at the designated post-load
            # repair hook, so nothing downstream has to care.
            heads = model.get("classifiers", {})
            if heads and any(isinstance(k, str) for k in heads):
                try:
                    model["classifiers"] = {int(k): v for k, v in heads.items()}
                except (TypeError, ValueError):
                    pass  # non-integer class labels: leave as-is
            for clf in model.get("classifiers", {}).values():
                self._repair_model_trees(clf)
        elif isinstance(model.get("tree"), str) and model.get("formula_id"):
            t = self.formula_library.get_formula_tree(model["formula_id"])
            if t is not None:
                model["tree"] = t

    def _raw_from_tree(
        self,
        tree: FormulaNode | dict,
        X: np.ndarray,
        feature_weights: np.ndarray | None,
        *,
        formula_id: str | None = None,
    ) -> np.ndarray:
        X_eff = self._prepare_X(X, feature_weights)
        return self.formula_engine.evaluate(
            self._coerce_tree(tree, formula_id=formula_id), {"X": X_eff},
        )

    def _scores_from_tree(
        self,
        tree: FormulaNode | dict,
        X: np.ndarray,
        feature_weights: np.ndarray | None,
        *,
        formula_id: str | None = None,
    ) -> np.ndarray:
        raw = self._raw_from_tree(tree, X, feature_weights, formula_id=formula_id)
        return np.clip(1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500))), 0.0, 1.0)

    def _to_binary_y(self, y: np.ndarray) -> tuple[np.ndarray, int | None]:
        unique = np.unique(y)
        if len(unique) == 2 and not np.array_equal(unique, [0, 1]):
            return (y == unique.max()).astype(int), int(unique.max())
        return y.astype(int), None

    # Base rate above which plain-accuracy threshold fitting degenerates.
    IMBALANCE_CUTOFF = 0.75

    def _fit_threshold(self, scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Pick the decision threshold on the fitted scores.

        Objective is accuracy on balanced data, but BALANCED accuracy once
        the majority class exceeds IMBALANCE_CUTOFF. Reason: at an 88/12
        base rate, predicting all-majority is *literally* the accuracy
        optimum whenever score separation is weak (bank_marketing: t=0.40
        -> train_acc 0.8815 using the signal vs t>=0.43 -> 0.8830 all-
        majority). Plain-accuracy fitting therefore threw the signal away
        and emitted a single-class model (AUC 0.60-0.65 -> balanced acc
        0.500). Balanced accuracy keeps the model discriminating; on such
        data headline accuracy dips a few tenths of a point, which is the
        honest trade -- a model that always answers "no" is useless
        whatever its accuracy.
        """
        y_bin, _ = self._to_binary_y(y)
        base_rate = max(float(np.mean(y_bin == 0)), float(np.mean(y_bin == 1)))
        imbalanced = base_rate >= self.IMBALANCE_CUTOFF

        def _score(pred: np.ndarray) -> float:
            if not imbalanced:
                return float(np.mean(pred == y_bin))
            # Balanced accuracy = mean per-class recall (undefined classes skipped).
            recalls = []
            for cls in (0, 1):
                mask = y_bin == cls
                if mask.any():
                    recalls.append(float(np.mean(pred[mask] == cls)))
            return float(np.mean(recalls)) if recalls else 0.0

        best_threshold, best_metric = 0.5, -1.0
        for t in self._THRESHOLD_GRID:
            m = _score((scores >= t).astype(int))
            if m > best_metric:
                best_metric, best_threshold = m, float(t)
        # Report plain accuracy at the chosen threshold so callers that
        # compare against accuracy elsewhere stay consistent.
        best_acc = float(np.mean((scores >= best_threshold).astype(int) == y_bin))
        return best_threshold, best_acc

    # Above this feature count, the formula grammar can only ever reference
    # a handful of columns (seeds cap at ~16, boosting adds ~1 feature/stage),
    # so on high-dim data (e.g. MNIST 784) the model references a tiny random
    # subset and collapses to majority-class -- while PSO + the 5-branch hybrid
    # churn over every column, costing hours. Preselect the top-K most
    # discriminative features and run the whole pipeline on that reduced,
    # RELEVANT basis. The selection is stored on the model (fully traceable /
    # overridable) and re-applied identically at predict time.
    FEATURE_SELECT_CAP = 64

    def fit_weighted(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_weights: np.ndarray,
        warm_start_model: object | None = None,
    ) -> object:
        X = np.asarray(X, dtype=float)
        fw_full = np.asarray(feature_weights, dtype=float) if feature_weights is not None else None

        # ── (a) high-dim feature preselection ────────────────────────
        selected = None
        if X.shape[1] > self.FEATURE_SELECT_CAP:
            if warm_start_model is not None and isinstance(warm_start_model, dict) \
                    and warm_start_model.get("selected_features") is not None:
                selected = np.asarray(warm_start_model["selected_features"], dtype=int)
            else:
                selected = self._select_features(X, y, fw_full, self.FEATURE_SELECT_CAP)
        if selected is not None:
            X = X[:, selected]
            fw = fw_full[selected] if fw_full is not None else None
        else:
            fw = fw_full

        n_features = X.shape[1]
        self._ensure_seeds(n_features)
        self._formulas_used_this_fit = []

        # Resolve the mode BEFORE the warm-start branch. It used to be
        # resolved after, so a warm-started fit never reached this line and
        # silently kept whatever mode it was copied with -- the search asks
        # for boosting/bagging/multi under stress and got no answer and no
        # error (bug #29).
        mode = self._resolve_mode(n_features, len(X))

        if warm_start_model is not None and isinstance(warm_start_model, dict):
            warmed = self._apply_warm_start(
                X, y, fw, warm_start_model, n_features, mode,
            )
            if warmed is not None:
                warmed["selected_features"] = (
                    selected.tolist() if selected is not None else None
                )
                return warmed

        X_fit, y_fit, self._train_augmented = ensure_min_rows(X, y, self._rng)

        if self.num_classes <= 2:
            model = self._fit_binary(X_fit, y_fit, fw, n_features, mode, eval_X=X, eval_y=y)
        else:
            model = self._fit_multiclass_ovr(X_fit, y_fit, fw, n_features, mode, eval_X=X, eval_y=y)

        model["feature_weights"] = fw.copy() if fw is not None else None
        model["n_features"] = n_features
        model["learning_mode"] = mode
        # None when no reduction happened; a list of original-column indices
        # otherwise. predict() reduces incoming X the same way.
        model["selected_features"] = selected.tolist() if selected is not None else None
        if model.get("kind") == "hybrid":
            for branch in model.get("branches", {}).values():
                branch["feature_weights"] = model["feature_weights"]
        return model

    def _select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_weights: np.ndarray | None,
        k: int,
    ) -> np.ndarray | None:
        """Top-k most discriminative feature indices (ANOVA F, fast).

        Blends the F-statistic with any informative PSO feature weights so
        the search's own relevance signal is respected. Returns sorted
        original-column indices, or None if selection can't be computed.
        """
        n = X.shape[1]
        if n <= k:
            return None
        try:
            from sklearn.feature_selection import f_classif
            score, _ = f_classif(X, y)
            score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            score = np.nan_to_num(X.var(axis=0), nan=0.0)
        if feature_weights is not None and feature_weights.shape[0] == n:
            fw = np.nan_to_num(feature_weights, nan=0.0)
            if fw.max() > fw.min():
                s = (score - score.min()) / (score.max() - score.min() + 1e-12)
                w = (fw - fw.min()) / (fw.max() - fw.min() + 1e-12)
                score = 0.7 * s + 0.3 * w
        idx = np.argsort(score)[::-1][:k]
        return np.sort(idx.astype(int))

    def _apply_warm_start(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_weights: np.ndarray | None,
        model: dict,
        n_features: int,
        mode: str | None = None,
    ) -> dict | None:
        """Reuse prior structure under new feature weights (fast resume).

        Returning None means "refit properly", which is the right answer
        whenever the copied structure would not actually answer the question
        being asked. Bug #29: this used to return a copy unconditionally for
        every mode except sparse_linear, so after the first fit of a run
        nothing was ever retrained -- not the formula choice, not the
        boosting rounds, not the coefficients, and not the decision
        threshold. The stage loop degenerated into re-weighting one frozen
        model, and evolved formulas from the library could never enter it.
        """
        if int(model.get("n_features", n_features)) != n_features:
            return None
        # sparse_linear coefficients are fitted jointly against a specific
        # basis; re-using them under new feature weights would serve stale
        # weights. Refit instead (it is a single L1 solve — cheap).
        if model.get("kind") == "sparse_linear":
            return None
        # The same reasoning applies to a MODE change: the search proposes
        # switching learner under stress, and silently handing back the old
        # architecture is a no-op disguised as a result.
        prior_mode = model.get("learning_mode")
        if mode is not None and prior_mode is not None and prior_mode != mode:
            return None
        import copy
        out = copy.deepcopy(model)
        self._repair_model_trees(out)
        out["feature_weights"] = (
            np.asarray(feature_weights, dtype=float).copy()
            if feature_weights is not None else None
        )
        if out.get("kind") == "hybrid":
            for branch in out.get("branches", {}).values():
                branch["feature_weights"] = out["feature_weights"]
        # New feature weights => new scores (the tree paths go through
        # _prepare_X), but the copied threshold was calibrated against the
        # OLD scores. Judging new numbers with an old ruler has no upside;
        # refitting is one pass over a small grid.
        self._refit_thresholds(out, X, y, feature_weights)
        ids = self._formula_ids_from_hybrid(out) if out.get("kind") == "hybrid" else []
        if not ids and out.get("formula_id"):
            ids = [out["formula_id"]]
        if ids:
            self._last_used_formula_id = ids[-1]
            self._formulas_used_this_fit = list(dict.fromkeys(ids))
        return out

    def _refit_thresholds(
        self,
        model: dict,
        X: np.ndarray,
        y: np.ndarray,
        feature_weights: np.ndarray | None,
    ) -> None:
        """Recalibrate a warm-started model's decision threshold(s) in place.

        Best-effort: a threshold that cannot be recomputed keeps its stored
        value, which is exactly the old behavior for that sub-model.
        """
        if model.get("kind") == "multiclass_ovr":
            for cls, head in model.get("classifiers", {}).items():
                try:
                    y_ovr = (np.asarray(y) == cls).astype(int)
                    scores = self._predict_binary_scores(head, X)
                    head["threshold"], _ = self._fit_threshold(scores, y_ovr)
                except Exception:
                    continue
            return
        try:
            y_bin, _ = self._to_binary_y(y)
            scores = self._predict_binary_scores(model, X)
            model["threshold"], _ = self._fit_threshold(scores, y_bin)
        except Exception:
            pass

    @staticmethod
    def _formula_ids_from_hybrid(model: dict) -> list[str]:
        ids: list[str] = []
        for name in model.get("branch_order", []):
            branch = model.get("branches", {}).get(name, {})
            ids.extend(SymbolicFormulaAdapter._formula_ids_from_branch(branch))
        return ids

    def _ensure_seeds(self, n_features: int) -> None:
        if not self.formula_library.get_active_formulas():
            seeds = self.formula_engine.get_seed_classifier_formulas(n_features)
            self.formula_library.seed_formulas(seeds)

    def _tree_or_seed(self, formula_id: str, n_features: int):
        """Resolve a formula's tree, falling back to a seed if it's gone.

        `_pick_formula` selects from the library listing, but the tree can
        come back None — the entry may have been pruned/archived between
        pick and fetch (the hybrid path fits branches in threads while the
        optimizer prunes), or its stored tree may be unrehydratable. That
        raised `TypeError: Expected FormulaNode or dict tree, got NoneType`
        and killed an entire dataset mid-benchmark. A missing formula is
        never worth crashing a run over.
        """
        tree = self.formula_library.get_formula_tree(formula_id)
        if tree is not None:
            return tree
        seeds = self.formula_engine.get_seed_classifier_formulas(n_features)
        if seeds:
            return next(iter(seeds.values()))
        return FormulaNode.const(0.5)

    # ── sparse linear over a composed symbolic basis (RuleFit-style) ──
    #
    # Why this exists: the boosting path adds ONE library formula per stage
    # and each seed formula references ~1 feature, so a 10-round fit touches
    # ~6 of 57 columns. Measured on spambase, the accuracy frontier for a
    # 6-term model is ~0.690 -- and boosting scored 0.685. The learner was
    # already at its ceiling FOR ITS SPARSITY; the gap was capacity, not
    # search quality. Reaching XGBoost parity needs ~66 jointly-fitted terms
    # (frontier: 6→0.690, 25→0.873, 43→0.921, 66→0.942 vs XGBoost 0.9435).
    #
    # Basis families (each entry stays a named, readable, editable term):
    #   raw      x_j
    #   formula  f(x) from the evolved library (the symbolic contribution)
    #   stump    1[x_j > theta]   (axis-aligned splits, as basis functions)
    # L1 logistic fits and SELECTS jointly -- selection by fitted utility,
    # not by a pre-hoc correlation screen. Measured (spambase / covertype):
    #   raw+L1 0.924/0.741 | symbolic 0.865/0.714 | stumps 0.934/0.723
    #   raw+symbolic+stumps 0.942/0.751 | XGBoost 0.944/0.780
    # Stumps help spambase but HURT covertype -> families are gated by config
    # so the router can pick per dataset fingerprint.
    STUMPS_PER_FEATURE = 3
    SPARSE_L1_C = 0.1
    MAX_SYMBOLIC_BASIS = 40
    # Research step 4: how many incubating (unproven) formulas may enter
    # the basis at once, and how strongly L1 is relaxed for them.
    MAX_NEWBORN_BASIS = 8
    NEWBORN_L1_BOOST = 2.0
    UTILITY_ADMIT_EPS = 1e-4
    # Direct-install: how strongly a forced formula rides in the free L1
    # path when we also keep preferred tags (belt-and-suspenders). The
    # primary install path is `_fit_sparse_linear_direct` (separate stage).
    FORCED_L1_BOOST = 25.0
    # Lowest scale a fully-deprioritised basis term can be pushed to. At
    # 0.25 a down-weighted feature carries 4x the effective L1 penalty of a
    # fully-weighted one — a real but bounded influence (matches the floor
    # already used by _prepare_X).
    WEIGHT_FLOOR = 0.25
    # Rule family (RuleFit-style): mine conjunctions from a shallow forest.
    RULE_TREES = 40      # trees whose leaf-paths become candidate rules
    RULE_DEPTH = 3       # max conditions per rule (interaction order)
    MAX_RULES = 40       # cap so the L1 problem stays bounded/readable

    def _extract_rules(
        self, X: np.ndarray, y_bin: np.ndarray, max_rules: int,
    ) -> list[list[dict]]:
        """Mine conjunctive rules (root->leaf paths of shallow trees).

        Each rule is a list of {feature, op, threshold} conditions AND-ed
        together — a readable, editable interaction term. Deduplicated
        across the forest; capped at max_rules.
        """
        if len(np.unique(y_bin)) < 2 or X.shape[0] < 8:
            return []
        try:
            from sklearn.ensemble import RandomForestClassifier
        except Exception:
            return []
        try:
            rf = RandomForestClassifier(
                n_estimators=self.RULE_TREES, max_depth=self.RULE_DEPTH,
                random_state=int(getattr(self, "seed", 0)), n_jobs=1,
            ).fit(X, y_bin)
        except Exception:
            return []
        rules: list[list[dict]] = []
        seen: set = set()
        for est in rf.estimators_:
            t = est.tree_

            def walk(node: int, conds: list) -> None:
                if len(rules) >= max_rules:
                    return
                left, right = t.children_left[node], t.children_right[node]
                if left == right:  # leaf
                    if conds:
                        key = tuple(sorted(
                            (c["feature"], c["op"], round(c["threshold"], 6))
                            for c in conds
                        ))
                        if key not in seen:
                            seen.add(key)
                            rules.append([dict(c) for c in conds])
                    return
                f, th = int(t.feature[node]), float(t.threshold[node])
                walk(left, conds + [{"feature": f, "op": "<=", "threshold": th}])
                walk(right, conds + [{"feature": f, "op": ">", "threshold": th}])

            walk(0, [])
            if len(rules) >= max_rules:
                break
        return rules[:max_rules]

    # Neural family. Small on purpose: these units compete under the SAME
    # L1 as the readable terms, and E5's lesson is that a large correlated
    # group starves (or drowns) the others unless the budget accounts for
    # it. 16 units is enough to carry interactions without taking over.
    NEURAL_UNITS = 16
    NEURAL_MAX_ITER = 300

    def _extract_neural_units(
        self, X: np.ndarray, y_bin: np.ndarray, n_units: int,
    ) -> list[dict]:
        """Fit a small MLP and return its hidden units as basis functions.

        The MLP is a FEATURE BUILDER here, not the classifier. Its first
        layer gives `n_units` learned non-linear projections; the sparse
        linear readout above then selects among them alongside the readable
        families. This is the honest middle of the readability trade: the
        top-level law stays a weighted sum of named terms, and a surviving
        neural term is inspectable (you can read its weight vector and see
        which inputs it leans on) without being as legible as a rule.
        """
        if len(np.unique(y_bin)) < 2 or X.shape[0] < 12:
            return []
        try:
            from sklearn.neural_network import MLPClassifier
        except Exception:
            return []
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                net = MLPClassifier(
                    hidden_layer_sizes=(n_units,),
                    activation="relu",
                    max_iter=self.NEURAL_MAX_ITER,
                    random_state=int(getattr(self, "seed", 0)),
                ).fit(X, y_bin)
        except Exception:
            return []
        W = np.asarray(net.coefs_[0], dtype=float)      # (n_features, n_units)
        b = np.asarray(net.intercepts_[0], dtype=float)  # (n_units,)
        units: list[dict] = []
        for u in range(W.shape[1]):
            w = W[:, u]
            if not np.isfinite(w).all() or float(np.max(np.abs(w))) < 1e-9:
                continue  # dead unit -- contributes a constant column
            units.append({
                "type": "neural",
                "unit": int(u),
                # Stored as plain lists so the model survives the JSON
                # round-trip the champion store puts it through (bug #28).
                "w": [float(v) for v in w],
                "b": float(b[u]) if u < b.size else 0.0,
                "act": "relu",
            })
        return units

    def _build_basis_spec(
        self, X: np.ndarray, y_bin: np.ndarray, n_features: int,
    ) -> list[dict]:
        """Compose the candidate basis; each entry is a readable term."""
        spec: list[dict] = []
        directive = getattr(self, "_learner_directive", None)
        use_sym = self.use_symbolic_basis
        if directive is not None and getattr(directive, "use_symbolic_basis", None) is not None:
            use_sym = bool(directive.use_symbolic_basis)
        prefer_ids = list(getattr(directive, "prefer_formula_ids", None) or [])

        if self.use_raw_basis:
            spec.extend({"type": "raw", "feature": int(j)} for j in range(n_features))
        if use_sym:
            seen: set[str] = set()
            forced_ids = set(getattr(self, "_residual_install_ids", None) or [])
            actives = self.formula_library.get_active_formulas()[: self.MAX_SYMBOLIC_BASIS]
            for f in actives:
                fid = f["formula_id"]
                tree = self.formula_library.get_formula_tree(fid)
                if tree is None:
                    continue
                origin = str(f.get("origin") or "")
                forced = (
                    fid in forced_ids
                    or origin == "residual_install"
                    or fid in prefer_ids
                )
                spec.append({
                    "type": "formula",
                    "formula_id": fid,
                    "tree": tree.to_dict() if hasattr(tree, "to_dict") else tree,
                    "preferred": fid in prefer_ids,
                    "forced": bool(forced and self.direct_formula_install),
                })
                seen.add(fid)
            # Translator: force preferred formulas into the basis even if
            # they were not in the active listing (HITL / episodic / graph).
            for fid in prefer_ids:
                if fid in seen:
                    continue
                tree = self.formula_library.get_formula_tree(fid)
                if tree is None:
                    continue
                spec.append({
                    "type": "formula",
                    "formula_id": fid,
                    "tree": tree.to_dict() if hasattr(tree, "to_dict") else tree,
                    "preferred": True,
                    "forced": bool(self.direct_formula_install),
                })
                seen.add(fid)
            # Residual installs may still be incubating if something else
            # demoted them — pull by id so the forced stage can see them.
            for fid in forced_ids:
                if fid in seen:
                    continue
                tree = self.formula_library.get_formula_tree(fid)
                if tree is None:
                    continue
                spec.append({
                    "type": "formula",
                    "formula_id": fid,
                    "tree": tree.to_dict() if hasattr(tree, "to_dict") else tree,
                    "preferred": fid in prefer_ids,
                    "forced": True,
                })
                seen.add(fid)
            # Research step 4: incubating formulas were previously EXCLUDED
            # from the sparse_linear basis (get_active_formulas skips them),
            # so a newborn could never be L1-selected, never earn
            # record_result credit, and never promote — a deadlock that
            # made the existence audit read `formulas` INERT. Opt-in:
            # let a capped set of incubating formulas compete, tagged
            # newborn so the L1 sub-budget can protect them.
            if self.utility_gated_formulas:
                for f in self.formula_library.get_incubating_formulas()[
                    : self.MAX_NEWBORN_BASIS
                ]:
                    fid = f.get("formula_id")
                    if not fid or fid in seen:
                        continue
                    tree = self.formula_library.get_formula_tree(fid)
                    if tree is None:
                        continue
                    spec.append({
                        "type": "formula",
                        "formula_id": fid,
                        "tree": tree.to_dict() if hasattr(tree, "to_dict") else tree,
                        "newborn": True,
                        "preferred": fid in prefer_ids,
                        "forced": False,
                    })
                    seen.add(fid)
        if self.use_stump_basis:
            qs = np.linspace(0.2, 0.8, self.STUMPS_PER_FEATURE)
            for j in range(n_features):
                for t in np.unique(np.quantile(X[:, j], qs)):
                    spec.append({"type": "stump", "feature": int(j), "threshold": float(t)})
        if self.use_rule_basis:
            for conds in self._extract_rules(X, y_bin, self.MAX_RULES):
                spec.append({"type": "rule", "conditions": conds})
        if self.use_neural_basis:
            spec.extend(self._extract_neural_units(X, y_bin, self.NEURAL_UNITS))
        if self.use_kan_basis:
            spec.extend(self._extract_kan_splines(X, y_bin))
        return spec

    def _extract_kan_splines(
        self, X: np.ndarray, y_bin: np.ndarray,
    ) -> list[dict]:
        """Univariate truncated-power basis (KAN-style, readable).

        For each of the top-K features (by |corr| with y), place knots at
        interior quantiles and emit (x_j - t)_+^degree terms. Each term is
        a named, editable cell; renderable as `(x{j}-{t})_+^{d}`.
        """
        n, f = X.shape
        if f == 0 or n < 8:
            return []
        # Rank features by absolute Pearson with the binary target.
        scores = []
        y = y_bin.astype(float)
        y = y - y.mean()
        for j in range(f):
            col = X[:, j]
            if float(np.std(col)) < 1e-12:
                scores.append((0.0, j))
                continue
            c = float(np.corrcoef(col, y_bin.astype(float))[0, 1])
            scores.append((abs(c) if np.isfinite(c) else 0.0, j))
        scores.sort(reverse=True)
        top = [j for _, j in scores[: max(1, int(self.KAN_FEATURES))]]
        deg = max(1, min(3, int(self.KAN_DEGREE)))
        n_knots = max(2, int(self.KAN_KNOTS))
        qs = np.linspace(0.15, 0.85, n_knots)
        out: list[dict] = []
        for j in top:
            knots = np.unique(np.quantile(X[:, j], qs))
            for t in knots:
                out.append({
                    "type": "kan",
                    "feature": int(j),
                    "knot": float(t),
                    "degree": deg,
                })
        return out

    def _basis_matrix(
        self, spec: list[dict], X: np.ndarray, feature_weights: np.ndarray | None,
    ) -> np.ndarray:
        """Evaluate a basis spec into a design matrix (same order as spec).

        NOTE — feature weights deliberately do NOT scale X here. The obvious
        "fix" (build the basis from _prepare_X(X, w)) is a proven no-op: this
        matrix is z-scored in _fit_sparse_linear, and scaling a column by a
        constant before standardizing cancels exactly. That was measured as
        arm B of the bug #25 experiment -- ablation delta 0.0000, with AND
        without the optimizer's normalization. A dead `X_eff = _prepare_X(...)`
        used to sit on this line and was the reason the dead cognitive channel
        went unnoticed for so long: it *looked* like weights were applied.

        The live route is _basis_weight_vector(), folded into `sd` below, so
        the weights ride INSIDE the standardization as an L1 penalty scale.
        (_prepare_X is still correct and still used on the tree/gradient paths,
        which have no standardization step to cancel it.)
        """
        n = X.shape[0]
        cols = []
        for entry in spec:
            kind = entry.get("type")
            if kind == "raw":
                j = entry["feature"]
                cols.append(X[:, j] if j < X.shape[1] else np.zeros(n))
            elif kind == "stump":
                j = entry["feature"]
                cols.append((X[:, j] > entry["threshold"]).astype(float)
                            if j < X.shape[1] else np.zeros(n))
            elif kind == "rule":
                # Conjunction: 1 where EVERY condition holds, else 0.
                mask = np.ones(n, dtype=float)
                for c in entry.get("conditions", []):
                    j = c["feature"]
                    if j >= X.shape[1]:
                        mask = np.zeros(n); break
                    if c["op"] == ">":
                        mask *= (X[:, j] > c["threshold"]).astype(float)
                    else:
                        mask *= (X[:, j] <= c["threshold"]).astype(float)
                cols.append(mask)
            elif kind == "neural":
                w = np.asarray(entry.get("w", []), dtype=float)
                if w.size != X.shape[1]:
                    cols.append(np.zeros(n))
                else:
                    z = X @ w + float(entry.get("b", 0.0))
                    cols.append(np.maximum(z, 0.0)
                                if entry.get("act", "relu") == "relu"
                                else np.tanh(z))
            elif kind == "kan":
                j = int(entry.get("feature", -1))
                if j < 0 or j >= X.shape[1]:
                    cols.append(np.zeros(n))
                else:
                    deg = max(1, int(entry.get("degree", 2)))
                    v = np.maximum(X[:, j] - float(entry.get("knot", 0.0)), 0.0)
                    cols.append(np.power(v, deg))
            else:  # formula
                try:
                    v = self._raw_from_tree(
                        entry["tree"], X, feature_weights,
                        formula_id=entry.get("formula_id"),
                    )
                    v = np.asarray(v, dtype=float).ravel()
                    if v.shape[0] != n or not np.isfinite(v).all():
                        v = np.zeros(n)
                except Exception:
                    v = np.zeros(n)
                cols.append(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
        return np.column_stack(cols) if cols else np.zeros((n, 1))

    def _basis_weight_vector(
        self, spec: list[dict], feature_weights: np.ndarray | None,
    ) -> np.ndarray:
        """Per-basis-column multiplier derived from the search's feature
        weights — the channel by which the cognitive layer reaches the fit.

        For an L1 model, scaling column j by c is EQUIVALENT to scaling its
        penalty by 1/c. So a down-weighted feature is simply dropped sooner.
        Returns all-ones when disabled, when weights are absent, or when the
        weights are uniform (no information to convey).

        Research step 4: when utility_gated_formulas is on, newborn
        (incubating) formula columns also get NEWBORN_L1_BOOST — the E5
        remedy applied to formulas so one new term is not starved by
        hundreds of entrenched raw/stump columns under a single L1.
        """
        n = len(spec)
        out = np.ones(n)
        if self.utility_gated_formulas:
            boost = float(self.NEWBORN_L1_BOOST)
            for i, e in enumerate(spec):
                if e.get("type") == "formula" and e.get("newborn"):
                    out[i] *= boost
        # Translator-preferred formulas get the same L1 boost family so
        # cognition can actually keep a named term in the fitted law.
        prefer_boost = float(getattr(self, "PREFER_L1_BOOST", 2.5))
        for i, e in enumerate(spec):
            if e.get("type") == "formula" and e.get("preferred"):
                out[i] *= prefer_boost
        # Direct-install forced terms: large boost if they ever share an
        # L1 with free columns (primary path is the separate stage).
        if self.direct_formula_install:
            forced_boost = float(self.FORCED_L1_BOOST)
            for i, e in enumerate(spec):
                if e.get("type") == "formula" and e.get("forced"):
                    out[i] *= forced_boost
        if not self.use_feature_weight_penalty or feature_weights is None:
            return out
        w = coerce_float_array(feature_weights)
        if w is None:
            return out
        w = np.asarray(w, dtype=float)
        if w.size == 0 or float(np.max(w) - np.min(w)) < 1e-9:
            return out          # uniform => carries no signal
        raw = safe_minmax(w)
        mult = self.WEIGHT_FLOOR + (1.0 - self.WEIGHT_FLOOR) * raw
        gt = getattr(self, "_guard_telemetry", None)
        if gt is not None and raw.size:
            # Floor binds when a feature was driven to 0 by minmax.
            gt.observe(
                "feature_weight_floor",
                float(np.min(raw)),
                float(np.min(mult)),
            )
        for i, e in enumerate(spec):
            kind = e.get("type")
            if kind in ("raw", "stump", "kan"):
                j = int(e.get("feature", -1))
                if 0 <= j < mult.size:
                    out[i] *= float(mult[j])
            elif kind == "rule":
                js = [int(c["feature"]) for c in e.get("conditions", [])
                      if 0 <= int(c.get("feature", -1)) < mult.size]
                if js:
                    out[i] *= float(np.mean([mult[j] for j in js]))
            elif kind == "neural":
                # A unit reads every feature, so weight it by how much of
                # its mass sits on features the search likes: |w|-weighted
                # mean of the multipliers. A unit built mostly on
                # deprioritised inputs is penalised more.
                ww = np.abs(np.asarray(e.get("w", []), dtype=float))
                if ww.size == mult.size and float(ww.sum()) > 1e-12:
                    out[i] *= float((ww * mult).sum() / ww.sum())
            # formula terms span many features -> keep newborn boost only
        return out

    def utility_test_candidate(
        self,
        tree,
        X: np.ndarray,
        y: np.ndarray,
        feature_weights: np.ndarray | None = None,
    ) -> tuple[bool, float]:
        """DreamCoder-style admission test (research step 4).

        Held-out: does adding this formula term to the sparse_linear basis
        improve accuracy vs the current active basis alone? Deterministic
        given the adapter seed. Returns (admit, delta) where delta is
        held-out acc_with − acc_without. Callers should only mint the
        formula when admit is True.

        Cheap: one train/val split, two L1 fits over the existing basis
        ± one column. Does not mutate the library.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[0] < 20:
            return False, 0.0
        classes = np.unique(y)
        if classes.size != 2:
            # Sparse_linear utility test is binary; multiclass OvR would
            # need per-head tests — leave that for a later gate.
            return False, 0.0
        pos = int(classes.max())
        y_bin = (y == pos).astype(int)
        if len(np.unique(y_bin)) < 2:
            return False, 0.0
        try:
            Xtr, Xva, ytr, yva = train_test_split(
                X, y_bin, test_size=0.25,
                random_state=(self._sk_random_state() + 901) & 0x7FFFFFFF,
                stratify=y_bin,
            )
        except ValueError:
            Xtr, Xva, ytr, yva = train_test_split(
                X, y_bin, test_size=0.25,
                random_state=(self._sk_random_state() + 901) & 0x7FFFFFFF,
            )
        n_features = X.shape[1]
        # Snapshot flag so _build_basis_spec does not pull other incubating
        # formulas into this probe (isolate the candidate under test).
        prev_gate = self.utility_gated_formulas
        self.utility_gated_formulas = False
        try:
            base_spec = self._build_basis_spec(Xtr, ytr, n_features)
        finally:
            self.utility_gated_formulas = prev_gate
        tree_dict = tree.to_dict() if hasattr(tree, "to_dict") else tree
        cand = {
            "type": "formula",
            "formula_id": "__utility_probe__",
            "tree": tree_dict,
            "newborn": True,
        }

        def _score(spec):
            if not spec:
                return 0.0
            Btr = self._basis_matrix(spec, Xtr, feature_weights)
            Bva = self._basis_matrix(spec, Xva, feature_weights)
            mu = Btr.mean(axis=0)
            sd = Btr.std(axis=0)
            sd = np.where(sd > 1e-9, sd, 1.0)
            wcol = self._basis_weight_vector(spec, feature_weights)
            sd = sd / np.where(wcol > 1e-9, wcol, 1.0)
            Ztr = (Btr - mu) / sd
            Zva = (Bva - mu) / sd
            if len(np.unique(ytr)) < 2:
                return float(np.mean(yva == ytr[0]))
            lr = LogisticRegression(
                penalty="l1", solver="liblinear", C=self.sparse_l1_c,
                max_iter=2000,
                random_state=(self._sk_random_state() + 902) & 0x7FFFFFFF,
            )
            lr.fit(Ztr, ytr)
            pred = lr.predict(Zva)
            return float(accuracy_score(yva, pred))

        # Force newborn boost for the probe column regardless of flag.
        prev_gate = self.utility_gated_formulas
        self.utility_gated_formulas = True
        try:
            acc_without = _score(base_spec)
            acc_with = _score(base_spec + [cand])
        finally:
            self.utility_gated_formulas = prev_gate
        delta = acc_with - acc_without
        return bool(delta > self.UTILITY_ADMIT_EPS), float(delta)

    def _ensure_residual_installed(
        self, X: np.ndarray, y_bin: np.ndarray,
        baseline_scores: np.ndarray | None = None,
    ) -> list[str]:
        """Step 1 — grow formulas on this dataset's residual/target.

        Trees live on the adapter for this fit (``_forced_residual_trees``)
        so candidate evaluations do not flood the formula library. A short
        list of formula_ids is minted once into the library for HITL /
        provenance when any term survives install.
        """
        from figpso_uil.cognition.residual_formula_install import grow_residual_terms

        if not self.direct_formula_install:
            return []
        terms = grow_residual_terms(
            X, y_bin,
            seed=self._sk_random_state() ^ 0x51D0,
            n_terms=int(self.RESIDUAL_INSTALL_TERMS),
            baseline_scores=baseline_scores,
            method="trial",  # offline mutate→score→keep; never API
            n_trials=int(getattr(self, "RESIDUAL_OFFLINE_TRIALS", 48)),
        )
        self._forced_residual_trees = [t.tree for t in terms]
        # Mint / refresh library ids only when empty or count changed —
        # reuse ids across candidates in the same process when possible.
        if len(self._residual_install_ids) != len(terms):
            ids: list[str] = []
            for i, term in enumerate(terms):
                fid = self.formula_library.add_formula(
                    name=f"residual_install_{i}",
                    tree=term.tree,
                    origin="residual_install",
                    mutation_type="residual_guided",
                    status="active",
                )
                ids.append(fid)
            self._residual_install_ids = ids
        else:
            # Refresh trees in place so residual tracks the free model.
            for fid, term in zip(self._residual_install_ids, terms):
                entry = self.formula_library._formulas.get(fid)
                if entry is not None:
                    entry["tree"] = term.tree.to_dict()
                    entry["origin"] = "residual_install"
        return list(self._residual_install_ids)

    def _l1_fit_spec(
        self,
        spec: list[dict],
        X: np.ndarray,
        y_bin: np.ndarray,
        feature_weights: np.ndarray | None,
        pos_class: int | None,
        C: float | None = None,
    ) -> dict:
        """Shared L1 fit over a basis spec (used by free + classic paths)."""
        from sklearn.linear_model import LogisticRegression

        if not spec:
            return {
                "kind": "sparse_linear", "basis": [], "coef": [],
                "mu": [], "sd": [], "intercept": 0.0,
                "positive_class": pos_class or 1,
            }
        B = self._basis_matrix(spec, X, feature_weights)
        mu = B.mean(axis=0)
        sd = B.std(axis=0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        wcol = self._basis_weight_vector(spec, feature_weights)
        sd = sd / np.where(wcol > 1e-9, wcol, 1.0)
        Bz = (B - mu) / sd
        if len(np.unique(y_bin)) < 2:
            return {
                "kind": "sparse_linear", "basis": [], "coef": [],
                "mu": [], "sd": [], "intercept": 0.0,
                "threshold": 0.5, "positive_class": pos_class or 1,
            }
        lr = LogisticRegression(
            penalty="l1", solver="liblinear",
            C=float(self.sparse_l1_c if C is None else C),
            max_iter=3000,
            random_state=self._sk_random_state(),
        )
        lr.fit(Bz, y_bin)
        coef = lr.coef_.ravel()
        keep = np.where(np.abs(coef) > 1e-8)[0]
        if keep.size == 0:
            keep = np.array([int(np.argmax(np.abs(coef)))])
        return {
            "kind": "sparse_linear",
            "basis": [spec[i] for i in keep],
            "coef": [float(coef[i]) for i in keep],
            "mu": [float(mu[i]) for i in keep],
            "sd": [float(sd[i]) for i in keep],
            "intercept": float(lr.intercept_[0]),
            "positive_class": pos_class or 1,
        }

    def _fit_sparse_linear_direct(
        self,
        X: np.ndarray,
        y_bin: np.ndarray,
        feature_weights: np.ndarray | None,
        n_features: int,
        pos_class: int | None,
    ) -> dict:
        """Steps 1–2 of direct influence: residual grow + forced install.

        Free families (raw/stump/rule/…) fit under L1 alone. Symbolic terms
        grown on the residual are then added with least-squares coefficients
        in score space — they never compete with 200 stumps for one L1
        budget. Predictions change whenever a residual term has nonzero gain.
        """
        from figpso_uil.cognition.symbolic_regressor import _ls
        from figpso_uil.cognition.residual_formula_install import (
            rank_formulas_by_residual,
        )

        # Free stage first (no formulas) so residual is well-defined.
        prev_sym = self.use_symbolic_basis
        self.use_symbolic_basis = False
        try:
            free_spec = self._build_basis_spec(X, y_bin, n_features)
        finally:
            self.use_symbolic_basis = prev_sym
        free_model = self._l1_fit_spec(
            free_spec, X, y_bin, feature_weights, pos_class,
        )
        free_scores = self._sparse_linear_scores(free_model, X, feature_weights)

        # Step 1: grow on residual of the free model.
        self._ensure_residual_installed(X, y_bin, baseline_scores=free_scores)

        # Full spec with forced tags; pick forced formula entries.
        full_spec = self._build_basis_spec(X, y_bin, n_features)
        forced_entries = [
            e for e in full_spec
            if e.get("type") == "formula" and e.get("forced")
        ]
        if not forced_entries:
            forced_entries = [
                e for e in full_spec
                if e.get("type") == "formula"
                and e.get("formula_id") in set(self._residual_install_ids)
            ]
        # Always include in-memory residual trees (authoritative for this fit).
        from figpso_uil.cognition.formula_engine import FormulaNode
        mem_trees = list(getattr(self, "_forced_residual_trees", None) or [])
        trees = []
        seen_fids: set[str] = set()
        for i, tree in enumerate(mem_trees):
            fid = (
                self._residual_install_ids[i]
                if i < len(self._residual_install_ids)
                else f"__residual_{i}"
            )
            node = tree if hasattr(tree, "is_op") else FormulaNode.from_dict(tree)
            trees.append((fid, node))
            seen_fids.add(fid)
        for e in forced_entries:
            fid = e.get("formula_id") or ""
            if fid in seen_fids:
                continue
            tree = e.get("tree")
            if tree is None:
                continue
            node = tree if hasattr(tree, "is_op") else FormulaNode.from_dict(tree)
            trees.append((fid, node))
            seen_fids.add(fid)

        p = 1.0 / (1.0 + np.exp(-np.clip(free_scores, -40.0, 40.0)))
        residual = y_bin.astype(float) - p
        ranked = rank_formulas_by_residual(
            self.formula_engine, trees, X, residual,
            top_k=int(self.MAX_FORCED_FORMULAS),
        )

        # Step 2: successive LS install into score space (dedicated budget).
        F = free_scores.astype(float).copy()
        formula_basis: list[dict] = []
        formula_coef: list[float] = []
        formula_mu: list[float] = []
        formula_sd: list[float] = []
        used_ids: list[str] = []
        for fid, tree, _gain in ranked:
            entry = {
                "type": "formula",
                "formula_id": fid,
                "tree": tree.to_dict() if hasattr(tree, "to_dict") else tree,
                "forced": True,
            }
            col = self._basis_matrix([entry], X, feature_weights)[:, 0]
            mu = float(col.mean())
            sd = float(col.std())
            if sd < 1e-9:
                sd = 1.0
            z = (col - mu) / sd
            target = y_bin.astype(float) - (
                1.0 / (1.0 + np.exp(-np.clip(F, -40.0, 40.0)))
            )
            a, b = _ls(z, target)
            if abs(a) < 1e-8:
                continue
            # Fold intercept tweak into the running score; store a on z.
            F = F + a * z + b
            formula_basis.append(entry)
            formula_coef.append(float(a))
            formula_mu.append(mu)
            formula_sd.append(sd)
            if fid:
                used_ids.append(fid)
            # Absorb b into free intercept below.

        model = {
            "kind": "sparse_linear",
            "basis": list(free_model.get("basis") or []) + formula_basis,
            "coef": list(free_model.get("coef") or []) + formula_coef,
            "mu": list(free_model.get("mu") or []) + formula_mu,
            "sd": list(free_model.get("sd") or []) + formula_sd,
            "intercept": float(free_model.get("intercept") or 0.0),
            "positive_class": pos_class or 1,
            "direct_install": True,
        }
        # Each LS step added `b` into F; fold the mean gap vs stored parts
        # into intercept so predict() matches the fitted scores.
        part_scores = self._sparse_linear_scores(model, X, feature_weights)
        model["intercept"] = float(model["intercept"]) + float(
            np.mean(F - part_scores)
        )

        self._formulas_used_this_fit = list(used_ids)
        if used_ids:
            self._last_used_formula_id = used_ids[-1]
        scores = self._sparse_linear_scores(model, X, feature_weights)
        model["threshold"], _ = self._fit_threshold(scores, y_bin)
        return model

    def _fit_sparse_linear(
        self,
        X: np.ndarray,
        y_bin: np.ndarray,
        feature_weights: np.ndarray | None,
        n_features: int,
        pos_class: int | None,
    ) -> dict:
        if self.direct_formula_install and self.use_symbolic_basis:
            return self._fit_sparse_linear_direct(
                X, y_bin, feature_weights, n_features, pos_class,
            )
        from sklearn.linear_model import LogisticRegression

        spec = self._build_basis_spec(X, y_bin, n_features)
        B = self._basis_matrix(spec, X, feature_weights)
        # Standardize the basis so L1 penalizes terms comparably.
        mu = B.mean(axis=0)
        sd = B.std(axis=0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        # Bug #25: fold the search's feature weights into the EFFECTIVE
        # scale. Dividing by (sd / w) == standardizing then multiplying by
        # w, so a low-weight term shrinks and L1 discards it first. Folding
        # into `sd` (rather than scaling Bz) makes predict() consistent for
        # free — _sparse_linear_scores already divides by the stored sd.
        wcol = self._basis_weight_vector(spec, feature_weights)
        sd = sd / np.where(wcol > 1e-9, wcol, 1.0)
        Bz = (B - mu) / sd
        if len(np.unique(y_bin)) < 2:
            return {"kind": "sparse_linear", "basis": [], "coef": [], "intercept": 0.0,
                    "mu": [], "sd": [], "threshold": 0.5, "positive_class": pos_class or 1}
        # Bug #31: liblinear is the ONE sklearn solver that consumes
        # `random_state` even for a deterministic-looking L1 fit -- it
        # randomly permutes the coordinate order of its descent. With
        # random_state=None sklearn draws that seed from the GLOBAL numpy
        # RNG, which is seeded from OS entropy at import, so it differs in
        # every process. Measured: 8 fits of identical data under 8 global
        # seeds produced 8 DIFFERENT coefficient vectors. That is the whole
        # of the documented multiclass nondeterminism (wine: n_selected
        # 9 vs 8 between two identical runs) -- three OvR heads give the
        # unstable L1 support three chances to flip. Pinning it to the
        # adapter's own seed makes sparse_linear reproducible on every
        # target arity, which every identity test depends on.
        lr = LogisticRegression(penalty="l1", solver="liblinear",
                                C=self.sparse_l1_c, max_iter=3000,
                                random_state=self._sk_random_state())
        lr.fit(Bz, y_bin)
        coef = lr.coef_.ravel()
        keep = np.where(np.abs(coef) > 1e-8)[0]
        if keep.size == 0:  # degenerate: keep the strongest term
            keep = np.array([int(np.argmax(np.abs(coef)))])
        model = {
            "kind": "sparse_linear",
            "basis": [spec[i] for i in keep],
            "coef": [float(coef[i]) for i in keep],
            "mu": [float(mu[i]) for i in keep],
            "sd": [float(sd[i]) for i in keep],
            "intercept": float(lr.intercept_[0]),
            "positive_class": pos_class or 1,
        }
        self._formulas_used_this_fit = [
            e["formula_id"] for e in model["basis"]
            if e.get("type") == "formula" and e.get("formula_id")
        ]
        if self._formulas_used_this_fit:
            self._last_used_formula_id = self._formulas_used_this_fit[-1]
        scores = self._sparse_linear_scores(model, X, feature_weights)
        model["threshold"], _ = self._fit_threshold(scores, y_bin)
        return model

    def _fit_ebm_cyclic(
        self,
        X: np.ndarray,
        y_bin: np.ndarray,
        feature_weights: np.ndarray | None,
        n_features: int,
        pos_class: int | None,
    ) -> dict:
        """EBM-style cyclic boosting over the composed symbolic basis.

        Research step 5 (InterpretML/EBM parity). Same readable basis as
        sparse_linear; fit is round-robin univariate GLM steps with a small
        learning rate instead of one joint L1 solve. Model shape matches
        sparse_linear so predict/readable paths are shared.
        """
        spec = self._build_basis_spec(X, y_bin, n_features)
        if not spec or len(np.unique(y_bin)) < 2:
            return {"kind": "sparse_linear", "basis": [], "coef": [], "intercept": 0.0,
                    "mu": [], "sd": [], "threshold": 0.5,
                    "positive_class": pos_class or 1, "fit": "ebm"}
        B = self._basis_matrix(spec, X, feature_weights)
        mu = B.mean(axis=0)
        sd = B.std(axis=0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        wcol = self._basis_weight_vector(spec, feature_weights)
        sd = sd / np.where(wcol > 1e-9, wcol, 1.0)
        Bz = (B - mu) / sd
        n, m = Bz.shape
        base = float(np.clip(y_bin.mean(), 1e-4, 1.0 - 1e-4))
        intercept = float(np.log(base / (1.0 - base)))
        coef = np.zeros(m, dtype=float)
        F = np.full(n, intercept, dtype=float)
        lr = float(self.ebm_learning_rate)
        rounds = max(1, int(self.ebm_rounds))
        order = list(range(m))  # fixed order — deterministic
        for _ in range(rounds):
            p = 1.0 / (1.0 + np.exp(-np.clip(F, -30.0, 30.0)))
            residual = y_bin.astype(float) - p
            w = np.clip(p * (1.0 - p), 1e-4, None)
            for j in order:
                bj = Bz[:, j]
                denom = float(np.sum(w * bj * bj)) + 1e-8
                step = lr * float(np.sum(w * bj * residual)) / denom
                if abs(step) < 1e-12:
                    continue
                coef[j] += step
                F += step * bj
                p = 1.0 / (1.0 + np.exp(-np.clip(F, -30.0, 30.0)))
                residual = y_bin.astype(float) - p
                w = np.clip(p * (1.0 - p), 1e-4, None)
        keep = np.where(np.abs(coef) > 1e-8)[0]
        if keep.size == 0:
            keep = np.array([int(np.argmax(np.abs(coef)))])
        model = {
            "kind": "sparse_linear",
            "fit": "ebm",
            "basis": [spec[i] for i in keep],
            "coef": [float(coef[i]) for i in keep],
            "mu": [float(mu[i]) for i in keep],
            "sd": [float(sd[i]) for i in keep],
            "intercept": intercept,
            "positive_class": pos_class or 1,
        }
        self._formulas_used_this_fit = [
            e["formula_id"] for e in model["basis"]
            if e.get("type") == "formula" and e.get("formula_id")
        ]
        if self._formulas_used_this_fit:
            self._last_used_formula_id = self._formulas_used_this_fit[-1]
        scores = self._sparse_linear_scores(model, X, feature_weights)
        model["threshold"], _ = self._fit_threshold(scores, y_bin)
        return model

    def _sparse_linear_scores(
        self, model: dict, X: np.ndarray, feature_weights: np.ndarray | None,
    ) -> np.ndarray:
        spec = model.get("basis", [])
        if not spec:
            return np.full(X.shape[0], 0.5)
        B = self._basis_matrix(spec, X, feature_weights)
        mu = np.asarray(model.get("mu", np.zeros(B.shape[1])), dtype=float)
        sd = np.asarray(model.get("sd", np.ones(B.shape[1])), dtype=float)
        sd = np.where(np.abs(sd) > 1e-9, sd, 1.0)
        Bz = (B - mu) / sd
        z = Bz @ np.asarray(model.get("coef", []), dtype=float) + float(model.get("intercept", 0.0))
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def readable_sparse_linear(self, model: dict, top: int | None = 12) -> str:
        """Render the fitted law as a readable weighted sum of named terms.

        Delegates to cognition.law_export, which folds the standardization
        (mu/sd) into raw-space coefficients and prints the fitted threshold
        and the original feature indices.

        This used to print STANDARDIZED coefficients against RAW-space term
        names, with no threshold and no mention of `selected_features`, so a
        reader who evaluated the printed expression got a different function
        than predict() computes -- measured at 3.8% of rows mislabeled on a
        clean 6-term model, and worse whenever `top` silently truncated.
        tests/test_law_fidelity.py::test_f6 pins both halves: the old
        arithmetic is wrong, and what is printed now round-trips exactly.

        Pass top=None for the complete law; truncation is now labelled.
        """
        from figpso_uil.cognition.law_export import (
            LawExportError, export_law, render_law,
        )
        try:
            return render_law(export_law(model), top=top)
        except LawExportError as exc:
            # Rendering must never abort a fit; say so rather than lie.
            return f"<no readable law: {exc}>"

    def _readable_sparse_linear_legacy(self, model: dict, top: int = 12) -> str:
        """The pre-2026-08 unfaithful renderer, kept only so the fidelity
        test can demonstrate the defect it fixed. Do not use."""
        terms = []
        order = np.argsort(-np.abs(np.asarray(model.get("coef", []), dtype=float)))
        for i in order[:top]:
            e = model["basis"][i]; w = model["coef"][i]
            if e["type"] == "raw":
                name = f"x{e['feature']}"
            elif e["type"] == "stump":
                name = f"1[x{e['feature']}>{e['threshold']:.3g}]"
            elif e["type"] == "rule":
                conds = " & ".join(
                    f"x{c['feature']}{c['op']}{c['threshold']:.3g}"
                    for c in e.get("conditions", [])
                )
                name = f"1[{conds}]"
            elif e["type"] == "neural":
                # Not fully legible, but not opaque either: name the unit
                # and the inputs it actually leans on, so a reader can see
                # WHAT it is built from even if not exactly what it computes.
                # NB: local must not be named `w` -- that is the coefficient.
                wv = np.asarray(e.get("w", []), dtype=float)
                top_in = np.argsort(-np.abs(wv))[:3] if wv.size else []
                lean = ",".join(f"x{int(j)}" for j in top_in)
                name = f"relu(u{e.get('unit', '?')}<-{lean})"
            elif e["type"] == "kan":
                name = (
                    f"(x{e['feature']}-{e['knot']:.3g})"
                    f"_+^{int(e.get('degree', 2))}"
                )
            else:
                name = e.get("formula_id", "f(x)")
            terms.append(f"{w:+.3f}*{name}")
        more = len(model.get("coef", [])) - len(terms)
        tail = f" ... (+{more} more terms)" if more > 0 else ""
        return f"sigmoid({' '.join(terms)} {model.get('intercept',0.0):+.3f}){tail}"

    def _fit_binary(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_weights: np.ndarray | None,
        n_features: int,
        mode: str,
        *,
        eval_X: np.ndarray | None = None,
        eval_y: np.ndarray | None = None,
    ) -> dict:
        y_bin, pos_class = self._to_binary_y(y)
        eval_X = X if eval_X is None else eval_X
        eval_y = y if eval_y is None else eval_y
        if mode == "sparse_linear":
            if self.use_ebm_fit:
                return self._fit_ebm_cyclic(
                    X, y_bin, feature_weights, n_features, pos_class,
                )
            return self._fit_sparse_linear(
                X, y_bin, feature_weights, n_features, pos_class,
            )
        if mode == "multi":
            return self._fit_hybrid_ensemble(
                X, y, y_bin, feature_weights, n_features, pos_class,
            )
        if mode == "boosting":
            return self._fit_boosting(
                X, y_bin, feature_weights, n_features, pos_class, variant="residual",
            )
        if mode == "bagging":
            return self._fit_bagging(X, y_bin, feature_weights, n_features, pos_class)
        return self._fit_single(X, y, y_bin, feature_weights, n_features, pos_class)

    def _fit_single(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_bin: np.ndarray,
        feature_weights: np.ndarray | None,
        n_features: int,
        pos_class: int | None,
    ) -> dict:
        chosen = self._pick_formula(feature_weights, n_features, X, y)
        self._last_used_formula_id = chosen["formula_id"]
        tree = self._tree_or_seed(chosen["formula_id"], n_features)
        scores = self._scores_from_tree(tree, X, feature_weights)
        threshold, _ = self._fit_threshold(scores, y_bin)
        return {
            "kind": "single",
            "formula_id": chosen["formula_id"],
            "tree": tree.to_dict() if hasattr(tree, "to_dict") else tree,
            "threshold": threshold,
            "positive_class": pos_class or 1,
        }

    def _fit_boosting(
        self,
        X: np.ndarray,
        y_bin: np.ndarray,
        feature_weights: np.ndarray | None,
        n_features: int,
        pos_class: int | None,
        *,
        variant: str = "residual",
        boosting_stages: int | None = None,
    ) -> dict:
        """Sequential formula boosting — residual, gradient, or Newton-style."""
        n = len(y_bin)
        F = np.zeros(n, dtype=float)
        stages: list[dict] = []
        y_float = y_bin.astype(float)
        n_stages = boosting_stages if boosting_stages is not None else self.boosting_stages

        X_eff = self._prepare_X(X, feature_weights)
        for _ in range(n_stages):
            scores = 1.0 / (1.0 + np.exp(-np.clip(F, -20, 20)))
            target = self._boosting_target(scores, y_float, variant)
            if np.std(target) < 1e-9:
                break

            chosen = self._pick_formula(
                feature_weights, n_features, X, y_bin, residual=target,
            )
            tree = self._tree_or_seed(chosen["formula_id"], n_features)
            # grad_fit (symbolic_gradient.py): refit the formula's constants
            # against the pseudo-residual, normalize, least-squares scale,
            # Newton line-search the step. The legacy constant-step
            # `learning_rate * (score - 0.5) * 4.0` produced near-constant
            # logits on real data — every prediction collapsed to the
            # majority class.
            fitted_tree, gf_params, contribution = fit_boost_stage(
                self.formula_engine, tree, X_eff, F, y_float,
                self.learning_rate, 3, self._rng,
            )
            F += contribution
            stages.append({
                "formula_id": chosen["formula_id"],
                "tree": fitted_tree.to_dict()
                if hasattr(fitted_tree, "to_dict") else fitted_tree,
                "variant": variant,
                "grad_fit": gf_params,
            })

        if stages:
            self._last_used_formula_id = stages[-1]["formula_id"]

        final_scores = 1.0 / (1.0 + np.exp(-np.clip(F, -20, 20)))
        threshold, _ = self._fit_threshold(final_scores, y_bin)
        return {
            "kind": "boosting",
            "boost_variant": variant,
            "stages": stages,
            "threshold": threshold,
            "positive_class": pos_class or 1,
        }

    @staticmethod
    def _boosting_target(
        scores: np.ndarray, y_float: np.ndarray, variant: str,
    ) -> np.ndarray:
        residual = y_float - scores
        if variant == "gradient":
            w = scores * (1.0 - scores)
            return residual * np.clip(w, 1e-6, 0.25)
        if variant == "newton":
            w = scores * (1.0 - scores)
            return residual / np.clip(w, 1e-3, 0.25)
        return residual

    def _branch_budgets(self) -> tuple[int, int, int]:
        """Split global budget across hybrid branches."""
        boost = max(2, self.boosting_stages // 3)
        bag = max(2, self.bagging_estimators // 3)
        mini_boost = max(2, min(4, self.boosting_stages // 5))
        return boost, bag, mini_boost

    def _fit_hybrid_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_bin: np.ndarray,
        feature_weights: np.ndarray | None,
        n_features: int,
        pos_class: int | None,
    ) -> dict:
        """Train distinct learning paths and stack them — not pick-one."""
        boost_stages, bag_est, mini_boost = self._branch_budgets()
        n_branches = max(3, min(len(self.HYBRID_BRANCHES), self.hybrid_branch_count))
        branch_names = list(self.HYBRID_BRANCHES[:n_branches])

        X_tr, y_tr, X_val, y_val = self._eval_holdout_split(X, y)
        y_val_bin, _ = self._to_binary_y(y_val)

        branch_workers = 1
        if self._branch_parallel and len(branch_names) > 1:
            branch_workers = max(
                1, min(self.parallel_eval_workers, len(branch_names)),
            )

        # Bug #31 (second axis): branch results must not depend on HOW MANY
        # threads the machine decided to give us. Serially, every branch drew
        # from one shared, advancing `self._rng`; in parallel each branch got
        # a clone whose `_rng` restarted at `seed` -- so the same run on a
        # 2-core and an 8-core box produced different models, and worse, all
        # parallel branches drew the IDENTICAL bootstrap stream (the branches
        # exist precisely to differ). Deriving each branch's stream from
        # (seed, branch name) makes serial and parallel byte-identical AND
        # restores the between-branch diversity the ensemble is built on.
        branch_seeds = {
            name: np.random.default_rng(
                [self._sk_random_state(), i, 0x5F0]
            )
            for i, name in enumerate(branch_names)
        }

        def _fit_branch(name: str) -> tuple[str, dict]:
            worker = (
                self.clone_for_parallel_eval(branch_parallel=False, parallel_workers=1)
                if branch_workers > 1
                else self
            )
            saved_rng = worker._rng
            worker._rng = branch_seeds[name]
            try:
                return _fit_branch_body(worker, name)
            finally:
                worker._rng = saved_rng

        def _fit_branch_body(worker, name: str) -> tuple[str, dict]:
            if name == "single":
                branch = worker._fit_single(
                    X, y, y_bin, feature_weights, n_features, pos_class,
                )
            elif name == "boosting":
                branch = worker._fit_boosting(
                    X, y_bin, feature_weights, n_features, pos_class,
                    variant="residual", boosting_stages=boost_stages,
                )
            elif name == "gradient":
                branch = worker._fit_boosting(
                    X, y_bin, feature_weights, n_features, pos_class,
                    variant="gradient", boosting_stages=boost_stages,
                )
            elif name == "bagging":
                branch = worker._fit_bagging(
                    X, y_bin, feature_weights, n_features, pos_class,
                    bagging_estimators=bag_est,
                )
            elif name == "boosted_bag":
                branch = worker._fit_boosted_bagging(
                    X, y_bin, feature_weights, n_features, pos_class,
                    bag_estimators=bag_est, boost_stages=mini_boost,
                )
            else:
                raise ValueError(f"unknown hybrid branch: {name}")
            branch["feature_weights"] = feature_weights
            return name, branch

        branches: dict[str, dict] = {}
        branch_val_scores: dict[str, np.ndarray] = {}

        blas_slots = branch_workers if branch_workers > 1 else 1

        if branch_workers > 1:
            def _fit_branch_threaded(name: str) -> tuple[str, dict]:
                with blas_thread_scope(
                    max(1, (os.cpu_count() or 4) // blas_slots),
                ):
                    return _fit_branch(name)

            with ThreadPoolExecutor(max_workers=branch_workers) as pool:
                for name, branch in pool.map(_fit_branch_threaded, branch_names):
                    branches[name] = branch
                    if len(X_val) > 0:
                        branch_val_scores[name] = self._predict_binary_scores(
                            branch, X_val,
                        )
        else:
            for name in branch_names:
                name, branch = _fit_branch(name)
                branches[name] = branch
                if len(X_val) > 0:
                    branch_val_scores[name] = self._predict_binary_scores(
                        branch, X_val,
                    )

        blend_weights = self._optimize_hybrid_blend(branch_val_scores, y_val_bin)
        blend_vec = np.array([blend_weights.get(n, 0.0) for n in branch_names], dtype=float)
        if blend_vec.sum() < 1e-9:
            blend_vec = np.ones(len(branch_names)) / len(branch_names)
        else:
            blend_vec /= blend_vec.sum()

        train_scores = self._blend_branch_scores(branches, branch_names, blend_vec, X)
        threshold, _ = self._fit_threshold(train_scores, y_bin)

        self._last_tournament = [
            {
                "formula_id": f"hybrid_{name}",
                "name": name,
                "score": round(float(blend_weights.get(name, 0.0)), 4),
            }
            for name in branch_names
        ]
        all_ids = []
        for b in branches.values():
            all_ids.extend(self._formula_ids_from_branch(b))
        if all_ids:
            self._last_used_formula_id = all_ids[-1]
        for fid in all_ids:
            if fid and fid not in self._formulas_used_this_fit:
                self._formulas_used_this_fit.append(fid)

        self._last_hybrid_blend = {
            n: float(blend_weights.get(n, 0.0)) for n in branch_names
        }

        return {
            "kind": "hybrid",
            "branch_order": branch_names,
            "branches": branches,
            "blend_weights": {n: float(blend_weights.get(n, 0.0)) for n in branch_names},
            "threshold": threshold,
            "positive_class": pos_class or 1,
        }

    def _fit_boosted_bagging(
        self,
        X: np.ndarray,
        y_bin: np.ndarray,
        feature_weights: np.ndarray | None,
        n_features: int,
        pos_class: int | None,
        *,
        bag_estimators: int,
        boost_stages: int,
    ) -> dict:
        """Bagging where each bootstrap estimator is a mini gradient boost."""
        n = len(y_bin)
        estimators: list[dict] = []
        saved = self.boosting_stages

        for _ in range(bag_estimators):
            idx = self._rng.integers(0, n, size=n)
            self.boosting_stages = boost_stages
            mini = self._fit_boosting(
                X[idx], y_bin[idx], feature_weights, n_features, pos_class,
                variant="gradient",
            )
            estimators.append({
                "formula_id": mini.get("stages", [{}])[-1].get("formula_id", "boosted_bag"),
                "submodel": mini,
            })

        self.boosting_stages = saved
        mean_scores = np.mean(
            np.column_stack([
                self._predict_binary_scores(e["submodel"], X) for e in estimators
            ]),
            axis=1,
        )
        threshold, _ = self._fit_threshold(mean_scores, y_bin)
        return {
            "kind": "boosted_bag",
            "estimators": estimators,
            "threshold": threshold,
            "positive_class": pos_class or 1,
        }

    def _blend_branch_scores(
        self,
        branches: dict[str, dict],
        branch_names: list[str],
        weights: np.ndarray,
        X: np.ndarray,
    ) -> np.ndarray:
        mats = [
            self._predict_binary_scores(branches[name], X)
            for name in branch_names
        ]
        return np.clip(np.column_stack(mats) @ weights, 0.0, 1.0)

    def _optimize_hybrid_blend(
        self,
        branch_val_scores: dict[str, np.ndarray],
        y_val_bin: np.ndarray,
    ) -> dict[str, float]:
        if not branch_val_scores or len(y_val_bin) == 0:
            n = max(1, len(branch_val_scores))
            return {k: 1.0 / n for k in branch_val_scores}

        names = list(branch_val_scores.keys())
        Z = np.column_stack([branch_val_scores[n] for n in names])
        best_w = np.ones(len(names)) / len(names)
        best_acc = -1.0

        for _ in range(120):
            w = self._rng.dirichlet(np.ones(len(names)))
            blended = np.clip(Z @ w, 0.0, 1.0)
            threshold, acc = self._fit_threshold(blended, y_val_bin)
            if acc > best_acc:
                best_acc = acc
                best_w = w

        # Coordinate refinement from best draw
        for _ in range(len(names) * 8):
            w = best_w.copy()
            i, j = self._rng.integers(0, len(names), size=2)
            delta = self._rng.uniform(0.02, 0.15)
            if w[i] > delta:
                w[i] -= delta
                w[j] += delta
            w = w / w.sum()
            blended = np.clip(Z @ w, 0.0, 1.0)
            _, acc = self._fit_threshold(blended, y_val_bin)
            if acc > best_acc:
                best_acc = acc
                best_w = w

        return {names[i]: float(best_w[i]) for i in range(len(names))}

    @staticmethod
    def _formula_ids_from_branch(branch: dict) -> list[str]:
        kind = branch.get("kind")
        if kind == "single":
            fid = branch.get("formula_id")
            return [fid] if fid else []
        if kind in ("boosting",):
            return [s["formula_id"] for s in branch.get("stages", []) if s.get("formula_id")]
        if kind == "bagging":
            return [e["formula_id"] for e in branch.get("estimators", []) if e.get("formula_id")]
        if kind == "boosted_bag":
            ids = []
            for e in branch.get("estimators", []):
                sub = e.get("submodel", {})
                ids.extend(
                    s["formula_id"] for s in sub.get("stages", []) if s.get("formula_id")
                )
            return ids
        return []

    def _fit_bagging(
        self,
        X: np.ndarray,
        y_bin: np.ndarray,
        feature_weights: np.ndarray | None,
        n_features: int,
        pos_class: int | None,
        *,
        bagging_estimators: int | None = None,
    ) -> dict:
        """Bootstrap-average multiple formulas (Random Forest spirit)."""
        n = len(y_bin)
        estimators: list[dict] = []
        n_est = bagging_estimators if bagging_estimators is not None else self.bagging_estimators

        for _ in range(n_est):
            idx = self._rng.integers(0, n, size=n)
            chosen = self._pick_formula(feature_weights, n_features, X[idx], y_bin[idx])
            tree = self._tree_or_seed(chosen["formula_id"], n_features)
            scores_boot = self._scores_from_tree(tree, X[idx], feature_weights)
            t_boot, _ = self._fit_threshold(scores_boot, y_bin[idx])
            estimators.append({
                "formula_id": chosen["formula_id"],
                "tree": tree.to_dict(),
                "threshold": t_boot,
            })

        if estimators:
            self._last_used_formula_id = estimators[-1]["formula_id"]

        all_scores = np.column_stack([
            self._scores_from_tree(e["tree"], X, feature_weights) for e in estimators
        ])
        mean_scores = np.mean(all_scores, axis=1)
        threshold, _ = self._fit_threshold(mean_scores, y_bin)
        return {
            "kind": "bagging",
            "estimators": estimators,
            "threshold": threshold,
            "positive_class": pos_class or 1,
        }

    def _fit_multiclass_ovr(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_weights: np.ndarray | None,
        n_features: int,
        mode: str,
        *,
        eval_X: np.ndarray | None = None,
        eval_y: np.ndarray | None = None,
    ) -> dict:
        classes = sorted(int(c) for c in np.unique(y))
        classifiers: dict[int, dict] = {}
        eval_X = X if eval_X is None else eval_X
        eval_y = y if eval_y is None else eval_y
        for c in classes:
            y_ovr = (y == c).astype(int)
            eval_ovr = (eval_y == c).astype(int)
            if mode == "sparse_linear":
                # One L1 model per class over the composed symbolic basis
                # == multinomial-style logistic over raw+formula+stump.
                # Far stronger than a single symbolic formula per head on
                # many-class problems (digits: single-formula heads top out
                # ~0.5; sparse_linear heads reach linear-model territory).
                clf = self._fit_sparse_linear(
                    X, y_ovr, feature_weights, n_features, c,
                )
            elif mode == "multi":
                clf = self._fit_hybrid_ensemble(
                    X, y, y_ovr, feature_weights, n_features, c,
                )
            elif mode == "boosting":
                clf = self._fit_boosting(
                    X, y_ovr, feature_weights, n_features, c, variant="residual",
                )
            elif mode == "bagging":
                clf = self._fit_bagging(X, y_ovr, feature_weights, n_features, c)
            else:
                clf = self._fit_single(X, y, y_ovr, feature_weights, n_features, c)
            # Platt-calibrate this head's raw score -> P(class==c) on the
            # eval set. Each one-vs-rest head is a DIFFERENT binary model
            # on a DIFFERENT (1-vs-rest, heavily imbalanced) problem, and
            # different `kind`s emit scores on different scales, so raw
            # scores are NOT comparable across heads. Without calibration
            # the multiclass argmax collapses onto a few hot heads (digits
            # emitted 3/10 classes -> 0.21 acc; MNIST 4/10). None when a
            # head can't be calibrated -> predict falls back to raw scores.
            cal = self._fit_platt(
                self._predict_binary_scores(clf, eval_X), eval_ovr,
            )
            if cal is not None:
                clf["ovr_calibration"] = cal
            classifiers[c] = clf
        self._last_used_formula_id = self._primary_formula_id(classifiers[classes[0]])
        return {"kind": "multiclass_ovr", "learning_mode": mode, "classifiers": classifiers}

    @staticmethod
    def _fit_platt(scores: np.ndarray, labels: np.ndarray) -> dict | None:
        """Fit sigmoid(a*s + b): map a one-vs-rest head's raw score to
        P(class==c), so OvR heads become comparable for argmax.

        Returns None when the head can't be calibrated (single-class eval
        labels or a constant score); predict() then falls back to the raw
        score (pre-fix behavior), so old persisted models still work.
        """
        s = np.asarray(scores, dtype=float).ravel()
        lab = np.asarray(labels, dtype=int).ravel()
        if s.size == 0 or lab.size != s.size or len(np.unique(lab)) < 2:
            return None
        if float(np.ptp(s)) < 1e-12:
            return None
        try:
            from sklearn.linear_model import LogisticRegression
            # lbfgs does not consume random_state, but pinning it anyway
            # keeps the bug #31 invariant ("every sklearn estimator gets
            # an explicit seed") checkable by grep rather than by knowing
            # which solvers are secretly stochastic.
            lr = LogisticRegression(
                solver="lbfgs", max_iter=1000, random_state=0,
            )
            lr.fit(s.reshape(-1, 1), lab)
            return {"a": float(lr.coef_.ravel()[0]), "b": float(lr.intercept_[0])}
        except Exception:
            return None

    @staticmethod
    def _apply_platt(scores: np.ndarray, cal: dict | None) -> np.ndarray:
        """Apply a stored Platt calibration; identity when cal is None."""
        s = np.asarray(scores, dtype=float).ravel()
        if not cal:
            return s
        z = float(cal.get("a", 1.0)) * s + float(cal.get("b", 0.0))
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _primary_formula_id(self, clf: dict) -> str | None:
        if clf.get("kind") == "hybrid":
            for name in clf.get("branch_order", []):
                sub = clf.get("branches", {}).get(name, {})
                fid = self._primary_formula_id(sub)
                if fid:
                    return fid
            return None
        if clf.get("formula_id"):
            return clf["formula_id"]
        stages = clf.get("stages") or []
        if stages:
            return stages[-1].get("formula_id")
        est = clf.get("estimators") or []
        if est:
            return est[-1].get("formula_id")
        return None

    def _pick_formula(
        self,
        feature_weights: np.ndarray | None,
        n_features: int,
        X: np.ndarray,
        y: np.ndarray,
        *,
        residual: np.ndarray | None = None,
    ) -> dict:
        if self._episodic_suggestion is not None and not self._used_episodic_suggestion:
            self._used_episodic_suggestion = True
            for f in self.formula_library.get_all():
                if f["formula_id"] == self._episodic_suggestion:
                    self._formulas_used_this_fit.append(f["formula_id"])
                    return f

        chosen = self._select_formula(
            feature_weights, n_features, X, y, residual=residual,
        )
        if chosen is not None:
            self._formulas_used_this_fit.append(chosen["formula_id"])
            return chosen

        seeds = self.formula_engine.get_seed_classifier_formulas(n_features)
        self.formula_library.seed_formulas(seeds)
        fallback = self.formula_library.get_best_formula(dataset_fingerprint=None)
        if fallback is None:
            raise RuntimeError("SymbolicFormula library empty after re-seed")
        self._formulas_used_this_fit.append(fallback["formula_id"])
        return fallback

    def _predict_binary_scores(self, model: dict, X: np.ndarray) -> np.ndarray:
        self._repair_model_trees(model)
        fw = model.get("feature_weights")
        if fw is not None:
            fw = coerce_float_array(fw)
        kind = model.get("kind")

        if kind == "hybrid":
            names = model.get("branch_order", [])
            branches = model.get("branches", {})
            weights = model.get("blend_weights", {})
            w = np.array([weights.get(n, 0.0) for n in names], dtype=float)
            if w.sum() < 1e-9:
                w = np.ones(len(names)) / max(1, len(names))
            else:
                w /= w.sum()
            return self._blend_branch_scores(branches, names, w, X)

        if kind == "sparse_linear":
            return self._sparse_linear_scores(model, X, fw)

        if kind == "boosting":
            F = np.zeros(X.shape[0], dtype=float)
            X_eff = self._prepare_X(X, fw)
            for stage in model.get("stages", []):
                gf_params = stage.get("grad_fit")
                if gf_params is not None:
                    F += stage_contribution(
                        self.formula_engine,
                        self._coerce_tree(
                            stage["tree"], formula_id=stage.get("formula_id"),
                        ),
                        X_eff, gf_params,
                    )
                else:
                    # Legacy pre-grad_fit persisted models
                    h = self._scores_from_tree(stage["tree"], X, fw) - 0.5
                    step = self.learning_rate
                    if stage.get("variant") == "newton":
                        step *= 0.85
                    F += step * h * 4.0
            return 1.0 / (1.0 + np.exp(-np.clip(F, -20, 20)))

        if kind == "boosted_bag":
            mats = [
                self._predict_binary_scores(e["submodel"], X)
                for e in model.get("estimators", [])
            ]
            if not mats:
                return np.full(X.shape[0], 0.5)
            return np.mean(np.column_stack(mats), axis=1)

        if kind == "bagging":
            mats = [
                self._scores_from_tree(
                    e["tree"], X, fw, formula_id=e.get("formula_id"),
                )
                for e in model.get("estimators", [])
            ]
            if not mats:
                return np.full(X.shape[0], 0.5)
            return np.mean(np.column_stack(mats), axis=1)

        return self._scores_from_tree(model["tree"], X, fw)

    def predict(self, model: object, X: np.ndarray) -> np.ndarray:
        # Re-apply the same feature preselection used at fit time, so
        # formula feature indices line up with the reduced basis.
        sel = model.get("selected_features")
        if sel is not None:
            X = np.asarray(X, dtype=float)[:, np.asarray(sel, dtype=int)]
        if model.get("kind") == "multiclass_ovr":
            # Sort NUMERICALLY and emit numeric labels even if the model was
            # loaded from JSON with stringified keys and never passed through
            # _repair_model_trees (bug #28). Plain sorted() on strings also
            # misorders >= 10 classes ('10' < '2'), which would mislabel every
            # prediction on digits/MNIST.
            classes = sorted(model["classifiers"].keys(), key=lambda c: float(c))
            class_labels = np.array([int(float(c)) for c in classes])
            # Calibrate each head's score to P(class==c) before argmax so
            # every class is reachable (fixes the OvR collapse). Heads
            # without a stored calibration fall back to raw scores.
            score_mat = np.column_stack([
                self._apply_platt(
                    self._predict_binary_scores(model["classifiers"][c], X),
                    model["classifiers"][c].get("ovr_calibration"),
                )
                for c in classes
            ])
            return class_labels[np.argmax(score_mat, axis=1)]

        scores = self._predict_binary_scores(model, X)
        preds = (scores >= model["threshold"]).astype(int)
        pos = model.get("positive_class", 1)
        if pos != 1:
            return np.where(preds == 1, pos, 0)
        return preds

    @staticmethod
    def _feature_usage_counts(tree, n_features: int) -> np.ndarray:
        counts = np.zeros(n_features)
        if isinstance(tree, dict):
            tree = FormulaNode.from_dict(tree)

        def _walk(node) -> None:
            if isinstance(node, dict):
                node = FormulaNode.from_dict(node)
            if node.is_feature and 0 <= node.feature < n_features:
                counts[node.feature] += 1
            for child in node.children:
                _walk(child)

        _walk(tree)
        return counts

    def _trees_from_model(self, model: dict) -> list:
        kind = model.get("kind")
        if kind == "sparse_linear":
            # Only the formula-family basis entries carry trees; raw/stump
            # terms are handled by get_selected_feature_indices directly.
            return [e["tree"] for e in model.get("basis", []) if e.get("type") == "formula"]
        if kind == "hybrid":
            trees = []
            for sub in model.get("branches", {}).values():
                trees.extend(self._trees_from_model(sub))
            return trees
        if kind == "boosted_bag":
            trees = []
            for e in model.get("estimators", []):
                trees.extend(self._trees_from_model(e.get("submodel", {})))
            return trees
        if kind == "boosting":
            return [s["tree"] for s in model.get("stages", [])]
        if kind == "bagging":
            return [e["tree"] for e in model.get("estimators", [])]
        if kind == "multiclass_ovr":
            trees = []
            for clf in model.get("classifiers", {}).values():
                trees.extend(self._trees_from_model(clf))
            return trees
        if "tree" in model:
            return [model["tree"]]
        return []

    def get_selected_feature_indices(self, model: object) -> np.ndarray:
        if model is None:
            return np.array([], dtype=int)
        n_features = int(model.get("n_features") or 10_000)
        indices: set[int] = set()
        for tree in self._trees_from_model(model):
            counts = self._feature_usage_counts(tree, n_features)
            indices.update(int(i) for i in np.where(counts > 0)[0])
        # sparse_linear: raw/stump basis entries name their column directly;
        # a rule references every feature in its conjunction.
        for e in (model.get("basis", []) if isinstance(model, dict) else []):
            if e.get("type") in ("raw", "stump"):
                indices.add(int(e["feature"]))
            elif e.get("type") == "rule":
                for c in e.get("conditions", []):
                    indices.add(int(c["feature"]))
            elif e.get("type") == "neural":
                # A unit technically touches every input, so crediting all
                # of them would make feat_ratio meaningless. Credit only the
                # inputs carrying real mass (>=10% of the unit's max |w|).
                w = np.abs(np.asarray(e.get("w", []), dtype=float))
                if w.size and float(w.max()) > 1e-12:
                    indices.update(
                        int(j) for j in np.where(w >= 0.1 * w.max())[0]
                    )
        # Map reduced-basis indices back to original columns when feature
        # preselection was applied, so callers see true dataset features.
        sel = model.get("selected_features")
        if sel is not None:
            sel = np.asarray(sel, dtype=int)
            return np.array(sorted(int(sel[i]) for i in indices if i < len(sel)), dtype=int)
        return np.array(sorted(indices), dtype=int)

    def get_feature_importance(self, model: object, n_features: int) -> np.ndarray:
        counts = np.zeros(n_features)
        for tree in self._trees_from_model(model):
            counts += self._feature_usage_counts(tree, n_features)
        if counts.sum() == 0:
            fw = model.get("feature_weights")
            if fw is not None and len(fw) == n_features:
                return safe_minmax(fw)
            return np.ones(n_features) / n_features
        return safe_minmax(counts)

    def _eval_formula_accuracy(
        self,
        tree: FormulaNode,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_weights: np.ndarray | None,
    ) -> float:
        y_bin, _ = self._to_binary_y(y_tr)
        y_val_bin, _ = self._to_binary_y(y_val)
        scores_tr = self._scores_from_tree(tree, X_tr, feature_weights)
        threshold, _ = self._fit_threshold(scores_tr, y_bin)
        scores_val = self._scores_from_tree(tree, X_val, feature_weights)
        return float(np.mean((scores_val >= threshold).astype(int) == y_val_bin))

    def _eval_holdout_split(
        self, X: np.ndarray, y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(y)
        if n < 12:
            split = max(1, int(0.7 * n))
            idx = np.arange(n)
            return X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]]

        from sklearn.model_selection import train_test_split
        stratify = y if len(np.unique(y)) == 2 and n >= 20 else None
        tr_idx, val_idx = train_test_split(
            np.arange(n),
            test_size=0.25,
            random_state=int(self.seed + n),
            stratify=stratify,
        )
        return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]

    def _select_formula(
        self,
        feature_weights: np.ndarray | None,
        n_features: int,
        X: np.ndarray,
        y: np.ndarray,
        *,
        residual: np.ndarray | None = None,
    ) -> dict | None:
        pool = [
            f for f in self.formula_library.get_all()
            if f["status"] in ("active", "seed", "incubating")
            and not self._should_skip_explored_formula(f)
        ]
        if not pool:
            return self.formula_library.get_best_formula(dataset_fingerprint=None)

        k = min(self.tournament_k, len(pool))
        progress = min(1.0, sum(f.get("evidence", 0) for f in pool) / max(1, len(pool) * 10))
        candidates = self.formula_library.select_tournament_candidates(
            self._rng, k=k, progress=progress,
        )
        if not candidates:
            candidates = pool[:k]

        X_eval, y_eval, _ = ensure_min_rows(X, y, self._rng)

        if len(X_eval) < 8:
            return self._select_formula_metadata(feature_weights, n_features, candidates)

        X_tr, y_tr, X_val, y_val = self._eval_holdout_split(X_eval, y_eval)
        if len(X_val) == 0:
            return self._select_formula_metadata(feature_weights, n_features, candidates)

        def _score_entry(entry: dict) -> tuple[dict, float, dict]:
            tree = self.formula_library.get_formula_tree(entry["formula_id"])
            if tree is None:
                return entry, -1.0, {}
            if residual is not None and len(residual) == len(X):
                h = self._scores_from_tree(tree, X, feature_weights) - 0.5
                score = abs(float(np.dot(h, residual)))
            else:
                score = self._eval_formula_accuracy(
                    tree, X_tr, y_tr, X_val, y_val, feature_weights,
                )
                score += self._graph_formula_bonus(tree, n_features)
                score += self._exploration_penalty(entry["formula_id"])
            log = {
                "formula_id": entry["formula_id"],
                "name": entry.get("name", entry["formula_id"]),
                "score": round(float(score), 4),
            }
            return entry, score, log

        tournament_log: list[dict] = []
        best_entry: dict | None = None
        best_score = -1.0

        workers = max(1, min(self.parallel_eval_workers, len(candidates)))
        if workers > 1 and len(candidates) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool_exec:
                scored = list(pool_exec.map(_score_entry, candidates))
        else:
            scored = [_score_entry(e) for e in candidates]

        for entry, score, log in scored:
            if log:
                tournament_log.append(log)
            if score > best_score:
                best_score = score
                best_entry = entry

        self._last_tournament = tournament_log
        if best_entry is not None:
            return best_entry
        return self._select_formula_metadata(feature_weights, n_features, candidates)

    def _select_formula_metadata(
        self,
        feature_weights: np.ndarray | None,
        n_features: int,
        candidates: list[dict] | None = None,
    ) -> dict | None:
        pool = candidates or [
            f for f in self.formula_library.get_all()
            if f["status"] in ("active", "seed", "incubating")
        ]
        if not pool:
            return self.formula_library.get_best_formula(dataset_fingerprint=None)

        baseline = self.formula_library.get_best_formula(dataset_fingerprint=None)
        if baseline is None:
            return pool[0]

        norm_weights = None
        if feature_weights is not None and len(feature_weights) == n_features:
            w = np.asarray(feature_weights, dtype=float)
            if np.std(w) >= 1e-9:
                norm_weights = w / (w.sum() + 1e-12)

        CONFIDENCE_BAND = 0.15
        filtered = [
            f for f in pool
            if f["evidence"] < FormulaLibrary.MIN_EVIDENCE_BOOTSTRAP
            or f["confidence"] >= baseline["confidence"] - CONFIDENCE_BAND
        ] or pool

        def score_formula(f: dict) -> float:
            fitness = f.get("confidence", 0.5) + 0.15 * min(1.0, f.get("evidence", 0) / 10)
            fitness += 0.35 * f.get("best_accuracy", 0.0)
            if str(f.get("name", "")).startswith("method_"):
                fitness += 0.05
            tree = self.formula_library.get_formula_tree(f["formula_id"])
            if tree and norm_weights is not None:
                usage = self._feature_usage_counts(tree, n_features)
                if usage.sum() > 0:
                    fitness += 0.25 * float(np.dot(usage / usage.sum(), norm_weights))
            if tree is not None:
                fitness += self._graph_formula_bonus(tree, n_features)
            return fitness

        return max(filtered, key=score_formula)

    def get_model_complexity(self, model: object) -> int:
        # Trees are stored in dict form (via FormulaNode.to_dict()) inside
        # the model, so rehydrate before counting -- same pattern as
        # _feature_usage_counts above.
        if isinstance(model, dict) and model.get("kind") == "sparse_linear":
            # Complexity = number of surviving (readable) terms.
            return len(model.get("coef", []))
        total = 0
        for t in self._trees_from_model(model):
            if isinstance(t, dict):
                t = FormulaNode.from_dict(t)
            total += t.node_count()
        return total

    def get_controllable_params(self) -> dict:
        params = {
            "mutation_rounds_per_fit": {
                "current": self.mutation_rounds_per_fit,
                "min": 0, "max": 3, "type": "int",
            },
            "boosting_stages": {
                "current": self.boosting_stages,
                "min": 1, "max": 50, "type": "int",
            },
            "bagging_estimators": {
                "current": self.bagging_estimators,
                "min": 1, "max": 20, "type": "int",
            },
        }
        # learning_mode is offered for exploration ONLY while it is "auto".
        # An explicit mode is an architectural decision made by the domain
        # router / operator (which basis families suit this fingerprint),
        # not a per-stage hyperparameter for PSO to randomize. Previously
        # the search overrode it every stage from a hardcoded list
        # (principle_guided_search.py) that didn't even contain
        # sparse_linear, so a router-selected mode never survived.
        if self.learning_mode == "auto":
            params["learning_mode"] = {
                "current": self.learning_mode,
                "options": [m for m in self.LEARNING_MODES if m != "auto"],
                "type": "str",
            }
        return params

    def get_architecture_axes(self) -> dict:
        """Axes the ARCHITECTURE RACE may vary (search/architecture_race.py).

        Deliberately SEPARATE from `get_controllable_params`. Those are
        hyperparameters the per-stage PSO search randomizes every stage;
        these are architecture decisions (which basis families exist, how
        hard the L1 selects) that bug #21's invariant forbids letting a
        per-stage search silently reset. The race is different in kind:
        it does not perturb these behind anyone's back, it evaluates each
        one on real held-out accuracy and keeps only a proven winner --
        so exposing them here does not weaken that invariant.

        Advertising them is also the fix for bug #25's ROOT CAUSE: the
        cognitive layer previously had exactly one word in its vocabulary
        (`max_penalty`, i.e. where to cut the feature list). These are the
        rest of the words. The adapter already accepted every one of these
        keys in `apply_config`; nothing ever emitted them.

        Only axes that MATTER for the current learning_mode are returned,
        so the race never spends a fit on an inert knob.
        """
        axes: dict[str, dict] = {}

        if self.learning_mode in ("sparse_linear", "auto"):
            # L1 strength. E5's finding: a large correlated group (rules)
            # competing under one strong L1 gets starved, so the rule axis
            # is worthless unless this one can move with it.
            axes["sparse_l1_c"] = {
                "current": self.sparse_l1_c,
                "values": [0.05, 0.1, 0.5, 1.0],
            }
            # Basis families. Both are known to help on some fingerprints
            # and hurt on others (stumps: spambase +0.021, covertype
            # -0.030; rules: sonar +0.038, ionosphere -0.048), which is
            # precisely why they must be raced per dataset rather than
            # switched on globally -- that global switch was E6, and it
            # regressed.
            axes["use_rule_basis"] = {
                "current": self.use_rule_basis,
                "values": [True, False],
            }
            # Research step 5: cyclic EBM fit vs joint L1 — same basis,
            # different fit algorithm. Raced, never a global switch.
            axes["use_ebm_fit"] = {
                "current": bool(self.use_ebm_fit),
                "values": [True, False],
            }
            axes["use_stump_basis"] = {
                "current": self.use_stump_basis,
                "values": [True, False],
            }
            # Neural units as basis functions: the dense-interaction
            # capacity the additive families provably lack, kept under the
            # same L1 readout so the top-level law stays a readable sum.
            # Raced, never global -- it costs an MLP fit per basis build and
            # trades away per-term legibility, so it must earn its place on
            # each fingerprint (same discipline as stumps and rules).
            axes["use_neural_basis"] = {
                "current": self.use_neural_basis,
                "values": [True, False],
            }
            # Research step 8 — KAN-style univariate splines. Raced, never
            # global (same discipline as neural/rules).
            axes["use_kan_basis"] = {
                "current": bool(self.use_kan_basis),
                "values": [True, False],
            }

        if self.learning_mode in ("boosting", "multi", "auto"):
            axes["boosting_stages"] = {
                "current": self.boosting_stages,
                "values": [5, 10, 25],
            }

        # learning_mode is deliberately NOT a race axis, even when it is
        # still "auto". `apply_config` LOCKS the mode the moment a
        # non-auto value is set, so a LOSING arm evaluated on the main
        # classifier (which happens whenever parallel workers are off)
        # would permanently pin the run's architecture to a config that
        # just lost its race -- a silent, one-way side effect of the kind
        # bug #21 was about. Mode selection stays with the domain router
        # (M3), which owns it by design and records its decision.

        return axes

    def apply_config(self, config: dict) -> None:
        if "mutation_rounds_per_fit" in config:
            self.mutation_rounds_per_fit = int(
                np.clip(config["mutation_rounds_per_fit"], 0, 3)
            )
        if config.get("lock_learning_mode"):
            self._mode_locked = True
        if (
            "learning_mode" in config
            and config["learning_mode"] in self.LEARNING_MODES
            and not self._mode_locked
        ):
            self.learning_mode = config["learning_mode"]
            # An operator/router explicitly choosing a non-auto mode locks
            # it against later PSO/champion overrides; the PSO search only
            # ever proposes a mode while it is still "auto" (see
            # get_controllable_params), so this never fires for exploration.
            if self.learning_mode != "auto":
                self._mode_locked = True
        if "boosting_stages" in config:
            self.boosting_stages = max(1, int(config["boosting_stages"]))
        if "bagging_estimators" in config:
            self.bagging_estimators = max(1, int(config["bagging_estimators"]))
        if "tournament_k" in config:
            self.tournament_k = max(3, int(config["tournament_k"]))
        if "hybrid_branch_count" in config:
            self.hybrid_branch_count = max(3, min(5, int(config["hybrid_branch_count"])))
        elif "multi_boost_variants" in config:
            self.hybrid_branch_count = max(3, min(5, int(config["multi_boost_variants"]) + 2))
        if "parallel_eval_workers" in config:
            self.parallel_eval_workers = max(1, int(config["parallel_eval_workers"]))
        # sparse_linear basis gates (router/fingerprint-controlled)
        for key in ("use_raw_basis", "use_symbolic_basis", "use_stump_basis",
                    "use_rule_basis", "use_neural_basis", "use_kan_basis",
                    "use_feature_weight_penalty", "utility_gated_formulas",
                    "use_ebm_fit", "direct_formula_install"):
            if key in config:
                setattr(self, key, bool(config[key]))
                if key == "direct_formula_install" and not bool(config[key]):
                    self._residual_install_ids = []
        if "sparse_l1_c" in config:
            self.sparse_l1_c = float(np.clip(config["sparse_l1_c"], 0.001, 10.0))
        if "ebm_rounds" in config:
            self.ebm_rounds = max(5, min(200, int(config["ebm_rounds"])))
        if "ebm_learning_rate" in config:
            self.ebm_learning_rate = float(np.clip(
                config["ebm_learning_rate"], 0.001, 0.5))

    def get_active_config_knobs(self) -> set:
        """Config keys that can actually change the fit under the CURRENT mode.

        A mined rule may constrain any `config.*` column the miner saw vary,
        but a knob that is INERT for the active learning mode produces a
        "true but useless" override. This is not hypothetical: the very
        first architecture rule the step-2+3 chain ever minted was
        `config_min.bagging_estimators` under `sparse_linear` (P-011) --
        provably a no-op, since `_fit_sparse_linear` never reads it.
        Reporting the live set lets the actuator drop those instead of
        applying them and journalling a knob that did nothing.

        Deliberately PERMISSIVE where uncertain: `auto`/`multi` resolve to
        (or run) several branches, so everything is reported live there.
        Wrongly calling a knob inert would silently drop a real override --
        strictly worse than allowing a harmless no-op -- so when in doubt
        this includes the knob.
        """
        # Live in every mode: these shape the symbolic basis / formula
        # selection regardless of which fit path runs.
        common = {
            "mutation_rounds_per_fit", "tournament_k",
            "parallel_eval_workers", "learning_mode",
        }
        basis = {
            "sparse_l1_c", "use_raw_basis", "use_symbolic_basis",
            "use_stump_basis", "use_rule_basis", "use_neural_basis",
            "use_kan_basis",
            "use_feature_weight_penalty", "use_ebm_fit",
            "ebm_rounds", "ebm_learning_rate",
        }
        mode = self.learning_mode
        if mode == "sparse_linear":
            return common | basis
        if mode == "boosting":
            return common | {"boosting_stages"}
        if mode == "bagging":
            return common | {"bagging_estimators"}
        # multi / auto / single / anything unrecognised: be permissive.
        return common | basis | {
            "boosting_stages", "bagging_estimators", "hybrid_branch_count",
        }

    def get_current_config(self) -> dict:
        return {
            "mutation_rounds_per_fit": self.mutation_rounds_per_fit,
            "learning_mode": self.learning_mode,
            "boosting_stages": self.boosting_stages,
            "bagging_estimators": self.bagging_estimators,
            "tournament_k": self.tournament_k,
            "hybrid_branch_count": self.hybrid_branch_count,
            "parallel_eval_workers": self.parallel_eval_workers,
            "use_raw_basis": self.use_raw_basis,
            "use_symbolic_basis": self.use_symbolic_basis,
            "use_stump_basis": self.use_stump_basis,
            "use_rule_basis": self.use_rule_basis,
            "use_neural_basis": self.use_neural_basis,
            "use_kan_basis": self.use_kan_basis,
            "use_feature_weight_penalty": self.use_feature_weight_penalty,
            "utility_gated_formulas": self.utility_gated_formulas,
            "direct_formula_install": self.direct_formula_install,
            "sparse_l1_c": self.sparse_l1_c,
        }

    def set_custom_objective(self, grad_fn: object | None, hess_fn: object | None) -> None:
        pass
