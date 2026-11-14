"""UIL Miner — Automatic Cause Discovery via Decision Tree Meta-Learning.

The Miner reads the raw numerical journal and trains a DecisionTreeRegressor
to discover WHAT conditions cause high regret. The tree's splits become
machine-discovered rules — thresholds the computer found, not the developer.

Two mining levels (closer to causal inference):
1. Within-dataset: same dataset, different actions → stronger causal evidence
2. Cross-dataset: different datasets → correlational, needs more evidence

V20+.2: Derived Feature Synthesis — when R² is low, the Miner auto-generates
interaction features (e.g., penalty * sparsity) and nonlinear transforms
(e.g., log(n_samples)) to discover complex relationships the base DT misses.

Reports R² score to the Stress Engine (low R² = blind to causes = higher stress).
"""

from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Column names for the meta-feature matrix
STATE_FEATURES = [
    "n_samples",
    "n_features",
    "num_classes",
    "entropy",
    "feat_ratio",
    # Rich derived signals
    "sample_feature_ratio",
    "feature_correlation_mean",
    "sparsity",
]

ACTION_FEATURES = [
    "penalty",
]

ALL_FEATURES = STATE_FEATURES + ACTION_FEATURES


class UIL_Miner:
    """Automatic Cause Discovery. No hardcoded rules. Pure math.

    Trains a Decision Tree on journal vectors to find conditions that
    predict high regret. The tree discovers thresholds like:
        penalty > 2.14 AND n_samples <= 47
    without anyone typing those numbers.

    V20+.2: When the base tree has low R² (< 0.5), the Miner synthesizes
    derived interaction features and re-mines with the expanded matrix.
    """

    # V20+.2: R² threshold below which derived features are synthesized
    DERIVED_FEATURE_R2_THRESHOLD = 0.5

    # M0.2 (v22 blueprint §4.3): tiny-data mining mode. Below min_entries
    # the miner used to sleep entirely -- the one regime where reasoning
    # should beat brute force (n<150) was the regime where the cognitive
    # layer was off. With TINY_MIN_ENTRIES..min_entries-1 entries we now
    # mine SHALLOW rules (depth 2) whose confidence is shrunk toward 0.5
    # by an evidence-based factor (the "wide confidence interval").
    TINY_MIN_ENTRIES = 12

    def __init__(self, min_entries: int = 40):
        self.min_entries = min_entries
        self._last_r2: float = 0.0
        self._last_tree: DecisionTreeRegressor | None = None
        # V20+.2: Track which derived features proved useful
        self._useful_derived: list[str] = []

    def get_last_r2_score(self) -> float:
        """Return the R² of the last mining attempt.
        
        Used by StressEngine as 'explanation failure' signal:
        High R² → we understand what causes regret → lower stress
        Low R² → we're blind → higher stress
        """
        return self._last_r2

    # ── Main Mining Entry Point ──────────────────────────────────────

    def mine(
        self,
        journal_entries: list[dict],
        stress_level: float = 0.0,
        dataset_filter: str | None = None,
    ) -> list[dict]:
        """Mine the journal for machine-discovered rules.
        
        Args:
            journal_entries: All journal entries.
            stress_level: Current system stress (governs tree depth).
            dataset_filter: If set, mine only this dataset (within-dataset).
        
        Returns:
            List of discovered rule dicts.
        """
        # Filter entries
        entries = [
            e for e in journal_entries
            if e.get("classifier") != "FAILSAFE"  # Skip failsafe entries
        ]
        if dataset_filter:
            entries = [e for e in entries if e.get("dataset") == dataset_filter]

        # M0.2: tiny-data mode engages between TINY_MIN_ENTRIES and
        # min_entries-1 instead of returning nothing.
        tiny_mode = len(entries) < self.min_entries
        min_rows = self.TINY_MIN_ENTRIES if tiny_mode else self.min_entries
        if len(entries) < min_rows:
            self._last_r2 = 0.0
            return []

        # Build feature matrix from journal
        X_meta, y_regret, valid_features = self._build_matrix(entries)

        if X_meta is None or len(X_meta) < min_rows:
            self._last_r2 = 0.0
            return []

        # Tree depth scales with stress: deeper tree when stressed.
        # Tiny mode is pinned shallow: with <40 rows a depth-3+ tree is
        # memorization, and there is no held-out split to catch it.
        if tiny_mode:
            max_depth = 2
        else:
            max_depth = 3 + int(stress_level * 2)  # 3 calm → 5 max stress

        # Held-out split so R² reflects generalization, not memorization.
        # With <40 rows, the split would leave fewer than ~10 test rows —
        # too noisy to be a meaningful signal. In that case fall back to
        # full-data R² (same as the old behaviour) and skip synthesis
        # (can't honestly evaluate derived features without held-out data).
        _MIN_FOR_HOLDOUT = 40
        if len(X_meta) >= _MIN_FOR_HOLDOUT:
            _all_idx = np.arange(len(X_meta))
            _tr_idx, _hld_idx = train_test_split(
                _all_idx, test_size=0.25, random_state=42
            )
            X_tr, X_hld = X_meta[_tr_idx], X_meta[_hld_idx]
            y_tr, y_hld = y_regret[_tr_idx], y_regret[_hld_idx]
            _has_holdout = True
        else:
            X_tr, X_hld = X_meta, X_meta
            y_tr, y_hld = y_regret, y_regret
            _tr_idx = _hld_idx = None
            _has_holdout = False

        # Train the Decision Tree — THE TREE DISCOVERS THE THRESHOLDS
        min_leaf = 4 if tiny_mode else max(5, len(X_tr) // 20)
        tree = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            random_state=42,
        )
        tree.fit(X_tr, y_tr)

        # Record R² for the Stress Engine — honest held-out score when
        # possible; in-sample fallback only when data is too scarce to
        # afford a split (in which case it's flagged as low-confidence).
        base_r2 = max(0.0, float(tree.score(X_hld, y_hld)))
        self._last_r2 = base_r2
        self._last_tree = tree

        # V20+.2: If R² is low, synthesize derived features and retry.
        # Guard: synthesis is only allowed when we have a held-out split
        # to evaluate the enhanced model honestly. Without it, all 15
        # derived-feature candidates are tested against the same training
        # data the tree was fit on — a p-hacking setup guaranteed to raise
        # in-sample R² while learning nothing about generalisation.
        if base_r2 < self.DERIVED_FEATURE_R2_THRESHOLD and _has_holdout:
            X_enhanced, enhanced_names = self._synthesize_derived_features(
                X_meta, valid_features
            )
            if X_enhanced is not None and X_enhanced.shape[1] > X_meta.shape[1]:
                tree_enhanced = DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_leaf=max(5, len(_tr_idx) // 20),
                    random_state=42,
                )
                # Fit on train split of the enhanced matrix, score on held-out.
                tree_enhanced.fit(X_enhanced[_tr_idx], y_tr)
                enhanced_r2 = max(
                    0.0, float(tree_enhanced.score(X_enhanced[_hld_idx], y_hld))
                )

                # Accept enhanced model only if it genuinely generalises better.
                if enhanced_r2 > base_r2 + 0.05:
                    self._last_r2 = enhanced_r2
                    self._last_tree = tree_enhanced
                    tree = tree_enhanced
                    X_meta = X_enhanced
                    valid_features = enhanced_names
                    # Track which derived features were useful
                    self._useful_derived = [
                        n for n in enhanced_names if n.startswith("derived.")
                    ]

        # Extract rules from the tree structure
        rules = self._extract_rules(
            tree, X_meta, y_regret, valid_features,
            min_evidence=2 if tiny_mode else min(5, min_leaf),
            tiny_mode=tiny_mode,
        )

        # Tag rules with mining level
        level = "within_dataset" if dataset_filter else "cross_dataset"
        # F6 — propensity / logging-policy footnote (Swaminathan & Joachims).
        # Rules inherit the search's action distribution; surface it so HITL
        # can see the confound. Does not change promotion math.
        logging_policy = self._logging_policy_footnote(entries)
        for rule in rules:
            rule["mining_level"] = level
            rule["logging_policy"] = logging_policy
            if dataset_filter:
                rule["dataset"] = dataset_filter

        return rules

    def mine_within_dataset(
        self,
        journal_entries: list[dict],
        dataset_name: str,
        stress_level: float = 0.0,
    ) -> list[dict]:
        """Mine within a single dataset — stronger causal evidence.
        
        Same dataset means n_samples, n_features, num_classes are constant.
        If penalty variation predicts regret, penalty is closer to causal.
        """
        return self.mine(
            journal_entries,
            stress_level=stress_level,
            dataset_filter=dataset_name,
        )

    def mine_cross_dataset(
        self,
        journal_entries: list[dict],
        stress_level: float = 0.0,
    ) -> list[dict]:
        """Mine across all datasets — correlational, needs more evidence."""
        return self.mine(
            journal_entries,
            stress_level=stress_level,
            dataset_filter=None,
        )

    # ── Feature Matrix Construction ──────────────────────────────────

    def _build_matrix(
        self, entries: list[dict]
    ) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
        """Build feature matrix X and target y from journal entries.
        
        V20: Dynamically discovers features from state, action, AND
        classifier_config (nested inside action). This lets the Miner
        discover rules about hyperparameters (e.g., "when max_depth > 8,
        regret increases").
        """
        # Discover available features
        available = set()
        for e in entries:
            state = e.get("state", {})
            action = e.get("action", {})
            for key in state:
                if isinstance(state[key], (int, float)):
                    available.add(("state", key))
            for key in action:
                if key == "classifier_config":
                    # V20: flatten classifier_config into separate features
                    cfg = action[key]
                    if isinstance(cfg, dict):
                        for ckey, cval in cfg.items():
                            if isinstance(cval, (int, float)):
                                available.add(("config", ckey))
                elif isinstance(action[key], (int, float)):
                    available.add(("action", key))

        if not available:
            return None, None, []

        # Sort for determinism
        available = sorted(available)
        feature_names = [f"{src}.{key}" for src, key in available]

        # Build matrix
        rows = []
        targets = []
        for e in entries:
            state = e.get("state", {})
            action = e.get("action", {})
            config = action.get("classifier_config", {}) if isinstance(action, dict) else {}
            regret = e.get("regret", 0.0)
            if not isinstance(regret, (int, float)) or not np.isfinite(float(regret)):
                continue

            row = []
            valid = True
            for src, key in available:
                if src == "state":
                    source = state
                elif src == "action":
                    source = action
                elif src == "config":
                    source = config if isinstance(config, dict) else {}
                else:
                    source = {}

                val = source.get(key)
                if val is None or not isinstance(val, (int, float)):
                    valid = False
                    break
                fval = float(val)
                # M0 hardening: a poisoned entry (NaN/Inf or beyond
                # float32 range) must not crash mining for everyone --
                # sklearn casts to float32 and rejects such values.
                if not np.isfinite(fval) or abs(fval) > 1e30:
                    valid = False
                    break
                row.append(fval)

            if valid:
                rows.append(row)
                targets.append(float(regret))

        # M0.2: TINY_MIN_ENTRIES, not min_entries -- mine() applies the
        # real (mode-dependent) row minimum after this returns; this
        # check only guards against a matrix too small to mine at all.
        if len(rows) < self.TINY_MIN_ENTRIES:
            return None, None, []

        return np.array(rows), np.array(targets), feature_names

    # ── V20+.2: Derived Feature Synthesis ─────────────────────────────

    def _synthesize_derived_features(
        self,
        X: np.ndarray,
        feature_names: list[str],
    ) -> tuple[np.ndarray | None, list[str]]:
        """Synthesize derived interaction and nonlinear features.

        When the base Decision Tree has low R², the raw features alone
        can't explain regret. This method generates:
        - Pairwise interactions (a * b)
        - Ratio features (a / b)
        - Nonlinear transforms (log, square, inverse)

        Only generates features where the base values have variance
        (constant columns produce useless derivations).

        Returns:
            (X_enhanced, enhanced_names) — the original matrix with
            derived columns appended, and the corresponding names.
        """
        n_samples, n_base = X.shape
        if n_base < 2:
            return None, feature_names

        derived_cols = []
        derived_names = []

        # Identify columns with non-trivial variance
        col_std = np.std(X, axis=0)
        active_cols = [i for i in range(n_base) if col_std[i] > 1e-8]

        # Cap the number of interactions to prevent explosion
        # At most 15 derived features
        max_derived = 15

        # 1. Pairwise interactions: a * b
        for i in range(len(active_cols)):
            if len(derived_cols) >= max_derived:
                break
            for j in range(i + 1, len(active_cols)):
                if len(derived_cols) >= max_derived:
                    break
                ci, cj = active_cols[i], active_cols[j]
                product = X[:, ci] * X[:, cj]
                if np.std(product) > 1e-8:
                    derived_cols.append(product)
                    derived_names.append(
                        f"derived.{feature_names[ci]}*{feature_names[cj]}"
                    )

        # 2. Nonlinear transforms on active columns
        for ci in active_cols:
            if len(derived_cols) >= max_derived:
                break
            col = X[:, ci]

            # log(1 + |x|) — captures diminishing returns
            log_col = np.log1p(np.abs(col))
            if np.std(log_col) > 1e-8:
                derived_cols.append(log_col)
                derived_names.append(f"derived.log1p({feature_names[ci]})")

            if len(derived_cols) >= max_derived:
                break

            # x² — captures polynomial relationships
            sq_col = col ** 2
            if np.std(sq_col) > 1e-8:
                derived_cols.append(sq_col)
                derived_names.append(f"derived.sq({feature_names[ci]})")

            if len(derived_cols) >= max_derived:
                break

            # 1 / (1 + |x|) — bounded inverse
            inv_col = 1.0 / (1.0 + np.abs(col))
            if np.std(inv_col) > 1e-8:
                derived_cols.append(inv_col)
                derived_names.append(f"derived.inv({feature_names[ci]})")

        if not derived_cols:
            return None, feature_names

        # Stack derived columns onto original matrix
        derived_matrix = np.column_stack(derived_cols)
        X_enhanced = np.hstack([X, derived_matrix])
        enhanced_names = feature_names + derived_names

        return X_enhanced, enhanced_names

    def get_useful_derived_features(self) -> list[str]:
        """Return the derived features that improved R² in the last mining.

        Useful for the journal schema — if derived features helped,
        future entries can precompute them.
        """
        return list(self._useful_derived)

    # ── Rule Extraction from Tree ────────────────────────────────────

    def _extract_rules(
        self,
        tree: DecisionTreeRegressor,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        min_evidence: int = 5,
        tiny_mode: bool = False,
    ) -> list[dict]:
        """Walk the decision tree and extract high-regret paths as rules.

        Each path from root to a high-regret leaf becomes a rule.

        M0.2: in tiny_mode, confidence is shrunk toward 0.5 by an
        evidence-based factor n/(n+8) -- a cheap wide-confidence-interval
        stand-in so few-sample rules enter the library as weak beliefs
        that must earn promotion, not as facts.
        """
        tree_struct = tree.tree_
        median_regret = float(np.median(y))
        rules = []

        def walk(node_id: int, conditions: list[dict]) -> None:
            left = tree_struct.children_left[node_id]
            right = tree_struct.children_right[node_id]

            # Leaf node
            if left == right:  # TREE_LEAF
                predicted_regret = float(tree_struct.value[node_id][0][0])
                n_samples = int(tree_struct.n_node_samples[node_id])

                # Only extract rules for HIGH regret leaves
                if predicted_regret > median_regret and n_samples >= min_evidence:
                    confidence = self._compute_confidence(
                        conditions, X, y, feature_names
                    )
                    if tiny_mode:
                        shrink = n_samples / (n_samples + 8.0)
                        confidence = 0.5 + (confidence - 0.5) * shrink
                    rule = {
                        "conditions": list(conditions),
                        "predicted_regret": round(predicted_regret, 4),
                        "evidence_count": n_samples,
                        "confidence": round(confidence, 4),
                    }
                    if tiny_mode:
                        rule["tiny_data"] = True
                    rules.append(rule)
                return

            feature_idx = tree_struct.feature[node_id]
            threshold = float(tree_struct.threshold[node_id])
            fname = (
                feature_names[feature_idx]
                if feature_idx < len(feature_names)
                else f"feature_{feature_idx}"
            )

            # Left child: feature <= threshold
            walk(
                left,
                conditions
                + [{"feature": fname, "op": "<=", "threshold": round(threshold, 4)}],
            )

            # Right child: feature > threshold
            walk(
                right,
                conditions
                + [{"feature": fname, "op": ">", "threshold": round(threshold, 4)}],
            )

        walk(0, [])
        return rules

    def _logging_policy_footnote(self, entries: list[dict]) -> dict:
        """Summarize the action distribution under which rules were mined.

        F6 / Swaminathan & Joachims: passive mining inherits the logging
        policy's biases. This footnote does not correct the rules; it makes
        the confound visible for HITL and lab reports.
        """
        penalties: list[float] = []
        config_numeric: dict[str, list[float]] = {}
        config_str: dict[str, list[str]] = {}
        classifiers: list[str] = []
        search_modes: list[str] = []

        for e in entries:
            clf = e.get("classifier")
            if clf:
                classifiers.append(str(clf))
            action = e.get("action") or {}
            if not isinstance(action, dict):
                continue
            pen = action.get("penalty")
            if isinstance(pen, (int, float)) and np.isfinite(float(pen)):
                penalties.append(float(pen))
            sm = action.get("search_mode") or e.get("search_mode")
            if sm is not None:
                search_modes.append(str(sm))
            cfg = action.get("classifier_config")
            if not isinstance(cfg, dict):
                continue
            for k, v in cfg.items():
                if isinstance(v, bool):
                    config_numeric.setdefault(k, []).append(1.0 if v else 0.0)
                elif isinstance(v, (int, float)) and np.isfinite(float(v)):
                    config_numeric.setdefault(k, []).append(float(v))
                elif isinstance(v, str):
                    config_str.setdefault(k, []).append(v)

        def _num_summary(vals: list[float]) -> dict:
            arr = np.asarray(vals, dtype=float)
            return {
                "n": int(arr.size),
                "mean": round(float(arr.mean()), 4),
                "std": round(float(arr.std()), 4),
                "min": round(float(arr.min()), 4),
                "max": round(float(arr.max()), 4),
                "n_distinct": int(len({round(float(v), 6) for v in vals})),
            }

        footnote: dict = {
            "n_journal_entries": len(entries),
            "classifiers": sorted(set(classifiers)),
            "search_modes": sorted(set(search_modes)),
        }
        if penalties:
            footnote["action.penalty"] = _num_summary(penalties)
        for k, vals in sorted(config_numeric.items()):
            footnote[f"config.{k}"] = _num_summary(vals)
        for k, vals in sorted(config_str.items()):
            footnote[f"config.{k}"] = {
                "n": len(vals),
                "n_distinct": len(set(vals)),
                "values": sorted(set(vals))[:20],
            }
        return footnote

    def _compute_confidence(
        self,
        conditions: list[dict],
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> float:
        """Compute confidence: fraction of matching samples that have high regret."""
        if not conditions or len(X) == 0:
            return 0.0

        median_regret = float(np.median(y))
        mask = np.ones(len(X), dtype=bool)

        for cond in conditions:
            fname = cond["feature"]
            if fname not in feature_names:
                continue
            idx = feature_names.index(fname)
            threshold = cond["threshold"]

            if cond["op"] == "<=":
                mask &= X[:, idx] <= threshold
            elif cond["op"] == ">":
                mask &= X[:, idx] > threshold
            elif cond["op"] == "<":
                mask &= X[:, idx] < threshold
            elif cond["op"] == ">=":
                mask &= X[:, idx] >= threshold

        matching = y[mask]
        if len(matching) == 0:
            return 0.0

        # Confidence = fraction of matching samples with above-median regret
        return float(np.mean(matching > median_regret))
