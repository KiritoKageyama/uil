"""UIL Failsafe Engine — Classifier diagnostic clutch.

NOT the main mechanism. The absolute LAST RESORT.
Activated ONLY when the model concludes it has genuinely failed.

Runs all available classifiers on the same data to diagnose:
1. Classifier mismatch: another classifier works → wrong tool for the job
2. Fundamental problem: no classifier works → data is the problem

Records rich mathematical explanations: WHY did each classifier succeed or fail,
based on data properties (correlation, sparsity, class balance, etc.)
"""

from __future__ import annotations
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


class FailsafeEngine:
    """Classifier diagnostic clutch. Last resort only.
    
    When the system has genuinely concluded failure, the failsafe:
    1. Runs all adapters on the same data
    2. Computes a data profile (mathematical fingerprint)
    3. Diagnoses whether the problem is classifier choice or data quality
    4. Records explanations for the journal
    """

    def __init__(self, all_adapters: list | None = None):
        self.all_adapters = all_adapters or []
        self._already_run: bool = False

    def diagnose(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_weights: np.ndarray,
        primary_classifier_name: str,
        seed: int = 42,
    ) -> dict:
        """Run all classifiers and diagnose the failure.
        
        Args:
            X: Training features.
            y: Training labels.
            feature_weights: Current best feature weights.
            primary_classifier_name: Name of the primary classifier that failed.
            seed: Random seed.
        
        Returns:
            Diagnosis dict with mathematical explanations.
        """
        self._already_run = True

        # Split for evaluation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )

        # Compute data profile (mathematical fingerprint)
        data_profile = self._compute_data_profile(X, y)

        # Run each adapter
        results = {}
        for adapter in self.all_adapters:
            try:
                start_time = time.perf_counter()
                model = adapter.fit_weighted(X_train, y_train, feature_weights)
                preds = adapter.predict(model, X_val)
                elapsed = time.perf_counter() - start_time

                acc = float(accuracy_score(y_val, preds))
                f1 = float(f1_score(y_val, preds, average="weighted"))

                results[adapter.name] = {
                    "accuracy": round(acc, 4),
                    "f1_score": round(f1, 4),
                    "cpu_time": round(elapsed, 3),
                    "complexity": adapter.get_model_complexity(model),
                }
            except Exception as e:
                results[adapter.name] = {
                    "accuracy": 0.0,
                    "f1_score": 0.0,
                    "cpu_time": 0.0,
                    "error": str(e),
                }

        # Diagnose
        primary_score = results.get(primary_classifier_name, {}).get("accuracy", 0.0)
        best_alt_name = max(
            (name for name in results if name != primary_classifier_name),
            key=lambda n: results[n].get("accuracy", 0.0),
            default=None,
        )

        if best_alt_name and results[best_alt_name]["accuracy"] > primary_score + 0.03:
            diagnosis = self._classifier_mismatch_diagnosis(
                primary_classifier_name,
                best_alt_name,
                results,
                data_profile,
            )
        else:
            diagnosis = self._fundamental_problem_diagnosis(
                results,
                data_profile,
            )

        diagnosis["data_profile"] = data_profile
        diagnosis["all_results"] = results
        diagnosis["dataset_fingerprint"] = self._compute_fingerprint(data_profile)

        return diagnosis

    def was_already_run(self) -> bool:
        """Whether failsafe has already been run this session."""
        return self._already_run

    def reset(self) -> None:
        """Reset the failsafe for a new session."""
        self._already_run = False

    # ── Data Profiling ───────────────────────────────────────────────

    def _compute_data_profile(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Compute mathematical properties of the dataset.
        
        These properties are used to explain WHY classifiers
        succeed or fail, and to identify similar datasets.
        """
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        profile = {
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "num_classes": int(len(unique_classes)),
            "sample_feature_ratio": round(n_samples / max(n_features, 1), 4),
        }

        # Class balance (entropy of class distribution)
        class_counts = np.bincount(y.astype(int))
        class_probs = class_counts / class_counts.sum()
        class_probs = class_probs[class_probs > 0]
        profile["class_balance_entropy"] = round(
            float(-np.sum(class_probs * np.log2(class_probs + 1e-10))), 4
        )

        # Sparsity
        profile["sparsity"] = round(float(np.mean(X == 0)), 4)

        # Feature correlation (sample if too many features)
        try:
            if n_features > 200:
                rng = np.random.default_rng(42)
                feat_idx = rng.choice(n_features, 200, replace=False)
                X_sub = X[:, feat_idx]
            else:
                X_sub = X

            corr = np.corrcoef(X_sub.T)
            np.fill_diagonal(corr, 0)
            corr = np.abs(corr)
            profile["feature_correlation_mean"] = round(float(np.nanmean(corr)), 4)
            profile["feature_correlation_max"] = round(float(np.nanmax(corr)), 4)
        except Exception:
            profile["feature_correlation_mean"] = 0.0
            profile["feature_correlation_max"] = 0.0

        return profile

    def _compute_fingerprint(self, profile: dict) -> list[float]:
        """Compute a numerical fingerprint for dataset similarity matching.
        
        Used to find similar datasets in future runs:
        if cosine_distance(fingerprint_A, fingerprint_B) < 0.3 → similar datasets.
        """
        return [
            np.log1p(profile.get("n_samples", 0)),
            np.log1p(profile.get("n_features", 0)),
            profile.get("sample_feature_ratio", 1.0),
            profile.get("feature_correlation_mean", 0.0),
            profile.get("class_balance_entropy", 0.0),
            profile.get("sparsity", 0.0),
        ]

    # ── Diagnosis Construction ───────────────────────────────────────

    def _classifier_mismatch_diagnosis(
        self,
        primary: str,
        best_alt: str,
        results: dict,
        data_profile: dict,
    ) -> dict:
        """Diagnose: another classifier outperforms → wrong tool for this data."""
        primary_acc = results[primary]["accuracy"]
        alt_acc = results[best_alt]["accuracy"]
        improvement = alt_acc - primary_acc

        # Mathematical explanation
        explanation = {
            "primary_classifier": primary,
            "primary_accuracy": primary_acc,
            "better_classifier": best_alt,
            "better_accuracy": alt_acc,
            "improvement": round(improvement, 4),
        }

        # WHY analysis based on data properties
        reasons = []
        sfr = data_profile.get("sample_feature_ratio", 1.0)
        corr = data_profile.get("feature_correlation_mean", 0.0)
        n_features = data_profile.get("n_features", 0)

        if best_alt == "RandomForest" and corr > 0.4:
            reasons.append(
                f"High feature correlation ({corr:.2f}) favors tree ensembles "
                f"that handle correlated features natively"
            )
        if best_alt == "KNN" and sfr > 5 and n_features < 50:
            reasons.append(
                f"Low dimensionality ({n_features} features) with good sample "
                f"coverage (ratio={sfr:.1f}) favors distance-based methods"
            )
        if best_alt == "TabPFN" and sfr > 2:
            reasons.append(
                f"Moderate dataset size with sufficient sample-feature ratio "
                f"({sfr:.1f}) is in TabPFN's sweet spot"
            )
        if primary == "XGBoost" and sfr < 2:
            reasons.append(
                f"Low sample-feature ratio ({sfr:.1f}) makes gradient "
                f"boosting prone to overfitting"
            )

        return {
            "diagnosis": "classifier_mismatch",
            "explanation": explanation,
            "reasons": reasons,
            "recommendation": best_alt,
            "lesson": {
                "type": "classifier_preference",
                "condition": f"sample_feature_ratio={sfr:.2f}, correlation={corr:.2f}",
                "result": f"{best_alt} outperforms {primary} by {improvement:.4f}",
            },
        }

    def _fundamental_problem_diagnosis(
        self,
        results: dict,
        data_profile: dict,
    ) -> dict:
        """Diagnose: no classifier works → fundamental data problem."""
        best_name = max(results, key=lambda n: results[n].get("accuracy", 0.0))
        best_acc = results[best_name]["accuracy"]

        reasons = []
        sfr = data_profile.get("sample_feature_ratio", 1.0)
        sparsity = data_profile.get("sparsity", 0.0)
        corr = data_profile.get("feature_correlation_mean", 0.0)

        if sfr < 1.0:
            reasons.append(
                f"Very low sample-feature ratio ({sfr:.2f}): more features "
                f"than samples makes learning extremely difficult"
            )
        if sparsity > 0.7:
            reasons.append(
                f"High sparsity ({sparsity:.2f}): most feature values are zero, "
                f"limiting information content"
            )
        if corr > 0.7:
            reasons.append(
                f"Very high feature correlation ({corr:.2f}): features may be "
                f"redundant, reducing effective dimensionality"
            )

        return {
            "diagnosis": "fundamental_problem",
            "best_achievable": {
                "classifier": best_name,
                "accuracy": best_acc,
            },
            "reasons": reasons,
            "recommendation": "escalate_stress",
            "lesson": {
                "type": "data_limitation",
                "condition": f"No classifier achieves threshold on this data profile",
                "result": f"Best achievable: {best_acc:.4f} with {best_name}",
            },
        }
