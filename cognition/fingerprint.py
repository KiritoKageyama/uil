"""Five-axis orthogonal statistical fingerprint (v22 §2.1).

Each axis isolates one property: center, spread, shape, dependence, sequence.
Store effect sizes / statistics only — never p-values.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from scipy import stats

from figpso_uil.cognition.paths import results_path


def _axis1_center(X: np.ndarray, y: np.ndarray | None) -> dict[str, float]:
    """Robust center measures."""
    out: dict[str, float] = {}
    for j in range(X.shape[1]):
        col = X[:, j]
        out[f"trimmed_mean_f{j}"] = float(stats.trim_mean(col, proportiontocut=0.1))
        out[f"median_f{j}"] = float(np.median(col))
    if y is not None and len(np.unique(y)) <= 20:
        for cls in np.unique(y):
            mask = y == cls
            if mask.sum() >= 3:
                out[f"class_{int(cls)}_median_f0"] = float(np.median(X[mask, 0]))
    return out


def _axis2_spread(X: np.ndarray, y: np.ndarray | None) -> dict[str, float]:
    """Dispersion — MAD and Levene across class groups."""
    out: dict[str, float] = {}
    for j in range(min(X.shape[1], 8)):
        col = X[:, j]
        med = float(np.median(col))
        mad = float(np.median(np.abs(col - med))) or 1e-9
        out[f"mad_f{j}"] = mad
    if y is not None and len(np.unique(y)) >= 2:
        groups = [X[y == c, 0] for c in np.unique(y) if (y == c).sum() >= 3]
        if len(groups) >= 2:
            try:
                stat, _ = stats.levene(*groups[:5])
                out["levene_f0"] = float(stat)
            except Exception:
                pass
    return out


def _axis3_shape(X: np.ndarray) -> dict[str, float]:
    """Skewness, kurtosis, normality distance on first features."""
    out: dict[str, float] = {}
    for j in range(min(X.shape[1], 5)):
        col = X[:, j]
        if np.std(col) < 1e-12:
            continue
        try:
            sk, _ = stats.skewtest(col)
            out[f"skew_f{j}"] = float(sk)
        except Exception:
            out[f"skew_f{j}"] = float(stats.skew(col))
        try:
            ku, _ = stats.kurtosistest(col)
            out[f"kurtosis_f{j}"] = float(ku)
        except Exception:
            out[f"kurtosis_f{j}"] = float(stats.kurtosis(col))
        if len(col) >= 8:
            try:
                w, _ = stats.shapiro(col[: min(5000, len(col))])
                out[f"shapiro_w_f{j}"] = float(w)
            except Exception:
                pass
    return out


def _axis4_dependence(X: np.ndarray, y: np.ndarray | None, seed: int) -> dict[str, float]:
    """Spearman summary, Cramér's V, optional probe scores."""
    out: dict[str, float] = {}
    n, f = X.shape
    if f >= 2:
        ranks = np.argsort(np.argsort(X[:, : min(f, 10)], axis=0), axis=0)
        rho = np.corrcoef(ranks.T)
        upper = rho[np.triu_indices_from(rho, k=1)]
        if upper.size:
            out["spearman_rho_mean"] = float(np.mean(np.abs(upper)))
            out["spearman_rho_max"] = float(np.max(np.abs(upper)))
    if y is not None:
        u = np.unique(y)
        if len(u) <= 20:
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import cross_val_score

                acc = cross_val_score(
                    LogisticRegression(max_iter=500, random_state=seed), X, y, cv=3,
                ).mean()
                out["linear_probe_acc"] = round(float(acc), 4)
            except Exception:
                pass
            # Non-linear probe: a shallow RandomForest captures feature
            # INTERACTIONS a linear model cannot. A large gap over the linear
            # probe means the class boundary NEEDS interactions (sonar:
            # linear ~0.77, RF ~0.85 -> gap ~0.08). The optimizer reads
            # interaction_gap to switch on the sparse_linear RULE basis (E5)
            # for such fingerprints. Subsampled + shallow to stay cheap.
            try:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import cross_val_score as _cvs
                Xp, yp = X, y
                if n > 2000:
                    _idx = np.random.default_rng(seed).choice(n, 2000, replace=False)
                    Xp, yp = X[_idx], y[_idx]
                nl = _cvs(
                    RandomForestClassifier(
                        n_estimators=60, max_depth=8, random_state=seed, n_jobs=1,
                    ), Xp, yp, cv=3,
                ).mean()
                out["nonlinear_probe_acc"] = round(float(nl), 4)
                if "linear_probe_acc" in out:
                    out["interaction_gap"] = round(
                        float(out["nonlinear_probe_acc"] - out["linear_probe_acc"]), 4,
                    )
            except Exception:
                pass
            # Research step 7 — TabPFN v2 as an honest ceiling probe (optional
            # dep). Soft-fails when tabpfn is not installed. Never becomes the
            # default learner here — fingerprint-only evidence for the router.
            try:
                from tabpfn import TabPFNClassifier
                if n <= 10_000 and f <= 500 and n >= 40 and len(u) >= 2:
                    Xp, yp = X, y
                    if n > 500:
                        _idx = np.random.default_rng(seed + 17).choice(
                            n, 500, replace=False,
                        )
                        Xp, yp = X[_idx], y[_idx]
                    # TabPFN APIs differ across versions; try common kwargs.
                    try:
                        _tp = TabPFNClassifier(device="cpu", random_state=seed)
                    except TypeError:
                        _tp = TabPFNClassifier(device="cpu")
                    # Single stratified split is enough for a ceiling probe;
                    # full CV is too expensive for a fingerprint step.
                    from sklearn.model_selection import train_test_split as _tts
                    try:
                        Xa, Xb, ya, yb = _tts(
                            Xp, yp, test_size=0.3, random_state=seed, stratify=yp,
                        )
                    except ValueError:
                        Xa, Xb, ya, yb = _tts(
                            Xp, yp, test_size=0.3, random_state=seed,
                        )
                    _tp.fit(Xa, ya)
                    tp_acc = float(np.mean(_tp.predict(Xb) == yb))
                    out["tabpfn_probe_acc"] = round(tp_acc, 4)
                    if "linear_probe_acc" in out:
                        out["tabpfn_gap"] = round(
                            tp_acc - float(out["linear_probe_acc"]), 4,
                        )
            except Exception:
                pass
        if len(u) <= 10 and n >= 30:
            try:
                ct = np.zeros((len(u), 2))
                for i, c in enumerate(u):
                    ct[i, 0] = (y == c).sum()
                    ct[i, 1] = n - ct[i, 0]
                chi2, _, _, _ = stats.chi2_contingency(ct + 0.5)
                n_tot = ct.sum()
                out["cramers_v"] = float(np.sqrt(chi2 / (n_tot * (min(ct.shape) - 1) + 1e-9)))
            except Exception:
                pass
    if y is not None and len(np.unique(y)) == 2 and n >= 40:
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score

            from figpso_uil.cognition.symbolic_regressor import SymbolicRegressor

            if np.issubdtype(y.dtype, np.floating) or len(np.unique(y)) > 2:
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y.astype(float), test_size=0.3, random_state=seed,
                )
                sr = SymbolicRegressor(seed=seed, n_terms=6).fit(Xtr, ytr)
                out["formula_probe_r2"] = round(float(r2_score(yte, sr.predict(Xte))), 4)
        except Exception:
            pass
    return out


def _axis5_sequence(X: np.ndarray) -> dict[str, float]:
    """Order-of-collection signals — lag autocorrelation + Ljung-Box proxy."""
    out: dict[str, float] = {}
    n = X.shape[0]
    if n < 20:
        return out
    proxy = np.asarray(X[:, 0] if X.shape[1] else np.arange(n, dtype=float), dtype=float)
    # Lag-1 autocorrelation catches AR(1) and ordered collection directly.
    if n > 2:
        c0 = proxy - proxy.mean()
        denom = float(np.dot(c0, c0))
        if denom > 1e-12:
            lag1 = float(np.dot(c0[1:], c0[:-1])) / denom
            out["lag1_autocorr"] = lag1
    for lag in (1, 5, 10):
        if n <= lag + 5:
            continue
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox

            lb = acorr_ljungbox(proxy, lags=[lag], return_df=True)
            out[f"ljung_box_lag{lag}"] = float(lb["lb_stat"].iloc[0])
        except Exception:
            diffs = np.diff(proxy)
            if len(diffs) > lag:
                ac = np.corrcoef(diffs[:-lag], diffs[lag:])[0, 1]
                out[f"autocorr_lag{lag}"] = float(ac) if np.isfinite(ac) else 0.0
    return out


def compute_fingerprint(
    X: np.ndarray,
    y: np.ndarray | None = None,
    *,
    seed: int = 0,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Build the 5-axis fingerprint dict."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y) if y is not None else None
    n, f = X.shape
    tt = "continuous"
    if y is not None:
        u = np.unique(y)
        if len(u) <= 2:
            tt = "binary"
        elif len(u) <= 20 and np.allclose(y, np.round(y)):
            tt = "multiclass"
    fp: dict[str, Any] = {
        "n_samples": int(n),
        "n_features": int(f),
        "target_type": tt,
        "tiny": n < 50,
        "axes": {},
    }
    fp["axes"]["center"] = _axis1_center(X, y)
    fp["axes"]["spread"] = _axis2_spread(X, y)
    fp["axes"]["shape"] = _axis3_shape(X)
    fp["axes"]["dependence"] = _axis4_dependence(X, y, seed)
    fp["axes"]["sequence"] = _axis5_sequence(X)

    seq = fp["axes"]["sequence"]
    lag1 = abs(seq.get("lag1_autocorr", 0.0))
    lb_vals = [v for k, v in seq.items() if k.startswith("ljung_box")]
    # Use lag-1 series autocorrelation only — diff-based proxies false-positive on IID data.
    fp["sequential"] = lag1 >= 0.35 or (lb_vals and max(lb_vals) > 8.0)

    dep = fp["axes"]["dependence"]
    fp["formula_probe_r2"] = dep.get("formula_probe_r2")
    fp["linear_probe_acc"] = dep.get("linear_probe_acc")
    fp["nonlinear_probe_acc"] = dep.get("nonlinear_probe_acc")
    fp["interaction_gap"] = dep.get("interaction_gap")
    fp["tabpfn_probe_acc"] = dep.get("tabpfn_probe_acc")
    fp["tabpfn_gap"] = dep.get("tabpfn_gap")

    if dataset_name:
        fp["dataset_name"] = dataset_name
    fp["fingerprint_id"] = hashlib.sha256(
        json.dumps(fp, sort_keys=True, default=str).encode(),
    ).hexdigest()[:16]
    return fp


def save_fingerprint(fp: dict, filepath: str | None = None) -> str:
    """Persist fingerprint as a versioned dataset cell JSON."""
    path = filepath or results_path("uil_fingerprints_v22.json")
    store: dict = {}
    try:
        with open(path, encoding="utf-8") as fh:
            store = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    key = fp.get("dataset_name") or fp.get("fingerprint_id", "unknown")
    store[key] = fp
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2, default=str)
    return path
