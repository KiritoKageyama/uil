"""Domain router — the meta-controller that decides *which learner* to use.

Looks at a dataset's characteristics (a fingerprint), classifies it into one of
four domains, and routes to the strategy that wins there. This is heuristic v1;
the self-improving version (Stage 4) records fingerprint -> strategy -> outcome
in UIL's episodic memory and promotes the routing rules that generalise, so the
router's accuracy climbs over runs.

Domains:
  D2_math            continuous target a compact formula fits well  -> SymbolicRegressor
  D1_noisy_regression continuous target, no compact formula          -> boosted regressor
  D3_tiny            very few samples                                 -> principle-primed few-shot
  D4_structured      classification, (near-)linearly separable        -> sparse readable model
  D1_noisy_tabular   classification, complex boundary                 -> grad_fit / XGBoost

The probes are the honest part: rather than guess from n/features alone, the
router *tries* a cheap symbolic fit (continuous) or a linear fit
(classification) on a held-out slice and reads the score — measured evidence,
not a heuristic on shape.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

from figpso_uil.cognition.paths import results_path
from figpso_uil.cognition.symbolic_regressor import SymbolicRegressor

TINY_N = 50
FORMULA_R2_MATH = 0.90      # continuous + probe above this => a law fits => D2
LINEAR_ACC_STRUCTURED = 0.92  # classification + linear probe above => D4


def _target_type(y: np.ndarray) -> str:
    u = np.unique(y)
    if len(u) <= 2:
        return "binary"
    if len(u) <= 20 and np.allclose(y, np.round(y)):
        return "multiclass"
    return "continuous"


def fingerprint(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    """Characterise a dataset — includes measured probe scores, not just shape."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n, f = X.shape
    tt = _target_type(y)
    fp = {
        "n_samples": int(n), "n_features": int(f),
        "target_type": tt, "tiny": n < TINY_N,
        "formula_probe_r2": None, "linear_probe_acc": None,
    }

    if tt == "continuous" and n >= 40:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
        sr = SymbolicRegressor(seed=seed, n_terms=6).fit(Xtr, ytr)
        fp["formula_probe_r2"] = round(float(r2_score(yte, sr.predict(Xte))), 3)

    if tt in ("binary", "multiclass") and n >= 30:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            acc = cross_val_score(
                LogisticRegression(max_iter=500, random_state=seed), X, y, cv=3,
            ).mean()
            fp["linear_probe_acc"] = round(float(acc), 3)
        except Exception:
            pass

    return fp


# Strategies are an OPEN registry, not a fixed set — register new learners freely.
STRATEGY_DESC = {
    "symbolic": "SymbolicRegressor (structure search; extrapolates)",
    "gbm": "gradient-boosted regressor (XGBoost/GBM)",
    "grad_fit": "SymbolicFormula grad_fit / XGBoost",
    "sparse_linear": "sparse linear over symbolic basis (readable)",
    "fewshot": "principle-primed few-shot (grad_fit, heavy regularization)",
    "tabpfn": "TabPFN v2 foundation model (optional dep; small-n / ceiling)",
}


def _tabpfn_available() -> bool:
    try:
        import tabpfn  # noqa: F401
        return True
    except ImportError:
        return False


def register_strategy(key: str, description: str) -> None:
    STRATEGY_DESC[key] = description


class Domain:
    """A named region of dataset-space with a matcher and a strategy.

    matcher(fp) -> (matches: bool, confidence: float). Priority breaks ties so
    specific domains beat fallbacks. Domains are objects, not hardcoded
    branches, so the taxonomy is open — register or discover as many as needed.
    """

    def __init__(self, name, matcher, strategy, priority: int = 0):
        self.name = name
        self.matcher = matcher
        self.strategy = strategy
        self.priority = priority


class DomainRegistry:
    """An open, ordered set of domains. The defaults below are *seeds, not a
    cap* — call register() to add domains, or let RoutedController discover
    them from data at runtime."""

    def __init__(self, domains: list[Domain] | None = None):
        self.domains: list[Domain] = list(domains) if domains else []

    def register(self, domain: Domain) -> Domain:
        self.domains.append(domain)
        return domain

    def names(self) -> list[str]:
        return [d.name for d in self.domains]

    def classify(self, fp: dict) -> tuple[str, str, str]:
        best, best_key = None, (-1, -1.0)
        for d in self.domains:
            ok, conf = d.matcher(fp)
            if ok and (d.priority, conf) > best_key:
                best, best_key = d, (d.priority, conf)
        if best is None:
            return "unknown", "grad_fit", STRATEGY_DESC.get("grad_fit", "")
        return best.name, best.strategy, STRATEGY_DESC.get(best.strategy, "")


def _default_registry() -> DomainRegistry:
    """The seed domains — a starting taxonomy, deliberately not exhaustive."""
    def m_tiny(fp):
        return bool(fp.get("tiny")), 1.0

    def m_tiny_tabpfn(fp):
        # Research step 7: TabPFN as router strategy for tiny / where the
        # ceiling probe beats linear. Soft-gated on optional dep.
        if not _tabpfn_available():
            return False, 0.0
        if fp.get("target_type") not in ("binary", "multiclass"):
            return False, 0.0
        n = int(fp.get("n_samples") or 0)
        if n > 10_000 or int(fp.get("n_features") or 0) > 500:
            return False, 0.0
        gap = fp.get("tabpfn_gap")
        if fp.get("tiny") or (gap is not None and float(gap) > 0.02):
            return True, 0.95 if fp.get("tiny") else 0.85
        return False, 0.0

    def m_math(fp):
        r2 = fp.get("formula_probe_r2")
        return (fp.get("target_type") == "continuous"
                and r2 is not None and r2 >= FORMULA_R2_MATH), float(r2 or 0.0)

    def m_noisy_reg(fp):
        return fp.get("target_type") == "continuous", 0.1

    def m_structured(fp):
        lin = fp.get("linear_probe_acc")
        return (fp.get("target_type") in ("binary", "multiclass")
                and lin is not None and lin >= LINEAR_ACC_STRUCTURED), float(lin or 0.0)

    def m_moderate_linear(fp):
        lin = fp.get("linear_probe_acc")
        return (
            fp.get("target_type") in ("binary", "multiclass")
            and lin is not None
            and 0.75 <= lin < LINEAR_ACC_STRUCTURED
        ), float(lin or 0.0)

    def m_noisy_tab(fp):
        return fp.get("target_type") in ("binary", "multiclass"), 0.1

    def m_hard_boundary(fp):
        lin = fp.get("linear_probe_acc")
        return (
            fp.get("target_type") in ("binary", "multiclass")
            and lin is not None
            and lin < 0.75
        ), float(0.75 - (lin or 0.0))

    def m_sequential(fp):
        return bool(fp.get("sequential")), 0.85

    return DomainRegistry([
        Domain("D3_tiny_tabpfn", m_tiny_tabpfn, "tabpfn", priority=105),
        Domain("D3_tiny", m_tiny, "gbm", priority=100),
        Domain("D5_sequential", m_sequential, "gbm", priority=90),
        Domain("D2_math", m_math, "symbolic", priority=50),
        # Linearly-separable (D4_structured) and near-linear
        # (D4_moderate_linear) classification -> the READABLE sparse-linear
        # model, which is both more accurate AND interpretable there than a
        # black-box GBM (cross-seed: breast_cancer 0.974, digits 0.948,
        # spambase 0.944, ionosphere 0.916 — ties/beats XGBoost with a
        # readable law). This is the blueprint's stated purpose for
        # sparse_linear ("sparse linear over symbolic basis (readable)").
        Domain("D4_structured", m_structured, "sparse_linear", priority=50),
        Domain("D4_moderate_linear", m_moderate_linear, "sparse_linear", priority=48),
        # Interaction-dense / complex boundary -> GBM. (The sparse-linear
        # RULE basis, E5, is auto-enabled by the optimizer's interaction_gap
        # probe when a SymbolicFormula runs here, keeping a readable option.)
        Domain("D1_hard_boundary", m_hard_boundary, "gbm", priority=45),
        Domain("D1_noisy_regression", m_noisy_reg, "gbm", priority=10),
        Domain("D1_noisy_tabular", m_noisy_tab, "grad_fit", priority=10),
    ])


# Module-level seed registry. Extend it: DEFAULT_REGISTRY.register(Domain(...)).
DEFAULT_REGISTRY = _default_registry()


def route(fp: dict, registry: DomainRegistry | None = None) -> tuple[str, str, str]:
    """Fingerprint -> (domain, strategy_key, description) via a domain registry.
    The default registry's domains are seeds, not a limit."""
    return (registry or DEFAULT_REGISTRY).classify(fp)


def explain(fp: dict, registry: DomainRegistry | None = None) -> str:
    domain, _, desc = route(fp, registry)
    bits = [f"n={fp['n_samples']}", f"feat={fp['n_features']}", f"target={fp['target_type']}"]
    if fp.get("formula_probe_r2") is not None:
        bits.append(f"formula_probe_R2={fp['formula_probe_r2']}")
    if fp.get("linear_probe_acc") is not None:
        bits.append(f"linear_probe_acc={fp['linear_probe_acc']}")
    return f"[{', '.join(bits)}]  ->  {domain}  ->  {desc}"


def _fp_vec(fp: dict) -> np.ndarray:
    """Numeric fingerprint key for nearest-neighbour routing memory."""
    tt = fp.get("target_type")
    r2 = fp.get("formula_probe_r2")
    lin = fp.get("linear_probe_acc")
    return np.array([
        np.log1p(fp.get("n_samples", 0)) / 7.0,
        fp.get("n_features", 0) / 50.0,
        1.0 if tt == "continuous" else 0.0,
        r2 if r2 is not None else -1.0,
        lin if lin is not None else -1.0,
        1.0 if fp.get("tiny") else 0.0,
    ], dtype=float)


class RoutingMemory:
    """Accumulates fingerprint -> strategy -> outcome, recommends by kNN.

    Same fingerprint-memory idea as episodic_attention (Part 13), but the value
    stored is *which learner won*, not a hyperparameter resolution. This is what
    lets the router learn its own routing rules and self-correct a heuristic that
    used a poor proxy.
    """

    def __init__(
        self,
        filepath: str | None = None,
        top_k: int = 6,
        min_support: int = 2,
        margin: float = 0.03,
    ):
        self.filepath = filepath
        self.top_k = top_k
        self.min_support = min_support
        self.margin = margin
        self._episodes: list[dict] = []
        if self.filepath:
            self._load()
            # Ensure the path exists even with zero episodes so isolated
            # state_dir arms don't look "empty" while shared results/ gets
            # the first write from a different code path (bug #30b).
            if not os.path.isfile(self.filepath):
                self._save()

    def record(self, fp: dict, strategy: str, score: float) -> None:
        self._episodes.append({
            "fingerprint": fp,
            "vec": _fp_vec(fp).tolist(),
            "strategy": strategy,
            "score": float(score),
        })
        if self.filepath:
            self._save()

    def novelty(self, fp: dict) -> float:
        """Distance to the nearest stored fingerprint — large => unseen region,
        a candidate for a brand-new domain the seed taxonomy doesn't cover."""
        if not self._episodes:
            return float("inf")
        q = _fp_vec(fp)
        return min(
            float(np.linalg.norm(q - np.array(ep["vec"]))) for ep in self._episodes
        )

    def recommend(self, fp: dict) -> dict | None:
        """Best strategy among nearest neighbours, if confidently better."""
        if len(self._episodes) < self.min_support:
            return None
        q = _fp_vec(fp)
        ranked_eps = sorted(
            self._episodes,
            key=lambda ep: float(np.linalg.norm(q - np.array(ep["vec"]))),
        )
        neigh = ranked_eps[: max(self.top_k, self.min_support)]

        agg: dict[str, list] = defaultdict(list)
        for ep in neigh:
            agg[ep["strategy"]].append(ep["score"])
        means = sorted(
            ((s, float(np.mean(v)), len(v)) for s, v in agg.items()),
            key=lambda t: t[1], reverse=True,
        )
        best_s, best_m, best_n = means[0]
        if best_n < self.min_support:
            return None
        if len(means) >= 2 and best_m - means[1][1] < self.margin:
            return None  # neighbours don't confidently prefer one strategy
        return {
            "strategy": best_s, "mean_score": round(best_m, 3),
            "support": best_n, "n_neighbors": len(neigh),
        }

    def _save(self) -> None:
        try:
            with open(self.filepath, "w") as f:
                json.dump({"episodes": self._episodes}, f)
        except OSError:
            pass

    def _load(self) -> None:
        try:
            with open(self.filepath, "r") as f:
                self._episodes = json.load(f).get("episodes", [])
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self._episodes = []


class RoutedController:
    """Heuristic route by default; override when memory has stronger evidence;
    and grow NEW domains when it meets data unlike anything it has seen.

    The domain set is not fixed at construction — `discover_domain` mints a new
    domain centred on a novel fingerprint region, so the taxonomy expands with
    experience rather than being capped at the seed set.
    """

    def __init__(
        self,
        memory: RoutingMemory | None = None,
        registry: DomainRegistry | None = None,
        novelty_threshold: float = 0.9,
        episodic_attention=None,
    ):
        self.episodic = episodic_attention
        self.memory = memory if memory is not None else RoutingMemory(
            filepath=results_path("uil_routing_memory.json"),
        )
        # Own registry (a fresh copy of the seeds) so discovered domains don't
        # leak into the module-global default.
        self.registry = registry or _default_registry()
        self.novelty_threshold = novelty_threshold
        self._discovered = 0

    def decide(self, fp: dict) -> dict:
        domain, heur_strategy, desc = route(fp, self.registry)
        nov = self.memory.novelty(fp)
        if self.episodic is not None:
            ep_rec = self.episodic.recommend_strategy(fp)
            if ep_rec is not None:
                return {
                    "strategy": ep_rec["strategy"],
                    "source": "episodic_routing",
                    "domain": domain,
                    "evidence": ep_rec,
                    "description": STRATEGY_DESC.get(ep_rec["strategy"], ""),
                    "novel": False,
                    "novelty": round(nov, 3) if np.isfinite(nov) else None,
                }
        novel = len(self.memory._episodes) > 0 and nov > self.novelty_threshold
        rec = self.memory.recommend(fp)

        if rec is not None and rec["strategy"] != heur_strategy:
            out = {"strategy": rec["strategy"], "source": "memory",
                   "overrode_heuristic": heur_strategy, "domain": domain,
                   "evidence": rec, "description": STRATEGY_DESC.get(rec["strategy"], "")}
        elif rec is not None:
            out = {"strategy": heur_strategy, "source": "memory+heuristic-agree",
                   "domain": domain, "evidence": rec, "description": desc}
        else:
            out = {"strategy": heur_strategy, "source": "heuristic",
                   "domain": domain, "evidence": None, "description": desc}
        out["novel"] = novel
        out["novelty"] = round(nov, 3) if np.isfinite(nov) else None
        return out

    def discover_domain(
        self, fp: dict, strategy: str, name: str | None = None,
    ) -> str:
        """Register a new domain covering the fingerprint region around `fp`.

        Called when the controller has met a novel region and learned which
        strategy wins there — the taxonomy grows past the seed domains.
        """
        center = _fp_vec(fp)
        radius = self.novelty_threshold
        nm = name or f"discovered_{self._discovered}"
        self._discovered += 1

        def matcher(f, c=center, r=radius):
            d = float(np.linalg.norm(_fp_vec(f) - c))
            return d <= r, (1.0 - d / r if r > 0 else 0.0)

        # priority above generic fallbacks, below hard rules like tiny
        self.registry.register(Domain(nm, matcher, strategy, priority=70))
        return nm

    def observe(self, fp: dict, strategy: str, score: float) -> None:
        self.memory.record(fp, strategy, score)
        if self.episodic is not None:
            self.episodic.store_routing_outcome(fp, strategy, score)
