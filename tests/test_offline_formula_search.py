"""Offline trial-and-error formula builder — no API key.

Run:
    python figpso_uil/tests/test_offline_formula_search.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from figpso_uil.cognition.llm_mutator import LLMMutator
from figpso_uil.cognition.offline_formula_search import trial_and_error_search
from figpso_uil.cognition.residual_formula_install import grow_residual_terms

_FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        _FAILURES.append(name)


def _xor(n=300, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X, y


def test_o1_trial_finds_signal():
    X, y = _xor()
    out = trial_and_error_search(
        X, y, seed=42, n_trials=40, n_keep=2,
    )
    check("O1 trial nonempty", len(out) >= 1, f"n={len(out)}")
    if out:
        check("O1 score positive", out[0].score > 0.01, f"score={out[0].score:.4f}")


def test_o2_deterministic():
    X, y = _xor(seed=1)
    a = trial_and_error_search(X, y, seed=7, n_trials=30, n_keep=2)
    b = trial_and_error_search(X, y, seed=7, n_trials=30, n_keep=2)
    keys_a = [r.mutation_type + str(r.score) for r in a]
    keys_b = [r.mutation_type + str(r.score) for r in b]
    # Same seed → same scores/order
    check("O2 same seed", keys_a == keys_b, f"{keys_a} vs {keys_b}")


def test_o3_mutator_offline_ignores_api_key(monkeypatch_env=True):
    """Even with OPENAI_API_KEY set, offline_only never marks LLM available."""
    old = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "sk-fake-should-not-be-used"
    try:
        m = LLMMutator(offline_only=True)
        check("O3 offline_only ignores key", m.available is False and m.api_key is None)
        # propose must not require network — engine/neural fallback only
        props = m.propose(fingerprint={}, cells=[], n_proposals=1)
        check("O3 propose offline works", len(props) >= 1)
        check(
            "O3 proposer not llm",
            all(p.proposer != "llm" for p in props),
            f"proposers={[p.proposer for p in props]}",
        )
    finally:
        if old is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = old


def test_o4_residual_default_is_trial():
    X, y = _xor()
    terms = grow_residual_terms(X, y, seed=3, n_terms=2, method="trial", n_trials=32)
    check("O4 residual trial grows", len(terms) >= 1)
    if terms:
        check(
            "O4 mutation offline_trial",
            "offline_trial" in terms[0].mutation_type,
            terms[0].mutation_type,
        )


if __name__ == "__main__":
    print("Offline formula search tests")
    test_o1_trial_finds_signal()
    test_o2_deterministic()
    test_o3_mutator_offline_ignores_api_key()
    test_o4_residual_default_is_trial()
    if _FAILURES:
        print(f"\n{len(_FAILURES)} FAILED: {_FAILURES}")
        sys.exit(1)
    print("\nAll O1-O4 passed.")
