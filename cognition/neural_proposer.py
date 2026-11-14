"""Neural proposer — R1 stage 1 (v22 blueprint §5.4): the own-transformer unit.

An OFFLINE, open-box replacement for the external-LLM slot in the mutation
proposal loop. One tiny single-head transformer encoder, implemented in
pure numpy (forward AND backward — no torch, no external framework),
trained on the system's OWN formula history: which formula shapes earned
confidence/evidence in the library, which got hurt.

Honest scope: this is NOT a language model. It never generates free-form
structure. It works generate-and-rank: the FormulaEngine's 7 grammar-safe
mutation operators generate candidates; this unit SCORES them, conditioned
on what the library history says good formulas look like. Grammar safety
is by construction; the neural part only ranks.

Distillation Round-Trip condition (§5.4): the discrete store stays the
ground truth. This unit:
- distills IN:  trained from versioned library cells (their trees + outcomes),
- distills OUT: `explain()` returns per-token preference scores readable by
  a human ("sigmoid +0.31, div −0.22"), and every proposal it emits is an
  ordinary MutationProposal (proposer="neural") that enters the normal
  proposal→evaluate→cell flow. Weights live in a readable JSON model card
  (results/uil_neural_proposer.json) with vocab, dims, and training stats.

LCA mapping (agent_architecture.md, dual-loop): this is the System-1 fast
proposer (<ms, offline, always available); an external LLM — when a key
exists — remains the optional System-2 slow path. See llm_mutator.py.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np

from figpso_uil.cognition.formula_engine import FormulaNode
from figpso_uil.cognition.paths import results_path

_MODEL_PATH = results_path("uil_neural_proposer.json")

# Token vocabulary is built from data plus these specials. Constants are
# bucketed (exact values would fragment the tiny vocab); feature leaves
# collapse to one token (which column matters less than shape).
_PAD = "[pad]"
_CLS = "[cls]"
_MAX_LEN = 48


def tokenize_tree(tree: dict | FormulaNode) -> list[str]:
    """Preorder token walk of a formula tree."""
    if isinstance(tree, FormulaNode):
        tree = tree.to_dict()
    out: list[str] = []

    def _walk(node: dict) -> None:
        if not isinstance(node, dict):
            return
        if "op" in node:
            out.append(str(node["op"]))
            for c in node.get("children", []):
                _walk(c)
        elif "var" in node:
            out.append(f"var:{node['var']}")
        elif "feature" in node:
            out.append("feat")
        elif "const" in node:
            c = float(node["const"])
            if c < 0:
                out.append("c_neg")
            elif c == 0:
                out.append("c_zero")
            elif c < 0.5:
                out.append("c_small")
            elif c <= 2.0:
                out.append("c_one")
            else:
                out.append("c_big")

    _walk(tree)
    return out[: _MAX_LEN - 1]


class TinyTransformer:
    """Single-head, single-block transformer encoder in pure numpy.

    [cls] + tokens → embed(+pos) → self-attention → FFN → mean-pool →
    sigmoid quality score. Every weight matrix is a plain array serialized
    to the JSON model card — nothing hidden.
    """

    def __init__(self, vocab: list[str], d: int = 16, h: int = 32, seed: int = 0):
        self.vocab = list(vocab)
        self.tok2id = {t: i for i, t in enumerate(self.vocab)}
        self.d, self.h = d, h
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d)
        self.E = rng.normal(0, s, (len(vocab), d))
        self.P = rng.normal(0, s, (_MAX_LEN, d))
        self.Wq = rng.normal(0, s, (d, d))
        self.Wk = rng.normal(0, s, (d, d))
        self.Wv = rng.normal(0, s, (d, d))
        self.W1 = rng.normal(0, s, (d, h))
        self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, s, (h, d))
        self.b2 = np.zeros(d)
        self.w_out = rng.normal(0, s, d)
        self.b_out = 0.0
        self.training_stats: dict[str, Any] = {}

    # ── forward / backward ───────────────────────────────────────────

    def _ids(self, tokens: list[str]) -> list[int]:
        unk = self.tok2id.get(_PAD, 0)
        ids = [self.tok2id.get(_CLS, 0)]
        ids += [self.tok2id.get(t, unk) for t in tokens]
        return ids[:_MAX_LEN]

    def _forward(self, ids: list[int]) -> tuple[float, dict]:
        L = len(ids)
        X = self.E[ids] + self.P[:L]
        Q, K, V = X @ self.Wq, X @ self.Wk, X @ self.Wv
        S = Q @ K.T / np.sqrt(self.d)
        S = S - S.max(axis=1, keepdims=True)
        A = np.exp(S)
        A = A / A.sum(axis=1, keepdims=True)
        H = X + A @ V
        Z1 = H @ self.W1 + self.b1
        R = np.maximum(Z1, 0.0)
        H2 = H + R @ self.W2 + self.b2
        p = H2.mean(axis=0)
        logit = float(p @ self.w_out + self.b_out)
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
        cache = {"ids": ids, "X": X, "Q": Q, "K": K, "V": V, "A": A,
                 "H": H, "Z1": Z1, "R": R, "H2": H2, "p": p, "prob": prob}
        return prob, cache

    def _backward(self, cache: dict, y: float, grads: dict) -> None:
        ids, X, A, V = cache["ids"], cache["X"], cache["A"], cache["V"]
        H, Z1, R, H2, p = cache["H"], cache["Z1"], cache["R"], cache["H2"], cache["p"]
        L = len(ids)
        dlogit = cache["prob"] - y
        grads["w_out"] += p * dlogit
        grads["b_out"] += dlogit
        dp = self.w_out * dlogit
        dH2 = np.tile(dp / L, (L, 1))
        dR = dH2 @ self.W2.T
        grads["W2"] += R.T @ dH2
        grads["b2"] += dH2.sum(axis=0)
        dZ1 = dR * (Z1 > 0)
        grads["W1"] += H.T @ dZ1
        grads["b1"] += dZ1.sum(axis=0)
        dH = dH2 + dZ1 @ self.W1.T
        dAV = dH
        dA = dAV @ V.T
        dV = A.T @ dAV
        # softmax rows backward
        dS = A * (dA - (dA * A).sum(axis=1, keepdims=True))
        dS /= np.sqrt(self.d)
        dQ = dS @ cache["K"]
        dK = dS.T @ cache["Q"]
        dX = dH + dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T
        grads["Wq"] += X.T @ dQ
        grads["Wk"] += X.T @ dK
        grads["Wv"] += X.T @ dV
        for row, tid in enumerate(ids):
            grads["E"][tid] += dX[row]
        grads["P"][:L] += dX

    _PARAMS = ("E", "P", "Wq", "Wk", "Wv", "W1", "b1", "W2", "b2", "w_out", "b_out")

    def train(
        self,
        samples: list[tuple[list[str], float]],
        epochs: int = 120,
        lr: float = 0.01,
        seed: int = 0,
        holdout: float = 0.25,
    ) -> dict:
        """Adam training on (tokens, label) pairs; returns stats."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(samples))
        n_hold = max(1, int(len(samples) * holdout))
        hold = [samples[i] for i in idx[:n_hold]]
        train = [samples[i] for i in idx[n_hold:]]

        m = {k: np.zeros_like(np.asarray(getattr(self, k), dtype=float)) for k in self._PARAMS}
        v = {k: np.zeros_like(np.asarray(getattr(self, k), dtype=float)) for k in self._PARAMS}
        t = 0
        for _epoch in range(epochs):
            rng.shuffle(train)
            grads = {k: np.zeros_like(np.asarray(getattr(self, k), dtype=float)) for k in self._PARAMS}
            for tokens, y in train:
                prob, cache = self._forward(self._ids(tokens))
                self._backward(cache, y, grads)
            t += 1
            for k in self._PARAMS:
                g = grads[k] / max(1, len(train))
                m[k] = 0.9 * m[k] + 0.1 * g
                v[k] = 0.999 * v[k] + 0.001 * g * g
                mh = m[k] / (1 - 0.9 ** t)
                vh = v[k] / (1 - 0.999 ** t)
                new = np.asarray(getattr(self, k), dtype=float) - lr * mh / (np.sqrt(vh) + 1e-8)
                setattr(self, k, float(new) if np.ndim(new) == 0 else new)

        def _acc(rows: list) -> float:
            if not rows:
                return 0.0
            hits = sum(
                1 for tok, y in rows
                if (self._forward(self._ids(tok))[0] >= 0.5) == (y >= 0.5)
            )
            return hits / len(rows)

        self.training_stats = {
            "n_train": len(train), "n_holdout": len(hold),
            "train_acc": round(_acc(train), 4),
            "holdout_acc": round(_acc(hold), 4),
            "epochs": epochs,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        return self.training_stats

    def score(self, tree: dict | FormulaNode) -> float:
        prob, _ = self._forward(self._ids(tokenize_tree(tree)))
        return float(prob)

    def explain(self) -> list[tuple[str, float]]:
        """Distill-out: per-token preference, model score of the lone token.

        Readable answer to "what did it learn": tokens (ops/leaf kinds)
        sorted by the quality the model assigns a formula made of just
        that token, centered on the [cls]-only baseline.
        """
        base, _ = self._forward(self._ids([]))
        out = []
        for tok in self.vocab:
            if tok in (_PAD, _CLS):
                continue
            prob, _ = self._forward(self._ids([tok]))
            out.append((tok, round(prob - base, 4)))
        out.sort(key=lambda x: -abs(x[1]))
        return out

    # ── open-box persistence ─────────────────────────────────────────

    def save(self, path: str = _MODEL_PATH) -> str:
        blob = {
            "kind": "tiny_transformer_v1",
            "vocab": self.vocab,
            "d": self.d, "h": self.h,
            "weights": {
                k: (getattr(self, k).tolist()
                    if hasattr(getattr(self, k), "tolist") else getattr(self, k))
                for k in self._PARAMS
            },
            "training_stats": self.training_stats,
            "explain": self.explain()[:15],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(blob, fh)
        return path

    @classmethod
    def load(cls, path: str = _MODEL_PATH) -> "TinyTransformer | None":
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                blob = json.load(fh)
            model = cls(blob["vocab"], d=blob["d"], h=blob["h"])
            for k, w in blob["weights"].items():
                setattr(model, k, np.asarray(w) if isinstance(w, list) else float(w))
            model.training_stats = blob.get("training_stats", {})
            return model
        except Exception:
            return None


# ── training-data extraction + the proposer facade ──────────────────


def build_training_samples(formulas: list[dict]) -> list[tuple[list[str], float]]:
    """(tree tokens, quality label) pairs from library formula cells.

    Label: helped-ratio with Laplace smoothing, thresholded at the corpus
    median — "does this formula shape belong to the better half of what
    the system has tried".
    """
    rows = []
    for f in formulas:
        tree = f.get("tree")
        if not tree:
            continue
        helped = float(f.get("times_helped", 0))
        hurt = float(f.get("times_hurt", 0))
        ratio = (helped + 1.0) / (helped + hurt + 2.0)
        conf = float(f.get("confidence", 0.5))
        quality = 0.5 * ratio + 0.5 * conf
        rows.append((tokenize_tree(tree), quality))
    if not rows:
        return []
    med = float(np.median([q for _, q in rows]))
    return [(tok, 1.0 if q >= med else 0.0) for tok, q in rows]


def build_vocab(samples: list[tuple[list[str], float]]) -> list[str]:
    vocab = {_PAD, _CLS}
    for tokens, _ in samples:
        vocab.update(tokens)
    return sorted(vocab)


class NeuralProposer:
    """Generate-and-rank mutation proposer backed by TinyTransformer."""

    def __init__(self, model: TinyTransformer):
        self.model = model

    @classmethod
    def load(cls) -> "NeuralProposer | None":
        model = TinyTransformer.load()
        return cls(model) if model is not None else None

    @classmethod
    def train_from_libraries(
        cls, formulas: list[dict], seed: int = 0,
    ) -> "NeuralProposer | None":
        samples = build_training_samples(formulas)
        if len(samples) < 20:
            return None  # not enough history yet — stay with engine fallback
        model = TinyTransformer(build_vocab(samples), seed=seed)
        model.train(samples, seed=seed)
        model.save()
        return cls(model)

    def rank_candidates(
        self, candidates: list[tuple[FormulaNode, str]],
    ) -> list[tuple[FormulaNode, str, float]]:
        scored = [
            (tree, mtype, self.model.score(tree)) for tree, mtype in candidates
        ]
        scored.sort(key=lambda x: -x[2])
        return scored
