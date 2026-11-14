"""LLM mutation proposal generator (v22 §5.2) — proposals only, never writes state."""

from __future__ import annotations

import json
import os
from typing import Any

from figpso_uil.cognition.knowledge_cell import MutationProposal
from figpso_uil.cognition.formula_engine import FormulaEngine, FormulaNode


class LLMMutator:
    """Generate MutationProposals from fingerprint + cells + residuals.

    Proposal sources, in LCA dual-loop order (agent_architecture.md):
    1. System 2 (slow, optional): external LLM when an API key exists
       AND ``offline_only`` is False.
    2. System 1 (fast, offline):  the own-transformer NeuralProposer
       (cognition/neural_proposer.py, R1 §5.4) — generate-and-rank over
       the formula grammar, trained on the system's own library history.
    3. Fallback: the formula engine's raw 7-mutation operators.

    Product default for formula *building* is offline trial-and-error
    (``cognition/offline_formula_search.py``) — this mutator is only a
    proposal helper. Pass ``offline_only=True`` to never touch the network
    even if ``OPENAI_API_KEY`` is set.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        offline_only: bool = False,
    ):
        self.offline_only = bool(offline_only)
        self.api_key = None if self.offline_only else (
            api_key
            or os.environ.get("UIL_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        self.model = model
        self._engine = FormulaEngine(seed=0)
        self._neural = None
        self._neural_checked = False

    @property
    def available(self) -> bool:
        """True when the external LLM (System 2) is reachable."""
        return (not self.offline_only) and bool(self.api_key)

    @property
    def has_proposer(self) -> bool:
        """True when ANY smart proposer exists — external LLM (System 2)
        or the offline own-transformer (System 1). Callers should gate on
        this, not `available`, so the neural path participates offline."""
        return bool(self.api_key) or self._neural_proposer() is not None

    def _neural_proposer(self):
        if not self._neural_checked:
            self._neural_checked = True
            try:
                from figpso_uil.cognition.neural_proposer import NeuralProposer
                self._neural = NeuralProposer.load()
            except Exception:
                self._neural = None
        return self._neural

    def propose(
        self,
        *,
        fingerprint: dict,
        cells: list[dict],
        residuals: dict | None = None,
        n_proposals: int = 3,
    ) -> list[MutationProposal]:
        if self.available:
            try:
                return self._propose_llm(
                    fingerprint=fingerprint,
                    cells=cells,
                    residuals=residuals,
                    n_proposals=n_proposals,
                )
            except Exception:
                pass
        neural = self._neural_proposer()
        if neural is not None:
            try:
                return self._propose_neural(neural, n_proposals=n_proposals)
            except Exception:
                pass
        return self._propose_engine(n_proposals=n_proposals)

    def _propose_neural(self, neural, n_proposals: int = 3) -> list[MutationProposal]:
        """Generate-and-rank: engine mutates (grammar-safe), model ranks."""
        seeds = self._engine.get_seed_formulas()
        parents = list(seeds.values()) or [FormulaNode.const(0.5)]
        candidates = []
        for i in range(max(n_proposals * 4, 8)):
            parent = parents[i % len(parents)]
            mutant, mtype = self._engine.mutate(parent, stress_level=0.5)
            ok, _ = self._engine.validate(mutant)
            if ok:
                candidates.append((mutant, mtype))
        ranked = neural.rank_candidates(candidates)
        return [
            MutationProposal(
                op="formula_mutate",
                payload={"tree": tree.to_dict()},
                rationale=f"neural rank {score:.3f} ({mtype})",
                proposer="neural",
            )
            for tree, mtype, score in ranked[:n_proposals]
        ]

    def _propose_engine(self, n_proposals: int = 3) -> list[MutationProposal]:
        seeds = self._engine.get_seed_formulas()
        trees = list(seeds.values()) or [FormulaNode.const(0.5)]
        proposals = []
        for i in range(n_proposals):
            tree = trees[i % len(trees)]
            mutant, _mt = self._engine.mutate(tree, stress_level=0.5)
            proposals.append(MutationProposal(
                op="formula_mutate",
                payload={"tree": mutant.to_dict()},
                rationale="formula_engine offline mutation",
                proposer="engine",
            ))
        return proposals

    def propose_goal(
        self,
        *,
        fingerprint: dict,
        residuals: dict | None = None,
        current_goal: dict | None = None,
    ) -> MutationProposal | None:
        """Propose a GoalCell payload — the LCA-inspired top-down director
        (2026-07-18). Never writes state; the caller (optimizer / HITL /
        webui) decides whether to install the proposal via
        goal_cell.record_goal_event() and its GoalStore.

        Uses a small heuristic ruleset over fingerprint + residuals so the
        proposer is useful offline; when the external LLM is available the
        heuristic is replaced by an LLM call with the same schema.
        """
        payload = self._heuristic_goal_payload(fingerprint, residuals, current_goal)
        if self.available:
            try:
                llm_payload = self._llm_goal_payload(
                    fingerprint=fingerprint,
                    residuals=residuals,
                    current_goal=current_goal,
                )
                if llm_payload:
                    payload = llm_payload
            except Exception:
                pass
        if payload is None:
            return None
        proposer = "llm" if self.available else "engine"
        return MutationProposal(
            op="set_goal",
            payload=payload,
            rationale=payload.pop("_rationale", "auto-generated goal proposal"),
            proposer=proposer,
        )

    @staticmethod
    def _heuristic_goal_payload(
        fingerprint: dict,
        residuals: dict | None,
        current_goal: dict | None,
    ) -> dict | None:
        """Small ruleset: pick a directive from fingerprint shape / residual
        signals. Deliberately conservative — a goal is a directive, not a
        theory, so we prefer producing None to producing a wrong one."""
        residuals = residuals or {}
        fp = fingerprint or {}
        # Feature-count / dimensionality signals
        n_features = int(fp.get("n_features", 0))
        sparsity = float(fp.get("sparsity", 0.0))
        smoothness = float(fp.get("smoothness", 0.5))
        # If we already have a live goal and nothing looks urgent, don't
        # override it — top-down direction should be sticky.
        already_active = bool(current_goal and current_goal.get("status") == "active")

        directive = None
        payload: dict = {
            "directive": "",
            "prefer_domains": [],
            "focus_features": [],
            "ignore_features": [],
            "prefer_relations": [],
            "goal_keywords": [],
            "confidence": 0.6,
        }

        if smoothness >= 0.7 and n_features <= 8:
            directive = "prefer readable symbolic model over marginal accuracy"
            payload["prefer_domains"] = ["symbolic", "symbolic_regression"]
            payload["prefer_relations"] = ["strengthens", "leads_to"]
            payload["goal_keywords"] = ["symbolic", "readable"]
        elif sparsity >= 0.6 and n_features >= 32:
            directive = "focus on the few features that carry signal; discount the sparse mass"
            payload["prefer_domains"] = ["sparse_linear"]
            payload["goal_keywords"] = ["sparse", "L1"]

        if directive is None or already_active:
            return None
        payload["directive"] = directive
        payload["_rationale"] = (
            f"heuristic goal from fingerprint("
            f"n_features={n_features}, sparsity={sparsity:.2f}, "
            f"smoothness={smoothness:.2f})"
        )
        return payload

    def _llm_goal_payload(
        self,
        *,
        fingerprint: dict,
        residuals: dict | None,
        current_goal: dict | None,
    ) -> dict | None:
        import urllib.request
        prompt = {
            "task": "propose a single GoalCell payload",
            "fingerprint": fingerprint,
            "residuals": residuals or {},
            "current_goal": current_goal or {},
            "schema": {
                "directive": "one-line human-readable directive",
                "prefer_domains": "list of strategy names",
                "focus_features": "list of feature names/indices",
                "ignore_features": "list of feature names/indices",
                "prefer_relations": "list of edge relations",
                "goal_keywords": "list of tag strings",
                "confidence": "float in [0, 1]",
            },
        }
        body = json.dumps({
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a UIL goal proposer. Return JSON matching "
                        "the schema. Prefer no directive over a bad one — "
                        "return null when uncertain."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt)},
            ],
            "temperature": 0.2,
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        text = data["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or not parsed.get("directive"):
            return None
        parsed.setdefault("confidence", 0.6)
        parsed["_rationale"] = "llm-proposed goal"
        return parsed

    def _propose_llm(
        self,
        *,
        fingerprint: dict,
        cells: list[dict],
        residuals: dict | None,
        n_proposals: int,
    ) -> list[MutationProposal]:
        import urllib.request

        readable = [c.get("readable", c.get("summary", str(c)[:200])) for c in cells[:8]]
        prompt = {
            "task": "propose formula or principle mutations",
            "fingerprint": fingerprint,
            "cells": readable,
            "residuals": residuals or {},
            "n": n_proposals,
        }
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return JSON list of mutation proposals only."},
                {"role": "user", "content": json.dumps(prompt)},
            ],
            "temperature": 0.3,
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        text = data["choices"][0]["message"]["content"]
        raw = json.loads(text)
        out: list[MutationProposal] = []
        for item in raw[:n_proposals]:
            out.append(MutationProposal(
                op=item.get("op", "mutate"),
                payload=item.get("payload", {}),
                rationale=item.get("rationale", ""),
                proposer="llm",
            ))
        return out
