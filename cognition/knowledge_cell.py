"""KnowledgeCell protocol — v22 §1.1 smallest self-modifiable knowledge unit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Provenance:
    journal_ids: list[str] = field(default_factory=list)
    trial_ids: list[str] = field(default_factory=list)
    fold_spread: float | None = None
    human_editor: str | None = None
    source: str = "mined"

    def to_dict(self) -> dict:
        return {
            "journal_ids": self.journal_ids,
            "trial_ids": self.trial_ids,
            "fold_spread": self.fold_spread,
            "human_editor": self.human_editor,
            "source": self.source,
        }


@dataclass
class MutationProposal:
    op: str
    payload: dict
    rationale: str = ""
    proposer: str = "engine"


@dataclass
class CellVerdict:
    applies: bool
    confidence: float
    action_override: dict = field(default_factory=dict)
    message: str = ""


@dataclass
class StructuralForm:
    encoding: str
    vocabulary: str = "ACTION/CONTEXT/OUTCOME"

    def to_dict(self) -> dict:
        return {"encoding": self.encoding, "vocabulary": self.vocabulary}


@runtime_checkable
class KnowledgeCell(Protocol):
    cell_id: str
    cell_type: str
    version: int
    provenance: Provenance
    confidence: float

    def evaluate(self, state: dict) -> CellVerdict: ...
    def mutate(self, proposal: MutationProposal) -> "KnowledgeCell": ...
    def compose(self, other: "KnowledgeCell") -> "KnowledgeCell | None": ...
    def to_readable(self) -> str: ...
    def to_structural(self) -> StructuralForm: ...


@dataclass
class PrincipleCell:
    """Thin wrapper over a principle dict."""

    cell_id: str
    cell_type: str = "principle"
    version: int = 1
    provenance: Provenance = field(default_factory=Provenance)
    confidence: float = 0.5
    conditions: list = field(default_factory=list)
    action_override: dict = field(default_factory=dict)
    predicted_regret: float = 0.0

    @classmethod
    def from_dict(cls, d: dict, cell_id: str | None = None) -> "PrincipleCell":
        return cls(
            cell_id=cell_id or d.get("principle_id") or d.get("id", d.get("cell_id", "P-?")),
            confidence=float(d.get("confidence", 0.5)),
            conditions=list(d.get("conditions", [])),
            action_override=dict(d.get("action_override", {})),
            predicted_regret=float(d.get("predicted_regret", 0.0)),
            version=int(d.get("version", 1)),
        )

    def evaluate(self, state: dict) -> CellVerdict:
        from figpso_uil.cognition.uil_executor import UIL_Executor

        ex = UIL_Executor()
        overrides = ex.get_overrides([{
            "conditions": self.conditions,
            "action_override": self.action_override,
            "confidence": self.confidence,
        }], state)
        applies = bool(overrides)
        return CellVerdict(
            applies=applies,
            confidence=self.confidence if applies else 0.0,
            action_override=overrides,
            message=f"principle {self.cell_id}",
        )

    def mutate(self, proposal: MutationProposal) -> "PrincipleCell":
        new = PrincipleCell(
            cell_id=self.cell_id,
            version=self.version + 1,
            provenance=Provenance(
                source=proposal.proposer,
                human_editor=proposal.proposer if proposal.proposer == "human" else None,
            ),
            confidence=float(proposal.payload.get("confidence", self.confidence)),
            conditions=list(proposal.payload.get("conditions", self.conditions)),
            action_override=dict(proposal.payload.get("action_override", self.action_override)),
            predicted_regret=float(proposal.payload.get("predicted_regret", self.predicted_regret)),
        )
        return new

    def compose(self, other: KnowledgeCell) -> "PrincipleCell | None":
        if not isinstance(other, PrincipleCell):
            return None
        if self.conditions != other.conditions:
            return None
        merged = dict(self.action_override)
        merged.update(other.action_override)
        return PrincipleCell(
            cell_id=self.cell_id,
            version=max(self.version, other.version) + 1,
            confidence=max(self.confidence, other.confidence),
            conditions=self.conditions,
            action_override=merged,
            predicted_regret=max(self.predicted_regret, other.predicted_regret),
        )

    def to_readable(self) -> str:
        conds = " AND ".join(
            f"{c.get('feature')}{c.get('op')}{c.get('threshold')}" for c in self.conditions
        )
        return f"IF {conds} THEN {self.action_override} (conf={self.confidence:.2f})"

    def to_structural(self) -> StructuralForm:
        return StructuralForm(encoding=self.to_readable())

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "cell_type": self.cell_type,
            "version": self.version,
            "confidence": self.confidence,
            "conditions": self.conditions,
            "action_override": self.action_override,
            "predicted_regret": self.predicted_regret,
            "provenance": self.provenance.to_dict(),
        }


@dataclass
class FormulaCell:
    """Readable wrapper over a formula library entry."""

    cell_id: str
    cell_type: str = "formula"
    version: int = 1
    provenance: Provenance = field(default_factory=Provenance)
    confidence: float = 0.5
    name: str = ""
    tree: dict | None = None
    status: str = "active"
    mutation_type: str = ""
    best_accuracy: float = 0.0

    @classmethod
    def from_dict(cls, d: dict, cell_id: str | None = None) -> "FormulaCell":
        return cls(
            cell_id=cell_id or d.get("formula_id", "F-?"),
            confidence=float(d.get("confidence", 0.5)),
            name=str(d.get("name", "")),
            tree=d.get("tree"),
            status=str(d.get("status", "active")),
            mutation_type=str(d.get("mutation_type", "")),
            best_accuracy=float(d.get("best_accuracy", 0.0)),
            version=int(d.get("version", 1)),
        )

    def evaluate(self, state: dict) -> CellVerdict:
        return CellVerdict(
            applies=self.status in ("active", "incubating", "seed"),
            confidence=self.confidence,
            message=f"formula {self.cell_id}",
        )

    def mutate(self, proposal: MutationProposal) -> "FormulaCell":
        return FormulaCell(
            cell_id=self.cell_id,
            version=self.version + 1,
            provenance=Provenance(
                source=proposal.proposer,
                human_editor=proposal.proposer if proposal.proposer == "human" else None,
            ),
            confidence=float(proposal.payload.get("confidence", self.confidence)),
            name=str(proposal.payload.get("name", self.name)),
            tree=proposal.payload.get("tree", self.tree),
            status=str(proposal.payload.get("status", self.status)),
            mutation_type=self.mutation_type,
            best_accuracy=float(proposal.payload.get("best_accuracy", self.best_accuracy)),
        )

    def compose(self, other: KnowledgeCell) -> "FormulaCell | None":
        return None

    def to_readable(self) -> str:
        from figpso_uil.cognition.formula_engine import FormulaNode

        expr = "?"
        if self.tree:
            try:
                expr = repr(FormulaNode.from_dict(self.tree))
            except Exception:
                expr = str(self.tree)[:80]
        return (
            f"FORMULA {self.name or self.cell_id}: y ~ {expr} "
            f"(conf={self.confidence:.2f}, status={self.status}, "
            f"best_acc={self.best_accuracy:.2f})"
        )

    def to_structural(self) -> StructuralForm:
        return StructuralForm(encoding=self.to_readable())

    def to_dict(self) -> dict:
        return {
            "formula_id": self.cell_id,
            "cell_type": self.cell_type,
            "name": self.name,
            "tree": self.tree,
            "confidence": self.confidence,
            "status": self.status,
            "mutation_type": self.mutation_type,
            "best_accuracy": self.best_accuracy,
            "version": self.version,
        }


@dataclass
class HypothesisCell:
    """Readable wrapper over a causal hypothesis."""

    cell_id: str
    cell_type: str = "causal_hypothesis"
    version: int = 1
    provenance: Provenance = field(default_factory=Provenance)
    confidence: float = 0.0
    variable: str = ""
    direction: str = ""
    threshold: float = 0.0
    status: str = "pending"
    conclusion: str = ""
    causal_strength: float = 0.0

    @classmethod
    def from_dict(cls, d: dict, cell_id: str | None = None) -> "HypothesisCell":
        return cls(
            cell_id=cell_id or d.get("hypothesis_id", "H-?"),
            confidence=float(d.get("causal_strength", d.get("confidence", 0.0))),
            variable=str(d.get("variable", "")),
            direction=str(d.get("direction", "")),
            threshold=float(d.get("threshold", 0.0)),
            status=str(d.get("status", "pending")),
            conclusion=str(d.get("conclusion", "")),
            causal_strength=float(d.get("causal_strength", 0.0)),
            version=int(d.get("version", 1)),
        )

    def evaluate(self, state: dict) -> CellVerdict:
        applies = self.status in ("pending", "testing", "confirmed_causal", "correlational_only")
        return CellVerdict(
            applies=applies,
            confidence=self.causal_strength,
            message=self.conclusion or f"hypothesis {self.variable} {self.direction}",
        )

    def mutate(self, proposal: MutationProposal) -> "HypothesisCell":
        return HypothesisCell(
            cell_id=self.cell_id,
            version=self.version + 1,
            provenance=Provenance(
                source=proposal.proposer,
                human_editor=proposal.proposer if proposal.proposer == "human" else None,
            ),
            confidence=float(proposal.payload.get("causal_strength", self.causal_strength)),
            variable=str(proposal.payload.get("variable", self.variable)),
            direction=str(proposal.payload.get("direction", self.direction)),
            threshold=float(proposal.payload.get("threshold", self.threshold)),
            status=str(proposal.payload.get("status", self.status)),
            conclusion=str(proposal.payload.get("conclusion", self.conclusion)),
            causal_strength=float(proposal.payload.get("causal_strength", self.causal_strength)),
        )

    def compose(self, other: KnowledgeCell) -> "HypothesisCell | None":
        return None

    def to_readable(self) -> str:
        return (
            f"HYPOTHESIS {self.cell_id}: {self.variable} is {self.direction} "
            f"(thr={self.threshold}, status={self.status}, "
            f"strength={self.causal_strength:.2f}) — {self.conclusion[:80]}"
        )

    def to_structural(self) -> StructuralForm:
        return StructuralForm(encoding=self.to_readable())

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.cell_id,
            "cell_type": self.cell_type,
            "variable": self.variable,
            "direction": self.direction,
            "threshold": self.threshold,
            "status": self.status,
            "conclusion": self.conclusion,
            "causal_strength": self.causal_strength,
            "version": self.version,
        }


@dataclass
class EdgeCell:
    """Readable wrapper over a knowledge-graph edge."""

    cell_id: str
    cell_type: str = "edge"
    version: int = 1
    provenance: Provenance = field(default_factory=Provenance)
    confidence: float = 0.5
    source: str = ""
    target: str = ""
    relation: str = ""
    weight: float = 0.0

    @classmethod
    def from_dict(cls, d: dict, cell_id: str | None = None) -> "EdgeCell":
        src = d.get("source", d.get("from", ""))
        tgt = d.get("target", d.get("to", ""))
        rel = d.get("relation", d.get("type", "leads_to"))
        eid = cell_id or d.get("edge_id") or f"{src}->{tgt}:{rel}"
        return cls(
            cell_id=eid,
            confidence=float(d.get("confidence", d.get("weight", 0.5))),
            source=str(src),
            target=str(tgt),
            relation=str(rel),
            weight=float(d.get("weight", 0.0)),
            version=int(d.get("version", 1)),
        )

    def evaluate(self, state: dict) -> CellVerdict:
        return CellVerdict(applies=True, confidence=self.confidence, message=self.to_readable())

    def mutate(self, proposal: MutationProposal) -> "EdgeCell":
        return EdgeCell(
            cell_id=self.cell_id,
            version=self.version + 1,
            provenance=Provenance(source=proposal.proposer),
            confidence=float(proposal.payload.get("confidence", self.confidence)),
            source=str(proposal.payload.get("source", self.source)),
            target=str(proposal.payload.get("target", self.target)),
            relation=str(proposal.payload.get("relation", self.relation)),
            weight=float(proposal.payload.get("weight", self.weight)),
        )

    def compose(self, other: KnowledgeCell) -> "EdgeCell | None":
        return None

    def to_readable(self) -> str:
        return (
            f"EDGE {self.source} --[{self.relation} w={self.weight:.2f}]--> "
            f"{self.target} (conf={self.confidence:.2f})"
        )

    def to_structural(self) -> StructuralForm:
        return StructuralForm(encoding=self.to_readable())

    def to_dict(self) -> dict:
        return {
            "edge_id": self.cell_id,
            "cell_type": self.cell_type,
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
            "confidence": self.confidence,
            "version": self.version,
        }


def cell_to_readable(cell_type: str, post: dict, cell_id: str) -> str:
    """Factory: render any cell bucket entry as human text."""
    def _goal():
        from figpso_uil.cognition.goal_cell import GoalCell
        return GoalCell.from_dict(post, cell_id).to_readable()

    factories = {
        "principle": lambda: PrincipleCell.from_dict(post, cell_id).to_readable(),
        "routing_rule": lambda: PrincipleCell.from_dict(post, cell_id).to_readable(),
        "formula": lambda: FormulaCell.from_dict(post, cell_id).to_readable(),
        "causal_hypothesis": lambda: HypothesisCell.from_dict(post, cell_id).to_readable(),
        "edge": lambda: EdgeCell.from_dict(post, cell_id).to_readable(),
        "node": lambda: PrincipleCell.from_dict(
            {
                "principle_id": cell_id,
                "conditions": post.get("conditions", []),
                "action_override": post.get("action_override", {}),
                "confidence": post.get("confidence", 0.5),
            },
            cell_id,
        ).to_readable(),
        "episode": lambda: (
            f"EPISODE strategy={post.get('resolution', {}).get('strategy', '?')} "
            f"score={post.get('resolution', {}).get('score', '?')}"
        ),
        "goal": _goal,
    }
    fn = factories.get(cell_type)
    if fn is None:
        return str(post)[:200]
    try:
        return fn()
    except Exception:
        return str(post)[:200]
