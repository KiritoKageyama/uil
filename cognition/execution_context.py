"""Execution Context — the shared input the Principle Executor and the
Editable Network both see when deciding overrides for the upcoming stage.

V20+.12: UIL_Executor.get_overrides() and EditableNetwork.traverse()/
traverse_with_attention() were both only ever given the dataset-state
dict (_build_state()'s output). But the Miner mines rules over THREE
namespaces -- state.*, action.*, and config.* (see uil_miner.py's
_build_matrix(), which flattens action["classifier_config"] into
"config.<key>" features and action's own numeric keys into
"action.<key>" features). Neither consumer ever received the
action/config namespaces, so any mined rule conditioned on them could
never match -- not a sparsity problem, a missing-input problem.

ExecutionContext is the one object both consumers now take. Its
to_flat_dict() produces the same flat, prefixed-key shape the Miner
mined against, so the unchanged NetworkNode._resolve() /
UIL_Executor._resolve_value() prefix-aware lookups just work.

Scope note: only `previous_action`/`previous_config` are populated by
the optimizer -- the action/config that WON the immediately preceding
stage's candidate competition (the only action/config data causally
available before THIS stage's own action is chosen; the current stage's
candidates don't exist yet at the point overrides are computed). A
future pass could evaluate principles/nodes per-candidate inside the
action-scoring loop instead (so a rule could ask "would THIS proposed
action be good," not just "was the last one") -- that needs the
override computation itself restructured to run once per candidate
rather than once per stage, which is out of scope here. current_action/
current_config exist as fields for that future pass and take priority
over previous_* if ever populated.
"""

from __future__ import annotations


class ExecutionContext:
    def __init__(
        self,
        state: dict,
        previous_action: dict | None = None,
        previous_config: dict | None = None,
        current_action: dict | None = None,
        current_config: dict | None = None,
        stress: float | None = None,
        regret: float | None = None,
        stage: int | None = None,
    ):
        self.state = state
        self.previous_action = previous_action or {}
        self.previous_config = previous_config or {}
        # Not populated by the optimizer yet -- see module docstring.
        self.current_action = current_action or {}
        self.current_config = current_config or {}
        self.stress = stress
        self.regret = regret
        self.stage = stage

    def to_flat_dict(self) -> dict:
        """Flatten into the namespaced shape rules were mined against:
        state keys as-is, action/config keys prefixed "action."/"config."
        so conditions like "config.max_depth" resolve via direct lookup.

        Only numeric values are carried over for action/config (mirrors
        uil_miner.py's own isinstance(..., (int, float)) feature filter --
        a non-numeric value like "description" was never a mineable
        feature, so it can't be a condition either).
        """
        merged = dict(self.state)
        for key, value in self.previous_action.items():
            if isinstance(value, (int, float)):
                merged[f"action.{key}"] = value
        for key, value in self.previous_config.items():
            if isinstance(value, (int, float)):
                merged[f"config.{key}"] = value
        # current_* takes priority when populated (future per-candidate pass).
        for key, value in self.current_action.items():
            if isinstance(value, (int, float)):
                merged[f"action.{key}"] = value
        for key, value in self.current_config.items():
            if isinstance(value, (int, float)):
                merged[f"config.{key}"] = value
        if self.stress is not None:
            merged["stress"] = self.stress
            merged["system_stress"] = self.stress
        if self.regret is not None:
            merged["regret"] = self.regret
        if self.stage is not None:
            merged["stage"] = self.stage
        return merged


def flatten_context(context: "ExecutionContext | dict") -> dict:
    """Normalize either an ExecutionContext or a plain state dict (the
    back-compat path every existing test and direct caller still uses)
    into the flat dict NetworkNode._resolve()/UIL_Executor._resolve_value()
    already know how to read.
    """
    if isinstance(context, ExecutionContext):
        return context.to_flat_dict()
    return context
