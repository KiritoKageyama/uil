"""Shared comparison for mined-rule conditions.

This logic existed in THREE places -- `uil_library._conditions_similar`,
`editable_network._conditions_similar` and
`journal_forgetting._conditions_overlap` -- and all three had the same
defect at the same time: each did `abs()` arithmetic on a threshold that
the v22 routing rules (`dataset.name ==`, `route.domain ==`) fill with a
STRING, so any run after the first routing rule was persisted died with
`TypeError: bad operand type for abs(): 'str'`. That is bug #17, and
fixing it meant finding and patching three copies.

The copies are the hazard, not the arithmetic: a fourth caller would
have reintroduced the same crash, and a future change to what "similar"
means has to be made three times to be made at all. Everything that
compares conditions now routes through here.

The two comparison MODES are preserved exactly as they were, because
they were never the same function:

  "reference" -- used by the library and the network, which compare a
      candidate against an existing principle. The tolerance is relative
      to the EXISTING threshold `a`, which is the reference point.

  "max" -- used by journal forgetting, which asks whether two conditions
      overlap. Neither side is a reference, so the denominator is the
      larger magnitude and the relation stays symmetric.

Unifying them into one rule would silently change which principles merge
and which journal entries are treated as redundant, so they stay
distinct and named.
"""

from __future__ import annotations

# Relative tolerance used by the library/network "similar" check.
DEFAULT_SIMILARITY_TOLERANCE = 0.10


def is_numeric(value) -> bool:
    """True for real numbers only.

    `bool` is excluded deliberately. It is a subclass of `int`, so a
    boolean gate (`use_rule_basis`) would otherwise be compared with
    arithmetic meant for continuous thresholds.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def thresholds_match(
    a,
    b,
    tolerance: float = DEFAULT_SIMILARITY_TOLERANCE,
    *,
    mode: str = "reference",
) -> bool:
    """Are two condition thresholds close enough to be the same rule?

    Non-numeric thresholds (routing-rule dataset/domain names) compare by
    exact equality -- there is no "within 10%" of the string "wine".
    This is the branch whose absence was bug #17.
    """
    if not (is_numeric(a) and is_numeric(b)):
        return a == b

    if mode == "max":
        denominator = max(abs(a), abs(b), 1e-6)
        return abs(a - b) / denominator < tolerance

    # mode == "reference": relative to `a`, with an explicit zero case --
    # when the reference is ~0 there is no scale to be relative to, so
    # only a `b` that is also ~0 counts as the same threshold.
    if abs(a) > 1e-6:
        return abs(a - b) / abs(a) <= tolerance
    return abs(b) <= 1e-6


# The feature names that ARE the penalty knob (exact match, never a
# substring test -- see sanitize_overrides).
PENALTY_FEATURES = frozenset({"penalty", "action.penalty"})

_PENALTY_OVERRIDE_KEYS = ("max_penalty", "min_penalty", "penalty_avoid")
_CONFIG_OVERRIDE_PREFIXES = ("config_max.", "config_min.", "config_avoid.")


def sanitize_overrides(entry: dict) -> bool:
    """Drop overrides that none of this rule's conditions supports.

    A range check cannot catch the corrupt data this exists for:
    `max_penalty: 0.0003` sits inside penalty's legitimate [0, 10]
    domain and only looks wrong once you notice the rule never mentions
    penalty -- its conditions are all on DERIVED products that merely
    contain the word ("penalty" in feature used to match
    `derived.action.penalty*config.mutation_rounds_per_fit`). So the
    test is PROVENANCE, not magnitude: an override survives only if some
    condition names the knob it constrains.

    Exact rather than heuristic, and it repairs state already on disk --
    18 of the 300 nodes in the live graph, plus their source principles.
    Rules minted after the generator fix are unaffected, so this is a
    no-op on healthy data. Returns True when something was removed.
    """
    override = entry.get("action_override")
    if not isinstance(override, dict) or not override:
        return False
    features = {c.get("feature", "") for c in entry.get("conditions", []) or []}
    has_penalty = bool(features & PENALTY_FEATURES)
    configs = {f.split(".", 1)[1] for f in features if f.startswith("config.")}

    drop = []
    for key in override:
        if key in _PENALTY_OVERRIDE_KEYS:
            if not has_penalty:
                drop.append(key)
        elif key.startswith(_CONFIG_OVERRIDE_PREFIXES):
            if key.split(".", 1)[1] not in configs:
                drop.append(key)
    for key in drop:
        override.pop(key, None)
    return bool(drop)


def condition_sets_similar(
    conds_a: list[dict],
    conds_b: list[dict],
    tolerance: float = DEFAULT_SIMILARITY_TOLERANCE,
) -> bool:
    """Do two condition sets describe the same rule?

    Same length, same features, same operators, and thresholds within
    tolerance. Sorted by feature name first so ordering does not matter.
    """
    if len(conds_a) != len(conds_b):
        return False

    for a, b in zip(
        sorted(conds_a, key=lambda c: c.get("feature", "")),
        sorted(conds_b, key=lambda c: c.get("feature", "")),
    ):
        if a.get("feature") != b.get("feature"):
            return False
        if a.get("op") != b.get("op"):
            return False
        if not thresholds_match(
            a.get("threshold", 0), b.get("threshold", 0),
            tolerance, mode="reference",
        ):
            return False
    return True
