"""Clean bootstrap knowledge for UIL + SymbolicFormula (Path A).

Seeds governance manifest + hypothesis principles + KG nodes without
polluting the journal with fake training rows.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from figpso_uil.cognition.paths import results_path

GOVERNANCE_MANIFEST = {
    "version": 1,
    "updated_at": None,
    "journal_rules": [
        "Journal = numeric facts only: state, action, outcome, stress, regret.",
        "Never store English lessons, hypotheses, or formula names in the journal.",
        "One row per stage per fold — raw observations for the Miner only.",
        "Regret investigations link to the journal entry that triggered them.",
    ],
    "principle_rules": [
        "Principles = testable hypotheses: conditions + action_override.",
        "Mined principles need evidence; seed principles start neutral (0.5).",
        "Retire principles that hurt more than they help after 10+ trials.",
        "Dataset-specific principles must generalize or stay tier-3 until promoted.",
    ],
    "symbolic_formula_rules": [
        "Formula tournament: evaluate on holdout before selecting a tree.",
        "Prefer boosting on hard/small data; hybrid multi on medium easy binary.",
        "Graph-activated features get weight in formula tournament scoring.",
        "Record formula outcomes per stage — do not reuse polluted library scores.",
    ],
    "path_a_curriculum": [
        "1) real_breast_cancer → target 0.80, boosting or multi, 40–80 stages",
        "2) real_spambase → boosting, 80–120 stages",
        "3) real_sonar → boosting only, 200+ stages, fresh symbolic library",
    ],
}

SEED_PRINCIPLES: list[dict] = [
    {
        "conditions": [{"feature": "state.n_samples", "op": "<", "threshold": 80}],
        "action_override": {"max_penalty": 1.4},
        "description": "GUIDE small-n: journal logs feat_ratio; explore wider feature weights.",
    },
    {
        "conditions": [{"feature": "state.feat_ratio", "op": ">", "threshold": 0.5}],
        "action_override": {"max_penalty": 1.2},
        "description": "GUIDE high-dim: cap penalty; principles should suggest sparsity not noise.",
    },
    {
        "conditions": [{"feature": "state.entropy", "op": "<", "threshold": 0.15}],
        "action_override": {"min_penalty": 0.8},
        "description": "GUIDE low entropy: journal state.entropy only; do not invent text causes.",
    },
    {
        "conditions": [{"feature": "decision_stress", "op": ">", "threshold": 0.45}],
        "action_override": {"max_penalty": 2.0},
        "description": "GUIDE stress: graph traversal fires; mine principles from regret not guesses.",
    },
    {
        "conditions": [{"feature": "regret", "op": ">", "threshold": 0.08}],
        "action_override": {"max_penalty": 1.5},
        "description": "GUIDE regret: investigation before new principles; journal holds the numbers.",
    },
    {
        "conditions": [{"feature": "state.num_classes", "op": ">", "threshold": 2}],
        "action_override": {"max_penalty": 1.6},
        "description": "GUIDE multiclass: SymbolicFormula OVR; journal logs num_classes.",
    },
    {
        "conditions": [{"feature": "outcome.accuracy", "op": "<", "threshold": 0.65}],
        "action_override": {"max_penalty": 2.2},
        "description": "GUIDE low acc: tournament more formulas; graph picks diverse trees.",
    },
    {
        "conditions": [{"feature": "state.sparsity", "op": ">", "threshold": 0.6}],
        "action_override": {"max_penalty": 1.3},
        "description": "GUIDE sparse data: prefer method-template seeds in symbolic library.",
    },
]


def governance_filepath(directory: str | None = None) -> str:
    if directory:
        return os.path.join(directory, "uil_governance_manifest.json")
    return results_path("uil_governance_manifest.json")


def save_governance_manifest(directory: str | None = None) -> str:
    path = governance_filepath(directory)
    manifest = dict(GOVERNANCE_MANIFEST)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def load_governance_manifest() -> dict:
    path = governance_filepath()
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_bootstrap(
    library,
    network,
    *,
    symbolic_library=None,
    symbolic_engine=None,
    force: bool = False,
    state_dir: str | None = None,
) -> dict:
    """Seed clean principles + KG + governance. Idempotent unless force=True."""
    gov_path = save_governance_manifest(state_dir)
    existing = library.get_principles() if hasattr(library, "get_principles") else []
    has_seeds = any(p.get("source") == "seed" for p in existing)

    principles_added = 0
    if force or not has_seeds:
        if hasattr(library, "accept_seed_principles"):
            principles_added = library.accept_seed_principles(SEED_PRINCIPLES)

    if network is not None and hasattr(network, "import_from_principles"):
        all_p = library.get_principles() if hasattr(library, "get_principles") else []
        if all_p:
            network.import_from_principles(all_p)

    symbolic_reset = False
    if symbolic_library is not None and symbolic_engine is not None and force:
        if symbolic_library.formula_count() > 40:
            symbolic_library._formulas = {}
            symbolic_library._save()
            symbolic_reset = True

    return {
        "principles_added": principles_added,
        "governance_path": gov_path,
        "symbolic_library_reset": symbolic_reset,
        "seed_principle_count": len(SEED_PRINCIPLES),
    }
