"""Lab report generator (v22 §6.1) — compiled from VersionStore event stream."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from figpso_uil.cognition.paths import results_path
from figpso_uil.cognition.version_store import VersionStore


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hypotheses_section() -> list[str]:
    """The scientist's experiments: tested hypotheses + their conclusions.

    Reads the causal store directly (not the event stream) so the report
    always reflects the CURRENT verdict of every hypothesis that has at
    least one control/treatment trial, plus how many are still waiting.
    """
    import json

    try:
        with open(results_path("uil_causal_v20.json"), encoding="utf-8") as fh:
            hypotheses = json.load(fh)
    except (OSError, ValueError):
        return []
    if not isinstance(hypotheses, list) or not hypotheses:
        return []

    tested = [h for h in hypotheses if h.get("evidence_trials")]
    untested = len(hypotheses) - len(tested)
    lines = ["## Hypotheses & experiments", ""]
    lines.append(
        f"- {len(hypotheses)} hypotheses held; {len(tested)} tested "
        f"in control/treatment trials; {untested} awaiting trials"
    )
    lines.append("")
    for h in sorted(
        tested, key=lambda x: -len(x.get("evidence_trials", []))
    )[:12]:
        trials = h.get("evidence_trials", [])
        lines.append(
            f"- **{h.get('hypothesis_id')}** `{h.get('variable')}` "
            f"({h.get('direction')}) — {h.get('status')}, "
            f"{len(trials)} trial(s)"
        )
        conclusion = h.get("conclusion")
        if conclusion:
            lines.append(f"  - {conclusion}")
    lines.append("")
    return lines


def generate_lab_report(
    *,
    run_id: str,
    dataset_name: str,
    route_decision: dict | None = None,
    fingerprint: dict | None = None,
    version_store: VersionStore | None = None,
    start_seq: int = 1,
    end_seq: int | None = None,
) -> str:
    """Build markdown lab report; returns filepath."""
    store = version_store or VersionStore.shared()
    events = list(store.events(start_seq=start_seq, end_seq=end_seq))

    lines = [
        f"# UIL Lab Report — {dataset_name}",
        f"",
        f"- **Run:** `{run_id}`",
        f"- **Generated:** {_utc_now()}",
        f"- **Events:** {len(events)} (seq {start_seq}–{end_seq or store.last_seq})",
        f"",
    ]

    if fingerprint:
        lines += [
            "## Dataset fingerprint",
            f"- Samples: {fingerprint.get('n_samples')} × {fingerprint.get('n_features')}",
            f"- Target: {fingerprint.get('target_type')}",
            f"- Sequential: {fingerprint.get('sequential', False)}",
            f"- ID: `{fingerprint.get('fingerprint_id', '?')}`",
            "",
        ]

    if route_decision:
        lines += [
            "## Routing decision",
            f"- Domain: **{route_decision.get('domain', '?')}**",
            f"- Strategy: `{route_decision.get('strategy', '?')}`",
            f"- Source: {route_decision.get('source', '?')}",
            f"- {route_decision.get('description', '')}",
            "",
        ]

    by_op: dict[str, int] = {}
    belief_changes: list[str] = []
    for ev in events:
        op = ev.get("op", "?")
        by_op[op] = by_op.get(op, 0) + 1
        post = (ev.get("payload") or {}).get("post_state")
        if isinstance(post, dict) and "confidence" in post:
            belief_changes.append(
                f"- seq {ev['seq']}: `{ev['cell_id']}` {op} "
                f"conf→{post['confidence']}"
            )

    lines += ["## Event summary", ""]
    for op, count in sorted(by_op.items(), key=lambda x: -x[1]):
        lines.append(f"- `{op}`: {count}")
    lines.append("")

    if belief_changes:
        lines += ["## Belief trajectories (sample)", ""]
        lines.extend(belief_changes[-20:])
        lines.append("")

    lines += _hypotheses_section()

    lines += [
        "## Notes",
        "Report compiled from VersionStore — use `reconstruct(seq)` for full state.",
        "",
    ]

    out_dir = results_path("reports")
    os.makedirs(out_dir, exist_ok=True)
    safe_name = dataset_name.replace(":", "_").replace("/", "_")
    path = os.path.join(out_dir, f"lab_{safe_name}_{run_id[:8]}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path
