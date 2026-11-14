"""Shared live observability feed — connects terminal runs to the web UI.

When the web console is running it creates results/.live_feed_enabled.
Any UniversalOptimizer.fit() (web OR terminal tests) then appends stage
snapshots to results/uil_live_feed.jsonl so the UI can display them in
real time without restarting tests through the browser.

Set UIL_LIVE_FEED=1 to force publishing even when the web UI is off
(useful for tail -f style debugging).
"""

from __future__ import annotations
import json
import os
import time
from datetime import datetime, timezone

from figpso_uil.cognition.paths import results_path

FEED_FILE = results_path("uil_live_feed.jsonl")
ENABLE_FLAG = results_path(".live_feed_enabled")
MAX_FEED_BYTES = 5 * 1024 * 1024  # rotate when feed exceeds 5 MB


def is_enabled() -> bool:
    if os.environ.get("UIL_LIVE_FEED", "").strip().lower() in ("0", "false", "no", "off"):
        return False
    # Always publish unless explicitly disabled — web UI polls the file
    return True


def enable() -> None:
    os.makedirs(os.path.dirname(ENABLE_FLAG), exist_ok=True)
    with open(ENABLE_FLAG, "w", encoding="utf-8") as f:
        f.write(datetime.now(timezone.utc).isoformat())


def disable() -> None:
    if os.path.isfile(ENABLE_FLAG):
        os.remove(ENABLE_FLAG)


def _maybe_rotate(feed_file: str = FEED_FILE) -> None:
    try:
        if os.path.isfile(feed_file) and os.path.getsize(feed_file) > MAX_FEED_BYTES:
            archived = feed_file + ".old"
            if os.path.isfile(archived):
                os.remove(archived)
            os.replace(feed_file, archived)
    except OSError:
        pass


def publish(
    snapshot: dict,
    meta: dict | None = None,
    *,
    filepath: str | None = None,
) -> None:
    """Append one observability snapshot. No-op if feed disabled.

    `filepath` scopes the feed for isolated `state_dir` runs (bug #30b) so
    experiment arms never append into the shared results/uil_live_feed.jsonl.
    """
    if not is_enabled():
        return
    feed_file = filepath or FEED_FILE
    _maybe_rotate(feed_file)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "unix": time.time(),
        **(meta or {}),
        **snapshot,
    }
    os.makedirs(os.path.dirname(feed_file) or ".", exist_ok=True)
    with open(feed_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def read_since(line_offset: int = 0, max_lines: int = 200) -> dict:
    """Read feed lines from byte/line offset for polling clients."""
    if not os.path.isfile(FEED_FILE):
        return {"lines": [], "next_offset": 0, "total_lines": 0, "enabled": is_enabled()}

    with open(FEED_FILE, encoding="utf-8") as f:
        all_lines = f.readlines()

    total = len(all_lines)
    chunk = all_lines[line_offset:line_offset + max_lines]
    parsed = []
    for raw in chunk:
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed.append(json.loads(raw))
        except json.JSONDecodeError:
            continue

    return {
        "lines": parsed,
        "next_offset": min(line_offset + len(chunk), total),
        "total_lines": total,
        "enabled": is_enabled(),
        "feed_path": FEED_FILE,
    }


def clear_feed() -> None:
    if os.path.isfile(FEED_FILE):
        os.remove(FEED_FILE)


def read_tail(max_lines: int = 100) -> dict:
    """Return the most recent feed events (for UI bootstrap)."""
    if not os.path.isfile(FEED_FILE):
        return {"lines": [], "total_lines": 0, "enabled": is_enabled(), "feed_path": FEED_FILE}
    with open(FEED_FILE, encoding="utf-8") as f:
        all_lines = f.readlines()
    total = len(all_lines)
    chunk = all_lines[max(0, total - max_lines):]
    parsed = []
    for raw in chunk:
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return {
        "lines": parsed,
        "next_offset": total,
        "total_lines": total,
        "enabled": is_enabled(),
        "feed_path": FEED_FILE,
    }
