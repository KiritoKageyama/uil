"""Recover broken symlink webui HTML/CSS from agent transcripts."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # == _recovery/ (script moved here 2026-08-30)
TRANSCRIPTS = Path.home() / ".cursor/projects/c-Users-arinb-Downloads-figpso-uil/agent-transcripts"
OUT = ROOT
OUT.mkdir(exist_ok=True)

TARGETS = {
    "webui/frontend/index.html": None,
    "webui/frontend/site.css": None,
    "webui/desktop/index.html": None,
    "webui/desktop/app.css": None,
}


def norm_path(p: str) -> str:
    p = p.replace("\\", "/")
    idx = p.lower().find("webui/")
    return p[idx:] if idx >= 0 else p


def strip_read_output(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    # Read tool format: "   123|content"
    if re.match(r"^\s*\d+\|", lines[0]):
        return "\n".join(
            ln.split("|", 1)[1] if re.match(r"^\s*\d+\|", ln) else ln for ln in lines
        )
    return text


def main() -> None:
    files = sorted(TRANSCRIPTS.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    fragments: dict[str, list[str]] = {k: [] for k in TARGETS}
    read_bodies: dict[str, str] = {}

    for fp in files:
        with fp.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                if "tool_use" not in line and "tool_result" not in line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                content = (obj.get("message") or {}).get("content")
                if not isinstance(content, list):
                    continue
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    typ = b.get("type")
                    name = b.get("name")
                    inp = b.get("input") or {}

                    if typ == "tool_use" and name == "Write" and "contents" in inp:
                        np = norm_path(inp.get("path", ""))
                        if np in TARGETS:
                            TARGETS[np] = inp["contents"]
                            print(f"WRITE {np} len={len(inp['contents'])}")

                    elif typ == "tool_use" and name == "StrReplace":
                        np = norm_path(inp.get("path", ""))
                        if np not in TARGETS:
                            continue
                        old, new = inp.get("old_string"), inp.get("new_string")
                        if old:
                            fragments[np].append(old)
                        if new:
                            fragments[np].append(new)
                        if TARGETS[np] is not None and old and new is not None:
                            cur = TARGETS[np]
                            if old in cur:
                                if inp.get("replace_all"):
                                    TARGETS[np] = cur.replace(old, new)
                                else:
                                    TARGETS[np] = cur.replace(old, new, 1)

                    elif typ == "tool_use" and name == "Read":
                        np = norm_path(inp.get("path", ""))
                        if np in TARGETS:
                            # stash path; result comes next as tool_result (separate msg)
                            pass

                    elif typ == "tool_result":
                        text = b.get("content")
                        if isinstance(text, list):
                            parts = []
                            for t in text:
                                if isinstance(t, dict) and "text" in t:
                                    parts.append(t["text"])
                                elif isinstance(t, str):
                                    parts.append(t)
                            text = "\n".join(parts)
                        if not isinstance(text, str) or len(text) < 1500:
                            continue
                        body = strip_read_output(text)
                        head = body.lstrip()[:60]
                        kind = None
                        if head.startswith("<!doctype") or head.startswith("<!DOCTYPE"):
                            if "Research Console" in body or "apiDot" in body or "view-overview" in body:
                                kind = "webui/desktop/index.html"
                            elif "data-view" in body or "site-field" in body or "uil-machine" in body or "FIGPSO" in body:
                                kind = "webui/frontend/index.html"
                        elif ":root" in body[:800] or (body.lstrip().startswith("/*") and "--" in body[:1500]):
                            if "--shell" in body or ".nav-item" in body or ".kg-workspace" in body:
                                kind = "webui/desktop/app.css"
                            elif ".hero" in body or ".site-" in body or "#siteField" in body:
                                kind = "webui/frontend/site.css"
                        if kind:
                            prev = read_bodies.get(kind, "")
                            if len(body) > len(prev):
                                read_bodies[kind] = body
                                print(f"READ-RESULT {kind} len={len(body)}")

    # Prefer full Write-replay; fall back to largest Read body
    for k, v in list(TARGETS.items()):
        if v is None and k in read_bodies:
            TARGETS[k] = read_bodies[k]
            print(f"FALLBACK-READ {k} len={len(read_bodies[k])}")

    for k, v in TARGETS.items():
        frag_path = OUT / (k.replace("/", "__") + ".fragments.txt")
        frag_path.write_text("\n\n=====FRAG=====\n\n".join(fragments[k]), encoding="utf-8")
        print(f"fragments {k}: {len(fragments[k])} pieces -> {frag_path.name}")
        if v:
            dest = OUT / k.replace("/", "__")
            dest.write_text(v, encoding="utf-8")
            print(f"saved {dest.name} ({len(v)} bytes)")
        else:
            print(f"MISSING {k}")


if __name__ == "__main__":
    main()
