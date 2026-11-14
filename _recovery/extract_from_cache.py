"""Recover the REAL webui from the Cursor browser cache.

The cache f_* files are clean, complete response BODIES (verified: they begin
at offset 0 with '<!doctype' / '/*' and end cleanly). They are named by a URL
hash, so we map each body to its filename by CONTENT: a self-naming header
comment where present, otherwise a unique signature.

Writes recovered files to _recovery/from_cache/ mirroring the webui tree, and
prints a mapping report. Nothing is installed here -- review, then install.
"""
import os
import re
import shutil

CACHE = r"C:\Users\arinb\Downloads".replace("Downloads", "")  # placeholder, set below
CACHE = r"C:\Users\arinb\AppData\Roaming\Cursor\Partitions\cursor-browser\Cache\Cache_Data"
ROOT = r"C:\Users\arinb\Downloads\figpso_uil"
OUT = os.path.join(ROOT, "_recovery", "from_cache")

# URL path -> on-disk destination under webui/. /app/* and /desktop/* are the
# same files served at two mounts; we recover the desktop/ copy (canonical).
def dest_for(path: str) -> str | None:
    if path in ("/", "/index.html"):
        return "webui/frontend/index.html"
    if path in ("/app/", "/desktop/"):
        return "webui/desktop/index.html"
    if path.startswith("/app/"):
        return "webui/desktop/" + path[len("/app/"):]
    if path.startswith("/desktop/"):
        return "webui/desktop/" + path[len("/desktop/"):]
    if path.startswith("/shared/"):
        return "webui/" + path.lstrip("/")
    if path.startswith("/js/") or path.startswith("/assets/") or path.startswith("/runtime/"):
        return "webui/frontend/" + path.lstrip("/")
    if path.endswith(".css"):
        return "webui/frontend/" + path.lstrip("/")
    return "webui/frontend/" + path.lstrip("/")


def sniff(body: bytes) -> str:
    h = body[:64].lstrip()
    if h[:2] == b"\xff\xd8":
        return "jpg"
    if body[:4] in (b"\x00\x00\x00\x18", b"\x00\x00\x00\x1c", b"\x00\x00\x00 ") or b"ftyp" in body[:32]:
        return "mp4"
    hs = h.lower()
    if hs.startswith(b"<!doctype") or hs.startswith(b"<html"):
        return "html"
    if hs.startswith(b"<svg") or (b"<svg" in body[:200] and b"xmlns" in body[:400]):
        return "svg"
    t = body[:6000]
    looks_css = t.count(b"{") > 3 and t.count(b"}") > 3 and b";" in t and b"</" not in t[:80]
    looks_js = any(k in t for k in (b"function", b"=>", b"const ", b"window.", b"document.", b"export "))
    if looks_js and not looks_css:
        return "js"
    if looks_css:
        return "css"
    if looks_js:
        return "js"
    return "other"


def self_name(body: bytes):
    """A header comment like '/* graph.js — ...' or 'site.js' → basename."""
    head = body[:300].decode("latin-1", "replace")
    m = re.search(r"([A-Za-z0-9_\-]+\.(?:js|css))", head)
    return m.group(1) if m else None


def classify(body: bytes, typ: str):
    """Return a set of candidate basenames this body could be."""
    s = body[:8000]
    def has(*subs): return all(sub in body for sub in subs)
    cands = set()
    if typ == "html":
        if b"a machine that remembers" in body:
            cands.add("__marketing_index__")
        elif b"Research Console" in body:
            cands.add("__console_index__")
        else:
            cands.add("__html__")
    elif typ == "css":
        if has(b".ub-node") and has(b".hero-title"):
            cands.add("site.css")
        elif has(b".sidebar") and (b".nav-item" in body):
            cands.add("app.css")
        elif b"--acc" in s and b":root" in s and b".ub-node" not in body and b".sidebar" not in body:
            cands.add("tokens.css")
        elif b".uil-logo" in body or b"uil-wordmark" in body and b"logo" in s:
            cands.add("logo.css")
        nm = self_name(body)
        if nm and nm.endswith(".css"):
            cands.add(nm)
    elif typ == "js":
        nm = self_name(body)
        if nm and nm.endswith(".js"):
            cands.add(nm)
        if b"window.NeuralField" in body or (b"NeuralField" in body and b".mount" in body):
            cands.add("neural-field.js")
        if b"class KnowledgeGraph" in body:
            cands.add("graph.js/neural-graph.js")
    return cands


def main():
    files = [f for f in os.listdir(CACHE) if f.startswith("f_")]
    report = []
    for f in files:
        p = os.path.join(CACHE, f)
        try:
            d = open(p, "rb").read()
        except Exception:
            continue
        if len(d) < 32:
            continue
        t = sniff(d)
        if t in ("other",):
            continue
        cands = classify(d, t)
        report.append((f, len(d), t, self_name(d), sorted(cands)))
    # print
    report.sort(key=lambda r: (r[2], -r[1]))
    print(f"{'cache':10} {'size':>8}  {'type':5} {'self-name':22} candidates")
    for f, sz, t, nm, cands in report:
        print(f"{f:10} {sz:>8}  {t:5} {str(nm):22} {', '.join(cands)}")


if __name__ == "__main__":
    main()
