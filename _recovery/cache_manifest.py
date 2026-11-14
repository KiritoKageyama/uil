"""Map every Cursor browser-cache entry to the URL it cached + sniff type.

Chromium 'simple cache' stores each response in an f_* file: a header, then
the request KEY (the URL as plaintext), then the data stream(s). We don't
need to parse the format precisely to IDENTIFY entries -- the URL is embedded
as readable text, and the body is the bulk text run. This builds a manifest
so we know which cache file holds which webui asset.
"""
import os
import re
import sys

CACHE = r"C:\Users\arinb\AppData\Roaming\Cursor\Partitions\cursor-browser\Cache\Cache_Data"

URL_RE = re.compile(rb"https?://[A-Za-z0-9._:/%\-?=&#~]+")


def sniff(body: bytes) -> str:
    head = body[:400].lstrip().lower()
    if head.startswith(b"<!doctype") or head.startswith(b"<html"):
        return "html"
    if b"function" in body[:2000] or b"=>" in body[:2000] or b"window." in body[:2000]:
        # could be js; css check below is stronger
        pass
    # CSS: lots of `{ ... }` with `:` and `;`
    sample = body[:4000]
    if sample.count(b"{") > 3 and sample.count(b"}") > 3 and b";" in sample and b"<" not in sample[:50]:
        return "css"
    if b"function" in body[:3000] or b"=>" in body[:3000] or b"const " in body[:3000] or b"window." in body[:3000]:
        return "js"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpg"
    return "other"


def urls_in(data: bytes):
    seen = []
    for m in URL_RE.findall(data):
        u = m.decode("latin-1")
        # trim trailing junk
        u = re.split(r'[\x00-\x1f"\'<>]', u)[0]
        if u not in seen:
            seen.append(u)
    return seen


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else ""
    rows = []
    for name in sorted(os.listdir(CACHE)):
        if not (name.startswith("f_") or name.startswith("data_")):
            continue
        p = os.path.join(CACHE, name)
        try:
            data = open(p, "rb").read()
        except Exception:
            continue
        urls = urls_in(data)
        # the localhost app URLs are what we care about
        app_urls = [u for u in urls if "localhost" in u or "127.0.0.1" in u]
        t = sniff(data)
        rows.append((name, len(data), t, app_urls[:4]))

    for name, size, t, urls in rows:
        line = f"{name:12} {size:>8}b  {t:6}  {', '.join(urls) if urls else '-'}"
        if not want or want.lower() in line.lower():
            print(line)


if __name__ == "__main__":
    main()
