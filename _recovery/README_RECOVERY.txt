2026-08-03 — webui recovered.
The files here are the REPAIRED versions now live in webui/. The original
botched auto-reassembly (build_console.py / mk.py + *.fragments.txt) produced
malformed CSS (=====FRAG===== markers, unbalanced braces) and replaced 9 real
files with symlinks into browser caches. All fixed:
  - site.css / app.css: repaired via repair_css.py (marker-free, brace-balanced)
  - 8 cache symlinks dereferenced to real files (graph.js, neural-graph.js,
    anime.esm.min.js, 5 hero photos)
  - neural-field.js: original unrecoverable (cache held junk); reimplemented
    from the caller API (see the note at the top of that file).
broken_snapshot_*/ holds the pre-repair CSS. Do NOT re-run mk.py/build_console.py
without checking — the *.fragments.txt are the raw, still-conflicting source.

--- SECOND PASS (real files recovered from Cursor browser cache) ---
The first pass produced VALID but STRIPPED files (about half the GUI). The
user's real, complete files were found in the Cursor browser cache
(AppData/Roaming/Cursor/Partitions/cursor-browser/Cache/Cache_Data) — the
site had been viewed there, so full response bodies were cached. Installed
the authentic originals (previous versions saved to pre_cache_restore_*/):
  webui/frontend/index.html  25.8KB(stripped) -> 31.7KB (real, 11 sections)
  webui/desktop/index.html   20.5KB(stripped) -> 48.3KB (real, 10 views)
  webui/frontend/site.css    21.0KB(stripped) -> 38.6KB (real, 301 rules)
  webui/desktop/app.css      (repaired)        -> 48.2KB (real, 426 rules)
JS files and SVG/photo assets were already present/real. neural-field.js is
still the reimplementation (the real one is inside a data_* cache block, not
yet extracted — the reimplementation works with no errors).
Cache extraction tooling: _recovery/cache_manifest.py, extract_from_cache.py.
