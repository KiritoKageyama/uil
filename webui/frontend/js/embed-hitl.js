/* ════════════════════════════════════════════════════════════════════
   embed-hitl.js — read-only knowledge-cell preview for the marketing
   site. Pulls /api/state/cells?limit=12 and renders a compact strip.
   View-only; the edit / reject / undo flow lives in the console.
   ════════════════════════════════════════════════════════════════════ */

"use strict";

(function () {
  const strip = document.getElementById("mktHitlCells");
  if (!strip) return;

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function render(cells) {
    if (!cells.length) {
      strip.innerHTML = `<div class="empty small"><p>No knowledge cells yet — run UIL in the console to grow beliefs, then review them here.</p></div>`;
      return;
    }
    strip.innerHTML = cells.map((c, i) => {
      const conf = Math.round((c.confidence ?? 0) * 100);
      const type = c.cell_type || "cell";
      return `<div class="cell-mini" style="--i:${i}">
        <div class="cell-mini-head">
          <span class="cell-type ct-${esc(type)}">${esc(type.replace(/_/g, " "))}</span>
          <span class="cell-id">${esc(c.cell_id || "")}</span>
        </div>
        <div class="cell-readable">${esc(c.readable || "")}</div>
        <div class="cell-conf"><i style="width:${conf}%"></i></div>
      </div>`;
    }).join("");
  }

  async function load() {
    try {
      const res = await fetch("/api/state/cells?limit=12");
      if (!res.ok) throw new Error(res.statusText);
      const data = await res.json();
      render(data.cells || []);
    } catch {
      strip.innerHTML = `<div class="empty small"><p>Backend offline — start uvicorn to preview live knowledge cells.</p></div>`;
    }
  }

  const section = document.getElementById("hitl");
  let loaded = false;
  const loadOnce = () => { if (!loaded) { loaded = true; load(); } };
  if (section && "IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) loadOnce(); });
    }, { threshold: 0.15 });
    io.observe(section);
    setTimeout(loadOnce, 2400); // safety net if the observer never fires
  } else { loadOnce(); }
})();
