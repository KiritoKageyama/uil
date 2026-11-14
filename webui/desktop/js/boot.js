/* boot.js — API health, initial view, ring CSS, global wiring */

"use strict";

(function initRingCss() {
  const circ = 2 * Math.PI * 52;
  const ring = $("#ringFill");
  if (ring) {
    ring.style.strokeDasharray = String(circ);
    ring.style.strokeDashoffset = String(circ);
  }
})();

async function probeApi() {
  try {
    await api("/runs");
    $("#apiDot").classList.add("ok");
    $("#apiDot").classList.remove("bad");
    $("#apiStatus").textContent = "API online";
    return true;
  } catch {
    $("#apiDot").classList.add("bad");
    $("#apiDot").classList.remove("ok");
    $("#apiStatus").textContent = "API offline";
    return false;
  }
}

const _HASH_VIEWS = new Set(["overview", "mission", "ops", "kg", "formulas", "principles", "memory", "hitl", "goal", "timeline"]);
function _viewFromHash() {
  const h = (location.hash || "").replace(/^#/, "").trim();
  return _HASH_VIEWS.has(h) ? h : null;
}

async function boot() {
  const ok = await probeApi();
  if (!ok) {
    toast("Backend not reachable — start via python figpso_uil/desktop.py", "err");
  }
  // honor deep links from the marketing site (e.g. /app/#kg, /app/#hitl)
  UIL.show(_viewFromHash() || "overview");
  window.addEventListener("hashchange", () => {
    const v = _viewFromHash();
    if (v) UIL.show(v);
  });
  // warm mission datasets in background
  try {
    const info = await api("/runs/datasets");
    const n = (info.datasets || info.info || []).length;
    if (n) $("#memFiles").textContent = `${n} curated datasets ready`;
  } catch { /* ignore */ }
}

document.addEventListener("DOMContentLoaded", boot);
window.addEventListener("focus", probeApi);
