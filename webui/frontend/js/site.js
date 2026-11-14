/* ════════════════════════════════════════════════════════════════════
   site.js — marketing site behaviour
   Scroll progress · sticky nav · active-section tracking · reveal
   observer · theme toggle · WebGL hero mount · live module counts and
   hero stats pulled from the running backend (/api). Degrades cleanly
   when the backend is unreachable or empty.
   ════════════════════════════════════════════════════════════════════ */

"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const API = "/api";

async function api(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error(res.statusText);
  return res.json();
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function fmtInt(n) { return n == null ? "—" : Number(n).toLocaleString("en-US"); }
function fmtBytes(b) {
  if (b == null) return "—";
  if (b < 1024) return b + " B";
  if (b < 1048576) return (b / 1024).toFixed(0) + " KB";
  if (b < 1073741824) return (b / 1048576).toFixed(1) + " MB";
  return (b / 1073741824).toFixed(2) + " GB";
}

/* ── theme ──────────────────────────────────────────────────────── */
(function initTheme() {
  const saved = localStorage.getItem("uil-theme");
  if (saved === "light" || saved === "dark") document.documentElement.dataset.theme = saved;
  $("#themeToggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("uil-theme", next);
    window.dispatchEvent(new Event("uil-theme-change"));
  });
})();

/* ── view mode: rendered (hyperreal) | wireframe (HUD FUI) ───────── */
(function initViewMode() {
  const root = document.documentElement;
  const btnR = $("#viewRendered");
  const btnW = $("#viewWireframe");
  if (!btnR || !btnW) return;

  function apply(mode) {
    const m = mode === "wireframe" ? "wireframe" : "rendered";
    root.dataset.view = m;
    localStorage.setItem("uil-view", m);
    btnR.setAttribute("aria-pressed", m === "rendered" ? "true" : "false");
    btnW.setAttribute("aria-pressed", m === "wireframe" ? "true" : "false");
    window.dispatchEvent(new CustomEvent("uil-view-mode", { detail: { mode: m } }));
  }

  const saved = localStorage.getItem("uil-view");
  apply(saved === "wireframe" || saved === "rendered" ? saved : "rendered");
  btnR.addEventListener("click", () => apply("rendered"));
  btnW.addEventListener("click", () => apply("wireframe"));
})();

/* ── scroll progress + sticky nav ───────────────────────────────── */
const nav = $("#topnav");
const progress = $("#scrollProgress");
function onScroll() {
  const st = window.scrollY || document.documentElement.scrollTop;
  const h = document.documentElement.scrollHeight - window.innerHeight;
  progress.style.width = (h > 0 ? (st / h) * 100 : 0) + "%";
  nav.classList.toggle("stuck", st > 30);
}
window.addEventListener("scroll", onScroll, { passive: true });
onScroll();

/* ── active section in nav ──────────────────────────────────────── */
const navLinks = $$("#topnavLinks a[data-nav]");
const sectionFor = new Map(navLinks.map((a) => [a.getAttribute("href").slice(1), a]));
const navObserver = new IntersectionObserver((entries) => {
  entries.forEach((e) => {
    if (e.isIntersecting) {
      navLinks.forEach((a) => a.classList.remove("active"));
      sectionFor.get(e.target.id)?.classList.add("active");
    }
  });
}, { rootMargin: "-45% 0px -50% 0px" });
["pillars", "what", "cases", "modules", "graph", "hitl", "vision"].forEach((id) => {
  const el = document.getElementById(id);
  if (el) navObserver.observe(el);
});

/* ── reveal fallback (anime-motion owns reveals when CDN loads) ─── */
if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
  $$(".reveal").forEach((el) => el.classList.add("in"));
  $("#pillarTransparent")?.classList.add("is-on");
} else {
  setTimeout(() => {
    if (!document.documentElement.classList.contains("anime-on")) {
      $$(".reveal").forEach((el) => el.classList.add("in"));
      $("#pillarTransparent")?.classList.add("is-on");
    }
  }, 2400);
}

/* ── full-site neural particle field ─────────────────────────────── */
let fieldCtl = null;
if (window.NeuralField) {
  const mountField = () => {
    if (fieldCtl) return;
    const canvas = $("#siteField");
    if (!canvas) return;
    fieldCtl = window.NeuralField.mount(canvas, {
      mode: "hero",
      density: 16000,
      maxNodes: 280,
      minNodes: 120,
      linkRadius: 0.32,
      globalPointer: true,
    });
    window.__uilField = fieldCtl;
  };
  if (document.documentElement.classList.contains("is-intro")) {
    window.addEventListener("uil-intro-done", mountField, { once: true });
    // Safety net if intro never signals done
    setTimeout(() => { if (!fieldCtl) mountField(); }, 20000);
  } else {
    mountField();
  }
}

function countUp(el, targetText) {
  if (!el) return;
  const raw = String(targetText).replace(/,/g, "");
  const num = parseFloat(raw);
  if (!Number.isFinite(num) || /[·×]/.test(String(targetText)) || String(targetText).includes(" ")) {
    el.textContent = targetText;
    el.classList.add("pop-in");
    return;
  }
  const isInt = Number.isInteger(num) && !String(targetText).includes(".");
  const t0 = performance.now();
  const dur = 700;
  const tick = (t) => {
    const p = Math.min(1, (t - t0) / dur);
    const ease = 1 - Math.pow(1 - p, 3);
    const v = num * ease;
    el.textContent = isInt ? Math.round(v).toLocaleString("en-US") : v.toFixed(1);
    if (p < 1) requestAnimationFrame(tick);
    else el.textContent = targetText;
  };
  requestAnimationFrame(tick);
}

/* ── live module counts + hero stats ────────────────────────────── */
const MODULES = [
  { key: "principles", name: "Principle Library", ic: "◇", c: "var(--acc)", desc: "4-tier mined hypotheses", file: "principles" },
  { key: "network", name: "Knowledge Graph", ic: "⬡", c: "var(--synapse)", desc: "neurons + synapses", live: true },
  { key: "formulas", name: "Formula Engine", ic: "ƒ", c: "var(--s2)", desc: "evolved loss + symbolic", file: "formulas" },
  { key: "symbolic_formulas", name: "Symbolic Classifiers", ic: "Σ", c: "var(--neural)", desc: "readable formula population", file: "symbolic_formulas" },
  { key: "causal", name: "Causal Engine", ic: "⇄", c: "var(--s3)", desc: "majority-of-3 hypotheses", file: "causal" },
  { key: "episodes", name: "Episodic Memory", ic: "◷", c: "var(--acc-2)", desc: "KDE warm-start", file: "episodes" },
  { key: "journal", name: "Journal", ic: "▤", c: "var(--synapse)", desc: "stage-by-stage record", file: "journal" },
  { key: "archive", name: "Knowledge Archive", ic: "⊟", c: "var(--neural)", desc: "nothing is ever lost", file: "knowledge_archive" },
];

function moduleCard(m, count, isLive) {
  return `<div class="mod-card" style="--c:${m.c}">
    <div class="mod-card-top">
      <div class="mod-ic">${m.ic}</div>
      <div class="mod-count ${isLive ? "live" : ""}">${count}</div>
    </div>
    <div class="mod-name">${esc(m.name)}</div>
    <div class="mod-desc">${esc(m.desc)}</div>
  </div>`;
}

async function loadModules() {
  const grid = $("#moduleGrid");
  // render skeleton immediately so the section is never empty
  grid.innerHTML = MODULES.map((m) => moduleCard(m, "—", false)).join("");

  let persistence = null, network = null;
  try { persistence = await api("/state/persistence"); } catch {}
  try { network = await api("/state/network"); } catch {}

  const files = persistence?.files || {};
  const rec = (k) => (files[k] && files[k].exists ? files[k].record_count : 0);

  const netSummary = network?.summary || {};
  const nodes = netSummary.nodes ?? (network?.nodes ? Object.keys(network.nodes).length : 0);
  const edges = netSummary.edges ?? (network?.edges ? network.edges.length : 0);

  const counts = {
    principles: rec("principles"),
    network: `${fmtInt(nodes)}·${fmtInt(edges)}`,
    formulas: rec("formulas"),
    symbolic_formulas: rec("symbolic_formulas"),
    causal: rec("causal"),
    episodes: rec("episodes"),
    journal: rec("journal"),
    archive: rec("knowledge_archive"),
  };
  grid.innerHTML = MODULES.map((m) => moduleCard(m, counts[m.key] ?? 0, !!m.live)).join("");

  // hero stats
  countUp($("#hsGraph"), `${fmtInt(nodes)} · ${fmtInt(edges)}`);
  if (persistence) countUp($("#hsMemory"), fmtBytes(persistence.total_bytes));
  $("#hsFiles").textContent = persistence
    ? Object.values(files).filter((f) => f.exists).length + " stores"
    : "—";
  $("#hsFiles").classList.add("pop-in");

  // events in history (versioned replay); tolerate absence
  try {
    const replay = await api("/state/replay");
    countUp($("#hsEvents"), fmtInt(replay.seq ?? 0));
  } catch { $("#hsEvents").textContent = "0"; }

  const online = !!(persistence || network);
  $("#footerStatus").textContent = online
    ? "backend online · counts are live from this instance"
    : "backend offline · showing the static pitch (start uvicorn to see live data)";
}

/* smooth-scroll offset for the fixed nav is handled by CSS scroll-margin;
   add it dynamically so anchor jumps don't hide under the bar */
$$('a[href^="#"]').forEach((a) => {
  a.addEventListener("click", (e) => {
    const id = a.getAttribute("href").slice(1);
    if (!id) return;
    const target = document.getElementById(id);
    if (!target) return;
    e.preventDefault();
    const y = target.getBoundingClientRect().top + window.scrollY - 72;
    window.scrollTo({ top: y, behavior: "smooth" });
  });
});

loadModules();

/* ══ izanami-style motion: inertial scroll · word cascades · parallax ══ */
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const TOUCH = window.matchMedia("(pointer: coarse)").matches;

/* masked word-cascade on chapter headings (plain-text headings only) */
if (!REDUCED) {
  $$(".chapter-head h2, .quiet-title").forEach((h) => {
    const words = h.textContent.trim().split(/\s+/);
    h.innerHTML = words.map((w, i) => `<span class="w"><i style="--wi:${i}">${esc(w)}</i></span>`).join(" ");
  });
}

/* Native scroll only — custom inertial wheel was a major lag source */
if (!REDUCED) {
  const heroPhoto = $(".hero-photo");
  let paraRaf = 0;
  window.addEventListener("scroll", () => {
    if (!heroPhoto || paraRaf) return;
    paraRaf = requestAnimationFrame(() => {
      paraRaf = 0;
      heroPhoto.style.transform = `translateY(${Math.min(120, scrollY * 0.12)}px)`;
    });
  }, { passive: true });
}
