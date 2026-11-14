/* ════════════════════════════════════════════════════════════════════
   core.js — shared plumbing for the FIGPSO UIL Research Console.
   API client, toasts, navigation, theme, shared edit modal, and a tiny
   event bus so each view module can subscribe to run/stage events.
   ════════════════════════════════════════════════════════════════════ */

"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const API = "/api";

/* ── tiny helpers ──────────────────────────────────────────────────── */
async function api(path, opts = {}) {
  const res = await fetch(API + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
}

function toast(msg, kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => { el.classList.add("out"); setTimeout(() => el.remove(), 350); }, 4600);
}

function fmt(v, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
  return Number(v).toFixed(digits);
}

function fmtBytes(b) {
  if (b === null || b === undefined) return "—";
  if (b < 1024) return b + " B";
  if (b < 1024 * 1024) return (b / 1024).toFixed(1) + " KB";
  if (b < 1024 * 1024 * 1024) return (b / (1024 * 1024)).toFixed(1) + " MB";
  return (b / (1024 * 1024 * 1024)).toFixed(2) + " GB";
}

function fmtInt(n) {
  if (n === null || n === undefined) return "—";
  return Number(n).toLocaleString("en-US");
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function timeShort(iso) {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
    });
  } catch { return String(iso); }
}

/* Horizontal labelled bars (used by KG stats, formula top, replay bars) */
function renderHBars(container, rows, { colorFn = () => "var(--acc)", wide = false, fmtV = fmtInt } = {}) {
  if (!rows.length) {
    container.innerHTML = `<div class="empty small"><p>Nothing to show yet.</p></div>`;
    return;
  }
  const max = Math.max(1, ...rows.map((r) => r.value));
  container.innerHTML = rows.map((r) => `
    <div class="hb-row ${wide ? "wide" : ""}" title="${esc(r.name)}">
      <span class="hb-name">${esc(r.name)}</span>
      <div class="hb-track"><div class="hb-fill" data-w="${Math.round((r.value / max) * 100)}"
        style="background:${colorFn(r)}"></div></div>
      <span class="hb-count">${fmtV(r.value)}</span>
    </div>`).join("");
  requestAnimationFrame(() => {
    container.querySelectorAll(".hb-fill").forEach((f) => { f.style.width = f.dataset.w + "%"; });
  });
}

/* Minimal markdown renderer for lab reports */
function mdToHtml(md) {
  const lines = String(md).split("\n");
  let html = "", inTable = false, inList = false;
  const inline = (s) => esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
  for (const line of lines) {
    if (/^\|/.test(line)) {
      if (/^[\s|:-]+$/.test(line)) continue;
      if (!inTable) { html += "<table>"; inTable = true; }
      html += `<tr>${line.split("|").slice(1, -1).map((c) => `<td>${inline(c.trim())}</td>`).join("")}</tr>`;
      continue;
    } else if (inTable) { html += "</table>"; inTable = false; }
    if (/^\s*[-*] /.test(line)) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${inline(line.replace(/^\s*[-*] /, ""))}</li>`;
      continue;
    } else if (inList) { html += "</ul>"; inList = false; }
    const h = line.match(/^(#{1,3}) (.*)/);
    if (h) { html += `<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`; continue; }
    if (line.trim()) html += `<p>${inline(line)}</p>`;
  }
  if (inTable) html += "</table>";
  if (inList) html += "</ul>";
  return html;
}

/* ── global namespace + event bus ──────────────────────────────────── */
const UIL = {
  currentRun: null,
  lastSnapshot: null,
  _showHooks: {},   // view -> [fn]
  _stageSubs: [],   // fn(snapshot)
  _runEventSubs: [], // fn(msg) — finished / error
  _pollSubs: [],    // fn() every 4s while tab visible

  registerShow(view, fn) { (this._showHooks[view] ??= []).push(fn); },
  onStage(fn) { this._stageSubs.push(fn); },
  onRunEvent(fn) { this._runEventSubs.push(fn); },
  onPoll(fn) { this._pollSubs.push(fn); },

  fireStage(snap) {
    this.lastSnapshot = snap;
    for (const fn of this._stageSubs) { try { fn(snap); } catch (e) { console.error(e); } }
  },
  fireRunEvent(msg) {
    for (const fn of this._runEventSubs) { try { fn(msg); } catch (e) { console.error(e); } }
  },

  activeView: "overview",
  show(view) {
    $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
    $$(".view").forEach((v) => v.classList.remove("active"));
    const target = $(`#view-${view}`);
    if (!target) return;
    void target.offsetWidth;
    target.classList.add("active");
    this.activeView = view;
    const meta = VIEW_META[view] || {};
    $("#topTitle").textContent = meta.title || view;
    $("#topCrumb").textContent = meta.crumb || "";
    $(".main").scrollTop = 0;
    // keep URL deep-linkable from the pitch site (/app/#kg)
    if (location.hash.replace(/^#/, "") !== view) {
      history.replaceState(null, "", `#${view}`);
    }
    for (const fn of this._showHooks[view] || []) { try { fn(); } catch (e) { console.error(e); } }
    window.dispatchEvent(new CustomEvent("uil-view-show", { detail: { view } }));
  },
};

const VIEW_META = {
  overview: { title: "What is UIL", crumb: "Overview" },
  mission: { title: "Mission Control", crumb: "Operate" },
  ops: { title: "Terminal Feed", crumb: "Operate" },
  kg: { title: "Knowledge Graph", crumb: "Knowledge" },
  formulas: { title: "Formula Library", crumb: "Knowledge" },
  principles: { title: "Principles & Causal", crumb: "Knowledge" },
  memory: { title: "Memory Proof", crumb: "Knowledge" },
  hitl: { title: "Human in the Loop", crumb: "Scientist" },
  goal: { title: "Goal Director", crumb: "Scientist" },
  timeline: { title: "Timeline & Replay", crumb: "Scientist" },
};

/* navigation wiring */
$$(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => UIL.show(btn.dataset.view));
});
document.addEventListener("click", (e) => {
  const goto = e.target.closest("[data-goto]");
  if (goto) UIL.show(goto.dataset.goto);
});
$("#runChip").addEventListener("click", () => UIL.show("mission"));

/* poll loop: every 4s notify subscribers (they decide if relevant) */
setInterval(() => {
  if (document.hidden) return;
  for (const fn of UIL._pollSubs) { try { fn(); } catch (e) { console.error(e); } }
}, 4000);

/* ── theme ─────────────────────────────────────────────────────────── */
(function initTheme() {
  const saved = localStorage.getItem("uil-theme");
  if (saved === "light" || saved === "dark") {
    document.documentElement.dataset.theme = saved;
  }
  $("#themeToggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("uil-theme", next);
    window.dispatchEvent(new Event("uil-theme-change"));
  });
})();

/* ── shared edit modal ─────────────────────────────────────────────── */
/* openEditModal({
     title, readable, confidence, status, statusOptions:[..]|null,
     tier|null, showReject, showUndo,
     onSave(fields) -> Promise, onReject() -> Promise, onUndo() -> Promise })  */
let _modalCfg = null;

function openEditModal(cfg) {
  _modalCfg = cfg;
  $("#modalTitle").textContent = cfg.title || "Edit";
  $("#modalReadable").textContent = cfg.readable || "";
  const confField = $("#modalConfField");
  if (cfg.confidence === null) confField.classList.add("hidden");
  else {
    confField.classList.remove("hidden");
    $("#modalConf").value = cfg.confidence ?? "";
  }
  const stField = $("#modalStatusField");
  if (cfg.statusOptions && cfg.statusOptions.length) {
    stField.classList.remove("hidden");
    $("#modalStatus").innerHTML = cfg.statusOptions
      .map((o) => `<option value="${esc(o)}" ${o === cfg.status ? "selected" : ""}>${esc(o)}</option>`)
      .join("");
  } else stField.classList.add("hidden");
  const tierField = $("#modalTierField");
  if (cfg.tier !== undefined && cfg.tier !== null) {
    tierField.classList.remove("hidden");
    $("#modalTier").value = String(cfg.tier);
  } else tierField.classList.add("hidden");
  $("#modalReject").classList.toggle("hidden", !cfg.onReject);
  $("#modalUndo").classList.toggle("hidden", !cfg.onUndo);
  $("#modalVeil").classList.remove("hidden");
}

function closeModal() { $("#modalVeil").classList.add("hidden"); _modalCfg = null; }

$("#modalCancel").addEventListener("click", closeModal);
$("#modalVeil").addEventListener("click", (e) => { if (e.target === e.currentTarget) closeModal(); });

$("#modalSave").addEventListener("click", async () => {
  if (!_modalCfg) return;
  const fields = {};
  if (_modalCfg.confidence !== null) {
    const conf = parseFloat($("#modalConf").value);
    if (Number.isNaN(conf) || conf < 0 || conf > 1) { toast("Confidence must be between 0 and 1", "err"); return; }
    fields.confidence = conf;
  }
  if (_modalCfg.statusOptions && _modalCfg.statusOptions.length) fields.status = $("#modalStatus").value;
  if (_modalCfg.tier !== undefined && _modalCfg.tier !== null) fields.tier = parseInt($("#modalTier").value, 10);
  try {
    await _modalCfg.onSave(fields);
    closeModal();
  } catch (err) { toast(err.message, "err"); }
});

$("#modalReject").addEventListener("click", async () => {
  if (!_modalCfg || !_modalCfg.onReject) return;
  try { await _modalCfg.onReject(); closeModal(); }
  catch (err) { toast(err.message, "err"); }
});

$("#modalUndo").addEventListener("click", async () => {
  if (!_modalCfg || !_modalCfg.onUndo) return;
  try { await _modalCfg.onUndo(); closeModal(); }
  catch (err) { toast(err.message, "err"); }
});
