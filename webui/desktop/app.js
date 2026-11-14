/* ═══════════════════════════════════════════════════════════════════
   UIL Research Console — app logic
   Views: experiments (runs + live charts), knowledge (cells + HITL),
   goal (GoalCell director), timeline (VersionStore replay).
   ═══════════════════════════════════════════════════════════════════ */

"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const API = "/api";

// ── tiny helpers ────────────────────────────────────────────────────
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
  setTimeout(() => { el.classList.add("out"); setTimeout(() => el.remove(), 350); }, 4200);
}

function fmt(v, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// ── navigation ──────────────────────────────────────────────────────
$$(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".nav-item").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const view = btn.dataset.view;
    $$(".view").forEach((v) => v.classList.remove("active"));
    const target = $(`#view-${view}`);
    target.classList.remove("active");
    void target.offsetWidth; // restart the enter animation
    target.classList.add("active");
    if (view === "knowledge") loadCells();
    if (view === "goal") loadGoal();
    if (view === "timeline") initTimeline();
  });
});

// ═══ Charts ═════════════════════════════════════════════════════════
// Two-series animated line chart on canvas. DPR-aware, recessive grid,
// crosshair + tooltip (dataviz: hover layer is default for line charts).
class LineChart {
  constructor(canvas, tipEl, series) {
    this.canvas = canvas;
    this.tip = tipEl;
    this.series = series; // [{key, label, color}]
    this.data = [];       // [{stage, ...values}]
    this.anim = 1;
    this._raf = null;
    const ro = new ResizeObserver(() => this.draw());
    ro.observe(canvas.parentElement);
    canvas.parentElement.addEventListener("mousemove", (e) => this.hover(e));
    canvas.parentElement.addEventListener("mouseleave", () => {
      this.hoverIdx = null; this.tip.classList.remove("show"); this.draw();
    });
  }

  push(pt) {
    this.data.push(pt);
    this.animateIn();
  }

  reset() { this.data = []; this.hoverIdx = null; this.draw(); }

  animateIn() {
    cancelAnimationFrame(this._raf);
    // rAF is throttled to zero in hidden/minimized windows — draw the
    // final frame synchronously so data is never invisible on restore.
    if (document.hidden) { this.anim = 1; this.draw(); return; }
    const t0 = performance.now();
    const tick = (t) => {
      this.anim = Math.min(1, (t - t0) / 420);
      this.draw();
      if (this.anim < 1) this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
  }

  layout() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: rect.width, h: rect.height, pad: { l: 40, r: 12, t: 10, b: 22 } };
  }

  scales(w, h, pad) {
    const n = this.data.length;
    const xs = (i) => pad.l + (n <= 1 ? 0 : (i / (n - 1)) * (w - pad.l - pad.r));
    let lo = Infinity, hi = -Infinity;
    for (const d of this.data) for (const s of this.series) {
      const v = d[s.key];
      if (v !== null && v !== undefined && !Number.isNaN(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    }
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (hi - lo < 0.02) { const m = (hi + lo) / 2; lo = m - 0.01; hi = m + 0.01; }
    const span = hi - lo;
    lo -= span * 0.12; hi += span * 0.12;
    const ys = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (h - pad.t - pad.b);
    return { xs, ys, lo, hi };
  }

  draw() {
    const { ctx, w, h, pad } = this.layout();
    ctx.clearRect(0, 0, w, h);
    const { xs, ys, lo, hi } = this.scales(w, h, pad);

    // recessive grid + axis labels
    ctx.strokeStyle = "#26262e";
    ctx.fillStyle = "#8a8b98";
    ctx.font = "10.5px 'Segoe UI', sans-serif";
    ctx.lineWidth = 1;
    for (let g = 0; g <= 3; g++) {
      const v = lo + ((hi - lo) * g) / 3;
      const y = ys(v);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.fillText(v.toFixed(2), 6, y + 3.5);
    }

    if (!this.data.length) {
      ctx.fillStyle = "#8a8b98";
      ctx.font = "12px 'Segoe UI', sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("waiting for stages…", w / 2, h / 2);
      ctx.textAlign = "left";
      return;
    }

    const n = this.data.length;
    const upto = Math.max(1, Math.ceil(n * (n <= 2 ? 1 : this.anim)));

    for (const s of this.series) {
      // area fill
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < upto; i++) {
        const v = this.data[i][s.key];
        if (v === null || v === undefined) continue;
        const x = xs(i), y = ys(v);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      }
      if (started) {
        const grad = ctx.createLinearGradient(0, pad.t, 0, h - pad.b);
        grad.addColorStop(0, s.color + "2e");
        grad.addColorStop(1, s.color + "00");
        ctx.lineTo(xs(upto - 1), h - pad.b);
        ctx.lineTo(xs(0), h - pad.b);
        ctx.closePath();
        ctx.fillStyle = grad;
        ctx.fill();
      }
      // stroke with glow
      ctx.beginPath();
      started = false;
      for (let i = 0; i < upto; i++) {
        const v = this.data[i][s.key];
        if (v === null || v === undefined) continue;
        const x = xs(i), y = ys(v);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.shadowColor = s.color;
      ctx.shadowBlur = 7;
      ctx.stroke();
      ctx.shadowBlur = 0;
      // end marker + direct label (selective, last point only)
      const last = this.data[upto - 1][s.key];
      if (last !== null && last !== undefined) {
        const x = xs(upto - 1), y = ys(last);
        ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = s.color; ctx.fill();
        ctx.strokeStyle = "#15151c"; ctx.lineWidth = 2; ctx.stroke();
      }
    }

    // crosshair
    if (this.hoverIdx !== null && this.hoverIdx !== undefined) {
      const x = xs(this.hoverIdx);
      ctx.strokeStyle = "rgba(255,255,255,0.22)";
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
      ctx.setLineDash([]);
      for (const s of this.series) {
        const v = this.data[this.hoverIdx][s.key];
        if (v === null || v === undefined) continue;
        ctx.beginPath(); ctx.arc(x, ys(v), 5, 0, Math.PI * 2);
        ctx.fillStyle = s.color; ctx.fill();
        ctx.strokeStyle = "#15151c"; ctx.lineWidth = 2; ctx.stroke();
      }
    }
  }

  hover(e) {
    if (!this.data.length) return;
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const { w, pad } = { w: rect.width, pad: { l: 40, r: 12 } };
    const mx = e.clientX - rect.left;
    const n = this.data.length;
    const frac = (mx - pad.l) / Math.max(1, w - pad.l - pad.r);
    const idx = Math.round(frac * (n - 1));
    this.hoverIdx = Math.max(0, Math.min(n - 1, idx));
    const d = this.data[this.hoverIdx];
    const rows = this.series
      .map((s) => `<div class="t-row"><i style="background:${s.color}"></i>${s.label}: <b>${fmt(d[s.key])}</b></div>`)
      .join("");
    this.tip.innerHTML = `<div class="t-row"><b>stage ${d.stage}</b></div>${rows}`;
    const xs = pad.l + (n <= 1 ? 0 : (this.hoverIdx / (n - 1)) * (w - pad.l - pad.r));
    this.tip.style.left = xs + "px";
    this.tip.style.top = "56px";
    this.tip.classList.add("show");
    this.draw();
  }
}

const css = getComputedStyle(document.documentElement);
const C = {
  s1: css.getPropertyValue("--s1").trim(),
  s2: css.getPropertyValue("--s2").trim(),
  s3: css.getPropertyValue("--s3").trim(),
  s4: css.getPropertyValue("--s4").trim(),
};

const accChart = new LineChart($("#accChart"), $("#accTip"), [
  { key: "accuracy", label: "stage acc", color: C.s1 },
  { key: "best_accuracy", label: "best acc", color: C.s2 },
]);
const stressChart = new LineChart($("#stressChart"), $("#stressTip"), [
  { key: "system_stress", label: "stress", color: C.s3 },
  { key: "regret", label: "regret", color: C.s4 },
]);

// ═══ Experiments: runs ══════════════════════════════════════════════
let currentRun = null;
let ws = null;
let heroTween = null;

function setHero(acc) {
  const el = $("#heroAcc");
  const from = parseFloat(el.dataset.v || "0") || 0;
  const to = acc || 0;
  el.dataset.v = to;
  cancelAnimationFrame(heroTween);
  const circ = 2 * Math.PI * 52;
  $("#ringFill").style.strokeDashoffset = String(circ * (1 - Math.max(0, Math.min(1, to))));
  // Hidden windows throttle rAF to zero — write the final value
  // directly so the number is correct the instant the app is restored.
  if (document.hidden) { el.textContent = (to * 100).toFixed(1) + "%"; return; }
  const t0 = performance.now();
  const tick = (t) => {
    const p = Math.min(1, (t - t0) / 700);
    const ease = 1 - Math.pow(1 - p, 3);
    const v = from + (to - from) * ease;
    el.textContent = (v * 100).toFixed(1) + "%";
    if (p < 1) heroTween = requestAnimationFrame(tick);
  };
  heroTween = requestAnimationFrame(tick);
}

function setRunChip(text, cls = "") {
  const chip = $("#runChip");
  chip.classList.remove("hidden", "err", "done");
  if (cls) chip.classList.add(cls);
  $("#runChipText").textContent = text;
}

function onStage(snap) {
  const pt = {
    stage: snap.stage,
    accuracy: snap.accuracy,
    best_accuracy: snap.best_accuracy,
    system_stress: snap.system_stress,
    regret: snap.regret,
  };
  accChart.push(pt);
  stressChart.push(pt);
  setHero(snap.best_accuracy);
  $("#hStage").textContent = `${snap.stage}/${snap.stages_total}`;
  $("#hStress").textContent = fmt(snap.system_stress, 2);
  $("#hRegret").textContent = fmt(snap.regret, 2);
  $("#hMaturity").textContent = snap.foundation_stage || snap.maturity || "—";
  setRunChip(`running · ${snap.dataset} · stage ${snap.stage}`);

  const feed = $("#stageFeed");
  $("#feedEmpty")?.remove();
  const item = document.createElement("div");
  item.className = "feed-item";
  const principles = (snap.triggered_principles || []).length;
  const nodes = (snap.activated_nodes || []).length;
  const tags = [];
  if (principles) tags.push(`<span class="ftag cool">${principles} principle${principles > 1 ? "s" : ""}</span>`);
  if (nodes) tags.push(`<span class="ftag">${nodes} node${nodes > 1 ? "s" : ""}</span>`);
  if (snap.intervention_variable) tags.push(`<span class="ftag warm">testing ${esc(snap.intervention_variable)}</span>`);
  if (snap.investigation_strategy) tags.push(`<span class="ftag warm">${esc(snap.investigation_strategy)}</span>`);
  if (snap.symbolic_formula_id) tags.push(`<span class="ftag">${esc(String(snap.symbolic_formula_id).slice(0, 14))}</span>`);
  item.innerHTML = `
    <span class="feed-stage">#${snap.stage}</span>
    <span class="feed-m"><b>${fmt(snap.accuracy)}</b><span>accuracy</span></span>
    <span class="feed-m"><b>${fmt(snap.system_stress, 2)}</b><span>stress</span></span>
    <span class="feed-m"><b>${fmt(snap.regret, 2)}</b><span>regret</span></span>
    <div class="feed-tags">${tags.join("")}</div>`;
  feed.appendChild(item);
  while (feed.children.length > 60) feed.removeChild(feed.firstChild);
}

function connectWs(runId) {
  if (ws) { try { ws.close(); } catch {} }
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/runs/${runId}`);
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === "stage") onStage(msg);
    else if (msg.type === "finished") {
      setRunChip(`${msg.status} · acc ${fmt(msg.accuracy)}`, "done");
      $("#runControls").classList.add("hidden");
      $("#startBtn").disabled = false;
      $("#startBtn .btn-label").textContent = "Start run";
      toast(`Run ${msg.status} — accuracy ${fmt(msg.accuracy)}`, "ok");
    } else if (msg.type === "error") {
      setRunChip("error", "err");
      $("#runControls").classList.add("hidden");
      $("#startBtn").disabled = false;
      $("#startBtn .btn-label").textContent = "Start run";
      toast(`Run error: ${msg.error}`, "err");
    }
  };
  ws.onclose = () => {};
}

$("#startBtn").addEventListener("click", async () => {
  const btn = $("#startBtn");
  const body = {
    dataset_name: $("#dsSelect").value,
    classifier: $("#clfSeg .on").dataset.v,
    seed: parseInt($("#seedInput").value, 10) || 42,
    stages: parseInt($("#stagesInput").value, 10) || 6,
    symbolic_learning_mode: $("#symMode").value,
  };
  btn.disabled = true;
  btn.querySelector(".btn-label").textContent = "Launching…";
  try {
    const res = await api("/runs", { method: "POST", body: JSON.stringify(body) });
    currentRun = res.run_id;
    accChart.reset(); stressChart.reset();
    $("#stageFeed").innerHTML = "";
    setHero(0);
    setRunChip(`starting · ${body.dataset_name}`);
    $("#runControls").classList.remove("hidden");
    $("#pauseBtn").textContent = "Pause";
    btn.querySelector(".btn-label").textContent = "Running…";
    connectWs(currentRun);
    toast(`Run started on ${body.dataset_name} (${body.classifier})`, "ok");
  } catch (err) {
    btn.disabled = false;
    btn.querySelector(".btn-label").textContent = "Start run";
    toast(`Could not start: ${err.message}`, "err");
  }
});

$("#stopBtn").addEventListener("click", async () => {
  if (!currentRun) return;
  try {
    await api(`/runs/${currentRun}/stop`, { method: "POST" });
    toast("Stop requested — finishing current stage", "");
  } catch (err) { toast(err.message, "err"); }
});

$("#pauseBtn").addEventListener("click", async () => {
  if (!currentRun) return;
  const btn = $("#pauseBtn");
  try {
    if (btn.textContent === "Pause") {
      await api(`/runs/${currentRun}/pause`, { method: "POST" });
      btn.textContent = "Resume";
      setRunChip("paused");
    } else {
      await api(`/runs/${currentRun}/resume`, { method: "POST" });
      btn.textContent = "Pause";
    }
  } catch (err) { toast(err.message, "err"); }
});

$$("#clfSeg button").forEach((b) => {
  b.addEventListener("click", () => {
    $$("#clfSeg button").forEach((x) => x.classList.remove("on"));
    b.classList.add("on");
    $("#symModeField").style.display = b.dataset.v === "SymbolicFormula" ? "" : "none";
  });
});

// ═══ Knowledge cells ════════════════════════════════════════════════
let cellFilter = "";
let cellCache = [];

async function loadCells() {
  try {
    const q = cellFilter ? `?cell_type=${cellFilter}&limit=60` : "?limit=60";
    const res = await api(`/state/cells${q}`);
    cellCache = res.cells;
    const grid = $("#cellGrid");
    if (!res.cells.length) {
      grid.innerHTML = `<div class="empty"><div class="empty-orb"></div><p>No cells of this type yet.</p></div>`;
      return;
    }
    grid.innerHTML = res.cells.map((c, i) => `
      <div class="cell-card" style="--i:${i}" data-idx="${i}">
        <div class="cell-head">
          <span class="cell-type ct-${esc(c.cell_type)}">${esc(c.cell_type.replace("_", " "))}</span>
          <span class="cell-id">${esc(c.cell_id)}</span>
        </div>
        <div class="cell-readable">${esc(c.readable || "")}</div>
        <div class="conf-bar"><div class="conf-fill" data-w="${Math.round((c.confidence || 0) * 100)}"></div></div>
      </div>`).join("");
    requestAnimationFrame(() => {
      $$(".conf-fill").forEach((f) => { f.style.width = f.dataset.w + "%"; });
    });
    $$(".cell-card").forEach((card) => {
      card.addEventListener("click", () => openModal(cellCache[parseInt(card.dataset.idx, 10)]));
    });
  } catch (err) { toast(`Cells: ${err.message}`, "err"); }
}

$$("#typeChips .chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    $$("#typeChips .chip").forEach((c) => c.classList.remove("on"));
    chip.classList.add("on");
    cellFilter = chip.dataset.t;
    loadCells();
  });
});
$("#cellsRefresh").addEventListener("click", loadCells);

// modal
let modalCell = null;
function openModal(cell) {
  modalCell = cell;
  $("#modalTitle").textContent = `${cell.cell_type} · ${cell.cell_id}`;
  $("#modalReadable").textContent = cell.readable || JSON.stringify(cell.post_state, null, 2);
  $("#modalConf").value = cell.confidence ?? "";
  $("#modalVeil").classList.remove("hidden");
}
$("#modalCancel").addEventListener("click", () => $("#modalVeil").classList.add("hidden"));
$("#modalVeil").addEventListener("click", (e) => { if (e.target === e.currentTarget) $("#modalVeil").classList.add("hidden"); });

$("#modalSave").addEventListener("click", async () => {
  if (!modalCell) return;
  const conf = parseFloat($("#modalConf").value);
  if (Number.isNaN(conf)) { toast("Enter a confidence between 0 and 1", "err"); return; }
  try {
    await api(`/state/cells/${encodeURIComponent(modalCell.cell_id)}?cell_type=${modalCell.cell_type}`, {
      method: "PATCH",
      body: JSON.stringify({ item_id: modalCell.cell_id, fields: { confidence: conf } }),
    });
    toast("Edit recorded (human_edit event)", "ok");
    $("#modalVeil").classList.add("hidden");
    loadCells();
  } catch (err) { toast(err.message, "err"); }
});

$("#modalReject").addEventListener("click", async () => {
  if (!modalCell) return;
  try {
    await api(`/state/cells/${encodeURIComponent(modalCell.cell_id)}/reject?cell_type=${modalCell.cell_type}`, { method: "POST" });
    toast("Cell rejected (human_reject event)", "ok");
    $("#modalVeil").classList.add("hidden");
    loadCells();
  } catch (err) { toast(err.message, "err"); }
});

// ═══ Goal Director ══════════════════════════════════════════════════
function renderActiveGoal(goal) {
  const box = $("#activeGoalBox");
  const hero = $("#heroGoal");
  if (!goal) {
    box.innerHTML = `<div class="empty small"><p>No active goal — the system runs pure bottom-up attention.</p></div>`;
    hero.innerHTML = "";
    return;
  }
  const tags = (list, cls) => (list || []).map((t) => `<span class="g-tag ${cls}">${esc(t)}</span>`).join("") || "<span class='muted'>—</span>";
  box.innerHTML = `
    <div class="g-directive">“${esc(goal.directive)}”</div>
    <div class="g-lists">
      <div class="g-line"><span>Focus</span>${tags(goal.focus_features, "focus")}</div>
      <div class="g-line"><span>Ignore</span>${tags(goal.ignore_features, "ignore")}</div>
      <div class="g-line"><span>Domains</span>${tags(goal.prefer_domains, "")}</div>
      <div class="g-line"><span>Relations</span>${tags(goal.prefer_relations, "")}</div>
      <div class="g-line"><span>Keywords</span>${tags(goal.goal_keywords, "")}</div>
      <div class="g-line"><span>Strength</span><b>${fmt(goal.confidence, 2)}</b></div>
      <div class="g-line"><span>Version</span>v${goal.version}</div>
    </div>`;
  hero.innerHTML = `<span class="goal-pill">${esc(goal.directive)}</span>`;
}

async function loadGoal() {
  try {
    const res = await api("/state/goal");
    renderActiveGoal(res.goal);
  } catch (err) { toast(`Goal: ${err.message}`, "err"); }
}

const csv = (v) => v.split(",").map((s) => s.trim()).filter(Boolean);
const selVals = (sel) => Array.from(sel.selectedOptions).map((o) => o.value);

$("#goalConf").addEventListener("input", () => {
  $("#goalConfVal").textContent = (parseInt($("#goalConf").value, 10) / 100).toFixed(2);
});

$("#goalSetBtn").addEventListener("click", async () => {
  const directive = $("#goalDirective").value.trim();
  if (!directive) { toast("A goal needs a directive", "err"); return; }
  const body = {
    directive,
    focus_features: csv($("#goalFocus").value),
    ignore_features: csv($("#goalIgnore").value),
    prefer_domains: selVals($("#goalDomains")),
    prefer_relations: selVals($("#goalRelations")),
    goal_keywords: csv($("#goalKeywords").value),
    confidence: parseInt($("#goalConf").value, 10) / 100,
  };
  try {
    const res = await api("/state/goal", { method: "POST", body: JSON.stringify(body) });
    renderActiveGoal(res.goal);
    toast(`Goal set (seq ${res.seq}) — next run will honor it`, "ok");
  } catch (err) { toast(err.message, "err"); }
});

$("#goalClearBtn").addEventListener("click", async () => {
  try {
    const res = await api("/state/goal/clear", { method: "POST" });
    renderActiveGoal(null);
    toast(res.cleared ? "Goal archived" : "No active goal to clear", "");
  } catch (err) { toast(err.message, "err"); }
});

// ═══ Timeline ═══════════════════════════════════════════════════════
const RB_COLORS = {
  principle: C.s1, formula: C.s2, causal_hypothesis: C.s3,
  edge: C.s4, node: "#6da7ec", routing_rule: "#d55181",
  episode: "#8a8b98", goal: "#c98500",
};
let seqDebounce = null;
let timelineReady = false;

async function initTimeline() {
  try {
    const res = await api("/state/replay");
    const last = res.seq || 1;
    const slider = $("#seqSlider");
    slider.max = String(last);
    if (!timelineReady) slider.value = String(last);
    $("#seqMax").textContent = `of ${last} events`;
    timelineReady = true;
    renderReplay(res);
    loadLabReport();
  } catch (err) { toast(`Timeline: ${err.message}`, "err"); }
}

function renderReplay(res) {
  $("#seqLabel").textContent = res.seq;
  const state = res.state || {};
  const rows = Object.entries(state)
    .map(([type, bucket]) => [type, Object.keys(bucket).length])
    .filter(([, n]) => n > 0)
    .sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...rows.map(([, n]) => n));
  $("#replayBars").innerHTML = rows.map(([type, n]) => `
    <div class="rb-row">
      <span class="rb-name">${esc(type.replace("_", " "))}</span>
      <div class="rb-track"><div class="rb-fill" data-w="${Math.round((n / max) * 100)}" style="background:${RB_COLORS[type] || "#8a8b98"}"></div></div>
      <span class="rb-count">${n}</span>
    </div>`).join("") || `<div class="empty small"><p>No events recorded yet.</p></div>`;
  requestAnimationFrame(() => {
    $$(".rb-fill").forEach((f) => { f.style.width = f.dataset.w + "%"; });
  });
}

$("#seqSlider").addEventListener("input", () => {
  $("#seqLabel").textContent = $("#seqSlider").value;
  clearTimeout(seqDebounce);
  seqDebounce = setTimeout(async () => {
    try {
      const res = await api(`/state/replay?seq=${$("#seqSlider").value}`);
      renderReplay(res);
    } catch (err) { toast(err.message, "err"); }
  }, 220);
});

function mdToHtml(md) {
  // minimal renderer: headings, bold, code, tables, lists
  const lines = md.split("\n");
  let html = "", inTable = false, inList = false;
  for (const line of lines) {
    if (/^\|/.test(line)) {
      if (/^[\s|:-]+$/.test(line)) continue; // header separator row
      if (!inTable) { html += "<table>"; inTable = true; }
      const cells = line.split("|").slice(1, -1).map((c) => `<td>${inline(c.trim())}</td>`).join("");
      html += `<tr>${cells}</tr>`;
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
  function inline(s) {
    return esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
  }
}

async function loadLabReport() {
  try {
    const res = await api("/state/lab-reports/latest");
    $("#labReport").innerHTML = mdToHtml(res.markdown);
  } catch {
    /* no reports yet — keep empty state */
  }
}

// Redraw everything when the window comes back from minimized/hidden —
// rAF-driven work skipped while hidden must catch up immediately.
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    accChart.draw();
    stressChart.draw();
    const el = $("#heroAcc");
    if (el.dataset.v) el.textContent = (parseFloat(el.dataset.v) * 100).toFixed(1) + "%";
  }
});

// ═══ Boot ═══════════════════════════════════════════════════════════
async function boot() {
  try {
    const ds = await api("/runs/datasets");
    const info = ds.info || [];
    $("#dsSelect").innerHTML = (info.length
      ? info.map((d) => `<option value="${esc(d.name)}">${esc(d.label || d.name)}</option>`)
      : ds.datasets.map((n) => `<option value="${esc(n)}">${esc(n)}</option>`)
    ).join("");
    $("#apiDot").className = "dot ok";
    $("#apiStatus").textContent = "backend online";
  } catch (err) {
    $("#apiDot").className = "dot bad";
    $("#apiStatus").textContent = "backend unreachable";
    toast(`Backend unreachable: ${err.message}`, "err");
    setTimeout(boot, 2500);
    return;
  }
  try {
    const p = await api("/state/persistence");
    const files = p.files ? p.files.length : (p.count ?? "");
    if (files) $("#memFiles").textContent = `${files} memory files on disk`;
  } catch {}
  loadGoal();
  // resume an in-flight run if the window reloaded
  try {
    const runs = await api("/runs");
    const active = (runs.runs || []).find((r) => r.status === "running" || r.status === "paused");
    if (active) {
      currentRun = active.run_id;
      setRunChip(`${active.status} · ${active.dataset_name}`);
      $("#runControls").classList.remove("hidden");
      $("#startBtn").disabled = true;
      $("#startBtn .btn-label").textContent = "Running…";
      connectWs(currentRun);
    }
  } catch {}
}

boot();
