/* Mission Control — launch runs, live charts, pipeline, memory counters */

"use strict";

const accChart = new LineChart($("#accChart"), $("#accTip"), [
  { key: "accuracy", label: "stage", colorVar: "--s1" },
  { key: "best_accuracy", label: "best", colorVar: "--s2" },
]);
const stressChart = new LineChart($("#stressChart"), $("#stressTip"), [
  { key: "system_stress", label: "stress", colorVar: "--s3" },
  { key: "regret", label: "regret", colorVar: "--s4" },
]);
const allocChart = new LineChart($("#allocChart"), $("#allocTip"), [
  { key: "pso_share", label: "PSO", colorVar: "--s5" },
  { key: "reason_share", label: "reasoning", colorVar: "--acc" },
]);

let _ws = null;
let _heroTween = null;

function setHero(acc) {
  const el = $("#heroAcc");
  const from = parseFloat(el.dataset.v || "0") || 0;
  const to = acc || 0;
  el.dataset.v = to;
  cancelAnimationFrame(_heroTween);
  const circ = 2 * Math.PI * 52;
  $("#ringFill").style.strokeDasharray = String(circ);
  $("#ringFill").style.strokeDashoffset = String(circ * (1 - Math.max(0, Math.min(1, to))));
  if (document.hidden) { el.textContent = (to * 100).toFixed(1) + "%"; return; }
  const t0 = performance.now();
  const tick = (t) => {
    const p = Math.min(1, (t - t0) / 700);
    const ease = 1 - Math.pow(1 - p, 3);
    el.textContent = ((from + (to - from) * ease) * 100).toFixed(1) + "%";
    if (p < 1) _heroTween = requestAnimationFrame(tick);
  };
  _heroTween = requestAnimationFrame(tick);
}

function setRunChip(text, cls = "") {
  const chip = $("#runChip");
  chip.classList.remove("hidden", "err", "done");
  if (cls) chip.classList.add(cls);
  $("#runChipText").textContent = text;
  $("#navRunDot").classList.toggle("hidden", !text || cls === "done" || cls === "err");
}

function renderPipeline(pipeline) {
  const el = $("#pipelineFlow");
  if (!el || !pipeline || !pipeline.length) return;
  el.innerHTML = pipeline.map((step, i) => `
    <div class="pipe-step ${step.active === false ? "dim" : "on"}" style="--i:${i}">
      <div class="pipe-ic">${esc(step.icon || "•")}</div>
      <div class="pipe-layer">${esc(step.layer || "")}</div>
      <div class="pipe-title">${esc(step.title || "")}</div>
      <div class="pipe-detail">${esc(step.detail || "")}</div>
    </div>`).join("");
}

function updateMemCounters(mem) {
  if (!mem) return;
  $("#mcJournal").textContent = fmtInt(mem.journal_entries ?? mem.journal ?? 0);
  $("#mcPrinciples").textContent = fmtInt(mem.principles ?? 0);
  const nodes = mem.network_nodes ?? mem.nodes ?? 0;
  const edges = mem.network_edges ?? mem.edges ?? 0;
  $("#mcNetwork").textContent = `${fmtInt(nodes)}/${fmtInt(edges)}`;
  $("#mcArchive").textContent = fmtInt(mem.archived_nodes_edges ?? mem.archive ?? 0);
  $("#mcEpisodes").textContent = fmtInt(mem.episodes ?? 0);
  $("#mcCausal").textContent = fmtInt(mem.causal_hypotheses ?? mem.causal ?? 0);
  $("#mcFormulas").textContent = fmtInt(mem.formulas ?? 0);
  $("#mcMinerR2").textContent = mem.miner_r2 != null ? fmt(mem.miner_r2, 2) : "—";
  $("#mcChampion").textContent = mem.champion_accuracy != null ? fmt(mem.champion_accuracy) : "—";
}

function onMissionStage(snap) {
  const alloc = snap.allocation || {};
  const pso = alloc.pso_fraction ?? alloc.pso ?? null;
  const reason = alloc.reasoning_fraction ?? alloc.reasoning ?? (pso != null ? 1 - pso : null);
  const pt = {
    stage: snap.stage,
    accuracy: snap.accuracy,
    best_accuracy: snap.best_accuracy,
    system_stress: snap.system_stress,
    regret: snap.regret,
    pso_share: pso,
    reason_share: reason,
  };
  accChart.push(pt);
  stressChart.push(pt);
  if (pso != null) allocChart.push(pt);
  setHero(snap.best_accuracy);
  $("#hStage").textContent = `${snap.stage}/${snap.stages_total ?? "?"}`;
  $("#hStress").textContent = fmt(snap.system_stress, 2);
  $("#hRegret").textContent = fmt(snap.regret, 2);
  $("#hMacroF1").textContent = snap.macro_f1 != null ? fmt(snap.macro_f1) : "—";
  $("#hMaturity").textContent = snap.foundation_stage || snap.maturity || "—";
  setRunChip(`running · ${snap.dataset || snap.dataset_name || "?"} · stage ${snap.stage}`);

  if (snap.decision_trace?.pipeline) renderPipeline(snap.decision_trace.pipeline);
  updateMemCounters(snap.memory);

  const feed = $("#stageFeed");
  $("#feedEmpty")?.remove();
  const item = document.createElement("div");
  item.className = "feed-item";
  const principles = (snap.triggered_principles || []).length;
  const nodes = (snap.activated_nodes || []).length;
  const tags = [];
  if (principles) tags.push(`<span class="ftag cool">${principles} principles</span>`);
  if (nodes) tags.push(`<span class="ftag">${nodes} nodes</span>`);
  if (snap.intervention_variable) tags.push(`<span class="ftag warm">test ${esc(snap.intervention_variable)}</span>`);
  item.innerHTML = `
    <span class="feed-stage">#${snap.stage}</span>
    <span class="feed-m"><b>${fmt(snap.accuracy)}</b><span>acc</span></span>
    <span class="feed-m"><b>${fmt(snap.system_stress, 2)}</b><span>stress</span></span>
    <span class="feed-m"><b>${fmt(snap.regret, 2)}</b><span>regret</span></span>
    <div class="feed-tags">${tags.join("")}</div>`;
  feed.appendChild(item);
  while (feed.children.length > 80) feed.removeChild(feed.firstChild);
}

function connectRunWs(runId) {
  if (_ws) { try { _ws.close(); } catch {} }
  const proto = location.protocol === "https:" ? "wss" : "ws";
  _ws = new WebSocket(`${proto}://${location.host}/ws/runs/${runId}`);
  _ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === "stage" || msg.stage != null) {
      UIL.fireStage(msg);
      onMissionStage(msg);
    } else if (msg.type === "finished") {
      setRunChip(`${msg.status} · acc ${fmt(msg.accuracy)}`, "done");
      $("#runControls").classList.add("hidden");
      $("#startBtn").disabled = false;
      $("#startBtn .btn-label").textContent = "Start run";
      toast(`Run ${msg.status} — accuracy ${fmt(msg.accuracy)}`, "ok");
      UIL.fireRunEvent(msg);
      refreshRunsTable();
    } else if (msg.type === "error") {
      setRunChip("error", "err");
      $("#runControls").classList.add("hidden");
      $("#startBtn").disabled = false;
      $("#startBtn .btn-label").textContent = "Start run";
      toast(`Run error: ${msg.error}`, "err");
      UIL.fireRunEvent(msg);
    }
  };
}

async function loadDatasets() {
  const sel = $("#dsSelect");
  try {
    const info = await api("/runs/datasets");
    const rows = info.info || info.datasets || [];
    sel.innerHTML = rows.map((d) => {
      const name = typeof d === "string" ? d : (d.name || d);
      const label = typeof d === "string" ? d : (d.label || d.name || name);
      const meta = typeof d === "object" && d.n_samples != null
        ? ` (${d.n_samples}×${d.n_features})` : "";
      return `<option value="${esc(name)}">${esc(label)}${esc(meta)}</option>`;
    }).join("");
  } catch (err) {
    toast(`Datasets: ${err.message}`, "err");
  }
}

async function loadSysResources() {
  try {
    const r = await api("/runs/system/resources");
    const b = r.budget || r;
    const el = $("#sysResources");
    if (!el) return;
    const ramGb = b.ram_gb ?? b.total_ram_gb ?? (b.ram_total_mb != null ? b.ram_total_mb / 1024 : null);
    el.innerHTML = [
      (b.cpu_count ?? b.cpu_count_logical) != null ? `CPU ${b.cpu_count ?? b.cpu_count_logical}` : null,
      ramGb != null ? `RAM ${fmt(ramGb, 1)} GB` : null,
      b.cpu_workers != null ? `${b.cpu_workers} workers` : null,
      b.gpu ? `GPU ${esc(String(b.gpu))}` : null,
    ].filter(Boolean).join(" · ") || "resource budget auto-scales on start";
  } catch { /* optional */ }
}

async function refreshRunsTable() {
  try {
    const res = await api("/runs");
    const runs = res.runs || [];
    const tb = $("#runsTable tbody");
    tb.innerHTML = runs.map((r) => `
      <tr>
        <td><code>${esc(r.run_id)}</code></td>
        <td>${esc(r.dataset_name)}</td>
        <td>${esc(r.classifier)}</td>
        <td><span class="badge st-${esc(r.status)}">${esc(r.status)}</span></td>
        <td>${r.last_stage ?? "—"} / ${r.stages_total ?? "—"}</td>
        <td class="muted">${esc(r.error || "")}</td>
      </tr>`).join("") || `<tr><td colspan="6" class="muted">No runs this session</td></tr>`;
  } catch (err) { toast(`Runs: ${err.message}`, "err"); }
}

$$("#clfSeg button").forEach((b) => {
  b.addEventListener("click", () => {
    $$("#clfSeg button").forEach((x) => x.classList.remove("on"));
    b.classList.add("on");
    $("#symModeField").style.display = b.dataset.v === "SymbolicFormula" ? "" : "none";
  });
});

$("#pathAPreset")?.addEventListener("click", () => {
  $$("#clfSeg button").forEach((x) => x.classList.toggle("on", x.dataset.v === "SymbolicFormula"));
  $("#symModeField").style.display = "";
  $("#symMode").value = "auto";
  const opt = [...$("#dsSelect").options].find((o) => /breast/i.test(o.value));
  if (opt) $("#dsSelect").value = opt.value;
  $("#targetAccInput").value = "0.80";
  $("#maxStagesInput").value = "300";
  $("#stagesInput").value = "80";
  $("#boostInput").value = "40";
  toast("Path A preset applied", "ok");
});

$("#startBtn").addEventListener("click", async () => {
  const btn = $("#startBtn");
  const targetRaw = $("#targetAccInput").value;
  const body = {
    dataset_name: $("#dsSelect").value,
    classifier: $("#clfSeg .on").dataset.v,
    seed: parseInt($("#seedInput").value, 10) || 42,
    stages: parseInt($("#stagesInput").value, 10) || 8,
    n_particles: parseInt($("#particlesInput").value, 10) || 15,
    pso_iters: parseInt($("#psoItersInput").value, 10) || 15,
    boosting_rounds: parseInt($("#boostInput").value, 10) || 30,
    max_stages: parseInt($("#maxStagesInput").value, 10) || 300,
    symbolic_learning_mode: $("#symMode").value,
  };
  if (targetRaw !== "") body.target_accuracy = parseFloat(targetRaw);
  btn.disabled = true;
  btn.querySelector(".btn-label").textContent = "Launching…";
  try {
    const res = await api("/runs", { method: "POST", body: JSON.stringify(body) });
    UIL.currentRun = res.run_id;
    accChart.reset(); stressChart.reset(); allocChart.reset();
    $("#stageFeed").innerHTML = "";
    setHero(0);
    setRunChip(`starting · ${body.dataset_name}`);
    $("#runControls").classList.remove("hidden");
    $("#pauseBtn").textContent = "Pause";
    btn.querySelector(".btn-label").textContent = "Running…";
    connectRunWs(res.run_id);
    toast(`Run started on ${body.dataset_name} (${body.classifier})`, "ok");
    refreshRunsTable();
  } catch (err) {
    btn.disabled = false;
    btn.querySelector(".btn-label").textContent = "Start run";
    toast(`Could not start: ${err.message}`, "err");
  }
});

$("#stopBtn").addEventListener("click", async () => {
  if (!UIL.currentRun) return;
  try {
    await api(`/runs/${UIL.currentRun}/stop`, { method: "POST" });
    toast("Stop requested — finishing current stage", "");
  } catch (err) { toast(err.message, "err"); }
});

$("#pauseBtn").addEventListener("click", async () => {
  if (!UIL.currentRun) return;
  const btn = $("#pauseBtn");
  try {
    if (btn.textContent === "Pause") {
      await api(`/runs/${UIL.currentRun}/pause`, { method: "POST" });
      btn.textContent = "Resume";
      setRunChip("paused");
    } else {
      await api(`/runs/${UIL.currentRun}/resume`, { method: "POST" });
      btn.textContent = "Pause";
    }
  } catch (err) { toast(err.message, "err"); }
});

$("#runsRefresh")?.addEventListener("click", refreshRunsTable);

UIL.registerShow("mission", () => {
  loadDatasets();
  loadSysResources();
  refreshRunsTable();
});
UIL.onPoll(() => { if (UIL.activeView === "mission") refreshRunsTable(); });

// formula usage log from stages
UIL.onStage((snap) => {
  if (!snap.symbolic_formula_id && !snap.formula_id) return;
  const log = $("#formulaUsageLog");
  if (!log) return;
  if (log.querySelector(".empty")) log.innerHTML = "";
  const row = document.createElement("div");
  row.className = "feed-item";
  row.innerHTML = `<span class="feed-stage">#${snap.stage}</span>
    <span>${esc(String(snap.symbolic_formula_id || snap.formula_id))}</span>
    <span class="muted">acc ${fmt(snap.accuracy)}</span>`;
  log.prepend(row);
});
