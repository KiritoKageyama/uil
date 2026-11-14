// Live decision trace — shows KG + DT + NN pipeline per stage.
const DecisionTrace = (() => {
  const MAX_LOG = 80;
  let history = [];

  function layerClass(layer) {
    const map = {
      ROUTE: "layer-route",
      STATE: "layer-state",
      EPISODIC: "layer-episodic",
      "DT→PRINCIPLES": "layer-dt",
      KG: "layer-kg",
      SEARCH: "layer-search",
      STRESS: "layer-stress",
      INVESTIGATE: "layer-investigate",
      CAUSAL: "layer-causal",
      "DT-MINER": "layer-miner",
    };
    return map[layer] || "layer-default";
  }

  function renderPipeline(pipeline) {
    const el = document.getElementById("pipeline-flow");
    if (!el || !pipeline) return;
    el.innerHTML = pipeline.map((step, i) => `
      <div class="pipeline-step ${layerClass(step.layer)} ${step.active === false ? "" : "active"}">
        <div class="step-num">${i + 1}</div>
        <div class="step-icon">${step.icon || "•"}</div>
        <div class="step-body">
          <div class="step-layer">${step.layer}</div>
          <div class="step-title">${step.title}</div>
          <div class="step-detail">${step.detail}</div>
          ${step.path && step.path.length ? `<div class="step-tags">${step.path.map(n => `<span class="tag">${n}</span>`).join("")}</div>` : ""}
          ${step.data && step.data.length ? `<pre class="step-data">${JSON.stringify(step.data, null, 2)}</pre>` : ""}
        </div>
      </div>
    `).join("");
  }

  function appendLog(s) {
    if (s.event === "started") {
      const route = s.decision_trace?.pipeline?.find(p => p.layer === "ROUTE");
      const routeTxt = route
        ? `route: ${route.detail}`
        : (s.route ? `route: ${s.route.domain}->${s.route.strategy}` : "");
      history.unshift({
        stage: "START",
        time: new Date().toLocaleTimeString(),
        summary: [routeTxt, route?.data?.fingerprint_id ? `fp=${route.data.fingerprint_id}` : ""]
          .filter(Boolean).join(" | ") || s.decision_trace?.pipeline?.[1]?.detail || "Run started",
        memory: s.memory,
      });
    } else if (s.event === "stage" || s.type === "stage") {
      const triggered = (s.triggered_principles || []).join(", ") || "none";
      const mined = s.decision_trace?.pipeline?.find(p => p.layer === "DT-MINER");
      history.unshift({
        stage: s.stage + 1,
        time: new Date().toLocaleTimeString(),
        summary: `acc=${(s.accuracy * 100).toFixed(1)}% stress=${s.system_stress.toFixed(2)} | principles: ${triggered}`,
        mined: mined ? mined.detail : null,
        memory: s.memory,
      });
    }
    history = history.slice(0, MAX_LOG);
    renderLog();
  }

  function renderLog() {
    const el = document.getElementById("decision-log");
    if (!el) return;
    el.innerHTML = history.map(h => `
      <div class="log-row">
        <span class="log-stage">S${h.stage}</span>
        <span class="log-time">${h.time}</span>
        <span class="log-text">${h.summary}</span>
        ${h.mined ? `<span class="log-mined">⛏ ${h.mined}</span>` : ""}
      </div>
    `).join("");
  }

  function onSnapshot(s) {
    if (s.decision_trace) renderPipeline(s.decision_trace.pipeline);
    appendLog(s);
    updateMemoryCounters(s.memory);
  }

  function updateMemoryCounters(mem) {
    if (!mem) return;
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    set("mem-journal", mem.journal_entries);
    set("mem-principles", mem.principles);
    set("mem-network", `${mem.network_nodes}/${mem.network_edges}`);
    set("mem-archive", `${mem.archive_nodes ?? 0}n / ${mem.archive_edges ?? 0}e`);
    set("mem-champion", mem.champion_accuracy != null
      ? `${(mem.champion_accuracy * 100).toFixed(1)}%` : "—");
    set("mem-episodes", mem.episodes);
    set("mem-causal", mem.causal_hypotheses);
    set("mem-miner-r2", mem.miner_r2?.toFixed?.(3) ?? mem.miner_r2);
  }

  function reset() {
    history = [];
    renderLog();
    const el = document.getElementById("pipeline-flow");
    if (el) el.innerHTML = "<p class='hint'>Start a run to see the decision pipeline live.</p>";
  }

  return { onSnapshot, reset };
})();
