// UIL Live Console — wires WebSocket snapshots to all visual panels.
const App = (() => {
  let currentRunId = null;

  function activeRunId() {
    return currentRunId;
  }

  function initTabs() {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.onclick = () => {
        document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
        if (btn.dataset.tab === "network") NetworkView.refresh(currentRunId);
        if (btn.dataset.tab === "formulas") FormulaView.refresh(currentRunId);
        if (btn.dataset.tab === "principles") { refreshPrinciples(); refreshCausal(); }
        if (btn.dataset.tab === "hitl") { HitlReview.loadCells(); HitlReview.loadLatestReport(); }
        if (btn.dataset.tab === "memory") { MemoryProof.refreshPersistence(); MemoryProof.loadEpisodes(); }
        if (btn.dataset.tab === "external") { ExternalFeed.loadAblationReports(); }
        if (btn.dataset.tab === "retrace") { /* loaded on demand */ }
      };
    });
  }

  function setBadge(status) {
    const el = document.getElementById("run-status-badge");
    el.className = `badge badge-${status}`;
    el.textContent = currentRunId ? `${status} · ${currentRunId}` : "no active run";
  }

  function onSnapshot(s) {
    if (!s || (s.stage == null && s.event !== "started" && s.type !== "started")) return;

    if (s.event === "started" || s.type === "started") {
      setBadge("running");
    } else if (s.stage >= 0) {
      setBadge("running");
    }

    Charts.onSnapshot(s);
    DecisionTrace.onSnapshot(s);
    if (s.memory) MemoryProof.onMemorySnapshot(s.memory);
    FormulaView.onSnapshot(s);
    NetworkView.onSnapshot(s);
    updateActiveNow(s);

    if (s.stage >= 0 && s.stage % 5 === 0) {
      MemoryProof.refreshPersistence();
    }
    if (s.stage >= 0 && s.stage % 10 === 0) {
      refreshPrinciples();
      refreshCausal();
    }
  }

  async function loadDatasets() {
    const { datasets, info } = await Api.get("/runs/datasets");
    const sel = document.getElementById("dataset-select");
    const byName = {};
    (info || []).forEach((d) => { byName[d.name] = d; });
    sel.innerHTML = datasets.map((d) => {
      const meta = byName[d];
      const label = meta
        ? `${d} — ${meta.n_samples}×${meta.n_features}, ${meta.n_classes} classes (${meta.category})`
        : d;
      return `<option value="${d}">${label}</option>`;
    }).join("");
  }

  async function startRun() {
    const btn = document.getElementById("btn-start-run");
    btn.disabled = true;
    try {
    MemoryProof.snapshotBaseline();
    DecisionTrace.reset();
    Charts.reset();
    MemoryProof.reset();
    FormulaView.reset();

    const targetRaw = document.getElementById("cfg-target-acc").value.trim();
    const stages = parseInt(document.getElementById("cfg-stages").value, 10);
    const n_particles = parseInt(document.getElementById("cfg-particles").value, 10);
    const pso_iters = parseInt(document.getElementById("cfg-pso-iters").value, 10);
    const boosting_rounds = parseInt(document.getElementById("cfg-boosting").value, 10);
    const max_stages = parseInt(document.getElementById("cfg-max-stages").value, 10);
    const seed = parseInt(document.getElementById("cfg-seed").value, 10);
    if ([stages, n_particles, pso_iters, boosting_rounds, max_stages, seed].some((n) => !Number.isFinite(n) || n < 1)) {
      throw new Error("Stages, particles, PSO iters, boosting, max stages, and seed must be positive numbers.");
    }
    const req = {
      dataset_name: document.getElementById("dataset-select").value,
      classifier: document.getElementById("classifier-select").value,
      seed,
      stages,
      n_particles,
      pso_iters,
      boosting_rounds,
      max_stages,
    };
    if (!req.dataset_name) throw new Error("Select a dataset first.");
    if (targetRaw) {
      const target = parseFloat(targetRaw);
      if (!Number.isFinite(target) || target <= 0 || target > 1) {
        throw new Error("Target accuracy must be between 0 and 1 (e.g. 0.80 for 80%).");
      }
      req.target_accuracy = target;
    }
    if (document.getElementById("classifier-select").value === "SymbolicFormula") {
      req.symbolic_learning_mode = document.getElementById("cfg-symbolic-mode").value;
    }
    const record = await Api.post("/runs", req);
    if (record.stopped_previous?.length) {
      console.warn("Stopped previous runs:", record.stopped_previous.join(", "));
    }
    currentRunId = record.run_id;
    setBadge(record.status);

    WsClient.onSnapshot = onSnapshot;
    WsClient.onFinished = (msg) => {
      stopHeartbeat();
      setBadge(msg.status || msg.target_reached ? "completed ✓ target" : "completed");
      if (msg.target_accuracy != null || msg.target_reached) {
        const hit = msg.target_reached ? "✓ target reached" : "✗ target not reached";
        const goal = msg.target_accuracy != null ? (msg.target_accuracy * 100).toFixed(0) : "80";
        alert(`Finished: ${((msg.accuracy || msg.best_accuracy || 0) * 100).toFixed(1)}% (goal ${goal}%) — ${hit}`);
      }
      MemoryProof.refreshPersistence();
      MemoryProof.loadEpisodes();
      refreshPrinciples();
      refreshCausal();
      FormulaView.refresh(currentRunId);
      NetworkView.refresh(currentRunId);
    };
    WsClient.onError = (msg) => { stopHeartbeat(); setBadge("error"); alert(`Run error: ${msg.error}`); };
    WsClient.connect(currentRunId);
    startHeartbeat();
    refreshRunsTable();
    document.querySelector('[data-tab="trace"]').click();
    } catch (e) {
      setBadge("error");
      alert(`Could not start run: ${e.message || e}`);
    } finally {
      btn.disabled = false;
    }
  }

  async function loadSystemResources() {
    const el = document.getElementById("system-resources");
    if (!el) return;
    try {
      const { budget } = await Api.get("/runs/system/resources");
      const gpu = budget.gpu_available
        ? `${budget.gpu_name || "GPU"} · ${(budget.gpu_vram_total_mb / 1024).toFixed(1)} GB VRAM (${Math.round(budget.target_fraction * 100)}% budget = ${(budget.gpu_vram_mb / 1024).toFixed(1)} GB)`
        : "GPU not detected (CPU-only path)";
      el.innerHTML = [
        `<div><span class="k">CPU</span>${budget.cpu_count_logical} threads · ${budget.cpu_workers} workers</div>`,
        `<div><span class="k">RAM</span>${(budget.ram_total_mb / 1024).toFixed(0)} GB total · ${(budget.ram_budget_mb / 1024).toFixed(0)} GB budget</div>`,
        `<div><span class="k">GPU</span>${gpu}</div>`,
        `<div><span class="k">Parallel eval</span>${budget.parallel_eval_workers} workers · ${budget.intra_fit_threads} BLAS threads/slot</div>`,
      ].join("");
    } catch (_) {
      el.textContent = "Could not load system resources.";
    }
  }

  async function refreshRunsTable() {
    const { runs } = await Api.get("/runs");
    const tbody = document.querySelector("#runs-table tbody");
    tbody.innerHTML = runs.map((r) => `
      <tr class="${r.status === "error" ? "row-error" : ""}">
        <td><a href="#" data-run="${r.run_id}" class="select-run">${r.run_id}</a></td>
        <td>${r.dataset_name}</td><td>${r.classifier}</td><td>${r.status}</td><td>${r.last_stage ?? ""}</td>
        <td class="err-cell">${r.error ? r.error.slice(0, 80) : ""}</td>
      </tr>
    `).join("");
    tbody.querySelectorAll(".select-run").forEach((a) => {
      a.onclick = (e) => {
        e.preventDefault();
        currentRunId = a.dataset.run;
        setBadge("watching");
        WsClient.onSnapshot = onSnapshot;
        WsClient.connect(currentRunId);
        startHeartbeat();
        pollRunHeartbeat();
      };
    });
  }

  function qs() {
    return currentRunId ? `?run_id=${currentRunId}` : "";
  }

  function updateActiveNow(s) {
    const grid = document.getElementById("active-now");
    const item = (k, v) => `<div><span class="k">${k}</span>${v}</div>`;
    grid.innerHTML = [
      item("stage", `${s.stage + 1} / ${s.stages_total}`),
      item("stage CPU", s.stage_cpu_seconds != null ? `${s.stage_cpu_seconds.toFixed(1)}s` : "—"),
      item("val accuracy", `${(s.accuracy * 100).toFixed(2)}% (best ${(s.best_accuracy * 100).toFixed(2)}%)`),
      item("target", s.target_accuracy != null
        ? `${(s.target_accuracy * 100).toFixed(0)}% — ${s.best_accuracy >= s.target_accuracy ? "HIT" : "not yet"}`
        : "(none — runs to max stages)"),
      item("maturity", `${s.foundation_stage} · ${(s.maturity * 100).toFixed(0)}%`),
      item("search mix", `${s.pso_actions} PSO + ${s.reasoning_actions} reasoning`),
      item("principles fired", (s.triggered_principles || []).join(", ") || "(none)"),
      item("KG activated", (s.activated_nodes || []).join(", ") || "(none)"),
      item("overrides", (s.action_overrides || []).join(", ") || "(none)"),
      item("investigation", s.investigation_strategy ?? "(none)"),
      item("causal test", s.intervention_variable ?? "(none)"),
      item("formula", s.symbolic_formula_id || s.formula_id || "(default loss)"),
      item("ensemble", (s.symbolic_formula_ids || []).join(", ") || "(single)"),
      item("hybrid blend", s.hybrid_blend
        ? Object.entries(s.hybrid_blend).map(([k, v]) => `${k}:${(v * 100).toFixed(0)}%`).join(" ")
        : "(n/a)"),
      item("warm-start", s.episodic_warm_start ? "YES (episodic)" : "no"),
      item("champion", s.memory?.champion_accuracy != null
        ? `${(s.memory.champion_accuracy * 100).toFixed(1)}% loaded`
        : "(none)"),
      item("resources", s.resource_budget
        ? `${s.resource_budget.cpu_workers} CPU · GPU ${s.resource_budget.gpu_vram_total_mb ? (s.resource_budget.gpu_vram_total_mb / 1024).toFixed(1) + " GB" : "n/a"} · ${s.resource_budget.parallel_eval_workers} parallel`
        : "(default)"),
      item("low-data aug", s.train_augmented ? "YES (bootstrap ≥15)" : "no"),
    ].join("");
  }

  async function loadGovernance() {
    const el = document.getElementById("governance-panel");
    if (!el) return;
    try {
      const { governance } = await Api.get("/state/governance");
      if (!governance || !governance.journal_rules) {
        el.innerHTML = "<p class=\"hint\">No governance manifest yet — starts on first run or memory clear.</p>";
        return;
      }
      const list = (items) => `<ul>${items.map((x) => `<li>${x}</li>`).join("")}</ul>`;
      el.innerHTML = `
        <h3>Journal (numbers only)</h3>${list(governance.journal_rules)}
        <h3>Principles (hypotheses)</h3>${list(governance.principle_rules)}
        <h3>SymbolicFormula</h3>${list(governance.symbolic_formula_rules)}
        <h3>Path A curriculum</h3>${list(governance.path_a_curriculum)}
      `;
    } catch (_) {
      el.innerHTML = "<p class=\"hint\">Governance manifest unavailable.</p>";
    }
  }

  async function refreshPrinciples() {
    const { principles } = await Api.get(`/state/principles${qs()}`);
    document.querySelector("#principles-table tbody").innerHTML = principles.map((p) => `
      <tr>
        <td>${p.principle_id}</td><td>${p.tier}</td>
        <td>${p.evidence_status || "hypothesis"}</td>
        <td>${p.confidence?.toFixed?.(2)}</td>
        <td>${p.evidence_count}</td>
        <td><code>${JSON.stringify(p.conditions)}</code></td>
      </tr>
    `).join("") || "<tr><td colspan='6'>No principles yet — need ~30 journal entries</td></tr>";
  }

  async function refreshCausal() {
    const { hypotheses } = await Api.get(`/state/causal${qs()}`);
    document.querySelector("#causal-table tbody").innerHTML = hypotheses.map((h) => `
      <tr>
        <td>${h.hypothesis_id}</td><td>${h.variable}</td><td>${h.direction}</td>
        <td>${h.status}</td><td>${h.causal_strength?.toFixed?.(2)}</td>
      </tr>
    `).join("") || "<tr><td colspan='5'>No causal hypotheses yet</td></tr>";
  }

  let heartbeatTimer = null;

  let lastSeenStage = -1;
  let lastStageWallMs = Date.now();

  async function pollRunHeartbeat() {
    if (!currentRunId) return;
    try {
      const r = await Api.get(`/runs/${currentRunId}`);
      if (r.status && r.status !== "running") {
        setBadge(r.status);
        stopHeartbeat();
        if (r.status === "error" && r.error) {
          document.getElementById("run-info").innerHTML =
            `<p class="err-cell"><strong>Run failed:</strong> ${r.error}</p>`;
          alert(`Run error: ${r.error}`);
        }
      } else if (r.status === "running") {
        const snap = r.last_snapshot;
        if (snap && snap.stage != null) {
          if (snap.stage > lastSeenStage) {
            lastSeenStage = snap.stage;
            lastStageWallMs = Date.now();
          }
          const elapsed = ((Date.now() - lastStageWallMs) / 1000).toFixed(0);
          const target = snap.target_accuracy;
          const best = snap.best_accuracy ?? 0;
          const tgtNote = target != null
            ? ` · goal ${(target * 100).toFixed(0)}% (best ${(best * 100).toFixed(1)}%)`
            : "";
          const slow = Date.now() - lastStageWallMs > 8000
            ? ` · computing stage ${snap.stage + 2}… (${elapsed}s)`
            : "";
          setBadge(`running${tgtNote}${slow}`);
        } else {
          setBadge("running");
        }
      }
      if (r.last_snapshot) {
        onSnapshot(r.last_snapshot);
        if (r.status === "completed" && r.last_snapshot.target_reached) {
          stopHeartbeat();
          setBadge("completed ✓ target");
        }
      }
    } catch (e) {
      console.warn("run heartbeat failed", e);
    }
  }

  function startHeartbeat() {
    if (heartbeatTimer) clearInterval(heartbeatTimer);
    heartbeatTimer = setInterval(pollRunHeartbeat, 3000);
    pollRunHeartbeat();
  }

  function stopHeartbeat() {
    if (heartbeatTimer) clearInterval(heartbeatTimer);
    heartbeatTimer = null;
  }

  function onExternalSnapshot(s) {
    DecisionTrace.onSnapshot(s);
    if (s.memory) MemoryProof.onMemorySnapshot(s.memory);
    FormulaView.onSnapshot(s);
    if (s.event === "stage" || s.type === "stage") Charts.onSnapshot(s);
  }

  function init() {
    initTabs();
    Charts.init();
    NetworkView.init();
    FormulaView.init();
    MemoryProof.init();
    ExternalFeed.init();
    Retrace.init();
    loadDatasets();
    loadSystemResources();
    refreshRunsTable();
    loadGovernance();

    document.getElementById("btn-start-run").onclick = startRun;
    document.getElementById("btn-path-a-preset").onclick = () => {
      document.getElementById("classifier-select").value = "SymbolicFormula";
      document.getElementById("cfg-symbolic-mode").value = "auto";
      document.getElementById("dataset-select").value = "real_breast_cancer";
      document.getElementById("cfg-target-acc").value = "0.80";
      document.getElementById("cfg-max-stages").value = "300";
      document.getElementById("cfg-stages").value = "80";
      document.getElementById("cfg-boosting").value = "40";
    };
    document.getElementById("btn-pause").onclick = () => currentRunId && Api.post(`/runs/${currentRunId}/pause`).then(() => setBadge("paused"));
    document.getElementById("btn-resume").onclick = () => currentRunId && Api.post(`/runs/${currentRunId}/resume`).then(() => setBadge("running"));
    document.getElementById("btn-stop").onclick = () => currentRunId && Api.post(`/runs/${currentRunId}/stop`).then(() => setBadge("stopped"));
    document.getElementById("btn-refresh-network").onclick = () => NetworkView.refresh(currentRunId);
    document.getElementById("btn-refresh-principles").onclick = () => {
      refreshPrinciples();
      loadGovernance();
    };
    document.getElementById("btn-refresh-causal").onclick = refreshCausal;

    const targetInput = document.getElementById("cfg-target-acc");
    const classifierSel = document.getElementById("classifier-select");
    classifierSel.onchange = () => {};
    targetInput.oninput = () => {};

    setInterval(refreshRunsTable, 8000);
    if (typeof HitlReview !== "undefined") HitlReview.init();
  }

  return { init, activeRunId, onExternalSnapshot, onSnapshot, loadGovernance };
})();

window.addEventListener("DOMContentLoaded", App.init);
