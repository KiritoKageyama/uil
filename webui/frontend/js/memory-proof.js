// Memory & persistence proof — shows disk files, archive, changelogs, champions.
const MemoryProof = (() => {
  let baseline = null;
  let pollTimer = null;
  let memoryHistory = [];

  function fmtBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / (1024 * 1024)).toFixed(2)} MB`;
  }

  function fmtTs(ts) {
    if (!ts) return "—";
    try {
      return new Date(ts).toISOString().replace("T", " ").slice(0, 19);
    } catch (_) {
      return String(ts).slice(0, 19);
    }
  }

  function fmtPct(v) {
    if (v == null || Number.isNaN(v)) return "—";
    return `${(Number(v) * 100).toFixed(1)}%`;
  }

  async function refreshPersistence() {
    const data = await Api.get("/state/persistence");
    renderFiles(data);
    updateProofBanner(data);
    return data;
  }

  async function refreshArchitecture(dataset) {
    const params = new URLSearchParams();
    if (dataset) params.set("dataset", dataset);
    const rid = typeof App !== "undefined" && App.activeRunId?.();
    if (rid) params.set("run_id", rid);
    const q = params.toString() ? `?${params}` : "";
    const arch = await Api.get(`/state/memory-architecture${q}`);
    renderArchitecture(arch);
    renderChampions(arch.champions || {});
    if (arch.consciousness) renderConsciousness(arch.consciousness);
    return arch;
  }

  async function refreshChangelog() {
    const data = await Api.get("/state/changelog?limit=35");
    renderChangelogFeed("graph-changelog", data.graph, "graph");
    renderChangelogFeed("formula-changelog", data.formula, "formula");
    return data;
  }

  async function refreshAll() {
    await refreshPersistence();
    const ds = document.getElementById("consciousness-dataset")?.value || "";
    await refreshArchitecture(ds || null);
    await refreshChangelog();
  }

  function renderFiles(data) {
    const tbody = document.querySelector("#persistence-table tbody");
    if (!tbody) return;
    tbody.innerHTML = Object.entries(data.files).map(([key, f]) => {
      const delta = baseline && baseline.files[key]
        ? f.bytes - baseline.files[key].bytes
        : 0;
      const deltaStr = delta > 0 ? `<span class="delta-up">+${fmtBytes(delta)}</span>` : "";
      const highlight = ["knowledge_archive", "graph_changelog", "formula_changelog"].includes(key)
        ? "row-highlight" : "";
      return `
        <tr class="${highlight}">
          <td>${key}</td>
          <td>${f.exists ? "✓" : "—"}</td>
          <td>${fmtBytes(f.bytes)} ${deltaStr}</td>
          <td>${f.record_count >= 0 ? f.record_count : "?"}</td>
          <td class="ts-cell">${fmtTs(f.modified_utc)}</td>
          <td class="path-cell">${f.path.split(/[/\\]/).slice(-2).join("/")}</td>
        </tr>`;
    }).join("");
    const dir = document.getElementById("results-dir");
    if (dir) dir.textContent = data.results_dir;
  }

  function renderArchitecture(arch) {
    const el = document.getElementById("memory-architecture-summary");
    if (!el) return;
    const w = arch.working || {};
    const a = arch.archive || {};
    const cl = arch.changelogs || {};
    const liveTag = w.source === "live_run"
      ? " <span class='hint'>(live run)</span>" : "";
    const diskHint = w.disk_working_edges != null && w.disk_working_edges !== w.working_edges
      ? `<div class='hint'>On disk until run ends: ${w.disk_working_nodes}n / ${w.disk_working_edges}e</div>`
      : "";
    const lp = arch.last_prune;
    const pruneHint = lp
      ? `<div class='hint'>Last KG prune: ${fmtTs(lp.at)} · archived ${lp.edges_archived ?? 0} edges → ${lp.working_nodes ?? "?"}n / ${lp.working_edges ?? "?"}e working</div>`
      : "";
    el.innerHTML = [
      item(`Working KG nodes${liveTag}`, w.working_nodes ?? 0),
      item(`Working KG edges${liveTag}`, w.working_edges ?? 0),
      diskHint,
      pruneHint,
      item("Archived nodes", a.archived_nodes ?? 0),
      item("Archived edges", a.archived_edges ?? 0),
      item("Archived formulas", a.archived_formulas ?? 0),
      item("Datasets tracked", a.datasets_tracked ?? 0),
      item("Archived journal", a.archived_journal ?? 0),
      item("Archived principles", a.archived_principles ?? 0),
      item("Archived hypotheses", a.archived_hypotheses ?? 0),
      item("Formula changelog events", cl.formula?.event_count ?? 0),
    ].join("");
    function item(k, v) {
      return `<div><span class="k">${k}</span><span>${v}</span></div>`;
    }
  }

  function renderChampions(champions) {
    const el = document.getElementById("champions-list");
    if (!el) return;
    const entries = Object.entries(champions);
    if (!entries.length) {
      el.innerHTML = "<p class='hint'>No dataset champions yet — complete a run that beats prior best accuracy.</p>";
      return;
    }
    el.innerHTML = entries.map(([ds, c]) => `
      <div class="episode-card champion-card">
        <div class="ep-title">${ds}</div>
        <div>val acc <strong>${fmtPct(c.best_accuracy)}</strong> · formula <code>${c.formula_id || "?"}</code></div>
        <div class="hint">saved ${fmtTs(c.saved_at)} · updated ${fmtTs(c.updated_at)}</div>
        ${c.learning_mode ? `<div>mode: ${c.learning_mode}</div>` : ""}
      </div>
    `).join("");
  }

  function renderConsciousness(brief) {
    const el = document.getElementById("consciousness-panel");
    if (!el || !brief) return;
    const lines = (brief.abstract_summary || []).map((t) => `<li>${t}</li>`).join("");
    const avoid = (brief.avoid_formulas || []).slice(0, 8).map((f) => `
      <div class="changelog-row avoid-row">
        <span class="cl-ts">${fmtTs(f.last_seen || "")}</span>
        <span class="cl-action">avoid</span>
        <span class="cl-id"><code>${f.formula_id}</code></span>
        <span class="cl-detail">${f.summary || f.verdict}</span>
      </div>
    `).join("");
    const good = (brief.known_good_formulas || []).slice(0, 5).map((f) => `
      <div class="changelog-row good-row">
        <span class="cl-action">${f.verdict}</span>
        <span class="cl-id"><code>${f.formula_id}</code></span>
        <span class="cl-detail">${f.summary || ""}</span>
      </div>
    `).join("");
    const champ = brief.champion;
    el.innerHTML = `
      ${champ ? `<div class="proof-banner proof-partial" style="margin-bottom:10px">
        <strong>Champion:</strong> ${fmtPct(champ.best_accuracy)} with <code>${champ.formula_id || "?"}</code>
        · saved ${fmtTs(champ.saved_at)}
      </div>` : ""}
      ${lines ? `<h4>Abstract memory</h4><ul class="consciousness-list">${lines}</ul>` : ""}
      ${avoid ? `<h4>Formulas to avoid (low accuracy)</h4>${avoid}` : ""}
      ${good ? `<h4>Known-good formulas</h4>${good}` : ""}
      <p class="hint">Archive: ${brief.archive_stats?.archived_nodes ?? 0} nodes,
        ${brief.archive_stats?.archived_edges ?? 0} edges,
        ${brief.archive_stats?.archived_formulas ?? 0} formulas</p>`;
  }

  function renderChangelogFeed(containerId, block, kind) {
    const el = document.getElementById(containerId);
    if (!el) return;
    const events = block?.events || [];
    if (!events.length) {
      el.innerHTML = `<p class="hint">No ${kind} changelog events yet — start a run to record timestamped changes.</p>`;
      return;
    }
    el.innerHTML = `
      <div class="changelog-meta">${block.total_events || events.length} total · updated ${fmtTs(block.updated_at)}</div>
      ${events.map((ev) => {
        const id = kind === "graph"
          ? (ev.entity_id || "?")
          : (ev.formula_id || "?");
        const action = ev.action || "?";
        const ds = ev.dataset ? `<span class="cl-dataset">${ev.dataset}</span>` : "";
        const detail = ev.details
          ? `<span class="cl-detail">${JSON.stringify(ev.details).slice(0, 120)}</span>`
          : "";
        return `
          <div class="changelog-row">
            <span class="cl-ts">${fmtTs(ev.timestamp)}</span>
            <span class="cl-action">${action}</span>
            <span class="cl-id">${kind === "formula" ? `<code>${id}</code>` : id}</span>
            ${ds}${detail}
          </div>`;
      }).join("")}
    `;
  }

  async function loadConsciousness() {
    const ds = document.getElementById("consciousness-dataset")?.value;
    if (!ds) {
      alert("Pick a dataset first.");
      return;
    }
    const brief = await Api.get(`/state/consciousness?dataset=${encodeURIComponent(ds)}`);
    renderConsciousness(brief);
  }

  async function populateDatasetSelect() {
    const sel = document.getElementById("consciousness-dataset");
    if (!sel) return;
    try {
      const { datasets } = await Api.get("/runs/datasets");
      const current = sel.value;
      sel.innerHTML = `<option value="">— pick dataset —</option>` +
        (datasets || []).map((d) => `<option value="${d.id}">${d.label || d.id}</option>`).join("");
      if (current) sel.value = current;
    } catch (_) { /* ignore */ }
  }

  function updateProofBanner(data) {
    const el = document.getElementById("learning-proof-banner");
    if (!el) return;
    const growing = Object.entries(data.files).some(([k, f]) => {
      if (!baseline) return f.bytes > 0;
      return f.bytes > (baseline.files[k]?.bytes || 0);
    });
    const hasArchive = (data.files.knowledge_archive?.exists);
    const hasChangelog = (data.files.graph_changelog?.exists || data.files.formula_changelog?.exists);
    const hasJournal = (data.files.journal?.record_count || 0) > 0;

    if (growing && (hasJournal || hasArchive || hasChangelog)) {
      el.className = "proof-banner proof-yes";
      el.innerHTML = "<strong>✓ Learning is persisting</strong> — working memory, archive, and changelogs are updating on disk.";
    } else if (data.learning_active) {
      el.className = "proof-banner proof-partial";
      el.innerHTML = "<strong>◐ Memory exists</strong> — run more stages to see archive + changelog growth.";
    } else {
      el.className = "proof-banner proof-no";
      el.innerHTML = "<strong>○ Cold start</strong> — no persisted memory yet. Run 30+ stages to mine first principles.";
    }
  }

  function snapshotBaseline() {
    refreshAll()
      .then(() => refreshPersistence())
      .then((p) => { baseline = JSON.parse(JSON.stringify(p)); });
  }

  function onMemorySnapshot(mem) {
    if (!mem) return;
    memoryHistory.push({ t: Date.now(), ...mem });
    if (memoryHistory.length > 200) memoryHistory.shift();
    renderMemoryChart();
  }

  function renderMemoryChart() {
    const canvas = document.getElementById("chart-memory");
    if (!canvas || !window.Chart) return;
    if (!MemoryProof._chart) {
      MemoryProof._chart = new Chart(canvas.getContext("2d"), {
        type: "line",
        data: {
          labels: [],
          datasets: [
            { label: "journal", data: [], borderColor: "#58a6ff", tension: 0.2, pointRadius: 0 },
            { label: "principles", data: [], borderColor: "#2ea043", tension: 0.2, pointRadius: 0 },
            { label: "KG nodes", data: [], borderColor: "#d29922", tension: 0.2, pointRadius: 0 },
            { label: "archive nodes", data: [], borderColor: "#a371f7", tension: 0.2, pointRadius: 0 },
          ],
        },
        options: {
          animation: false,
          plugins: { title: { display: true, text: "In-memory knowledge growth (live)", color: "#8b949e" } },
        },
      });
    }
    const c = MemoryProof._chart;
    const last = memoryHistory.slice(-50);
    c.data.labels = last.map((_, i) => i);
    c.data.datasets[0].data = last.map(m => m.journal_entries);
    c.data.datasets[1].data = last.map(m => m.principles);
    c.data.datasets[2].data = last.map(m => m.network_nodes);
    c.data.datasets[3].data = last.map(m => m.archive_nodes || 0);
    c.update("none");
  }

  async function loadEpisodes() {
    const q = App.activeRunId() ? `?run_id=${App.activeRunId()}` : "";
    const { episodes, count } = await Api.get(`/state/episodes${q}`);
    const el = document.getElementById("episodes-list");
    if (!el) return;
    if (!count) {
      el.innerHTML = "<p class='hint'>No episodic memories yet. Complete a run — the next similar dataset will warm-start from it.</p>";
      return;
    }
    el.innerHTML = episodes.slice(-5).reverse().map((e, i) => `
      <div class="episode-card">
        <div class="ep-title">Episode ${count - i}</div>
        <div>n=${e.fingerprint?.n_samples}, f=${e.fingerprint?.n_features}, sparsity=${e.fingerprint?.sparsity?.toFixed?.(3)}</div>
        <div>final acc=${(e.trajectory?.final_accuracy * 100)?.toFixed?.(1)}%, stress=${e.trajectory?.max_stress?.toFixed?.(2)}</div>
        <div>principles used: ${(e.resolution?.principle_ids || []).length}</div>
      </div>
    `).join("");
  }

  function startPolling() {
    stopPolling();
    pollTimer = setInterval(() => {
      if (document.getElementById("tab-memory")?.classList.contains("active")) {
        refreshAll();
      } else {
        refreshPersistence();
      }
    }, 3000);
  }

  function stopPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;
  }

  function init() {
    populateDatasetSelect();
    snapshotBaseline();
    startPolling();
    document.getElementById("btn-refresh-persistence")?.addEventListener("click", refreshAll);
    document.getElementById("btn-snapshot-baseline")?.addEventListener("click", snapshotBaseline);
    document.getElementById("btn-load-consciousness")?.addEventListener("click", loadConsciousness);
    document.getElementById("consciousness-dataset")?.addEventListener("change", loadConsciousness);
    document.getElementById("btn-load-episodes")?.addEventListener("click", loadEpisodes);
    document.getElementById("btn-clear-memory")?.addEventListener("click", async () => {
      if (!confirm("Clear ALL UIL memory?\n\nJournal, principles, KG, episodes reset — then clean Path-A seed principles + governance are reinstalled.")) return;
      const result = await Api.post("/state/memory/clear");
      alert(result.message || "Memory cleared and bootstrapped.");
      baseline = null;
      memoryHistory = [];
      await refreshAll();
      if (window.App && App.loadGovernance) App.loadGovernance();
    });
    document.getElementById("btn-learning-test")?.addEventListener("click", () => {
      alert(
        "Learning transfer test:\n\n" +
        "1. Clear memory (button above)\n" +
        "2. Run breast_cancer Path A twice\n" +
        "3. Memory Proof — watch archive + changelogs grow\n" +
        "4. Run 2 — champion should load (~86% start, not ~63%)"
      );
    });
    document.querySelector('[data-tab="memory"]')?.addEventListener("click", () => {
      setTimeout(refreshAll, 100);
    });
  }

  return {
    init, onMemorySnapshot, refreshPersistence, refreshAll,
    snapshotBaseline, loadEpisodes, startPolling, stopPolling,
    reset: () => { memoryHistory = []; baseline = null; },
  };
})();
