// Real-time formula library — loss objectives + symbolic classifier formulas.
const FormulaView = (() => {
  let pollTimer = null;
  let selectedId = null;
  let activeFormulaId = null;
  let lastFormulas = [];
  let lastSymbolic = [];
  let usageLog = []; // { t, stage, formula_id, accuracy, dataset }

  function treeToString(node) {
    if (!node) return "?";
    if (node.var) return node.var;
    if (node.const != null) return Number(node.const).toFixed(3);
    if (node.feature != null) return `x[${node.feature}]`;
    if (node.op) {
      const args = (node.children || []).map(treeToString).join(", ");
      return `${node.op}(${args})`;
    }
    return "?";
  }

  function treeDepth(node) {
    if (!node || !node.children?.length) return 1;
    return 1 + Math.max(...node.children.map(treeDepth));
  }

  function statusClass(status) {
    if (status === "active") return "status-active";
    if (status === "seed") return "status-seed";
    if (status === "incubating") return "status-incubating";
    if (status === "retired") return "status-retired";
    return "";
  }

  function winRate(f) {
    const e = f.evidence || 0;
    if (!e) return "—";
    return `${((f.times_helped || 0) / e * 100).toFixed(0)}%`;
  }

  function fmtTs(ts) {
    if (!ts) return "—";
    try {
      return new Date(ts).toISOString().replace("T", " ").slice(0, 16);
    } catch (_) {
      return String(ts).slice(0, 16);
    }
  }

  function renderActiveBanner() {
    const el = document.getElementById("formula-active-banner");
    if (!el) return;
    if (!activeFormulaId) {
      el.className = "formula-active-banner idle";
      el.innerHTML = "<strong>No formula selected this stage</strong> — XGBoost uses default loss; SymbolicFormula picks via tournament.";
      return;
    }
    const f = [...lastFormulas, ...lastSymbolic].find((x) => x.formula_id === activeFormulaId);
    const expr = f ? treeToString(f.tree) : "(loading…)";
    el.className = "formula-active-banner live";
    el.innerHTML = `
      <strong>Active now:</strong> <code>${activeFormulaId}</code>
      ${f ? ` · ${f.name} · conf ${(f.confidence ?? 0).toFixed(2)} · best ${((f.best_accuracy ?? 0) * 100).toFixed(1)}%` : ""}
      <div class="formula-expr">${expr}</div>`;
  }

  function renderTable(formulas, tbodyId, kind) {
    const tbody = document.querySelector(`#${tbodyId} tbody`);
    if (!tbody) return;

    const filter = document.getElementById("formula-filter")?.value || "all";
    const sortBy = document.getElementById("formula-sort")?.value || "confidence";

    let rows = [...formulas];
    if (filter !== "all") rows = rows.filter((f) => f.status === filter);
    rows.sort((a, b) => {
      if (sortBy === "evidence") return (b.evidence || 0) - (a.evidence || 0);
      if (sortBy === "best_accuracy") return (b.best_accuracy || 0) - (a.best_accuracy || 0);
      if (sortBy === "name") return (a.name || "").localeCompare(b.name || "");
      return (b.confidence || 0) - (a.confidence || 0);
    });

    tbody.innerHTML = rows.map((f) => {
      const isActive = f.formula_id === activeFormulaId;
      const isSelected = f.formula_id === selectedId;
      return `
        <tr class="formula-row ${isActive ? "row-active" : ""} ${isSelected ? "row-selected" : ""}"
            data-id="${f.formula_id}" data-kind="${kind}">
          <td><code>${f.formula_id}</code>${isActive ? ' <span class="live-dot">●</span>' : ""}</td>
          <td>${f.name || "—"}</td>
          <td><span class="status-pill ${statusClass(f.status)}">${f.status}</span></td>
          <td>${(f.confidence ?? 0).toFixed(2)}</td>
          <td>${f.evidence ?? 0}</td>
          <td>${f.times_helped ?? 0}/${f.times_hurt ?? 0}</td>
          <td>${winRate(f)}</td>
          <td>${((f.best_accuracy ?? 0) * 100).toFixed(1)}%</td>
          <td class="ts-cell" title="${f.created_at || ""}">${fmtTs(f.updated_at || f.created_at)}</td>
          <td>${treeDepth(f.tree)}</td>
          <td class="origin-cell">${(f.origin || "—").replace("mutation_of_", "↳ ")}</td>
        </tr>`;
    }).join("") || `<tr><td colspan="11">No ${kind} formulas yet — run with XGBoost (custom loss) or SymbolicFormula classifier.</td></tr>`;

    tbody.querySelectorAll(".formula-row").forEach((tr) => {
      tr.onclick = () => selectFormula(tr.dataset.id, tr.dataset.kind);
    });
  }

  function renderDetail() {
    const el = document.getElementById("formula-detail");
    if (!el) return;
    const f = [...lastFormulas, ...lastSymbolic].find((x) => x.formula_id === selectedId);
    if (!f) {
      el.innerHTML = "<p class='hint'>Click a formula row to see its expression tree and performance history.</p>";
      return;
    }
    const expr = treeToString(f.tree);
    el.innerHTML = `
      <h3>${f.formula_id} · ${f.name}</h3>
      <div class="formula-meta">
        <span class="status-pill ${statusClass(f.status)}">${f.status}</span>
        <span>confidence ${(f.confidence ?? 0).toFixed(3)}</span>
        <span>evidence ${f.evidence ?? 0}</span>
        <span>helped ${f.times_helped ?? 0} / hurt ${f.times_hurt ?? 0}</span>
        <span>best acc ${((f.best_accuracy ?? 0) * 100).toFixed(2)}%</span>
        ${f.created_at ? `<span>created ${fmtTs(f.created_at)}</span>` : ""}
        ${f.updated_at ? `<span>updated ${fmtTs(f.updated_at)}</span>` : ""}
        ${f.last_change ? `<span>last: ${f.last_change}</span>` : ""}
        ${f.mutation_type ? `<span>mutation: ${f.mutation_type}</span>` : ""}
      </div>
      <pre class="formula-expr-block">${expr}</pre>
      ${f.dataset_types?.length ? `<p class="hint">Worked on: ${f.dataset_types.join(", ")}</p>` : ""}
      <details><summary>Raw tree JSON</summary><pre class="formula-json">${JSON.stringify(f.tree, null, 2)}</pre></details>`;
  }

  function renderUsageLog() {
    const el = document.getElementById("formula-usage-log");
    if (!el) return;
    const tail = usageLog.slice(-30).reverse();
    el.innerHTML = tail.map((u) => {
      const tour = (u.tournament || [])
        .sort((a, b) => (b.score || 0) - (a.score || 0))
        .slice(0, 3)
        .map((t) => `${t.formula_id?.slice(0, 12) || "?"}:${t.score}`)
        .join(" · ");
      return `
      <div class="log-row">
        <span class="log-stage">S${u.stage + 1}</span>
        <span class="log-time">${u.dataset || "?"}</span>
        <span class="log-text"><code>${u.formula_id || "(default)"}</code> → val ${(u.accuracy * 100).toFixed(1)}%</span>
        ${tour ? `<span class="log-tournament hint">tournament: ${tour}</span>` : ""}
      </div>`;
    }).join("") || "<p class='hint'>Formula tournament picks per stage appear here during a live run.</p>";
  }

  function renderChart() {
    const canvas = document.getElementById("chart-formulas");
    if (!canvas || !window.Chart) return;
    const top = [...lastFormulas]
      .filter((f) => (f.evidence || 0) > 0)
      .sort((a, b) => (b.confidence || 0) - (a.confidence || 0))
      .slice(0, 8);
    const labels = top.map((f) => f.formula_id.replace("SEED-", "S-"));
    const conf = top.map((f) => f.confidence ?? 0);
    const best = top.map((f) => (f.best_accuracy ?? 0) * 100);

    if (!FormulaView._chart) {
      FormulaView._chart = new Chart(canvas.getContext("2d"), {
        type: "bar",
        data: {
          labels,
          datasets: [
            { label: "confidence", data: conf, backgroundColor: "#58a6ff88" },
            { label: "best acc %", data: best, backgroundColor: "#3fb95088" },
          ],
        },
        options: {
          responsive: true,
          plugins: { legend: { labels: { color: "#8b949e" } } },
          scales: {
            x: { ticks: { color: "#8b949e", maxRotation: 45 } },
            y: { ticks: { color: "#8b949e" }, beginAtZero: true, max: 100 },
          },
        },
      });
      return;
    }
    FormulaView._chart.data.labels = labels;
    FormulaView._chart.data.datasets[0].data = conf;
    FormulaView._chart.data.datasets[1].data = best;
    FormulaView._chart.update("none");
  }

  function selectFormula(id, kind) {
    selectedId = id;
    renderTable(lastFormulas, "formulas-loss-table", "loss");
    renderTable(lastSymbolic, "formulas-symbolic-table", "symbolic");
    renderDetail();
  }

  async function refresh(runId) {
    const q = runId ? `?run_id=${runId}` : "";
    try {
      const data = await Api.get(`/state/formulas${q}`);
      lastFormulas = data.formulas || [];
      lastSymbolic = data.symbolic_formulas || [];
      if (data.active_formula_id) activeFormulaId = data.active_formula_id;

      const summary = document.getElementById("formula-summary");
      if (summary) {
        summary.textContent = `${lastFormulas.length} loss formulas · ${lastSymbolic.length} symbolic · ${lastFormulas.filter((f) => f.status === "active").length} active`;
      }

      renderActiveBanner();
      renderTable(lastFormulas, "formulas-loss-table", "loss");
      renderTable(lastSymbolic, "formulas-symbolic-table", "symbolic");
      if (selectedId) renderDetail();
      renderChart();
    } catch (e) {
      console.warn("formula refresh failed", e);
    }
  }

  function onSnapshot(s) {
    if (s.formula_id) activeFormulaId = s.formula_id;
    if (s.symbolic_formula_id) activeFormulaId = s.symbolic_formula_id;
    if (s.stage >= 0 && (s.formula_id || s.symbolic_formula_id)) {
      usageLog.push({
        t: Date.now(),
        stage: s.stage,
        formula_id: s.formula_id || s.symbolic_formula_id,
        accuracy: s.accuracy ?? 0,
        tournament: s.formula_tournament || [],
        dataset: s.dataset,
      });
      if (usageLog.length > 200) usageLog.shift();
    }
    renderActiveBanner();
    renderUsageLog();
    if (document.getElementById("tab-formulas")?.classList.contains("active")) {
      refresh(App.activeRunId());
    }
  }

  function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(() => {
      if (document.getElementById("tab-formulas")?.classList.contains("active")) {
        refresh(App.activeRunId());
      }
    }, 3000);
  }

  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  function reset() {
    usageLog = [];
    activeFormulaId = null;
    renderUsageLog();
    renderActiveBanner();
  }

  function init() {
    document.getElementById("btn-refresh-formulas")?.addEventListener("click", () => refresh(App.activeRunId()));
    document.getElementById("formula-filter")?.addEventListener("change", () => {
      renderTable(lastFormulas, "formulas-loss-table", "loss");
      renderTable(lastSymbolic, "formulas-symbolic-table", "symbolic");
    });
    document.getElementById("formula-sort")?.addEventListener("change", () => {
      renderTable(lastFormulas, "formulas-loss-table", "loss");
      renderTable(lastSymbolic, "formulas-symbolic-table", "symbolic");
    });
    startPolling();
    refresh(null);
  }

  return { init, refresh, onSnapshot, reset, stopPolling };
})();
