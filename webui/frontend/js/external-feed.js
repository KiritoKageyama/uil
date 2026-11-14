// Polls uil_live_feed.jsonl for terminal / external runs while web UI runs.
const ExternalFeed = (() => {
  let offset = 0;
  let pollTimer = null;
  let activeSources = {};

  function sourceLabel(msg) {
    if (msg.source === "webui") return `Web ${msg.run_id || ""}`.trim();
    const script = (msg.script || "terminal").replace(/\.py$/, "");
    return `Terminal · ${script} · pid ${msg.pid}`;
  }

  function renderSources() {
    const el = document.getElementById("external-sources");
    if (!el) return;
    const rows = Object.values(activeSources);
    if (!rows.length) {
      el.innerHTML = "<p class='hint'>Waiting for terminal or web runs… Start the web UI, then run tests in another terminal.</p>";
      return;
    }
    el.innerHTML = rows.map((r) => `
      <div class="external-source-card ${r.source === "webui" ? "src-web" : "src-term"}">
        <div class="src-title">${sourceLabel(r)}</div>
        <div>${r.dataset || "?"} · ${r.classifier || "?"}</div>
        <div>stage ${r.last_stage >= 0 ? r.last_stage + 1 : "start"} / ${r.stages_total || "?"} · acc ${((r.accuracy || 0) * 100).toFixed(1)}%</div>
        ${r.isolated_state ? "<div class='warn-tag'>isolated state — not writing to shared memory</div>" : "<div class='ok-tag'>shared memory</div>"}
      </div>
    `).join("");
  }

  function renderFeed(lines) {
    const el = document.getElementById("external-feed-log");
    if (!el) return;
    const tail = lines.slice(-40).reverse();
    el.innerHTML = tail.map((m) => {
      const ev = m.event || "stage";
      const st = m.stage >= 0 ? `S${m.stage + 1}` : "START";
      return `<div class="log-row">
        <span class="log-stage">${st}</span>
        <span class="log-time">${sourceLabel(m)}</span>
        <span class="log-text">${ev === "started" ? "run started" : `acc=${((m.accuracy||0)*100).toFixed(1)}% stress=${(m.system_stress||0).toFixed(2)}`}</span>
      </div>`;
    }).join("");
  }

  function processLines(lines) {
    if (!lines || !lines.length) return;
    for (const msg of lines) {
      const key = msg.run_id || `${msg.source}-${msg.pid}-${msg.dataset}-${msg.ts}`;
      activeSources[key] = {
        source: msg.source,
        run_id: msg.run_id,
        pid: msg.pid,
        script: msg.script,
        dataset: msg.dataset,
        classifier: msg.classifier,
        isolated_state: msg.isolated_state,
        last_stage: msg.stage,
        stages_total: msg.stages_total,
        accuracy: msg.accuracy,
      };
      if (msg.source === "webui" && msg.run_id && App.activeRunId() === msg.run_id && App.onSnapshot) {
        App.onSnapshot(msg);
      } else if (msg.source !== "webui" && msg.decision_trace && App.onExternalSnapshot) {
        App.onExternalSnapshot(msg);
      }
    }
    renderFeed(lines);
    renderSources();
    if (lines.length) {
      MemoryProof.refreshPersistence();
    }
  }

  async function bootstrap() {
    try {
      const data = await Api.get("/state/live-feed?tail=80");
      if (data.lines?.length) {
        processLines(data.lines);
        offset = data.next_offset ?? data.total_lines ?? 0;
      }
      updateStatus(data);
    } catch (e) {
      console.warn("live-feed bootstrap failed", e);
    }
  }

  function updateStatus(data) {
    const status = document.getElementById("feed-status");
    if (!status || !data) return;
    status.textContent = data.enabled
      ? `Live feed ON · ${data.total_lines || 0} events · ${(data.feed_path || "").split(/[/\\]/).pop()}`
      : "Live feed disabled (set UIL_LIVE_FEED=0 to turn off)";
  }

  async function poll() {
    try {
      const data = await Api.get(`/state/live-feed?offset=${offset}&max_lines=50`);
      if (data.lines && data.lines.length) processLines(data.lines);
      offset = data.next_offset ?? offset;
      updateStatus(data);
    } catch (e) {
      console.warn("live-feed poll failed", e);
    }
  }

  function start() {
    stop();
    poll();
    pollTimer = setInterval(poll, 1500);
  }

  function stop() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;
  }

  function init() {
    document.getElementById("btn-clear-feed")?.addEventListener("click", async () => {
      await Api.post("/state/live-feed/clear");
      offset = 0;
      activeSources = {};
      renderSources();
      document.getElementById("external-feed-log").innerHTML = "";
    });
    document.getElementById("btn-refresh-ablation")?.addEventListener("click", loadAblationReports);
    bootstrap();
    pollTimer = setInterval(poll, 1500);
  }

  async function loadAblationReports() {
    const { reports } = await Api.get("/state/ablation-reports");
    const el = document.getElementById("ablation-reports-list");
    if (!el) return;
    if (!reports.length) {
      el.innerHTML = "<p class='hint'>No ablation reports yet. Run: python -m figpso_uil.experiments.ablation_harness</p>";
      return;
    }
    el.innerHTML = `<table><thead><tr><th>file</th><th>classifier</th><th>stages</th><th>datasets</th><th>generated</th></tr></thead><tbody>${
      reports.map((r) => `
        <tr>
          <td>${r.filename}</td>
          <td>${r.classifier || "—"}</td>
          <td>${r.stages ?? "—"}</td>
          <td>${(r.datasets || []).join(", ")}</td>
          <td>${r.generated_at || "—"}</td>
        </tr>`).join("")
    }</tbody></table>`;
  }

  function reset() {
    offset = 0;
    activeSources = {};
    renderSources();
    const el = document.getElementById("external-feed-log");
    if (el) el.innerHTML = "";
  }

  return { init, start, stop, reset, bootstrap, loadAblationReports };
})();
