/* Memory Proof — persistence, architecture, champions, consciousness, changelogs */

"use strict";

let _memBaseline = null;
const memGrowthChart = new LineChart($("#memGrowthChart"), $("#memGrowthTip"), [
  { key: "principles", label: "principles", colorVar: "--s1" },
  { key: "nodes", label: "KG nodes", colorVar: "--s2" },
  { key: "journal", label: "journal", colorVar: "--s3" },
]);

function persistRows(persist) {
  const files = persist.files || persist.stores || [];
  if (Array.isArray(files)) return files;
  return Object.entries(files).map(([name, v]) => ({ name, ...v }));
}

async function loadPersistence() {
  try {
    const persist = await api("/state/persistence");
    $("#resultsDir").textContent = persist.results_dir || "results/";
    const rows = persistRows(persist);
    const base = _memBaseline || {};
    $("#persistTable tbody").innerHTML = rows.map((f) => {
      const name = f.name || f.store || f.path || "?";
      const size = f.size ?? f.bytes ?? 0;
      const delta = base[name] != null ? size - base[name] : null;
      return `<tr>
        <td>${esc(name)}</td>
        <td>${f.exists === false ? "no" : "yes"}</td>
        <td>${fmtBytes(size)}</td>
        <td>${delta == null ? "—" : (delta >= 0 ? "+" : "") + fmtBytes(Math.abs(delta))}</td>
        <td>${fmtInt(f.records ?? f.count ?? "—")}</td>
        <td>${esc(timeShort(f.mtime || f.modified))}</td>
      </tr>`;
    }).join("") || `<tr><td colspan="6" class="muted">No persistence stats</td></tr>`;

    const growing = rows.some((f) => f.exists !== false && (f.size || 0) > 0);
    const banner = $("#proofBanner");
    banner.innerHTML = growing
      ? `<b>●</b><span>Persisted memory present — ${rows.filter((f) => f.exists !== false).length} stores on disk.</span>`
      : `<b>○</b><span>Cold start — run stages to grow memory files.</span>`;
    banner.className = `proof-banner card ${growing ? "ok" : ""}`;
  } catch (err) { toast(`Persistence: ${err.message}`, "err"); }
}

async function loadMemArch() {
  try {
    const a = await api("/state/memory-architecture");
    const el = $("#memArch");
    const pairs = [
      ["working graph nodes", a.working_nodes ?? a.graph?.nodes],
      ["working edges", a.working_edges ?? a.graph?.edges],
      ["archived principles", a.archived_principles],
      ["archived formulas", a.archived_formulas],
      ["archive nodes/edges", a.archived_nodes_edges],
      ["principles (active)", a.principles],
      ["formulas (active)", a.formulas],
    ];
    el.innerHTML = pairs.map(([k, v]) =>
      `<div class="kv"><span>${esc(k)}</span><b>${esc(String(v ?? "—"))}</b></div>`
    ).join("");

    const champs = a.champions || a.dataset_champions || [];
    const list = Array.isArray(champs) ? champs : Object.entries(champs).map(([k, v]) => ({ dataset: k, ...v }));
    $("#championsList").innerHTML = list.length
      ? list.map((c) => `<div class="kv"><span>${esc(c.dataset || c.name)}</span><b>${fmt(c.best_accuracy ?? c.accuracy)}</b></div>`).join("")
      : `<div class="empty small"><p>No champions yet</p></div>`;
  } catch {
    $("#memArch").innerHTML = `<div class="empty small"><p>Architecture summary unavailable</p></div>`;
  }
}

async function loadConsciousness() {
  const ds = $("#consciousnessDs").value;
  if (!ds) return;
  try {
    const brief = await api(`/state/consciousness?dataset=${encodeURIComponent(ds)}`);
    $("#consciousnessPanel").innerHTML = `
      <pre class="pre-panel">${esc(typeof brief === "string" ? brief : JSON.stringify(brief, null, 2))}</pre>`;
  } catch (err) {
    $("#consciousnessPanel").innerHTML = `<div class="empty small"><p>${esc(err.message)}</p></div>`;
  }
}

async function loadEpisodes() {
  try {
    const res = await api("/state/episodes");
    const eps = res.episodes || [];
    $("#episodesList").innerHTML = eps.slice(-20).reverse().map((e, i) => `
      <div class="ep-card">
        <b>#${eps.length - i}</b>
        <span class="muted">${esc(JSON.stringify(e.fingerprint || {}).slice(0, 80))}</span>
        <div>strategy: ${esc(e.resolution?.strategy || e.resolution?.preferred_strategy || "—")}
          · score ${fmt(e.resolution?.score ?? e.trajectory?.final_accuracy)}</div>
      </div>`).join("") || `<div class="empty small"><p>No episodes yet</p></div>`;
  } catch (err) { toast(`Episodes: ${err.message}`, "err"); }
}

async function loadChangelogs() {
  try {
    const res = await api("/state/changelog");
    const g = res.graph || res.graph_changelog || [];
    const f = res.formula || res.formula_changelog || [];
    $("#graphChangelog").innerHTML = (Array.isArray(g) ? g : []).slice(-30).reverse().map((e) =>
      `<div class="feed-item"><span>${esc(e.action || e.op || "?")}</span><span class="muted">${esc(e.entity_id || e.node_id || "")}</span></div>`
    ).join("") || `<div class="empty small"><p>No graph changes</p></div>`;
    $("#formulaChangelog").innerHTML = (Array.isArray(f) ? f : []).slice(-30).reverse().map((e) =>
      `<div class="feed-item"><span>${esc(e.action || e.op || "?")}</span><span class="muted">${esc(e.formula_id || "")}</span></div>`
    ).join("") || `<div class="empty small"><p>No formula changes</p></div>`;
  } catch { /* optional */ }
}

async function fillConsciousnessDatasets() {
  try {
    const info = await api("/runs/datasets");
    const rows = info.info || info.datasets || [];
    const sel = $("#consciousnessDs");
    const cur = sel.value;
    sel.innerHTML = `<option value="">— dataset —</option>` +
      rows.map((d) => {
        const name = typeof d === "string" ? d : d.name;
        return `<option value="${esc(name)}">${esc(name)}</option>`;
      }).join("");
    if (cur) sel.value = cur;
  } catch { /* ignore */ }
}

function loadMemoryView() {
  loadPersistence();
  loadMemArch();
  loadChangelogs();
  fillConsciousnessDatasets();
}

$("#memRefresh").addEventListener("click", loadMemoryView);
$("#memBaseline").addEventListener("click", async () => {
  const persist = await api("/state/persistence");
  _memBaseline = {};
  for (const f of persistRows(persist)) {
    _memBaseline[f.name || f.store || f.path] = f.size ?? f.bytes ?? 0;
  }
  toast("Baseline set — Δ column will show growth", "ok");
  loadPersistence();
});
$("#memTransferHelp").addEventListener("click", () => {
  toast("1) Set baseline  2) Run dataset A  3) Run dataset B  4) Refresh — look for Δ growth & episodes", "ok");
});
$("#memClear").addEventListener("click", async () => {
  if (!confirm("Clear ALL UIL memory?\n\nJournal, principles, KG, episodes reset — seed principles reinstalled.")) return;
  try {
    await api("/state/memory/clear", { method: "POST" });
    toast("Memory cleared and bootstrapped", "ok");
    loadMemoryView();
  } catch (err) { toast(err.message, "err"); }
});
$("#consciousnessLoad").addEventListener("click", loadConsciousness);
$("#episodesLoad").addEventListener("click", loadEpisodes);

UIL.onStage((snap) => {
  const mem = snap.memory || {};
  memGrowthChart.push({
    stage: snap.stage,
    principles: mem.principles ?? 0,
    nodes: mem.network_nodes ?? mem.nodes ?? 0,
    journal: mem.journal_entries ?? mem.journal ?? 0,
  });
});

UIL.registerShow("memory", loadMemoryView);
UIL.onPoll(() => { if (UIL.activeView === "memory") loadPersistence(); });
