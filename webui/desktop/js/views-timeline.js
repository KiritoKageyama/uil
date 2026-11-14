/* Timeline & Replay — VersionStore scrubber + journal retrace */

"use strict";

async function initTimeline() {
  try {
    const latest = await api("/state/replay");
    const max = Math.max(1, latest.seq || 1);
    const slider = $("#seqSlider");
    slider.max = max;
    slider.value = max;
    $("#seqMax").textContent = `of ${fmtInt(max)}`;
    $("#seqLabel").textContent = fmtInt(max);
    await renderReplayBars(max);
    await loadJournal();
    await fillRetraceDatasets();
  } catch (err) { toast(`Timeline: ${err.message}`, "err"); }
}

async function renderReplayBars(seq) {
  try {
    const data = await api(`/state/replay?seq=${seq}`);
    const state = data.state || {};
    const rows = Object.entries(state).map(([name, bucket]) => ({
      name,
      value: bucket && typeof bucket === "object" ? Object.keys(bucket).length : 0,
    }));
    renderHBars($("#replayBars"), rows, { wide: true });
  } catch {
    $("#replayBars").innerHTML = `<div class="empty small"><p>Cannot reconstruct at this seq</p></div>`;
  }
}

$("#seqSlider").addEventListener("input", async () => {
  const seq = parseInt($("#seqSlider").value, 10);
  $("#seqLabel").textContent = fmtInt(seq);
  await renderReplayBars(seq);
});

async function loadJournal() {
  try {
    const res = await api("/state/journal/recent");
    const rows = res.entries || res.journal || [];
    $("#journalTable tbody").innerHTML = rows.slice(0, 80).map((e) => `
      <tr>
        <td>${esc(e.dataset_name || e.dataset || "—")}</td>
        <td>${esc(String(e.stage ?? "—"))}</td>
        <td>${fmt(e.accuracy ?? e.outcome?.accuracy)}</td>
        <td>${fmt(e.system_stress ?? e.stress, 2)}</td>
        <td>${fmt(e.regret, 2)}</td>
        <td class="muted">${esc(Object.keys(e.action || e.best_action || {}).slice(0, 4).join(", ") || "—")}</td>
      </tr>`).join("") || `<tr><td colspan="6" class="muted">No journal entries</td></tr>`;
  } catch {
    $("#journalTable tbody").innerHTML = `<tr><td colspan="6" class="muted">Journal unavailable</td></tr>`;
  }
}

async function fillRetraceDatasets() {
  try {
    const info = await api("/runs/datasets");
    const rows = info.info || info.datasets || [];
    const list = rows.map((d) => (typeof d === "string" ? d : d.name)).filter(Boolean);
    $("#retraceDatasets").innerHTML = list.map((n) => `<option value="${esc(n)}">`).join("");
  } catch { /* ignore */ }
}

let _retraceEntries = [];

$("#retraceLoad").addEventListener("click", async () => {
  const ds = $("#retraceDataset").value.trim();
  if (!ds) { toast("Enter a dataset name", "err"); return; }
  const fold = $("#retraceFold").value;
  try {
    let path = `/retrace/${encodeURIComponent(ds)}`;
    if (fold !== "") path += `?fold=${encodeURIComponent(fold)}`;
    const res = await api(path);
    _retraceEntries = res.entries || res.stages || [];
    const scrub = $("#retraceScrubber");
    scrub.max = Math.max(0, _retraceEntries.length - 1);
    scrub.value = 0;
    showRetrace(0);
    toast(`Loaded ${_retraceEntries.length} journal stages`, "ok");
  } catch (err) { toast(err.message, "err"); }
});

$("#retraceScrubber").addEventListener("input", () => {
  showRetrace(parseInt($("#retraceScrubber").value, 10) || 0);
});

function showRetrace(i) {
  const e = _retraceEntries[i];
  $("#retracePos").textContent = e ? `entry ${i + 1} / ${_retraceEntries.length}` : "";
  $("#retraceDetail").innerHTML = e
    ? `<pre class="pre-panel">${esc(JSON.stringify(e, null, 2))}</pre>`
    : `<div class="empty small"><p>No entry</p></div>`;
}

$("#journalRefresh").addEventListener("click", loadJournal);
UIL.registerShow("timeline", initTimeline);
