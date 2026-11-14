/* Terminal / external feed + ablation reports */

"use strict";

let _feedCursor = 0;

async function pollLiveFeed() {
  try {
    const res = await api(`/state/live-feed?since=${_feedCursor}`);
    const events = res.events || res.entries || [];
    if (res.enabled === false) {
      $("#feedStatus").textContent = "feed disabled (set UIL_LIVE_FEED=1)";
      return;
    }
    $("#feedStatus").textContent = events.length
      ? `${events.length} new · cursor ${_feedCursor}`
      : "listening…";
    if (!events.length) return;
    const log = $("#feedLog");
    if (log.querySelector(".empty")) log.innerHTML = "";
    const sources = new Map();
    for (const ev of events) {
      _feedCursor = Math.max(_feedCursor, ev.seq ?? ev.offset ?? _feedCursor + 1);
      const src = ev.source || ev.dataset || "terminal";
      sources.set(src, (sources.get(src) || 0) + 1);
      const row = document.createElement("div");
      row.className = "feed-item";
      row.innerHTML = `<span class="feed-stage">${esc(src)}</span>
        <span>${esc(ev.event || ev.type || "stage")} ${ev.stage != null ? "#" + ev.stage : ""}</span>
        <span class="muted">${ev.accuracy != null ? "acc " + fmt(ev.accuracy) : ""}</span>`;
      log.prepend(row);
    }
    while (log.children.length > 120) log.removeChild(log.lastChild);
    $("#feedSources").innerHTML = [...sources.entries()].map(([k, v]) =>
      `<div class="kv"><span>${esc(k)}</span><b>${fmtInt(v)} events</b></div>`
    ).join("") || `<div class="empty small"><p>No feed activity yet.</p></div>`;
  } catch {
    $("#feedStatus").textContent = "feed unavailable";
  }
}

async function loadAblationReports() {
  try {
    const res = await api("/state/ablation-reports");
    const reports = res.reports || [];
    $("#ablationList").innerHTML = reports.length
      ? reports.map((r) => `
        <div class="ep-card">
          <b>${esc(r.filename)}</b>
          <div class="muted">${esc(r.classifier || "")} · stages ${esc(String(r.stages ?? "—"))}</div>
          <div>${esc((r.datasets || []).join(", ") || "")}</div>
        </div>`).join("")
      : `<div class="empty small"><p>No ablation reports yet. Run: <code>python -m figpso_uil.experiments.ablation_harness</code></p></div>`;
  } catch {
    $("#ablationList").innerHTML = `<div class="empty small"><p>Reports unavailable</p></div>`;
  }
}

$("#feedClear").addEventListener("click", async () => {
  try {
    await api("/state/live-feed/clear", { method: "POST" });
    _feedCursor = 0;
    $("#feedLog").innerHTML = `<div class="empty small"><p>Feed cleared.</p></div>`;
    toast("Feed cleared", "ok");
  } catch (err) { toast(err.message, "err"); }
});
$("#ablationRefresh").addEventListener("click", loadAblationReports);

UIL.registerShow("ops", () => {
  pollLiveFeed();
  loadAblationReports();
});
UIL.onPoll(() => {
  if (UIL.activeView === "ops") pollLiveFeed();
});
