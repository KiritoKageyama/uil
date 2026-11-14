/* Goal Director — set / clear top-down attention goals */

"use strict";

function renderActiveGoal(goal) {
  const box = $("#activeGoalBox");
  const hero = $("#heroGoal");
  if (!goal) {
    box.innerHTML = `<div class="empty small"><p>No active goal — the system runs pure bottom-up attention.</p></div>`;
    if (hero) hero.innerHTML = "";
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
      <div class="g-line"><span>Version</span>v${goal.version ?? 1}</div>
    </div>`;
  if (hero) hero.innerHTML = `<span class="goal-pill">${esc(goal.directive)}</span>`;
}

async function loadGoal() {
  try {
    const res = await api("/state/goal");
    renderActiveGoal(res.goal);
  } catch (err) { toast(`Goal: ${err.message}`, "err"); }
}

const csv = (v) => String(v || "").split(",").map((s) => s.trim()).filter(Boolean);
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
    renderActiveGoal(res.goal || body);
    toast("Goal set — next run will honor this directive", "ok");
  } catch (err) { toast(err.message, "err"); }
});

$("#goalClearBtn").addEventListener("click", async () => {
  try {
    await api("/state/goal/clear", { method: "POST" });
    renderActiveGoal(null);
    toast("Goal cleared", "ok");
  } catch (err) { toast(err.message, "err"); }
});

UIL.registerShow("goal", loadGoal);
