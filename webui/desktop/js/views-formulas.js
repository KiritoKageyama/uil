/* Formulas view — loss + symbolic libraries, detail, edit */

"use strict";

let _lossFormulas = [];
let _symFormulas = [];

function formulaDepth(tree) {
  if (!tree || typeof tree !== "object") return 0;
  if (!tree.children) return 1;
  return 1 + Math.max(0, ...(tree.children || []).map(formulaDepth));
}

function formulaExpr(f) {
  try {
    if (f.readable) return f.readable;
    if (f.tree) return JSON.stringify(f.tree).slice(0, 120);
    return f.name || f.formula_id || "—";
  } catch { return "—"; }
}

function filterSort(list) {
  const status = $("#formulaFilter").value;
  const sort = $("#formulaSort").value;
  let rows = list.slice();
  if (status !== "all") rows = rows.filter((f) => (f.status || "active") === status);
  rows.sort((a, b) => {
    if (sort === "name") return String(a.name || "").localeCompare(String(b.name || ""));
    return (b[sort] ?? 0) - (a[sort] ?? 0);
  });
  return rows;
}

function renderFormulaTable(tbody, list) {
  const rows = filterSort(list);
  tbody.innerHTML = rows.map((f) => `
    <tr data-id="${esc(f.formula_id)}">
      <td>${esc(f.name || f.formula_id)}</td>
      <td><span class="badge st-${esc(f.status || "active")}">${esc(f.status || "active")}</span></td>
      <td>${fmt(f.confidence)}</td>
      <td>${fmtInt(f.evidence ?? 0)}</td>
      <td>${fmtInt(f.times_helped ?? 0)}/${fmtInt(f.times_hurt ?? 0)}</td>
      <td>${fmt(f.best_accuracy)}</td>
      <td>${formulaDepth(f.tree)}</td>
    </tr>`).join("") || `<tr><td colspan="7" class="muted">No formulas yet</td></tr>`;
  tbody.querySelectorAll("tr[data-id]").forEach((tr) => {
    tr.addEventListener("click", () => {
      const f = [..._lossFormulas, ..._symFormulas].find((x) => x.formula_id === tr.dataset.id);
      if (f) showFormulaDetail(f);
    });
  });
}

function showFormulaDetail(f) {
  const el = $("#formulaDetail");
  el.innerHTML = `
    <div class="insp">
      <div class="insp-id">${esc(f.name || f.formula_id)}</div>
      <pre class="pre-panel">${esc(formulaExpr(f))}</pre>
      <div class="kv"><span>status</span><b>${esc(f.status || "active")}</b></div>
      <div class="kv"><span>confidence</span><b>${fmt(f.confidence)}</b></div>
      <div class="kv"><span>best accuracy</span><b>${fmt(f.best_accuracy)}</b></div>
      <div class="kv"><span>origin</span><b>${esc(f.origin || "—")}</b></div>
      <div class="btn-row" style="margin-top:12px">
        <button class="btn primary sm" id="formulaEditBtn">Edit</button>
        <button class="btn danger ghost sm" id="formulaRejectBtn">Reject</button>
      </div>
    </div>`;
  $("#formulaEditBtn").onclick = () => openEditModal({
    title: f.formula_id,
    readable: formulaExpr(f),
    confidence: f.confidence ?? 0.5,
    status: f.status || "active",
    statusOptions: ["active", "incubating", "seed", "proven", "retired"],
    onSave: async (fields) => {
      await api("/state/formulas", {
        method: "PATCH",
        body: JSON.stringify({ item_id: f.formula_id, fields }),
      });
      toast("Formula edit saved", "ok");
      loadFormulas();
    },
    onReject: async () => {
      await api(`/state/cells/${encodeURIComponent(f.formula_id)}/reject?cell_type=formula`, { method: "POST" });
      toast("Formula rejected", "ok");
      loadFormulas();
    },
  });
  $("#formulaRejectBtn").onclick = () => $("#formulaEditBtn").onclick();
}

async function loadFormulas() {
  try {
    const res = await api("/state/formulas");
    _lossFormulas = res.formulas || res.loss_formulas || [];
    _symFormulas = res.symbolic_formulas || res.symbolic || [];
    renderFormulaTable($("#lossTable tbody"), _lossFormulas);
    renderFormulaTable($("#symbolicTable tbody"), _symFormulas);
    const all = [..._lossFormulas, ..._symFormulas];
    $("#formulaSummary").textContent = `${all.length} formulas · ${_lossFormulas.length} loss · ${_symFormulas.length} symbolic`;
    const top = all.sort((a, b) => (b.confidence || 0) - (a.confidence || 0)).slice(0, 10);
    renderHBars($("#formulaTop"), top.map((f) => ({
      name: f.name || f.formula_id,
      value: f.confidence || 0,
    })), { fmtV: (v) => fmt(v, 2) });
  } catch (err) { toast(`Formulas: ${err.message}`, "err"); }
}

UIL.onStage((snap) => {
  const banner = $("#formulaBanner");
  if (!banner) return;
  if (snap.symbolic_formula_id || snap.formula_id || snap.custom_objective) {
    banner.classList.remove("idle");
    banner.innerHTML = `<b>Active formula</b><span> — ${esc(String(snap.symbolic_formula_id || snap.formula_id || "custom objective"))} · stage ${snap.stage}</span>`;
  }
});

$("#formulasRefresh").addEventListener("click", loadFormulas);
$("#formulaFilter").addEventListener("change", loadFormulas);
$("#formulaSort").addEventListener("change", loadFormulas);
UIL.registerShow("formulas", loadFormulas);
