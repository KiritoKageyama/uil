/* HITL — sneak peek guide + cell review + lab report + replay */

"use strict";

let _cellFilter = "";
let _cellCache = [];
let _hitlStep = 0;

function animateHitlSteps() {
  const steps = $$("#hitlSteps .hstep");
  steps.forEach((s, i) => s.classList.toggle("lit", i === _hitlStep));
  _hitlStep = (_hitlStep + 1) % Math.max(1, steps.length);
}

setInterval(() => {
  if (UIL.activeView === "hitl") animateHitlSteps();
}, 2800);

async function loadHitlCells() {
  try {
    const q = _cellFilter ? `?cell_type=${encodeURIComponent(_cellFilter)}&limit=80` : "?limit=80";
    const res = await api(`/state/cells${q}`);
    _cellCache = res.cells || [];
    $("#hitlLastSeq").textContent = `seq ${res.last_seq ?? "—"}`;
    $("#hitlSeqNow").textContent = fmtInt(res.last_seq ?? 0);
    $("#hitlCellsNow").textContent = fmtInt(_cellCache.length);

    // count human ops via replay state keys is approximate
    try {
      const rp = await api("/state/replay");
      let human = 0;
      // trajectory not available for all — estimate from lab report later
      $("#hitlHumanNow").textContent = fmtInt(human || "—");
    } catch { $("#hitlHumanNow").textContent = "—"; }

    const grid = $("#cellGrid");
    if (!_cellCache.length) {
      grid.innerHTML = `<div class="empty"><div class="empty-orb"></div><p>No versioned cells yet — run UIL with shared memory.</p></div>`;
      return;
    }
    grid.innerHTML = _cellCache.map((c, i) => `
      <div class="cell-card" style="--i:${i}" data-idx="${i}">
        <div class="cell-head">
          <span class="cell-type ct-${esc(c.cell_type)}">${esc(c.cell_type.replace(/_/g, " "))}</span>
          <span class="cell-id">${esc(c.cell_id)}</span>
        </div>
        <div class="cell-readable">${esc(c.readable || "")}</div>
        <div class="conf-bar"><div class="conf-fill" data-w="${Math.round((c.confidence || 0) * 100)}"></div></div>
      </div>`).join("");
    requestAnimationFrame(() => {
      $$(".conf-fill").forEach((f) => { f.style.width = f.dataset.w + "%"; });
    });
    $$(".cell-card").forEach((card) => {
      card.addEventListener("click", () => openCellModal(_cellCache[+card.dataset.idx]));
    });
  } catch (err) { toast(`HITL cells: ${err.message}`, "err"); }
}

function openCellModal(cell) {
  openEditModal({
    title: `${cell.cell_type} · ${cell.cell_id}`,
    readable: cell.readable || JSON.stringify(cell.post_state, null, 2),
    confidence: cell.confidence ?? 0.5,
    status: cell.post_state?.status,
    statusOptions: cell.cell_type === "formula"
      ? ["active", "incubating", "retired"]
      : cell.cell_type === "causal_hypothesis"
        ? ["pending", "testing", "correlational_only", "confirmed_causal", "refuted"]
        : null,
    onSave: async (fields) => {
      await api(`/state/cells/${encodeURIComponent(cell.cell_id)}?cell_type=${encodeURIComponent(cell.cell_type)}`, {
        method: "PATCH",
        body: JSON.stringify({ item_id: cell.cell_id, fields }),
      });
      toast("human_edit recorded + materialized", "ok");
      loadHitlCells();
    },
    onReject: async () => {
      await api(`/state/cells/${encodeURIComponent(cell.cell_id)}/reject?cell_type=${encodeURIComponent(cell.cell_type)}`, {
        method: "POST",
      });
      toast("human_reject recorded", "ok");
      loadHitlCells();
    },
    onUndo: async () => {
      const seq = Math.max(1, (await api("/state/replay")).seq - 1);
      const data = await api(`/state/replay?seq=${seq}`);
      $("#hitlReplayPreview").textContent = JSON.stringify(data.state, null, 2);
      toast(`Replay preview at seq ${seq}`, "ok");
    },
  });
}

async function loadLabReport() {
  try {
    const res = await api("/state/lab-reports/latest");
    $("#labReport").innerHTML = `<p class="hint">${esc(res.filename)}</p>${mdToHtml(res.markdown || "")}`;
  } catch {
    $("#labReport").innerHTML = `<div class="empty small"><p>No lab report yet — complete a UIL run.</p></div>`;
  }
}

$$("#typeChips .chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    $$("#typeChips .chip").forEach((c) => c.classList.remove("on"));
    chip.classList.add("on");
    _cellFilter = chip.dataset.t || "";
    loadHitlCells();
  });
});
$("#cellsRefresh").addEventListener("click", () => { loadHitlCells(); loadLabReport(); });
$("#labReportRefresh")?.addEventListener("click", loadLabReport);
$("#hitlPeek")?.addEventListener("click", async () => {
  await loadHitlCells();
  animateHitlSteps();
  toast("Live VersionStore state loaded into the review grid", "ok");
});

UIL.registerShow("hitl", () => {
  _hitlStep = 0;
  animateHitlSteps();
  loadHitlCells();
  loadLabReport();
});
