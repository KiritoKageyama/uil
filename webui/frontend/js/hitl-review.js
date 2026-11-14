/** HITL (Human-In-The-Loop) — review, edit, reject UIL knowledge between runs. */
const HitlReview = (() => {
  let lastSeq = 0;

  async function loadCells() {
    const el = document.getElementById("hitl-cells");
    if (!el) return;
    const filter = document.getElementById("hitl-type-filter")?.value || "";
    const qs = filter ? `?cell_type=${encodeURIComponent(filter)}&limit=80` : "?limit=80";
    try {
      const { cells, last_seq } = await Api.get(`/state/cells${qs}`);
      lastSeq = last_seq;
      document.getElementById("hitl-last-seq").textContent = `seq ${last_seq}`;
      if (!cells.length) {
        el.innerHTML = "<p class='hint'>No versioned cells yet — run UIL with shared memory.</p>";
        return;
      }
      el.innerHTML = `
        <p class="hint">HITL = you review what UIL learned. Edit confidence/status, reject bad cells, preview undo via replay.</p>
        <table class="hitl-table">
        <thead><tr><th>id</th><th>type</th><th>conf</th><th>readable</th><th>actions</th></tr></thead>
        <tbody>${cells.map((c) => rowHtml(c)).join("")}</tbody>
      </table>`;
      el.querySelectorAll("[data-reject]").forEach((btn) => {
        btn.onclick = () => rejectCell(btn.dataset.reject, btn.dataset.ctype);
      });
      el.querySelectorAll("[data-edit]").forEach((btn) => {
        btn.onclick = () => editCell(btn.dataset.edit, btn.dataset.ctype);
      });
      el.querySelectorAll("[data-replay]").forEach((btn) => {
        btn.onclick = () => replayAt(btn.dataset.replay);
      });
    } catch (e) {
      el.innerHTML = `<p class="hint">Failed to load cells: ${e.message}</p>`;
    }
  }

  function rowHtml(c) {
    const conf = c.confidence != null ? Number(c.confidence).toFixed(2) : "—";
    const readable = (c.readable || "").slice(0, 160);
    return `<tr>
      <td><code>${c.cell_id}</code></td>
      <td>${c.cell_type}</td>
      <td>${conf}</td>
      <td class="readable-col" title="${(c.readable || "").replace(/"/g, "&quot;")}">${readable}</td>
      <td>
        <button data-edit="${c.cell_id}" data-ctype="${c.cell_type}">Edit</button>
        <button data-reject="${c.cell_id}" data-ctype="${c.cell_type}">Reject</button>
        <button data-replay="${Math.max(1, lastSeq - 1)}">Undo preview</button>
      </td>
    </tr>`;
  }

  async function rejectCell(cellId, cellType) {
    if (!confirm(`Reject ${cellId}? This records a human_reject event (undo = replay).`)) return;
    await Api.post(`/state/cells/${encodeURIComponent(cellId)}/reject?cell_type=${encodeURIComponent(cellType)}`);
    await loadCells();
  }

  async function editCell(cellId, cellType) {
    let fields = {};
    if (cellType === "formula") {
      const conf = prompt("Formula confidence (0–1):", "0.5");
      if (conf == null) return;
      fields.confidence = parseFloat(conf);
      const status = prompt("Status (active/incubating/retired):", "active");
      if (status) fields.status = status;
    } else if (cellType === "causal_hypothesis") {
      const st = prompt("Status (pending/confirmed_causal/refuted):", "pending");
      if (st == null) return;
      fields.status = st;
      const cs = prompt("Causal strength (0–1):", "0.5");
      if (cs) fields.causal_strength = parseFloat(cs);
    } else {
      const val = prompt("New confidence (0–1):", "0.5");
      if (val == null) return;
      fields.confidence = parseFloat(val);
    }
    await Api.patch(`/state/cells/${encodeURIComponent(cellId)}?cell_type=${encodeURIComponent(cellType)}`, {
      item_id: cellId,
      fields,
    });
    await loadCells();
  }

  async function replayAt(seq) {
    const pre = document.getElementById("hitl-replay-preview");
    const data = await Api.get(`/state/replay?seq=${seq}`);
    pre.textContent = JSON.stringify(data.state, null, 2);
  }

  async function loadLatestReport() {
    const el = document.getElementById("hitl-lab-report");
    if (!el) return;
    try {
      const res = await Api.get("/state/lab-reports/latest");
      el.innerHTML = `<p class="hint">${res.filename}</p><pre class="lab-report-pre">${(res.markdown || "").replace(/</g, "&lt;")}</pre>`;
    } catch {
      el.innerHTML = "<p class='hint'>No lab report yet — complete a UIL run (reports auto-generated at fit() end).</p>";
    }
  }

  function init() {
    document.getElementById("hitl-type-filter")?.addEventListener("change", loadCells);
    document.getElementById("btn-hitl-refresh")?.addEventListener("click", () => {
      loadCells();
      loadLatestReport();
    });
    document.getElementById("btn-hitl-refresh")?.click();
  }

  return { init, loadCells, loadLatestReport };
})();
