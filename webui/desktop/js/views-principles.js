/* Principles & Causal view */

"use strict";

function condStr(p) {
  return (p.conditions || []).map((c) =>
    `${c.feature || "?"}${c.op || ""}${c.threshold ?? ""}`).join(" ∧ ") || "—";
}

async function loadGovernance() {
  try {
    const g = await api("/state/governance");
    const el = $("#governancePanel");
    const seeds = g.seed_principles || g.seeds || [];
    el.innerHTML = `
      <p class="hint">${esc(g.summary || "Seed rules keep the journal numeric and principles hypothesis-only.")}</p>
      <div class="kv-rows">
        ${(seeds.length ? seeds : [{ principle_id: "—", note: "no seed listing" }]).slice(0, 8).map((s) =>
          `<div class="kv"><span>${esc(s.principle_id || s.id || "seed")}</span><b>${esc(s.note || condStr(s))}</b></div>`
        ).join("")}
      </div>`;
  } catch {
    $("#governancePanel").innerHTML = `<div class="empty small"><p>Governance manifest unavailable.</p></div>`;
  }
}

async function loadPrinciples() {
  try {
    const res = await api("/state/principles");
    const rows = res.principles || [];
    $("#principlesTable tbody").innerHTML = rows.map((p) => `
      <tr data-id="${esc(p.principle_id)}">
        <td><code>${esc(p.principle_id)}</code></td>
        <td>${esc(String(p.tier ?? "—"))}</td>
        <td><span class="badge st-${esc(p.status || "active")}">${esc(p.status || "active")}</span></td>
        <td>${fmt(p.confidence)}</td>
        <td>${fmtInt(p.evidence_count ?? p.evidence ?? 0)}</td>
        <td class="wrap">${esc(condStr(p))}</td>
      </tr>`).join("") || `<tr><td colspan="6" class="muted">No principles yet — need journal history</td></tr>`;
    $$("#principlesTable tbody tr[data-id]").forEach((tr) => {
      tr.addEventListener("click", () => {
        const p = rows.find((x) => x.principle_id === tr.dataset.id);
        if (!p) return;
        openEditModal({
          title: p.principle_id,
          readable: condStr(p),
          confidence: p.confidence ?? 0.5,
          tier: p.tier ?? 3,
          status: p.status || "active",
          statusOptions: ["active", "quarantined", "retired"],
          onSave: async (fields) => {
            await api("/state/principles", {
              method: "PATCH",
              body: JSON.stringify({ item_id: p.principle_id, fields }),
            });
            toast("Principle updated (human_edit)", "ok");
            loadPrinciples();
          },
          onReject: async () => {
            await api(`/state/cells/${encodeURIComponent(p.principle_id)}/reject?cell_type=principle`, { method: "POST" });
            toast("Principle rejected", "ok");
            loadPrinciples();
          },
        });
      });
    });
  } catch (err) { toast(`Principles: ${err.message}`, "err"); }
}

async function loadCausal() {
  try {
    const res = await api("/state/causal");
    const rows = res.hypotheses || res.causal || [];
    $("#causalTable tbody").innerHTML = rows.map((h) => `
      <tr data-id="${esc(h.hypothesis_id)}">
        <td><code>${esc(h.hypothesis_id)}</code></td>
        <td>${esc(h.variable)}</td>
        <td>${esc(h.direction)}</td>
        <td><span class="badge st-${esc((h.status || "").replace(/_/g, "-"))}">${esc(h.status)}</span></td>
        <td>${fmt(h.causal_strength)}</td>
        <td>${fmtInt((h.evidence_trials || h.interventions || []).length)}</td>
      </tr>`).join("") || `<tr><td colspan="6" class="muted">No causal hypotheses yet</td></tr>`;
    $$("#causalTable tbody tr[data-id]").forEach((tr) => {
      tr.addEventListener("click", () => {
        const h = rows.find((x) => x.hypothesis_id === tr.dataset.id);
        if (!h) return;
        openEditModal({
          title: h.hypothesis_id,
          readable: h.conclusion || `${h.variable} ${h.direction}`,
          confidence: h.causal_strength ?? 0,
          status: h.status,
          statusOptions: ["pending", "testing", "correlational_only", "confirmed_causal", "refuted"],
          onSave: async (fields) => {
            const patch = { ...fields };
            if (patch.confidence != null) {
              patch.causal_strength = patch.confidence;
              delete patch.confidence;
            }
            await api("/state/causal", {
              method: "PATCH",
              body: JSON.stringify({ item_id: h.hypothesis_id, fields: patch }),
            });
            toast("Hypothesis updated", "ok");
            loadCausal();
          },
        });
      });
    });
  } catch (err) { toast(`Causal: ${err.message}`, "err"); }
}

function loadPrinciplesView() {
  loadGovernance();
  loadPrinciples();
  loadCausal();
}

$("#principlesRefresh").addEventListener("click", loadPrinciples);
$("#causalRefresh").addEventListener("click", loadCausal);
UIL.registerShow("principles", loadPrinciplesView);
UIL.onPoll(() => { if (UIL.activeView === "principles") loadPrinciplesView(); });
