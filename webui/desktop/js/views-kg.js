/* Knowledge Graph — modular neural workspace, inline edit, synapse inspector */

"use strict";

const kg = new KnowledgeGraph($("#kgCanvas"));
let _kgRaw = { nodes: [], edges: [] };
let _saveTimer = null;

function normalizeNetwork(res) {
  const rawNodes = Array.isArray(res.nodes)
    ? res.nodes
    : Object.values(res.nodes || {});
  const nodes = rawNodes.map((n) => ({
    id: n.node_id || n.id,
    label: n.node_id || n.id,
    confidence: n.confidence ?? 0.5,
    tier: n.tier,
    meta: n,
  }));
  const edges = (res.edges || []).map((e) => ({
    source: e.source || e.from || e.source_id,
    target: e.target || e.to || e.target_id,
    type: e.edge_type || e.relation || e.type || "leads_to",
    weight: e.weight ?? e.confidence ?? 0.5,
  }));
  return { nodes, edges, summary: res.summary || {} };
}

function setModOpen(mod, open) {
  const el = $(`.kg-mod[data-mod="${mod}"]`);
  if (!el) return;
  el.classList.toggle("collapsed", !open);
  el.querySelector(".kg-mod-toggle")?.setAttribute("aria-expanded", open ? "true" : "false");
}

function renderKgVitals(summary, nodes, edges) {
  const s = summary || {};
  $("#kgVitals").innerHTML = [
    ["neurons", s.nodes ?? nodes.length],
    ["synapses", s.edges ?? edges.length],
    ["contradicts", s.edge_types?.contradicts ?? edges.filter((e) => e.type === "contradicts").length],
    ["strengthens", s.edge_types?.strengthens ?? edges.filter((e) => e.type === "strengthens").length],
  ].map(([k, v]) => `<div class="kv"><span>${esc(k)}</span><b class="count-up">${esc(String(v))}</b></div>`).join("");

  const types = { ...(s.edge_types || {}) };
  if (!Object.keys(types).length) {
    for (const e of edges) types[e.type] = (types[e.type] || 0) + 1;
  }
  renderHBars($("#kgTypeBars"), Object.entries(types).map(([name, value]) => ({ name, value })), {
    colorFn: (r) => EDGE_COLORS[r.name] || "var(--acc)",
  });

  const activated = s.most_activated || [];
  const top = activated.length
    ? activated.map((a) => nodes.find((n) => n.id === a.node_id) || { id: a.node_id, confidence: 0.5, activations: a.activations })
    : [...nodes].sort((a, b) => (b.confidence || 0) - (a.confidence || 0)).slice(0, 8);

  $("#kgTopNodes").innerHTML = top.map((n) => `
    <button class="top-node" data-id="${esc(n.id)}">
      <span class="pulse-dot"></span>
      <span>${esc(n.id)}</span>
      <b>${n.activations != null ? fmtInt(n.activations) + "×" : fmt(n.confidence)}</b>
    </button>`).join("") || `<div class="empty small"><p>No neurons yet — run UIL to grow the graph.</p></div>`;

  $$("#kgTopNodes .top-node").forEach((btn) => {
    btn.addEventListener("click", () => focusNode(btn.dataset.id));
  });
}

function clearFocus() {
  kg.clearFocus();
  $("#kgFocusLabel").textContent = "";
  $("#kgClearFocus").classList.add("hidden");
}

function focusNode(id) {
  const n = kg.nodeById.get(id);
  if (!n) return;
  kg.selectedId = id;
  kg.setFocus(id);
  $("#kgFocusLabel").textContent = `focus · ${id}`;
  $("#kgClearFocus").classList.remove("hidden");
  renderNodeEditor(n);
  setModOpen("node", true);
  toast(`Focused neighborhood · ${id}`, "");
}

async function saveNodeFields(nodeId, fields) {
  await api("/state/network/node", {
    method: "PATCH",
    body: JSON.stringify({ item_id: nodeId, fields }),
  });
  toast("Neuron updated · persisted to Editable Network", "ok");
  loadKg();
}

function renderNodeEditor(node) {
  const el = $("#kgNodeEditor");
  if (!node) {
    el.innerHTML = `<div class="empty small"><p>Select a neuron (hex node) to inspect and edit confidence / tier inline.</p></div>`;
    return;
  }
  const m = node.meta || {};
  const conds = (m.conditions || []).map((c) =>
    `${c.feature || "?"} ${c.op || ""} ${c.threshold ?? ""}`).join(" ∧ ") || "—";

  el.innerHTML = `
    <div class="insp animate-in">
      <div class="insp-head">
        <span class="neuron-badge">NEURON</span>
        <code class="insp-id">${esc(node.id)}</code>
      </div>
      <div class="insp-conds">${esc(conds)}</div>
      <label class="slider-field">
        <span>Confidence <b id="kgConfLive">${fmt(node.confidence)}</b></span>
        <input type="range" id="kgConfSlider" min="0" max="100" value="${Math.round((node.confidence || 0) * 100)}">
      </label>
      <label class="field slim">
        <span>Tier</span>
        <select id="kgTierSelect">
          ${[1, 2, 3, 4].map((t) => `<option value="${t}" ${(m.tier ?? node.tier) === t ? "selected" : ""}>T${t}</option>`).join("")}
        </select>
      </label>
      <div class="btn-row">
        <button class="btn primary sm" id="kgSaveNode">Save to graph</button>
        <button class="btn ghost sm" id="kgFocusNode">Focus synapses</button>
        <button class="btn ghost sm" id="kgAdvEdit">Advanced</button>
      </div>
    </div>`;

  const slider = $("#kgConfSlider");
  slider.addEventListener("input", () => {
    $("#kgConfLive").textContent = fmt(parseInt(slider.value, 10) / 100);
    node.confidence = parseInt(slider.value, 10) / 100;
    node.r = 7 + Math.min(12, node.confidence * 14);
  });

  $("#kgSaveNode").onclick = () => saveNodeFields(node.id, {
    confidence: parseInt(slider.value, 10) / 100,
    tier: parseInt($("#kgTierSelect").value, 10),
  });

  $("#kgFocusNode").onclick = () => {
    kg.setFocus(node.id);
    $("#kgFocusLabel").textContent = `focus · ${node.id}`;
    $("#kgClearFocus").classList.remove("hidden");
  };

  $("#kgAdvEdit").onclick = () => {
    openEditModal({
      title: `Neuron ${node.id}`,
      readable: conds,
      confidence: node.confidence,
      tier: m.tier ?? node.tier ?? 3,
      statusOptions: null,
      onSave: async (fields) => saveNodeFields(node.id, fields),
    });
  };
}

function renderEdgeInspector(edge) {
  const el = $("#kgEdgeEditor");
  if (!edge) {
    el.innerHTML = `<div class="empty small"><p>Click a synapse (edge) to inspect relation type and weight.</p></div>`;
    return;
  }
  const col = EDGE_COLORS[edge.type] || "var(--acc)";
  el.innerHTML = `
    <div class="insp animate-in">
      <div class="insp-head">
        <span class="synapse-badge" style="--c:${col}">SYNAPSE</span>
        <span class="badge">${esc(edge.type)}</span>
      </div>
      <div class="kv"><span>from</span><code>${esc(edge.source)}</code></div>
      <div class="kv"><span>to</span><code>${esc(edge.target)}</code></div>
      <div class="kv"><span>weight</span><b>${fmt(edge.weight)}</b></div>
      <div class="btn-row">
        <button class="btn ghost sm" id="kgJumpSource">Go to source</button>
        <button class="btn ghost sm" id="kgJumpTarget">Go to target</button>
      </div>
      <p class="hint">Edge weight edits route through HITL cells when the backend exposes them. Use Principles for causal overrides.</p>
    </div>`;
  $("#kgJumpSource").onclick = () => focusNode(edge.source);
  $("#kgJumpTarget").onclick = () => focusNode(edge.target);
}

function renderInspector(node) {
  renderNodeEditor(node);
  if (node) setModOpen("node", true);
}

async function loadKg() {
  try {
    const res = await api("/state/network");
    _kgRaw = normalizeNetwork(res);
    const MAX_N = 200;
    let nodes = _kgRaw.nodes;
    let edges = _kgRaw.edges;
    if (nodes.length > MAX_N) {
      nodes = [...nodes].sort((a, b) => (b.confidence || 0) - (a.confidence || 0)).slice(0, MAX_N);
      const keep = new Set(nodes.map((n) => n.id));
      edges = edges.filter((e) => keep.has(e.source) && keep.has(e.target));
    }
    kg.edgeFilter = $$("#kgEdgeChips .chip.on")[0]?.dataset.t || "";
    kg.search = $("#kgSearch").value.trim();
    kg.showLabels = $("#kgLabels").checked;
    kg.physics = $("#kgPhysics").checked && nodes.length <= 140;
    kg.neuralPulse = $("#kgPulse").checked;
    kg.setData({ nodes, edges });
    renderKgVitals(_kgRaw.summary, _kgRaw.nodes, _kgRaw.edges);
    if (kg.selectedId) {
      const n = kg.nodeById.get(kg.selectedId);
      if (n) renderNodeEditor(n);
    }
  } catch (err) { toast(`KG: ${err.message}`, "err"); }
}

kg.onSelect = (n) => {
  renderInspector(n);
  renderEdgeInspector(null);
};
kg.onEdgeSelect = (e) => {
  renderEdgeInspector(e);
  setModOpen("edge", true);
};
kg.onHover = (n, e, edge) => {
  const card = $("#kgHover");
  if (!n && !edge) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");
  if (n) {
    card.innerHTML = `<span class="neuron-badge sm">NEURON</span><b>${esc(n.id)}</b><span>conf ${fmt(n.confidence)} · T${n.tier ?? 3}</span>`;
  } else {
    card.innerHTML = `<span class="synapse-badge sm">SYNAPSE</span><b>${esc(edge.type)}</b><span>${esc(edge.source)} → ${esc(edge.target)}</span>`;
  }
  const rect = $("#kgCanvas").parentElement.getBoundingClientRect();
  card.style.left = Math.min(rect.width - 200, Math.max(12, e.clientX - rect.left + 14)) + "px";
  card.style.top = Math.min(rect.height - 70, Math.max(12, e.clientY - rect.top + 14)) + "px";
};

$$("#kgModeSeg button").forEach((b) => {
  b.addEventListener("click", () => {
    $$("#kgModeSeg button").forEach((x) => x.classList.remove("on"));
    b.classList.add("on");
    kg.setMode(b.dataset.v);
  });
});

$$("#kgEdgeChips .chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    $$("#kgEdgeChips .chip").forEach((c) => c.classList.remove("on"));
    chip.classList.add("on");
    kg.edgeFilter = chip.dataset.t || "";
  });
});

$$(".kg-mod-toggle").forEach((btn) => {
  btn.addEventListener("click", () => {
    const mod = btn.closest(".kg-mod");
    mod.classList.toggle("collapsed");
    const open = !mod.classList.contains("collapsed");
    btn.setAttribute("aria-expanded", open ? "true" : "false");
    const icon = btn.querySelector("i");
    if (icon) icon.textContent = open ? "−" : "+";
  });
});

$("#kgSearch").addEventListener("input", () => { kg.search = $("#kgSearch").value.trim(); });
$("#kgLabels").addEventListener("change", () => { kg.showLabels = $("#kgLabels").checked; });
$("#kgPhysics").addEventListener("change", () => { kg.physics = $("#kgPhysics").checked; });
$("#kgPulse").addEventListener("change", () => { kg.neuralPulse = $("#kgPulse").checked; });
$("#kgRefresh").addEventListener("click", loadKg);
$("#kgFit").addEventListener("click", () => kg.fit());
$("#kgClearFocus").addEventListener("click", clearFocus);

$("#kgDetail")?.addEventListener("change", () => {
  kg.showLabels = $("#kgLabels").checked && $("#kgDetail").checked;
  const legend = $("#kgLegend");
  if (legend) legend.classList.toggle("hidden", !$("#kgDetail").checked);
});

window.addEventListener("keydown", (e) => {
  if (UIL.activeView !== "kg") return;
  const tag = (e.target && e.target.tagName) || "";
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
  if (e.key === "f" || e.key === "F") { e.preventDefault(); kg.fit(); }
  if (e.key === "Escape") { clearFocus(); kg.selectedId = null; renderNodeEditor(null); }
});

UIL.registerShow("kg", () => {
  loadKg();
  setModOpen("vitals", true);
  setModOpen("node", true);
});
UIL.onStage(() => { if (UIL.activeView === "kg") loadKg(); });
UIL.onPoll(() => { if (UIL.activeView === "kg") loadKg(); });
