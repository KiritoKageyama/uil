/* ════════════════════════════════════════════════════════════════════
   embed-kg.js — read-only Knowledge Graph sneak peek for the marketing
   site. Reuses the SHARED neural-graph engine against /api/state/network.
   Inspect only: click a neuron to read its conditions / tier / confidence.
   No edits here — editing lives in the console (/app/#kg).
   Degrades to a friendly message when the API is empty or unreachable.
   ════════════════════════════════════════════════════════════════════ */

"use strict";

(function () {
  const canvas = document.getElementById("mktKgCanvas");
  if (!canvas || !window.KnowledgeGraph) return;

  const hint = document.getElementById("mktKgHint");
  const hover = document.getElementById("mktKgHover");
  const vitals = document.getElementById("mktKgVitals");
  const inspect = document.getElementById("mktKgInspect");

  const EDGE_COLORS = window.EDGE_COLORS || {};
  const graph = new KnowledgeGraph(canvas, { readOnly: true, physics: true });

  function normalize(res) {
    const rawNodes = Array.isArray(res.nodes) ? res.nodes : Object.values(res.nodes || {});
    const nodes = rawNodes.map((n) => ({
      id: n.node_id || n.id,
      label: n.node_id || n.id,
      confidence: n.confidence ?? 0.5,
      tier: n.tier,
      meta: n,
    }));
    const edges = (res.edges || []).map((e) => ({
      source: e.source || e.source_id || e.from,
      target: e.target || e.target_id || e.to,
      type: e.edge_type || e.relation || e.type || "leads_to",
      weight: e.weight ?? e.confidence ?? 0.5,
    }));
    return { nodes, edges, summary: res.summary || {} };
  }

  function renderVitals(summary, nodes, edges) {
    const s = summary || {};
    const rows = [
      ["neurons", s.nodes ?? nodes.length],
      ["synapses", s.edges ?? edges.length],
      ["contradicts", s.edge_types?.contradicts ?? edges.filter((e) => e.type === "contradicts").length],
      ["strengthens", s.edge_types?.strengthens ?? edges.filter((e) => e.type === "strengthens").length],
    ];
    vitals.innerHTML = rows.map(([k, v]) =>
      `<div class="kv-tile"><b>${Number(v).toLocaleString("en-US")}</b><span>${k}</span></div>`).join("");
  }

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function renderInspect(node) {
    if (!node) {
      inspect.innerHTML = `<div class="empty small"><p>Click a neuron to inspect its conditions, tier, and confidence.</p></div>`;
      return;
    }
    const m = node.meta || {};
    const conds = (m.conditions || []).map((c) => `${c.feature ?? "?"} ${c.op ?? ""} ${c.threshold ?? ""}`).join(" ∧ ") || "—";
    inspect.innerHTML = `
      <code class="insp-id">${esc(node.id)}</code>
      <div class="insp-conds">${esc(conds)}</div>
      <div class="insp-row"><span>confidence</span><b>${(node.confidence ?? 0).toFixed(3)}</b></div>
      <div class="insp-row"><span>tier</span><b>T${m.tier ?? node.tier ?? 3}</b></div>
      <div class="insp-row"><span>activations</span><b>${m.activation_count ?? 0}</b></div>
      <div class="insp-row"><span>evidence</span><b>${m.evidence_count ?? 0}</b></div>`;
  }

  graph.onSelect = (n) => renderInspect(n);
  graph.onEdgeSelect = () => {};
  graph.onHover = (n, e, edge) => {
    if (!n && !edge) { hover.classList.add("hidden"); return; }
    hover.classList.remove("hidden");
    if (n) {
      hover.innerHTML = `<b>${esc(n.id)}</b>conf ${(n.confidence ?? 0).toFixed(2)} · T${n.tier ?? 3}`;
    } else {
      const col = EDGE_COLORS[edge.type] || "var(--acc)";
      hover.innerHTML = `<b style="color:${col}">${esc(edge.type)}</b>${esc(edge.source)} → ${esc(edge.target)}`;
    }
    const rect = canvas.parentElement.getBoundingClientRect();
    hover.style.left = Math.min(rect.width - 200, Math.max(10, e.clientX - rect.left + 14)) + "px";
    hover.style.top = Math.min(rect.height - 70, Math.max(10, e.clientY - rect.top + 14)) + "px";
  };

  async function load() {
    try {
      const res = await fetch("/api/state/network");
      if (!res.ok) throw new Error(res.statusText);
      const data = normalize(await res.json());
      let { nodes, edges } = data;
      if (!nodes.length) {
        hint.textContent = "graph is empty — run UIL in the console to grow it";
        renderVitals(data.summary, [], []);
        return;
      }
      // cap for a snappy embed; keep the highest-confidence neighbourhood
      const MAX = 120;
      if (nodes.length > MAX) {
        nodes = [...nodes].sort((a, b) => (b.confidence || 0) - (a.confidence || 0)).slice(0, MAX);
        const keep = new Set(nodes.map((n) => n.id));
        edges = edges.filter((e) => keep.has(e.source) && keep.has(e.target));
      }
      graph.physics = nodes.length <= 90;
      graph.setData({ nodes, edges });
      renderVitals(data.summary, data.nodes, data.edges);
      hint.textContent = `${nodes.length} neurons · ${edges.length} synapses · drag · zoom · click`;
    } catch (err) {
      hint.textContent = "backend offline — start uvicorn to see the live graph";
      vitals.innerHTML = `<div class="kv-tile"><b>—</b><span>offline</span></div>`;
    }
  }

  // load when the graph section scrolls into view (defer heavy work)
  const section = document.getElementById("graph");
  let loaded = false;
  const loadOnce = () => { if (!loaded) { loaded = true; load(); } };
  if (section && "IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) loadOnce(); });
    }, { threshold: 0.15 });
    io.observe(section);
    // safety net so the embed never stays stuck on "loading…"
    setTimeout(loadOnce, 2200);
  } else { loadOnce(); }
})();
