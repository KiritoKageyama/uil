/* Overview — what is UIL, howto, live memory stats, module map + mindmap */

"use strict";

const MODULES = [
  { view: "mission", title: "Mission Control", blurb: "Launch runs, watch charts & decision pipeline live." },
  { view: "kg", title: "Knowledge Graph", blurb: "Network + mindmap of principles and causal edges." },
  { view: "formulas", title: "Formulas", blurb: "Loss & symbolic classifier libraries — editable cells." },
  { view: "principles", title: "Principles & Causal", blurb: "DT-mined rules + majority-of-3 causal status." },
  { view: "memory", title: "Memory Proof", blurb: "On-disk stores, champions, archive, consciousness." },
  { view: "hitl", title: "Human in the Loop", blurb: "Review, edit, reject beliefs; undo via replay." },
  { view: "goal", title: "Goal Director", blurb: "Top-down attention bias the next run must honor." },
  { view: "timeline", title: "Timeline & Replay", blurb: "Scrub VersionStore seq + journal retrace." },
  { view: "ops", title: "Terminal Feed", blurb: "External / terminal runs stream into the console." },
];

let _ovKg = null;

function demoMindmap() {
  const hub = "UIL";
  const spokes = ["PSO", "Journal", "Miner", "Principles", "KG", "Causal", "Formulas", "HITL"];
  const nodes = [{ id: hub, label: hub, confidence: 1 }].concat(
    spokes.map((s, i) => ({ id: s, label: s, confidence: 0.55 + (i % 4) * 0.1 })),
  );
  const edges = spokes.map((s) => ({
    source: hub, target: s, type: "leads_to", weight: 0.7,
  }));
  edges.push(
    { source: "Journal", target: "Miner", type: "depends_on", weight: 0.8 },
    { source: "Miner", target: "Principles", type: "leads_to", weight: 0.75 },
    { source: "Principles", target: "KG", type: "strengthens", weight: 0.65 },
    { source: "Causal", target: "Principles", type: "contradicts", weight: 0.4 },
  );
  return { nodes, edges };
}

function ensureOverviewMindmap() {
  const canvas = $("#ovMindmap");
  if (!canvas || _ovKg) return _ovKg;
  _ovKg = new KnowledgeGraph(canvas);
  _ovKg.showLabels = true;
  _ovKg.physics = false;
  _ovKg.setMode("brain");
  _ovKg.setData(demoMindmap());
  return _ovKg;
}

function feedOverviewMindmap(net) {
  const kg = ensureOverviewMindmap();
  if (!kg) return;
  const rawNodes = Array.isArray(net.nodes) ? net.nodes : Object.values(net.nodes || {});
  if (rawNodes.length < 2) {
    kg.setMode("mindmap");
    kg.setData(demoMindmap());
    return;
  }
  // Cap for hero performance
  const ranked = [...rawNodes]
    .sort((a, b) => (b.confidence || 0) - (a.confidence || 0))
    .slice(0, 36);
  const keep = new Set(ranked.map((n) => n.node_id || n.id));
  const nodes = ranked.map((n) => ({
    id: n.node_id || n.id,
    label: (n.node_id || n.id || "").slice(0, 18),
    confidence: n.confidence ?? 0.5,
  }));
  const edges = (net.edges || [])
    .filter((e) => keep.has(e.source_id || e.source) && keep.has(e.target_id || e.target))
    .slice(0, 80)
    .map((e) => ({
      source: e.source_id || e.source,
      target: e.target_id || e.target,
      type: e.edge_type || e.type || "leads_to",
      weight: e.weight ?? 0.5,
    }));
  kg.setMode("mindmap");
  kg.setData({ nodes, edges });
}

function renderModuleMap() {
  const el = $("#moduleMap");
  if (!el) return;
  el.innerHTML = MODULES.map((m, i) => `
    <button class="mod-tile" data-goto="${m.view}" style="--i:${i}">
      <b>${esc(m.title)}</b>
      <span>${esc(m.blurb)}</span>
    </button>`).join("");
}

async function refreshOverviewStats() {
  try {
    const [persist, net, replay] = await Promise.all([
      api("/state/persistence"),
      api("/state/network").catch(() => ({ nodes: [], edges: [], summary: {} })),
      api("/state/replay").catch(() => ({ seq: 0 })),
    ]);
    const files = (persist.files || persist.stores || []);
    const list = Array.isArray(files) ? files : Object.values(files);
    const existing = list.filter((f) => f.exists !== false);
    const bytes = existing.reduce((s, f) => s + (f.size || f.bytes || 0), 0);
    $("#ovStatFiles").textContent = fmtInt(existing.length || list.length);
    $("#ovStatBytes").textContent = fmtBytes(bytes);
    const nodeCount = Array.isArray(net.nodes)
      ? net.nodes.length
      : Object.keys(net.nodes || {}).length;
    const nNodes = net.summary?.nodes ?? nodeCount;
    const nEdges = net.summary?.edges ?? (net.edges || []).length;
    $("#ovStatGraph").textContent = `${fmtInt(nNodes)} / ${fmtInt(nEdges)}`;
    $("#ovStatSeq").textContent = fmtInt(replay.seq ?? 0);
    $("#memFiles").textContent = `${existing.length || list.length} memory files`;
    feedOverviewMindmap(net);
  } catch (err) {
    ensureOverviewMindmap();
  }
}

UIL.registerShow("overview", () => {
  renderModuleMap();
  ensureOverviewMindmap();
  refreshOverviewStats();
});
UIL.onPoll(() => {
  if (UIL.activeView === "overview") refreshOverviewStats();
});
