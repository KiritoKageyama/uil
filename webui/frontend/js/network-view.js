// Editable Network graph view via vis-network. Click a node to edit
// its confidence/action_override through the existing edit_node() API.
const NetworkView = (() => {
  let network = null;
  let nodesDS = null;
  let edgesDS = null;
  let lastRefreshStage = -1;

  function init() {
    const container = document.getElementById("network-graph");
    nodesDS = new vis.DataSet([]);
    edgesDS = new vis.DataSet([]);
    network = new vis.Network(container, { nodes: nodesDS, edges: edgesDS }, {
      nodes: { shape: "dot", size: 14, font: { color: "#e6e6e6" } },
      edges: { arrows: "to", color: { color: "#30363d" } },
      physics: { stabilization: true },
    });
    network.on("click", (params) => {
      if (params.nodes.length) onNodeClick(params.nodes[0]);
    });
  }

  function colorForEdgeType(t) {
    if (t === "contradicts") return "#da3633";
    if (t === "strengthens") return "#2ea043";
    return "#58a6ff"; // leads_to / depends_on
  }

  async function refresh(runId) {
    const q = runId ? `?run_id=${runId}` : "";
    const data = await Api.get(`/state/network${q}`);
    document.getElementById("network-summary").textContent =
      `${data.nodes ? Object.keys(data.nodes).length : 0} nodes, ${data.edges ? data.edges.length : 0} edges`;

    const nodes = Object.values(data.nodes || {}).map((n) => ({
      id: n.node_id,
      label: `${n.node_id}\n(tier ${n.tier}, conf ${n.confidence?.toFixed?.(2) ?? n.confidence})`,
      title: JSON.stringify(n.conditions),
    }));
    const edges = (data.edges || []).map((e, i) => ({
      id: i,
      from: e.source_id,
      to: e.target_id,
      color: { color: colorForEdgeType(e.edge_type) },
      title: `${e.edge_type} (w=${e.weight?.toFixed?.(2) ?? e.weight})`,
    }));

    nodesDS.clear();
    edgesDS.clear();
    nodesDS.add(nodes);
    edgesDS.add(edges);

    window.__lastNetworkData = data;
  }

  function onSnapshot(s) {
    if (!s?.memory) return;
    const el = document.getElementById("network-summary");
    if (el) {
      el.textContent = `${s.memory.network_nodes} nodes, ${s.memory.network_edges} edges (live)`;
    }
    if (s.stage >= 0 && s.stage % 3 === 0 && s.stage !== lastRefreshStage) {
      lastRefreshStage = s.stage;
      refresh(App.activeRunId());
    }
  }

  function onNodeClick(nodeId) {
    const data = window.__lastNetworkData;
    const node = data && data.nodes ? data.nodes[nodeId] : null;
    const container = document.getElementById("network-edit");
    if (!node) {
      container.innerHTML = "";
      return;
    }
    container.innerHTML = `
      <h3>Edit node ${nodeId}</h3>
      <label>Confidence <input id="edit-node-confidence" type="number" step="0.01" value="${node.confidence}"></label>
      <button id="edit-node-save">Save</button>
    `;
    document.getElementById("edit-node-save").onclick = async () => {
      const confidence = parseFloat(document.getElementById("edit-node-confidence").value);
      await Api.patch("/state/network/node", { item_id: nodeId, fields: { confidence } });
      refresh(App.activeRunId());
    };
  }

  return { init, refresh, onSnapshot };
})();
