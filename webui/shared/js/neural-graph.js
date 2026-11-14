/* ════════════════════════════════════════════════════════════════════
   neural-graph.js — SHARED neural knowledge-graph engine
   Force network · radial mindmap · synapse brain · orbit layouts.
   Interactive: pan / zoom / drag neurons, select node & edge, focus a
   neighborhood. Self-contained (inline color helper) so it can be loaded
   by the marketing site AND the research console. Editing (PATCH) is done
   by the host page via the onSelect/onEdgeSelect callbacks — the engine
   itself is render-only, so it is safe to embed read-only.
   ════════════════════════════════════════════════════════════════════ */

"use strict";

function _kgCssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback || "";
}

const EDGE_COLORS = {
  leads_to: "#4aa8ff",
  strengthens: "#2fe0a0",
  contradicts: "#ff5470",
  depends_on: "#ffb43c",
};

const TIER_COLORS = ["#b48bff", "#4aa8ff", "#94a3b8", "#64748b"];

class KnowledgeGraph {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.readOnly = !!opts.readOnly;
    this.nodes = [];
    this.edges = [];
    this.nodeById = new Map();
    this.mode = "network";
    this.edgeFilter = "";
    this.search = "";
    this.showLabels = opts.showLabels !== false;
    this.physics = opts.physics !== false;
    this.neuralPulse = true;
    this.selectedId = null;
    this.selectedEdge = null;
    this.hoverId = null;
    this.hoverEdge = null;
    this.focusId = null;
    this.scale = 1;
    this.ox = 0;
    this.oy = 0;
    this._dragNode = null;
    this._panning = false;
    this._last = null;
    this._t0 = performance.now();
    this._enter = 0;
    this.onSelect = null;
    this.onEdgeSelect = null;
    this.onHover = null;

    const ro = new ResizeObserver(() => this._resize());
    ro.observe(canvas.parentElement);
    this._bind();
    this._resize();
    this._loop();
    window.addEventListener("uil-theme-change", () => this.draw());
  }

  _bind() {
    const c = this.canvas;
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 0.9 : 1.11;
      const rect = c.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const wx = (mx - this.ox) / this.scale;
      const wy = (my - this.oy) / this.scale;
      this.scale = Math.max(0.2, Math.min(4, this.scale * factor));
      this.ox = mx - wx * this.scale;
      this.oy = my - wy * this.scale;
    }, { passive: false });

    c.addEventListener("mousedown", (e) => {
      const p = this._screenToWorld(e);
      const hit = this._hit(p.x, p.y);
      const edge = hit ? null : this._hitEdge(p.x, p.y);
      if (edge && !hit) {
        this.selectedEdge = edge;
        this.selectedId = null;
        if (this.onEdgeSelect) this.onEdgeSelect(edge);
        return;
      }
      if (hit) {
        this._dragNode = hit;
        this.selectedId = hit.id;
        this.selectedEdge = null;
        if (this.onSelect) this.onSelect(hit);
      } else {
        this.selectedId = null;
        this.selectedEdge = null;
        this._panning = true;
        c.classList.add("dragging");
      }
      this._last = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener("mousemove", (e) => {
      if (this._dragNode) {
        const p = this._screenToWorld(e);
        this._dragNode.x = p.x;
        this._dragNode.y = p.y;
        this._dragNode.fx = p.x;
        this._dragNode.fy = p.y;
        return;
      }
      if (this._panning && this._last) {
        this.ox += e.clientX - this._last.x;
        this.oy += e.clientY - this._last.y;
        this._last = { x: e.clientX, y: e.clientY };
        return;
      }
      const rect = this.canvas.getBoundingClientRect();
      if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) {
        if (this.hoverId || this.hoverEdge) { this.hoverId = null; this.hoverEdge = null; if (this.onHover) this.onHover(null, e, null); }
        return;
      }
      const p = this._screenToWorld(e);
      const hit = this._hit(p.x, p.y);
      const edge = hit ? null : this._hitEdge(p.x, p.y);
      const nid = hit ? hit.id : null;
      if (nid !== this.hoverId || (edge?.source !== this.hoverEdge?.source)) {
        this.hoverId = nid;
        this.hoverEdge = edge;
        c.style.cursor = hit ? "pointer" : edge ? "crosshair" : "grab";
        if (this.onHover) this.onHover(hit, e, edge);
      }
    });

    window.addEventListener("mouseup", () => {
      if (this._dragNode) { this._dragNode.fx = null; this._dragNode.fy = null; }
      this._dragNode = null;
      this._panning = false;
      this._last = null;
      this.canvas.classList.remove("dragging");
    });

    c.addEventListener("dblclick", () => this.fit());
  }

  _resize() {
    const parent = this.canvas.parentElement;
    const rect = parent.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, rect.width * dpr);
    this.canvas.height = Math.max(1, rect.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = rect.width;
    this.h = rect.height;
    if (!this._centered && this.nodes.length) this.fit();
  }

  setData({ nodes = [], edges = [] } = {}) {
    const prev = new Map(this.nodes.map((n) => [n.id, n]));
    this.nodes = nodes.map((n, i) => {
      const old = prev.get(n.id);
      const angle = (i / Math.max(1, nodes.length)) * Math.PI * 2;
      const conf = n.confidence ?? 0.5;
      return {
        id: n.id,
        label: n.label || n.id,
        confidence: conf,
        tier: n.tier ?? 3,
        meta: n.meta || {},
        x: old ? old.x : Math.cos(angle) * (100 + (i % 6) * 32),
        y: old ? old.y : Math.sin(angle) * (100 + (i % 6) * 32),
        vx: 0, vy: 0, fx: null, fy: null,
        r: 7 + Math.min(12, conf * 14),
        _born: performance.now(),
      };
    });
    this.nodeById = new Map(this.nodes.map((n) => [n.id, n]));
    this.edges = edges
      .filter((e) => this.nodeById.has(e.source) && this.nodeById.has(e.target))
      .map((e) => ({
        source: e.source, target: e.target,
        type: e.type || e.relation || "leads_to",
        weight: e.weight ?? 0.5,
      }));
    this._enter = performance.now();
    if (this.mode === "mindmap" || this.mode === "brain") this._layoutMindmap();
    if (this.mode === "orbit") this._layoutOrbit();
    this._centered = false;
    this.fit();
  }

  setMode(mode) {
    this.mode = mode;
    for (const n of this.nodes) { n.fx = null; n.fy = null; }
    if (mode === "mindmap" || mode === "brain") this._layoutMindmap();
    else if (mode === "orbit") this._layoutOrbit();
    this.fit();
  }

  setFocus(nodeId) { this.focusId = nodeId; }
  clearFocus() { this.focusId = null; }

  neighbors(id, depth = 1) {
    const out = new Set([id]);
    let frontier = [id];
    for (let d = 0; d < depth; d++) {
      const next = [];
      for (const e of this.edges) {
        if (frontier.includes(e.source) && !out.has(e.target)) { out.add(e.target); next.push(e.target); }
        if (frontier.includes(e.target) && !out.has(e.source)) { out.add(e.source); next.push(e.source); }
      }
      frontier = next;
    }
    return out;
  }

  _layoutMindmap() {
    if (!this.nodes.length) return;
    const deg = new Map(this.nodes.map((n) => [n.id, 0]));
    for (const e of this.edges) {
      deg.set(e.source, (deg.get(e.source) || 0) + 1);
      deg.set(e.target, (deg.get(e.target) || 0) + 1);
    }
    const root = [...this.nodes].sort(
      (a, b) => (deg.get(b.id) + b.confidence) - (deg.get(a.id) + a.confidence),
    )[0];
    const children = new Map();
    for (const e of this.edges) {
      if (!children.has(e.source)) children.set(e.source, []);
      children.get(e.source).push(e.target);
    }
    const placed = new Set();
    const place = (id, depth, a0, a1) => {
      const n = this.nodeById.get(id);
      if (!n || placed.has(id)) return;
      placed.add(id);
      const mid = (a0 + a1) / 2;
      const rad = depth * (this.mode === "brain" ? 120 : 140);
      n.x = Math.cos(mid) * rad; n.y = Math.sin(mid) * rad;
      n.fx = n.x; n.fy = n.y;
      const kids = (children.get(id) || []).filter((k) => !placed.has(k));
      kids.forEach((kid, i) => {
        const span = (a1 - a0) / Math.max(1, kids.length);
        place(kid, depth + 1, a0 + i * span, a0 + (i + 1) * span);
      });
    };
    place(root.id, 0, 0, Math.PI * 2);
    let oi = 0;
    for (const n of this.nodes) {
      if (placed.has(n.id)) continue;
      const a = (oi++ / Math.max(1, this.nodes.length - placed.size)) * Math.PI * 2;
      n.x = Math.cos(a) * 260; n.y = Math.sin(a) * 260; n.fx = n.x; n.fy = n.y;
    }
  }

  _layoutOrbit() {
    this.nodes.forEach((n, i) => {
      const a = (i / this.nodes.length) * Math.PI * 2;
      const rad = 80 + (i % 4) * 55;
      n.x = Math.cos(a) * rad; n.y = Math.sin(a) * rad; n.fx = n.x; n.fy = n.y;
    });
  }

  fit() {
    if (!this.nodes.length || !this.w) return;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const n of this.nodes) {
      minX = Math.min(minX, n.x); maxX = Math.max(maxX, n.x);
      minY = Math.min(minY, n.y); maxY = Math.max(maxY, n.y);
    }
    const pad = 70;
    const bw = Math.max(40, maxX - minX);
    const bh = Math.max(40, maxY - minY);
    this.scale = Math.min((this.w - pad * 2) / bw, (this.h - pad * 2) / bh, 1.8);
    this.ox = this.w / 2 - ((minX + maxX) / 2) * this.scale;
    this.oy = this.h / 2 - ((minY + maxY) / 2) * this.scale;
    this._centered = true;
  }

  _screenToWorld(e) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left - this.ox) / this.scale,
      y: (e.clientY - rect.top - this.oy) / this.scale,
    };
  }

  _hit(x, y) {
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const n = this.nodes[i];
      if (!this._nodeVisible(n)) continue;
      const dx = n.x - x, dy = n.y - y;
      if (dx * dx + dy * dy <= (n.r + 6) * (n.r + 6)) return n;
    }
    return null;
  }

  _distToSeg(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1, dy = y2 - y1;
    const len2 = dx * dx + dy * dy || 0.001;
    let t = ((px - x1) * dx + (py - y1) * dy) / len2;
    t = Math.max(0, Math.min(1, t));
    const lx = x1 + t * dx, ly = y1 + t * dy;
    const ddx = px - lx, ddy = py - ly;
    return Math.sqrt(ddx * ddx + ddy * ddy);
  }

  _hitEdge(x, y) {
    const thresh = 10 / this.scale;
    let best = null, bestD = thresh;
    for (const e of this._visibleEdges()) {
      const a = this.nodeById.get(e.source);
      const b = this.nodeById.get(e.target);
      const d = this._distToSeg(x, y, a.x, a.y, b.x, b.y);
      if (d < bestD) { bestD = d; best = e; }
    }
    return best;
  }

  _nodeVisible(n) {
    if (this.focusId) {
      const hood = this.neighbors(this.focusId, 2);
      if (!hood.has(n.id)) return false;
    }
    if (!this.search) return true;
    const q = this.search.toLowerCase();
    return n.id.toLowerCase().includes(q) || (n.label || "").toLowerCase().includes(q);
  }

  _visibleEdges() {
    return this.edges.filter((e) => {
      if (this.edgeFilter && e.type !== this.edgeFilter) return false;
      const a = this.nodeById.get(e.source);
      const b = this.nodeById.get(e.target);
      return a && b && this._nodeVisible(a) && this._nodeVisible(b);
    });
  }

  _tickPhysics() {
    if (!this.physics || this.mode === "mindmap" || this.mode === "brain" || this.mode === "orbit") return;
    if (this.nodes.length > 200) return;
    const N = this.nodes;
    for (let i = 0; i < N.length; i++) {
      for (let j = i + 1; j < N.length; j++) {
        let dx = N[j].x - N[i].x;
        let dy = N[j].y - N[i].y;
        let d2 = dx * dx + dy * dy || 0.01;
        const f = 2400 / d2;
        const d = Math.sqrt(d2);
        dx = (dx / d) * f; dy = (dy / d) * f;
        N[i].vx -= dx; N[i].vy -= dy;
        N[j].vx += dx; N[j].vy += dy;
      }
    }
    for (const e of this._visibleEdges()) {
      const a = this.nodeById.get(e.source);
      const b = this.nodeById.get(e.target);
      let dx = b.x - a.x, dy = b.y - a.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const ideal = 95;
      const f = (dist - ideal) * 0.022 * (0.35 + (e.weight || 0.5));
      dx = (dx / dist) * f; dy = (dy / dist) * f;
      a.vx += dx; a.vy += dy;
      b.vx -= dx; b.vy -= dy;
    }
    for (const n of N) {
      if (n.fx != null) { n.x = n.fx; n.y = n.fy; n.vx = 0; n.vy = 0; continue; }
      n.vx = (n.vx - n.x * 0.0006) * 0.86;
      n.vy = (n.vy - n.y * 0.0006) * 0.86;
      n.x += n.vx; n.y += n.vy;
    }
  }

  _loop() {
    const step = () => {
      if (!document.hidden) { this._tickPhysics(); this.draw(); }
      this._raf = requestAnimationFrame(step);
    };
    this._raf = requestAnimationFrame(step);
  }

  _drawEdge(ctx, e, a, b, t, sel, hov) {
    const col = EDGE_COLORS[e.type] || "#64748b";
    const w = 0.9 + (e.weight || 0.5) * 2.2;
    const curved = this.mode === "mindmap" || this.mode === "brain";
    ctx.beginPath();
    ctx.strokeStyle = col;
    ctx.globalAlpha = sel ? 0.95 : hov ? 0.75 : 0.28 + 0.35 * (e.weight || 0.5);
    ctx.lineWidth = sel ? w + 1.5 : w;
    if (curved) {
      const cx = (a.x + b.x) / 2 - (b.y - a.y) * 0.18;
      const cy = (b.y + a.y) / 2 + (b.x - a.x) * 0.18;
      ctx.moveTo(a.x, a.y);
      ctx.quadraticCurveTo(cx, cy, b.x, b.y);
    } else {
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
    }
    ctx.stroke();

    if (this.neuralPulse || sel || hov) {
      ctx.setLineDash([5, 12]);
      ctx.lineDashOffset = -t * 32;
      ctx.globalAlpha = sel ? 0.9 : 0.45;
      ctx.lineWidth = 1.2;
      ctx.stroke();
      ctx.setLineDash([]);
      const prog = ((t * 0.35 + e.source.charCodeAt(0) * 0.01) % 1);
      const px = curved
        ? this._qBezier(a.x, a.y, (a.x + b.x) / 2 - (b.y - a.y) * 0.18, (a.y + b.y) / 2 + (b.x - a.x) * 0.18, b.x, b.y, prog)
        : { x: a.x + (b.x - a.x) * prog, y: a.y + (b.y - a.y) * prog };
      ctx.beginPath();
      ctx.fillStyle = col;
      ctx.globalAlpha = 0.85;
      ctx.arc(px.x, px.y, sel ? 3.2 : 2.2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  _qBezier(x0, y0, cx, cy, x1, y1, t) {
    const u = 1 - t;
    return {
      x: u * u * x0 + 2 * u * t * cx + t * t * x1,
      y: u * u * y0 + 2 * u * t * cy + t * t * y1,
    };
  }

  _drawNeuron(ctx, n, t, sel, hov, fade) {
    const pulse = this.neuralPulse
      ? 1 + 0.1 * Math.sin(t * 2.2 + n.x * 0.02 + (n.phase || 0))
      : 1;
    const r = n.r * pulse;
    const tierCol = TIER_COLORS[(n.tier || 3) - 1] || TIER_COLORS[2];
    ctx.globalAlpha = fade;

    if (sel || hov) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 10, 0, Math.PI * 2);
      const glow = ctx.createRadialGradient(n.x, n.y, r, n.x, n.y, r + 10);
      glow.addColorStop(0, "rgba(74, 168, 255, 0.35)");
      glow.addColorStop(1, "transparent");
      ctx.fillStyle = glow;
      ctx.fill();
    }

    ctx.save();
    ctx.translate(n.x, n.y);
    ctx.rotate(Math.PI / 6);
    const sides = 6;
    ctx.beginPath();
    for (let i = 0; i <= sides; i++) {
      const a = (i / sides) * Math.PI * 2;
      const rr = r * (i % 2 === 0 ? 1 : 0.92);
      const px = Math.cos(a) * rr;
      const py = Math.sin(a) * rr;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.closePath();
    const g = ctx.createRadialGradient(-r * 0.2, -r * 0.2, 1, 0, 0, r);
    g.addColorStop(0, tierCol);
    g.addColorStop(1, _kgCssVar("--acc", "#3b74ff"));
    ctx.fillStyle = g;
    ctx.fill();
    ctx.strokeStyle = sel ? "#f8fafc" : "rgba(255,255,255,0.45)";
    ctx.lineWidth = sel ? 2 : 1;
    ctx.stroke();
    ctx.restore();

    if (this.showLabels && (this.scale > 0.45 || sel || hov)) {
      ctx.font = `${sel ? 600 : 500} ${10.5 / this.scale}px Inter, sans-serif`;
      ctx.fillStyle = _kgCssVar("--text", "#e2e8f0");
      ctx.textAlign = "center";
      ctx.fillText((n.label || n.id).slice(0, 24), n.x, n.y + r + 11 / this.scale);
    }
    ctx.globalAlpha = 1;
  }

  draw() {
    const ctx = this.ctx;
    const w = this.w || 1, h = this.h || 1;
    ctx.clearRect(0, 0, w, h);

    ctx.save();
    ctx.translate(this.ox, this.oy);
    ctx.scale(this.scale, this.scale);

    const t = (performance.now() - this._t0) / 1000;
    const edges = this._visibleEdges();
    const focusSet = this.focusId ? this.neighbors(this.focusId, 2) : null;

    for (const e of edges) {
      const a = this.nodeById.get(e.source);
      const b = this.nodeById.get(e.target);
      const sel = this.selectedEdge && this.selectedEdge.source === e.source
        && this.selectedEdge.target === e.target && this.selectedEdge.type === e.type;
      const hov = this.hoverEdge && this.hoverEdge.source === e.source
        && this.hoverEdge.target === e.target;
      this._drawEdge(ctx, e, a, b, t, sel, hov);
    }

    for (const n of this.nodes) {
      if (!this._nodeVisible(n)) continue;
      const sel = n.id === this.selectedId;
      const hov = n.id === this.hoverId;
      let fade = 1;
      if (focusSet && !focusSet.has(n.id)) fade = 0.12;
      n.phase = n.phase ?? n.x * 0.01;
      this._drawNeuron(ctx, n, t, sel, hov, fade);
    }
    ctx.restore();
  }
}

window.KnowledgeGraph = KnowledgeGraph;
window.EDGE_COLORS = EDGE_COLORS;
