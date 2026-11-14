/* ════════════════════════════════════════════════════════════════════
   charts.js — DPR-aware animated canvas line chart with crosshair +
   tooltip. Theme-aware: colors resolved from CSS variables at draw
   time so a theme toggle repaints correctly.
   ════════════════════════════════════════════════════════════════════ */

"use strict";

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

class LineChart {
  /* series: [{key, label, colorVar}] — colorVar is a CSS custom property name */
  constructor(canvas, tipEl, series) {
    this.canvas = canvas;
    this.tip = tipEl;
    this.series = series;
    this.data = [];
    this.anim = 1;
    this._raf = null;
    this.hoverIdx = null;
    const ro = new ResizeObserver(() => this.draw());
    ro.observe(canvas.parentElement);
    canvas.parentElement.addEventListener("mousemove", (e) => this.hover(e));
    canvas.parentElement.addEventListener("mouseleave", () => {
      this.hoverIdx = null; this.tip.classList.remove("show"); this.draw();
    });
    window.addEventListener("uil-theme-change", () => this.draw());
  }

  push(pt) { this.data.push(pt); this.animateIn(); }
  setData(rows) { this.data = rows; this.hoverIdx = null; this.animateIn(); }
  reset() { this.data = []; this.hoverIdx = null; this.draw(); }

  animateIn() {
    cancelAnimationFrame(this._raf);
    // rAF is throttled to zero in hidden windows — draw synchronously
    // so data is never invisible when the window is restored.
    if (document.hidden) { this.anim = 1; this.draw(); return; }
    const t0 = performance.now();
    const tick = (t) => {
      this.anim = Math.min(1, (t - t0) / 420);
      this.draw();
      if (this.anim < 1) this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
  }

  layout() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, rect.width * dpr);
    this.canvas.height = Math.max(1, rect.height * dpr);
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: rect.width, h: rect.height, pad: { l: 42, r: 12, t: 10, b: 22 } };
  }

  scales(w, h, pad) {
    const n = this.data.length;
    const xs = (i) => pad.l + (n <= 1 ? 0 : (i / (n - 1)) * (w - pad.l - pad.r));
    let lo = Infinity, hi = -Infinity;
    for (const d of this.data) for (const s of this.series) {
      const v = d[s.key];
      if (v !== null && v !== undefined && !Number.isNaN(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    }
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (hi - lo < 0.02) { const m = (hi + lo) / 2; lo = m - 0.01; hi = m + 0.01; }
    const span = hi - lo;
    lo -= span * 0.12; hi += span * 0.12;
    const ys = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (h - pad.t - pad.b);
    return { xs, ys, lo, hi };
  }

  draw() {
    const { ctx, w, h, pad } = this.layout();
    ctx.clearRect(0, 0, w, h);
    const { xs, ys, lo, hi } = this.scales(w, h, pad);
    const gridCol = cssVar("--chart-grid") || "rgba(128,128,128,0.12)";
    const textCol = cssVar("--text-3") || "#888";
    const surface = cssVar("--surface") || "#15151c";

    ctx.strokeStyle = gridCol;
    ctx.fillStyle = textCol;
    ctx.font = "10.5px 'Segoe UI', sans-serif";
    ctx.lineWidth = 1;
    for (let g = 0; g <= 3; g++) {
      const v = lo + ((hi - lo) * g) / 3;
      const y = ys(v);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.fillText(v.toFixed(2), 6, y + 3.5);
    }

    if (!this.data.length) {
      ctx.fillStyle = textCol;
      ctx.font = "12px 'Segoe UI', sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("waiting for data…", w / 2, h / 2);
      ctx.textAlign = "left";
      return;
    }

    const n = this.data.length;
    const upto = Math.max(1, Math.ceil(n * (n <= 2 ? 1 : this.anim)));

    for (const s of this.series) {
      const color = cssVar(s.colorVar) || "#7c6cff";
      // area fill
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < upto; i++) {
        const v = this.data[i][s.key];
        if (v === null || v === undefined) continue;
        const x = xs(i), y = ys(v);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      }
      if (started) {
        const grad = ctx.createLinearGradient(0, pad.t, 0, h - pad.b);
        grad.addColorStop(0, color + "30");
        grad.addColorStop(1, color + "00");
        ctx.lineTo(xs(upto - 1), h - pad.b);
        ctx.lineTo(xs(0), h - pad.b);
        ctx.closePath();
        ctx.fillStyle = grad;
        ctx.fill();
      }
      // stroke
      ctx.beginPath();
      started = false;
      for (let i = 0; i < upto; i++) {
        const v = this.data[i][s.key];
        if (v === null || v === undefined) continue;
        const x = xs(i), y = ys(v);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.shadowColor = color;
      ctx.shadowBlur = 6;
      ctx.stroke();
      ctx.shadowBlur = 0;
      // end marker
      const last = this.data[upto - 1][s.key];
      if (last !== null && last !== undefined) {
        const x = xs(upto - 1), y = ys(last);
        ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = color; ctx.fill();
        ctx.strokeStyle = surface; ctx.lineWidth = 2; ctx.stroke();
      }
    }

    // crosshair
    if (this.hoverIdx !== null && this.hoverIdx !== undefined) {
      const x = xs(this.hoverIdx);
      ctx.strokeStyle = "rgba(128,128,160,0.35)";
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
      ctx.setLineDash([]);
      for (const s of this.series) {
        const v = this.data[this.hoverIdx][s.key];
        if (v === null || v === undefined) continue;
        ctx.beginPath(); ctx.arc(x, ys(v), 5, 0, Math.PI * 2);
        ctx.fillStyle = cssVar(s.colorVar);
        ctx.fill();
        ctx.strokeStyle = surface; ctx.lineWidth = 2; ctx.stroke();
      }
    }
  }

  hover(e) {
    if (!this.data.length) return;
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const pad = { l: 42, r: 12 };
    const w = rect.width;
    const mx = e.clientX - rect.left;
    const n = this.data.length;
    const frac = (mx - pad.l) / Math.max(1, w - pad.l - pad.r);
    const idx = Math.round(frac * (n - 1));
    this.hoverIdx = Math.max(0, Math.min(n - 1, idx));
    const d = this.data[this.hoverIdx];
    const rows = this.series
      .map((s) => `<div class="t-row"><i style="background:${cssVar(s.colorVar)}"></i>${esc(s.label)}: <b>${fmt(d[s.key])}</b></div>`)
      .join("");
    const head = d.stage !== undefined ? `stage ${d.stage}` : `#${this.hoverIdx}`;
    this.tip.innerHTML = `<div class="t-row"><b>${esc(head)}</b></div>${rows}`;
    const xsPos = pad.l + (n <= 1 ? 0 : (this.hoverIdx / (n - 1)) * (w - pad.l - pad.r));
    this.tip.style.left = Math.max(60, Math.min(w - 60, xsPos)) + "px";
    this.tip.style.top = "48px";
    this.tip.classList.add("show");
    this.draw();
  }
}

/* redraw all charts when the window returns from hidden/minimized */
const _allCharts = [];
function registerChart(c) { _allCharts.push(c); return c; }
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) _allCharts.forEach((c) => c.draw());
});
