/* ════════════════════════════════════════════════════════════════════
   neural-field.js — animated connected-node background field.

   RECONSTRUCTION NOTE (2026-08-03): the original of this file was lost in a
   botched recovery (its entry had been replaced by a symlink into a browser
   cache that no longer held the real bytes). No copy survived in any sibling
   project, the _recovery fragments, or VS Code history. This is a clean,
   self-contained reimplementation that honors the exact public API the two
   callers use — `window.NeuralField.mount(canvas, opts)` returning a
   controller with `setIntensity(n)` and `destroy()` — so both surfaces keep
   an animated backdrop and nothing throws. It is NOT guaranteed to match the
   original look pixel-for-pixel; swap in the real file if it ever resurfaces.

   Callers (webui/desktop/js/neural-bg.js, webui/frontend/js/site.js) pass:
     { mode, density, maxNodes, minNodes, linkRadius, globalPointer }
   ════════════════════════════════════════════════════════════════════ */
(function () {
  "use strict";

  function accent() {
    // Read the live theme accent so the field recolors with light/dark.
    const s = getComputedStyle(document.documentElement);
    const a = (s.getPropertyValue("--acc") || "#3b74ff").trim();
    const b = (s.getPropertyValue("--acc-2") || a).trim();
    return { a, b };
  }

  function hexToRgb(hex) {
    const m = /^#?([0-9a-f]{6})$/i.exec(hex || "");
    if (!m) return [123, 140, 255];
    const n = parseInt(m[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  function mount(canvas, opts) {
    if (!canvas || !canvas.getContext) return { setIntensity() {}, destroy() {} };
    const o = Object.assign(
      { density: 13000, maxNodes: 420, minNodes: 150, linkRadius: 0.34, globalPointer: true },
      opts || {}
    );
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return { setIntensity() {}, destroy() {} };

    let dpr = Math.min(window.devicePixelRatio || 1, 2);
    let W = 0, H = 0, nodes = [], raf = 0, running = true, intensity = 1;
    const pointer = { x: -1e5, y: -1e5, sx: -1e5, sy: -1e5, on: false };

    function count() {
      const area = W * H;
      const n = Math.round(area / (o.density * dpr));
      return Math.max(o.minNodes, Math.min(o.maxNodes, n));
    }

    function seed() {
      const n = count();
      nodes = new Array(n).fill(0).map(() => ({
        x: Math.random() * W,
        y: Math.random() * H,
        vx: 0, vy: 0,                                    // interaction velocity
        dvx: (Math.random() - 0.5) * 0.14 * dpr,         // idle drift
        dvy: (Math.random() - 0.5) * 0.14 * dpr,
        r: (Math.random() * 1.3 + 0.6) * dpr,
      }));
    }

    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      W = Math.max(1, Math.round((rect.width || window.innerWidth) * dpr));
      H = Math.max(1, Math.round((rect.height || window.innerHeight) * dpr));
      canvas.width = W;
      canvas.height = H;
      seed();
    }

    function onMove(e) {
      const rect = canvas.getBoundingClientRect();
      pointer.x = (e.clientX - rect.left) * dpr;
      pointer.y = (e.clientY - rect.top) * dpr;
      if (!pointer.on) { pointer.sx = pointer.x; pointer.sy = pointer.y; } // no jump on first move
      pointer.on = true;
    }
    function onLeave() { pointer.on = false; pointer.x = pointer.y = -1e5; }

    function step() {
      if (!running) return;
      ctx.clearRect(0, 0, W, H);
      const { a, b } = accent();
      const ca = hexToRgb(a), cb = hexToRgb(b);
      const linkPx = o.linkRadius * Math.min(W, H);
      const link2 = linkPx * linkPx;

      // smoothed cursor for fluid influence
      if (pointer.on) {
        pointer.sx += (pointer.x - pointer.sx) * 0.2;
        pointer.sy += (pointer.y - pointer.sy) * 0.2;
      }
      const R = 190 * dpr, R2 = R * R;
      for (const p of nodes) {
        if (pointer.on) {
          const dx = pointer.sx - p.x, dy = pointer.sy - p.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < R2 && d2 > 1) {
            const d = Math.sqrt(d2);
            const pull = (1 - d / R);
            if (d < 46 * dpr) {
              // very close: gently push out so the cursor "parts" the field
              p.vx -= (dx / d) * pull * 0.9;
              p.vy -= (dy / d) * pull * 0.9;
            } else {
              // mid range: draw in with a tangential swirl (stirring)
              p.vx += (dx / d) * pull * 0.42 + (-dy / d) * pull * 0.28;
              p.vy += (dy / d) * pull * 0.42 + (dx / d) * pull * 0.28;
            }
          }
        }
        p.vx *= 0.94; p.vy *= 0.94;            // damping keeps it calm
        p.x += p.vx; p.y += p.vy;
        // gentle drift so idle field still breathes
        p.x += p.dvx; p.y += p.dvy;
        if (p.x < 0 || p.x > W) { p.dvx *= -1; p.x = Math.max(0, Math.min(W, p.x)); }
        if (p.y < 0 || p.y > H) { p.dvy *= -1; p.y = Math.max(0, Math.min(H, p.y)); }
      }

      // links
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y;
          const d2 = dx * dx + dy * dy;
          if (d2 < link2) {
            const t = 1 - d2 / link2;
            const m = t; // mix a→b by proximity
            const r = Math.round(ca[0] + (cb[0] - ca[0]) * m);
            const g = Math.round(ca[1] + (cb[1] - ca[1]) * m);
            const bl = Math.round(ca[2] + (cb[2] - ca[2]) * m);
            ctx.strokeStyle = `rgba(${r},${g},${bl},${(t * 0.28 * intensity).toFixed(3)})`;
            ctx.lineWidth = Math.max(0.5, t * 1.1 * dpr);
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.stroke();
          }
        }
      }
      // cursor constellation: bright links from the cursor to nearby stars,
      // and a soft glow that follows it — the interactive layer
      if (pointer.on) {
        const CR = 200 * dpr, CR2 = CR * CR;
        const glow = ctx.createRadialGradient(pointer.sx, pointer.sy, 0, pointer.sx, pointer.sy, CR * 0.9);
        glow.addColorStop(0, `rgba(${cb[0]},${cb[1]},${cb[2]},${(0.10 * intensity).toFixed(3)})`);
        glow.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = glow;
        ctx.beginPath(); ctx.arc(pointer.sx, pointer.sy, CR * 0.9, 0, Math.PI * 2); ctx.fill();
        for (const p of nodes) {
          const dx = p.x - pointer.sx, dy = p.y - pointer.sy, d2 = dx * dx + dy * dy;
          if (d2 < CR2) {
            const t = 1 - d2 / CR2;
            ctx.strokeStyle = `rgba(${cb[0]},${cb[1]},${cb[2]},${(t * 0.5 * intensity).toFixed(3)})`;
            ctx.lineWidth = Math.max(0.5, t * 1.4 * dpr);
            ctx.beginPath(); ctx.moveTo(pointer.sx, pointer.sy); ctx.lineTo(p.x, p.y); ctx.stroke();
            // brighten the star itself when the cursor is near
            p._hot = t;
          } else if (p._hot) { p._hot *= 0.9; }
        }
      }
      // nodes
      for (const p of nodes) {
        const hot = p._hot || 0;
        const a = (0.6 + hot * 0.4) * intensity;
        ctx.fillStyle = hot > 0.05
          ? `rgba(${Math.round(ca[0] + (255 - ca[0]) * hot)},${Math.round(ca[1] + (255 - ca[1]) * hot)},${Math.round(ca[2] + (255 - ca[2]) * hot)},${a.toFixed(3)})`
          : `rgba(${ca[0]},${ca[1]},${ca[2]},${(0.62 * intensity).toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r * (1 + hot * 0.9), 0, Math.PI * 2);
        ctx.fill();
      }
      raf = requestAnimationFrame(step);
    }

    resize();
    window.addEventListener("resize", resize, { passive: true });
    if (o.globalPointer) {
      window.addEventListener("mousemove", onMove, { passive: true });
      window.addEventListener("mouseleave", onLeave, { passive: true });
    }
    raf = requestAnimationFrame(step);

    return {
      setIntensity(n) { intensity = Math.max(0, Math.min(1.5, Number(n) || 0)); },
      destroy() {
        running = false;
        cancelAnimationFrame(raf);
        window.removeEventListener("resize", resize);
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseleave", onLeave);
        ctx.clearRect(0, 0, W, H);
      },
    };
  }

  window.NeuralField = { mount };
})();
