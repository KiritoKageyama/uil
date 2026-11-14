/* Ambient neural field behind the console — full-viewport WebGL mesh.
   Matches the pitch site particle language so both surfaces feel one system. */

"use strict";

(function initNeuralBg() {
  function ensureCanvas() {
    let c = document.getElementById("neuralBg");
    if (!c) {
      c = document.createElement("canvas");
      c.id = "neuralBg";
      c.setAttribute("aria-hidden", "true");
      document.body.prepend(c);
    }
    // Force true fullscreen — inset alone can collapse before layout
    Object.assign(c.style, {
      position: "fixed",
      top: "0",
      left: "0",
      right: "0",
      bottom: "0",
      width: "100vw",
      height: "100vh",
      zIndex: "0",
      pointerEvents: "none",
      display: "block",
    });
    return c;
  }

  function boot() {
    const c = ensureCanvas();
    if (window.NeuralField) {
      const ctl = window.NeuralField.mount(c, {
        mode: "hero",
        density: 11000,
        maxNodes: 480,
        minNodes: 180,
        linkRadius: 0.36,
        globalPointer: true,
      });
      if (!c.classList.contains("no-webgl")) {
        window.__uilField = ctl || { setIntensity() {} };
        // Re-sync after fonts/layout settle so the field covers the full window
        const kick = () => window.dispatchEvent(new Event("resize"));
        requestAnimationFrame(() => requestAnimationFrame(kick));
        setTimeout(kick, 120);
        setTimeout(kick, 600);
        return;
      }
      c.remove();
    }

    // ── 2D fullscreen fallback ──
    const canvas = ensureCanvas();
    const ctx = canvas.getContext("2d");
    let w = 0, h = 0, nodes = [], edges = [], raf = 0, t0 = performance.now();
    const ptr = { x: 0, y: 0, on: false };

    function css(name, fb) {
      const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      return v || fb;
    }

    function seed() {
      const n = Math.min(140, Math.max(70, Math.floor((w * h) / 14000)));
      nodes = Array.from({ length: n }, (_, i) => ({
        x: Math.random() * w, y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.32, vy: (Math.random() - 0.5) * 0.32,
        r: 1.5 + Math.random() * 2.4, phase: Math.random() * Math.PI * 2,
        kind: i % 7 === 0 ? "core" : "node",
      }));
      edges = [];
      for (let i = 0; i < nodes.length; i++)
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y;
          if (dx * dx + dy * dy < 220 * 220) edges.push({ a: i, b: j });
        }
    }

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = window.innerWidth; h = window.innerHeight;
      canvas.width = w * dpr; canvas.height = h * dpr;
      canvas.style.width = w + "px"; canvas.style.height = h + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      seed();
    }

    function tick() {
      const t = (performance.now() - t0) / 1000;
      ctx.clearRect(0, 0, w, h);
      const acc = css("--acc-2", "#7aa6ff"), accSoft = css("--acc", "#3b74ff"), dim = css("--text-3", "#64748b");
      for (const n of nodes) {
        n.x += n.vx; n.y += n.vy;
        if (n.x < 0 || n.x > w) n.vx *= -1;
        if (n.y < 0 || n.y > h) n.vy *= -1;
        if (ptr.on) {
          const dx = n.x - ptr.x, dy = n.y - ptr.y, d = Math.hypot(dx, dy) || 1;
          if (d < 150) {
            const f = (1 - d / 150) * 1.9;
            n.x += (dx / d) * f; n.y += (dy / d) * f;
          }
        }
      }
      for (const e of edges) {
        const a = nodes[e.a], b = nodes[e.b];
        const dx = b.x - a.x, dy = b.y - a.y, dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > 230) continue;
        ctx.beginPath(); ctx.strokeStyle = dim; ctx.globalAlpha = (1 - dist / 230) * 0.38;
        ctx.lineWidth = 0.75; ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
      for (const n of nodes) {
        const pulse = n.kind === "core" ? 1 + 0.35 * Math.sin(t * 2 + n.phase) : 1 + 0.15 * Math.sin(t * 1.6 + n.phase);
        ctx.beginPath(); ctx.fillStyle = n.kind === "core" ? acc : accSoft;
        ctx.globalAlpha = n.kind === "core" ? 0.95 : 0.75;
        ctx.arc(n.x, n.y, n.r * pulse, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
      raf = requestAnimationFrame(tick);
    }

    window.addEventListener("pointermove", (e) => { ptr.x = e.clientX; ptr.y = e.clientY; ptr.on = true; }, { passive: true });
    window.addEventListener("pointerleave", () => { ptr.on = false; });
    window.addEventListener("resize", resize);
    resize(); tick();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
