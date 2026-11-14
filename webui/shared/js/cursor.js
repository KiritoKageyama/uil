/* ════════════════════════════════════════════════════════════════════
   cursor.js — fluid ink pointer (izanami-style).
   The mouse drags a ribbon of FLUID through space: soft blobs are
   spawned along the pointer path, inherit its velocity, drift with a
   curl-noise field, expand and dissolve. On dark themes the fluid glows
   like nebula gas (additive); on light themes it behaves like ink in
   water (multiply pigment). Click = a splash burst.
   Plus a bright core dot + eased halo ring. Fine pointers only;
   touch + reduced-motion keep the native cursor. Self-contained.
   ════════════════════════════════════════════════════════════════════ */

"use strict";

(function () {
  const finePointer = window.matchMedia("(pointer: fine)").matches;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!finePointer || reduced) return;

  const css = `
    .cur-on, .cur-on * { cursor: none !important; }
    .cur-on input, .cur-on textarea, .cur-on select, .cur-on [contenteditable] { cursor: text !important; }
    #curFluid { position: fixed; inset: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: 99997; }
    .cur-core { position: fixed; top: 0; left: 0; width: 6px; height: 6px; border-radius: 50%;
      background: #ffffff; box-shadow: 0 0 10px var(--acc, #3b74ff), 0 0 22px rgba(59,116,255,0.5);
      transform: translate(-50%, -50%); pointer-events: none; z-index: 100000; will-change: transform; }
    [data-theme="light"] .cur-core { background: var(--acc, #2b55d6); box-shadow: 0 0 10px rgba(43,85,214,0.55); }
    .cur-ring { position: fixed; top: 0; left: 0; width: 30px; height: 30px; border-radius: 50%;
      border: 1px solid rgba(122, 166, 255, 0.75); transform: translate(-50%, -50%); pointer-events: none;
      z-index: 99999; mix-blend-mode: screen; will-change: transform, width, height;
      transition: width .28s var(--ease-out, ease), height .28s var(--ease-out, ease),
        border-color .28s, background .28s, opacity .28s; }
    [data-theme="light"] .cur-ring { mix-blend-mode: normal; border-color: rgba(43, 85, 214, 0.5); }
    .cur-ring.hot { width: 48px; height: 48px; border-color: var(--neural, #b48bff);
      background: rgba(180, 139, 255, 0.07); }
    .cur-ring.down { width: 20px; height: 20px; border-color: var(--c-yellow, #ffd23f); }
    .cur-ring.hidden { opacity: 0; }
  `;
  const style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  const fluid = document.createElement("canvas");
  fluid.id = "curFluid";
  fluid.setAttribute("aria-hidden", "true");
  const core = document.createElement("div"); core.className = "cur-core";
  const ring = document.createElement("div"); ring.className = "cur-ring hidden";
  document.body.appendChild(fluid); document.body.appendChild(core); document.body.appendChild(ring);
  document.documentElement.classList.add("cur-on");

  const ctx = fluid.getContext("2d");
  let W = 0, H = 0, dpr = 1;
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 1.5); // fluid is soft — cap DPR for perf
    W = innerWidth; H = innerHeight;
    fluid.width = W * dpr; fluid.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener("resize", resize);
  resize();

  /* ── fluid blobs ── */
  const INK_VARS = ["--p1", "--p5", "--p1", "--p2", "--p6"]; // blue-leaning mix
  function cssv(n, fb) { const v = getComputedStyle(document.documentElement).getPropertyValue(n).trim(); return v || fb; }
  function hexRgb(hex) {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [59, 116, 255];
  }

  const blobs = [];
  const MAX_BLOBS = 150;
  function spawn(x, y, vx, vy, big) {
    if (blobs.length >= MAX_BLOBS) blobs.shift();
    blobs.push({
      x, y,
      vx: vx + (Math.random() - 0.5) * 0.6,
      vy: vy + (Math.random() - 0.5) * 0.6,
      r: (big ? 22 : 10) + Math.random() * (big ? 26 : 14),
      grow: big ? 1.028 : 1.018,
      life: 1,
      decay: big ? 0.016 : 0.024,
      col: hexRgb(cssv(INK_VARS[(Math.random() * INK_VARS.length) | 0], "#3b74ff")),
      seed: Math.random() * 6.28,
    });
  }

  function isInteractive(el) {
    return !!(el && el.closest && el.closest('a, button, [role="button"], .btn, .chip, .seg button, .nav-item, .tab-btn, label, input[type="range"], .cell-card, .cell-mini, .top-node, .case, .vision-card, .film, .mod-card, .um-bay, .ub-node, .pillar, .intro-skip'));
  }

  const target = { x: innerWidth / 2, y: innerHeight / 2 };
  const rpos = { x: target.x, y: target.y };
  let last = null, shown = false;

  window.addEventListener("pointermove", (e) => {
    if (e.pointerType === "touch") return;
    target.x = e.clientX; target.y = e.clientY;
    core.style.transform = `translate(${e.clientX}px, ${e.clientY}px) translate(-50%,-50%)`;
    if (!shown) { shown = true; ring.classList.remove("hidden"); }
    ring.classList.toggle("hot", isInteractive(e.target));
    // seed fluid along the swept path so fast moves leave a continuous ribbon
    if (last) {
      const dx = e.clientX - last.x, dy = e.clientY - last.y;
      const dist = Math.hypot(dx, dy);
      const steps = Math.max(1, Math.min(6, (dist / 14) | 0));
      for (let i = 1; i <= steps; i++) {
        const t = i / steps;
        spawn(last.x + dx * t, last.y + dy * t, dx * 0.055, dy * 0.055, false);
      }
    } else spawn(e.clientX, e.clientY, 0, 0, false);
    last = { x: e.clientX, y: e.clientY };
  }, { passive: true });

  window.addEventListener("pointerdown", (e) => {
    if (e.pointerType === "touch") return;
    ring.classList.add("down");
    // splash: a burst of fluid thrown outward
    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * Math.PI * 2 + Math.random() * 0.4;
      const sp = 1.6 + Math.random() * 2.4;
      spawn(e.clientX, e.clientY, Math.cos(a) * sp, Math.sin(a) * sp, true);
    }
  });
  window.addEventListener("pointerup", () => ring.classList.remove("down"));
  window.addEventListener("pointerleave", () => { shown = false; ring.classList.add("hidden"); last = null; });
  document.addEventListener("mouseleave", () => { shown = false; ring.classList.add("hidden"); last = null; });

  let t0 = performance.now();
  function frame() {
    const t = (performance.now() - t0) / 1000;
    // halo ring eases toward the pointer
    rpos.x += (target.x - rpos.x) * 0.18;
    rpos.y += (target.y - rpos.y) * 0.18;
    ring.style.transform = `translate(${rpos.x}px, ${rpos.y}px) translate(-50%,-50%)`;

    ctx.clearRect(0, 0, W, H);
    if (blobs.length) {
      const light = document.documentElement.dataset.theme === "light";
      ctx.globalCompositeOperation = light ? "multiply" : "lighter";
      for (let i = blobs.length - 1; i >= 0; i--) {
        const b = blobs[i];
        // curl-ish drift: a slowly rotating flow field bends the ribbon
        const ang = Math.sin(b.x * 0.006 + t * 0.7 + b.seed) + Math.cos(b.y * 0.007 - t * 0.5);
        b.vx += Math.cos(ang * Math.PI) * 0.028;
        b.vy += Math.sin(ang * Math.PI) * 0.028;
        b.vx *= 0.955; b.vy *= 0.955;
        b.x += b.vx; b.y += b.vy;
        b.r *= b.grow;
        b.life -= b.decay;
        if (b.life <= 0.02) { blobs.splice(i, 1); continue; }
        const a = light ? b.life * 0.16 : b.life * 0.22;
        const g = ctx.createRadialGradient(b.x, b.y, 0, b.x, b.y, b.r);
        const [cr, cg, cb] = b.col;
        g.addColorStop(0, `rgba(${cr},${cg},${cb},${a})`);
        g.addColorStop(0.55, `rgba(${cr},${cg},${cb},${a * 0.5})`);
        g.addColorStop(1, `rgba(${cr},${cg},${cb},0)`);
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalCompositeOperation = "source-over";
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  // tiny debug hook so tooling can confirm the fluid is alive
  window.CursorFluid = { count: () => blobs.length };
})();
