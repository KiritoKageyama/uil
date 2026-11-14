/* ════════════════════════════════════════════════════════════════════
   logo.js — FIGPSO UIL · glass gravity brain
   No cartoon face. Realistic brain lobes · glass hex · orbits.
   Particle / motion physics: motes fall into the well (heavy pull).
   ════════════════════════════════════════════════════════════════════ */

"use strict";

(function () {
  const UID = () => `uil${Math.random().toString(36).slice(2, 9)}`;
  const VER = "6";
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function markSvg(uid) {
    const g = (n) => `${uid}-${n}`;
    return `
<svg class="ul-svg" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">
  <defs>
    <radialGradient id="${g("well")}" cx="50%" cy="48%" r="55%">
      <stop offset="0%" stop-color="#dce8ff" stop-opacity="0.28"/>
      <stop offset="28%" stop-color="#5a8fff" stop-opacity="0.2"/>
      <stop offset="62%" stop-color="#3b74ff" stop-opacity="0.07"/>
      <stop offset="100%" stop-color="#050506" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="${g("glass")}" cx="34%" cy="28%" r="72%">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0.55"/>
      <stop offset="22%" stop-color="#b8d0ff" stop-opacity="0.28"/>
      <stop offset="55%" stop-color="#3b74ff" stop-opacity="0.55"/>
      <stop offset="100%" stop-color="#0a1838" stop-opacity="0.92"/>
    </radialGradient>
    <radialGradient id="${g("sphere")}" cx="38%" cy="32%" r="68%">
      <stop offset="0%" stop-color="#2c4574"/>
      <stop offset="50%" stop-color="#0d1730"/>
      <stop offset="100%" stop-color="#05070e"/>
    </radialGradient>
    <linearGradient id="${g("brainL")}" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#9eb8e8" stop-opacity="0.55"/>
      <stop offset="55%" stop-color="#5a7fc4" stop-opacity="0.4"/>
      <stop offset="100%" stop-color="#2a4a8a" stop-opacity="0.25"/>
    </linearGradient>
    <linearGradient id="${g("brainR")}" x1="1" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#8fd4c0" stop-opacity="0.4"/>
      <stop offset="55%" stop-color="#5a8fff" stop-opacity="0.38"/>
      <stop offset="100%" stop-color="#2a4a8a" stop-opacity="0.25"/>
    </linearGradient>
    <linearGradient id="${g("oa")}" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#cfe0ff" stop-opacity="0.15"/>
      <stop offset="45%" stop-color="#a8c8ff"/>
      <stop offset="100%" stop-color="#6f9dff" stop-opacity="0.2"/>
    </linearGradient>
    <linearGradient id="${g("ob")}" x1="1" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#21d99a" stop-opacity="0.25"/>
      <stop offset="45%" stop-color="#8fd4c0"/>
      <stop offset="100%" stop-color="#6f9dff" stop-opacity="0.2"/>
    </linearGradient>
    <linearGradient id="${g("oc")}" x1="0" y1="1" x2="1" y2="0">
      <stop offset="0%" stop-color="#ff8a34" stop-opacity="0.3"/>
      <stop offset="45%" stop-color="#ffc08a"/>
      <stop offset="100%" stop-color="#9ec0ff" stop-opacity="0.2"/>
    </linearGradient>
    <filter id="${g("soft")}" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="0.7"/>
    </filter>
    <filter id="${g("glow")}" x="-60%" y="-60%" width="220%" height="220%">
      <feGaussianBlur stdDeviation="1.2" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <clipPath id="${g("clip")}"><circle cx="32" cy="32" r="13.8"/></clipPath>
  </defs>

  <!-- gravity well -->
  <ellipse class="ul-well" cx="32" cy="33" rx="31" ry="28" fill="url(#${g("well")})"/>

  <!-- field lines bending INTO the well -->
  <g class="ul-field" fill="none" stroke-linecap="round">
    <path d="M4 16 C20 22, 44 22, 60 16" stroke="#9eb8e8" stroke-width="0.5" opacity="0.35"/>
    <path d="M2 32 C18 36, 46 36, 62 32" stroke="#8eb4ff" stroke-width="0.65" opacity="0.45"/>
    <path d="M4 48 C20 42, 44 42, 60 48" stroke="#9eb8e8" stroke-width="0.5" opacity="0.35"/>
    <path d="M12 6 C22 24, 22 40, 12 58" stroke="#6f9dff" stroke-width="0.45" opacity="0.3"/>
    <path d="M52 6 C42 24, 42 40, 52 58" stroke="#6f9dff" stroke-width="0.45" opacity="0.3"/>
  </g>

  <!-- orbits (equal weight) -->
  <g class="ul-orbit-a" fill="none">
    <ellipse cx="32" cy="32" rx="25.5" ry="9" stroke="url(#${g("oa")})"
      stroke-width="1.15" transform="rotate(26 32 32)"/>
  </g>
  <g class="ul-orbit-b" fill="none">
    <ellipse cx="32" cy="32" rx="25.5" ry="9" stroke="url(#${g("ob")})"
      stroke-width="1.15" transform="rotate(-38 32 32)"/>
  </g>
  <g class="ul-orbit-c" fill="none">
    <ellipse cx="32" cy="32" rx="24" ry="8.4" stroke="url(#${g("oc")})"
      stroke-width="1.1" transform="rotate(76 32 32)"/>
  </g>

  <g class="ul-sats" filter="url(#${g("glow")})">
    <g transform="rotate(26 32 32)"><circle cx="57.5" cy="32" r="1.55" fill="#eaf2ff"/></g>
    <g transform="rotate(-38 32 32)"><circle cx="57.5" cy="32" r="1.35" fill="#21d99a"/></g>
    <g transform="rotate(76 32 32)"><circle cx="56.4" cy="32" r="1.3" fill="#ff8a34"/></g>
  </g>

  <!-- anatomical-ish brain lobes (realistic proportions, soft tissue tones) -->
  <g class="ul-brain" filter="url(#${g("soft")})">
    <path fill="url(#${g("brainL")})" d="
      M17.2 30.5
      C15.4 24.2, 18.8 17.8, 24.6 16.6
      C27.8 15.9, 30.2 17.8, 30.8 20.8
      C31.6 18.4, 29.8 15.2, 26.4 14.8
      C21.2 14.2, 16.2 18.6, 15.8 24.5
      C15.5 29.2, 17.5 34.8, 20.8 38.6
      C23.6 41.8, 27.2 42.6, 29.4 40.8
      C26.2 41.2, 22.4 38.4, 20.6 34.2
      C18.8 30.5, 18.2 30.8, 17.2 30.5 Z"/>
    <path fill="url(#${g("brainR")})" d="
      M46.8 30.5
      C48.6 24.2, 45.2 17.8, 39.4 16.6
      C36.2 15.9, 33.8 17.8, 33.2 20.8
      C32.4 18.4, 34.2 15.2, 37.6 14.8
      C42.8 14.2, 47.8 18.6, 48.2 24.5
      C48.5 29.2, 46.5 34.8, 43.2 38.6
      C40.4 41.8, 36.8 42.6, 34.6 40.8
      C37.8 41.2, 41.6 38.4, 43.4 34.2
      C45.2 30.5, 45.8 30.8, 46.8 30.5 Z"/>
    <!-- gyri (folds) -->
    <g fill="none" stroke="#c5d4f0" stroke-width="0.45" opacity="0.45">
      <path d="M19.5 22 C22 24.5, 24 27, 23.2 31"/>
      <path d="M20.5 26 C23.5 28, 25.5 31, 24.8 34.5"/>
      <path d="M44.5 22 C42 24.5, 40 27, 40.8 31"/>
      <path d="M43.5 26 C40.5 28, 38.5 31, 39.2 34.5"/>
    </g>
  </g>

  <!-- glass sphere -->
  <circle cx="32" cy="32" r="13.8" fill="url(#${g("sphere")})"/>
  <circle cx="32" cy="32" r="13.8" fill="url(#${g("glass")})" opacity="0.55"/>
  <circle cx="32" cy="32" r="13.8" fill="none" stroke="#cfe0ff" stroke-width="0.7" opacity="0.55"/>
  <!-- glass rim highlight -->
  <path d="M22 22 A13.8 13.8 0 0 1 44 20" fill="none" stroke="#ffffff" stroke-width="0.9" opacity="0.35" stroke-linecap="round"/>

  <g class="ul-mesh" clip-path="url(#${g("clip")})" fill="none" stroke="#9eb8e8" stroke-width="0.35" opacity="0.55">
    <ellipse cx="32" cy="32" rx="13.8" ry="4.2"/>
    <ellipse cx="32" cy="27.5" rx="12.2" ry="3.6"/>
    <ellipse cx="32" cy="36.5" rx="12.2" ry="3.6"/>
    <ellipse cx="32" cy="32" rx="4.2" ry="13.8"/>
    <ellipse cx="32" cy="32" rx="9.2" ry="13.8"/>
  </g>

  <!-- glass hex kernel (no face) -->
  <g class="ul-hex">
    <path d="M32 21.4 L40.8 26.5 L40.8 36.5 L32 41.6 L23.2 36.5 L23.2 26.5 Z"
      fill="url(#${g("glass")})" fill-opacity="0.85"
      stroke="#eaf2ff" stroke-width="0.7" opacity="0.95"/>
    <path d="M32 21.4 L40.8 26.5 L32 31.5 Z" fill="#ffffff" fill-opacity="0.22"/>
    <path d="M32 31.5 L40.8 36.5 L32 41.6 Z" fill="#050a14" fill-opacity="0.25"/>
    <path d="M32 24.8 L37.6 28.1 L37.6 34.9 L32 38.2 L26.4 34.9 L26.4 28.1 Z"
      fill="none" stroke="#ffffff" stroke-width="0.35" opacity="0.35"/>
  </g>

  <!-- dense nucleus — the mass -->
  <circle class="ul-core" cx="32" cy="32" r="3.1" fill="#0a1224" opacity="0.55"/>
  <circle class="ul-spark" cx="32" cy="32" r="1.85" fill="#f4f8ff" filter="url(#${g("glow")})"/>
  <circle cx="30.6" cy="30.8" r="0.55" fill="#ffffff" opacity="0.7"/>
</svg>`;
  }

  /* ── gravity particle physics (pulls motes into the core) ───────── */
  function mountPhysics(host) {
    if (REDUCED || host.querySelector(".ul-phys")) return;
    const size = host.getAttribute("data-size") || "md";
    // Skip tiny nav marks — physics reads at md+
    if (size === "sm") return;

    const canvas = document.createElement("canvas");
    canvas.className = "ul-phys";
    canvas.setAttribute("aria-hidden", "true");
    host.appendChild(canvas);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let w = 0, h = 0, dpr = 1, raf = 0, running = true;
    const particles = [];
    const N = size === "hero" || size === "xl" ? 48 : size === "lg" ? 28 : 18;

    function resize() {
      const r = host.getBoundingClientRect();
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = Math.max(1, r.width);
      h = Math.max(1, r.height);
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = w + "px";
      canvas.style.height = h + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function spawn(atEdge) {
      const cx = w / 2, cy = h / 2;
      let x, y;
      if (atEdge) {
        const a = Math.random() * Math.PI * 2;
        const R = Math.max(w, h) * (0.55 + Math.random() * 0.35);
        x = cx + Math.cos(a) * R;
        y = cy + Math.sin(a) * R;
      } else {
        x = Math.random() * w;
        y = Math.random() * h;
      }
      const colors = [
        [207, 224, 255],
        [111, 157, 255],
        [33, 217, 154],
        [255, 138, 52],
        [180, 210, 255],
      ];
      const col = colors[(Math.random() * colors.length) | 0];
      return {
        x, y,
        vx: (Math.random() - 0.5) * 0.15,
        vy: (Math.random() - 0.5) * 0.15,
        r: 0.6 + Math.random() * 1.4,
        life: 0.55 + Math.random() * 0.45,
        col,
        mass: 0.6 + Math.random() * 1.2,
      };
    }

    for (let i = 0; i < N; i++) particles.push(spawn(true));

    function step() {
      if (!running) return;
      const cx = w / 2, cy = h / 2;
      ctx.clearRect(0, 0, w, h);

      // soft gravity disk
      const grd = ctx.createRadialGradient(cx, cy, w * 0.05, cx, cy, w * 0.55);
      grd.addColorStop(0, "rgba(110,157,255,0.08)");
      grd.addColorStop(1, "rgba(110,157,255,0)");
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.ellipse(cx, cy, w * 0.48, h * 0.42, 0, 0, Math.PI * 2);
      ctx.fill();

      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        const dx = cx - p.x;
        const dy = cy - p.y;
        const d = Math.hypot(dx, dy) || 0.001;
        // inverse-square-ish gravity (clamped) — heavy pull
        const g = Math.min(0.22, (38 * p.mass) / (d * d + 80));
        p.vx += (dx / d) * g;
        p.vy += (dy / d) * g;
        // slight tangential swirl (orbit before fall)
        p.vx += (-dy / d) * 0.012;
        p.vy += (dx / d) * 0.012;
        // drag
        p.vx *= 0.978;
        p.vy *= 0.978;
        p.x += p.vx;
        p.y += p.vy;
        p.life -= 0.0025;

        // absorbed by core
        if (d < w * 0.08 || p.life <= 0) {
          particles[i] = spawn(true);
          continue;
        }

        const a = Math.max(0.15, Math.min(0.95, p.life * (0.35 + (1 - d / (w * 0.7)) * 0.65)));
        ctx.beginPath();
        ctx.fillStyle = `rgba(${p.col[0]},${p.col[1]},${p.col[2]},${a})`;
        ctx.shadowColor = `rgba(${p.col[0]},${p.col[1]},${p.col[2]},0.6)`;
        ctx.shadowBlur = 6;
        ctx.arc(p.x, p.y, p.r * (0.7 + (1 - d / (w * 0.8)) * 0.8), 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      }
      raf = requestAnimationFrame(step);
    }

    const ro = new ResizeObserver(() => resize());
    ro.observe(host);
    resize();
    raf = requestAnimationFrame(step);

    host._ulPhysStop = () => {
      running = false;
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }

  function enhance(el) {
    if (!el) return;
    if (el.dataset.uilReady === VER) return;
    // tear down old physics if re-enhancing
    if (el._ulPhysStop) try { el._ulPhysStop(); } catch {}
    const size = el.getAttribute("data-size") || "md";
    el.classList.add("uil-logo");
    el.setAttribute("data-size", size);
    el.setAttribute("role", "img");
    if (!el.getAttribute("aria-label")) el.setAttribute("aria-label", "FIGPSO UIL");
    el.innerHTML = markSvg(UID());
    el.dataset.uilReady = VER;
    mountPhysics(el);
  }

  function mountAll(root) {
    (root || document).querySelectorAll("[data-uil-logo]").forEach(enhance);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => mountAll(), { once: true });
  } else {
    mountAll();
  }

  const mo = new MutationObserver((muts) => {
    for (const m of muts) {
      m.addedNodes.forEach((n) => {
        if (n.nodeType !== 1) return;
        if (n.matches?.("[data-uil-logo]")) enhance(n);
        else if (n.querySelectorAll) mountAll(n);
      });
    }
  });
  mo.observe(document.documentElement, { childList: true, subtree: true });

  window.UilLogo = { enhance, mountAll, markSvg };
})();
