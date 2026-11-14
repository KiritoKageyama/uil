/* ════════════════════════════════════════════════════════════════════
   intro-engine.js — the atmospheric first screen.
   A spring-physics "reactor" (anime.js, like the animejs.com hero engine)
   whose pistons fire around a rotating core, plus live dual clocks, an
   rpm count-up, and a masked poem reveal. Degrades gracefully with no
   anime.js and respects reduced-motion.
   ════════════════════════════════════════════════════════════════════ */

"use strict";

(async function () {
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ── live dual clocks ──────────────────────────────────────────────
  const clkLocal = document.getElementById("clkLocal");
  const clkTokyo = document.getElementById("clkTokyo");
  const clkZone = document.getElementById("clkLocalZone");
  if (clkLocal && clkTokyo) {
    try {
      const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || "LOCAL";
      if (clkZone) clkZone.textContent = tz.split("/").pop().replace(/_/g, " ").toUpperCase();
    } catch {}
    const fmt = (opts) => new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false, ...opts });
    const local = fmt({}), tokyo = fmt({ timeZone: "Asia/Tokyo" });
    const tick = () => {
      const now = new Date();
      clkLocal.textContent = local.format(now);
      clkTokyo.textContent = tokyo.format(now);
    };
    tick(); setInterval(tick, 1000);
  }

  // ── masked poem reveal (word cascade) ─────────────────────────────
  const poem = document.getElementById("introPoem");
  if (poem) {
    const lines = poem.innerHTML.split(/<br\s*\/?>/i);
    poem.innerHTML = lines.map((line) =>
      line.trim().split(/\s+/).map((w) => `<span class="w"><i>${w}</i></span>`).join(" ")
    ).join("<br>");
    if (REDUCED) poem.classList.add("in");
    else setTimeout(() => poem.classList.add("in"), 700);
  }

  const engine = document.getElementById("introEngine");
  if (!engine) return;
  const pistonsRoot = document.getElementById("iePistons");
  const spokesRoot = document.getElementById("ieSpokes");
  const NS = "http://www.w3.org/2000/svg";
  const ROD_COLORS = ["var(--acc)", "var(--c-orange)", "var(--c-green)", "var(--c-red)"];
  const N = 8;

  // build spokes + spring pistons
  for (let i = 0; i < N; i++) {
    const a = (i / N) * 360;
    const sp = document.createElementNS(NS, "line");
    sp.setAttribute("x1", 0); sp.setAttribute("y1", 0);
    sp.setAttribute("x2", 0); sp.setAttribute("y2", -60);
    sp.setAttribute("transform", `rotate(${a})`);
    spokesRoot?.appendChild(sp);

    const g = document.createElementNS(NS, "g");
    g.setAttribute("transform", `rotate(${a})`);
    const inner = document.createElementNS(NS, "g");
    inner.className.baseVal = "ie-piston";
    const rod = document.createElementNS(NS, "line");
    rod.setAttribute("class", "ie-rod");
    rod.setAttribute("x1", 0); rod.setAttribute("y1", -60);
    rod.setAttribute("x2", 0); rod.setAttribute("y2", -84);
    rod.setAttribute("stroke", ROD_COLORS[i % ROD_COLORS.length]);
    const head = document.createElementNS(NS, "circle");
    head.setAttribute("class", "ie-head");
    head.setAttribute("cx", 0); head.setAttribute("cy", -86); head.setAttribute("r", 4.5);
    head.setAttribute("fill", ROD_COLORS[i % ROD_COLORS.length]);
    inner.appendChild(rod); inner.appendChild(head);
    g.appendChild(inner);
    pistonsRoot?.appendChild(g);
  }
  const pistons = Array.from(pistonsRoot?.querySelectorAll(".ie-piston") || []);

  // rpm readout
  const rpmEl = document.getElementById("ieRpm");

  let animate = null, stagger = null, createSpring = null;
  try {
    let mod;
    try { mod = await import("/shared/vendor/anime.esm.min.js"); }
    catch { mod = await import("https://cdn.jsdelivr.net/npm/animejs@4.2.2/+esm"); }
    animate = mod.animate; stagger = mod.stagger; createSpring = mod.createSpring;
  } catch { animate = null; }

  if (REDUCED || !animate) {
    if (rpmEl) rpmEl.textContent = "2 400";
    return;
  }

  // pistons fire on a spring, staggered around the ring — an idling engine
  const spring = createSpring ? createSpring({ stiffness: 90, damping: 9, mass: 1 }) : "outElastic";
  animate(pistons, {
    translateY: [0, -13],
    loop: true,
    alternate: true,
    ease: spring,
    duration: 900,
    delay: stagger(110, { from: "first" }),
  });

  // core spark breathes
  animate(".intro-engine .ie-core", { scale: [1, 1.14], opacity: [0.9, 1], loop: true, alternate: true, ease: "inOutSine", duration: 1400 });
  animate(".intro-engine .ie-spark", { scale: [0.7, 1.5], opacity: [1, 0.4], loop: true, alternate: true, ease: "inOutQuad", duration: 900 });

  // rev up: rpm count + a subtle overall scale kick on load
  if (rpmEl) {
    const state = { v: 0 };
    animate(state, { v: 2400, duration: 2200, ease: "outExpo",
      onUpdate: () => { rpmEl.textContent = Math.round(state.v).toLocaleString("en-US").replace(/,/g, " "); },
      onComplete: () => {
        // idle flutter
        animate(state, { v: 2320, duration: 1600, loop: true, alternate: true, ease: "inOutSine",
          onUpdate: () => { rpmEl.textContent = Math.round(state.v).toLocaleString("en-US").replace(/,/g, " "); } });
      },
    });
  }
})();
