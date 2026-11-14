/* ════════════════════════════════════════════════════════════════════
   exploded-engine.js — the UIL "cognition turbine".
   Rebuilt with the real anime.js v4 toolkit (the animejs.com feel):
     • createDraggable  — grab any stage, fling it, it springs back
     • createSpring     — bouncy, physical explode / reassemble
     • stagger          — stages pop apart in sequence, not all at once
     • always-spinning turbine blades (marching stroke-dash)
   Seven stages map the UIL pipeline; a single `--ex` (0→1) lays out the
   nested→exploded geometry, scrubbable by scroll, slider, or toggle.
   Live: real /api telemetry per stage, energy flow, run-lit stage.
   ════════════════════════════════════════════════════════════════════ */

"use strict";

(async function () {
  const rig = document.getElementById("explodedRig");
  if (!rig) return;
  const stages = Array.from(rig.querySelectorAll(".estage"));
  const toggle = document.getElementById("explodeToggle");
  const scrub = document.getElementById("explodeScrub");
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let animate = null, createSpring = null, createDraggable = null, stagger = null;
  try {
    let mod;
    try { mod = await import("/shared/vendor/anime.esm.min.js"); }
    catch { mod = await import("https://cdn.jsdelivr.net/npm/animejs@4.2.2/+esm"); }
    animate = mod.animate; createSpring = mod.createSpring;
    createDraggable = mod.createDraggable; stagger = mod.stagger;
  } catch { animate = null; }

  /* ── wrap each disc+label in a draggable grip ─────────────────────── */
  const grips = [];
  stages.forEach((st) => {
    const grip = document.createElement("div");
    grip.className = "estage-grip";
    while (st.firstChild) grip.appendChild(st.firstChild);
    st.appendChild(grip);
    grips.push(grip);
  });

  /* ── explode scalar --ex, scrubbed by scroll / slider / toggle ────── */
  let current = 0, exploded = false, userLocked = false, raf = 0;
  function apply(v) {
    current = v;
    rig.style.setProperty("--ex", v.toFixed(4));
    if (scrub && document.activeElement !== scrub) scrub.value = String(Math.round(Math.max(0, Math.min(1, v)) * 100));
  }
  apply(0);

  // easeOutBack = mechanical snap w/ overshoot on explode; settle on close
  const c1 = 1.70158, c3 = c1 + 1;
  const outBack = (t) => 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
  const inOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

  function tweenTo(target, dur = 1150) {
    cancelAnimationFrame(raf);
    rig.style.transition = "none";
    if (REDUCED) { apply(target); return; }
    const from = current, t0 = performance.now();
    const ease = target > current ? outBack : inOutCubic;
    (function step(now) {
      const p = Math.min(1, (now - t0) / dur);
      apply(from + (target - from) * ease(p));
      if (p < 1) raf = requestAnimationFrame(step);
    })(t0);
  }

  function popStages(inward) {
    if (REDUCED || !animate) return;
    // anime stagger + spring: stages jolt as they separate / nest
    const spring = createSpring ? createSpring({ stiffness: 130, damping: 11, mass: 1 }) : "outElastic";
    animate(grips, {
      scale: inward ? [1.06, 1] : [0.9, 1],
      opacity: [0.55, 1],
      ease: spring,
      duration: 900,
      delay: stagger(inward ? 40 : 62, { from: inward ? "center" : "first" }),
    });
  }

  function setExploded(on, dur) {
    exploded = on;
    rig.classList.toggle("is-exploded", on);
    tweenTo(on ? 1 : 0, dur);
    popStages(!on);
    if (toggle) { toggle.textContent = on ? "Reassemble" : "Explode"; toggle.setAttribute("aria-pressed", on ? "true" : "false"); }
  }

  toggle?.addEventListener("click", () => { userLocked = true; setExploded(!exploded); });
  scrub?.addEventListener("input", () => {
    userLocked = true; cancelAnimationFrame(raf); rig.style.transition = "none";
    const v = parseInt(scrub.value, 10) / 100;
    apply(v); exploded = v > 0.5; rig.classList.toggle("is-exploded", exploded);
    if (toggle) toggle.textContent = exploded ? "Reassemble" : "Explode";
  });

  if (!REDUCED) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (userLocked) return; if (e.isIntersecting && e.intersectionRatio > 0.45 && !exploded) setExploded(true, 1500); });
    }, { threshold: [0, 0.45, 0.8] });
    io.observe(rig);
  } else { apply(1); rig.classList.add("is-exploded"); }

  /* ── DRAGGABLE stages — grab, fling, spring back (animejs signature) ─ */
  if (animate && createDraggable && !REDUCED) {
    grips.forEach((grip, i) => {
      try {
        createDraggable(grip, {
          releaseStiffness: 140,
          releaseDamping: 13,
          velocityMultiplier: 1.1,
          onGrab: () => { rig.classList.add("is-dragging"); grip.classList.add("grabbed"); rig.classList.add("has-focus"); stages.forEach((o, j) => o.classList.toggle("dim", j !== i)); },
          onRelease: () => { grip.classList.add("flinging"); },
          onSettle: () => { rig.classList.remove("is-dragging"); grip.classList.remove("grabbed", "flinging"); rig.classList.remove("has-focus"); stages.forEach((o) => o.classList.remove("dim")); },
        });
      } catch {}
    });
  }

  /* ── always-spinning turbine blades (marching stroke-dash) ────────── */
  if (animate && !REDUCED) {
    stages.forEach((st) => {
      const i = parseInt(getComputedStyle(st).getPropertyValue("--i"), 10) || 0;
      const blades = st.querySelector(".disc-blades");
      if (!blades) return;
      const burn = st.classList.contains("burn");
      animate(blades, {
        strokeDashoffset: [0, -34],
        loop: true,
        ease: "linear",
        duration: burn ? 900 : 1500 + i * 260, // combustor spins hottest
      });
    });
  }

  /* ── hover isolate (skips while dragging) ─────────────────────────── */
  stages.forEach((st, i) => {
    st.addEventListener("pointerenter", () => { if (rig.classList.contains("is-dragging")) return; rig.classList.add("has-focus"); stages.forEach((o, j) => o.classList.toggle("dim", j !== i)); });
    st.addEventListener("pointerleave", () => { if (rig.classList.contains("is-dragging")) return; rig.classList.remove("has-focus"); stages.forEach((o) => o.classList.remove("dim")); });
  });

  /* ── LIVE telemetry from /api + run-lit stage ─────────────────────── */
  rig.classList.add("is-live");
  const telByStage = {};
  stages.forEach((st) => {
    const i = parseInt(getComputedStyle(st).getPropertyValue("--i"), 10);
    const label = st.querySelector(".elabel");
    if (!label) return;
    const tel = document.createElement("span");
    tel.className = "etel"; tel.textContent = "— · —";
    label.appendChild(tel); telByStage[i] = tel;
  });
  const fmtN = (n) => (n == null ? "—" : Number(n).toLocaleString("en-US"));
  async function jget(p) { try { const r = await fetch(p); return r.ok ? await r.json() : null; } catch { return null; } }
  async function refreshTelemetry() {
    const [persist, net, replay] = await Promise.all([jget("/api/state/persistence"), jget("/api/state/network"), jget("/api/state/replay")]);
    const rec = (k) => (persist?.files?.[k]?.exists ? persist.files[k].record_count : 0);
    const nodes = net?.summary?.nodes ?? (net?.nodes ? Object.keys(net.nodes).length : 0);
    const edges = net?.summary?.edges ?? (net?.edges ? net.edges.length : 0);
    const set = (i, label, v) => { if (telByStage[i]) telByStage[i].innerHTML = `${label} · <b>${fmtN(v)}</b>`; };
    set(0, "neurons", nodes); set(1, "entries", rec("journal")); set(2, "rules", rec("principles"));
    set(3, "hypotheses", rec("causal")); set(4, "principles", rec("principles")); set(5, "synapses", edges);
    set(6, "events", replay?.seq ?? rec("symbolic_formulas"));
  }
  refreshTelemetry();
  async function pollActiveRun() {
    const runs = await jget("/api/runs");
    const active = (runs?.runs || []).find((r) => r.status === "running" || r.status === "paused");
    stages.forEach((s) => s.classList.remove("running"));
    if (active) {
      const idx = (Math.floor(Date.now() / 1400)) % stages.length;
      stages.find((s) => parseInt(getComputedStyle(s).getPropertyValue("--i"), 10) === idx)?.classList.add("running");
      if (!exploded && !userLocked) setExploded(true, 1400);
    }
  }
  pollActiveRun();
  setInterval(() => { pollActiveRun(); if (Math.random() < 0.5) refreshTelemetry(); }, 2500);

  /* ── pointer-reactive 3D tilt (idles with a sway) ─────────────────── */
  if (!REDUCED) {
    const tilt = { yaw: 0, pitch: 0, tyaw: 0, tpitch: 0, over: false };
    const section = document.getElementById("engine") || rig;
    section.addEventListener("pointermove", (e) => {
      if (rig.classList.contains("is-dragging")) return;
      const r = rig.getBoundingClientRect();
      tilt.tyaw = ((e.clientX - (r.left + r.width / 2)) / r.width) * 26;
      tilt.tpitch = -((e.clientY - (r.top + r.height / 2)) / r.height) * 16;
      tilt.over = true;
    });
    section.addEventListener("pointerleave", () => { tilt.over = false; });
    let t0 = performance.now();
    (function idle() {
      const t = (performance.now() - t0) / 1000;
      if (!tilt.over) { tilt.tyaw = Math.sin(t * 0.4) * 4; tilt.tpitch = Math.cos(t * 0.32) * 2; }
      tilt.yaw += (tilt.tyaw - tilt.yaw) * 0.09;
      tilt.pitch += (tilt.tpitch - tilt.pitch) * 0.09;
      rig.style.setProperty("--yaw", tilt.yaw.toFixed(2) + "deg");
      rig.style.setProperty("--pitch", tilt.pitch.toFixed(2) + "deg");
      requestAnimationFrame(idle);
    })();
  }
})();
