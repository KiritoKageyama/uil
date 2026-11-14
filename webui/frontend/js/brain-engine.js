/* ════════════════════════════════════════════════════════════════════
   brain-engine.js — the UIL cognitive brain, dissected (canvas).
   A procedural anatomical wireframe brain — frontal, parietal, occipital,
   temporal lobes + limbic core, cerebellum (striped) and brainstem — that
   pulls apart into its regions as you scroll (a single `--ex` 0→1, driven by
   scroll position, slider, or the Dissect toggle, with per-lobe stagger for
   the anime.js "explode" feel). Each region maps to a UIL subsystem; hover /
   click shows its spec. Deterministic mesh (seeded), DPR-aware, reduced-motion
   aware.
   ════════════════════════════════════════════════════════════════════ */

"use strict";

const SPECS = {
  router:   { tag:"01 · STRATEGY · Prefrontal cortex", title:"Domain Router",
    lead:"The executive. Fingerprints the dataset on five statistical axes, then routes it to the learner that fits — the brain deciding which region should think.",
    bullets:["Five-axis fingerprint: shape, scale, dependence, class structure, size","Routes to sparse_linear, gbm, symbolic, or structured domains","Architectural knobs stay locked — search cannot override them"],
    file:"cognition/domain_router.py" },
  causal:   { tag:"02 · SCIENCE · Frontal lobe", title:"Causal Engine",
    lead:"Reasoning and planning. Correlation is not enough — the scientist designs interventions and only confirms causality when a majority of recent trials agree.",
    bullets:["Hypotheses scheduled by expected information gain","CONFIRMED_CAUSAL needs majority-of-3, not one lucky trial","Derived factors decompose to controllable actions"],
    file:"cognition/causal_intervention.py" },
  library:  { tag:"03 · BELIEF · Parietal lobe", title:"Principle Library",
    lead:"Integration of what is known. Principles carry confidence and climb a four-tier hierarchy — or get demoted and quarantined when they fail.",
    bullets:["Seed knowledge starts humble (confidence 0.5)","Promotion / demotion / quarantine keep the pool honest","Meta-principles merge similar rules across runs"],
    file:"cognition/uil_library.py" },
  symbolic: { tag:"04 · LAW · Occipital lobe", title:"Symbolic Formula",
    lead:"Where thought becomes something you can see. Readable models — formulas, sparse-linear over a symbolic basis, rules — a law you can inspect, not a weight blob.",
    bullets:["Modes: single, boosting, bagging, sparse_linear, multiclass OvR","Basis families: raw · formula · stump · rule (opt-in)","Wins categorically on extrapolation; stays editable"],
    file:"adapters/symbolic_formula_adapter.py" },
  miner:    { tag:"05 · DISCOVERY · Temporal lobe", title:"DT Miner",
    lead:"Pattern recognition. Decision trees mine readable if→then principles from the journal — gated on held-out R² so rules are not memorised noise.",
    bullets:["Within-dataset and cross-dataset rule mining","Cross rules penalised so they do not overclaim","Can synthesize derived features when they truly help"],
    file:"cognition/uil_miner.py" },
  graph:    { tag:"06 · STRUCTURE · Limbic network", title:"Knowledge Graph",
    lead:"The connective web. Principles become neurons; lift-based edges become synapses — leads_to, strengthens, contradicts, depends_on — that you can traverse and edit.",
    bullets:["Lift-over-base-rate edge classification","Graph attention + co-activation bonuses during search","Working graph is pruned; nothing is lost — it archives"],
    file:"cognition/editable_network.py" },
  version:  { tag:"07 · HISTORY · Cerebellum", title:"VersionStore",
    lead:"Coordination through time. Event-sourced lossless history — reconstruct the entire belief state at any sequence number. Lab reports and undos are just replay.",
    bullets:["Append-only events with replayable reconstruct(seq)","Snapshots accelerate; events are never deleted","Human edits are recorded as first-class events"],
    file:"cognition/version_store.py" },
  executor: { tag:"08 · ACTION · Brainstem", title:"Executor",
    lead:"Beliefs that only sit in a library are decoration. The executor evaluates active principles against the live state and drives the next stage — the mind moving the body.",
    bullets:["Matches principle conditions to current run context","Overrides and bonuses flow into search proposals","Shared execution context keeps network + library in sync"],
    file:"cognition/uil_executor.py" },
};

(function () {
  const lab = document.getElementById("brainLab");
  const cv = document.getElementById("brainCanvas");
  if (!lab || !cv) return;
  const ctx = cv.getContext("2d");
  const toggle = document.getElementById("brainToggle");
  const scrub = document.getElementById("brainScrub");
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const W = 600, H = 470, CX = 298, CY = 214;
  const R = [
    { id:"router",  x:432,y:196,rx:50,ry:46,c:"#6ff0c0" },
    { id:"causal",  x:372,y:148,rx:56,ry:46,c:"#ffb46a" },
    { id:"library", x:286,y:138,rx:58,ry:46,c:"#f5c451" },
    { id:"symbolic",x:198,y:186,rx:50,ry:46,c:"#7fd0ff" },
    { id:"miner",   x:360,y:250,rx:56,ry:38,c:"#b39dff" },
    { id:"graph",   x:296,y:206,rx:44,ry:34,c:"#6f9dff" },
    { id:"version", x:222,y:286,rx:54,ry:42,c:"#ff7f9c", cerebellum:true },
    { id:"executor",x:298,y:322,rx:22,ry:64,c:"#34d399", stem:true },
  ];
  let _s = 12345; const rnd = () => { _s = (_s * 1664525 + 1013904223) & 0x7fffffff; return _s / 0x7fffffff; };
  R.forEach((rg) => {
    let dx = rg.x - CX, dy = rg.y - CY, L = Math.hypot(dx, dy) || 1;
    if (rg.stem) { dx = 0; dy = 1; } else { dx /= L; dy /= L; }
    rg.dir = [dx, dy]; rg.spread = rg.stem ? 70 : 132;
    rg.pts = []; rg.edges = []; rg.dots = []; rg.gyri = [];
    if (rg.cerebellum) {
      for (let k = -4; k <= 4; k++) {
        const yy = rg.y + k * 7.5, w = rg.rx * Math.sqrt(Math.max(0, 1 - (k * 7.5 / rg.ry) ** 2));
        rg.gyri.push([[rg.x - w, yy], [rg.x - w * 0.3, yy - 3.5], [rg.x + w * 0.3, yy - 3.5], [rg.x + w, yy]]);
      }
    } else if (rg.stem) {
      for (let k = 0; k < 7; k++) { const yy = rg.y - rg.ry + k * (rg.ry * 2 / 6), w = rg.rx * (1 - k * 0.09); rg.gyri.push([[rg.x - w, yy], [rg.x + w, yy]]); }
    } else {
      const N = Math.round(rg.rx * rg.ry / 70);
      for (let k = 0; k < N; k++) { const a = rnd() * Math.PI * 2, rr = Math.sqrt(rnd()); rg.pts.push([rg.x + Math.cos(a) * rr * rg.rx * 0.92, rg.y + Math.sin(a) * rr * rg.ry * 0.92]); }
      const ring = [], M = 16;
      for (let k = 0; k < M; k++) { const a = k / M * Math.PI * 2, jr = 0.86 + 0.16 * rnd(); ring.push([rg.x + Math.cos(a) * rg.rx * jr, rg.y + Math.sin(a) * rg.ry * jr]); }
      rg.ring = ring; rg.pts = rg.pts.concat(ring);
      rg.pts.forEach((p, pi) => {
        const d = rg.pts.map((q, qi) => [qi, (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2]).filter((e) => e[0] !== pi).sort((a, b) => a[1] - b[1]);
        for (let n = 0; n < 3; n++) { if (d[n] && pi < d[n][0]) rg.edges.push([pi, d[n][0]]); }
      });
      for (let g = 0; g < 3; g++) { const yy = rg.y - rg.ry * 0.4 + g * rg.ry * 0.4, seg = []; for (let s = 0; s <= 6; s++) { const xx = rg.x - rg.rx * 0.7 + s * (rg.rx * 1.4 / 6); seg.push([xx, yy + Math.sin(s * 1.3 + g) * 4]); } rg.gyri.push(seg); }
      for (let k = 0; k < 3; k++) { const p = rg.pts[(rnd() * rg.pts.length) | 0]; rg.dots.push([p[0], p[1], k % 2 ? "#4aa8ff" : "#ff5470"]); }
    }
  });

  let ex = 0, exT = 0, active = "router", auto = true, autoI = 0, tAuto = 0, userLocked = false;
  const el = (id) => document.getElementById(id);
  function showSpec(id) {
    const s = SPECS[id]; if (!s) return;
    if (el("bpTag")) el("bpTag").textContent = s.tag;
    if (el("bpTitle")) { el("bpTitle").textContent = s.title; el("bpTitle").style.color = (R.find((r) => r.id === id) || {}).c || ""; }
    if (el("bpLead")) el("bpLead").textContent = s.lead;
    if (el("bpBullets")) el("bpBullets").innerHTML = s.bullets.map((t) => `<li>${t}</li>`).join("");
    if (el("bpFile")) el("bpFile").textContent = s.file;
  }
  showSpec(active);

  const regCenter = (rg, exi) => [rg.x + rg.dir[0] * rg.spread * exi, rg.y + rg.dir[1] * rg.spread * exi];
  const exiFor = (i) => { const st = i * 0.05; return Math.max(0, Math.min(1, (ex - st) / (1 - 0.35))); };

  let dpr = 1, scaleF = 1, offX = 0, offY = 0;
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = cv.clientWidth || 600, h = cv.clientHeight || 470;
    cv.width = w * dpr; cv.height = h * dpr;
    scaleF = Math.min(w / W, h / H); offX = (w - W * scaleF) / 2; offY = (h - H * scaleF) / 2;
  }
  window.addEventListener("resize", resize); resize();

  const toWorld = (cx, cy) => { const r = cv.getBoundingClientRect(); return [((cx - r.left) - offX) / scaleF, ((cy - r.top) - offY) / scaleF]; };
  function hit(mx, my) { for (const rg of R) { const p = regCenter(rg, exiFor(R.indexOf(rg))); if (((mx - p[0]) / (rg.rx + 8)) ** 2 + ((my - p[1]) / (rg.ry + 8)) ** 2 < 1) return rg.id; } return null; }
  cv.addEventListener("mousemove", (e) => { const w = toWorld(e.clientX, e.clientY); const h = hit(w[0], w[1]); cv.style.cursor = h ? "pointer" : "default"; if (h) { auto = false; if (h !== active) { active = h; showSpec(h); } } });
  cv.addEventListener("mouseleave", () => { auto = true; });
  cv.addEventListener("click", (e) => { const w = toWorld(e.clientX, e.clientY); const h = hit(w[0], w[1]); if (h) { auto = false; active = h; showSpec(h); } });

  /* ── controls ─────────────────────────────────────────────────────── */
  toggle && toggle.addEventListener("click", () => {
    userLocked = true; const on = exT < 0.5; exT = on ? 1 : 0;
    toggle.textContent = on ? "Reassemble" : "Dissect"; toggle.setAttribute("aria-pressed", on ? "true" : "false");
  });
  scrub && scrub.addEventListener("input", () => { userLocked = true; exT = (+scrub.value) / 100; });

  /* ── scroll drives the dissection ─────────────────────────────────── */
  function scrollProgress() {
    const r = lab.getBoundingClientRect();
    const vh = window.innerHeight || document.documentElement.clientHeight || 0;
    if (!vh) return exT;
    const start = vh * 0.82, end = vh * 0.24, denom = (start - end) || 1;
    const p = (start - r.top) / denom;
    return Number.isFinite(p) ? Math.max(0, Math.min(1, p)) : 0;
  }
  let sPending = false;
  window.addEventListener("scroll", () => {
    if (userLocked || sPending) return; sPending = true;
    requestAnimationFrame(() => { sPending = false; exT = scrollProgress(); });
  }, { passive: true });
  if (REDUCED) { ex = exT = 0.55; }
  else exT = scrollProgress();

  /* ── render loop ──────────────────────────────────────────────────── */
  function draw() {
    ex += (exT - ex) * 0.12; if (Math.abs(exT - ex) < 0.001) ex = exT;
    const stage = document.getElementById("brainStage"); if (stage) stage.style.setProperty("--ex", ex.toFixed(3));
    if (scrub && document.activeElement !== scrub) scrub.value = Math.round(ex * 100);
    if (auto && !REDUCED) { tAuto++; if (tAuto > 170) { tAuto = 0; autoI = (autoI + 1) % R.length; active = R[autoI].id; showSpec(active); } }

    const c = ctx; c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, cv.width, cv.height);
    c.save(); c.translate(offX, offY); c.scale(scaleF, scaleF);

    // connectome
    c.lineWidth = 0.8;
    for (let i = 0; i < R.length; i++) for (let j = i + 1; j < R.length; j++) {
      const a = regCenter(R[i], exiFor(i)), b = regCenter(R[j], exiFor(j)), d = Math.hypot(a[0] - b[0], a[1] - b[1]); if (d > 200) continue;
      c.strokeStyle = "rgba(90,140,230," + (0.05 + 0.16 * (1 - d / 200)) + ")"; c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.stroke();
    }
    // core glow
    const cg = c.createRadialGradient(CX, CY, 2, CX, CY, 60);
    cg.addColorStop(0, "rgba(160,210,255," + (0.32 * (1 - ex * 0.7)) + ")"); cg.addColorStop(1, "rgba(90,140,255,0)");
    c.fillStyle = cg; c.beginPath(); c.arc(CX, CY, 60, 0, 7); c.fill();

    const order = R.map((r, i) => i).sort((a, b) => R[a].y - R[b].y);
    for (const i of order) {
      const rg = R[i], exi = exiFor(i), pc = regCenter(rg, exi), cx = pc[0], cy = pc[1];
      const on = rg.id === active, tx = cx - rg.x, ty = cy - rg.y;
      c.save(); c.translate(tx, ty);
      const glow = on ? rg.c : "#6fb0ff";
      c.shadowColor = on ? rg.c : "transparent"; c.shadowBlur = on ? 14 : 0;
      c.fillStyle = "rgba(20,40,80," + (on ? 0.5 : 0.34) + ")"; c.beginPath(); c.ellipse(rg.x, rg.y, rg.rx, rg.ry, 0, 0, 7); c.fill();
      c.strokeStyle = on ? rg.c : "rgba(120,180,255," + (0.4 + 0.25 * exi) + ")"; c.lineWidth = on ? 1 : 0.7;
      c.beginPath(); rg.edges.forEach((e) => { const p = rg.pts[e[0]], q = rg.pts[e[1]]; c.moveTo(p[0], p[1]); c.lineTo(q[0], q[1]); }); c.stroke();
      c.strokeStyle = on ? "rgba(230,245,255,0.7)" : "rgba(160,200,255,0.45)"; c.lineWidth = 1;
      rg.gyri.forEach((seg) => { c.beginPath(); seg.forEach((p, k) => k ? c.lineTo(p[0], p[1]) : c.moveTo(p[0], p[1])); c.stroke(); });
      if (rg.ring) { c.strokeStyle = glow; c.lineWidth = on ? 1.6 : 1; c.globalAlpha = 0.6 + 0.4 * exi; c.beginPath(); rg.ring.forEach((p, k) => k ? c.lineTo(p[0], p[1]) : c.moveTo(p[0], p[1])); c.closePath(); c.stroke(); c.globalAlpha = 1; }
      c.shadowBlur = 0; rg.dots.forEach((d) => { c.fillStyle = d[2]; c.beginPath(); c.arc(d[0], d[1], on ? 2.6 : 2, 0, 7); c.fill(); });
      c.restore();
      if (exi > 0.18) {
        const lx = cx + rg.dir[0] * (rg.rx + 26), ly = cy + rg.dir[1] * (rg.ry + 18);
        c.globalAlpha = Math.min(1, (exi - 0.18) * 2.2);
        c.strokeStyle = on ? rg.c : "rgba(150,190,255,0.5)"; c.lineWidth = on ? 1.1 : 0.7;
        c.beginPath(); c.moveTo(cx + rg.dir[0] * rg.rx, cy + rg.dir[1] * rg.ry); c.lineTo(lx, ly); c.stroke();
        c.fillStyle = on ? "#eaf3ff" : "#b8c8e8"; c.font = (on ? "600 " : "400 ") + "12px 'IBM Plex Mono',monospace";
        c.textAlign = rg.dir[0] < -0.2 ? "end" : (rg.dir[0] > 0.2 ? "start" : "center");
        c.fillText(SPECS[rg.id].title, lx + rg.dir[0] * 4, ly - 2);
        c.fillStyle = on ? rg.c : "#7f95bb"; c.font = "9px 'IBM Plex Mono',monospace";
        c.fillText(SPECS[rg.id].tag.split("·").pop().trim().toUpperCase(), lx + rg.dir[0] * 4, ly + 10);
        c.globalAlpha = 1;
      }
    }
    c.restore();
    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
  window.__uilBrain = { select: (id) => { auto = false; active = id; showSpec(id); }, dissect: (v) => { userLocked = true; exT = v ? 1 : 0; } };
})();
