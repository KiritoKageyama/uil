/* ════════════════════════════════════════════════════════════════════
   uil-brain.js — anime.js-style cognitive brain for UIL's 12 components
   Orbiting lobes · synaptic SVG · springy swarm · draggable nodes
   Inspired by the animejs.com engine / spring / stagger demos.
   ════════════════════════════════════════════════════════════════════ */

"use strict";

const SPECS = {
  router: {
    tag: "01 · STRATEGY", title: "Domain Router",
    lead: "Fingerprints the dataset on five statistical axes, then picks a learner strategy — so UIL does not force one classifier onto every problem.",
    bullets: [
      "Five-axis fingerprint: shape, scale, dependence, class structure, size",
      "Routes to domains such as sparse_linear, gbm, symbolic, structured",
      "Architectural knobs stay locked — search cannot silently override them",
    ],
    file: "cognition/domain_router.py",
  },
  pso: {
    tag: "02 · SEARCH", title: "PSO Search",
    lead: "A particle swarm explores classifier hyperparameters and actions. As the system matures, principle-guided search can detach from pure PSO.",
    bullets: [
      "Swarm particles propose configs each stage",
      "Hybrid search blends principles with exploration",
      "Maturity stages gradually reduce blind PSO reliance",
    ],
    file: "search/pso_search.py",
  },
  journal: {
    tag: "03 · MEMORY", title: "Journal",
    lead: "The lab notebook. Every stage writes state → action → outcome, stress, and regret so later mining has real evidence — not vibes.",
    bullets: [
      "Persistent stage-by-stage record across runs",
      "Feeds the miner, episodic memory, and lab reports",
      "Forgetting decays old noise so the book stays useful",
    ],
    file: "cognition/uil_journal.py",
  },
  stress: {
    tag: "04 · SIGNALS", title: "Stress & Regret",
    lead: "Dual signals that tell the machine when it is failing and how much better a counterfactual action would have been.",
    bullets: [
      "System stress rises when accuracy stalls or collapses",
      "k-fold regret estimates the cost of bad choices honestly",
      "High stress unlocks investigation and formula evolution",
    ],
    file: "cognition/uil_stress.py",
  },
  miner: {
    tag: "05 · DISCOVERY", title: "DT Miner",
    lead: "Decision trees mine readable if→then principles from the journal — gated on held-out R² so rules are not just memorizing training noise.",
    bullets: [
      "Within-dataset and cross-dataset rule mining",
      "Cross rules penalised so they do not overclaim",
      "Can synthesize derived features when they truly help",
    ],
    file: "cognition/uil_miner.py",
  },
  library: {
    tag: "06 · BELIEF", title: "Principle Library",
    lead: "The working set of beliefs. Principles carry confidence and climb a four-tier hierarchy — or get demoted and quarantined when they fail.",
    bullets: [
      "Seed knowledge starts humble (confidence 0.5)",
      "Promotion / demotion / quarantine keep the pool honest",
      "Meta-principles merge similar rules across runs",
    ],
    file: "cognition/uil_library.py",
  },
  graph: {
    tag: "07 · STRUCTURE", title: "Knowledge Graph",
    lead: "Principles become neurons; lift-based edges become synapses — leads_to, strengthens, contradicts, depends_on — that you can traverse and edit.",
    bullets: [
      "Lift-over-base-rate edge classification",
      "Graph attention + co-activation bonuses during search",
      "Working graph is pruned; nothing is lost — it archives",
    ],
    file: "cognition/editable_network.py",
  },
  causal: {
    tag: "08 · SCIENCE", title: "Causal Engine",
    lead: "Correlation is not enough. The scientist designs interventions and only confirms causality when a majority of recent trials agree.",
    bullets: [
      "Hypotheses scheduled by expected information gain",
      "CONFIRMED_CAUSAL needs majority-of-3 — not one lucky trial",
      "Derived factors decompose to controllable actions",
    ],
    file: "cognition/causal_intervention.py",
  },
  executor: {
    tag: "09 · ACTION", title: "Executor",
    lead: "Beliefs that only sit in a library are decoration. The executor evaluates active principles against the live state and steers the next stage.",
    bullets: [
      "Matches principle conditions to current run context",
      "Overrides and bonuses flow into search proposals",
      "Shared execution context keeps network + library in sync",
    ],
    file: "cognition/uil_executor.py",
  },
  symbolic: {
    tag: "10 · LAW", title: "Symbolic Formula",
    lead: "Readable models — formulas, sparse linear over a symbolic basis, rules — so the output is a law you can inspect, not a sealed weight dump.",
    bullets: [
      "Modes: single, boosting, bagging, sparse_linear, multiclass OvR",
      "Basis families: raw · formula · stump · rule (opt-in)",
      "Wins categorically on extrapolation; stays editable",
    ],
    file: "adapters/symbolic_formula_adapter.py",
  },
  version: {
    tag: "11 · HISTORY", title: "VersionStore",
    lead: "Event-sourced lossless history. Reconstruct the entire belief state at any sequence number — lab reports and undos are just replay.",
    bullets: [
      "Append-only events with replayable reconstruct(seq)",
      "Snapshots accelerate; events are never deleted",
      "Human edits are recorded as first-class events",
    ],
    file: "cognition/version_store.py",
  },
  hitl: {
    tag: "12 · OVERRIDE", title: "Human-in-the-loop",
    lead: "Between runs you can accept, edit, or reject any belief. The machine is a scientist — you are the peer reviewer.",
    bullets: [
      "Edits apply to live cells and the event stream",
      "Survives snapshots and reconstruct/replay",
      "Lab reports compile what changed and why",
    ],
    file: "cognition/lab_report.py",
  },
};

const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

async function loadAnime() {
  try {
    return await import("/shared/vendor/anime.esm.min.js");
  } catch {
    try {
      return await import("https://cdn.jsdelivr.net/npm/animejs@4.2.2/+esm");
    } catch {
      return null;
    }
  }
}

function layoutNodes(stage, nodes) {
  const rect = stage.getBoundingClientRect();
  const cx = rect.width / 2;
  const cy = rect.height / 2;
  const rx = Math.min(rect.width * 0.38, 280);
  const ry = Math.min(rect.height * 0.34, 200);
  const n = nodes.length;
  nodes.forEach((el, i) => {
    const a = (i / n) * Math.PI * 2 - Math.PI / 2;
    const x = cx + Math.cos(a) * rx;
    const y = cy + Math.sin(a) * ry;
    el.style.left = `${x}px`;
    el.style.top = `${y}px`;
    el.dataset.hx = String(x);
    el.dataset.hy = String(y);
    el.dataset.angle = String(a);
  });
  return { cx, cy, rx, ry };
}

function drawSynapses(svg, stage, nodes, activeId) {
  const rect = stage.getBoundingClientRect();
  svg.setAttribute("viewBox", `0 0 ${rect.width} ${rect.height}`);
  svg.innerHTML = "";
  const cx = rect.width / 2;
  const cy = rect.height / 2;
  const NS = "http://www.w3.org/2000/svg";

  nodes.forEach((el) => {
    const x = Number(el.dataset.hx);
    const y = Number(el.dataset.hy);
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", cx);
    line.setAttribute("y1", cy);
    line.setAttribute("x2", x);
    line.setAttribute("y2", y);
    line.setAttribute("class", el.dataset.id === activeId ? "ub-link on" : "ub-link");
    svg.appendChild(line);
  });

  // ring connecting neighbors
  for (let i = 0; i < nodes.length; i++) {
    const a = nodes[i];
    const b = nodes[(i + 1) % nodes.length];
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", a.dataset.hx);
    line.setAttribute("y1", a.dataset.hy);
    line.setAttribute("x2", b.dataset.hx);
    line.setAttribute("y2", b.dataset.hy);
    line.setAttribute("class", "ub-ring-link");
    svg.appendChild(line);
  }
}

function showSpec(id, nodes) {
  const spec = SPECS[id];
  if (!spec) return;
  nodes.forEach((n) => n.classList.toggle("is-on", n.dataset.id === id));
  const tag = document.getElementById("ubTag");
  const title = document.getElementById("ubTitle");
  const lead = document.getElementById("ubLead");
  const bullets = document.getElementById("ubBullets");
  const file = document.getElementById("ubFile");
  if (tag) tag.textContent = spec.tag;
  if (title) title.textContent = spec.title;
  if (lead) lead.textContent = spec.lead;
  if (bullets) bullets.innerHTML = spec.bullets.map((t) => `<li>${t}</li>`).join("");
  if (file) file.textContent = spec.file;
}

async function boot() {
  const root = document.getElementById("uilBrain");
  const stage = document.getElementById("ubStage");
  const svg = document.getElementById("ubSynapses");
  const swarm = document.getElementById("ubSwarm");
  const core = document.getElementById("ubCore");
  if (!root || !stage) return;

  const nodes = Array.from(stage.querySelectorAll(".ub-node"));
  let active = nodes[0]?.dataset.id || "router";
  let auto = true;
  let idx = 0;

  const place = () => {
    layoutNodes(stage, nodes);
    drawSynapses(svg, stage, nodes, active);
  };
  place();
  showSpec(active, nodes);
  window.addEventListener("resize", place);

  const anime = await loadAnime();
  const animate = anime?.animate;
  const stagger = anime?.stagger;
  const createTimeline = anime?.createTimeline;
  const createSpring = anime?.createSpring;

  // ── swarm motes (animejs.com-style scatter / converge) ───────────
  const MOTES = REDUCED ? 12 : 42;
  swarm.innerHTML = "";
  const motes = [];
  for (let i = 0; i < MOTES; i++) {
    const m = document.createElement("span");
    m.className = "ub-mote";
    swarm.appendChild(m);
    motes.push(m);
  }

  function moteTargets(mode) {
    const rect = stage.getBoundingClientRect();
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    return motes.map((_, i) => {
      if (mode === "scatter") {
        return {
          x: (Math.random() - 0.5) * rect.width * 0.7,
          y: (Math.random() - 0.5) * rect.height * 0.65,
        };
      }
      // converge toward active node, else core
      const on = nodes.find((n) => n.dataset.id === active) || nodes[0];
      const tx = Number(on.dataset.hx) - cx;
      const ty = Number(on.dataset.hy) - cy;
      const a = (i / MOTES) * Math.PI * 2;
      const r = 18 + (i % 6) * 7;
      return { x: tx + Math.cos(a) * r, y: ty + Math.sin(a) * r };
    });
  }

  let swarmTimer = 0;
  function runSwarm() {
    if (REDUCED || !animate) return;
    clearTimeout(swarmTimer);
    const from = moteTargets("scatter");
    motes.forEach((m, i) => {
      m.style.transform = `translate(-50%, -50%) translate(${from[i].x}px, ${from[i].y}px)`;
    });
    const to = moteTargets("converge");
    animate(motes, {
      translateX: (_, i) => to[i].x,
      translateY: (_, i) => to[i].y,
      opacity: [0.3, 1],
      scale: [0.55, 1],
      duration: 1200,
      delay: stagger ? stagger(16) : 0,
      ease: "inOutQuad",
      onComplete: () => {
        swarmTimer = setTimeout(runSwarm, 420);
      },
    });
  }

  // ── core pulse ───────────────────────────────────────────────────
  if (!REDUCED && animate && core) {
    animate(core.querySelector(".ub-core-glow"), {
      scale: [1, 1.12, 1],
      opacity: [0.55, 0.95, 0.55],
      duration: 2400,
      ease: "inOutSine",
      loop: true,
    });
    animate(core.querySelector(".ub-core-ring.r1"), {
      rotate: 360,
      duration: 14000,
      ease: "linear",
      loop: true,
    });
    animate(core.querySelector(".ub-core-ring.r2"), {
      rotate: -360,
      duration: 22000,
      ease: "linear",
      loop: true,
    });
  }

  // ── select lobe ──────────────────────────────────────────────────
  function select(id, user) {
    if (user) auto = false;
    active = id;
    idx = Math.max(0, nodes.findIndex((n) => n.dataset.id === id));
    showSpec(id, nodes);
    drawSynapses(svg, stage, nodes, active);
    const on = nodes.find((n) => n.dataset.id === id);
    if (on && animate) {
      animate(on, {
        scale: [1, 1.14, 1],
        duration: 520,
        ease: "outExpo",
      });
    }
  }

  nodes.forEach((n) => {
    n.addEventListener("click", () => select(n.dataset.id, true));
    n.addEventListener("pointerenter", () => {
      if (window.matchMedia("(pointer: fine)").matches) select(n.dataset.id, true);
    });
  });

  root.addEventListener("pointerleave", () => { auto = true; });

  if (!REDUCED) {
    setInterval(() => {
      if (!auto) return;
      idx = (idx + 1) % nodes.length;
      select(nodes[idx].dataset.id, false);
    }, 2800);
  }

  // ── springy fling (animejs.com draggable spirit) ─────────────────
  if (!REDUCED && animate) {
    nodes.forEach((n) => {
      let ox = 0, oy = 0, dragging = false;
      n.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        dragging = true;
        auto = false;
        n.setPointerCapture(e.pointerId);
        ox = e.clientX - Number(n.dataset.hx);
        oy = e.clientY - Number(n.dataset.hy);
        n.classList.add("is-grab");
        select(n.dataset.id, true);
      });
      n.addEventListener("pointermove", (e) => {
        if (!dragging) return;
        n.style.left = `${e.clientX - ox}px`;
        n.style.top = `${e.clientY - oy}px`;
      });
      const release = () => {
        if (!dragging) return;
        dragging = false;
        n.classList.remove("is-grab");
        const hx = Number(n.dataset.hx);
        const hy = Number(n.dataset.hy);
        const ease = createSpring
          ? createSpring({ stiffness: 160, damping: 14 })
          : "outElastic(1, 0.55)";
        animate(n, {
          left: `${hx}px`,
          top: `${hy}px`,
          duration: 780,
          ease,
          onComplete: () => drawSynapses(svg, stage, nodes, active),
        });
      };
      n.addEventListener("pointerup", release);
      n.addEventListener("pointercancel", release);
    });
  }

  // entrance stagger
  if (!REDUCED && animate && createTimeline) {
    nodes.forEach((n) => {
      n.style.opacity = "0";
      n.style.transform = "translate(-50%, -50%) scale(0.55)";
    });
    createTimeline({ defaults: { ease: "outExpo" } })
      .add(core, { scale: [0.7, 1], opacity: [0, 1], duration: 900 }, 0)
      .add(nodes, {
        opacity: [0, 1],
        scale: [0.55, 1],
        delay: stagger ? stagger(55) : 0,
        duration: 750,
        onComplete: () => {
          nodes.forEach((n) => {
            n.style.transform = "translate(-50%, -50%)";
          });
        },
      }, 220);
  } else {
    nodes.forEach((n) => {
      n.style.transform = "translate(-50%, -50%)";
    });
  }

  runSwarm();
}

boot();
