/* anime-motion.js — anime.js v4 motion for the pitch site
   Intro scroll gate · smoother reveals · UIL cognition engine (Engine-inspired) */

"use strict";

const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const STAGES = ["PSO", "Journal", "Miner", "Library", "Executor"];

async function boot() {
  let animate, stagger, createTimeline, engine, utils;
  try {
    let mod;
    try {
      // vendored locally so the app works offline / self-contained
      mod = await import("/shared/vendor/anime.esm.min.js");
    } catch {
      // CDN fallback if the vendor file is ever missing
      mod = await import("https://cdn.jsdelivr.net/npm/animejs@4.2.2/+esm");
    }
    animate = mod.animate;
    stagger = mod.stagger;
    createTimeline = mod.createTimeline;
    engine = mod.engine;
    utils = mod.utils;
  } catch {
    bootFallback();
    return;
  }

  if (engine?.defaults) {
    engine.defaults.ease = "outExpo";
    engine.defaults.duration = 900;
  }

  document.documentElement.classList.add("anime-on");
  // Intro gate / video owned by intro-scrub.js — do not toggle is-intro here

  /* ── intro chrome (video owned by intro-scrub.js) ───────────────── */
  const sticky = document.getElementById("introSticky");

  /* intro entrance — logo + copy */
  if (!REDUCED && sticky) {
    createTimeline({ defaults: { ease: "outExpo" } })
      .add(".intro-logo-wrap", { opacity: [0, 1], scale: [0.72, 1], duration: 1200 }, 0)
      .add(".intro-kicker", { opacity: [0, 1], y: [16, 0], duration: 850 }, 220)
      .add(".intro-title", { opacity: [0, 1], y: [40, 0], duration: 1100 }, 320)
      .add(".intro-poem", { opacity: [0, 1], y: [18, 0], duration: 900 }, 520)
      .add(".intro-foot", { opacity: [0, 1], y: [12, 0], duration: 700 }, 700);
  }

  /* ── hero (after intro) ─────────────────────────────────────────── */
  const hero = document.querySelector("#hero .hero-inner");
  if (hero && !REDUCED) {
    const heroIO = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        heroIO.unobserve(e.target);
        e.target.classList.add("in");
        createTimeline({ defaults: { ease: "outExpo" } })
          .add(".hero-kicker", { opacity: [0, 1], y: [18, 0], duration: 900 }, 0)
          .add(".hero-title", { opacity: [0, 1], y: [36, 0], duration: 1100 }, 120)
          .add(".hero-sub", { opacity: [0, 1], y: [22, 0], duration: 950 }, 280)
          .add(".hero-cta .btn", { opacity: [0, 1], y: [20, 0], delay: stagger(90), duration: 850 }, 420)
          .add(".hstat", { opacity: [0, 1], y: [14, 0], delay: stagger(70), duration: 750 }, 580);
      });
    }, { threshold: 0.28 });
    heroIO.observe(hero);
  } else if (hero) {
    hero.classList.add("in");
  }

  /* ── smooth scroll reveals ──────────────────────────────────────── */
  if (!REDUCED) {
    const revealIO = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          const el = entry.target;
          revealIO.unobserve(el);
          el.classList.add("in");
          animate(el, {
            opacity: [0, 1],
            y: [28, 0],
            duration: 900,
            ease: "outQuint",
          });
          const kids = el.querySelectorAll(
            ".mod-card, .case, .vision-card, .hitl-step, .film, .loop-node, .pillar"
          );
          if (kids.length) {
            animate(kids, {
              opacity: [0, 1],
              y: [22, 0],
              delay: stagger(70, { start: 120 }),
              duration: 900,
              ease: "outExpo",
            });
          }
          const words = el.querySelectorAll(".w i");
          if (words.length) {
            animate(words, {
              y: ["115%", "0%"],
              delay: stagger(42),
              duration: 920,
              ease: "outExpo",
            });
          }
          if (el.id === "pillarTransparent" || el.closest?.("#pillarTransparent")) {
            document.getElementById("pillarTransparent")?.classList.add("is-on");
          }
          if (el.classList.contains("pillar") && el.id === "pillarTransparent") {
            el.classList.add("is-on");
          }
        });
      },
      { threshold: 0.16, rootMargin: "0px 0px -8% 0px" }
    );

    document.querySelectorAll(".reveal").forEach((el) => {
      if (el.closest("#hero .hero-inner")) return;
      revealIO.observe(el);
    });

    const glassIO = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add("is-on");
            glassIO.unobserve(e.target);
          }
        });
      },
      { threshold: 0.4 }
    );
    const pt = document.getElementById("pillarTransparent");
    if (pt) glassIO.observe(pt);
  } else {
    document.querySelectorAll(".reveal, #pillarTransparent").forEach((el) => {
      el.classList.add("in", "is-on");
    });
  }

  /* ── button micro-interactions ──────────────────────────────────── */
  if (!REDUCED) {
    document.querySelectorAll(".btn.primary, .btn.ghost").forEach((btn) => {
      btn.addEventListener("pointerenter", () => {
        animate(btn, { scale: 1.03, duration: 280, ease: "outQuad" });
      });
      btn.addEventListener("pointerleave", () => {
        animate(btn, { scale: 1, duration: 420, ease: "outExpo" });
      });
    });
  }

  window.addEventListener("uil-view-mode", () => {
    const pulse = document.querySelector(".view-toggle");
    if (!pulse || REDUCED) return;
    animate(pulse, { scale: [1, 1.05, 1], duration: 480, ease: "outExpo" });
  });

  /* ── UIL cognition engine (anime.js Engine spirit) ──────────────── */
  bootCognitionEngine({ animate, stagger, createTimeline, engine, utils });
}

function bootFallback() {
  document.querySelectorAll(".reveal").forEach((el) => el.classList.add("in"));
  document.getElementById("pillarTransparent")?.classList.add("is-on");
}

function bootCognitionEngine({ animate, createTimeline, engine }) {
  const dotsRoot = document.getElementById("engineDots");
  const ring = document.getElementById("engineRing");
  const ring2 = document.getElementById("engineRing2");
  const stageEl = document.getElementById("engineStage");
  if (!dotsRoot || !ring) return;

  const N = 28;
  dotsRoot.innerHTML = "";
  const dots = [];
  for (let i = 0; i < N; i++) {
    const d = document.createElement("span");
    d.className = "engine-dot";
    dotsRoot.appendChild(d);
    dots.push(d);
  }

  let paused = false;
  let turbo = false;
  let stageIdx = 0;
  let swarmAnim = null;
  let ringAnim = null;
  let meterLoop = null;

  const meters = {
    speed: document.getElementById("meterSpeed"),
    swarm: document.getElementById("meterSwarm"),
    stress: document.getElementById("meterStress"),
    fps: document.getElementById("meterFps"),
  };
  const vals = {
    speed: document.getElementById("valSpeed"),
    swarm: document.getElementById("valSwarm"),
    stress: document.getElementById("valStress"),
    fps: document.getElementById("valFps"),
  };

  function setMeter(key, pct, label) {
    if (meters[key]) meters[key].style.width = `${Math.max(4, Math.min(100, pct))}%`;
    if (vals[key] && label != null) vals[key].textContent = label;
  }

  function scatter() {
    const R = turbo ? 28 : 58;
    return dots.map(() => ({
      x: (Math.random() - 0.5) * R * 2,
      y: (Math.random() - 0.5) * R * 2,
    }));
  }

  function converge() {
    return dots.map((_, i) => {
      const a = (i / N) * Math.PI * 2;
      const r = turbo ? 18 + (i % 5) * 3 : 26 + (i % 7) * 4;
      return { x: Math.cos(a) * r, y: Math.sin(a) * r };
    });
  }

  function runSwarmCycle() {
    if (REDUCED || paused) return;
    const from = scatter();
    const to = converge();
    dots.forEach((d, i) => {
      d.style.transform = `translate(-50%, -50%) translate(${from[i].x}px, ${from[i].y}px)`;
    });
    swarmAnim = animate(dots, {
      translateX: (_, i) => to[i].x,
      translateY: (_, i) => to[i].y,
      opacity: [0.4, 1],
      duration: turbo ? 700 : 1400,
      delay: stagger(turbo ? 12 : 28),
      ease: "inOutQuad",
      onComplete: () => {
        if (paused) return;
        stageIdx = (stageIdx + 1) % STAGES.length;
        if (stageEl) {
          stageEl.textContent = STAGES[stageIdx];
          animate(stageEl, {
            opacity: [0, 1],
            translateY: [8, 0],
            duration: 420,
            ease: "outExpo",
          });
        }
        setTimeout(runSwarmCycle, turbo ? 280 : 620);
      },
    });
  }

  function spinRings() {
    if (REDUCED) return;
    ringAnim = createTimeline({ defaults: { ease: "linear", loop: true } })
      .add(ring, { rotate: 360, duration: turbo ? 1800 : 4200 }, 0);
    if (ring2) {
      animate(ring2, {
        rotate: -360,
        duration: turbo ? 2800 : 6400,
        ease: "linear",
        loop: true,
      });
    }
  }

  function tickMeters() {
    if (paused) return;
    const spd = turbo ? 1.75 : 1;
    if (engine) engine.speed = spd;
    setMeter("speed", spd * 55, `${spd.toFixed(2)}×`);
    setMeter("swarm", 40 + Math.random() * 50, String(48 + ((Math.random() * 24) | 0)));
    setMeter("stress", 25 + Math.random() * 45, (0.25 + Math.random() * 0.4).toFixed(2));
    setMeter("fps", turbo ? 92 : 78, turbo ? "90" : "60");
  }

  if (!REDUCED) {
    spinRings();
    runSwarmCycle();
    meterLoop = setInterval(tickMeters, 900);
    tickMeters();
  } else {
    setMeter("speed", 55, "1.00×");
    setMeter("swarm", 70, "64");
    setMeter("stress", 42, "0.42");
    setMeter("fps", 78, "60");
    dots.forEach((d, i) => {
      const a = (i / N) * Math.PI * 2;
      d.style.transform = `translate(${Math.cos(a) * 30}px, ${Math.sin(a) * 30}px)`;
    });
  }

  const btnPause = document.getElementById("enginePause");
  const btnBoost = document.getElementById("engineBoost");

  btnPause?.addEventListener("click", () => {
    paused = !paused;
    btnPause.setAttribute("aria-pressed", paused ? "true" : "false");
    btnPause.textContent = paused ? "Resume" : "Pause";
    if (engine) {
      if (paused) engine.pause();
      else engine.resume();
    }
    if (!paused && !REDUCED) runSwarmCycle();
  });

  btnBoost?.addEventListener("click", () => {
    turbo = !turbo;
    btnBoost.setAttribute("aria-pressed", turbo ? "true" : "false");
    if (engine) engine.speed = turbo ? 1.75 : 1;
    tickMeters();
    if (!paused && !REDUCED) {
      try { swarmAnim?.pause?.(); } catch {}
      runSwarmCycle();
    }
  });
}

boot();
