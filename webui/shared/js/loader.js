/* ════════════════════════════════════════════════════════════════════
   loader.js — logo intro / loading screen.
   A full-screen splash where the Orbital Core assembles (orbits draw in,
   core ignites, particles converge, wordmark rises), holds a minimum
   beat, then dissolves into the app. Self-contained CSS + inline SVG.
   Shared by the marketing site and the console.
   ════════════════════════════════════════════════════════════════════ */

"use strict";

(function () {
  if (document.getElementById("uilLoader")) return;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const css = `
    #uilLoader { position: fixed; inset: 0; z-index: 100001; display: grid; place-items: center;
      background:
        radial-gradient(700px 500px at 50% 42%, rgba(59,116,255,0.10), transparent 60%),
        var(--bg, #04050b);
      transition: opacity .6s var(--ease-out, ease), visibility .6s; }
    #uilLoader.out { opacity: 0; visibility: hidden; }
    .ldr-wrap { display: flex; flex-direction: column; align-items: center; gap: 26px; }
    .ldr-mark { width: 120px; height: 120px; }
    .ldr-mark .uil-logo { width: 120px; height: 120px; --uil-logo-size: 120px;
      filter: drop-shadow(0 0 18px rgba(59,116,255,0.55)); }
    .ldr-word { display: flex; align-items: baseline; gap: 8px; opacity: 0;
      animation: ldrRise .7s var(--ease-out, ease) .55s forwards; }
    @keyframes ldrRise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
    .ldr-word b { font: 700 22px var(--font-display, var(--font), sans-serif); letter-spacing: .5px; color: var(--text, #eef2fb); }
    .ldr-word b i { font-style: normal; color: var(--acc-2, #6f9dff); }
    .ldr-word span { font: 500 10px var(--mono, monospace); letter-spacing: 3px; text-transform: uppercase; color: var(--text-3, #647089); }
    .ldr-bar { width: 190px; height: 3px; border-radius: 3px; background: var(--surface-3, #223); overflow: hidden; opacity: 0;
      animation: ldrRise .5s ease .75s forwards; }
    .ldr-bar i { display: block; height: 100%; width: 30%; border-radius: 3px;
      background: linear-gradient(90deg, var(--acc, #3b74ff), var(--neural, #21d99a));
      animation: ldrSweep 1.1s var(--ease-inout, ease) infinite; }
    @keyframes ldrSweep { 0% { transform: translateX(-120%); width: 40%; } 60% { width: 65%; } 100% { transform: translateX(320%); width: 30%; } }
    .ldr-status { font: 500 10px var(--mono, monospace); letter-spacing: 2px; text-transform: uppercase;
      color: var(--text-3, #647089); opacity: 0; animation: ldrRise .5s ease .85s forwards; }
  `;
  const style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  // Ensure shared logo stylesheet is present for the animated mark
  if (!document.querySelector('link[href*="logo.css"]')) {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "/shared/css/logo.css";
    document.head.appendChild(link);
  }

  const el = document.createElement("div");
  el.id = "uilLoader";
  el.innerHTML = `
    <div class="ldr-wrap">
      <div class="ldr-mark"><span data-uil-logo data-size="xl"></span></div>
      <div class="ldr-word"><b>FIGPSO <i>UIL</i></b><span>Universal Intelligence Layer</span></div>
      <div class="ldr-bar"><i></i></div>
      <div class="ldr-status" id="ldrStatus">initializing cognitive core</div>
    </div>`;

  function mount() {
    (document.body || document.documentElement).appendChild(el);
    document.documentElement.style.overflow = "hidden";
    // Enhance logo if logo.js already loaded; else inject SVG via UilLogo later
    if (window.UilLogo) window.UilLogo.mountAll(el);
    else {
      const s = document.createElement("script");
      s.src = "/shared/js/logo.js";
      s.onload = () => window.UilLogo?.mountAll(el);
      document.head.appendChild(s);
    }
  }
  if (document.body) mount();
  else document.addEventListener("DOMContentLoaded", mount, { once: true });

  const START = performance.now();
  const MIN_MS = reduced ? 300 : 1700;
  const HARD_CAP = 6500;
  let done = false;

  function dismiss() {
    if (done) return; done = true;
    const wait = Math.max(0, MIN_MS - (performance.now() - START));
    setTimeout(() => {
      const st = document.getElementById("ldrStatus"); if (st) st.textContent = "core online";
      el.classList.add("out");
      document.documentElement.style.overflow = "";
      setTimeout(() => el.remove(), 650);
    }, wait);
  }

  if (document.readyState === "complete") setTimeout(dismiss, 60);
  else window.addEventListener("load", () => setTimeout(dismiss, 120), { once: true });
  setTimeout(dismiss, HARD_CAP); // never trap the user
})();
