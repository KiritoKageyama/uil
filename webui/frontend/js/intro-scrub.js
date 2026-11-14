/* ════════════════════════════════════════════════════════════════════
   intro-scrub.js — NASA black-hole plunge, SCROLL-SCRUBBED.
   The intro is a tall pinned section. As you scroll through it the video
   scrubs from start to end and the logo / text fade away (--intro-p);
   at the end you're already in the hero. Native scroll — nothing is
   locked. Muted video is scrubbed via currentTime (eased for smoothness).
   NASA SVS 14576 · public domain.
   ════════════════════════════════════════════════════════════════════ */

"use strict";

(function () {
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const VIDEO_SRC = "/assets/video/nasa-bh-plunge-firstperson.mp4";

  /* ── live dual clocks ─────────────────────────────────────────────── */
  const clkLocal = document.getElementById("clkLocal");
  const clkTokyo = document.getElementById("clkTokyo");
  const clkZone = document.getElementById("clkLocalZone");
  if (clkLocal && clkTokyo) {
    try {
      const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || "LOCAL";
      if (clkZone) clkZone.textContent = tz.split("/").pop().replace(/_/g, " ").toUpperCase();
    } catch {}
    const fmt = (o) => new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false, ...o });
    const local = fmt({}), tokyo = fmt({ timeZone: "Asia/Tokyo" });
    const tick = () => { const n = new Date(); clkLocal.textContent = local.format(n); clkTokyo.textContent = tokyo.format(n); };
    tick(); setInterval(tick, 1000);
  }

  /* ── masked poem reveal ───────────────────────────────────────────── */
  const poem = document.getElementById("introPoem");
  if (poem) {
    poem.innerHTML = poem.innerHTML.split(/<br\s*\/?>/i)
      .map((line) => line.trim().split(/\s+/).map((w) => `<span class="w"><i>${w}</i></span>`).join(" "))
      .join("<br>");
    if (REDUCED) poem.classList.add("in"); else setTimeout(() => poem.classList.add("in"), 500);
  }

  const video = document.getElementById("introVideo");
  const sticky = document.getElementById("introSticky");
  const pin = document.getElementById("intro");
  const bar = document.getElementById("introBar");
  const skipBtn = document.getElementById("introSkip");
  const html = document.documentElement;
  if (!video || !sticky || !pin) { html.classList.remove("is-intro"); return; }

  // muted + inline so it can be scrubbed without a user gesture
  video.muted = true; video.defaultMuted = true; video.playsInline = true;
  video.setAttribute("muted", ""); video.setAttribute("playsinline", "");
  video.preload = "auto"; video.controls = false; video.pause();
  if (!video.getAttribute("src")) video.src = VIDEO_SRC;
  try { video.load(); } catch {}

  let duration = 0, ready = false, targetT = 0, done = false;

  function onMeta() {
    duration = Number(video.duration) || 0;
    if (duration > 0.2 && Number.isFinite(duration)) { ready = true; video.classList.add("is-ready"); sticky.classList.add("video-on"); }
  }
  video.addEventListener("loadedmetadata", onMeta);
  video.addEventListener("loadeddata", () => { video.classList.add("is-ready"); sticky.classList.add("video-on"); });
  video.addEventListener("error", () => {
    // never leave a blank hole — keep the logo/text over the fallback wash
    console.warn("[intro] video unavailable, using static intro");
    sticky.classList.add("video-fail", "video-on");
    setProgress(0);
  });
  if (video.readyState >= 1) onMeta();

  /* progress → fade + bar (video seek handled by the eased loop below) */
  function setProgress(p) {
    const x = Math.max(0, Math.min(1, p));
    sticky.style.setProperty("--intro-p", x.toFixed(4));
    if (bar) bar.style.transform = `scaleX(${x})`;
    sticky.classList.toggle("deep", x > 0.5);
    // topnav + particle field come alive once we're basically through
    const intro = x < 0.985;
    html.classList.toggle("is-intro", intro);
    html.classList.toggle("intro-done", !intro);
    // fire once when the intro completes so the interactive starfield mounts
    if (!intro && !done) { done = true; window.dispatchEvent(new Event("uil-intro-done")); }
  }

  /* scroll position within the pin → 0..1 */
  function scrollProgress() {
    const rect = pin.getBoundingClientRect();
    const travel = Math.max(1, pin.offsetHeight - window.innerHeight);
    return Math.min(1, Math.max(0, -rect.top / travel));
  }

  let rafPending = false;
  function onScroll() {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(() => {
      rafPending = false;
      const p = scrollProgress();
      setProgress(p);
      if (ready && duration) targetT = p * (duration - 0.05);
    });
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll, { passive: true });

  /* ease the video's currentTime toward the scroll target — scrubbing a
     muted <video> frame-by-frame is smoother when lerped than hard-set */
  function seekLoop() {
    if (ready && !document.hidden) {
      const cur = video.currentTime || 0;
      const d = targetT - cur;
      if (Math.abs(d) > 0.008) {
        try { video.currentTime = cur + d * (REDUCED ? 1 : 0.18); } catch {}
      }
    }
    requestAnimationFrame(seekLoop);
  }
  requestAnimationFrame(seekLoop);

  skipBtn?.addEventListener("click", (e) => {
    e.preventDefault();
    const hero = document.getElementById("hero") || document.getElementById("top");
    if (hero) window.scrollTo({ top: hero.getBoundingClientRect().top + window.scrollY - 8, behavior: REDUCED ? "auto" : "smooth" });
  });

  setProgress(0);
  onScroll();
  window.__uilIntroScrub = { progress: scrollProgress };
})();
