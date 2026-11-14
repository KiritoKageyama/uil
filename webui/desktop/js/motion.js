/* Page motion — staggered reveals, hover lift, nav ripple
   (anime.js-inspired: staggered siblings, outExpo-ish CSS timing) */

"use strict";

function staggerCards(root) {
  const cards = root.querySelectorAll(".card, .mod-tile, .kg-mod, .hstep, .loop-step, .ostat, .mc");
  cards.forEach((el, i) => {
    el.classList.add("reveal");
    /* anime stagger(55) spirit — capped so long grids don’t crawl */
    el.style.setProperty("--delay", `${Math.min(i * 55, 480)}ms`);
  });
}

const _io = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (e.isIntersecting) {
      e.target.classList.add("in");
      _io.unobserve(e.target);
    }
  }
}, { threshold: 0.08, rootMargin: "0px 0px -40px 0px" });

function observeReveals() {
  document.querySelectorAll(".reveal:not(.in)").forEach((el) => _io.observe(el));
}

UIL.registerShow("*", (view) => {
  const panel = $(`#view-${view}`);
  if (!panel) return;
  requestAnimationFrame(() => {
    staggerCards(panel);
    observeReveals();
  });
});

document.addEventListener("uil-view-show", (e) => {
  const panel = $(`#view-${e.detail.view}`);
  if (!panel) return;
  requestAnimationFrame(() => {
    staggerCards(panel);
    observeReveals();
  });
});

document.addEventListener("DOMContentLoaded", () => {
  staggerCards($("#view-overview") || document.body);
  observeReveals();

  $$(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      btn.classList.remove("pop");
      void btn.offsetWidth;
      btn.classList.add("pop");
    });
  });
});
