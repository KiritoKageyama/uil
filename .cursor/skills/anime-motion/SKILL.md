---
name: anime-motion
description: >-
  Anime.js-inspired motion for UIL pitch (/) and Research Console (/app).
  Use when adding or polishing UI animations, page transitions, staggered
  reveals, scroll-driven motion, or interactive micro-interactions. Prefer
  anime.js API patterns (stagger, spring/ease, timelines) while respecting
  prefers-reduced-motion and existing CSS tokens.
---

# Anime.js motion (UIL)

Take inspiration from [anime.js](https://animejs.com/) (v4: `animate`, `stagger`, timelines) for UIL’s marketing site and console — not generic CSS fade soup.

## When to apply

- Pitch site (`webui/frontend/`), console (`webui/desktop/`), shared embeds
- Hero/CTA entrances, chapter reveals, module tiles, KG HUD chrome, button ripples
- Any request: “animate”, “stagger”, “spring”, “motion polish”, “anime.js”

## Principles (anime.js spirit)

1. **Stagger over simultaneous** — siblings enter with `stagger(40–80ms)`; from `first` / `center` / `last` when it clarifies hierarchy.
2. **Ease with character** — prefer `inOutQuad` / `inOutQuint` / `outExpo` / springy overshoots for CTAs; avoid linear except loaders.
3. **Timelines for story** — sequence brand → headline → sub → CTA → stats; don’t dump everything at `t=0`.
4. **Transform + opacity** — animate `translateY`, `scale`, `rotate`, `opacity`; avoid layout thrash (`top`/`height` unless necessary).
5. **Reduced motion** — if `prefers-reduced-motion: reduce`, skip stagger/parallax; show final state immediately.
6. **One job per motion** — entrance OR hover OR scroll accent; don’t stack competing loops on the same element.
7. **Defer to content** — motion supports the AI/brain/space pitch; never obscure copy or the live KG.

## UIL wiring

| Surface | File | Notes |
|---------|------|--------|
| Pitch motion module | `webui/frontend/js/anime-motion.js` | ESM; loads anime.js from CDN; enhances hero + reveals |
| Pitch behaviour | `webui/frontend/js/site.js` | Scroll, stats, modules; keep compatible with anime enhancements |
| Console motion | `webui/desktop/js/motion.js` | CSS stagger fallback; mirror anime easing when touching |

CDN (pitch, ESM):

```js
import { animate, stagger, createTimeline } from "https://cdn.jsdelivr.net/npm/animejs@4.2.2/+esm";
```

Offline / no CDN: keep CSS `.reveal` / `--delay` paths working (already in `site.css` / `motion.js`).

## Recipe snippets

**Hero timeline**

```js
createTimeline({ defaults: { ease: "outExpo" } })
  .add(".hero-kicker", { opacity: [0, 1], y: [12, 0], duration: 600 })
  .add(".hero-title", { opacity: [0, 1], y: [28, 0], duration: 900 }, "-=400")
  .add(".hero-cta .btn", { opacity: [0, 1], y: [16, 0], delay: stagger(70), duration: 650 }, "-=500")
  .add(".hstat", { opacity: [0, 1], y: [10, 0], delay: stagger(55), duration: 500 }, "-=350");
```

**Scroll reveal + stagger children**

```js
animate(el, { opacity: [0, 1], y: [26, 0], filter: ["blur(8px)", "blur(0px)"], duration: 750, ease: "outQuint" });
animate(el.querySelectorAll(".mod-card, .card"), {
  opacity: [0, 1], y: [18, 0], delay: stagger(55, { start: 80 }), duration: 600, ease: "outQuad",
});
```

**Hover micro (console nav / buttons)**

```js
el.addEventListener("pointerenter", () => animate(el, { scale: 1.03, duration: 220, ease: "outQuad" }));
el.addEventListener("pointerleave", () => animate(el, { scale: 1, duration: 280, ease: "outExpo" }));
```

## Anti-patterns

- Infinite bounce on primary CTA
- Purple glow / glassmorphism as a substitute for motion craft
- Animating every list item on every re-render (debounce; unobserve after first reveal)
- Fighting the inertial scroll in `site.js` — don’t add a second scroll library on the pitch

## Quick checklist

- [ ] `prefers-reduced-motion` respected  
- [ ] Stagger used for sibling groups  
- [ ] Easing named (not default linear)  
- [ ] CSS fallback still readable if CDN fails  
- [ ] No motion on KG canvas internals (graph has its own sim)
