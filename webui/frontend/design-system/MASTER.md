# FIGPSO UIL — Design System Master

Generated via ui-ux-pro-max (`--design-system --motion 8 --variance 6 --density 3`) and specialized with **HUD / Sci-Fi FUI** + **3D & Hyperrealism** styles.

## Dual view modes

| Mode | Style source | Intent |
|------|--------------|--------|
| **Rendered** (default) | 3D & Hyperrealism + Modern Dark Cinema + Scroll storytelling | Photographic depth, WebGL field, soft glass, parallax layers |
| **Wireframe** | HUD / Sci-Fi FUI | Blueprint grid, 1px technical lines, mono type, scanlines, corner brackets |

Toggle: `html[data-view="rendered|wireframe"]` · persisted in `localStorage.uil-view`.

## Brand (shared)

- Primary: space blue `#3b74ff` / signal `#4aa8ff` / neural purple `#b48bff`
- Avoid: Inter/Roboto/Arial; pure `#00FFFF` neon (a11y); purple-on-white cliché
- Display: **Syne** · Body: **Sora** · Mono: **IBM Plex Mono**
- Motion: anime.js stagger / `outExpo` · respect `prefers-reduced-motion`
- Density: spacious (marketing) · Hero budget: brand + one headline + one sub + CTA + dominant visual

## Pattern

Scroll-triggered storytelling chapters: Intro → What → Cases → Modules → Live graph → HITL → Vision → Honest science → Climax CTA.

## Anti-patterns

Heavy chrome · emoji icons · card soup in hero · forced scroll-jack without reduced-motion · Inter stack · pure black OLED smear `#000`
