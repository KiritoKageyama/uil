/* ════════════════════════════════════════════════════════════════════
   hero-webgl.js — raw WebGL / GLSL neural-mesh hero engine
   Additive-glow neurons + synapse lines + traveling signal pulses, with
   mouse parallax. No three.js: hand-written shaders keep the whole app
   self-contained and offline-safe. Degrades gracefully:
     • prefers-reduced-motion  → one static frame, no animation loop
     • no WebGL context         → adds .no-webgl (CSS shows a static field)
   Public: HeroWebGL.mount(canvas, { density }) -> { destroy() }
   ════════════════════════════════════════════════════════════════════ */

"use strict";

(function () {
  const VERT_POINTS = `
    attribute vec2 a_pos;
    attribute float a_seed;
    attribute float a_size;
    uniform float u_dpr;
    varying float v_seed;
    void main() {
      gl_Position = vec4(a_pos, 0.0, 1.0);
      gl_PointSize = a_size * u_dpr;
      v_seed = a_seed;
    }`;

  const FRAG_POINTS = `
    precision mediump float;
    varying float v_seed;
    uniform vec3 u_cyan;
    uniform vec3 u_violet;
    uniform float u_boost;
    void main() {
      vec2 d = gl_PointCoord - vec2(0.5);
      float r = length(d);
      if (r > 0.5) discard;
      float core = smoothstep(0.5, 0.0, r);
      float halo = smoothstep(0.5, 0.15, r) * 0.6;
      vec3 col = mix(u_cyan, u_violet, v_seed);
      float a = (core + halo) * u_boost;
      gl_FragColor = vec4(col * (0.7 + core * 0.6), a);
    }`;

  const VERT_LINES = `
    attribute vec2 a_pos;
    attribute float a_alpha;
    varying float v_alpha;
    void main() { gl_Position = vec4(a_pos, 0.0, 1.0); v_alpha = a_alpha; }`;

  const FRAG_LINES = `
    precision mediump float;
    varying float v_alpha;
    uniform vec3 u_col;
    void main() { gl_FragColor = vec4(u_col, v_alpha); }`;

  function compile(gl, type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      console.warn("shader compile failed:", gl.getShaderInfoLog(sh));
      return null;
    }
    return sh;
  }

  function program(gl, vsrc, fsrc) {
    const vs = compile(gl, gl.VERTEX_SHADER, vsrc);
    const fs = compile(gl, gl.FRAGMENT_SHADER, fsrc);
    if (!vs || !fs) return null;
    const p = gl.createProgram();
    gl.attachShader(p, vs); gl.attachShader(p, fs);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      console.warn("program link failed:", gl.getProgramInfoLog(p));
      return null;
    }
    return p;
  }

  function hexToRgb(hex, fb) {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec((hex || "").trim());
    if (!m) return fb;
    return [parseInt(m[1], 16) / 255, parseInt(m[2], 16) / 255, parseInt(m[3], 16) / 255];
  }

  function cssColor(name, fb) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return hexToRgb(v, fb);
  }

  function mount(canvas, opts = {}) {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let gl = null;
    try {
      gl = canvas.getContext("webgl", { alpha: true, antialias: true, premultipliedAlpha: false })
        || canvas.getContext("experimental-webgl");
    } catch (e) { gl = null; }

    if (!gl) {
      canvas.classList.add("no-webgl");
      return { destroy() {} };
    }

    const progPoints = program(gl, VERT_POINTS, FRAG_POINTS);
    const progLines = program(gl, VERT_LINES, FRAG_LINES);
    if (!progPoints || !progLines) {
      canvas.classList.add("no-webgl");
      return { destroy() {} };
    }

    let W = 0, H = 0, aspect = 1, dpr = 1;
    let nodes = [], edges = [];
    let posArr, seedArr, sizeArr;
    let lineArr, lineAlphaArr;
    let pulses = [];
    const parallax = { x: 0, y: 0, tx: 0, ty: 0 };
    let raf = 0, t0 = performance.now(), running = true;

    const posBuf = gl.createBuffer();
    const seedBuf = gl.createBuffer();
    const sizeBuf = gl.createBuffer();
    const lineBuf = gl.createBuffer();
    const lineAlphaBuf = gl.createBuffer();
    const pulsePosBuf = gl.createBuffer();
    const pulseSeedBuf = gl.createBuffer();
    const pulseSizeBuf = gl.createBuffer();

    function seed() {
      const area = W * H;
      const density = opts.density || 46000;
      const N = Math.max(48, Math.min(180, Math.floor(area / density)));
      nodes = Array.from({ length: N }, () => {
        const bx = (Math.random() * 2 - 1) * aspect;
        const by = Math.random() * 2 - 1;
        return {
          bx, by, x: bx, y: by,
          seed: Math.random(),
          phase: Math.random() * Math.PI * 2,
          amp: 0.01 + Math.random() * 0.03,
          depth: 0.3 + Math.random() * 0.7,
          spd: 0.15 + Math.random() * 0.4,
        };
      });
      // connect near neighbours
      edges = [];
      const R = 0.42;
      for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
          const dx = (nodes[i].bx - nodes[j].bx) / aspect;
          const dy = nodes[i].by - nodes[j].by;
          const d2 = dx * dx + dy * dy;
          if (d2 < R * R) edges.push({ i, j, d: Math.sqrt(d2) });
        }
      }
      posArr = new Float32Array(N * 2);
      seedArr = new Float32Array(N);
      sizeArr = new Float32Array(N);
      for (let k = 0; k < N; k++) {
        seedArr[k] = nodes[k].seed;
        sizeArr[k] = (2.4 + nodes[k].seed * 5.5);
      }
      lineArr = new Float32Array(edges.length * 4);
      lineAlphaArr = new Float32Array(edges.length * 2);
      gl.bindBuffer(gl.ARRAY_BUFFER, seedBuf);
      gl.bufferData(gl.ARRAY_BUFFER, seedArr, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, sizeBuf);
      gl.bufferData(gl.ARRAY_BUFFER, sizeArr, gl.STATIC_DRAW);
      pulses = edges.length
        ? Array.from({ length: Math.min(26, Math.floor(edges.length / 3)) }, () => ({
            e: (Math.random() * edges.length) | 0,
            t: Math.random(),
            spd: 0.25 + Math.random() * 0.5,
          }))
        : [];
    }

    function resize() {
      const rect = canvas.getBoundingClientRect();
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = rect.width; H = rect.height;
      aspect = W / Math.max(1, H);
      canvas.width = Math.max(1, W * dpr);
      canvas.height = Math.max(1, H * dpr);
      gl.viewport(0, 0, canvas.width, canvas.height);
      seed();
      if (reduced || !running) frame(true);
    }

    function frame(staticOnly) {
      const t = (performance.now() - t0) / 1000;
      parallax.x += (parallax.tx - parallax.x) * 0.05;
      parallax.y += (parallax.ty - parallax.y) * 0.05;

      for (let k = 0; k < nodes.length; k++) {
        const n = nodes[k];
        const dx = staticOnly ? 0 : Math.sin(t * n.spd + n.phase) * n.amp;
        const dy = staticOnly ? 0 : Math.cos(t * n.spd * 0.8 + n.phase) * n.amp;
        n.x = n.bx + dx + parallax.x * n.depth * aspect;
        n.y = n.by + dy + parallax.y * n.depth;
        posArr[k * 2] = n.x;
        posArr[k * 2 + 1] = n.y;
      }

      for (let e = 0; e < edges.length; e++) {
        const A = nodes[edges[e].i], B = nodes[edges[e].j];
        lineArr[e * 4] = A.x; lineArr[e * 4 + 1] = A.y;
        lineArr[e * 4 + 2] = B.x; lineArr[e * 4 + 3] = B.y;
        const a = Math.max(0, (1 - edges[e].d / 0.42)) * 0.16;
        lineAlphaArr[e * 2] = a; lineAlphaArr[e * 2 + 1] = a;
      }

      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // additive glow

      // ── lines ──
      if (edges.length) {
        gl.useProgram(progLines);
        gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
        gl.bufferData(gl.ARRAY_BUFFER, lineArr, gl.DYNAMIC_DRAW);
        const lp = gl.getAttribLocation(progLines, "a_pos");
        gl.enableVertexAttribArray(lp);
        gl.vertexAttribPointer(lp, 2, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, lineAlphaBuf);
        gl.bufferData(gl.ARRAY_BUFFER, lineAlphaArr, gl.DYNAMIC_DRAW);
        const la = gl.getAttribLocation(progLines, "a_alpha");
        gl.enableVertexAttribArray(la);
        gl.vertexAttribPointer(la, 1, gl.FLOAT, false, 0, 0);
        gl.uniform3fv(gl.getUniformLocation(progLines, "u_col"), cssColor("--synapse", [0.22, 0.74, 0.97]));
        gl.drawArrays(gl.LINES, 0, edges.length * 2);
      }

      // ── neurons ──
      gl.useProgram(progPoints);
      gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
      gl.bufferData(gl.ARRAY_BUFFER, posArr, gl.DYNAMIC_DRAW);
      const pp = gl.getAttribLocation(progPoints, "a_pos");
      gl.enableVertexAttribArray(pp);
      gl.vertexAttribPointer(pp, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, seedBuf);
      const ps = gl.getAttribLocation(progPoints, "a_seed");
      gl.enableVertexAttribArray(ps);
      gl.vertexAttribPointer(ps, 1, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, sizeBuf);
      const pz = gl.getAttribLocation(progPoints, "a_size");
      gl.enableVertexAttribArray(pz);
      gl.vertexAttribPointer(pz, 1, gl.FLOAT, false, 0, 0);
      gl.uniform1f(gl.getUniformLocation(progPoints, "u_dpr"), dpr);
      gl.uniform1f(gl.getUniformLocation(progPoints, "u_boost"), 0.85);
      gl.uniform3fv(gl.getUniformLocation(progPoints, "u_cyan"), cssColor("--acc-2", [0.4, 0.9, 0.98]));
      gl.uniform3fv(gl.getUniformLocation(progPoints, "u_violet"), cssColor("--neural", [0.65, 0.55, 0.98]));
      gl.drawArrays(gl.POINTS, 0, nodes.length);

      // ── travelling signal pulses ──
      if (!staticOnly && pulses.length) {
        const pPos = new Float32Array(pulses.length * 2);
        const pSeed = new Float32Array(pulses.length);
        const pSize = new Float32Array(pulses.length);
        for (let i = 0; i < pulses.length; i++) {
          const pl = pulses[i];
          pl.t += pl.spd * 0.016;
          if (pl.t > 1) { pl.t = 0; pl.e = (Math.random() * edges.length) | 0; }
          const ed = edges[pl.e];
          const A = nodes[ed.i], B = nodes[ed.j];
          pPos[i * 2] = A.x + (B.x - A.x) * pl.t;
          pPos[i * 2 + 1] = A.y + (B.y - A.y) * pl.t;
          pSeed[i] = 0.15;
          pSize[i] = 3.5 + Math.sin(pl.t * Math.PI) * 3.0;
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, pulsePosBuf);
        gl.bufferData(gl.ARRAY_BUFFER, pPos, gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(pp);
        gl.vertexAttribPointer(pp, 2, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, pulseSeedBuf);
        gl.bufferData(gl.ARRAY_BUFFER, pSeed, gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(ps);
        gl.vertexAttribPointer(ps, 1, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, pulseSizeBuf);
        gl.bufferData(gl.ARRAY_BUFFER, pSize, gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(pz);
        gl.vertexAttribPointer(pz, 1, gl.FLOAT, false, 0, 0);
        gl.uniform1f(gl.getUniformLocation(progPoints, "u_boost"), 1.5);
        gl.drawArrays(gl.POINTS, 0, pulses.length);
      }
    }

    function loop() {
      if (!running) return;
      if (!document.hidden) frame(false);
      raf = requestAnimationFrame(loop);
    }

    // pointer parallax (skip when reduced-motion)
    function onMove(e) {
      const rect = canvas.getBoundingClientRect();
      parallax.tx = ((e.clientX - rect.left) / rect.width - 0.5) * 0.16;
      parallax.ty = ((e.clientY - rect.top) / rect.height - 0.5) * -0.16;
    }

    gl.clearColor(0, 0, 0, 0);
    const ro = new ResizeObserver(() => resize());
    ro.observe(canvas.parentElement || canvas);
    resize();

    if (reduced) {
      frame(true);
    } else {
      window.addEventListener("pointermove", onMove);
      window.addEventListener("uil-theme-change", () => frame(false));
      loop();
    }

    return {
      destroy() {
        running = false;
        cancelAnimationFrame(raf);
        ro.disconnect();
        window.removeEventListener("pointermove", onMove);
      },
    };
  }

  window.HeroWebGL = { mount };
})();
