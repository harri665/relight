// The viewer's runtime: owns the engine, the lights, the render loop and the viewport canvases.
// React components read its fields directly and re-render on emit() (see useRelighter.js).
// It lives for the whole page load, so leaving the viewer route does not reload the scene.
import { NRPEngine, GLEngine, LightOptimizer, MAX_LIGHTS, discPixels } from "../engine/nrp.js";
import { hexToRgb, rgbToHex, srgbToLin, fmt } from "./color.js";

const params = new URLSearchParams(location.search);
// Without ?scene=, the first load picks "cornell" or else the first scene in the index (see resolveScene).
export let SCENE = params.get("scene") || "";
export const BACKEND = params.get("backend");  // "webgpu" or "webgl" forces one
export const RES = Number(params.get("res")) || null;  // alternative image size, see export.py --res
export { MAX_LIGHTS };

// While lights change, a light whose full-resolution evaluation would not fit the frame budget is
// evaluated on every s-th pixel instead (a preview, upsampled along the geometry by the composite).
// Once nothing has changed for REFINE_MS, previews are re-evaluated at full resolution.
// Fast GPUs stay at full resolution throughout.
const FRAME_BUDGET_MS = 30, REFINE_MS = 150;

// Engines own the canvas context, so switching backend or scene reloads the page. Lights, exposure
// and selection are carried over in sessionStorage so both backends can be compared on the same lighting.
const CARRY_KEY = "nrp-carry";

// Starting lights for scenes exported before scene.json listed its own.
const FALLBACK_LIGHTS = [
  { name: "Key", pos: [0.1, 0.8, 0.1], radius: 0.12, color: [1, 0.86, 0.68], intensity: 18 },
  { name: "Fill", pos: [-0.72, 0.15, 0.75], radius: 0.09, color: [0.6, 0.75, 1], intensity: 10 },
];

async function resolveScene() {
  if (SCENE) return;
  let index = [];
  try { index = await (await fetch("/scenes/index.json")).json(); } catch { /* fall through to cornell */ }
  SCENE = index.some((s) => s.name === "cornell") ? "cornell" : index[0]?.name ?? "cornell";
}

export const HELP_HINT = "drag lights · wheel = depth · shift+wheel = radius · double-click a surface to place";

class Relighter {
  constructor() {
    this.listeners = new Set();
    this.version = 0;
    this.engine = null;
    this.scene = null;
    this.W = 0; this.H = 0; this.NP = 0;

    this.state = {
      lights: [], sel: -1, tab: "lights", nextId: 1,
      freeSlots: [...Array(MAX_LIGHTS).keys()].reverse(),
      drag: null, hover: -1,
      needsRender: true, optimizing: false, stopOpt: false, undo: null,
      refIndex: -1, stash: null, lossHist: [],
    };
    // Values of the panel controls that the runtime reads (brush, optimizer settings, exposure).
    this.ui = {
      exposure: 0,
      brushColor: "#ffd9a0", brushSize: 36, brushAlpha: 0.6, preserve: 0.3,
      iters: 200, frac: 4096,
      optPos: true, optRadius: true, optColor: true, optSelected: false,
    };
    this.status = { phase: "loading", message: "starting…" };
    this.hint = "";
    this.stats = { fps: null, perLight: null, gpu: "" };
    this.optStatus = "";
    this.metric = null;
    this.targetImg = null;

    this.gpuCanvas = this.makeCanvas();
    this.overlay = this.makeCanvas();
    this.overlay.style.cursor = "crosshair";
    this.overlay.style.touchAction = "none";
    this.octx = this.overlay.getContext("2d");
    this.paint = { canvas: document.createElement("canvas"), drawing: false, last: null, cursor: null, erase: false };
    this.bindViewport();

    this.started = false;
    this.raf = 0;
    this.frames = 0; this.lastFps = performance.now();
    this.lastChange = 0; this.sinceProbe = 0;
    this.loop = this.loop.bind(this);
    this.resizeObserver = new ResizeObserver(() => this.resizeOverlay());
  }

  makeCanvas() {
    const c = document.createElement("canvas");
    c.className = "absolute inset-0 h-full w-full";
    return c;
  }

  // ---------------------------------------------------------------- store
  subscribe = (fn) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };
  getVersion = () => this.version;
  emit() {
    this.version++;
    this.listeners.forEach((fn) => fn());
  }
  get ready() { return this.status.phase === "ready"; }

  // ---------------------------------------------------------------- mounting
  /** Puts the canvases into `stage` and runs the render loop; loads the engine on first use. */
  mount(stage) {
    stage.append(this.gpuCanvas, this.overlay);
    this.resizeObserver.observe(this.overlay);
    this.raf = requestAnimationFrame(this.loop);
    if (!this.started) { this.started = true; this.start(); }
  }
  unmount() {
    cancelAnimationFrame(this.raf);
    this.resizeObserver.disconnect();
    this.gpuCanvas.remove();
    this.overlay.remove();
  }

  /** WebGPU first, then WebGL2. Returns the first backend that loads the scene. */
  async startEngine(setMsg) {
    await resolveScene();
    const failed = [];
    for (const [name, Engine] of [["WebGPU", NRPEngine], ["WebGL2", GLEngine]]) {
      if (BACKEND && !name.toLowerCase().startsWith(BACKEND.toLowerCase())) continue;
      let e;
      try {
        setMsg(`starting ${name}…`);
        e = await Engine.create(this.gpuCanvas);
        const sc = await e.load(`/scenes/${SCENE}`, (m) => setMsg(`loading ${m} (${name})…`), RES);
        return [e, sc];
      } catch (err) {
        console.warn(`${name} failed:`, err);
        failed.push(`${name}: ${err.message}`);
        e?.device?.destroy();
        // A canvas keeps the first kind of context it was given, so the next backend gets a fresh one.
        const fresh = this.gpuCanvas.cloneNode(false);
        if (this.gpuCanvas.parentNode) this.gpuCanvas.replaceWith(fresh);
        this.gpuCanvas = fresh;
      }
    }
    throw new Error(failed.join("\n") || `unknown backend "${BACKEND}" (use webgpu or webgl)`);
  }

  async start() {
    const setMsg = (m) => { this.status = { phase: "loading", message: m }; this.emit(); };
    try {
      [this.engine, this.scene] = await this.startEngine(setMsg);
    } catch (e) {
      console.error(e);
      this.status = {
        phase: "error",
        message: `${e.message}\n\nThis demo needs a browser with WebGPU, or WebGL2 with floating-point render targets.`,
      };
      this.emit();
      return;
    }
    const engine = this.engine;
    const gpu = engine.adapterInfo.description || engine.adapterInfo.architecture;
    this.stats.gpu = gpu ? `${engine.backend} · ${gpu}` : engine.backend;
    this.W = engine.W; this.H = engine.H; this.NP = engine.NP;
    this.paint.canvas.width = this.W; this.paint.canvas.height = this.H;
    this.paint.ctx = this.paint.canvas.getContext("2d", { willReadFrequently: true });
    this.status = { phase: "ready", message: "" };
    this.setExposure(this.ui.exposure);
    if (!this.restoreCarried()) this.defaultLights();
    this.hint = HELP_HINT;
    this.resizeOverlay();
    this.emit();
    window.nrp = { relighter: this, engine, state: this.state };  // handy for scripting / debugging
  }

  // ---------------------------------------------------------------- lights
  makeLight(o = {}) {
    const { state } = this;
    if (!state.freeSlots.length) return null;
    const r = o.radius ?? 0.12, id = state.nextId++;
    return {
      id, name: o.name || `Light ${id}`,
      pos: o.pos ? [...o.pos] : [0, 0.3, 0.3], radius: r,
      color: o.color ? [...o.color] : [1, 1, 1],
      intensity: o.intensity ?? 0.3 / (r * r),
      enabled: o.enabled ?? true, slot: state.freeSlots.pop(), dirty: true, optimize: true,
    };
  }
  addLight(o = { pos: [0, 0.2, 0.3] }) {
    const l = this.makeLight(o);
    if (!l) { this.flash(`At most ${MAX_LIGHTS} lights`); return null; }
    this.engine.clampLight(l);
    this.state.lights.push(l);
    this.state.sel = this.state.lights.length - 1;
    this.onLightsChanged();
    return l;
  }
  removeLight(i) {
    const { state } = this;
    const [l] = state.lights.splice(i, 1);
    state.freeSlots.push(l.slot);
    state.sel = Math.min(state.sel, state.lights.length - 1);
    this.onLightsChanged();
  }
  setLights(list) {
    const { state } = this;
    state.lights.forEach((l) => state.freeSlots.push(l.slot));
    state.lights = [];
    list.forEach((o) => { const l = this.makeLight(o); if (l) { this.engine.clampLight(l); state.lights.push(l); } });
    state.sel = state.lights.length ? 0 : -1;
    this.onLightsChanged();
  }
  cloneLights() {
    return this.state.lights.map((l) => ({ ...l, pos: [...l.pos], color: [...l.color] }));
  }
  restoreLights(snap) {
    const { state } = this;
    state.lights.forEach((l) => state.freeSlots.push(l.slot));
    state.freeSlots = state.freeSlots.filter((s) => !snap.some((l) => l.slot === s));
    state.lights = snap.map((l) => ({ ...l, pos: [...l.pos], color: [...l.color], dirty: true }));
    state.sel = Math.min(state.sel, state.lights.length - 1);
    this.onLightsChanged();
  }
  touch(l, geometry = true) {
    if (geometry) { this.engine.clampLight(l); l.dirty = true; }
    this.state.needsRender = true;
    this.emit();
  }
  onLightsChanged() {
    this.state.lights.forEach((l) => (l.dirty = true));
    this.state.needsRender = true;
    this.emit();
  }
  get selected() { return this.state.lights[this.state.sel] ?? null; }

  selectLight(i) {
    this.state.sel = i;
    this.state.needsRender = true;
    this.emit();
  }
  toggleLight(i) {
    const l = this.state.lights[i];
    l.enabled = !l.enabled;
    this.state.needsRender = true;
    this.emit();
  }
  /** Edits the selected light: pos index 0-2, "radius", "logIntensity" or "color" (hex). */
  editSelected(key, value) {
    const l = this.selected;
    if (!l) return;
    if (typeof key === "number") { l.pos[key] = value; this.touch(l); }
    else if (key === "radius") { l.radius = value; this.touch(l); }
    else if (key === "logIntensity") { l.intensity = 10 ** value; this.touch(l, false); }
    else if (key === "color") { l.color = hexToRgb(value); this.touch(l, false); }
  }

  defaultLights() {
    this.setLights(this.scene.default_lights ?? FALLBACK_LIGHTS);
  }
  randomLights() {
    const { lo, hi, rmin, rmax } = this.engine;
    const n = 2 + Math.floor(Math.random() * 3);
    const hues = [[1, 0.8, 0.55], [0.55, 0.7, 1], [1, 0.45, 0.35], [0.5, 1, 0.6], [1, 1, 1], [0.9, 0.5, 1]];
    this.setLights(Array.from({ length: n }, (_, i) => {
      const r = rmin + (rmax - rmin) * (0.15 + 0.55 * Math.random());
      return {
        name: `Light ${i + 1}`,
        pos: [0, 1, 2].map((k) => lo[k] + (Math.min(hi[k], 0.95) - lo[k]) * Math.random()),
        radius: r, color: hues[Math.floor(Math.random() * hues.length)],
        intensity: (0.5 + Math.random()) * (0.25 / (r * r)),
      };
    }));
  }

  setExposure(ev) {
    this.ui.exposure = ev;
    if (this.engine) { this.engine.exposure = 2 ** ev; this.state.needsRender = true; }
    this.emit();
  }
  setUi(key, value) {
    this.ui[key] = value;
    if (key === "brushSize") this.drawOverlay();
    this.emit();
  }

  setTab(tab) {
    this.state.tab = tab;
    if (this.engine) {
      if (tab !== "compare" && this.engine.mode !== 0) this.setViewMode(0);
      this.overlay.style.cursor = tab === "paint" ? "none" : "crosshair";
      this.drawOverlay();
    }
    this.emit();
  }

  flash(msg) { this.hint = msg; this.emit(); }

  // ---------------------------------------------------------------- viewport interaction
  eventToImage(e) {
    const r = this.overlay.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * this.W, y: ((e.clientY - r.top) / r.height) * this.H, sx: r.width / this.W };
  }
  projectLight(l) {
    const p = this.engine.project(l.pos);
    const pr = (l.radius / p.z / (2 * this.engine.cam.tx)) * this.W;
    return { x: p.u * this.W, y: p.v * this.H, z: p.z, r: pr };
  }
  hitLight(pt) {
    let best = -1, bd = Infinity;
    this.state.lights.forEach((l, i) => {
      const p = this.projectLight(l);
      const d = Math.hypot(p.x - pt.x, p.y - pt.y);
      if (d < Math.max(p.r, 9 / pt.sx) && p.z < bd) { best = i; bd = p.z; }
    });
    return best;
  }

  bindViewport() {
    const overlay = this.overlay, state = this.state, paint = this.paint;
    overlay.addEventListener("pointerdown", (e) => {
      if (!this.ready) return;
      const pt = this.eventToImage(e);
      overlay.setPointerCapture(e.pointerId);
      if (state.tab === "paint") {
        paint.drawing = true; paint.last = pt; paint.erase = e.ctrlKey || e.button === 2;
        this.stroke(pt, pt);
        return;
      }
      const i = this.hitLight(pt);
      if (i >= 0) {
        this.selectLight(i);
        const p = this.projectLight(state.lights[i]);
        state.drag = { i, z: p.z, dx: p.x - pt.x, dy: p.y - pt.y };
      }
      this.drawOverlay();
    });
    overlay.addEventListener("pointermove", (e) => {
      if (!this.ready) return;
      const pt = this.eventToImage(e);
      if (state.tab === "paint") {
        paint.cursor = pt;
        if (paint.drawing) { this.stroke(paint.last, pt); paint.last = pt; }
        this.drawOverlay();
        return;
      }
      if (state.drag) {
        const l = state.lights[state.drag.i];
        l.pos = this.engine.unproject((pt.x + state.drag.dx) / this.W, (pt.y + state.drag.dy) / this.H, state.drag.z);
        this.touch(l);
      } else {
        const h = this.hitLight(pt);
        if (h !== state.hover) { state.hover = h; this.drawOverlay(); }
        const px = Math.floor(pt.x), py = Math.floor(pt.y);
        const p = Math.min(this.NP - 1, py * this.W + px);
        const g = this.engine.geom;
        this.flash(g[p * 4 + 3] > 0
          ? `pixel ${px},${py} · surface (${fmt(g[p * 4])}, ${fmt(g[p * 4 + 1])}, ${fmt(g[p * 4 + 2])})`
          : `pixel ${px},${py}`);
      }
    });
    const end = () => { state.drag = null; paint.drawing = false; };
    overlay.addEventListener("pointerup", end);
    overlay.addEventListener("pointercancel", end);
    overlay.addEventListener("pointerleave", () => { paint.cursor = null; this.drawOverlay(); });
    overlay.addEventListener("contextmenu", (e) => e.preventDefault());

    overlay.addEventListener("dblclick", (e) => {
      if (!this.ready || state.tab === "paint") return;
      const pt = this.eventToImage(e);
      const p = Math.floor(pt.y) * this.W + Math.floor(pt.x);
      const g = this.engine.geom, n = this.engine.normal;
      if (!(g[p * 4 + 3] > 0)) return;
      const l = this.selected ?? this.addLight({});
      if (!l) return;
      const off = l.radius + 0.06;
      l.pos = [0, 1, 2].map((k) => g[p * 4 + k] + n[p * 3 + k] * off);
      this.touch(l);
    });

    overlay.addEventListener("wheel", (e) => {
      const l = this.selected;
      if (!this.ready || !l || state.tab === "paint") return;
      e.preventDefault();
      const d = Math.sign(e.deltaY) * Math.min(Math.abs(e.deltaY), 100);
      if (e.shiftKey) { l.radius *= Math.exp(-d * 0.002); this.touch(l); }
      else if (e.altKey) { l.intensity *= Math.exp(-d * 0.003); this.touch(l, false); }
      else {
        const O = this.engine.cam.O;
        const s = 1 + d * 0.0012;
        l.pos = l.pos.map((x, k) => O[k] + (x - O[k]) * s);
        this.touch(l);
      }
    }, { passive: false });
  }

  stroke(a, b) {
    const c = this.paint.ctx, erase = this.paint.erase;
    const { brushSize: size, brushAlpha: alpha, brushColor } = this.ui;
    const steps = Math.max(1, Math.ceil(Math.hypot(b.x - a.x, b.y - a.y) / (size * 0.15)));
    const [r, gg, bb] = hexToRgb(brushColor).map((v) => Math.round(v * 255));
    c.save();
    c.globalCompositeOperation = erase ? "destination-out" : "source-over";
    for (let s = 0; s <= steps; s++) {
      const x = a.x + ((b.x - a.x) * s) / steps, y = a.y + ((b.y - a.y) * s) / steps;
      const g = c.createRadialGradient(x, y, 0, x, y, size / 2);
      g.addColorStop(0, `rgba(${r},${gg},${bb},${alpha * 0.35})`);
      g.addColorStop(0.6, `rgba(${r},${gg},${bb},${alpha * 0.25})`);
      g.addColorStop(1, `rgba(${r},${gg},${bb},0)`);
      c.fillStyle = erase ? "rgba(0,0,0,0.35)" : g;
      c.beginPath(); c.arc(x, y, size / 2, 0, Math.PI * 2); c.fill();
    }
    c.restore();
  }

  // ---------------------------------------------------------------- overlay drawing
  resizeOverlay() {
    const r = this.overlay.getBoundingClientRect(), dpr = devicePixelRatio || 1;
    this.overlay.width = Math.round(r.width * dpr); this.overlay.height = Math.round(r.height * dpr);
    this.drawOverlay();
  }
  drawOverlay() {
    const { engine, octx, overlay, state, paint, W, H } = this;
    if (!engine || !W) return;
    const cw = overlay.width, ch = overlay.height, s = cw / W, dpr = devicePixelRatio || 1;
    octx.clearRect(0, 0, cw, ch);
    if (state.tab === "paint") {
      octx.globalAlpha = 0.85;
      octx.drawImage(paint.canvas, 0, 0, cw, ch);
      octx.globalAlpha = 1;
      if (paint.cursor) {
        octx.strokeStyle = "rgba(255,255,255,.8)"; octx.lineWidth = 1.5;
        octx.beginPath(); octx.arc(paint.cursor.x * s, paint.cursor.y * s, (this.ui.brushSize / 2) * s, 0, Math.PI * 2); octx.stroke();
      }
    }
    if (state.tab === "compare" && engine.mode !== 0) return;
    const floorY = engine.lo[1];
    state.lights.forEach((l, i) => {
      const p = this.projectLight(l);
      if (p.z <= 0) return;
      const sel = i === state.sel, hov = i === state.hover;
      const x = p.x * s, y = p.y * s, r = Math.max(p.r * s, 5);
      // drop line to the floor for depth perception
      const f = engine.project([l.pos[0], floorY, l.pos[2]]);
      octx.setLineDash([3 * s, 4 * s]);
      octx.strokeStyle = "rgba(255,255,255,.28)"; octx.lineWidth = 1;
      octx.beginPath(); octx.moveTo(x, y); octx.lineTo(f.u * W * s, f.v * H * s); octx.stroke();
      octx.beginPath(); octx.ellipse(f.u * W * s, f.v * H * s, 6 * s * 0.6, 2.5 * s * 0.6, 0, 0, Math.PI * 2); octx.stroke();
      octx.setLineDash([]);
      octx.lineWidth = sel ? 2.2 : 1.3;
      octx.strokeStyle = sel ? "#ffb44d" : hov ? "rgba(255,255,255,.9)" : "rgba(255,255,255,.45)";
      octx.beginPath(); octx.arc(x, y, r + 3, 0, Math.PI * 2); octx.stroke();
      octx.fillStyle = rgbToHex(l.color);
      octx.beginPath(); octx.arc(x, y, 3.2 * dpr, 0, Math.PI * 2); octx.fill();
      octx.font = `${11 * dpr}px ui-monospace, monospace`;
      octx.fillStyle = sel ? "#ffb44d" : "rgba(255,255,255,.75)";
      octx.fillText(l.name, x + r + 6, y - r * 0.4);
    });
  }

  // ---------------------------------------------------------------- paint & optimize
  clearPaint() {
    this.paint.ctx?.clearRect(0, 0, this.W, this.H);
    this.drawOverlay();
  }
  async loadTarget(file) {
    const { W, H } = this;
    const bmp = await createImageBitmap(file);
    const c = document.createElement("canvas"); c.width = W; c.height = H;
    const cx = c.getContext("2d");
    const sc = Math.max(W / bmp.width, H / bmp.height);
    cx.drawImage(bmp, (W - bmp.width * sc) / 2, (H - bmp.height * sc) / 2, bmp.width * sc, bmp.height * sc);
    this.targetImg = cx.getImageData(0, 0, W, H).data;
    this.optStatus = `target image loaded (${bmp.width}×${bmp.height}); press Optimize`;
    this.emit();
  }
  clearTarget() {
    this.targetImg = null;
    this.optStatus = "";
    this.emit();
  }
  toggleOptimize() {
    if (this.state.optimizing) this.state.stopOpt = true;
    else this.optimize();
  }
  undoOptimize() {
    if (!this.state.undo) return;
    this.restoreLights(this.state.undo);
    this.state.undo = null;
    this.emit();
  }
  setOptStatus(msg) { this.optStatus = msg; this.emit(); }

  async optimize() {
    const { engine, state, ui, paint, NP, W, H, targetImg } = this;
    const lights = state.lights.filter((l) => l.enabled);
    if (!lights.length) { this.setOptStatus("no enabled lights to optimize"); return; }
    lights.forEach((l) => (l.optimize = !ui.optSelected || l === this.selected));
    // Current rendering (display space, full resolution) is the base of the target.
    this.setViewMode(0);
    this.sharpen();
    engine.render(state.lights);
    const base = await engine.readDisplay();
    const pdata = paint.ctx.getImageData(0, 0, W, H).data;
    const tone = new Float32Array(NP * 3), wpx = new Float32Array(NP);
    const painted = [], rest = [];
    const preserve = ui.preserve;
    for (let p = 0; p < NP; p++) {
      const a = pdata[p * 4 + 3] / 255;
      for (let k = 0; k < 3; k++) {
        const b = targetImg ? targetImg[p * 4 + k] : base[p * 4 + k];
        tone[p * 3 + k] = srgbToLin((b * (1 - a) + pdata[p * 4 + k] * a) / 255);
      }
      if (targetImg) { wpx[p] = 1; (a > 0.02 ? painted : rest).push(p); }
      else if (a > 0.02) { wpx[p] = Math.min(1, a * 2); painted.push(p); }
      else { wpx[p] = preserve; rest.push(p); }
    }
    if (!targetImg && !painted.length) { this.setOptStatus("paint on the image first (or load a target image)"); return; }

    state.undo = this.cloneLights();
    state.optimizing = true; state.stopOpt = false;
    this.emit();
    const iters = ui.iters, K = ui.frac;
    const opt = new LightOptimizer(engine, lights, { pos: ui.optPos, radius: ui.optRadius, color: ui.optColor, lr: 0.05 });
    const idx = new Uint32Array(K), tgt = new Float32Array(K * 4);
    // Painting: the lights' own discs are not part of the target (the artist paints illumination),
    // so pixels showing a light, at its original or current position, are left out of the loss.
    // Target images: discs stay in, and their rims get a dedicated stratum so the analytic
    // direct term constrains radius vs. intensity.
    const baseLights = this.cloneLights().filter((l) => l.enabled);
    const excl = new Uint8Array(NP);
    state.lossHist = [];
    const t0 = performance.now();
    for (let it = 0; it < iters && !state.stopOpt; it++) {
      // Stratified stochastic pixel subset (paper Sec. 5.3, mini-batch over pixels):
      // (1) light-disc rims [target mode], (2) painted pixels, (3) the rest. Weights keep it unbiased.
      excl.fill(0);
      let E = [];
      if (targetImg) {
        E = discPixels(engine, lights, "edge", 2.5, excl).list;
      } else {
        discPixels(engine, baseLights, "disc", 3, excl);
        discPixels(engine, lights, "disc", 3, excl);
      }
      let nP = 0, nR = 0;
      for (const p of painted) nP += excl[p] ? 0 : 1;
      for (const p of rest) nR += excl[p] ? 0 : 1;
      const kE = Math.min(E.length, K >> 2);
      const useP = nP > 0, useR = nR > 0 && (targetImg || preserve > 0);
      const kP = useP ? (useR ? (K - kE) >> 1 : K - kE) : 0;
      const kR = useR ? K - kE - kP : 0;
      let j = 0;
      const put = (p, w) => {
        idx[j] = p;
        tgt[j * 4] = tone[p * 3]; tgt[j * 4 + 1] = tone[p * 3 + 1]; tgt[j * 4 + 2] = tone[p * 3 + 2];
        tgt[j * 4 + 3] = w; j++;
      };
      for (let n = 0; n < kE; n++) { const p = E[(Math.random() * E.length) | 0]; put(p, (wpx[p] * E.length) / kE / NP); }
      for (const [set, k, size] of [[painted, kP, nP], [rest, kR, nR]]) {
        for (let n = 0; n < k && size > 0;) {
          const p = set[(Math.random() * set.length) | 0];
          if (excl[p]) continue;
          // Target image: unbiased full-image mean. Painting: each region is averaged on its own,
          // so the loss is mean(painted) + keep * mean(unpainted), independent of stroke size.
          put(p, targetImg ? (wpx[p] * size) / k / NP : wpx[p] / k); n++;
        }
      }
      while (j < K) put(0, 0);
      const { grads, loss } = await engine.gradStep(lights, idx, tgt);
      opt.step(grads);
      state.lossHist.push(loss);
      if (it % 3 === 0 || it === iters - 1) {
        // Render directly (not via rAF, which background tabs throttle) and yield to the UI.
        this.planPreview();
        engine.render(state.lights);
        this.drawOverlay();
        this.setOptStatus(`iter ${it + 1}/${iters} · loss ${loss.toExponential(3)} · ${fmt((performance.now() - t0) / 1000, 1)} s`);
        await new Promise((r) => setTimeout(r, 0));
      }
    }
    state.optimizing = false;
    this.lastChange = performance.now();  // previews from the run are sharpened once it settles
    state.needsRender = true;
    this.emit();
  }

  // ---------------------------------------------------------------- accuracy
  setViewMode(m) {
    this.engine.mode = m;
    this.state.needsRender = true;
    this.drawOverlay();
    this.emit();
  }
  requestViewMode(m) {
    if (this.state.refIndex < 0) { this.metric = { message: "pick a test light first" }; this.emit(); return; }
    this.setViewMode(m);
  }
  async selectRef(i) {
    const { engine, scene, state, NP } = this;
    const ref = await engine.loadReference(i);
    // Pick an intensity that exposes the reference nicely (references are per unit radiance).
    let m = 0;
    for (let p = 0; p < NP; p++) m += 0.2126 * ref[p * 4] + 0.7152 * ref[p * 4 + 1] + 0.0722 * ref[p * 4 + 2];
    m /= NP;
    const intensity = 0.15 / Math.max(m, 1e-6);
    const L = scene.refs[i].light;
    if (!state.stash) state.stash = this.cloneLights();
    this.setLights([{ name: `Test ${i + 1}`, pos: L.slice(0, 3), radius: L[3], color: [1, 1, 1], intensity }]);
    state.refIndex = i;
    engine.refLight = 0;
    this.metric = { message: "evaluating…" };
    this.emit();
    engine.render(state.lights);
    const hdr = await engine.readHDR();
    const T = (x) => { x = Math.max(0, x); return x / (1 + x); };
    let se = 0, ae = 0, sr = 0;
    for (let p = 0; p < NP; p++) for (let k = 0; k < 3; k++) {
      const a = hdr[p * 4 + k], b = ref[p * 4 + k] * intensity;
      se += (T(a) - T(b)) ** 2; ae += Math.abs(a - b); sr += Math.abs(b);
    }
    this.metric = { psnr: -10 * Math.log10(se / (NP * 3)), mae: (100 * ae) / sr, light: L };
    this.emit();
  }
  restoreMyLights() {
    const { state } = this;
    if (state.stash) this.restoreLights(state.stash);
    state.stash = null; state.refIndex = -1;
    this.metric = null;
    this.setViewMode(0);
  }

  // ---------------------------------------------------------------- adaptive resolution
  previewStride(nDirty) {
    const engine = this.engine;
    if (engine.evalCost(1) == null) return 8;  // nothing measured yet: start with the cheapest preview
    const cost = (s) => nDirty * engine.evalCost(s);
    let s = [1, 2, 4].find((k) => cost(k) <= FRAME_BUDGET_MS) ?? 8;
    // Predictions for finer strides go stale, e.g. when they were measured while the GPU was clocked
    // down, and light preview work never clocks it up. So every few frames, try the next finer stride
    // if it is plausibly close to the budget.
    if (s > 1 && ++this.sinceProbe >= 8 && cost(s / 2) <= 3 * FRAME_BUDGET_MS) { this.sinceProbe = 0; s /= 2; }
    return s;
  }
  /** Marks dirty lights for a preview-resolution evaluation where needed. */
  planPreview() {
    const dirty = this.state.lights.filter((l) => l.dirty);
    if (!dirty.length) return;
    const s = this.previewStride(dirty.length);
    dirty.forEach((l) => (l.stride = s));
    this.lastChange = performance.now();
  }
  /** Marks every preview-resolution light for re-evaluation at full resolution. */
  sharpen() {
    let n = 0;
    for (const l of this.state.lights) if (l.stride > 1) { l.stride = 1; l.dirty = true; n++; }
    return n;
  }

  // ---------------------------------------------------------------- main loop
  loop() {
    const { engine, state } = this;
    const now = performance.now();
    if (this.ready) {
      if (state.needsRender) {
        state.needsRender = false;
        this.planPreview();
        engine.render(state.lights);
        this.drawOverlay();
        this.frames++;
      } else if (!state.optimizing && now - this.lastChange > REFINE_MS && this.sharpen()) {
        engine.render(state.lights);
        this.frames++;
      }
    }
    if (now - this.lastFps > 500) {
      this.stats.fps = this.frames ? Math.round((this.frames * 1000) / (now - this.lastFps)) : null;
      if (engine?.timing.perLight) this.stats.perLight = engine.timing.perLight;
      this.frames = 0; this.lastFps = now;
      this.emit();
    }
    this.raf = requestAnimationFrame(this.loop);
  }

  // ---------------------------------------------------------------- backend / scene switching
  reloadWith(changes) {
    if (this.ready) {
      try {
        sessionStorage.setItem(CARRY_KEY, JSON.stringify({
          scene: SCENE, family: this.sceneFamily(), sel: this.state.sel, exposure: this.ui.exposure,
          lights: this.state.lights.map(({ name, pos, radius, color, intensity, enabled }) => ({ name, pos, radius, color, intensity, enabled })),
        }));
      } catch { /* storage unavailable: switch without carrying state */ }
    }
    const q = new URLSearchParams(location.search);
    for (const [k, v] of Object.entries(changes)) (v ? q.set(k, v) : q.delete(k));
    location.search = q.toString();
  }
  sceneFamily() {
    const { scene } = this;
    return scene.source_scene ?? JSON.stringify([scene.camera.to_world, scene.light_bbox]);
  }
  restoreCarried() {
    let c = null;
    try { c = JSON.parse(sessionStorage.getItem(CARRY_KEY)); sessionStorage.removeItem(CARRY_KEY); } catch { return false; }
    // Lights only mean the same thing in exports of the same scene (e.g. another network or size).
    if (!c?.lights?.length || c.family !== this.sceneFamily()) return false;
    this.setLights(c.lights);
    this.state.sel = Math.min(c.sel ?? 0, this.state.lights.length - 1);
    this.setExposure(c.exposure ?? 0);
    return true;
  }
}

let instance = null;
/** The page's single viewer runtime. */
export function getRelighter() {
  instance ??= new Relighter();
  return instance;
}
