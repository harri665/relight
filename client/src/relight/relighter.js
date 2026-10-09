// The viewer's runtime: owns the engine, the lights, the render loop and the viewport canvases.
// React components read its fields directly and re-render on emit() (see useRelighter.js).
// It lives for the whole page load, so leaving the viewer route does not reload the scene.
import { NRPEngine, GLEngine, LightOptimizer, MAX_LIGHTS, SPARE, discPixels } from "../engine/nrp.js";
import { MIN_BAND_ROWS } from "../engine/engine-base.js";
import { hexToRgb, rgbToHex, srgbToLin, fmt } from "./color.js";
import {
  AdaptiveQuality, previewStride, refineStride, benchStride, gpuName, loadProfile, saveProfile, seedCosts, measuredCosts,
  upgradeTier, nextVisitTier, switchNetwork, networkKey, NETWORKS, compactDevice, slowConnection,
} from "./quality.js";

const params = new URLSearchParams(location.search);
// Without ?scene=, the first load picks "cornell" or else the first scene in the index (see resolveScene).
export let SCENE = params.get("scene") || "";
export const BACKEND = params.get("backend");  // "webgpu" or "webgl" forces one
// ?res=N pins an image size exported with export.py --res N, and ?res=native the size the paths were
// traced at. Without it the size follows the GPU (AUTO): a first visit starts at the native size (the
// smallest on phones and slow connections), a larger one is fetched in the background if the GPU has
// room for it, and the next visit starts at the size this one settled on.
export let RES = Number(params.get("res")) || null;
export const AUTO = !params.get("res");
// A scene the viewer picked itself (no ?scene=) also changes to the lighter network of the same scene
// on a GPU too slow to refine the larger one past every 4th pixel (quality.js switchNetwork).
const AUTO_NETWORK = AUTO && !params.get("scene");
export { MAX_LIGHTS };

/** Exports of the same scene (other networks), and the image sizes an export comes in. */
export const familyOf = (e) => e?.family ?? e?.name.replace(/-(hq|lite)$/, "");
export const nativeOf = (e) => e?.size ?? 512;
export const tiersOf = (e) => [...new Set([nativeOf(e), ...(e?.tiers ?? [])])].sort((a, b) => a - b);

// While lights change, a light whose full-resolution evaluation would not fit the network's budget for
// a frame (quality.js) is evaluated on every s-th pixel instead (a preview, upsampled along the
// geometry by the composite). Once nothing has changed for REFINE_DELAY_MS, previews are refined, a
// band of rows a frame within the budget, into the spare slot, which then takes the light's place.
// Fast GPUs stay at full resolution throughout; slow ones stop at the stride they can refine to.
const REFINE_DELAY_MS = 120;
// While the lights rest, a light is timed at the finest stride the GPU can take (benchStride), at
// most BENCH_LIMIT times an engine.
const BENCH_REST_MS = 500, BENCH_LIMIT = 16;
// The budget only follows late frames once the scene has run this long (loading made frames late).
const SETTLE_MS = 1500;
const SAVE_EVERY_MS = 5000;

// Engines own the canvas context, so switching backend or scene reloads the page. Lights, exposure
// and selection are carried over in sessionStorage so both backends can be compared on the same lighting.
const CARRY_KEY = "nrp-carry";

// Starting lights for scenes exported before scene.json listed its own.
const FALLBACK_LIGHTS = [
  { name: "Key", pos: [0.1, 0.8, 0.1], radius: 0.12, color: [1, 0.86, 0.68], intensity: 18 },
  { name: "Fill", pos: [-0.72, 0.15, 0.75], radius: 0.09, color: [0.6, 0.75, 1], intensity: 10 },
];

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
    this.lastChange = 0; this.lastFrame = 0; this.sinceProbe = 0;
    // the slot a resting light is refined into (see refine), and that refinement
    this.spare = SPARE;
    this.refining = null;
    this.quality = new AdaptiveQuality();
    this.gpu = null;
    this.savedAt = 0;
    // scene index, and changing image size, network or backend in place (see adapt, swapEngine)
    this.index = [];
    this.swapping = null;
    this.upgradeFailed = false;
    this.adaptedAt = 0;
    this.notGPU = null;
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

  /**
   * Picks the scene and image size to start with: ?scene= or "cornell" (or the first in the index),
   * and in AUTO mode the size, and for a scene picked here the network, an earlier visit settled on.
   */
  async resolveScene() {
    try { this.index = await (await fetch("/scenes/index.json")).json(); } catch { /* no index: just the scene */ }
    if (!SCENE) SCENE = this.index.some((s) => s.name === "cornell") ? "cornell" : this.index[0]?.name ?? "cornell";
    let entry = this.index.find((e) => e.name === SCENE);
    if (!entry) return;
    const fam = familyOf(entry);
    const net = this.profile?.networks?.[fam];
    if (AUTO_NETWORK && NETWORKS.includes(entry.network) && NETWORKS.includes(net)) {
      const sibling = this.siblings(entry).find((e) => e.network === net);
      if (sibling) { SCENE = sibling.name; entry = sibling; }
    }
    if (!AUTO) return;
    const tiers = tiersOf(entry), native = nativeOf(entry);
    let tier = this.profile?.tiers?.[fam];
    // the largest size is only ever reached by timing the GPU
    if (!tiers.includes(tier)) tier = compactDevice() || slowConnection() ? Math.min(native, tiers[0]) : native;
    RES = tier === native ? null : tier;
  }

  get entry() { return this.index.find((e) => e.name === SCENE); }
  siblings(entry = this.entry) { return this.index.filter((e) => familyOf(e) === familyOf(entry)); }
  /** Image sizes of the current scene, and its native one. */
  get tiers() { return tiersOf(this.entry); }
  get native() { return nativeOf(this.entry); }

  /** WebGPU first, then WebGL2. Returns the first backend that loads the scene. */
  async startEngine(setMsg) {
    await this.resolveScene();
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
        if (Engine === NRPEngine) this.notGPU = `WebGPU couldn't start (${err.message})`;
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
    // What an earlier visit learned on this GPU: the network's budget and costs, the image size and network.
    this.gpu = gpuName();
    this.profile = loadProfile(this.gpu);
    if (BACKEND === "webgl") this.notGPU = "the page was asked for WebGL (?backend=webgl)";
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
    this.adoptEngine(engine);
    this.paint.canvas.width = this.W; this.paint.canvas.height = this.H;
    this.paint.ctx = this.paint.canvas.getContext("2d", { willReadFrequently: true });
    this.gpu ||= this.stats.gpu;
    this.fromProfile = !!this.profile;
    this.quality = new AdaptiveQuality({ budget: this.profile?.budget });
    seedCosts(engine, this.profile);
    this.readyAt = performance.now();
    addEventListener("pagehide", () => this.saveProfile());
    this.status = { phase: "ready", message: "" };
    this.setExposure(this.ui.exposure);
    if (!this.restoreCarried()) this.defaultLights();
    this.hint = HELP_HINT;
    this.resizeOverlay();
    this.emit();
    window.nrp = { relighter: this, engine, state: this.state };  // handy for scripting / debugging
  }

  /** Makes `engine` the one the viewer runs: sizes, GPU name, and falling back to WebGL if its device is lost. */
  adoptEngine(engine) {
    this.engine = engine;
    this.scene = engine.scene;
    this.W = engine.W; this.H = engine.H; this.NP = engine.NP;
    const gpu = engine.adapterInfo.description || engine.adapterInfo.architecture;
    this.stats.gpu = gpu ? `${engine.backend} · ${gpu}` : engine.backend;
    engine.onLost = (info) => {
      this.notGPU = `its WebGPU device was lost (${info.message || info.reason})`;
      this.swapEngine({ Engine: GLEngine });
    };
    if (window.nrp) window.nrp.engine = engine;
  }

  /**
   * Loads `name` at image size `res` (null: native) with `Engine`, on a canvas of its own, and swaps
   * it in for the running engine, keeping the lights (exports of the same scene share their light
   * space). The old engine runs until then. Returns whether it swapped.
   */
  async swapEngine({ name = SCENE, res = RES, Engine = this.engine.constructor } = {}) {
    if (this.swapping) return false;
    this.swapping = { name, res, backend: Engine === NRPEngine ? "WebGPU" : "WebGL2" };
    this.emit();
    const canvas = this.makeCanvas();
    let e = null;
    try {
      e = await Engine.create(canvas);
      await e.load(`/scenes/${name}`, () => {}, res);
      if (this.state.optimizing) throw new Error("the optimizer started");
      const old = this.engine;
      seedCosts(e, measuredCosts(old));
      this.adoptEngine(e);
      SCENE = name; RES = res;
      this.gpuCanvas.replaceWith(canvas);
      this.gpuCanvas = canvas;
      // paint strokes and a target image follow the new size
      const strokes = document.createElement("canvas");
      strokes.width = this.paint.canvas.width; strokes.height = this.paint.canvas.height;
      strokes.getContext("2d").drawImage(this.paint.canvas, 0, 0);
      this.paint.canvas.width = this.W; this.paint.canvas.height = this.H;
      this.paint.ctx.drawImage(strokes, 0, 0, this.W, this.H);
      if (this.targetSource) this.fitTarget();
      e.exposure = 2 ** this.ui.exposure;
      this.refining = null;
      this.spare = SPARE;
      this.onLightsChanged();
      this.resizeOverlay();
      old.dispose();
      return true;
    } catch (err) {
      console.warn(`staying at ${this.W} px (${this.swapping.backend}, ${name} at ${res ?? "native size"}):`, err);
      try { e?.dispose(); } catch { /* never got that far */ }
      this.upgradeFailed = true;
      return false;
    } finally {
      this.swapping = null;
      this.emit();
    }
  }

  /**
   * In AUTO mode, about once a second once the scene has settled: changes to the lighter network of
   * this scene if the GPU can't refine this one (only for a scene the viewer picked), or else fetches
   * a larger image size if the GPU has room for it and the screen shows the image larger than now.
   */
  adapt(now) {
    const { engine, quality, state } = this;
    if (!AUTO || this.swapping || this.upgradeFailed || state.optimizing || state.tab === "compare") return;
    if (now - this.readyAt < 2000 || now - this.adaptedAt < 1000) return;
    this.adaptedAt = now;
    if (AUTO_NETWORK) {
      const net = switchNetwork(engine, quality.budget);
      const sibling = net && this.siblings().find((e) => e.network === net);
      if (sibling) {
        const res = tiersOf(sibling).includes(this.W) && this.W !== nativeOf(sibling) ? this.W : null;
        this.swapEngine({ name: sibling.name, res });
        return;
      }
    }
    const r = this.overlay.getBoundingClientRect();
    const shownPx = Math.max(r.width, r.height) * Math.min(devicePixelRatio || 1, 2);
    const tier = upgradeTier(engine, this.tiers, quality.budget, shownPx);
    if (tier) this.swapEngine({ res: tier === this.native ? null : tier });
  }

  // ---------------------------------------------------------------- lights
  makeLight(o = {}) {
    const { state } = this;
    if (!state.freeSlots.length) return null;
    const r = o.radius ?? 0.12, id = state.nextId++;
    return {
      id, name: o.name || `Light ${id}`,
      pos: o.pos ? [...o.pos] : this.newLightPos(), radius: r,
      color: o.color ? [...o.color] : [1, 1, 1],
      intensity: o.intensity ?? 0.3 / (r * r),
      enabled: o.enabled ?? true, slot: state.freeSlots.pop(), dirty: true, optimize: true,
    };
  }
  /** Where a new light starts: the middle of the scene's random-light region (in front of the camera). */
  newLightPos() {
    const b = this.scene?.random_bbox;
    return b ? b[0].map((a, k) => (a + b[1][k]) / 2) : [0, 0.2, 0.3];
  }
  addLight(o = {}) {
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
    if (this.refining?.light === l) this.refining = null;
    state.freeSlots.push(l.slot);
    state.sel = Math.min(state.sel, state.lights.length - 1);
    this.onLightsChanged();
  }
  setLights(list) {
    const { state } = this;
    this.refining = null;
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
    this.refining = null;
    state.lights.forEach((l) => state.freeSlots.push(l.slot));
    // slots move between lights and the spare as lights are refined, so each gets a free one again
    state.lights = snap.map((l) => ({ ...l, pos: [...l.pos], color: [...l.color], slot: state.freeSlots.pop(), dirty: true }));
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
    const { rmin, rmax } = this.engine;
    // Older Cornell exports have no random_bbox; there the cap keeps lights inside the box.
    const [lo, hi] = this.scene?.random_bbox || [this.engine.lo, this.engine.hi.map((h) => Math.min(h, 0.95))];
    const n = 2 + Math.floor(Math.random() * 3);
    const hues = [[1, 0.8, 0.55], [0.55, 0.7, 1], [1, 0.45, 0.35], [0.5, 1, 0.6], [1, 1, 1], [0.9, 0.5, 1]];
    this.setLights(Array.from({ length: n }, (_, i) => {
      const r = rmin + (rmax - rmin) * (0.15 + 0.55 * Math.random());
      return {
        name: `Light ${i + 1}`,
        pos: [0, 1, 2].map((k) => lo[k] + (hi[k] - lo[k]) * Math.random()),
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
        const surf = this.engine.surfaceAt(p);
        this.flash(surf ? `pixel ${px},${py} · surface (${surf.pos.map((x) => fmt(x)).join(", ")})` : `pixel ${px},${py}`);
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
      const surf = this.engine.surfaceAt(p);
      if (!surf) return;
      const l = this.selected ?? this.addLight({});
      if (!l) return;
      const off = l.radius + 0.06;
      l.pos = surf.pos.map((x, k) => x + surf.normal[k] * off);
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
    const bmp = await createImageBitmap(file);
    this.targetSource = bmp;
    this.fitTarget();
    this.optStatus = `target image loaded (${bmp.width}×${bmp.height}); press Optimize`;
    this.emit();
  }
  /** The target image cropped and scaled to the image size. */
  fitTarget() {
    const { W, H } = this, bmp = this.targetSource;
    const c = document.createElement("canvas"); c.width = W; c.height = H;
    const cx = c.getContext("2d");
    const sc = Math.max(W / bmp.width, H / bmp.height);
    cx.drawImage(bmp, (W - bmp.width * sc) / 2, (H - bmp.height * sc) / 2, bmp.width * sc, bmp.height * sc);
    this.targetImg = cx.getImageData(0, 0, W, H).data;
  }
  clearTarget() {
    this.targetImg = null;
    this.targetSource = null;
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
    this.refining = null;
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
  /** Marks dirty lights for a preview-resolution evaluation that fits the budget between them. */
  planPreview() {
    const dirty = this.state.lights.filter((l) => l.dirty);
    if (!dirty.length) return;
    const budget = this.quality.budget / dirty.length;
    let s = previewStride(this.engine, budget);
    // Costs timed while the lights rested ran on a GPU clocked down (an RTX 3080 took 55 ms for a
    // light it takes 6 for when busy), and preview work at a coarse stride never clocks it up. So
    // every few frames the next finer stride is tried if it is plausibly close to the budget.
    if (s > 1 && ++this.sinceProbe >= 8 && this.engine.evalCost(s / 2) <= 3 * budget) { this.sinceProbe = 0; s /= 2; }
    dirty.forEach((l) => (l.stride = s));
    if (dirty.includes(this.refining?.light)) this.refining = null;
    this.lastChange = performance.now();
  }
  /** Marks every preview-resolution light for re-evaluation at full resolution (all at once). */
  sharpen() {
    let n = 0;
    for (const l of this.state.lights) if (l.stride > 1) { l.stride = 1; l.dirty = true; n++; }
    return n;
  }

  /**
   * Refines a resting light a band of rows a frame, as far as the budget allows (refineStride), into
   * the spare slot while its own slot stays on screen; when the band reaches the bottom the two swap.
   * Returns whether the network ran.
   */
  refine(now) {
    const { engine, quality } = this;
    if (now - this.lastChange < REFINE_DELAY_MS) return false;
    const finest = refineStride(engine, quality.budget);
    let job = this.refining;
    if (job && (job.stride !== finest || !this.state.lights.includes(job.light))) job = null;
    if (!job) {
      const light = this.state.lights.find((l) => l.enabled && (l.stride || 1) > finest);
      if (!light) return false;
      job = this.refining = { light, stride: finest, row: 0 };
    }
    const rows = engine.rows(finest), cols = engine.cols(finest);
    const cost = engine.evalCost(finest), rowCost = cost && cost * engine.share(finest, 0, 1, 0, cols);
    const band = rowCost ? Math.floor(quality.budget / rowCost / 4) * 4 : MIN_BAND_ROWS;
    const to = Math.min(rows, job.row + Math.max(MIN_BAND_ROWS, band));
    engine.evaluate(job.light, this.spare, finest, job.row, to);
    job.row = to;
    if (to >= rows) {
      [job.light.slot, this.spare] = [this.spare, job.light.slot];
      job.light.stride = finest;
      this.refining = null;
      engine.composite(this.state.lights);
    }
    return true;
  }

  /** While the lights rest, times a light at the finest stride the GPU can take (see benchStride). */
  bench(now) {
    const { engine } = this;
    if (this.refining || engine.timing.pending || (engine.benchRuns ?? 0) >= BENCH_LIMIT) return false;
    if (now - this.lastChange < BENCH_REST_MS) return false;
    const stride = benchStride(engine);
    if (!stride) return false;
    engine.benchRuns = (engine.benchRuns ?? 0) + 1;
    engine.evaluate(engine.midLight(), this.spare, stride);
    return true;
  }

  /** Keeps what this visit learned for the next: per scene, the size and network it settled on. */
  saveProfile() {
    const { engine, quality } = this;
    if (!engine?.scene) return;
    const prev = this.profile ?? {}, fam = familyOf(this.entry), net = networkKey(engine);
    // an engine not timed yet keeps the costs measured before
    const costs = Object.keys(engine.timing.byStride).length && !engine.timing.seeded ? measuredCosts(engine) : {};
    this.profile = {
      ...prev, gpu: this.gpu, fps: 30, ...costs, budget: quality.budget,
      tiers: { ...prev.tiers, ...(AUTO && fam && { [fam]: nextVisitTier(engine, this.tiers, quality.budget) }) },
      networks: { ...prev.networks, ...(AUTO_NETWORK && fam && NETWORKS.includes(net) && { [fam]: net }) },
    };
    saveProfile(this.profile);
  }

  // ---------------------------------------------------------------- main loop
  loop(now = performance.now()) {
    const { engine, state, quality } = this;
    const delta = this.lastFrame ? (now - this.lastFrame) / 1000 : 0;
    this.lastFrame = now;
    if (this.ready) {
      engine.pollTiming();
      quality.frame(delta, now - this.readyAt > SETTLE_MS, state.optimizing || !!this.swapping);
      this.adapt(now);
      let worked = false;
      if (state.needsRender) {
        state.needsRender = false;
        this.planPreview();
        worked = state.lights.some((l) => l.dirty);
        engine.render(state.lights);
        this.drawOverlay();
        this.frames++;
      } else if (!state.optimizing) {
        worked = this.refine(now) || this.bench(now);
        if (worked) this.frames++;
      }
      if (worked) quality.worked();
      if (now - this.savedAt > SAVE_EVERY_MS) { this.savedAt = now; this.saveProfile(); }
    }
    if (now - this.lastFps > 500) {
      this.stats.fps = this.frames ? Math.round((this.frames * 1000) / (now - this.lastFps)) : null;
      this.stats.perLight = engine?.evalCost(1) ?? null;
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
