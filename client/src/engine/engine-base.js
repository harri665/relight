// Backend-independent parts of the runtime: scene loading, camera maths, light normalisation and
// the cost model the runtime schedules evaluations with.
export const MAX_LIGHTS = 8;
// Output slots: one a light, and a spare. A resting light is refined into the spare while its own
// slot stays on screen, and the two swap when the refinement is done (see Relighter.refine).
export const SLOTS = MAX_LIGHTS + 1;
export const SPARE = MAX_LIGHTS;
// Below this many rows a band's fixed cost outweighs its work (many times over on WebGL).
export const MIN_BAND_ROWS = 16;
// Thin bands make the GPU look slow: 16 rows at 768 px read ~16 ms for a light that takes ~8. Only
// evaluations of at least this share of a light are timed.
const MIN_TIMED_SHARE = 0.25;
// ms of unpacking between yields to the page while a scene loads
const SLICE_MS = 6;

/**
 * How each engine runs the network best on this GPU (its kernel shape, tuned at load) is kept in
 * localStorage under `key`, so a returning visitor skips the timing. `?kernel=...` overrides it, and
 * `?kernel=tune` times the shapes again. Storage can be missing or blocked; then it just tunes.
 */
export function chooseKernel(key) {
  const want = new URLSearchParams(location.search).get("kernel");
  if (want === "tune") return { name: null, forced: false };
  if (want) return { name: want, forced: true };
  try { return { name: localStorage.getItem(key), forced: false }; } catch { return { name: null, forced: false }; }
}
export function rememberKernel(key, name) {
  try { localStorage.setItem(key, name); } catch { /* not kept; it is timed again next visit */ }
}

const yieldChannel = new MessageChannel(), yieldWaiters = [];
yieldChannel.port1.onmessage = () => yieldWaiters.shift()();
/** Yields to the event loop for one task (unlike setTimeout, not clamped or throttled). */
export const yieldTask = () => new Promise((r) => { yieldWaiters.push(r); yieldChannel.port2.postMessage(0); });

export const f16tab = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    t[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return t;
})();

/** float32 -> float16 bits, rounded to nearest even */
export function toHalf(values) {
  const out = new Uint16Array(values.length);
  const f = new Float32Array(1), bits = new Uint32Array(f.buffer);
  for (let k = 0; k < values.length; k++) {
    f[0] = values[k];
    const x = bits[0], sign = (x >>> 16) & 0x8000, e = ((x >>> 23) & 0xff) - 112;
    let m = x & 0x7fffff;
    if (e <= 0) out[k] = e < -10 ? sign : sign | ((m | 0x800000) >> (14 - e));
    else if (e >= 31) out[k] = sign | 0x7c00;
    else {
      const rest = m & 0x1fff;
      m >>= 13;
      let h = sign | (e << 10) | m;
      if (rest > 0x1000 || (rest === 0x1000 && m & 1)) h += 1;
      out[k] = h;
    }
  }
  return out;
}

export function typed(buf, entry) {
  const n = entry.shape.reduce((a, b) => a * b, 1);
  if (entry.dtype === "float32") return new Float32Array(buf, entry.offset, n);
  const h = new Uint16Array(buf, entry.offset, n);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = f16tab[h[i]];
  return out;
}

/** An entry's values as float16 bits, whatever it was stored as. */
function halves(buf, entry) {
  const n = entry.shape.reduce((a, b) => a * b, 1);
  return entry.dtype === "float16" ? new Uint16Array(buf, entry.offset, n) : toHalf(new Float32Array(buf, entry.offset, n));
}

/**
 * Undoes nrp/pixels.py's planes (pixel format 2): per channel, 16-bit values stored as differences
 * from the value to their left, low bytes then high bytes. Writes channel k of pixel p to
 * out[at(p, k)], yielding to the page every SLICE_MS so a large scene doesn't stall it.
 */
async function unpackPlanes(bytes, W, H, channels, at, out) {
  const NP = W * H;
  let slice = performance.now();
  for (let k = 0; k < channels; k++) {
    const lo = k * 2 * NP, hi = lo + NP;
    for (let y = 0; y < H; y++) {
      let v = 0;
      for (let p = y * W, end = p + W; p < end; p++) {
        v = (v + (bytes[lo + p] | (bytes[hi + p] << 8))) & 0xffff;
        out[at(p, k)] = v;
      }
      if ((y & 31) === 31 && performance.now() - slice > SLICE_MS) {
        await yieldTask();
        slice = performance.now();
      }
    }
  }
  return out;
}

/** Format-1 positions ([NP, 4] f32: position, camera distance) as format 2's 16-bit values. */
function quantisePositions(geom, NP) {
  let lo = Infinity, hi = -Infinity;
  for (let p = 0; p < NP; p++) {
    if (!(geom[p * 4 + 3] > 0)) continue;
    for (let k = 0; k < 3; k++) { lo = Math.min(lo, geom[p * 4 + k]); hi = Math.max(hi, geom[p * 4 + k]); }
  }
  if (!(hi > lo)) { lo = -1; hi = 1; }
  const pad = (hi - lo) * 1e-4;
  lo -= pad; hi += pad;
  const step = (hi - lo) / 65534, pos = new Uint16Array(NP * 4);
  for (let p = 0; p < NP; p++) {
    if (!(geom[p * 4 + 3] > 0)) continue;
    for (let k = 0; k < 3; k++) pos[p * 4 + k] = Math.round((geom[p * 4 + k] - lo) / step) + 1;
  }
  return { pos, range: [lo, hi] };
}

/**
 * Both engines evaluate a light either at every pixel (stride 1) or, as a fast preview, at every
 * s-th pixel in x and y. A light at stride s is stored compactly (ceil(W/s) x ceil(H/s) items,
 * row-major) in its output slot, and the composite upsamples it along the geometry. evaluate() can
 * fill any band of rows and columns of those items, so a light can be refined a band a frame.
 *
 * The per-pixel data stays as the 16-bit values it ships as, and the GPU turns it into the
 * network's inputs and the geometry: aux is the network's per-pixel features as halves, in planes
 * of 4 channels ([group][pixel][4]); pos is each pixel's surface position as 16-bit values over
 * pixels.pos.range from 1, 0 where the pixel sees nothing ([pixel][4]).
 */
export class EngineBase {
  /**
   * Fetches and unpacks the scene; returns the pieces the backend uploads. `res` picks an
   * alternative image size exported with `export.py --res N` (scene-N.json / pixels-N.bin).
   */
  async fetchScene(base, onProgress, res = null) {
    const suffix = res ? `-${res}` : "";
    const fetchBin = async (f) => {
      const r = await fetch(`${base}/${f}`);
      if (!r.ok) throw new Error(`failed to load ${f}`);
      return r.arrayBuffer();
    };
    onProgress("scene description");
    const sr = await fetch(`${base}/scene${suffix}.json`);
    if (!sr.ok) throw new Error(`failed to load scene${suffix}.json`);
    const scene = await sr.json();
    this.scene = scene;
    onProgress("network weights");
    // A smaller network can share a larger one's pixels (pixels.file), as they don't depend on it.
    const [model, pixels] = await Promise.all([fetchBin("model.bin"), fetchBin(scene.pixels.file ?? `pixels${suffix}.bin`)]);
    this.refsBuf = null;
    // Reference renders exist only at the resolution the paths were traced at.
    this.refsPromise = scene.refs.length
      ? fetchBin("refs.bin").then((b) => (this.refsBuf = b)).catch(() => null)
      : Promise.resolve(null);

    const W = scene.width, H = scene.height, NP = W * H;
    this.W = W; this.H = H; this.NP = NP;
    this.canvas.width = W; this.canvas.height = H;

    let gridLen = 0;
    const gridOff = scene.model.grids.map((g) => { const o = gridLen; gridLen += g.shape[0] * g.shape[1] * g.shape[2]; return o; });
    const grid = new Uint16Array(gridLen);
    scene.model.grids.forEach((g, i) => grid.set(halves(model, g), gridOff[i]));
    const layers = scene.model.layers.map((l) => ({ w: typed(model, l.weight), b: typed(model, l.bias), shape: l.weight.shape }));
    this.paramCount = grid.length + layers.reduce((a, l) => a + l.w.length + l.b.length, 0);
    this.modelBytes = model.byteLength;

    const M = scene.camera.to_world;
    const col = (j) => [M[0][j], M[1][j], M[2][j]];
    const tx = Math.tan((scene.camera.fov * Math.PI) / 360);
    this.cam = { X: col(0), Y: col(1), Z: col(2), O: col(3), tx, ty: (tx * H) / W };

    onProgress("pixel buffers");
    const auxDim = scene.network.aux_dim ?? 7, AG = Math.ceil(auxDim / 4);
    const planar = (p, k) => ((k >> 2) * NP + p) * 4 + (k & 3);
    let aux, pos, range;
    if (scene.pixels.format === 2) {
      const px = scene.pixels, plane = (e) => new Uint8Array(pixels, e.offset, e.bytes);
      aux = await unpackPlanes(plane(px.aux), W, H, px.aux.channels, planar, new Uint16Array(NP * 4 * AG));
      pos = await unpackPlanes(plane(px.pos), W, H, 3, (p, k) => p * 4 + k, new Uint16Array(NP * 4));
      range = px.pos.range;
    } else {
      // Format 1 (unpacked): the same data in the same form; its normals are the aux features'.
      const a = halves(pixels, scene.pixels.aux);
      aux = new Uint16Array(NP * 4 * AG);
      for (let p = 0; p < NP; p++) for (let k = 0; k < auxDim; k++) aux[planar(p, k)] = a[p * auxDim + k];
      ({ pos, range } = quantisePositions(typed(pixels, scene.pixels.geom), NP));
    }
    this.aux = aux; this.pos = pos; this.auxGroups = AG;
    this.posRange = range;
    this.posStep = (range[1] - range[0]) / 65534;
    this.lo = scene.light_bbox[0]; this.hi = scene.light_bbox[1];
    [this.rmin, this.rmax] = scene.radius_range;

    this.exposure = 1;
    this.mode = 0;
    this.refLight = 0;
    this.timing = { byStride: {}, samples: 0, pending: null, seeded: false };
    return { scene, grid, gridOff, layers, aux, pos };
  }

  /** The surface pixel p sees: world position, distance from the camera and normal; null where it sees nothing. */
  surfaceAt(p) {
    const q = this.pos, o = p * 4;
    if (!q[o]) return null;
    const { O } = this.cam;
    const pos = [0, 1, 2].map((k) => this.posRange[0] + (q[o + k] - 1) * this.posStep);
    const normal = [3, 4, 5].map((j) => f16tab[this.aux[((j >> 2) * this.NP + p) * 4 + (j & 3)]]);
    return { pos, dist: Math.hypot(pos[0] - O[0], pos[1] - O[1], pos[2] - O[2]), normal };
  }

  // ---------------------------------------------------------------- cost model
  /** Item rows / columns of a light evaluated at stride s. */
  rows(s = 1) { return Math.ceil(this.H / s); }
  cols(s = 1) { return Math.ceil(this.W / s); }
  /** Share of a whole light at stride s that rows [r0, r1) x columns [c0, c1) are. */
  share(s, r0, r1, c0, c1) { return ((r1 - r0) * (c1 - c0)) / (this.rows(s) * this.cols(s)); }
  /** Whether an evaluation of that band is worth timing (see MIN_TIMED_SHARE), and no other is being timed. */
  timeable(s, r0, r1, c0, c1) {
    const rows = this.rows(s), least = Math.min(1, Math.max(MIN_BAND_ROWS / rows, MIN_TIMED_SHARE));
    return !this.timing.pending && r1 - r0 >= Math.min(rows, MIN_BAND_ROWS) && this.share(s, r0, r1, c0, c1) >= least;
  }

  /** Records that evaluating `share` of a light at stride s took ms (null: no reading). */
  recordTiming(s, share, ms) {
    const t = this.timing;
    t.samples++;
    // The first runs include shader compiles.
    if (ms === null || !(ms > 0) || t.samples <= 2) return;
    const full = ms / share;
    // Costs carried over from an earlier visit or another image size give way to the first reading.
    if (t.seeded) { t.seeded = false; t.byStride = {}; }
    const b = t.byStride;
    // Falls quickly and rises slowly: one-off stalls (loading, other tabs) must not make a fast GPU
    // look slow for long.
    b[s] = !b[s] ? full : full < b[s] ? 0.5 * b[s] + 0.5 * full : 0.9 * b[s] + 0.1 * full;
  }

  /** Starts from costs measured before (ms a whole light, by stride), until this GPU is timed. */
  seedTiming(byStride) {
    const t = {};
    for (const [s, ms] of Object.entries(byStride || {})) if (Number(s) >= 1 && Number.isFinite(ms) && ms > 0) t[s] = ms;
    if (Object.keys(t).length) { this.timing.byStride = t; this.timing.seeded = true; }
  }

  /**
   * Estimated ms to evaluate one whole light at stride s, or null before anything was measured.
   * Each measured stride k predicts t[k] * (k / s)^2; the smallest prediction wins. Scaling up from a
   * coarser stride also scales its fixed overheads, so those predictions err on the slow side, and a
   * recent fast measurement at any stride corrects a stale slow one (e.g. taken while the GPU was
   * still clocked down).
   */
  evalCost(s) {
    const t = this.timing.byStride, known = Object.keys(t).map(Number);
    if (!known.length) return null;
    return Math.min(...known.map((k) => t[k] * (k / s) ** 2));
  }

  /** A light in the middle of the light box. */
  midLight() {
    return { pos: [0, 1, 2].map((i) => (this.lo[i] + this.hi[i]) / 2), radius: (this.rmin + this.rmax) / 2 };
  }

  /**
   * The finest stride (1, 2, 4 or 8) a light can be evaluated at within `ms`, from an untimed run at
   * stride 4 (after one that sets the pipeline up). Kernels are tuned at it: at full resolution a
   * slow GPU took seconds.
   */
  async affordableStride(ms = 30) {
    const light = this.midLight();
    let t = 0;
    await this.untimed(async () => {
      this.evaluate(light, SPARE, 4);
      await this.finish();
      const t0 = performance.now();
      this.evaluate(light, SPARE, 4);
      await this.finish();
      t = performance.now() - t0;
    });
    return [1, 2, 4].find((s) => t * (4 / s) ** 2 <= ms) ?? 8;
  }

  /** Runs `fn` (which evaluates) without its timings reaching the cost model. */
  async untimed(fn) {
    const saved = this.timing;
    this.timing = { ...saved, pending: {} };
    try { return await fn(); } finally { this.timing = saved; }
  }

  async referenceData(i) {
    await this.refsPromise;
    if (!this.refsBuf) throw new Error("no reference images");
    const img = typed(this.refsBuf, this.scene.refs[i].image);
    const r4 = new Float32Array(this.NP * 4);
    for (let p = 0; p < this.NP; p++) { r4[p * 4] = img[p * 3]; r4[p * 4 + 1] = img[p * 3 + 1]; r4[p * 4 + 2] = img[p * 3 + 2]; }
    return r4;
  }

  /**
   * Sizes the canvas to the pixels it is shown at (never below the image's own size: smaller, the
   * browser scales it down), and draws the image to it again; the engine upscales it (present).
   */
  setDisplaySize(w, h) {
    w = Math.max(this.W, Math.round(w)); h = Math.max(this.H, Math.round(h));
    if (this.canvas.width === w && this.canvas.height === h) return;
    this.canvas.width = w; this.canvas.height = h;
    if (this.composited) this.present();
  }

  /** Evaluates the dirty lights (each at its stride) and composites. */
  render(lights) {
    for (const l of lights) {
      if (!l.dirty) continue;
      this.evaluate(l, l.slot, l.stride || 1);
      l.dirty = false;
    }
    this.composite(lights);
  }

  normLight(l) {
    const { lo, hi } = this;
    return [0, 1, 2].map((i) => (2 * (l.pos[i] - lo[i])) / (hi[i] - lo[i]) - 1)
      .concat([(2 * (l.radius - this.rmin)) / (this.rmax - this.rmin) - 1]);
  }

  project(p) {
    const { X, Y, Z, O, tx, ty } = this.cam;
    const v = [p[0] - O[0], p[1] - O[1], p[2] - O[2]];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const xc = dot(v, X), yc = dot(v, Y), zc = dot(v, Z);
    return { u: 0.5 - xc / (zc * 2 * tx), v: 0.5 - yc / (zc * 2 * ty), z: zc };
  }

  unproject(u, v, zc) {
    const { X, Y, Z, O, tx, ty } = this.cam;
    const xc = (0.5 - u) * 2 * tx * zc, yc = (0.5 - v) * 2 * ty * zc;
    return [0, 1, 2].map((i) => O[i] + X[i] * xc + Y[i] * yc + Z[i] * zc);
  }

  clampLight(l) {
    for (let i = 0; i < 3; i++) l.pos[i] = Math.min(this.hi[i], Math.max(this.lo[i], l.pos[i]));
    l.radius = Math.min(this.rmax, Math.max(this.rmin, l.radius));
  }

  /**
   * CPU copy of the soft-edged direct-view term (directSoft in shaders.js): coverage D of pixel p
   * by light l and dD/d(center.xyz, radius) in world units.
   */
  directSoft(p, l) {
    const zero = { D: 0, g: [0, 0, 0, 0] };
    const { X, Y, Z, O, tx, ty } = this.cam, W = this.W, H = this.H;
    const pu = ((p % W) + 0.5) / W, pv = (Math.floor(p / W) + 0.5) / H;
    const a = (0.5 - pu) * 2 * tx, b = (0.5 - pv) * 2 * ty;
    const d = [0, 1, 2].map((i) => X[i] * a + Y[i] * b + Z[i]);
    const dn = Math.hypot(d[0], d[1], d[2]);
    for (let i = 0; i < 3; i++) d[i] /= dn;
    const r = l.radius;
    const v = [0, 1, 2].map((i) => l.pos[i] - O[i]);
    const Ln = Math.hypot(v[0], v[1], v[2]);
    if (Ln <= r * 1.001) return zero;
    const u = v.map((x) => x / Ln);
    const surf = this.surfaceAt(p)?.dist ?? 1e9;
    if (Ln - r > surf || u[0] * Z[0] + u[1] * Z[1] + u[2] * Z[2] <= 0) return zero;
    const ct = Math.min(1, Math.max(-1, d[0] * u[0] + d[1] * u[1] + d[2] * u[2]));
    const th = Math.acos(ct), sa = r / Ln, al = Math.asin(sa);
    const ca = Math.sqrt(Math.max(1 - sa * sa, 1e-8));
    const delta = (2 * tx) / W;
    const s = (al - th) / delta + 0.5;
    if (s <= 0) return zero;
    if (s >= 1) return { D: 1, g: [0, 0, 0, 0] };
    const st = Math.max(Math.sqrt(Math.max(1 - ct * ct, 0)), 1e-6);
    const g = [0, 1, 2].map((i) => (-(sa / (Ln * ca)) * u[i] + (d[i] - ct * u[i]) / (st * Ln)) / delta);
    g.push(1 / (Ln * ca * delta));
    return { D: s, g };
  }
}
