// Backend-independent parts of the runtime: scene loading, camera maths, light normalisation.
export const MAX_LIGHTS = 8;

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

export const f16tab = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    t[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return t;
})();

export function typed(buf, entry) {
  const n = entry.shape.reduce((a, b) => a * b, 1);
  if (entry.dtype === "float32") return new Float32Array(buf, entry.offset, n);
  const h = new Uint16Array(buf, entry.offset, n);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = f16tab[h[i]];
  return out;
}

/**
 * Unpacks pixel format 2 (nrp/pixels.py): per channel, a plane of 16-bit values stored as
 * differences from the value to their left, low bytes then high bytes. Returns what format 1
 * stored: aux [NP * C] (the network's features), geom [NP * 4] (position, camera distance; 0 where
 * the pixel sees nothing) and normal [NP * 3] (the aux features' normals).
 */
function unpackPixels(buf, px, W, H, O) {
  const NP = W * H;
  const planes = (entry, put) => {
    const bytes = new Uint8Array(buf, entry.offset, entry.bytes);
    for (let k = 0; k < entry.channels; k++) {
      const lo = k * 2 * NP, hi = lo + NP;
      for (let y = 0; y < H; y++) {
        let v = 0;
        for (let p = y * W, end = p + W; p < end; p++) {
          v = (v + (bytes[lo + p] | (bytes[hi + p] << 8))) & 0xffff;
          put(p, k, v);
        }
      }
    }
  };
  const C = px.aux.channels;
  const aux = new Float32Array(NP * C);
  planes(px.aux, (p, k, v) => { aux[p * C + k] = f16tab[v]; });
  const q = new Uint16Array(NP * 3);
  planes(px.pos, (p, k, v) => { q[p * 3 + k] = v; });
  const [lo, hi] = px.pos.range, step = (hi - lo) / 65534;
  const geom = new Float32Array(NP * 4), normal = new Float32Array(NP * 3);
  for (let p = 0; p < NP; p++) {
    for (let k = 0; k < 3; k++) normal[p * 3 + k] = aux[p * C + 3 + k];
    if (!q[p * 3]) continue;
    let d2 = 0;
    for (let k = 0; k < 3; k++) {
      const x = lo + (q[p * 3 + k] - 1) * step;
      geom[p * 4 + k] = x;
      d2 += (x - O[k]) ** 2;
    }
    geom[p * 4 + 3] = Math.sqrt(d2);
  }
  return { aux, geom, normal };
}

/**
 * Both engines evaluate a light either at every pixel (stride 1) or, as a fast preview, at every
 * s-th pixel in x and y (light.stride = s). A preview is stored compactly (ceil(W/s) x ceil(H/s),
 * row-major) in the light's output slot, and the composite upsamples it along the geometry.
 */
export class EngineBase {
  /**
   * Fetches and unpacks the scene; returns the raw pieces the backend uploads. `res` picks an
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
    const [model, pixels] = await Promise.all([fetchBin("model.bin"), fetchBin(`pixels${suffix}.bin`)]);
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
    const grid = new Float32Array(gridLen);
    scene.model.grids.forEach((g, i) => grid.set(typed(model, g), gridOff[i]));
    const layers = scene.model.layers.map((l) => ({ w: typed(model, l.weight), b: typed(model, l.bias), shape: l.weight.shape }));
    this.paramCount = grid.length + layers.reduce((a, l) => a + l.w.length + l.b.length, 0);
    this.modelBytes = model.byteLength;

    const M = scene.camera.to_world;
    const col = (j) => [M[0][j], M[1][j], M[2][j]];
    const tx = Math.tan((scene.camera.fov * Math.PI) / 360);
    this.cam = { X: col(0), Y: col(1), Z: col(2), O: col(3), tx, ty: (tx * H) / W };

    let aux;
    if (scene.pixels.format === 2) {
      ({ aux, geom: this.geom, normal: this.normal } = unpackPixels(pixels, scene.pixels, W, H, this.cam.O));
    } else {
      aux = typed(pixels, scene.pixels.aux);
      this.geom = typed(pixels, scene.pixels.geom);
      this.normal = typed(pixels, scene.pixels.normal);
    }
    this.lo = scene.light_bbox[0]; this.hi = scene.light_bbox[1];
    [this.rmin, this.rmax] = scene.radius_range;

    this.exposure = 1;
    this.mode = 0;
    this.refLight = 0;
    this.timing = { perLight: 0, byStride: {} };
    return { scene, grid, gridOff, layers, aux };
  }

  /** Records the time (ms) of evaluating one light at stride s (every s-th pixel), after warm-up. */
  recordTiming(s, ms) {
    this._evals = (this._evals || 0) + 1;
    if (this._evals > 2) {
      // Falls quickly and rises slowly: one-off stalls (loading, other tabs) must not make a fast
      // GPU look slow for long.
      const t = this.timing.byStride;
      t[s] = !t[s] ? ms : ms < t[s] ? 0.5 * t[s] + 0.5 * ms : 0.9 * t[s] + 0.1 * ms;
      this.timing.perLight = t[1] || 0;
    }
    this._timing = false;
  }

  /**
   * Estimated ms to evaluate one light at stride s, or null before anything was measured. Each
   * measured stride k predicts t[k] * (k / s)^2; the smallest prediction wins. Scaling up from a
   * coarser stride also scales its fixed overheads, so those predictions err on the slow side,
   * and a recent fast measurement at any stride corrects a stale slow one (e.g. taken while the
   * GPU was still clocked down).
   */
  evalCost(s) {
    const t = this.timing.byStride, known = Object.keys(t).map(Number);
    if (!known.length) return null;
    return Math.min(...known.map((k) => t[k] * (k / s) ** 2));
  }

  async referenceData(i) {
    await this.refsPromise;
    if (!this.refsBuf) throw new Error("no reference images");
    const img = typed(this.refsBuf, this.scene.refs[i].image);
    const r4 = new Float32Array(this.NP * 4);
    for (let p = 0; p < this.NP; p++) { r4[p * 4] = img[p * 3]; r4[p * 4 + 1] = img[p * 3 + 1]; r4[p * 4 + 2] = img[p * 3 + 2]; }
    return r4;
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
    const gw = this.geom[p * 4 + 3];
    const surf = gw > 0 ? gw : 1e9;
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
