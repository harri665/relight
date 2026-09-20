// WebGPU runtime for a neural render proxy exported by nrp/export.py.
import { makeShaders } from "./shaders.js";
import { EngineBase, MAX_LIGHTS } from "./engine-base.js";

export { MAX_LIGHTS };

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export class NRPEngine extends EngineBase {
  static async create(canvas) {
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter found.");
    const L = adapter.limits;
    if (L.maxComputeWorkgroupStorageSize < 24576) throw new Error("GPU lacks 24 KB workgroup memory.");
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
        maxBufferSize: L.maxBufferSize,
        maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
        maxStorageBuffersPerShaderStage: Math.min(10, L.maxStorageBuffersPerShaderStage),
      },
    });
    const e = new NRPEngine();
    e.device = device;
    e.backend = "WebGPU";
    e.adapterInfo = adapter.info || {};
    e.canvas = canvas;
    e.ctx = canvas.getContext("webgpu");
    e.format = navigator.gpu.getPreferredCanvasFormat();
    e.ctx.configure({ device, format: e.format, alphaMode: "opaque" });
    device.lost.then((info) => console.error("WebGPU device lost:", info.message));
    return e;
  }

  async load(base, onProgress = () => {}) {
    const dev = this.device;
    const { scene, grid, gridOff, layers, aux } = await this.fetchScene(base, onProgress);
    const { W, H, NP } = this;
    const net = scene.network;
    const WD = net.width, NH = net.hidden - 1;
    const auxDim = net.aux_dim ?? 7, geo = !!net.geo, mul = net.head === "mul";
    if (net.light_grid) throw new Error("light_grid models are not supported by this viewer");
    const NO = mul ? 6 : 3;

    // --- weights. First-layer columns: [pixel grid | aux | light (4) | geo (6)].
    const IN = layers[0].shape[1], PIXIN = net.grid_res.length * net.feats + auxDim;
    const GC = PIXIN + 4;
    const off = {}; let cur = 0;
    const alloc = (k, n) => { off[k] = cur; cur += Math.ceil(n / 4) * 4; };
    alloc("W0", WD * IN); alloc("B0", WD); alloc("W0G", geo ? 6 * WD : 0);
    alloc("HID", NH * (WD * WD + WD)); alloc("HIDB", NH * WD * WD);
    alloc("WO", NO * WD); alloc("BO", 8);
    const P = new Float32Array(cur);
    P.set(layers[0].w, off.W0); P.set(layers[0].b, off.B0);
    if (geo) for (let j = 0; j < 6; j++) for (let c = 0; c < WD; c++) P[off.W0G + j * WD + c] = layers[0].w[c * IN + GC + j];
    // Light columns stay on the CPU: add = W0[:, light] . ln is computed once per light.
    this.W0L = new Float32Array(WD * 4);
    for (let c = 0; c < WD; c++) for (let j = 0; j < 4; j++) this.W0L[c * 4 + j] = layers[0].w[c * IN + PIXIN + j];
    this.WD = WD;
    for (let l = 0; l < NH; l++) {
      const { w, b } = layers[l + 1];
      const o = off.HID + l * (WD * WD + WD);
      for (let c = 0; c < WD; c++) for (let k = 0; k < WD; k++) P[o + k * WD + c] = w[c * WD + k];
      P.set(b, o + WD * WD);
      P.set(w, off.HIDB + l * WD * WD);
    }
    P.set(layers[NH + 1].w, off.WO); P.set(layers[NH + 1].b, off.BO);

    // --- per-pixel buffers
    const AUXS = Math.ceil(auxDim / 4) * 4;
    const auxB = new Float32Array(NP * AUXS);
    for (let p = 0; p < NP; p++) for (let j = 0; j < auxDim; j++) auxB[p * AUXS + j] = aux[p * auxDim + j];
    const pgeo = new Float32Array(NP * 8);   // (position, camera distance), (normal, 0)
    for (let p = 0; p < NP; p++) {
      for (let k = 0; k < 4; k++) pgeo[p * 8 + k] = this.geom[p * 4 + k];
      for (let k = 0; k < 3; k++) pgeo[p * 8 + 4 + k] = this.normal[p * 3 + k];
    }

    onProgress("compiling kernels");
    const S = makeShaders({ WD, NH, W, H, levels: net.grid_res.length, feats: net.feats, gridRes: net.grid_res,
      gridOff, off, auxDim, geo, mul, IN });
    this.S = S;
    const mod = (code, label) => dev.createShaderModule({ code, label });
    const cp = (code, label) => dev.createComputePipelineAsync({ layout: "auto", compute: { module: mod(code, label), entryPoint: "main" }, label });
    const [pPre, pFwd, pComp, pLoss, pGrad] = await Promise.all([
      cp(S.precompute, "precompute"), cp(S.forward, "forward"), cp(S.composite, "composite"),
      cp(S.lossPrep, "lossPrep"), cp(S.grad, "grad")]);
    const blitMod = mod(S.blit, "blit");
    const pBlit = await dev.createRenderPipelineAsync({
      layout: "auto",
      vertex: { module: blitMod, entryPoint: "vs" },
      fragment: { module: blitMod, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });
    Object.assign(this, { pFwd, pComp, pLoss, pGrad, pBlit });

    // --- GPU buffers
    const SU = GPUBufferUsage;
    const buf = (size, usage, data) => {
      const b = dev.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage, mappedAtCreation: !!data });
      if (data) { new data.constructor(b.getMappedRange()).set(data); b.unmap(); }
      return b;
    };
    const ST = SU.STORAGE | SU.COPY_DST | SU.COPY_SRC;
    this.wBuf = buf(P.byteLength, ST, P);
    const gridBuf = buf(grid.byteLength, ST, grid);
    const auxBuf = buf(auxB.byteLength, ST, auxB);
    this.pgeoBuf = buf(pgeo.byteLength, ST, pgeo);
    this.h0Buf = buf(NP * WD * 4, ST);
    this.outBuf = buf(MAX_LIGHTS * NP * 16, ST);
    this.refBuf = buf(NP * 16, ST);
    this.dispBuf = buf(NP * 4, ST);
    this.hdrBuf = buf(NP * 16, ST);
    this.frameBuf = buf(96 + MAX_LIGHTS * 48, SU.UNIFORM | SU.COPY_DST);
    this.jobsBuf = buf(MAX_LIGHTS * S.JOB * 16, ST);
    this.fwdU = buf(16, SU.UNIFORM | SU.COPY_DST, new Uint32Array([NP, 0, 1, NP]));

    // optimisation buffers
    this.K = 4096;
    this.nTiles = Math.ceil(this.K / S.TP);
    const K = this.K;
    this.subIdxBuf = buf(K * 4, ST);
    this.subOutBuf = buf(MAX_LIGHTS * K * 16, ST);
    this.tgtBuf = buf(K * 16, ST);
    this.dLdIBuf = buf(K * 16, ST);
    this.subDirBuf = buf(MAX_LIGHTS * K * 32, ST);
    this.scrBuf = buf(MAX_LIGHTS * this.nTiles * (NH + 1) * S.TP * WD * 4, ST);
    this.partBuf = buf(MAX_LIGHTS * this.nTiles * S.PS * 4, ST);
    this.partRead = buf(MAX_LIGHTS * this.nTiles * S.PS * 4, SU.MAP_READ | SU.COPY_DST);
    this.jobsSubBuf = buf(MAX_LIGHTS * S.JOB * 16, ST);
    this.fwdSubU = buf(16, SU.UNIFORM | SU.COPY_DST, new Uint32Array([K, 1, 1, K]));
    this.optU = buf(32, SU.UNIFORM | SU.COPY_DST);

    const bg = (pipe, group, bufs) => dev.createBindGroup({
      layout: pipe.getBindGroupLayout(group),
      entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
    this.bgFwd = bg(pFwd, 0, [this.wBuf, this.h0Buf, this.jobsBuf, this.fwdU, this.subIdxBuf, this.pgeoBuf, this.outBuf]);
    this.bgFwdSub = bg(pFwd, 0, [this.wBuf, this.h0Buf, this.jobsSubBuf, this.fwdSubU, this.subIdxBuf, this.pgeoBuf, this.subOutBuf]);
    this.bgComp = bg(pComp, 0, [this.frameBuf, this.outBuf, this.pgeoBuf, this.refBuf, this.dispBuf, this.hdrBuf]);
    this.bgBlit = bg(pBlit, 0, [this.dispBuf]);
    this.bgLoss = bg(pLoss, 0, [this.frameBuf, this.optU, this.subIdxBuf, this.subOutBuf, this.tgtBuf, this.pgeoBuf, this.dLdIBuf, this.subDirBuf]);
    this.bgGrad0 = bg(pGrad, 0, [this.wBuf, this.h0Buf, this.jobsSubBuf, this.fwdSubU, this.subIdxBuf, this.pgeoBuf]);
    this.bgGrad1 = bg(pGrad, 1, [this.frameBuf, this.optU, this.dLdIBuf, this.subOutBuf, this.subDirBuf, this.scrBuf, this.partBuf]);

    // --- precompute the light-independent part of layer 0
    onProgress("precomputing pixel features");
    const bgPre = bg(pPre, 0, [gridBuf, auxBuf, this.wBuf, this.h0Buf]);
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pPre); pass.setBindGroup(0, bgPre);
    pass.dispatchWorkgroups(Math.ceil(NP / 64));
    pass.end();
    dev.queue.submit([enc.finish()]);
    await dev.queue.onSubmittedWorkDone();
    gridBuf.destroy(); auxBuf.destroy();

    return scene;
  }

  // ------------------------------------------------------------------ helpers
  writeFrame(lights) {
    const f = new ArrayBuffer(96 + MAX_LIGHTS * 48);
    const F = new Float32Array(f), U = new Uint32Array(f);
    const { X, Y, Z, O, tx, ty } = this.cam;
    F.set(O, 0); F.set(X, 4); F.set(Y, 8); F.set(Z, 12); F.set([tx, ty], 16);
    F[20] = this.exposure; U[21] = this.mode; U[22] = lights.length; U[23] = this.refLight;
    lights.forEach((l, i) => {
      const o = 24 + i * 12;
      F.set([...l.pos, l.radius], o);
      F.set(l.color.map((c) => c * l.intensity), o + 4);
      U[o + 8] = l.slot; U[o + 9] = l.enabled ? 1 : 0; U[o + 10] = l.stride || 1;
    });
    this.device.queue.writeBuffer(this.frameBuf, 0, f);
  }

  /** Job j: [0] center, radius  [1].x slot  [2..] add = W0[:, light] . normalised light params. */
  writeJobs(buffer, entries) {
    const JOB = this.S.JOB, WD = this.WD;
    const F = new Float32Array(MAX_LIGHTS * JOB * 4);
    entries.forEach(([l, slot], j) => {
      const o = j * JOB * 4;
      F.set([...l.pos, l.radius], o);
      F[o + 4] = slot;
      const ln = this.normLight(l);
      for (let c = 0; c < WD; c++) {
        const w = c * 4;
        F[o + 8 + c] = this.W0L[w] * ln[0] + this.W0L[w + 1] * ln[1] + this.W0L[w + 2] * ln[2] + this.W0L[w + 3] * ln[3];
      }
    });
    this.device.queue.writeBuffer(buffer, 0, F);
  }

  // ------------------------------------------------------------------ render
  /** lights: [{pos, radius, color, intensity, enabled, slot, dirty, stride?}] */
  render(lights) {
    const dev = this.device;
    this.writeFrame(lights);
    const dirty = lights.filter((l) => l.dirty);
    // Dirty lights, one dispatch per evaluation stride (see EngineBase).
    const groups = new Map();
    for (const l of dirty) {
      const s = l.stride || 1;
      groups.set(s, [...(groups.get(s) || []), l]);
      l.dirty = false;
    }
    const t0 = performance.now();
    for (const [s, ls] of groups) {
      const n = Math.ceil(this.W / s) * Math.ceil(this.H / s);
      dev.queue.writeBuffer(this.fwdU, 0, new Uint32Array([n, 0, s, this.NP]));
      this.writeJobs(this.jobsBuf, ls.map((l) => [l, l.slot]));
      const enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
      pass.setPipeline(this.pFwd); pass.setBindGroup(0, this.bgFwd);
      pass.dispatchWorkgroups(Math.ceil(n / this.S.TP), ls.length);
      pass.end();
      dev.queue.submit([enc.finish()]);
    }
    if (groups.size === 1 && !this._timing) {
      this._timing = true;
      const [s] = groups.keys();
      dev.queue.onSubmittedWorkDone().then(() => this.recordTiming(s, (performance.now() - t0) / dirty.length));
    }
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pComp); pass.setBindGroup(0, this.bgComp);
    pass.dispatchWorkgroups(Math.ceil(this.NP / 64));
    pass.end();
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }],
    });
    rp.setPipeline(this.pBlit); rp.setBindGroup(0, this.bgBlit); rp.draw(3); rp.end();
    dev.queue.submit([enc.finish()]);
  }

  async readBuffer(src, size) {
    const dev = this.device;
    const rb = dev.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, rb, 0, size);
    dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = rb.getMappedRange().slice(0);
    rb.unmap(); rb.destroy();
    return out;
  }

  /** Current display image as RGBA bytes. */
  async readDisplay() {
    return new Uint8ClampedArray(await this.readBuffer(this.dispBuf, this.NP * 4));
  }

  async readHDR() {
    return new Float32Array(await this.readBuffer(this.hdrBuf, this.NP * 16));
  }

  async loadReference(i) {
    const r4 = await this.referenceData(i);
    this.device.queue.writeBuffer(this.refBuf, 0, r4);
    return r4;
  }

  // ------------------------------------------------------------------ gradients
  /**
   * One stochastic gradient evaluation (paper Sec. 5.3, mini-batch over pixels).
   * idx: Uint32Array(K) pixel indices; tgt: Float32Array(K*4) tonemapped target rgb + weight.
   * Returns per light d loss / d(normalised params) and d loss / dE, plus the loss.
   */
  async gradStep(lights, idx, tgt) {
    const dev = this.device, K = idx.length, nL = lights.length;
    if (K > this.K) throw new Error(`at most ${this.K} pixels per step`);
    const nTiles = Math.ceil(K / this.S.TP);
    dev.queue.writeBuffer(this.fwdSubU, 0, new Uint32Array([K, 1, 1, K]));
    const ou = new ArrayBuffer(32);
    new Uint32Array(ou, 0, 4).set([K, nTiles, 0, 0]);
    // world -> normalised light-parameter units for the direct-term gradient
    new Float32Array(ou, 16, 4).set([0, 1, 2].map((i) => (this.hi[i] - this.lo[i]) / 2).concat([(this.rmax - this.rmin) / 2]));
    dev.queue.writeBuffer(this.optU, 0, ou);
    dev.queue.writeBuffer(this.subIdxBuf, 0, idx);
    dev.queue.writeBuffer(this.tgtBuf, 0, tgt);
    this.writeFrame(lights);
    this.writeJobs(this.jobsSubBuf, lights.map((l, j) => [l, j]));
    const enc = dev.createCommandEncoder();
    let pass = enc.beginComputePass();
    pass.setPipeline(this.pFwd); pass.setBindGroup(0, this.bgFwdSub);
    pass.dispatchWorkgroups(nTiles, nL);
    pass.setPipeline(this.pLoss); pass.setBindGroup(0, this.bgLoss);
    pass.dispatchWorkgroups(Math.ceil(K / 64));
    pass.setPipeline(this.pGrad); pass.setBindGroup(0, this.bgGrad0); pass.setBindGroup(1, this.bgGrad1);
    pass.dispatchWorkgroups(nTiles, nL);
    pass.end();
    const PS = this.S.PS, WD = this.WD;
    const bytes = nL * nTiles * PS * 4;
    enc.copyBufferToBuffer(this.partBuf, 0, this.partRead, 0, bytes);
    dev.queue.submit([enc.finish()]);
    await this.partRead.mapAsync(GPUMapMode.READ, 0, bytes);
    const part = new Float32Array(this.partRead.getMappedRange(0, bytes).slice(0));
    this.partRead.unmap();
    const res = lights.map(() => ({ dln: [0, 0, 0, 0], dE: [0, 0, 0] }));
    let loss = 0;
    const g0 = new Float64Array(WD);
    for (let l = 0; l < nL; l++) {
      g0.fill(0);
      for (let t = 0; t < nTiles; t++) {
        const o = (l * nTiles + t) * PS;
        for (let c = 0; c < WD; c++) g0[c] += part[o + c];
        for (let j = 0; j < 4; j++) res[l].dln[j] += part[o + WD + j];
        for (let j = 0; j < 3; j++) res[l].dE[j] += part[o + WD + 4 + j];
        if (l === 0) loss += part[o + WD + 7];
      }
      // Back through the light columns of the first layer (applied on the CPU in writeJobs).
      for (let c = 0; c < WD; c++) for (let j = 0; j < 4; j++) res[l].dln[j] += this.W0L[c * 4 + j] * g0[c];
    }
    return { grads: res, loss };
  }
}

/** Screen-space footprint of each light's visible disc: (cx, cy, radius) in pixels. */
function discs(engine, lights) {
  const { W, H } = engine, O = engine.cam.O;
  return lights.filter((l) => l.enabled).map((l) => {
    const p = engine.project(l.pos);
    const dist = Math.hypot(...l.pos.map((x, i) => x - O[i]));
    if (p.z <= l.radius) return null;
    return { cx: p.u * W, cy: p.v * H, r: (Math.asin(Math.min(1, l.radius / dist)) / (2 * engine.cam.tx)) * W };
  }).filter(Boolean);
}

/**
 * Pixels near the visible light discs. mode "edge": within `band` px of the rim (where the
 * direct term's gradient lives); mode "disc": the whole disc grown by `band` px.
 * Marks `mask` (Uint8Array, optional) and returns the newly marked pixel indices.
 */
export function discPixels(engine, lights, mode = "edge", band = 2.5, mask = null) {
  const { W, H } = engine, out = [];
  mask = mask || new Uint8Array(W * H);
  for (const { cx, cy, r } of discs(engine, lights)) {
    const R = r + band;
    for (let y = Math.max(0, Math.floor(cy - R)); y <= Math.min(H - 1, Math.ceil(cy + R)); y++) {
      for (let x = Math.max(0, Math.floor(cx - R)); x <= Math.min(W - 1, Math.ceil(cx + R)); x++) {
        const d = Math.hypot(x + 0.5 - cx, y + 0.5 - cy);
        const i = y * W + x;
        const hit = mode === "edge" ? Math.abs(d - r) < band : d < R;
        if (hit && !mask[i]) { mask[i] = 1; out.push(i); }
      }
    }
  }
  return { list: out, mask };
}
export const edgePixels = (engine, lights, band = 2.5) => discPixels(engine, lights, "edge", band);

// ---------------------------------------------------------------- optimiser
/** Adam over the paper's unconstrained reparameterisation (sigmoid box / softplus colour). */
export class LightOptimizer {
  constructor(engine, lights, opts) {
    this.e = engine;
    this.lights = lights;
    this.opts = opts; // { pos, radius, color, lr }
    const logit = (x) => { x = Math.min(1 - 1e-4, Math.max(1e-4, x)); return Math.log(x / (1 - x)); };
    const invSoftplus = (y) => { y = Math.max(y, 1e-4); return y > 20 ? y : Math.log(Math.expm1(y)); };
    const { lo, hi, rmin, rmax } = engine;
    this.theta = lights.map((l) => [
      ...[0, 1, 2].map((i) => logit((l.pos[i] - lo[i]) / (hi[i] - lo[i]))),
      logit((l.radius - rmin) / (rmax - rmin)),
      ...l.color.map((c) => invSoftplus(c * l.intensity)),
    ]);
    this.m = this.theta.map((t) => t.map(() => 0));
    this.v = this.theta.map((t) => t.map(() => 0));
    this.t = 0;
  }

  apply() {
    const { lo, hi, rmin, rmax } = this.e;
    this.lights.forEach((l, k) => {
      const th = this.theta[k];
      l.pos = [0, 1, 2].map((i) => lo[i] + (hi[i] - lo[i]) * sigmoid(th[i]));
      l.radius = rmin + (rmax - rmin) * sigmoid(th[3]);
      const E = th.slice(4).map((v) => (v > 20 ? v : Math.log1p(Math.exp(v))));
      const I = Math.max(...E, 1e-6);
      l.intensity = I;
      l.color = E.map((x) => x / I);
      l.dirty = true;
    });
  }

  step(grads) {
    const { lr } = this.opts;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    this.t++;
    grads.forEach((g, k) => {
      const th = this.theta[k];
      if (!this.lights[k].optimize) return;
      const d = new Array(7).fill(0);
      for (let i = 0; i < 3; i++) if (this.opts.pos) { const s = sigmoid(th[i]); d[i] = g.dln[i] * 2 * s * (1 - s); }
      if (this.opts.radius) { const s = sigmoid(th[3]); d[3] = g.dln[3] * 2 * s * (1 - s); }
      if (this.opts.color) for (let i = 0; i < 3; i++) d[4 + i] = g.dE[i] * sigmoid(th[4 + i]);
      for (let i = 0; i < 7; i++) {
        if (d[i] === 0) continue;
        this.m[k][i] = b1 * this.m[k][i] + (1 - b1) * d[i];
        this.v[k][i] = b2 * this.v[k][i] + (1 - b2) * d[i] * d[i];
        const mh = this.m[k][i] / (1 - b1 ** this.t), vh = this.v[k][i] / (1 - b2 ** this.t);
        // colour lives on a log-ish scale; scale its step with its magnitude
        const scale = i >= 4 ? Math.max(1, Math.abs(th[i])) : 1;
        th[i] -= (lr * scale * mh) / (Math.sqrt(vh) + eps);
      }
    });
    this.apply();
  }
}
