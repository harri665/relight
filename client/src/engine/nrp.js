// WebGPU runtime for a neural render proxy exported by nrp/export.py.
import { makeShaders } from "./shaders.js";
import { EngineBase, MAX_LIGHTS, SLOTS, SPARE, chooseKernel, rememberKernel, toHalf } from "./engine-base.js";

export { MAX_LIGHTS, SLOTS, SPARE };
export { GLEngine } from "./nrp-gl.js";

// Shapes of the display kernel (shaders.js fastForward): threads in a workgroup, and pixels each
// thread computes. More pixels per thread reuse each weight more often, for more registers; which
// is fastest depends on the GPU, so load() times them (tune) and remembers the fastest
// per GPU (chooseKernel). `?kernel=128x4` forces one.
export const KERNELS = [
  { threads: 256, pixels: 4 }, { threads: 128, pixels: 4 }, { threads: 256, pixels: 8 },
  { threads: 128, pixels: 8 }, { threads: 64, pixels: 8 }, { threads: 64, pixels: 4 },
];
export const kernelKey = (k) => `${k.threads}x${k.pixels}`;

// ms: queue completions closer than this get reported together
const QUEUE_GRAIN = 0.1;

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export class NRPEngine extends EngineBase {
  static async create(canvas) {
    if (!navigator.gpu) throw new Error(window.isSecureContext ? "this browser has no WebGPU" : "WebGPU needs HTTPS (or localhost)");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("the browser offers no WebGPU adapter for this GPU");
    const L = adapter.limits;
    if (L.maxComputeWorkgroupStorageSize < 24576) throw new Error("GPU lacks 24 KB workgroup memory.");
    // 16-bit arithmetic where the GPU has it (about a third faster on an RTX 3080), and timestamps
    // to time the network on the GPU itself
    const device = await adapter.requestDevice({
      requiredFeatures: ["shader-f16", "timestamp-query"].filter((f) => adapter.features.has(f)),
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
    // A lost device (GPU reset, driver update) is reported to the runtime, which falls back to WebGL.
    device.lost.then((info) => {
      if (info.reason === "destroyed") return;
      console.error("WebGPU device lost:", info.message);
      e.onLost?.(info);
    });
    return e;
  }

  async load(base, onProgress = () => {}, res = null) {
    const dev = this.device;
    const { scene, grid, gridOff, layers, aux, pos } = await this.fetchScene(base, onProgress, res);
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
    // W0T: the pixel columns input-major, padded with zero rows to a whole vec4 of inputs
    alloc("W0T", Math.ceil(PIXIN / 4) * 4 * WD); alloc("W0G", geo ? 6 * WD : 0);
    alloc("HID", NH * (WD * WD + WD)); alloc("HIDB", NH * WD * WD);
    alloc("WO", NO * WD); alloc("BO", 8);
    const P = new Float32Array(cur);
    for (let k = 0; k < PIXIN; k++) for (let c = 0; c < WD; c++) P[off.W0T + k * WD + c] = layers[0].w[c * IN + k];
    if (geo) for (let j = 0; j < 6; j++) for (let c = 0; c < WD; c++) P[off.W0G + j * WD + c] = layers[0].w[c * IN + GC + j];
    // Light columns and the layer-0 bias stay on the CPU: add = b0 + W0[:, light] . ln, once per light.
    this.B0 = layers[0].b;
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

    onProgress("compiling kernels");
    const S = makeShaders({ WD, NH, W, H, levels: net.grid_res.length, feats: net.feats, gridRes: net.grid_res,
      gridOff, off, auxDim, geo, mul, IN, posRange: this.posRange, camO: this.cam.O });
    this.S = S;
    const half = dev.features.has("shader-f16");
    this.half = half;
    const lim = dev.limits;
    // Kernel shapes that fit this device: workgroup memory, invocations, and a light's workgroups
    this.kernels = KERNELS.filter((k) => {
      const tile = (k.threads / S.CG) * k.pixels;
      const bytes = tile * WD * (half ? 2 : 4) + tile * 32 + tile * S.NO * 4;
      return k.threads % S.CG === 0 && k.threads <= lim.maxComputeInvocationsPerWorkgroup &&
        bytes <= lim.maxComputeWorkgroupStorageSize && Math.ceil(NP / tile) <= lim.maxComputeWorkgroupsPerDimension;
    });
    if (!this.kernels.length) throw new Error("not enough workgroup memory for the display kernel");
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
    this.buffers = [];
    const buf = (size, usage, data) => {
      const b = dev.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage, mappedAtCreation: !!data });
      if (data) { new data.constructor(b.getMappedRange()).set(data); b.unmap(); }
      this.buffers.push(b);
      return b;
    };
    const ST = SU.STORAGE | SU.COPY_DST | SU.COPY_SRC;
    this.wBuf = buf(P.byteLength, ST, P);
    this.xBuf = buf(NP * S.XW * 4, ST);   // pixel inputs as f16 pairs (was a WD-float first-layer cache per pixel)
    this.pgeoBuf = buf(NP * 32, ST);
    this.outBuf = buf(SLOTS * NP * S.NW * 4, ST);   // display outputs as halves: rgb, or a and b of a * G + b
    this.refBuf = buf(NP * 16, ST);
    this.dispBuf = buf(NP * 4, ST);
    this.hdrBuf = buf(NP * 16, ST);
    this.frameBuf = buf(96 + MAX_LIGHTS * 48, SU.UNIFORM | SU.COPY_DST);
    this.jobsBuf = buf(MAX_LIGHTS * S.JOB * 16, ST);
    this.bandU = buf(32, SU.UNIFORM | SU.COPY_DST);
    this.bandWords = new Uint32Array(8);
    if (dev.features.has("timestamp-query")) {
      this.querySet = dev.createQuerySet({ type: "timestamp", count: 2 });
      this.queryBuf = buf(16, SU.QUERY_RESOLVE | SU.COPY_SRC);
      this.stampBuf = buf(16, SU.MAP_READ | SU.COPY_DST);
    }

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
    this.bgFwdSub = bg(pFwd, 0, [this.wBuf, this.xBuf, this.jobsSubBuf, this.fwdSubU, this.subIdxBuf, this.pgeoBuf, this.subOutBuf]);
    this.bgComp = bg(pComp, 0, [this.frameBuf, this.outBuf, this.pgeoBuf, this.refBuf, this.dispBuf, this.hdrBuf]);
    this.bgBlit = bg(pBlit, 0, [this.dispBuf]);
    this.bgLoss = bg(pLoss, 0, [this.frameBuf, this.optU, this.subIdxBuf, this.subOutBuf, this.tgtBuf, this.pgeoBuf, this.dLdIBuf, this.subDirBuf]);
    this.bgGrad0 = bg(pGrad, 0, [this.wBuf, this.xBuf, this.jobsSubBuf, this.fwdSubU, this.subIdxBuf, this.pgeoBuf]);
    this.bgGrad1 = bg(pGrad, 1, [this.frameBuf, this.optU, this.dLdIBuf, this.subOutBuf, this.subDirBuf, this.scrBuf, this.partBuf]);

    // --- each pixel's network inputs and geometry, built on the GPU from the 16-bit data
    onProgress("precomputing pixel features");
    const upload = (data) => {
      const b = dev.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 16) * 16), usage: SU.STORAGE, mappedAtCreation: true });
      new Uint16Array(b.getMappedRange()).set(data);
      b.unmap();
      return b;
    };
    const inputs = [upload(grid), upload(aux), upload(pos)];
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pPre); pass.setBindGroup(0, bg(pPre, 0, [...inputs, this.xBuf, this.pgeoBuf]));
    pass.dispatchWorkgroups(Math.ceil(NP / 64));
    pass.end();
    dev.queue.submit([enc.finish()]);
    await dev.queue.onSubmittedWorkDone();
    inputs.forEach((b) => b.destroy());

    // --- the display kernel: f16 weights beside the f32 ones the gradient path reads
    this.wBufH = half ? buf(P.length * 2, ST, toHalf(P)) : this.wBuf;
    // remembered per GPU, network shape and image size (chooseKernel)
    const ai = this.adapterInfo;
    this.kernelStore = `relight-kernel:webgpu:${ai.vendor}/${ai.architecture}/${ai.description}:${W}x${H}:${WD}x${NH}:${half ? 16 : 32}`;
    const { name, forced } = chooseKernel(this.kernelStore);
    const known = this.kernels.find((k) => kernelKey(k) === name);
    await this.setKernel(known || this.kernels[0]);
    this.tuned = !!known || forced || this.kernels.length < 2;
    if (!this.tuned) {
      onProgress("timing kernel shapes");
      await this.tune(await this.affordableStride());
      rememberKernel(this.kernelStore, kernelKey(this.kernel));
    }

    return scene;
  }

  dispose() {
    this.querySet?.destroy();
    this.buffers?.forEach((b) => b.destroy());
    this.buffers = [];
    this.ctx?.unconfigure();
    this.device.destroy();
  }

  // ------------------------------------------------------------------ display kernel
  get kernelName() { return kernelKey(this.kernel); }
  get precision() { return this.half ? "16-bit floats (shader-f16)" : "32-bit floats"; }

  async kernelPipeline(k) {
    const code = this.S.fastForward({ ...k, half: this.half });
    return this.device.createComputePipelineAsync({
      layout: "auto", compute: { module: this.device.createShaderModule({ code, label: `forward ${kernelKey(k)}` }), entryPoint: "main" } });
  }

  async setKernel(k) {
    const pipeline = await this.kernelPipeline(k);
    this.kernel = k;
    this.pFast = pipeline;
    // subIdx (binding 4) is not read by the display kernel, so it is not in its layout
    const list = [[0, this.wBufH], [1, this.xBuf], [2, this.jobsBuf], [3, this.bandU], [5, this.pgeoBuf], [6, this.outBuf]];
    this.bgFast = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries: list.map(([binding, b]) => ({ binding, resource: { buffer: b } })) });
    this.tile = (k.threads / this.S.CG) * k.pixels;
  }

  /**
   * Times each kernel shape that fits on a light in the middle of the box (in the spare slot, at
   * `stride`), and keeps the fastest.
   */
  async tune(stride = 1, runs = 3) {
    const light = this.midLight();
    const done = () => this.device.queue.onSubmittedWorkDone();
    const times = {};
    let best = null;
    await this.untimed(async () => {
      for (const k of this.kernels) {
        await this.setKernel(k);
        this.evaluate(light, SPARE, stride);          // the first run of a pipeline sets it up
        await done();
        const t0 = performance.now();
        for (let i = 0; i < runs; i++) this.evaluate(light, SPARE, stride);
        await done();
        const ms = (performance.now() - t0) / runs;
        times[kernelKey(k)] = ms;
        if (!best || ms < best.ms) best = { k, ms };
      }
    });
    await this.setKernel(best.k);
    this.kernelTimes = times;
    this.tuned = true;
    console.info("kernel shapes (ms per light):", times, "->", kernelKey(best.k));
  }

  /**
   * Evaluates light l into output slot `slot` at stride s: the item rows [r0, r1) and columns
   * [c0, c1) of its ceil(W/s) x ceil(H/s) grid (by default all of them). Large enough evaluations
   * are timed on the GPU (see pollTiming).
   */
  evaluate(l, slot, s = 1, r0 = 0, r1 = this.rows(s), c0 = 0, c1 = this.cols(s)) {
    const dev = this.device;
    r1 = Math.min(r1, this.rows(s)); c1 = Math.min(c1, this.cols(s));
    const xw = c1 - c0, i0 = r0 * xw, i1 = r1 * xw;
    if (i1 <= i0 || xw <= 0) return;
    this.bandWords.set([i0, i1, s, c0, xw]);
    dev.queue.writeBuffer(this.bandU, 0, this.bandWords);
    this.writeJobs(this.jobsBuf, [[l, slot]]);
    const timed = this.timeable(s, r0, r1, c0, c1);
    const stamped = timed && this.querySet && this.stampBuf.mapState === "unmapped";
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass(stamped
      ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } }
      : undefined);
    pass.setPipeline(this.pFast); pass.setBindGroup(0, this.bgFast);
    pass.dispatchWorkgroups(Math.ceil((i1 - i0) / this.tile));
    pass.end();
    if (stamped) {
      enc.resolveQuerySet(this.querySet, 0, 2, this.queryBuf, 0);
      enc.copyBufferToBuffer(this.queryBuf, 0, this.stampBuf, 0, 16);
    }
    // Without timestamps: from when the work queued before this finished to when this did
    const queued = timed && !stamped ? this.queueDone() : null;
    dev.queue.submit([enc.finish()]);
    if (timed) this.time(s, this.share(s, r0, r1, c0, c1), stamped, queued);
  }

  queueDone() { return this.device.queue.onSubmittedWorkDone().then(() => performance.now()); }

  // Many phones have no timestamp queries. Timing from submit counted the rest of the frame and
  // made the network look several times slower than it is.
  time(s, share, stamped, queued) {
    const pending = { stride: s, share, done: false, ms: null };
    this.timing.pending = pending;
    const start = performance.now();
    const settle = (ms) => { pending.ms = ms > 0 ? ms : null; pending.done = true; };
    if (stamped) {
      this.stampBuf.mapAsync(GPUMapMode.READ).then(() => {
        const t = new BigUint64Array(this.stampBuf.getMappedRange());
        const ms = Number(t[1] - t[0]) / 1e6;
        this.stampBuf.unmap();
        settle(ms);
      }).catch(() => settle(null));
    } else {
      // Chrome sometimes resolves both at once behind a long queue; then time since submit.
      Promise.all([queued, this.queueDone()]).then(([before, after]) => {
        const ms = after - Math.max(start, before);
        settle(ms > QUEUE_GRAIN ? ms : after - start);
      }).catch(() => settle(null));
    }
  }

  /** Records the evaluation being timed once its reading is in; call once a frame. */
  pollTiming() {
    const p = this.timing.pending;
    if (p?.done) { this.timing.pending = null; this.recordTiming(p.stride, p.share, p.ms); }
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
    const F = new Float32Array(entries.length * JOB * 4);
    entries.forEach(([l, slot], j) => {
      const o = j * JOB * 4;
      F.set([...l.pos, l.radius], o);
      F[o + 4] = slot;
      const ln = this.normLight(l);
      for (let c = 0; c < WD; c++) {
        const w = c * 4;
        F[o + 8 + c] = this.B0[c] + this.W0L[w] * ln[0] + this.W0L[w + 1] * ln[1] + this.W0L[w + 2] * ln[2] + this.W0L[w + 3] * ln[3];
      }
    });
    this.device.queue.writeBuffer(buffer, 0, F);
  }

  // ------------------------------------------------------------------ render
  /**
   * Puts the lights' slots together into the image and shows it.
   * lights: [{pos, radius, color, intensity, enabled, slot, stride?}], pos and radius as evaluated.
   */
  composite(lights) {
    const dev = this.device;
    this.writeFrame(lights);
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pComp); pass.setBindGroup(0, this.bgComp);
    pass.dispatchWorkgroups(Math.ceil(this.NP / 64));
    pass.end();
    dev.queue.submit([enc.finish()]);
    this.present();
  }

  /** Draws the last composited image to the canvas. */
  present() {
    const dev = this.device;
    const enc = dev.createCommandEncoder();
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }],
    });
    rp.setPipeline(this.pBlit); rp.setBindGroup(0, this.bgBlit); rp.draw(3); rp.end();
    dev.queue.submit([enc.finish()]);
  }

  /** Resolves once the GPU has finished the work submitted so far. */
  finish() { return this.device.queue.onSubmittedWorkDone(); }

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
