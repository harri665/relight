// ?bench: no viewer starts, and window.__relightBench builds engines on their own for
// perf/relight-bench.mjs: open / time / tune / score / close. Never runs on a normal visit.
// Backported from the portfolio's relight bench.
import { NRPEngine, GLEngine, SPARE } from "../engine/nrp.js";

let engine = null;

function close() {
  engine?.dispose();
  engine = null;
}

/** Loads `scene` at image size `res` (null: native) on `backend` ("webgpu" or "webgl"). */
async function open({ scene = "cornell", res = null, backend = "webgpu" } = {}) {
  close();
  const Engine = backend === "webgl" ? GLEngine : NRPEngine;
  const e = await Engine.create(document.createElement("canvas"));
  await e.load(`/scenes/${scene}`, () => {}, res);
  engine = e;
  // warm up, so shader compiles aren't timed
  await e.untimed(async () => { e.evaluate(e.midLight(), SPARE, 8); await e.finish(); });
  const net = e.scene.network;
  return {
    backend: e.backend, size: e.W, half: !!e.half, precision: e.precision, network: `${net.width}x${net.hidden}`,
    kernel: e.kernelName, gpu: e.adapterInfo.description || e.adapterInfo.architecture || "", refs: e.scene.refs.length,
  };
}

/** Median ms of a whole light at `stride`, over `batches` batches of `n` evaluations back to back. */
async function time({ stride = 1, n = 8, batches = 3 } = {}) {
  const e = engine, light = e.midLight(), runs = [];
  await e.untimed(async () => {
    for (let b = 0; b < batches; b++) {
      await e.finish();
      const t0 = performance.now();
      for (let i = 0; i < n; i++) e.evaluate(light, SPARE, stride);
      await e.finish();
      runs.push((performance.now() - t0) / n);
    }
  });
  runs.sort((a, b) => a - b);
  return { ms: runs[runs.length >> 1], runs, items: e.rows(stride) * e.cols(stride) };
}

/** Times the kernel shapes (WebGPU) or output groups a pass (WebGL) at `stride`, and keeps the fastest. */
async function tune({ stride = 1, runs = 4 } = {}) {
  await engine.tune(stride, runs);
  return { kernel: engine.kernelName, times: engine.kernelTimes };
}

/**
 * Scores the scene's test lights at `stride` against their path-traced references, as the Accuracy
 * tab and nrp/evaluate.py do: PSNR after Reinhard, at an intensity that puts the reference's mean
 * luminance at 0.15. Only at the native size, where the references are.
 */
async function score({ stride = 1 } = {}) {
  const e = engine, T = (x) => { x = Math.max(0, x); return x / (1 + x); };
  const psnr = [];
  for (let i = 0; i < e.scene.refs.length; i++) {
    const ref = await e.referenceData(i);
    let m = 0;
    for (let p = 0; p < e.NP; p++) m += 0.2126 * ref[p * 4] + 0.7152 * ref[p * 4 + 1] + 0.0722 * ref[p * 4 + 2];
    const intensity = 0.15 / Math.max(m / e.NP, 1e-6);
    const L = e.scene.refs[i].light;
    const light = { pos: L.slice(0, 3), radius: L[3], color: [1, 1, 1], intensity, enabled: true, slot: 0, stride };
    await e.untimed(async () => { e.evaluate(light, 0, stride); e.composite([light]); });
    const hdr = await e.readHDR();
    let se = 0;
    for (let p = 0; p < e.NP; p++) for (let k = 0; k < 3; k++) se += (T(hdr[p * 4 + k]) - T(ref[p * 4 + k] * intensity)) ** 2;
    psnr.push(-10 * Math.log10(se / (e.NP * 3)));
  }
  return { psnr, mean: psnr.reduce((a, b) => a + b, 0) / Math.max(1, psnr.length) };
}

export function installBench() {
  window.__relightBench = { open, time, tune, score, close };
}
