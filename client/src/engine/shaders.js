// WGSL kernels for the neural render proxy, specialised to one network shape.
//
// Weight buffer layout (f32, every offset a multiple of 4) — see load() in nrp.js:
//   W0T   [PIN4*4][WD] first layer's pixel columns, input-major (row k = input k, zero rows pad to
//                    a whole vec4 of inputs); the layer-0 bias is part of `add` below
//   W0G   [6][WD]    geo-feature columns of the first layer, transposed      (if GEO)
//   HID   NH x ([WD][WD] transposed (k-major) + bias [WD])   forward
//   HIDB  NH x [WD][WD] row-major (c-major)                  backward
//   WO    [NO][WD], BO [8]      NO = 3, or 6 for the multiplicative head (out = a * G + b)
//
// The light-dependent part of layer 0 (bias and light parameters) is the same for every pixel, so
// nrp.js evaluates it on the CPU once per light and passes it in the job buffer as `add` [WD].
// The per-pixel part of layer 0 runs in the kernels, from X: each pixel's network inputs (grid
// encoding and aux features) as pairs of f16 packed in u32, XW words a pixel. That is 20 words
// instead of the WD floats of a cached first-layer output (80 B against 512 B a pixel).
//
// Job buffer (array<vec4f>, JOB = 2 + WD/4 entries per light):
//   [0] light center xyz, radius   [1].x slot (as float)   [2..] add
//
// Pixel geometry buffer pgeo (array<vec4f>, 2 per pixel): (position, camera distance; 0 = no hit),
// (shading normal, 0).

import { MAX_LIGHTS } from "./engine-base.js";
export { MAX_LIGHTS };

export function makeShaders(c) {
  const { WD, NH, W, H, levels, feats, gridRes, gridOff, off, auxDim, geo, mul } = c;
  const CG = WD / 4;          // channel groups of 4 per row
  const PG = 256 / CG;        // pixel groups per workgroup
  const TP = PG * 4;          // pixels per workgroup tile
  const ENC = levels * feats;
  const PIXIN = ENC + auxDim;
  const AUXS = Math.ceil(auxDim / 4) * 4;
  const XW = Math.ceil(PIXIN / 2);
  const IN = c.IN;
  const NP = W * H;
  const NO = mul ? 6 : 3;
  const JOB = 2 + CG;
  const PS = WD + 8;          // per-tile partial sums: sum g0 [WD], dln extra [4], dE [3], loss
  const u = (x) => `${x >>> 0}u`;
  const arr = (a) => `array<u32, ${a.length}>(${a.map(u).join(", ")})`;
  const HSTRIDE = WD * WD + WD;

  const structs = /* wgsl */ `
struct Light { pr: vec4f, e: vec4f, info: vec4u };
struct Frame {
  camO: vec4f, camX: vec4f, camY: vec4f, camZ: vec4f,
  tanxy: vec4f,          // tan(fov_x/2), tan(fov_y/2)
  exposure: f32, mode: u32, nLights: u32, refLight: u32,
  lights: array<Light, ${MAX_LIGHTS}>,
};
`;

  // Geometric features of pixel position x / normal n w.r.t. a sphere light (c, r):
  // f = [l, cos, 0.5 ln d, 0.25 ln(omega + 1e-6)], G = omega max(cos, 0) / pi. Mirrors
  // geo_features() in nrp/model.py.
  const geoFns = /* wgsl */ `
const PI = 3.14159265358979;
struct Geo { f0: vec4f, f1: vec4f };   // f0 = (l, cos), f1 = (0.5 ln d, 0.25 ln omega, G, valid)
fn geoFeatures(x: vec4f, n: vec3f, lc: vec4f) -> Geo {
  var o: Geo;
  o.f0 = vec4f(0.0); o.f1 = vec4f(0.0);
  if (x.w <= 0.0) { return o; }
  let v = lc.xyz - x.xyz;
  let d = max(length(v), 1e-4);
  let l = v / d;
  let cs = dot(n, l);
  let s = min(lc.w / d, 1.0);
  let om = 2.0 * PI * s * s / (1.0 + sqrt(max(1.0 - s * s, 0.0)));   // = 2 pi (1 - sqrt(1 - s^2))
  o.f0 = vec4f(l, cs);
  o.f1 = vec4f(0.5 * log(d), 0.25 * log(om + 1e-6), om * max(cs, 0.0) / PI, 1.0);
  return o;
}
// Chain rule: given dL/df (6) and dL/dG, return dL/d(center xyz, radius).
fn geoBackward(x: vec4f, n: vec3f, lc: vec4f, gl: vec3f, gcos0: f32, gld: f32, glo: f32, gG: f32) -> vec4f {
  if (x.w <= 0.0) { return vec4f(0.0); }
  let v = lc.xyz - x.xyz;
  let d = max(length(v), 1e-4);
  let l = v / d;
  let cs = dot(n, l);
  let s = min(lc.w / d, 1.0);
  let sq = sqrt(max(1.0 - s * s, 0.0));
  let om = 2.0 * PI * s * s / (1.0 + sq);
  let gcos = gcos0 + select(0.0, gG * om / PI, cs > 0.0);
  let gom = glo * 0.25 / (om + 1e-6) + gG * max(cs, 0.0) / PI;
  var dc = (gl - dot(gl, l) * l) / d + gcos * (n - cs * l) / d + gld * 0.5 * l / d;
  var dr = 0.0;
  if (s < 1.0) {
    let domds = 2.0 * PI * s / max(sq, 1e-3);
    dc += gom * domds * (-(s / d) * l);
    dr = gom * domds / d;
  }
  return vec4f(dc, dr);
}
`;

  const common = structs + /* wgsl */ `
const W = ${u(W)}; const H = ${u(H)}; const NP = ${u(NP)};

fn camDir(uv: vec2f) -> vec3f {
  let d = frame.camX.xyz * ((0.5 - uv.x) * 2.0 * frame.tanxy.x)
        + frame.camY.xyz * ((0.5 - uv.y) * 2.0 * frame.tanxy.y)
        + frame.camZ.xyz;
  return normalize(d);
}

// Fraction of the pixel footprint (4x4 subsamples) that sees sphere light l
// before the first surface: the analytic segment-0 term.
fn directCov(p: u32, l: u32) -> f32 {
  let L = frame.lights[l];
  let c = L.pr.xyz; let r = L.pr.w;
  let g = pgeo[2u * p];
  let surf = select(1e9, g.w, g.w > 0.0);
  let o = frame.camO.xyz;
  let oc = o - c;
  let px = f32(p % W); let py = f32(p / W);
  // cheap reject using the pixel-centre ray
  let d0 = camDir(vec2f((px + 0.5) / f32(W), (py + 0.5) / f32(H)));
  let b0 = dot(d0, oc);
  let pixAng = 2.0 * frame.tanxy.x / f32(W) * 1.5;
  let dist2 = dot(oc, oc) - b0 * b0;
  let slack = r + pixAng * length(oc);
  if (dist2 > slack * slack || b0 > 0.0) { return 0.0; }
  var cnt = 0.0;
  for (var sy = 0u; sy < 4u; sy++) {
    for (var sx = 0u; sx < 4u; sx++) {
      let uv = vec2f((px + (f32(sx) + 0.5) / 4.0) / f32(W), (py + (f32(sy) + 0.5) / 4.0) / f32(H));
      let d = camDir(uv);
      let b = dot(d, oc);
      let disc = b * b - dot(oc, oc) + r * r;
      if (disc >= 0.0) {
        let t0 = -b - sqrt(disc);
        if (t0 > 0.0 && t0 < surf) { cnt += 1.0; }
      }
    }
  }
  return cnt / 16.0;
}

// Differentiable version for optimisation: coverage ramps linearly across a
// one-pixel band around the disc edge (angular distance vs angular radius).
// Returns D and dD/d(center.xyz, radius) in world units.
struct DGrad { D: f32, g: vec4f };
fn directSoft(p: u32, l: u32) -> DGrad {
  var o: DGrad;
  o.D = 0.0; o.g = vec4f(0.0);
  let L = frame.lights[l];
  let c = L.pr.xyz; let r = L.pr.w;
  let d = camDir(vec2f((f32(p % W) + 0.5) / f32(W), (f32(p / W) + 0.5) / f32(H)));
  let v = c - frame.camO.xyz;
  let Ln = length(v);
  if (Ln <= r * 1.001) { return o; }
  let u = v / Ln;
  let gm = pgeo[2u * p];
  let surf = select(1e9, gm.w, gm.w > 0.0);
  if (Ln - r > surf || dot(u, frame.camZ.xyz) <= 0.0) { return o; }
  let ct = clamp(dot(d, u), -1.0, 1.0);
  let th = acos(ct);
  let sa = r / Ln;
  let al = asin(sa);
  let ca = sqrt(max(1.0 - sa * sa, 1e-8));
  let delta = 2.0 * frame.tanxy.x / f32(W);
  let s = (al - th) / delta + 0.5;
  if (s <= 0.0) { return o; }
  if (s >= 1.0) { o.D = 1.0; return o; }
  o.D = s;
  let st = max(sqrt(max(1.0 - ct * ct, 0.0)), 1e-6);
  let dth_dc = -(d - ct * u) / (st * Ln);
  let dal_dc = -(sa / (Ln * ca)) * u;
  o.g = vec4f((dal_dc - dth_dc) / delta, 1.0 / (Ln * ca * delta));
  return o;
}
`;

  // ---------------------------------------------------------------- precompute
  const precompute = /* wgsl */ `
@group(0) @binding(0) var<storage, read> grid: array<f32>;
@group(0) @binding(1) var<storage, read> auxB: array<f32>;
@group(0) @binding(2) var<storage, read_write> X: array<u32>;
var<private> RES: array<u32, ${levels}> = ${arr(gridRes)};
var<private> GOFF: array<u32, ${levels}> = ${arr(gridOff)};
const W = ${u(W)}; const H = ${u(H)}; const NP = ${u(NP)};
const F = ${u(feats)};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let p = gid.x;
  if (p >= NP) { return; }
  let u = (f32(p % W) + 0.5) / f32(W);
  let v = (f32(p / W) + 0.5) / f32(H);
  var x: array<f32, ${XW * 2}>;
  for (var l = 0u; l < ${u(levels)}; l++) {
    let R = RES[l];
    let fx = u * f32(R - 1u); let fy = v * f32(R - 1u);
    let x0 = min(u32(floor(fx)), R - 2u); let y0 = min(u32(floor(fy)), R - 2u);
    let tx = fx - f32(x0); let ty = fy - f32(y0);
    let b = GOFF[l];
    for (var f = 0u; f < F; f++) {
      let v00 = grid[b + (y0 * R + x0) * F + f];
      let v01 = grid[b + (y0 * R + x0 + 1u) * F + f];
      let v10 = grid[b + ((y0 + 1u) * R + x0) * F + f];
      let v11 = grid[b + ((y0 + 1u) * R + x0 + 1u) * F + f];
      x[l * F + f] = mix(mix(v00, v01, tx), mix(v10, v11, tx), ty);
    }
  }
  for (var j = 0u; j < ${u(auxDim)}; j++) { x[${u(ENC)} + j] = auxB[p * ${u(AUXS)} + j]; }
  for (var j = 0u; j < ${u(XW)}; j++) { X[p * ${u(XW)} + j] = pack2x16float(vec2f(x[2u * j], x[2u * j + 1u])); }
}
`;

  // ---------------------------------------------------------- shared MLP body
  // Per-pixel geo features -> workgroup memory, layer 0 (+ light, + geo) and the hidden
  // layers into workgroup memory `act` ([TP][WD]). If SCRATCH, every post-activation layer
  // is also stored for the backward pass.
  const mlpBody = (scratch) => /* wgsl */ `
  let pg = t / ${u(CG)};
  let cg = t % ${u(CG)};
  let jb = wg.y * ${u(JOB)};
  let lc = jobs[jb];
  if (t < ${u(TP)}) {
    let i = base + t;
    var gf: Geo;
    gf.f0 = vec4f(0.0); gf.f1 = vec4f(0.0);
    if (i < n) {
      let p = pixelOf(i);
      gf = geoFeatures(pgeo[2u * p], pgeo[2u * p + 1u].xyz, lc);
    }
    pf[2u * t] = gf.f0; pf[2u * t + 1u] = gf.f1;
  }
  workgroupBarrier();
  let add = jobs[jb + 2u + cg];
  ${geo ? `let wg0 = Wv[${u(off.W0G / 4)} + 0u * ${u(CG)} + cg]; let wg1 = Wv[${u(off.W0G / 4)} + 1u * ${u(CG)} + cg];
  let wg2 = Wv[${u(off.W0G / 4)} + 2u * ${u(CG)} + cg]; let wg3 = Wv[${u(off.W0G / 4)} + 3u * ${u(CG)} + cg];
  let wg4 = Wv[${u(off.W0G / 4)} + 4u * ${u(CG)} + cg]; let wg5 = Wv[${u(off.W0G / 4)} + 5u * ${u(CG)} + cg];` : ""}
  // The tile's pixel inputs, unpacked into act (the first XW * 2 of a pixel's WD slots)
  for (var e = t; e < ${u(TP * XW)}; e += 256u) {
    let pl = e / ${u(XW)}; let j = e % ${u(XW)};
    var v = vec2f(0.0);
    if (base + pl < n) { v = unpack2x16float(X[pixelOf(base + pl) * ${u(XW)} + j]); }
    act[pl * ${u(WD)} + 2u * j] = v.x; act[pl * ${u(WD)} + 2u * j + 1u] = v.y;
  }
  workgroupBarrier();
  // Layer 0: per-pixel columns from the inputs, then the light's add and the geo columns
  var h = array<vec4f, 4>(add, add, add, add);
  for (var k = 0u; k < ${u(PIXIN)}; k++) {
    let w = Wv[${u(off.W0T / 4)} + k * ${u(CG)} + cg];
    for (var q = 0u; q < 4u; q++) { h[q] += act[(pg * 4u + q) * ${u(WD)} + k] * w; }
  }
  ${geo ? `for (var q = 0u; q < 4u; q++) {
    let f0 = pf[2u * (pg * 4u + q)]; let f1 = pf[2u * (pg * 4u + q) + 1u];
    h[q] += wg0 * f0.x + wg1 * f0.y + wg2 * f0.z + wg3 * f0.w + wg4 * f1.x + wg5 * f1.y;
  }` : ""}
  workgroupBarrier();
  for (var q = 0u; q < 4u; q++) {
    let pl = pg * 4u + q;
    let v = select(vec4f(0.0), max(h[q], vec4f(0.0)), base + pl < n);
    let a = pl * ${u(WD)} + cg * 4u;
    act[a] = v.x; act[a + 1u] = v.y; act[a + 2u] = v.z; act[a + 3u] = v.w;
    ${scratch ? "scr[sbase + a / 4u] = v;" : ""}
  }
  workgroupBarrier();
  for (var l = 0u; l < ${u(NH)}; l++) {
    let wo = ${u(off.HID / 4)} + l * ${u(HSTRIDE / 4)};
    let bias = Wv[wo + ${u((WD * WD) / 4)} + cg];
    var a0 = bias; var a1 = bias; var a2 = bias; var a3 = bias;
    let r0 = (pg * 4u) * ${u(WD)};
    for (var k = 0u; k < ${u(WD)}; k++) {
      let w = Wv[wo + k * ${u(CG)} + cg];
      a0 += act[r0 + k] * w;
      a1 += act[r0 + ${u(WD)} + k] * w;
      a2 += act[r0 + ${u(2 * WD)} + k] * w;
      a3 += act[r0 + ${u(3 * WD)} + k] * w;
    }
    workgroupBarrier();
    var res = array<vec4f, 4>(max(a0, vec4f(0.0)), max(a1, vec4f(0.0)), max(a2, vec4f(0.0)), max(a3, vec4f(0.0)));
    for (var q = 0u; q < 4u; q++) {
      let a = (pg * 4u + q) * ${u(WD)} + cg * 4u;
      let v = res[q];
      act[a] = v.x; act[a + 1u] = v.y; act[a + 2u] = v.z; act[a + 3u] = v.w;
      ${scratch ? `scr[sbase + (l + 1u) * ${u((TP * WD) / 4)} + a / 4u] = v;` : ""}
    }
    workgroupBarrier();
  }
`;

  // Output row o (0..NO-1) of the last layer for tile pixel pl.
  const outRow = /* wgsl */ `
fn outRow(o: u32, pl: u32) -> f32 {
  var acc = wf(${u(off.BO)} + o);
  let row = (${u(off.WO)} + o * ${u(WD)}) / 4u;
  for (var k = 0u; k < ${u(CG)}; k++) {
    let w = Wv[row + k];
    let b = pl * ${u(WD)} + k * 4u;
    acc += w.x * act[b] + w.y * act[b + 1u] + w.z * act[b + 2u] + w.w * act[b + 3u];
  }
  return acc;
}
`;

  const mlpDecls = /* wgsl */ `
// n items; subset: items are subIdx pixels; stride s > 1: item i is pixel (i % cw, i / cw) * s with
// cw = ceil(W / s); outputs go to outB[slot * slotStride + i].
struct FwdU { n: u32, subset: u32, stride: u32, slotStride: u32 };
@group(0) @binding(0) var<storage, read> Wv: array<vec4f>;
@group(0) @binding(1) var<storage, read> X: array<u32>;
@group(0) @binding(2) var<storage, read> jobs: array<vec4f>;
@group(0) @binding(3) var<uniform> fu: FwdU;
@group(0) @binding(4) var<storage, read> subIdx: array<u32>;
@group(0) @binding(5) var<storage, read> pgeo: array<vec4f>;
var<workgroup> act: array<f32, ${TP * WD}>;
var<workgroup> pf: array<vec4f, ${2 * TP}>;
fn pixelOf(i: u32) -> u32 {
  if (fu.subset == 1u) { return subIdx[i]; }
  if (fu.stride <= 1u) { return i; }
  let cw = (${u(W)} + fu.stride - 1u) / fu.stride;
  return (i / cw) * fu.stride * ${u(W)} + (i % cw) * fu.stride;
}
fn wf(i: u32) -> f32 { return Wv[i / 4u][i % 4u]; }
${geoFns}
${outRow}
`;

  // ---------------------------------------------------------------- forward
  const forward = /* wgsl */ `
${mlpDecls}
@group(0) @binding(6) var<storage, read_write> outB: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let base = wg.x * ${u(TP)};
  let n = fu.n;
${mlpBody(false)}
  if (t < ${u(TP * 3)}) {
    let pl = t / 3u; let o = t % 3u;
    let i = base + pl;
    if (i < n) {
      ${mul ? "let val = outRow(o, pl) * pf[2u * pl + 1u].z + outRow(o + 3u, pl);" : "let val = outRow(o, pl);"}
      outB[(u32(jobs[jb + 1u].x) * fu.slotStride + i) * 4u + o] = val;
    }
  }
}
`;


  // ------------------------------------------------------- fast forward (display path)
  // The same network as `forward`, for evaluating lights to display, shaped by (threads, pixels):
  // a thread computes 4 channels of `pixels` pixels, and reuses each weight it loads for all of
  // them. Activations are vec4s (a pixel's four channels in one workgroup-memory load), and with
  // `half` the arithmetic and workgroup memory are 16-bit floats from 16-bit weights (shader-f16).
  // Which shape is fastest depends on the GPU (NRPEngine.tune). The gradient path keeps `forward`:
  // its backward pass recomputes activations in f32.
  const fastForward = ({ threads, pixels: P, half }) => {
    const PGf = threads / CG, TPf = PGf * P;
    const Q = [...Array(P).keys()];
    const V = half ? "vec4h" : "vec4f";
    const PIN4 = Math.ceil(PIXIN / 4);
    const X4 = Math.ceil(XW / 2);
    const fma = (accs, x, w) => accs
      .map((acc, q) => `${acc} += ${x(q)}.x * ${w}0 + ${x(q)}.y * ${w}1 + ${x(q)}.z * ${w}2 + ${x(q)}.w * ${w}3;`).join("\n      ");
    const rows4 = (at) => [0, 1, 2, 3].map((j) => `let w${j} = Wv[${at} + ${u(j * CG)}];`).join(" ");
    return /* wgsl */ `
${half ? "enable f16;" : ""}
struct FwdU { n: u32, subset: u32, stride: u32, slotStride: u32 };
@group(0) @binding(0) var<storage, read> Wv: array<${V}>;
@group(0) @binding(1) var<storage, read> X: array<u32>;
@group(0) @binding(2) var<storage, read> jobs: array<vec4f>;
@group(0) @binding(3) var<uniform> fu: FwdU;
@group(0) @binding(5) var<storage, read> pgeo: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> outB: array<f32>;
var<workgroup> act: array<${V}, ${TPf * CG}>;
var<workgroup> pf: array<vec4f, ${2 * TPf}>;
var<workgroup> res: array<f32, ${TPf * NO}>;
${geoFns}
fn pixelOf(i: u32) -> u32 {
  if (fu.stride <= 1u) { return i; }
  let cw = (${u(W)} + fu.stride - 1u) / fu.stride;
  return (i / cw) * fu.stride * ${u(W)} + (i % cw) * fu.stride;
}
fn outRow(o: u32, pl: u32) -> f32 {
  var acc = f32(Wv[${u(off.BO / 4)} + o / 4u][o % 4u]);
  let row = ${u(off.WO / 4)} + o * ${u(CG)};
  for (var k = 0u; k < ${u(CG)}; k++) {
    acc += f32(dot(Wv[row + k], act[pl * ${u(CG)} + k]));
  }
  return acc;
}

@compute @workgroup_size(${threads})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let n = fu.n;
  let base = wg.x * ${u(TPf)};
  let pg = t / ${u(CG)};
  let cg = t % ${u(CG)};
  let jb = wg.y * ${u(JOB)};
  let lc = jobs[jb];

  // The tile's pixel inputs, unpacked into act, and its geometric features
  for (var e = t; e < ${u(TPf * X4)}; e += ${u(threads)}) {
    let pl = e / ${u(X4)}; let q = e % ${u(X4)};
    var v = vec4f(0.0);
    if (base + pl < n) {
      let at = pixelOf(base + pl) * ${u(XW)} + 2u * q;
      // an odd word count leaves the last vec4 half empty
      let hi = ${XW % 2 ? `select(vec2f(0.0), unpack2x16float(X[at + 1u]), 2u * q + 1u < ${u(XW)})` : "unpack2x16float(X[at + 1u])"};
      v = vec4f(unpack2x16float(X[at]), hi);
    }
    act[pl * ${u(CG)} + q] = ${V}(v);
  }
  if (t < ${u(TPf)}) {
    var gf: Geo;
    gf.f0 = vec4f(0.0); gf.f1 = vec4f(0.0);
    if (base + t < n) {
      let p = pixelOf(base + t);
      gf = geoFeatures(pgeo[2u * p], pgeo[2u * p + 1u].xyz, lc);
    }
    pf[2u * t] = gf.f0; pf[2u * t + 1u] = gf.f1;
  }
  workgroupBarrier();

  // Layer 0
  let add = ${V}(jobs[jb + 2u + cg]);
  var h = array<${V}, ${P}>(${Q.map(() => "add").join(", ")});
  let r0 = pg * ${u(P * CG)};
  for (var k = 0u; k < ${u(PIN4)}; k++) {
    ${rows4(`${u(off.W0T / 4)} + k * ${u(4 * CG)} + cg`)}
    ${fma(Q.map((q) => `h[${q}]`), (q) => `act[r0 + ${u(q * CG)} + k]`, "w")}
  }
  ${geo ? `for (var q = 0u; q < ${u(P)}; q++) {
    let f0 = ${V}(pf[2u * (pg * ${u(P)} + q)]); let f1 = ${V}(pf[2u * (pg * ${u(P)} + q) + 1u]);
    h[q] += Wv[${u(off.W0G / 4)} + cg] * f0.x + Wv[${u(off.W0G / 4 + CG)} + cg] * f0.y
          + Wv[${u(off.W0G / 4 + 2 * CG)} + cg] * f0.z + Wv[${u(off.W0G / 4 + 3 * CG)} + cg] * f0.w
          + Wv[${u(off.W0G / 4 + 4 * CG)} + cg] * f1.x + Wv[${u(off.W0G / 4 + 5 * CG)} + cg] * f1.y;
  }` : ""}
  workgroupBarrier();
  for (var q = 0u; q < ${u(P)}; q++) {
    let pl = pg * ${u(P)} + q;
    act[pl * ${u(CG)} + cg] = select(${V}(0.0), max(h[q], ${V}(0.0)), base + pl < n);
  }
  workgroupBarrier();

  // Hidden layers
  for (var l = 0u; l < ${u(NH)}; l++) {
    let wo = ${u(off.HID / 4)} + l * ${u(HSTRIDE / 4)};
    let bias = Wv[wo + ${u(WD * CG)} + cg];
    ${Q.map((q) => `var a${q} = bias;`).join(" ")}
    for (var k = 0u; k < ${u(CG)}; k++) {
      ${rows4(`wo + k * ${u(4 * CG)} + cg`)}
      ${fma(Q.map((q) => `a${q}`), (q) => `act[r0 + ${u(q * CG)} + k]`, "w")}
    }
    workgroupBarrier();
    ${Q.map((q) => `act[r0 + ${u(q * CG)} + cg] = max(a${q}, ${V}(0.0));`).join("\n    ")}
    workgroupBarrier();
  }

  // Output rows (rgb, or the 'mul' head's a and b), then a * G + b per pixel
  for (var e = t; e < ${u(TPf * NO)}; e += ${u(threads)}) {
    res[e] = outRow(e % ${u(NO)}, e / ${u(NO)});
  }
  workgroupBarrier();
  if (t < ${u(TPf)} && base + t < n) {
    let r = t * ${u(NO)};
    let o = (u32(jobs[jb + 1u].x) * fu.slotStride + base + t) * 4u;
    ${mul
      ? `let G = pf[2u * t + 1u].z;
    outB[o] = res[r] * G + res[r + 3u]; outB[o + 1u] = res[r + 1u] * G + res[r + 4u]; outB[o + 2u] = res[r + 2u] * G + res[r + 5u];`
      : "outB[o] = res[r]; outB[o + 1u] = res[r + 1u]; outB[o + 2u] = res[r + 2u];"}
  }
}
`;
  };

  // ---------------------------------------------------------------- composite
  const composite = /* wgsl */ `
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> outB: array<vec4f>;
@group(0) @binding(2) var<storage, read> pgeo: array<vec4f>;
@group(0) @binding(3) var<storage, read> refB: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> disp: array<u32>;
@group(0) @binding(5) var<storage, read_write> hdr: array<vec4f>;
${common}
fn tone(x: vec3f) -> vec3f { let y = max(x, vec3f(0.0)) * frame.exposure; return y / (1.0 + y); }
fn srgb(c: vec3f) -> vec3f {
  return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, 12.92 * c, c <= vec3f(0.0031308));
}
fn heat(x: f32) -> vec3f {
  let t = clamp(x, 0.0, 1.0);
  return clamp(vec3f(1.5 * t, 1.5 * t - 0.5, 3.0 * t - 2.0) + vec3f(0.0, 0.0, 0.25 * (1.0 - t) * t * 4.0), vec3f(0.0), vec3f(1.0));
}
// Network output of light slot "slot" at pixel p. A preview at stride s > 1 holds every s-th pixel
// (compact, ceil(W/s) columns): interpolate its 4 nearest samples bilinearly, down-weighting samples
// whose surface point is far from this pixel's (joint bilateral upsampling), so light does not
// bleed across object edges.
fn netAt(slot: u32, s: u32, p: u32) -> vec3f {
  let base = slot * NP;
  if (s <= 1u) { return max(outB[base + p].xyz, vec3f(0.0)); }
  let cw = (W + s - 1u) / s; let ch = (H + s - 1u) / s;
  let fx = f32(p % W) / f32(s); let fy = f32(p / W) / f32(s);
  let x0 = min(u32(fx), cw - 1u); let y0 = min(u32(fy), ch - 1u);
  let x1 = min(x0 + 1u, cw - 1u); let y1 = min(y0 + 1u, ch - 1u);
  let tx = fx - f32(x0); let ty = fy - f32(y0);
  let g = pgeo[2u * p];
  let sig = 1.5 * f32(s) * 2.0 * frame.tanxy.x / f32(W) * max(g.w, 1e-3);  // ~1.5 sample spacings
  var acc = vec3f(0.0); var ws = 0.0;
  for (var k = 0u; k < 4u; k++) {
    let cx = select(x0, x1, (k & 1u) == 1u); let cy = select(y0, y1, k >= 2u);
    let wb = select(1.0 - tx, tx, (k & 1u) == 1u) * select(1.0 - ty, ty, k >= 2u);
    let q = pgeo[2u * (cy * s * W + cx * s)];
    let dq = q.xyz - g.xyz;
    var wg = exp(-dot(dq, dq) / (2.0 * sig * sig));
    if ((q.w > 0.0) != (g.w > 0.0)) { wg = 0.0; }
    let w = wb * (wg + 1e-4);
    acc += w * max(outB[base + cy * cw + cx].xyz, vec3f(0.0));
    ws += w;
  }
  return acc / max(ws, 1e-12);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let p = gid.x;
  if (p >= NP) { return; }
  var I = vec3f(0.0);
  for (var l = 0u; l < frame.nLights; l++) {
    let L = frame.lights[l];
    if (L.info.y == 0u) { continue; }
    let nrp = netAt(L.info.x, L.info.z, p);
    I += L.e.xyz * (nrp + directCov(p, l));
  }
  hdr[p] = vec4f(I, 0.0);
  var c = tone(I);
  if (frame.mode >= 1u) {
    let R = frame.lights[frame.refLight].e.xyz * refB[p].xyz;
    if (frame.mode == 1u) { c = tone(R); }
    else { c = heat(length(tone(I) - tone(R)) * 8.0); }
  }
  let s = clamp(srgb(c), vec3f(0.0), vec3f(1.0));
  let q = vec3u(round(s * 255.0));
  disp[p] = q.x | (q.y << 8u) | (q.z << 16u) | (255u << 24u);
}
`;

  const blit = /* wgsl */ `
@group(0) @binding(0) var<storage, read> disp: array<u32>;
struct VO { @builtin(position) pos: vec4f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VO; o.pos = vec4f(p[i], 0.0, 1.0); return o;
}
@fragment fn fs(v: VO) -> @location(0) vec4f {
  let x = min(u32(v.pos.x), ${u(W - 1)}); let y = min(u32(v.pos.y), ${u(H - 1)});
  let c = disp[y * ${u(W)} + x];
  return vec4f(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u), 255.0) / 255.0;
}
`;

  // ---------------------------------------------------------------- loss prep
  // Per sampled pixel: total image I = sum_l E_l (N_l + D_l), Reinhard-tonemapped
  // loss against the target, and dL/dI. Also stores D_l and the direct term's gradient.
  const lossPrep = /* wgsl */ `
struct OptU { K: u32, nTiles: u32, _a: u32, _b: u32, sc: vec4f };
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> ou: OptU;
@group(0) @binding(2) var<storage, read> subIdx: array<u32>;
@group(0) @binding(3) var<storage, read> subOut: array<vec4f>;
@group(0) @binding(4) var<storage, read> tgtB: array<vec4f>;
@group(0) @binding(5) var<storage, read> pgeo: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> dLdI: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> subDir: array<vec4f>;  // [2*(l*K+i)] = (D,..), [+1] = dL/dln_direct
${common}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= ou.K) { return; }
  let p = subIdx[i];
  var I = vec3f(0.0);
  var gs: array<vec4f, ${MAX_LIGHTS}>;
  for (var l = 0u; l < frame.nLights; l++) {
    let L = frame.lights[l];
    let ds = directSoft(p, l);
    gs[l] = ds.g;
    subDir[2u * (l * ou.K + i)] = vec4f(ds.D, 0.0, 0.0, 0.0);
    if (L.info.y == 0u) { continue; }
    I += L.e.xyz * (max(subOut[l * ou.K + i].xyz, vec3f(0.0)) + ds.D);
  }
  let e = frame.exposure;
  let x = max(I, vec3f(0.0)) * e;
  let tn = x / (1.0 + x);
  let tg = tgtB[i];
  let r = tn - tg.xyz;
  let g = 2.0 * tg.w * r * e / ((1.0 + x) * (1.0 + x));
  dLdI[i] = vec4f(g, tg.w * dot(r, r));
  for (var l = 0u; l < frame.nLights; l++) {
    subDir[2u * (l * ou.K + i) + 1u] = gs[l] * dot(g, frame.lights[l].e.xyz) * ou.sc;
  }
}
`;

  // ---------------------------------------------------------------- gradient
  // One workgroup = one tile of sampled pixels for one light (wg.y). Forward (storing
  // activations), then backprop. Per-tile partial sums (PS floats):
  //   [0, WD)       sum over pixels of dL/d(layer-0 pre-activation)  -> light columns on the CPU
  //   [WD, WD+4)    dL/dln from the geo features and the direct term
  //   [WD+4, WD+7)  dL/dE,  [WD+7] loss
  const grad = /* wgsl */ `
${mlpDecls}
${structs}
struct OptU { K: u32, nTiles: u32, _a: u32, _b: u32, sc: vec4f };
@group(1) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(1) var<uniform> ou: OptU;
@group(1) @binding(2) var<storage, read> dLdI: array<vec4f>;
@group(1) @binding(3) var<storage, read> subOut: array<vec4f>;
@group(1) @binding(4) var<storage, read> subDir: array<vec4f>;
@group(1) @binding(5) var<storage, read_write> scr: array<vec4f>;
@group(1) @binding(6) var<storage, read_write> part: array<f32>;
var<workgroup> gout: array<vec4f, ${2 * TP}>;   // per pixel: dL/d(out rows 0..2), dL/d(out rows 3..5)
var<workgroup> gG: array<f32, ${TP}>;           // per pixel: dL/dG (mul head)
var<workgroup> red: array<f32, ${TP * 8}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let base = wg.x * ${u(TP)};
  let n = ou.K;
  let li = wg.y;
  let sbase = ((li * ou.nTiles + wg.x) * ${u(NH + 1)}) * ${u((TP * WD) / 4)};
${mlpBody(true)}
  // dL/d(output rows) for this light, and the per-pixel colour-gradient / loss terms.
  if (t < ${u(TP)}) {
    let i = base + t;
    var ga = vec4f(0.0); var gb = vec4f(0.0);
    var gg = 0.0;
    for (var j = 0u; j < 8u; j++) { red[t * 8u + j] = 0.0; }
    if (i < n) {
      let dI = dLdI[i];
      let raw = subOut[li * n + i].xyz;
      let E = frame.lights[li].e.xyz;
      let go = dI.xyz * E * select(vec3f(0.0), vec3f(1.0), raw > vec3f(0.0));
      ${mul ? `let G = pf[2u * t + 1u].z;
      ga = vec4f(go * G, 0.0);
      gb = vec4f(go, 0.0);
      gg = dot(go, vec3f(outRow(0u, t), outRow(1u, t), outRow(2u, t)));` : "ga = vec4f(go, 0.0);"}
      let contrib = max(raw, vec3f(0.0)) + subDir[2u * (li * n + i)].x;
      red[t * 8u + 4u] = dI.x * contrib.x;
      red[t * 8u + 5u] = dI.y * contrib.y;
      red[t * 8u + 6u] = dI.z * contrib.z;
      red[t * 8u + 7u] = dI.w;
    }
    gout[2u * t] = ga; gout[2u * t + 1u] = gb; gG[t] = gg;
  }
  workgroupBarrier();
  // Back through the output layer: g[k] = sum_o Wo[o][k] gout[o], masked by a_NH > 0.
  let topL = ${u(NH)} * ${u((TP * WD) / 4)};
  var gr: array<vec4f, 4>;
  for (var q = 0u; q < 4u; q++) {
    let pl = pg * 4u + q;
    let ga = gout[2u * pl];
    var s = Wv[${u(off.WO / 4)} + cg] * ga.x
          + Wv[${u((off.WO + WD) / 4)} + cg] * ga.y
          + Wv[${u((off.WO + 2 * WD) / 4)} + cg] * ga.z;
    ${mul ? `let gb = gout[2u * pl + 1u];
    s += Wv[${u((off.WO + 3 * WD) / 4)} + cg] * gb.x
       + Wv[${u((off.WO + 4 * WD) / 4)} + cg] * gb.y
       + Wv[${u((off.WO + 5 * WD) / 4)} + cg] * gb.z;` : ""}
    let a = scr[sbase + topL + (pl * ${u(WD)} + cg * 4u) / 4u];
    gr[q] = select(vec4f(0.0), s, a > vec4f(0.0));
  }
  workgroupBarrier();
  for (var q = 0u; q < 4u; q++) {
    let a = (pg * 4u + q) * ${u(WD)} + cg * 4u;
    act[a] = gr[q].x; act[a + 1u] = gr[q].y; act[a + 2u] = gr[q].z; act[a + 3u] = gr[q].w;
  }
  workgroupBarrier();
  // Hidden layers, last to first: g_in[k] = sum_c W[c][k] g[c], masked by a_l > 0.
  for (var s = 0u; s < ${u(NH)}; s++) {
    let l = ${u(NH - 1)} - s;
    let wo = ${u(off.HIDB / 4)} + l * ${u((WD * WD) / 4)};
    var a0 = vec4f(0.0); var a1 = vec4f(0.0); var a2 = vec4f(0.0); var a3 = vec4f(0.0);
    let r0 = (pg * 4u) * ${u(WD)};
    for (var c = 0u; c < ${u(WD)}; c++) {
      let w = Wv[wo + c * ${u(CG)} + cg];
      a0 += act[r0 + c] * w;
      a1 += act[r0 + ${u(WD)} + c] * w;
      a2 += act[r0 + ${u(2 * WD)} + c] * w;
      a3 += act[r0 + ${u(3 * WD)} + c] * w;
    }
    var res = array<vec4f, 4>(a0, a1, a2, a3);
    for (var q = 0u; q < 4u; q++) {
      let a = scr[sbase + l * ${u((TP * WD) / 4)} + ((pg * 4u + q) * ${u(WD)} + cg * 4u) / 4u];
      res[q] = select(vec4f(0.0), res[q], a > vec4f(0.0));
    }
    workgroupBarrier();
    for (var q = 0u; q < 4u; q++) {
      let a = (pg * 4u + q) * ${u(WD)} + cg * 4u;
      act[a] = res[q].x; act[a + 1u] = res[q].y; act[a + 2u] = res[q].z; act[a + 3u] = res[q].w;
    }
    workgroupBarrier();
  }
  // act now holds g0 = dL/d(layer-0 pre-activation) per pixel.
  let pbase = (li * ou.nTiles + wg.x) * ${u(PS)};
  // (1) sum over the tile's pixels, per channel (the light columns are applied on the CPU)
  if (t < ${u(WD)}) {
    var s = 0.0;
    for (var pl = 0u; pl < ${u(TP)}; pl++) { s += act[pl * ${u(WD)} + t]; }
    part[pbase + t] = s;
  }
  // (2) per pixel: geo features / G and the direct term -> dL/dln
  if (t < ${u(TP)}) {
    let i = base + t;
    var dl = vec4f(0.0);
    if (i < n) {
      ${geo || mul ? `var gf = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
      ${geo ? `for (var j = 0u; j < 6u; j++) {
        var s = 0.0;
        let row = ${u(off.W0G)} + j * ${u(WD)};
        for (var c = 0u; c < ${u(WD)}; c++) { s += wf(row + c) * act[t * ${u(WD)} + c]; }
        gf[j] = s;
      }` : ""}
      let p = pixelOf(i);
      let dcr = geoBackward(pgeo[2u * p], pgeo[2u * p + 1u].xyz, lc,
                            vec3f(gf[0], gf[1], gf[2]), gf[3], gf[4], gf[5], gG[t]);
      dl = dcr * ou.sc;` : ""}
      dl += subDir[2u * (li * n + i) + 1u];
    }
    red[t * 8u + 0u] = dl.x; red[t * 8u + 1u] = dl.y; red[t * 8u + 2u] = dl.z; red[t * 8u + 3u] = dl.w;
  }
  workgroupBarrier();
  if (t < 8u) {
    var s = 0.0;
    for (var pl = 0u; pl < ${u(TP)}; pl++) { s += red[pl * 8u + t]; }
    part[pbase + ${u(WD)} + t] = s;
  }
}
`;

  return { precompute, forward, fastForward, XW, composite, blit, lossPrep, grad, TP, CG, IN, PIXIN, JOB, PS, NO };
}
