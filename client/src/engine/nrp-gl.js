// WebGL2 fallback runtime, for browsers or GPUs without WebGPU. Same scene files and the same
// interface as NRPEngine (nrp.js).
//
// WebGL2 has no compute shaders, so the MLP runs as a chain of fragment-shader passes over a band
// of pixels. Each pass writes OUT groups of 4 channels of the next layer (one render target per
// group, into layers of a texture array), reading every channel of the previous layer from the
// other texture array and the weights from a uniform buffer.
//
// The per-pixel data stays as it ships: the aux features as 16-bit floats (auxT, planes of 4
// channels) and positions as 16-bit integers (posT). An encoding pass turns them and the grid into
// the network's inputs (xT), and the shaders read position and normal from posT and auxT.
//
// Inverse lighting: colour and the direct-view term have exact gradients (computed on the CPU from
// the network outputs). For the 4 light inputs of the network, central finite differences are used,
// evaluated in one batch with the base value on the sampled pixels.
import { EngineBase, MAX_LIGHTS, SLOTS, SPARE, f16tab, chooseKernel, rememberKernel, yieldTask } from "./engine-base.js";

const MAXV = 9 * MAX_LIGHTS;  // variants per gradient batch: base, then ±h on each of 4 inputs, per light
const SUB_W = 256;            // item-grid width for pixel-subset evaluation
const SUB_K = 4096;           // max pixels per gradient step
// Target size of one activation texture array: 32 MB runs ~20% faster than 8, but phones don't
// have the memory.
const COMPACT = "(max-width: 768px), (pointer: coarse)";
const BAND_BYTES = matchMedia?.(COMPACT).matches ? 16 << 20 : 32 << 20;
// Output groups (of 4 channels) a pass of a layer writes, for displayed lights. Fewer mean more
// passes, each reading every input again, but fewer accumulators each: on an RTX 3080 one a pass
// runs the network about twice as fast as eight. Which is fastest depends on the GPU, so load()
// times them (tune) and remembers the result (chooseKernel); until then the default runs.
const OUTPUTS = [1, 2, 4, 8];
const DEFAULT_OUTPUTS = 2;
const outputsName = (n) => `o${n}`;
// ms after which a fence that hasn't passed stops being timed
const FENCE_GIVE_UP = 1000;

const VS = `#version 300 es
void main() {
  gl_Position = vec4(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0, 0.0, 1.0);
}`;

const HEAD = `#version 300 es
precision highp float; precision highp int;
precision highp sampler2D; precision highp sampler2DArray; precision highp usampler2D;
`;

// A pixel's surface position and camera distance (0 where it sees nothing), and its normal (aux
// features 3..5), from the 16-bit data.
const GEOM = `
uniform usampler2D posT; uniform sampler2DArray auxT;
uniform vec2 posMap; uniform vec3 camO;  // posMap: lowest position, step
vec4 geomAt(ivec2 px) {
  uvec4 q = texelFetch(posT, px, 0);
  if (q.x == 0u) return vec4(0.0);
  vec3 w = posMap.x + (vec3(q.xyz) - 1.0) * posMap.y;
  return vec4(w, length(w - camO));
}
vec3 normalAt(ivec2 px) {
  return vec3(texelFetch(auxT, ivec3(px, 0), 0).w, texelFetch(auxT, ivec3(px, 1), 0).xy);
}`;

/** Writes input groups k0 .. k0+n-1 of each pixel's network inputs: the grid encoding, then the aux features. */
function encodeFS({ W, H, levels, feats, gridRes, gridOff, auxDim, GW, k0, n }) {
  const ENC = levels * feats;
  const input = (i) => {
    if (i < ENC) {
      const l = Math.floor(i / feats);
      return `bil(${gridRes[l]}, ${gridOff[l]}, ${i % feats}, uv)`;
    }
    if (i < ENC + auxDim) {
      const j = i - ENC;
      return `texelFetch(auxT, ivec3(px, ${j >> 2}), 0).${"xyzw"[j & 3]}`;
    }
    return "0.0";
  };
  const J = [...Array(n).keys()];
  return HEAD + `
uniform sampler2D gridT; uniform sampler2DArray auxT;
${J.map((j) => `layout(location = ${j}) out vec4 o${j};`).join("\n")}
float gridAt(int i) { return texelFetch(gridT, ivec2(i % ${GW}, i / ${GW}), 0).x; }
float bil(int R, int off, int f, vec2 uv) {
  vec2 g = uv * float(R - 1);
  ivec2 c = min(ivec2(floor(g)), ivec2(R - 2));
  vec2 t = g - vec2(c);
  int i00 = off + (c.y * R + c.x) * ${feats} + f;
  int i10 = i00 + R * ${feats};
  float top = mix(gridAt(i00), gridAt(i00 + ${feats}), t.x);
  float bot = mix(gridAt(i10), gridAt(i10 + ${feats}), t.x);
  return mix(top, bot, t.y);
}
void main() {
  ivec2 px = ivec2(gl_FragCoord.xy);
  vec2 uv = (vec2(px) + 0.5) / vec2(${W}.0, ${H}.0);
  ${J.map((j) => `o${j} = vec4(${[0, 1, 2, 3].map((c) => input(4 * (k0 + j) + c)).join(", ")});`).join("\n  ")}
}`;
}

/**
 * One pass of a layer: `out` output groups from `kin` input groups.
 * stage "l0":  first layer. Pixel features come from `src` (xg groups), then the normalised light,
 *              then (geo) the 6 geometric features in 2 groups.
 *       "hid": hidden layer, activations from `src` at the band-local position.
 *       "out": output layer, activations at the band-local position, written at the item's
 *              absolute row. head "mul": out = h[0..2] * G + h[3..5].
 * items "full": a light's item grid at stride `stride`; the viewport's x is the item column and
 *              row (band-local) + yOff its row (light from `ln`); "sub": a pixel subset, rowsPerVar
 *              rows per light variant (light from lnv[variant]).
 */
function layerFS({ kin, out, relu, stage, items, W, xg = 0, geo = false, mul = false }) {
  const J = [...Array(out).keys()];
  const acc = (x, b) => J.map((j) => `a${j} += w[${b} + ${4 * j}] * ${x}.x + w[${b} + ${4 * j + 1}] * ${x}.y + ` +
    `w[${b} + ${4 * j + 2}] * ${x}.z + w[${b} + ${4 * j + 3}] * ${x}.w;`).join("\n    ");
  // Displayed lights with the 'mul' head write a and b of a * G + b as they are, into two targets;
  // the composite applies G per pixel, so a preview keeps the light's shape (see netAt there)
  const split = stage === "out" && mul && items === "full";
  const needItem = stage === "l0" || (stage === "out" && mul && !split);
  const nTex = stage === "l0" ? xg : kin;
  const item = {
    full: "ivec2 itemPixel(int x, int row, out vec4 L) { L = ln; return ivec2(x, row) * stride; }",
    sub: `ivec2 itemPixel(int x, int row, out vec4 L) {
  int v = row / rowsPerVar;
  int p = int(texelFetch(idxT, ivec2(x, row - v * rowsPerVar), 0).r);
  L = lnv[v];
  return ivec2(p % ${W}, p / ${W});
}`,
  }[items];
  const targets = split ? 2 : stage === "out" ? 1 : out;
  return HEAD + `
uniform sampler2DArray src;
layout(std140) uniform Wt { vec4 w[${kin * out * 4}]; };
uniform vec4 bias[${out}];
uniform int yOff;
${needItem && items === "full" ? "uniform vec4 ln; uniform int stride;" : ""}
${needItem && items === "sub" ? `uniform usampler2D idxT; uniform vec4 lnv[${MAXV}]; uniform int rowsPerVar;` : ""}
${needItem ? item : ""}
${geo || mul ? `${GEOM}
uniform vec3 boxLo, boxHi; uniform vec2 radRange;
// Geometric features of paper-external option "geo" and the irradiance factor G (nrp/model.py geo_features).
// The light is rebuilt from its normalised parameters, so finite differences also move these.
void geoFeatures(ivec2 px, vec4 L, out vec4 g0, out vec4 g1, out float G) {
  vec3 c = boxLo + (L.xyz + 1.0) * 0.5 * (boxHi - boxLo);
  float r = radRange.x + (L.w + 1.0) * 0.5 * (radRange.y - radRange.x);
  vec4 gm = geomAt(px);
  float valid = gm.w > 0.0 ? 1.0 : 0.0;
  vec3 v = c - gm.xyz;
  float d = max(length(v), 1e-4);
  vec3 l = v / d;
  float cosv = dot(normalAt(px), l);
  float s = min(r / d, 1.0);
  float omega = 6.283185307 * s * s / (1.0 + sqrt(max(1.0 - s * s, 0.0)));  // = 2 pi (1 - sqrt(1 - s^2)), without cancellation
  g0 = vec4(l, cosv) * valid;
  g1 = vec4(0.5 * log(d), 0.25 * log(omega + 1e-6), 0.0, 0.0) * valid;
  G = omega * max(cosv, 0.0) / 3.141592654 * valid;
}` : ""}
${[...Array(targets).keys()].map((j) => `layout(location = ${j}) out vec4 o${j};`).join("\n")}
void main() {
  ivec2 q = ivec2(gl_FragCoord.xy);
  ${stage === "l0" ? "vec4 L; ivec2 px = itemPixel(q.x, q.y + yOff, L);" : "ivec2 px = ivec2(q.x, q.y - yOff);"}
  ${stage === "out" && mul && !split ? "vec4 L; ivec2 ip = itemPixel(q.x, q.y, L);" : ""}
  ${J.map((j) => `vec4 a${j} = bias[${j}];`).join(" ")}
  for (int k = 0; k < ${nTex}; k++) {
    vec4 x = texelFetch(src, ivec3(px, k), 0);
    int b = k * ${out * 4};
    ${acc("x", "b")}
  }
  ${stage === "l0" ? acc("L", xg * out * 4) : ""}
  ${stage === "l0" && geo ? `vec4 g0, g1; float G;
  geoFeatures(px, L, g0, g1, G);
  ${acc("g0", (xg + 1) * out * 4)}
  ${acc("g1", (xg + 2) * out * 4)}` : ""}
  ${split ? "o0 = vec4(a0.xyz, 0.0); o1 = vec4(a0.w, a1.xy, 0.0);"
    : stage === "out" && mul ? `vec4 g0, g1; float G;
  geoFeatures(ip, L, g0, g1, G);
  o0 = vec4(a0.xyz * G + vec3(a0.w, a1.xy), 0.0);`
    : stage === "out" ? "o0 = a0;"
    : J.map((j) => `o${j} = ${relu ? `max(a${j}, vec4(0.0))` : `a${j}`};`).join("\n  ")}
}`;
}

// Port of `composite` in shaders.js. Writes the display colour and the HDR image.
const compositeFS = (W, H, mul) => HEAD + `
uniform sampler2DArray outT; uniform sampler2D refT;
${GEOM}
uniform vec3 camX, camY, camZ; uniform vec2 tanxy;
uniform float exposure; uniform int mode, nLights, refLight;
uniform vec4 lPR[${MAX_LIGHTS}]; uniform vec3 lE[${MAX_LIGHTS}]; uniform ivec3 lInfo[${MAX_LIGHTS}];  // slot, enabled, stride
layout(location = 0) out vec4 disp;
layout(location = 1) out vec4 hdr;
const float W = ${W}.0, H = ${H}.0;

vec3 camDir(vec2 uv) {
  return normalize(camX * ((0.5 - uv.x) * 2.0 * tanxy.x) + camY * ((0.5 - uv.y) * 2.0 * tanxy.y) + camZ);
}
float directCov(ivec2 pix, int l) {
  vec3 c = lPR[l].xyz; float r = lPR[l].w;
  vec4 g = geomAt(pix);
  float surf = g.w > 0.0 ? g.w : 1e9;
  vec3 oc = camO - c;
  float px = float(pix.x), py = float(pix.y);
  vec3 d0 = camDir(vec2((px + 0.5) / W, (py + 0.5) / H));
  float b0 = dot(d0, oc);
  float pixAng = 2.0 * tanxy.x / W * 1.5;
  float dist2 = dot(oc, oc) - b0 * b0;
  float slack = r + pixAng * length(oc);
  if (dist2 > slack * slack || b0 > 0.0) return 0.0;
  float cnt = 0.0;
  for (int sy = 0; sy < 4; sy++) {
    for (int sx = 0; sx < 4; sx++) {
      vec3 d = camDir(vec2((px + (float(sx) + 0.5) / 4.0) / W, (py + (float(sy) + 0.5) / 4.0) / H));
      float b = dot(d, oc);
      float disc = b * b - dot(oc, oc) + r * r;
      if (disc >= 0.0) {
        float t0 = -b - sqrt(disc);
        if (t0 > 0.0 && t0 < surf) cnt += 1.0;
      }
    }
  }
  return cnt / 16.0;
}
// A slot's output at item c: rgb (a), or with the 'mul' head a and b of a * G + b (two layers a slot)
void outAt(ivec2 c, int slot, out vec3 a, out vec3 b) {
  ${mul ? `a = texelFetch(outT, ivec3(c, 2 * slot), 0).xyz;
  b = texelFetch(outT, ivec3(c, 2 * slot + 1), 0).xyz;`
    : `a = max(texelFetch(outT, ivec3(c, slot), 0).xyz, 0.0);
  b = vec3(0.0);`}
}
// Light l's unshadowed irradiance factor G at pix (geo_features in nrp/model.py)
float irradiance(ivec2 pix, int l) {
  vec4 gm = geomAt(pix);
  if (gm.w <= 0.0) return 0.0;
  vec3 v = lPR[l].xyz - gm.xyz;
  float d = max(length(v), 1e-4);
  float cosv = dot(normalAt(pix), v / d);
  float s = min(lPR[l].w / d, 1.0);
  float omega = 6.283185307 * s * s / (1.0 + sqrt(max(1.0 - s * s, 0.0)));
  return omega * max(cosv, 0.0) / 3.141592654;
}
// Network output of light l (in its slot) at pix; a preview at stride s > 1 is upsampled like
// netAt in shaders.js, with a and b interpolated apart before G is applied at the pixel itself.
vec3 netAt(int l, int slot, int s, ivec2 pix) {
  vec3 a, b;
  if (s <= 1) {
    outAt(pix, slot, a, b);
  } else {
  ivec2 last = ivec2((${W} + s - 1) / s - 1, (${H} + s - 1) / s - 1);
  vec2 f = vec2(pix) / float(s);
  ivec2 c0 = min(ivec2(f), last), c1 = min(c0 + 1, last);
  vec2 t = f - vec2(c0);
  vec4 g = geomAt(pix);
  float sig = 1.5 * float(s) * 2.0 * tanxy.x / W * max(g.w, 1e-3);
  vec3 accA = vec3(0.0), accB = vec3(0.0); float ws = 0.0;
  for (int k = 0; k < 4; k++) {
    ivec2 c = ivec2((k & 1) == 1 ? c1.x : c0.x, k >= 2 ? c1.y : c0.y);
    float wb = ((k & 1) == 1 ? t.x : 1.0 - t.x) * (k >= 2 ? t.y : 1.0 - t.y);
    vec4 q = geomAt(c * s);
    vec3 dq = q.xyz - g.xyz;
    float wg = (q.w > 0.0) == (g.w > 0.0) ? exp(-dot(dq, dq) / (2.0 * sig * sig)) : 0.0;
    float w = wb * (wg + 1e-4);
    vec3 ca, cb;
    outAt(c, slot, ca, cb);
    accA += w * ca; accB += w * cb;
    ws += w;
  }
  a = accA / max(ws, 1e-12); b = accB / max(ws, 1e-12);
  }
  return ${mul ? "max(a * irradiance(pix, l) + b, 0.0)" : "a"};
}
vec3 tone(vec3 x) { vec3 y = max(x, 0.0) * exposure; return y / (1.0 + y); }
vec3 srgb(vec3 c) { return mix(1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, 12.92 * c, lessThanEqual(c, vec3(0.0031308))); }
vec3 heat(float x) {
  float t = clamp(x, 0.0, 1.0);
  return clamp(vec3(1.5 * t, 1.5 * t - 0.5, 3.0 * t - 2.0) + vec3(0.0, 0.0, 0.25 * (1.0 - t) * t * 4.0), 0.0, 1.0);
}
void main() {
  ivec2 pix = ivec2(gl_FragCoord.xy);
  vec3 I = vec3(0.0);
  for (int l = 0; l < ${MAX_LIGHTS}; l++) {
    if (l >= nLights) break;
    if (lInfo[l].y == 0) continue;
    vec3 nrp = netAt(l, lInfo[l].x, lInfo[l].z, pix);
    I += lE[l] * (nrp + directCov(pix, l));
  }
  hdr = vec4(I, 0.0);
  vec3 c = tone(I);
  if (mode >= 1) {
    vec3 R = lE[refLight] * texelFetch(refT, pix, 0).xyz;
    c = mode == 1 ? tone(R) : heat(length(tone(I) - tone(R)) * 8.0);
  }
  disp = vec4(clamp(srgb(c), 0.0, 1.0), 1.0);
}`;

/**
 * A layer's weights split into passes of `out` output groups. Uniform block of pass p:
 * w[(k * out + j) * 4 + c] = weights from input (group k, component c) to outputs 4(p*out + j) .. +3.
 * inCol(k, c) maps an input slot to a weight column, or -1 for padding.
 */
function packLayer(layer, nIn, kin, out, inCol) {
  const nOut = layer.shape[0], passes = Math.ceil(Math.ceil(nOut / 4) / out);
  return Array.from({ length: passes }, (_, p) => {
    const w = new Float32Array(kin * out * 16), b = new Float32Array(out * 4);
    for (let j = 0; j < out; j++) for (let r = 0; r < 4; r++) {
      const o = 4 * (p * out + j) + r;
      if (o >= nOut) continue;
      b[j * 4 + r] = layer.b[o];
      for (let k = 0; k < kin; k++) for (let c = 0; c < 4; c++) {
        const i = inCol(k, c);
        if (i >= 0) w[((k * out + j) * 4 + c) * 4 + r] = layer.w[o * nIn + i];
      }
    }
    return { w, b };
  });
}

export class GLEngine extends EngineBase {
  static async create(canvas) {
    const gl = canvas.getContext("webgl2", {
      antialias: false, depth: false, stencil: false, alpha: false, powerPreference: "high-performance",
    });
    if (!gl) throw new Error("WebGL2 is not available in this browser.");
    const f32 = !!gl.getExtension("EXT_color_buffer_float");
    if (!f32 && !gl.getExtension("EXT_color_buffer_half_float")) {
      throw new Error("This GPU cannot render to floating-point textures (EXT_color_buffer_float).");
    }
    const e = new GLEngine();
    e.gl = gl;
    e.canvas = canvas;
    e.backend = "WebGL2";
    e.f32 = f32;
    e.timer = gl.getExtension("EXT_disjoint_timer_query_webgl2");
    let renderer = gl.getParameter(gl.RENDERER);
    if (/^WebKit/.test(renderer)) {
      const dbg = gl.getExtension("WEBGL_debug_renderer_info");
      if (dbg) renderer = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL);
    }
    e.adapterInfo = { description: renderer };
    canvas.addEventListener("webglcontextlost", (ev) => { ev.preventDefault(); console.error("WebGL context lost"); });
    return e;
  }

  async load(base, onProgress = () => {}, res = null) {
    const gl = this.gl;
    const { scene, grid, gridOff, layers, aux, pos } = await this.fetchScene(base, onProgress, res);
    const { W, H } = this;
    const net = scene.network;
    const WD = net.width, NH = net.hidden - 1, G = WD / 4;
    // Optional inputs/heads (nrp/model.py); older exports lack the keys.
    const auxDim = net.aux_dim ?? 7, geo = !!net.geo, mul = net.head === "mul";
    if (net.light_grid) throw new Error("the WebGL2 viewer does not support light_grid models yet");
    const ENC = net.grid_res.length * net.feats, PIXIN = ENC + auxDim, XG = Math.ceil(PIXIN / 4);
    const IN = layers[0].shape[1];
    if (IN !== PIXIN + 4 + (geo ? 6 : 0)) throw new Error(`unexpected first-layer width ${IN}`);
    const outGroups = mul ? 2 : 1;
    this.NH = NH;
    this.K = SUB_K;
    // Finite-difference step in normalised light units; fp16 activations need a larger one.
    this.fdStep = this.f32 ? 2e-3 : 3e-2;
    this.resources = { textures: [], framebuffers: [], programs: [], buffers: [] };

    // --- output groups per pass: bounded by render targets, 64 B of targets per pixel, uniform block size
    const P = (p) => gl.getParameter(p);
    const maxBlock = P(gl.MAX_UNIFORM_BLOCK_SIZE), align = P(gl.UNIFORM_BUFFER_OFFSET_ALIGNMENT);
    const maxRT = Math.min(P(gl.MAX_DRAW_BUFFERS), P(gl.MAX_COLOR_ATTACHMENTS));
    const pickOut = (bytes) => {
      let o = Math.max(1, Math.min(maxRT, Math.floor(64 / bytes), Math.floor(maxBlock / (G * 64))));
      while (G % o) o--;
      return o;
    };

    // --- weights into one uniform buffer, one aligned block per (layer, pass)
    onProgress("packing weights");
    const blocks = [];
    let uboLen = 0;
    const place = ({ w, b }) => {
      const off = uboLen;
      uboLen += Math.ceil(w.byteLength / align) * align;
      blocks.push([off, w]);
      return { off, size: w.byteLength, b };
    };
    const packed = new Map();
    const netFor = (out) => {
      if (!packed.has(out)) {
        const l0 = packLayer(layers[0], IN, XG + 1 + (geo ? 2 : 0), out, (k, c) => {
          if (k < XG) return 4 * k + c < PIXIN ? 4 * k + c : -1;
          const i = PIXIN + 4 * (k - XG) + c;  // light (4), then geo (6)
          return i < IN ? i : -1;
        });
        const hid = layers.slice(1, NH + 1).map((L) => packLayer(L, WD, G, out, (k, c) => 4 * k + c));
        packed.set(out, [l0, ...hid].map((passes) => passes.map(place)));
      }
      return packed.get(out);
    };
    const F16 = { ifmt: gl.RGBA16F, bytes: 8 }, F32 = { ifmt: gl.RGBA32F, bytes: 16 };
    const subFmt = this.f32 ? F32 : F16;
    const outSub = pickOut(subFmt.bytes), wSub = netFor(outSub);
    // displayed lights: every output count that fits, packed now as the uniform buffer is built once
    this.outOptions = OUTPUTS.filter((n) => n <= maxRT && G % n === 0 && G * 64 * n <= maxBlock && n * F16.bytes <= 64);
    const wByOut = new Map(this.outOptions.map((n) => [n, netFor(n)]));
    this.wOut = place(packLayer(layers[NH + 1], WD, G, outGroups, (k, c) => 4 * k + c)[0]);
    const all = new Float32Array(uboLen / 4);
    for (const [off, w] of blocks) all.set(w, off / 4);
    this.ubo = gl.createBuffer();
    this.resources.buffers.push(this.ubo);
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.ubo);
    gl.bufferData(gl.UNIFORM_BUFFER, all, gl.STATIC_DRAW);

    // --- textures: the 16-bit pixel data as it ships, then the network's inputs encoded from it
    onProgress("pixel features");
    const A2 = gl.TEXTURE_2D_ARRAY, T2 = gl.TEXTURE_2D;
    const tex = (target, ifmt, w, h, n = 1) => {
      const t = gl.createTexture();
      this.resources.textures.push(t);
      gl.bindTexture(target, t);
      if (target === A2) gl.texStorage3D(target, 1, ifmt, w, h, n);
      else gl.texStorage2D(target, 1, ifmt, w, h);
      gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(target, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(target, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    };
    this.tex = tex;
    this.posMap = [this.posRange[0], this.posStep];
    this.auxTex = tex(A2, gl.RGBA16F, W, H, this.auxGroups);
    gl.texSubImage3D(A2, 0, 0, 0, 0, W, H, this.auxGroups, gl.RGBA, gl.HALF_FLOAT, aux);
    this.posTex = tex(T2, gl.RGBA16UI, W, H);
    gl.texSubImage2D(T2, 0, 0, 0, W, H, gl.RGBA_INTEGER, gl.UNSIGNED_SHORT, pos);
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    this.xTex = tex(A2, gl.RGBA16F, W, H, XG);
    this.encodeInputs({ grid, gridOff, XG, auxDim, maxRT });
    await yieldTask();

    this.refTex = tex(T2, gl.RGBA32F, W, H);
    this.idxTex = tex(T2, gl.R32UI, SUB_W, SUB_K / SUB_W);
    // a slot a layer, or two with the 'mul' head (a, then b)
    const outTex = tex(A2, gl.RGBA16F, W, H, SLOTS * (mul ? 2 : 1));
    this.outTex = outTex;
    this.outFbos = Array.from({ length: SLOTS }, (_, s) => this.fbo(mul ? [[outTex, 2 * s], [outTex, 2 * s + 1]] : [[outTex, s]]));
    const subOut = tex(T2, subFmt.ifmt, SUB_W, (SUB_K / SUB_W) * MAXV);
    this.subFbo = this.fbo([[subOut]]);
    const dispTex = tex(T2, gl.RGBA8, W, H), hdrTex = tex(T2, this.f32 ? gl.RGBA32F : gl.RGBA16F, W, H);
    this.dispTex = dispTex;
    this.compFbo = this.fbo([[dispTex], [hdrTex]]);
    this.dispFbo = this.fbo([[dispTex]]);
    this.hdrFbo = this.fbo([[hdrTex]]);

    // --- programs and activation buffers
    onProgress("compiling shaders");
    this.pComp = this.program(compositeFS(W, H, mul), "composite");
    const evaluator = (fmt, out, weights, gw, rows, items) => {
      const th = Math.max(8, Math.min(rows, Math.floor(BAND_BYTES / (gw * G * fmt.bytes))));
      const arrs = [0, 1].map(() => tex(A2, fmt.ifmt, gw, th, G));
      const fbos = arrs.map((t) => Array.from({ length: G / out }, (_, p) =>
        this.fbo(Array.from({ length: out }, (_, j) => [t, p * out + j]))));
      return {
        outN: out, gw, th, arrs, fbos, w: weights,
        l0: this.program(layerFS({ kin: XG + 1 + (geo ? 2 : 0), out, relu: true, stage: "l0", items, W, xg: XG, geo }), `layer 0 (${items})`),
        hid: this.program(layerFS({ kin: G, out, relu: true, stage: "hid", items, W }), "hidden layer"),
        out: this.program(layerFS({ kin: G, out: outGroups, relu: false, stage: "out", items, W, mul }), `output layer (${items})`),
        mul,
      };
    };
    this.sub = evaluator(subFmt, outSub, wSub, SUB_W, (SUB_K / SUB_W) * MAXV, "sub");
    this.makeFull = (out) => evaluator(F16, out, wByOut.get(out), W, H, "full");

    // remembered per GPU, network shape and image size (chooseKernel)
    this.kernelStore = `relight-kernel:webgl:${this.adapterInfo.description}:${W}x${H}:${WD}x${NH}`;
    const { name, forced } = chooseKernel(this.kernelStore);
    const known = this.outOptions.find((n) => outputsName(n) === name);
    this.setOutputs(known ?? this.outOptions.filter((n) => n <= DEFAULT_OUTPUTS).pop() ?? this.outOptions[0]);
    this.tuned = known !== undefined || forced || this.outOptions.length < 2;
    if (!this.tuned) {
      onProgress("timing shader passes");
      await this.tune();
      rememberKernel(this.kernelStore, this.kernelName);
    }
    return scene;
  }

  /** Encodes each pixel's network inputs (grid encoding, then aux features) into xTex, on the GPU. */
  encodeInputs({ grid, gridOff, XG, auxDim, maxRT }) {
    const gl = this.gl, { W, H } = this, net = this.scene.network;
    const GW = Math.min(4096, gl.getParameter(gl.MAX_TEXTURE_SIZE));
    const rows = Math.ceil(grid.length / GW);
    const padded = new Uint16Array(GW * rows);
    padded.set(grid);
    const gridTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, gridTex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R16F, GW, rows);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, GW, rows, gl.RED, gl.HALF_FLOAT, padded);
    gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D, gridTex);
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.auxTex);
    gl.viewport(0, 0, W, H);
    const shapes = { W, H, levels: net.grid_res.length, feats: net.feats, gridRes: net.grid_res, gridOff, auxDim, GW };
    const n = Math.min(maxRT, 8);
    for (let k0 = 0; k0 < XG; k0 += n) {
      const count = Math.min(n, XG - k0);
      const { p } = this.program(encodeFS({ ...shapes, k0, n: count }), "input encoding");
      const fbo = this.fbo(Array.from({ length: count }, (_, j) => [this.xTex, k0 + j]));
      gl.useProgram(p);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, fbo);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.deleteFramebuffer(fbo); this.resources.framebuffers.pop();
      gl.deleteProgram(p); this.resources.programs.pop();
    }
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
    gl.deleteTexture(gridTex);
  }

  dispose() {
    const gl = this.gl, pending = this.timing?.pending;
    if (pending?.query) gl.deleteQuery(pending.query);
    if (pending && !pending.query) {
      pending.cancelled = true;
      clearTimeout(pending.timer);
      [pending.before, pending.sync].forEach((s) => s && gl.deleteSync(s));
    }
    const r = this.resources;
    if (r) {
      r.textures.forEach((t) => gl.deleteTexture(t));
      r.framebuffers.forEach((f) => gl.deleteFramebuffer(f));
      r.programs.forEach((p) => gl.deleteProgram(p));
      r.buffers.forEach((b) => gl.deleteBuffer(b));
    }
    gl.deleteVertexArray(this.vao);
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  }

  // ------------------------------------------------------------------ display passes
  get kernelName() { return outputsName(this.full.outN); }
  get precision() { return "32-bit maths, 16-bit activations"; }

  /** Runs displayed lights with passes of `out` output groups. */
  setOutputs(out) {
    if (this.full?.outN === out) return;
    const old = this.full;
    this.full = this.makeFull(out);
    if (old) this.disposeEvaluator(old);
  }

  disposeEvaluator(E) {
    const gl = this.gl, r = this.resources;
    const drop = (list, items, del) => items.forEach((x) => { del(x); list.splice(list.indexOf(x), 1); });
    drop(r.textures, E.arrs, (t) => gl.deleteTexture(t));
    drop(r.framebuffers, E.fbos.flat(), (f) => gl.deleteFramebuffer(f));
    drop(r.programs, [E.l0, E.hid, E.out].map((P) => P.p), (p) => gl.deleteProgram(p));
  }

  /** Times each output count on a light in the middle of the box (in the spare slot, at `stride`), and keeps the fastest. */
  async tune(stride = 1, runs = 3) {
    const light = { pos: [0, 1, 2].map((i) => (this.lo[i] + this.hi[i]) / 2), radius: (this.rmin + this.rmax) / 2 };
    const times = {};
    let best = null;
    await this.untimed(async () => {
      for (const n of this.outOptions) {
        this.setOutputs(n);
        this.evaluate(light, SPARE, stride);   // the first run compiles the shaders
        await this.finish();
        const t0 = performance.now();
        for (let i = 0; i < runs; i++) this.evaluate(light, SPARE, stride);
        await this.finish();
        const ms = (performance.now() - t0) / runs;
        times[outputsName(n)] = ms;
        if (!best || ms < best.ms) best = { n, ms };
      }
    });
    this.setOutputs(best.n);
    this.kernelTimes = times;
    this.tuned = true;
    console.info("outputs per pass (ms per light):", times, "->", outputsName(best.n));
  }

  // ------------------------------------------------------------------ GL helpers
  program(fs, label) {
    const gl = this.gl;
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
    const vs = sh(gl.VERTEX_SHADER, VS), f = sh(gl.FRAGMENT_SHADER, fs);
    const p = gl.createProgram();
    gl.attachShader(p, vs); gl.attachShader(p, f);
    gl.linkProgram(p);
    const log = gl.getProgramParameter(p, gl.LINK_STATUS) ? null : gl.getShaderInfoLog(f) || gl.getShaderInfoLog(vs) || gl.getProgramInfoLog(p);
    gl.deleteShader(vs); gl.deleteShader(f);
    if (log !== null) { gl.deleteProgram(p); throw new Error(`${label} shader: ${log}`); }
    this.resources.programs.push(p);
    const u = {};
    for (let i = 0, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i < n; i++) {
      const name = gl.getActiveUniform(p, i).name;
      u[name.replace(/\[0\]$/, "")] = gl.getUniformLocation(p, name);
    }
    const bi = gl.getUniformBlockIndex(p, "Wt");
    if (bi !== gl.INVALID_INDEX) gl.uniformBlockBinding(p, bi, 0);
    gl.useProgram(p);
    for (const [name, unit] of [["src", 0], ["outT", 0], ["idxT", 1], ["posT", 2], ["refT", 3], ["auxT", 4], ["gridT", 6]]) {
      if (u[name]) gl.uniform1i(u[name], unit);
    }
    if (u.posMap) { gl.uniform2fv(u.posMap, this.posMap); gl.uniform3fv(u.camO, this.cam.O); }
    if (u.boxLo) {
      gl.uniform3fv(u.boxLo, this.lo); gl.uniform3fv(u.boxHi, this.hi);
      gl.uniform2f(u.radRange, this.rmin, this.rmax);
    }
    return { p, u };
  }

  /** Framebuffer over [texture, layer?] attachments, one draw buffer each. */
  fbo(attach) {
    const gl = this.gl, f = gl.createFramebuffer();
    this.resources.framebuffers.push(f);
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    attach.forEach(([t, layer], j) => (layer === undefined
      ? gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + j, gl.TEXTURE_2D, t, 0)
      : gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + j, t, 0, layer)));
    gl.drawBuffers(attach.map((_, j) => gl.COLOR_ATTACHMENT0 + j));
    const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (st !== gl.FRAMEBUFFER_COMPLETE) throw new Error(`framebuffer incomplete (0x${st.toString(16)})`);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return f;
  }

  /**
   * Resolves once the GPU has finished all commands issued so far. Polls through a MessageChannel,
   * which background tabs do not throttle the way they throttle timers; long waits back off to timers.
   */
  async finish() {
    const gl = this.gl, s = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0), t0 = performance.now();
    gl.flush();
    while (gl.clientWaitSync(s, 0, 0) === gl.TIMEOUT_EXPIRED && !gl.isContextLost()) {
      await (performance.now() - t0 < 200 ? yieldTask() : new Promise((r) => setTimeout(r, 4)));
    }
    gl.deleteSync(s);
  }

  /** Reads the first w x h RGBA texels of fbo's attachment 0 without stalling the page. */
  async read(fbo, w, h, float) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fbo);
    let type = gl.UNSIGNED_BYTE;
    if (float) type = !this.f32 && gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_TYPE) === gl.HALF_FLOAT ? gl.HALF_FLOAT : gl.FLOAT;
    const out = new ({ [gl.FLOAT]: Float32Array, [gl.HALF_FLOAT]: Uint16Array }[type] || Uint8Array)(w * h * 4);
    const pbo = gl.createBuffer();
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
    gl.bufferData(gl.PIXEL_PACK_BUFFER, out.byteLength, gl.STREAM_READ);
    gl.readPixels(0, 0, w, h, gl.RGBA, type, 0);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    await this.finish();
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
    gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, out);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    gl.deleteBuffer(pbo);
    return type === gl.HALF_FLOAT ? Float32Array.from(out, (x) => f16tab[x]) : out;
  }

  /**
   * Runs the MLP over item rows [r0, r1) and columns [c0, c1) of E's item grid, band by band.
   * setItems(uniforms) sets the per-item light inputs of the programs that read them (layer 0, and
   * the output layer of a "mul" head); bindTarget() binds the framebuffer that receives the output rows.
   */
  runNet(E, r0, r1, setItems, bindTarget, c0 = 0, c1 = E.gw) {
    const gl = this.gl, gw = c1 - c0, draw = (ch, P) => {
      gl.bindBufferRange(gl.UNIFORM_BUFFER, 0, this.ubo, ch.off, ch.size);
      gl.uniform4fv(P.u.bias, ch.b);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    };
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.posTex);
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.auxTex);
    for (let y0 = r0; y0 < r1; y0 += E.th) {
      const h = Math.min(E.th, r1 - y0);
      let cur = -1;
      for (let l = 0; l <= this.NH; l++) {
        const P = l === 0 ? E.l0 : E.hid, dst = l === 0 ? 0 : 1 - cur;
        gl.useProgram(P.p);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D_ARRAY, l === 0 ? this.xTex : E.arrs[cur]);
        gl.uniform1i(P.u.yOff, l === 0 ? y0 : 0);
        if (l === 0) setItems(P.u);
        gl.viewport(c0, 0, gw, h);
        E.w[l].forEach((ch, p) => { gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, E.fbos[dst][p]); draw(ch, P); });
        cur = dst;
      }
      const P = E.out;
      gl.useProgram(P.p);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, E.arrs[cur]);
      gl.uniform1i(P.u.yOff, y0);
      if (E.mul) setItems(P.u);
      bindTarget();
      gl.viewport(c0, y0, gw, h);
      draw(this.wOut, P);
    }
  }

  /**
   * Evaluates light l into output slot `slot` at stride s: the item rows [r0, r1) and columns
   * [c0, c1) of its ceil(W/s) x ceil(H/s) grid (by default all of them). Large enough evaluations
   * are timed on the GPU (see pollTiming).
   */
  evaluate(l, slot, s = 1, r0 = 0, r1 = this.rows(s), c0 = 0, c1 = this.cols(s)) {
    const gl = this.gl;
    r1 = Math.min(r1, this.rows(s)); c1 = Math.min(c1, this.cols(s));
    if (r1 <= r0 || c1 <= c0) return;
    const ln = this.normLight(l);
    const timed = this.timeable(s, r0, r1, c0, c1) && this.startTiming(s, this.share(s, r0, r1, c0, c1));
    this.runNet(this.full, r0, r1, (u) => { gl.uniform4fv(u.ln, ln); gl.uniform1i(u.stride, s); },
      () => gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.outFbos[slot]), c0, c1);
    if (timed) this.stopTiming();
  }

  // With timer queries the GPU times the passes itself. Without them (most phones), two fences
  // bracket the work and are watched about every ms: checking once a frame made every evaluation
  // read as at least a frame long.
  startTiming(stride, share) {
    const gl = this.gl;
    const pending = { stride, share, start: performance.now() };
    if (this.timer) {
      pending.query = gl.createQuery();
      gl.beginQuery(this.timer.TIME_ELAPSED_EXT, pending.query);
    } else {
      pending.before = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();
    }
    this.timing.pending = pending;
    return true;
  }

  stopTiming() {
    const gl = this.gl, pending = this.timing.pending;
    if (pending.query) { gl.endQuery(this.timer.TIME_ELAPSED_EXT); return; }
    pending.sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
    const passed = (sync) => gl.clientWaitSync(sync, 0, 0) !== gl.TIMEOUT_EXPIRED;
    const check = () => {
      pending.timer = null;
      if (pending.cancelled) return;
      const now = performance.now();
      if (pending.before && passed(pending.before)) {
        pending.beforeAt = now;
        gl.deleteSync(pending.before);
        pending.before = null;
      }
      if (!pending.before && passed(pending.sync)) {
        pending.ms = now - Math.max(pending.start, pending.beforeAt ?? pending.start);
      } else if (now - pending.start < FENCE_GIVE_UP) {
        pending.timer = setTimeout(check, 1);
        return;
      }
      if (pending.before) { gl.deleteSync(pending.before); pending.before = null; }
      gl.deleteSync(pending.sync);
      pending.sync = null;
      pending.done = true;
    };
    check();
  }

  /** Records the evaluation being timed once its reading is in; call once a frame. */
  pollTiming() {
    const gl = this.gl, pending = this.timing.pending;
    if (!pending?.stride) return;
    let ms = null;
    if (pending.query) {
      if (!gl.getQueryParameter(pending.query, gl.QUERY_RESULT_AVAILABLE)) return;
      if (!gl.getParameter(this.timer.GPU_DISJOINT_EXT)) ms = gl.getQueryParameter(pending.query, gl.QUERY_RESULT) / 1e6;
      gl.deleteQuery(pending.query);
    } else {
      if (!pending.done) return;
      ms = pending.ms ?? null;
    }
    this.timing.pending = null;
    this.recordTiming(pending.stride, pending.share, ms);
  }

  /**
   * Puts the lights' slots together into the image and shows it.
   * lights: [{pos, radius, color, intensity, enabled, slot, stride?}], pos and radius as evaluated.
   */
  composite(lights) {
    const gl = this.gl, { u, p } = this.pComp, { X, Y, Z, O, tx, ty } = this.cam, { W, H } = this;
    gl.useProgram(p);
    gl.uniform3fv(u.camO, O); gl.uniform3fv(u.camX, X); gl.uniform3fv(u.camY, Y); gl.uniform3fv(u.camZ, Z);
    gl.uniform2f(u.tanxy, tx, ty);
    gl.uniform1f(u.exposure, this.exposure);
    gl.uniform1i(u.mode, this.mode); gl.uniform1i(u.nLights, lights.length); gl.uniform1i(u.refLight, this.refLight);
    const pr = new Float32Array(MAX_LIGHTS * 4), E = new Float32Array(MAX_LIGHTS * 3), info = new Int32Array(MAX_LIGHTS * 3);
    lights.forEach((l, i) => {
      pr.set([...l.pos, l.radius], i * 4);
      E.set(l.color.map((c) => c * l.intensity), i * 3);
      info.set([l.slot, l.enabled ? 1 : 0, l.stride || 1], i * 3);
    });
    gl.uniform4fv(u.lPR, pr); gl.uniform3fv(u.lE, E); gl.uniform3iv(u.lInfo, info);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.outTex);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.posTex);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, this.refTex);
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.auxTex);
    gl.bindVertexArray(this.vao);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.compFbo);
    gl.viewport(0, 0, W, H);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.present();
  }

  /** Draws the last composited image to the canvas. */
  present() {
    const gl = this.gl, { W, H } = this;
    // Texture rows are image rows (top first); the canvas has y up, so flip while copying.
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.dispFbo);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
    gl.blitFramebuffer(0, 0, W, H, 0, H, W, 0, gl.COLOR_BUFFER_BIT, gl.NEAREST);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
  }

  /** Current display image as RGBA bytes. */
  async readDisplay() {
    return new Uint8ClampedArray((await this.read(this.dispFbo, this.W, this.H, false)).buffer);
  }

  async readHDR() {
    return this.read(this.hdrFbo, this.W, this.H, true);
  }

  async loadReference(i) {
    const r4 = await this.referenceData(i);
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.refTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.W, this.H, gl.RGBA, gl.FLOAT, r4);
    return r4;
  }

  // ------------------------------------------------------------------ gradients
  /**
   * Network outputs for every light-parameter variant on the pixels idx.
   * Returns N (rgba per item) and R (item rows per variant): item (v, i) is at
   * ((v * R + floor(i / SUB_W)) * SUB_W + i % SUB_W) * 4.
   */
  async evalSubset(variants, idx) {
    const gl = this.gl, R = Math.ceil(idx.length / SUB_W);
    const ids = new Uint32Array(R * SUB_W);
    ids.set(idx);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.idxTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, SUB_W, R, gl.RED_INTEGER, gl.UNSIGNED_INT, ids);
    const lnv = new Float32Array(MAXV * 4);
    variants.forEach((v, i) => lnv.set(v, i * 4));
    const rows = R * variants.length;
    this.runNet(this.sub, 0, rows, (u) => {
      gl.uniform4fv(u.lnv, lnv);
      gl.uniform1i(u.rowsPerVar, R);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.idxTex);
    }, () => gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.subFbo));
    return { N: await this.read(this.subFbo, SUB_W, rows, true), R };
  }

  /**
   * One stochastic gradient evaluation (paper Sec. 5.3, mini-batch over pixels); same contract as
   * NRPEngine.gradStep. idx: Uint32Array(K) pixel indices; tgt: Float32Array(K*4) tonemapped target
   * rgb + weight. Returns per light d loss / d(normalised params) and d loss / dE, plus the loss.
   */
  async gradStep(lights, idx, tgt) {
    const K = idx.length, nL = lights.length, h = this.fdStep;
    if (K > this.K) throw new Error(`at most ${this.K} pixels per step`);
    const variants = [];
    for (const l of lights) {
      const ln = this.normLight(l);
      variants.push(ln);
      for (let a = 0; a < 4; a++) for (const s of [h, -h]) variants.push(ln.map((x, j) => (j === a ? x + s : x)));
    }
    const { N, R } = await this.evalSubset(variants, idx);
    const at = (v, i) => ((v * R + Math.floor(i / SUB_W)) * SUB_W + (i % SUB_W)) * 4;

    const e = this.exposure;
    const E = lights.map((l) => l.color.map((c) => c * l.intensity));
    // world -> normalised light-parameter units for the direct-term gradient
    const sc = [0, 1, 2].map((i) => (this.hi[i] - this.lo[i]) / 2).concat([(this.rmax - this.rmin) / 2]);
    const res = lights.map(() => ({ dln: [0, 0, 0, 0], dE: [0, 0, 0] }));
    const ds = new Array(nL), I = [0, 0, 0], g = [0, 0, 0];
    let loss = 0;
    for (let i = 0; i < K; i++) {
      const w = tgt[i * 4 + 3];
      if (!w) continue;
      const p = idx[i];
      I.fill(0);
      for (let l = 0; l < nL; l++) {
        ds[l] = this.directSoft(p, lights[l]);
        if (!lights[l].enabled) continue;
        const o = at(l * 9, i);
        for (let k = 0; k < 3; k++) I[k] += E[l][k] * (Math.max(N[o + k], 0) + ds[l].D);
      }
      for (let k = 0; k < 3; k++) {
        const x = Math.max(I[k], 0) * e, r = x / (1 + x) - tgt[i * 4 + k];
        g[k] = (2 * w * r * e) / ((1 + x) * (1 + x));
        loss += w * r * r;
      }
      for (let l = 0; l < nL; l++) {
        const o = at(l * 9, i), El = E[l];
        const gE = g[0] * El[0] + g[1] * El[1] + g[2] * El[2];
        for (let k = 0; k < 3; k++) res[l].dE[k] += g[k] * (Math.max(N[o + k], 0) + ds[l].D);
        for (let a = 0; a < 4; a++) {
          const op = at(l * 9 + 1 + 2 * a, i), om = at(l * 9 + 2 + 2 * a, i);
          let s = 0;
          for (let k = 0; k < 3; k++) if (N[o + k] > 0) s += g[k] * El[k] * (N[op + k] - N[om + k]);
          res[l].dln[a] += s / (2 * h) + ds[l].g[a] * sc[a] * gE;
        }
      }
    }
    return { grads: res, loss };
  }
}
