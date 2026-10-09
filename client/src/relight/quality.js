// Fits the viewer to the device: how long the network may run each frame (its budget), and so how
// coarse moving lights are previewed and how far resting ones are refined, plus which image size
// and network to run. Quality goes up as long as the page holds 30 fps, and what was learned is
// kept per GPU for the next visit. Backported from the portfolio's relit-room backdrop.

export const MIN_FPS = 30;
// ms. The slack is for frames that vsync rounds just past 33 ms (5 refreshes at 144 Hz, 9 at 240 Hz).
const LATE_SLACK = 5;
// s: a longer gap is a hidden tab or a stall, not a frame
const GAP = 0.25;
// frames after the network ran that count as its work
const WORK_FRAMES = 2;
const RATE_ALPHA = 0.05;
// how much more often frames with network work must run late than frames without it to blame it
const BLAME = 0.03;

// The budget backs off on a late frame and caps itself at 90% of where it failed, then creeps back
// up, so it settles just under what the GPU can take instead of missing a frame every second.
const BUDGET_MIN = 1;
const BUDGET_SHARE = 0.7;
const START_BUDGET = 7;
const GROW = 1.02;
const BACK_OFF = 0.75;
const CREEP = 1.0005;

// A resting light is refined as finely as ~REFINE_FRAMES frames of budget allow: slow GPUs stop at a
// softer image instead of stalling for seconds.
export const REFINE_FRAMES = 40;
// A larger image size is fetched if the GPU could refine a light at it within this many frames.
const UPGRADE_FRAMES = 12;
// ms: the most a timing evaluation may take while the page is showing
const BENCH_MAX_MS = 30;

export class AdaptiveQuality {
  constructor({ budget = START_BUDGET, minFps = MIN_FPS } = {}) {
    this.lateMs = 1000 / minFps + LATE_SLACK;
    this.budgetMax = (1000 / minFps) * BUDGET_SHARE;
    this.frameNo = 0;
    this.lastWork = -Infinity;
    this.budget = clamp(Number.isFinite(budget) && budget > 0 ? budget : START_BUDGET, BUDGET_MIN, this.budgetMax);
    this.ceiling = Infinity;
    this.lateRate = { work: 0, idle: 0 };
    this.stats = { frames: 0, seconds: 0, late: 0 };
  }

  /**
   * Called every frame with its length (s). `settled` stays off while the scene loads (shader
   * compiles made frames late, and the budget went to the floor); `hold` skips the frame (loading
   * another size, timing kernels).
   */
  frame(delta, settled, hold = false) {
    this.frameNo++;
    if (!(delta > 0) || delta > GAP || hold) return;
    const recent = this.frameNo - this.lastWork <= WORK_FRAMES;
    const late = delta * 1000 > this.lateMs;
    this.stats.frames++;
    this.stats.seconds += delta;
    this.stats.late += late ? 1 : 0;
    const rates = this.lateRate, kind = recent ? "work" : "idle";
    if (settled) rates[kind] += RATE_ALPHA * ((late ? 1 : 0) - rates[kind]);
    // Late frames only count against the network when frames with its work run late more often than
    // frames without: on a slow CPU the page itself misses frames.
    if (recent && late && settled && rates.work > rates.idle + BLAME) {
      this.ceiling = this.budget * 0.9;
      this.budget *= BACK_OFF;
    } else if (recent && !late) {
      this.ceiling *= CREEP;
      this.budget = Math.min(this.ceiling, this.budget * GROW);
    }
    this.budget = clamp(this.budget, BUDGET_MIN, this.budgetMax);
  }

  /** Marks this frame as one in which the network ran. */
  worked() { this.lastWork = this.frameNo; }
}

/** Stride moving lights are previewed at: the finest whose evaluation fits `budget` ms. */
export function previewStride(engine, budget) {
  if (engine.evalCost(1) === null) return 4;
  return [1, 2, 4, 8].find((s) => engine.evalCost(s) <= budget) ?? 16;
}

/** Stride resting lights are refined to, a band a frame. */
export function refineStride(engine, budget) {
  if (engine.evalCost(1) === null) return 4;
  return [1, 2, 4].find((s) => engine.evalCost(s) <= budget * REFINE_FRAMES) ?? 8;
}

/**
 * The stride to time a light at while the lights rest, or null. A whole light at every 16th pixel
 * is so few pixels that the fixed cost of a pass dominates (an AMD 860M read 330 ms for a light that
 * takes 47 and stayed at every 16th pixel), so the finest stride the GPU can take is timed, until
 * costs come from one that fine.
 */
export function benchStride(engine) {
  const known = Object.keys(engine.timing.byStride).map(Number);
  if (!known.length) return 8;
  const finest = [1, 2, 4].find((s) => engine.evalCost(s) <= BENCH_MAX_MS) ?? 8;
  // costs carried over from before are replaced by this GPU's own
  if (engine.timing.seeded) return finest;
  return finest < Math.min(...known) ? finest : null;
}

// ---------------------------------------------------------------- image size and network

/** Whether the connection is slow or saving data; then nothing larger is fetched. */
export function slowConnection() {
  const c = navigator.connection;
  return !!c && (c.saveData || /(^|-)[23]g$/.test(c.effectiveType || ""));
}
export const compactDevice = () => matchMedia?.("(max-width: 768px), (pointer: coarse)").matches ?? false;

/**
 * The next larger image size to fetch, or null: if the GPU could refine a light at it within
 * UPGRADE_FRAMES frames of budget, and the screen shows the image larger than now (shownPx).
 */
export function upgradeTier(engine, tiers, budget, shownPx) {
  const next = tiers.find((t) => t > engine.W);
  const cost = engine.evalCost(1);
  if (!next || cost === null || engine.timing.seeded || slowConnection()) return null;
  // the largest size needs ~100 MB while it is prepared
  if (next > 512 && navigator.deviceMemory && navigator.deviceMemory < 4) return null;
  if (shownPx <= engine.W) return null;
  return cost * (next / engine.W) ** 2 <= budget * UPGRADE_FRAMES ? next : null;
}

/** The size the next visit loads: this one, or the one below if a resting light couldn't be refined fully. */
export function nextVisitTier(engine, tiers, budget) {
  const below = [...tiers].reverse().find((t) => t < engine.W);
  const cost = engine.evalCost(1);
  return below && cost !== null && cost > budget * REFINE_FRAMES ? below : engine.W;
}

// The small network (64 wide) is ~3.3x faster than the 128-wide one but ~2 dB worse against the
// path-traced references; the larger one wins down to every 4th pixel, so the small one is for GPUs
// that can't refine the larger one past that.
export const NETWORKS = ["128x4", "64x4"];
const SMALL_SPEEDUP = 3.3;

/** The network to change to, or null; with hysteresis, so a GPU near the line doesn't flip back and forth. */
export function switchNetwork(engine, budget) {
  const cost = engine.evalCost(1), i = NETWORKS.indexOf(networkKey(engine));
  if (cost === null || engine.timing.seeded || i < 0) return null;
  const refines = (full, room) => full / 16 <= budget * REFINE_FRAMES * room;
  if (i === 0 && !refines(cost, 1)) return NETWORKS[1];
  if (i === 1 && refines(cost * SMALL_SPEEDUP, 0.5)) return NETWORKS[0];
  return null;
}

export function networkKey(engine) {
  const net = engine.scene.network;
  return `${net.width}x${net.hidden}`;
}

// ---------------------------------------------------------------- per-GPU profile

const STORAGE_KEY = "relight-quality";
const MAX_AGE = 30 * 24 * 3600 * 1000;

/** The GPU's name, from a WebGL context (both backends name it the same way). */
export function gpuName() {
  try {
    const gl = document.createElement("canvas").getContext("webgl2") || document.createElement("canvas").getContext("webgl");
    if (!gl) return null;
    let name = String(gl.getParameter(gl.RENDERER));
    // Chrome and Safari say "WebKit WebGL" unless asked for the unmasked name; Firefox warns if asked.
    if (/^webkit/i.test(name)) {
      const ext = gl.getExtension("WEBGL_debug_renderer_info");
      if (ext) name = String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL));
    }
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return name;
  } catch { return null; }
}

/** What an engine measured, to seed another (another size of the same network on the same backend). */
export function measuredCosts(engine) {
  return { res: engine.W, network: networkKey(engine), backend: engine.backend, byStride: { ...engine.timing.byStride } };
}

/** Seeds an engine's costs from earlier ones, scaled by pixel count; only for the same network and backend. */
export function seedCosts(engine, measured) {
  if (!measured?.byStride || measured.network !== networkKey(engine) || measured.backend !== engine.backend || !(measured.res > 0)) return;
  const scale = (engine.W / measured.res) ** 2, byStride = {};
  for (const [s, ms] of Object.entries(measured.byStride)) byStride[s] = ms * scale;
  engine.seedTiming(byStride);
}

// localStorage can be missing or throw (private windows); then the defaults are used.
export function loadProfile(gpu) {
  try {
    const p = JSON.parse(localStorage.getItem(STORAGE_KEY));
    return p?.gpu === gpu && (p.fps ?? MIN_FPS) === MIN_FPS && Date.now() - p.at < MAX_AGE ? p : null;
  } catch { return null; }
}

export function saveProfile(profile) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...profile, at: Date.now() })); } catch { /* nowhere to keep it */ }
}

function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
