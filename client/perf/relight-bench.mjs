// Times the network at each stride, image size and backend, and scores the test lights against
// their path-traced references (PSNR after Reinhard, as nrp/evaluate.py and the Accuracy tab do).
// Opens the viewer with ?bench (see src/relight/bench.js), so a dev or preview server must be up.
// Backported from the portfolio's perf/relight-bench.mjs.
//
//     npm install                                   (in this folder, once)
//     node relight-bench.mjs                        cornell, WebGPU and WebGL, native size
//     node relight-bench.mjs --scenes cornell,cornell-lite --sizes native,384,768 --tune
//     node relight-bench.mjs --gpu software         SwiftShader stands in for a weak GPU
import fs from 'node:fs'
import path from 'node:path'
import { parseArgs } from 'node:util'
import { chromium } from 'playwright'

const HERE = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'))
const RESULTS = path.join(HERE, 'results')

const { values: args } = parseArgs({
  options: {
    url: { type: 'string', default: 'http://localhost:5173' },
    scenes: { type: 'string', default: 'cornell' },
    backends: { type: 'string', default: 'webgpu,webgl' },
    sizes: { type: 'string', default: 'native' },
    strides: { type: 'string', default: '1,2,4,8' },
    gpu: { type: 'string', default: 'hardware' },
    tune: { type: 'boolean', default: false },
    'no-quality': { type: 'boolean', default: false },
    headed: { type: 'boolean', default: false },
    label: { type: 'string', default: '' },
  },
})
const list = (s) => s.split(',').filter(Boolean)
const strides = list(args.strides).map(Number).sort((a, b) => b - a)

// ms: batches are sized to take about this long; a light slower than MAX_EVAL_MS ends a size's strides
const BATCH_MS = 150
const MAX_EVAL_MS = 3000

const flags = [
  '--enable-unsafe-webgpu',
  ...(args.gpu === 'software' ? ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] : ['--ignore-gpu-blocklist']),
]
const browser = await chromium
  .launch({ channel: 'chrome', headless: !args.headed, args: flags })
  .catch(() => chromium.launch({ headless: !args.headed, args: flags }))
const results = []

for (const backend of list(args.backends)) {
  const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage()
  await page.goto(`${args.url}/?bench`, { waitUntil: 'load', timeout: 120000 })
  await page.waitForFunction(() => window.__relightBench, null, { timeout: 120000 })
  const bench = (fn, o) => page.evaluate(([f, a]) => window.__relightBench[f](a), [fn, o])

  for (const scene of list(args.scenes)) {
    const entry = { backend, scene, sizes: {}, quality: null, errors: [] }
    results.push(entry)
    for (const size of list(args.sizes)) {
      const res = size === 'native' ? null : Number(size)
      const t0 = Date.now()
      let info
      try {
        info = await bench('open', { scene, res, backend })
      } catch (error) {
        const msg = error.message.split('\n')[0]
        entry.errors.push(`${size}: ${msg}`)
        console.log(`${scene} ${backend} ${size}: ${msg}`)
        continue
      }
      if (args.tune) {
        const probe = await bench('time', { stride: 4, n: 1, batches: 1 })
        const stride = [1, 2, 4, 8].find((s) => probe.ms * (4 / s) ** 2 <= 30) ?? 8
        info.kernel = (await bench('tune', { stride, runs: 4 })).kernel
      }
      const row = { ...info, buildMs: Date.now() - t0, strides: {} }
      entry.sizes[size] = row
      for (const stride of strides) {
        const probe = await bench('time', { stride, n: 1, batches: 1 })
        if (probe.ms > MAX_EVAL_MS) {
          row.strides[stride] = { ms: probe.ms, items: probe.items, probeOnly: true }
          break
        }
        const n = Math.max(1, Math.min(16, Math.round(BATCH_MS / Math.max(probe.ms, 0.1))))
        const t = await bench('time', { stride, n, batches: 3 })
        row.strides[stride] = { ms: t.ms, items: t.items, nsPerItem: (t.ms * 1e6) / t.items }
      }
      const line = Object.entries(row.strides).sort((a, b) => a[0] - b[0]).map(([s, v]) => `s${s} ${v.ms.toFixed(2)} ms`).join('  ')
      console.log(`${scene} ${info.backend}${info.half ? ' f16' : ''} ${info.size}px ${info.network} ${info.kernel}: ${line}`)

      // the references are at the native size
      if (!args['no-quality'] && res === null && info.refs) {
        entry.quality = {}
        for (const stride of strides) {
          if (row.strides[stride]?.probeOnly) continue
          const q = await bench('score', { stride })
          entry.quality[stride] = q
          console.log(`  quality s${stride}: ${q.mean.toFixed(2)} dB vs path-traced  [${q.psnr.map((x) => x.toFixed(1)).join(' ')}]`)
        }
      }
    }
  }
  await bench('close')
  await page.context().close()
}
await browser.close()

fs.mkdirSync(RESULTS, { recursive: true })
const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
const file = path.join(RESULTS, `relight-${stamp}${args.label ? `-${args.label}` : ''}.json`)
fs.writeFileSync(file, JSON.stringify({ date: new Date().toISOString(), args, results }, null, 1))
console.log(`\nSaved ${path.relative(process.cwd(), file)}`)
