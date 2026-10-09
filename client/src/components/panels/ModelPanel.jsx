import { useRelighter } from '../../relight/useRelighter.js'
import { AUTO } from '../../relight/relighter.js'
import { previewStride, refineStride } from '../../relight/quality.js'
import { fmt } from '../../relight/color.js'
import BrowserSupport from './BrowserSupport.jsx'

const ordinal = (n) => `${n}${n === 2 ? 'nd' : n === 3 ? 'rd' : 'th'}`
const every = (s) => (s <= 1 ? 'every pixel' : `every ${ordinal(s)} pixel`)
// WebGPU: '<threads>x<pixels>', WebGL: 'o<output groups a pass>'
function kernelText(name) {
  const gpu = /^(\d+)x(\d+)$/.exec(name), gl = /^o(\d+)$/.exec(name)
  if (gpu) return `${gpu[1]} threads a workgroup, ${gpu[2]} pixels each`
  if (gl) return `${gl[1]} output group${gl[1] === '1' ? '' : 's'} a pass`
  return name
}

function Table({ rows }) {
  return (
    <table className="w-full border-collapse text-xs">
      <tbody>
        {rows.map(([k, v, note]) => (
          <tr key={k} className="border-b border-line align-top">
            <td className="py-0.5 pr-2 text-dim">{k}</td>
            <td className="py-0.5 text-right">
              {v}
              {note && <div className="text-dim">{note}</div>}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export default function ModelPanel() {
  const r = useRelighter()
  const { engine, scene } = r
  const n = scene.network, t = scene.train
  const last = t.log?.[0]
  return (
    <>
      <h3 className="mb-1 font-bold">Network</h3>
      <Table
        rows={[
          ['MLP', `${n.width} × ${n.hidden}`],
          ['pixel encoding', `${n.grid_res.length} grids × ${n.feats} features, ${n.grid_res[0]}²–${n.grid_res.at(-1)}²`],
          ['geometric light inputs', n.geo ? 'yes' : 'no'],
          ['output head', n.head === 'mul' ? 'multiplicative (a·G + b)' : 'linear'],
          ['parameters', engine.paramCount.toLocaleString()],
          ['weights', `${fmt(engine.modelBytes / 1e6, 1)} MB`],
        ]}
      />
      <h3 className="mt-4 mb-1 font-bold">Training</h3>
      <Table
        rows={[
          ['iterations', t.iters.toLocaleString()],
          ['path samples', `${t.spp} spp, up to ${t.max_seg} segments`],
          ...(last
            ? [
                ['final loss (rel. MSE)', last.loss.toExponential(2)],
                ['PSNR vs. denoised target', `${fmt(last.psnr_den, 1)} dB`],
                ['training time', `${fmt(last.time / 60, 0)} min`],
              ]
            : []),
        ]}
      />
      <h3 className="mt-4 mb-1 font-bold">Renderer</h3>
      <Renderer r={r} />
      <BrowserSupport />
    </>
  )
}

/** How the viewer runs on this device, live: what quality.js has settled on so far. */
function Renderer({ r }) {
  const { engine, quality } = r
  const cost = engine.evalCost(1), budget = quality.budget
  const { frames, late } = quality.stats
  const gpu = engine.backend === 'WebGPU'
  const rows = [
    ['backend', gpu ? 'WebGPU (compute shaders)' : 'WebGL2 (fragment passes)', !gpu && r.notGPU ? `as ${r.notGPU}` : null],
    ['precision', engine.precision],
    ['image', `${r.W} × ${r.H}${AUTO ? ', sized to this GPU' : ''}`,
      r.swapping ? `loading ${r.swapping.name} at ${r.swapping.res ?? 'its native size'} on ${r.swapping.backend}` : null],
    ['GPU', r.stats.gpu],
    ['kernel', `${kernelText(engine.kernelName)}${engine.tuned ? '' : ' (untuned)'}`],
    ['one light, every pixel', cost === null ? 'measuring…' : `${fmt(cost, 1)} ms`, engine.timing.seeded ? 'from an earlier visit' : null],
    ['network budget', `${fmt(budget, 1)} ms a frame`],
    ['moving light', every(previewStride(engine, budget))],
    ['resting light', `refined to ${every(refineStride(engine, budget))}`],
    ['display', `${engine.canvas.width} × ${engine.canvas.height} px, Catmull-Rom upscaled`],
    ['frames held 30 fps', frames ? `${fmt(100 - (100 * late) / frames, 1)} %` : '–'],
    ['settings', r.fromProfile ? 'tuned on an earlier visit' : 'being tuned on this visit'],
  ]
  return <Table rows={rows} />
}
