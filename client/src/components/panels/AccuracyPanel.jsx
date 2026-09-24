import { useRelighter } from '../../relight/useRelighter.js'
import { fmt } from '../../relight/color.js'
import { Button, Segmented } from '../ui.jsx'

const MODES = [
  { value: 0, label: 'Neural proxy' },
  { value: 1, label: 'Reference' },
  { value: 2, label: 'Error ×8' },
]

function Metric({ metric }) {
  if (!metric) return null
  if (metric.message) return <p className="text-xs text-dim">{metric.message}</p>
  const { psnr, mae, light: L } = metric
  return (
    <div className="space-y-0.5 text-xs text-dim">
      <p><b className="text-lg font-normal text-fg">{fmt(psnr, 2)} dB</b> PSNR (Reinhard-tonemapped)</p>
      <p>relative MAE {fmt(mae, 1)} %</p>
      <p>light at ({L.slice(0, 3).map((x) => fmt(x)).join(', ')}), r = {fmt(L[3], 3)}</p>
      <p>Move the light to leave the reference setup.</p>
    </div>
  )
}

export default function AccuracyPanel() {
  const r = useRelighter()
  const refs = r.scene.refs
  const { refIndex } = r.state
  if (!refs.length) {
    return <p className="text-dim">No reference renders at this resolution. Switch to the native resolution to compare against them.</p>
  }
  return (
    <>
      <p className="mb-3 text-dim">
        Held-out test lights, path traced and denoised. Pick one to place it and compare the proxy against
        the reference.
      </p>
      <div className="mb-3 grid grid-cols-3 gap-1">
        {refs.map((ref, i) => (
          <Button key={i} small variant={i === refIndex ? 'active' : 'default'} className="flex flex-col" onClick={() => r.selectRef(i)}>
            test {i + 1}
            <span className="text-dim">r {fmt(ref.light[3])}</span>
          </Button>
        ))}
      </div>
      <Segmented className="mb-3" options={MODES} value={r.engine.mode} onChange={(m) => r.requestViewMode(m)} />
      <Metric metric={r.metric} />
      {refIndex >= 0 && (
        <Button small variant="ghost" className="mt-3" onClick={() => r.restoreMyLights()}>Restore my lights</Button>
      )}
    </>
  )
}
