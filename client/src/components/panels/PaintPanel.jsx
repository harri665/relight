import { useEffect, useRef } from 'react'
import { useRelighter } from '../../relight/useRelighter.js'
import { BRUSH_SWATCHES, fmt } from '../../relight/color.js'
import { Button, Field, Range, SectionTitle, Swatches } from '../ui.jsx'

function LossPlot({ history, version }) {
  const ref = useRef(null)
  useEffect(() => {
    const c = ref.current, x = c.getContext('2d')
    x.clearRect(0, 0, c.width, c.height)
    if (history.length < 2) return
    const lg = history.map((v) => Math.log10(Math.max(v, 1e-12)))
    const mn = Math.min(...lg), mx = Math.max(...lg), pad = 10
    x.strokeStyle = '#2563eb'; x.lineWidth = 2; x.beginPath()
    lg.forEach((v, i) => {
      const px = pad + ((c.width - 2 * pad) * i) / (lg.length - 1)
      const py = pad + (c.height - 2 * pad) * (1 - (v - mn) / Math.max(1e-6, mx - mn))
      if (i) x.lineTo(px, py)
      else x.moveTo(px, py)
    })
    x.stroke()
    x.fillStyle = '#6b7280'; x.font = '20px ui-monospace, monospace'
    x.fillText('loss (log)', pad + 4, c.height - pad - 4)
  }, [history, history.length, version])
  return <canvas ref={ref} width={600} height={110} className="mt-2 h-[70px] w-full border border-line" />
}

function Check({ checked, onChange, children }) {
  return (
    <label className="flex items-center gap-1.5 text-xs">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {children}
    </label>
  )
}

export default function PaintPanel() {
  const r = useRelighter()
  const { ui, state, NP } = r
  const set = (k) => (v) => r.setUi(k, v)
  return (
    <>
      <p className="mb-3 text-dim">
        Paint where you want more or less light, then let gradient descent through the network find light
        parameters that produce it (paper §6.2). You can also load a target image, e.g. from an image-editing
        model (§6.3). Ctrl-drag or right-drag erases.
      </p>
      <div className="flex items-end gap-2">
        <Field label="Brush color" className="flex-1">
          <Swatches colors={BRUSH_SWATCHES} onPick={set('brushColor')} />
        </Field>
        <input type="color" className="mb-2" value={ui.brushColor} onChange={(e) => r.setUi('brushColor', e.target.value)} />
      </div>
      <Field label="Brush size" value={`${ui.brushSize}px`}>
        <Range min={4} max={120} value={ui.brushSize} onChange={set('brushSize')} />
      </Field>
      <Field label="Brush opacity" value={fmt(ui.brushAlpha)}>
        <Range min={0.1} max={1} step={0.05} value={ui.brushAlpha} onChange={set('brushAlpha')} />
      </Field>
      <Field label="Keep unpainted regions" value={fmt(ui.preserve)}>
        <Range min={0} max={1} step={0.05} value={ui.preserve} onChange={set('preserve')} />
      </Field>
      <div className="mb-2 flex flex-wrap gap-2">
        <Button small variant="ghost" onClick={() => r.clearPaint()}>Clear paint</Button>
        <label className="cursor-pointer rounded border border-line px-2.5 py-1 text-xs hover:border-dim">
          Load target image…
          <input
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => {
              const f = e.target.files[0]
              if (f) r.loadTarget(f)
              e.target.value = ''
            }}
          />
        </label>
        {r.targetImg && <Button small variant="ghost" onClick={() => r.clearTarget()}>Remove target</Button>}
      </div>

      <SectionTitle>Optimize</SectionTitle>
      <div className="mb-2 grid grid-cols-2 gap-1">
        <Check checked={ui.optPos} onChange={set('optPos')}>position</Check>
        <Check checked={ui.optRadius} onChange={set('optRadius')}>radius</Check>
        <Check checked={ui.optColor} onChange={set('optColor')}>color &amp; intensity</Check>
        <Check checked={ui.optSelected} onChange={set('optSelected')}>selected light only</Check>
      </div>
      <div className="flex gap-2">
        <Field label="Iterations" value={ui.iters} className="flex-1">
          <Range min={20} max={600} step={10} value={ui.iters} onChange={set('iters')} />
        </Field>
        <Field label="Pixels / step" value={`${ui.frac} (${fmt((100 * ui.frac) / NP, 1)}%)`} className="flex-1">
          <Range min={512} max={4096} step={512} value={ui.frac} onChange={set('frac')} />
        </Field>
      </div>
      <div className="flex gap-2">
        <Button variant={state.optimizing ? 'danger' : 'primary'} className="flex-1" onClick={() => r.toggleOptimize()}>
          {state.optimizing ? 'Stop' : 'Optimize lights'}
        </Button>
        <Button variant="ghost" disabled={!state.undo || state.optimizing} onClick={() => r.undoOptimize()}>Undo</Button>
      </div>
      <LossPlot history={state.lossHist} version={r.version} />
      <div className="mt-1 font-mono text-xs text-dim">{r.optStatus}</div>
    </>
  )
}
