import { useRelighter } from '../../relight/useRelighter.js'
import { MAX_LIGHTS } from '../../relight/relighter.js'
import { LIGHT_SWATCHES, rgbToHex, fmt } from '../../relight/color.js'
import { Button, Field, Range, SectionTitle, Swatches } from '../ui.jsx'

function LightRow({ light, index, selected }) {
  const r = useRelighter()
  const color = rgbToHex(light.color)
  return (
    <li
      className={`flex cursor-pointer items-center gap-2 border px-1.5 py-1 ${selected ? 'border-accent' : 'border-transparent'} ${light.enabled ? '' : 'opacity-50'}`}
      onClick={() => r.selectLight(index)}
    >
      <span className="h-3 w-3 rounded-full" style={{ background: color }} />
      <span className="flex-1">{light.name}</span>
      <span className="text-[11px] text-dim">r {fmt(light.radius)} · {fmt(light.intensity, 0)}</span>
      <button
        type="button"
        title="toggle"
        className="cursor-pointer px-1"
        onClick={(e) => { e.stopPropagation(); r.toggleLight(index) }}
      >
        {light.enabled ? '◉' : '○'}
      </button>
      <button
        type="button"
        title="remove"
        className="cursor-pointer px-1"
        onClick={(e) => { e.stopPropagation(); r.removeLight(index) }}
      >
        ✕
      </button>
    </li>
  )
}

function LightEditor() {
  const r = useRelighter()
  const { engine } = r
  const l = r.selected
  const disabled = !l
  const pos = l?.pos ?? [0, 0, 0]
  return (
    <div className={disabled ? 'pointer-events-none opacity-40' : ''}>
      {['X', 'Y', 'Z'].map((axis, i) => (
        <Field key={axis} label={axis} value={fmt(pos[i])}>
          <Range min={engine.lo[i]} max={engine.hi[i]} step={0.001} value={pos[i]} onChange={(v) => r.editSelected(i, v)} />
        </Field>
      ))}
      <Field label="Radius" value={fmt(l?.radius ?? 0, 3)}>
        <Range min={engine.rmin} max={engine.rmax} step={0.001} value={l?.radius ?? engine.rmin} onChange={(v) => r.editSelected('radius', v)} />
      </Field>
      <Field label="Intensity" value={fmt(l?.intensity ?? 0, 1)}>
        <Range min={-1} max={3} step={0.01} value={Math.log10(l?.intensity ?? 1)} onChange={(v) => r.editSelected('logIntensity', v)} />
      </Field>
      <div className="mb-2 flex flex-wrap items-center gap-1.5">
        <span className="text-xs text-dim">Color</span>
        <input type="color" value={rgbToHex(l?.color ?? [1, 1, 1])} onChange={(e) => r.editSelected('color', e.target.value)} />
        <Swatches colors={LIGHT_SWATCHES} onPick={(c) => r.editSelected('color', c)} />
      </div>
    </div>
  )
}

export default function LightsPanel() {
  const r = useRelighter()
  const { lights, sel } = r.state
  const ev = r.ui.exposure
  return (
    <>
      <Field label="Exposure" value={`${ev >= 0 ? '+' : ''}${fmt(ev, 1)} EV`}>
        <Range min={-4} max={4} step={0.05} value={ev} onChange={(v) => r.setExposure(v)} />
      </Field>

      <SectionTitle
        actions={
          <>
            <Button small disabled={lights.length >= MAX_LIGHTS} onClick={() => r.addLight()}>+ Add</Button>
            <Button small variant="ghost" onClick={() => r.randomLights()}>Randomize</Button>
          </>
        }
      >
        Lights
      </SectionTitle>
      <ul className="mb-3">
        {lights.map((l, i) => <LightRow key={l.id} light={l} index={i} selected={i === sel} />)}
      </ul>

      <LightEditor />
    </>
  )
}
