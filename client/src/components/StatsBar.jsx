import { useRelighter } from '../relight/useRelighter.js'
import { SCENE, BACKEND, RES, AUTO, tiersOf, nativeOf } from '../relight/relighter.js'
import { fmt } from '../relight/color.js'
import { Segmented } from './ui.jsx'

const hasWebGPU = !!navigator.gpu
const hasWebGL2 = (() => {
  try { return !!document.createElement('canvas').getContext('webgl2') } catch { return false }
})()

const SELECT = 'rounded border border-line bg-[#f3f4f6] px-1.5 py-0.5 text-xs text-fg'

export default function StatsBar() {
  const r = useRelighter()
  const scenes = r.index
  const entry = scenes.find((s) => s.name === SCENE)
  const tiers = tiersOf(entry), native = nativeOf(entry)
  // '' = auto (the size follows the GPU), 'native', or a pinned size
  const size = AUTO ? '' : RES ?? 'native'
  const shown = r.W ? `${r.W} × ${r.H}` : '…'
  const want = (BACKEND || '').toLowerCase()
  const backends = [
    { value: '', label: r.engine && !want ? `Auto (${r.engine.backend})` : 'Auto' },
    { value: 'webgpu', label: 'WebGPU', disabled: !hasWebGPU, title: hasWebGPU ? undefined : 'WebGPU is not available in this browser' },
    { value: 'webgl', label: 'WebGL2', disabled: !hasWebGL2, title: hasWebGL2 ? undefined : 'WebGL2 is not available in this browser' },
  ]
  return (
    <div className="flex flex-wrap items-center gap-4 text-xs text-dim">
      <span title="Rendering backend (reloads, keeps your lights)">
        <Segmented options={backends} value={want} onChange={(b) => b !== want && r.reloadWith({ backend: b })} />
      </span>
      {scenes.length > 1 && (
        <select title="model" className={SELECT} value={SCENE} onChange={(e) => r.reloadWith({ scene: e.target.value, res: null })}>
          {scenes.map((s) => <option key={s.name} value={s.name}>{s.label}</option>)}
        </select>
      )}
      {entry && tiers.length > 1 && (
        <select title="image size: auto follows what the GPU can take (smaller is faster on weak GPUs)" className={SELECT} value={size} onChange={(e) => r.reloadWith({ res: e.target.value })}>
          <option value="">auto ({r.swapping ? `${shown}, loading another` : shown})</option>
          {tiers.map((t) => <option key={t} value={t === native ? 'native' : t}>{t} × {t}{t === native ? ' (native)' : ''}</option>)}
        </select>
      )}
      <span><b className="font-normal text-fg">{r.stats.fps ?? '–'}</b> fps</span>
      <span><b className="font-normal text-fg">{r.stats.perLight ? fmt(r.stats.perLight, 1) : '–'}</b> ms / light eval</span>
      <span className="max-w-sm truncate" title={r.stats.gpu}>{r.stats.gpu}</span>
    </div>
  )
}
