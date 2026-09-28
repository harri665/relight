import { useEffect, useState } from 'react'
import { useRelighter } from '../relight/useRelighter.js'
import { SCENE, BACKEND, RES } from '../relight/relighter.js'
import { fmt } from '../relight/color.js'
import { Segmented } from './ui.jsx'

const hasWebGPU = !!navigator.gpu
const hasWebGL2 = (() => {
  try { return !!document.createElement('canvas').getContext('webgl2') } catch { return false }
})()

function useSceneIndex() {
  const [scenes, setScenes] = useState([])
  useEffect(() => {
    fetch('/scenes/index.json')
      .then((res) => (res.ok ? res.json() : []))
      .then(setScenes)
      .catch(() => {})
  }, [])
  return scenes
}

const SELECT = 'rounded border border-line bg-[#222] px-1.5 py-0.5 text-xs text-fg'

export default function StatsBar() {
  const r = useRelighter()
  const scenes = useSceneIndex()
  const tiers = scenes.find((s) => s.name === SCENE)?.tiers ?? []
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
      {tiers.length > 0 && (
        <select title="image size (smaller is faster on weak GPUs)" className={SELECT} value={RES ?? ''} onChange={(e) => r.reloadWith({ res: e.target.value })}>
          <option value="">native size</option>
          {tiers.map((t) => <option key={t} value={t}>{t} × {t}</option>)}
        </select>
      )}
      <span><b className="font-normal text-fg">{r.stats.fps ?? '–'}</b> fps</span>
      <span><b className="font-normal text-fg">{r.stats.perLight ? fmt(r.stats.perLight, 1) : '–'}</b> ms / light eval</span>
      <span className="max-w-sm truncate" title={r.stats.gpu}>{r.stats.gpu}</span>
    </div>
  )
}
