import { useEffect, useState } from 'react'
import { useRelighter } from '../relight/useRelighter.js'
import { SCENE } from '../relight/relighter.js'
import { fmt } from '../relight/color.js'

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
  return (
    <div className="flex flex-wrap items-center gap-4 text-xs text-dim">
      {scenes.length > 1 && (
        <select title="model" className={SELECT} value={SCENE} onChange={(e) => r.reloadWith({ scene: e.target.value })}>
          {scenes.map((s) => <option key={s.name} value={s.name}>{s.label}</option>)}
        </select>
      )}
      <span><b className="font-normal text-fg">{r.stats.fps ?? '–'}</b> fps</span>
      <span><b className="font-normal text-fg">{r.stats.perLight ? fmt(r.stats.perLight, 1) : '–'}</b> ms / light eval</span>
      <span className="max-w-sm truncate" title={r.stats.gpu}>{r.stats.gpu}</span>
    </div>
  )
}
