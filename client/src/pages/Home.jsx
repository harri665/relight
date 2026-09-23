import { useRelighter } from '../relight/useRelighter.js'
import Viewport from '../components/Viewport.jsx'
import LightsPanel from '../components/panels/LightsPanel.jsx'
import PaintPanel from '../components/panels/PaintPanel.jsx'

const TABS = [
  { id: 'lights', label: 'Lights', Panel: LightsPanel },
  { id: 'paint', label: 'Paint & Optimize', Panel: PaintPanel },
]

export default function Home() {
  const r = useRelighter()
  const { tab } = r.state
  const Panel = TABS.find((t) => t.id === tab).Panel
  return (
    <main className="grid min-h-0 flex-1 grid-cols-[1fr_340px] max-[800px]:grid-cols-1">
      <Viewport />
      <aside className="overflow-y-auto border-l border-line bg-panel max-[800px]:border-t max-[800px]:border-l-0">
        <nav className="sticky top-0 z-10 flex border-b border-line bg-panel" role="tablist">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={t.id === tab}
              className={`flex-1 cursor-pointer whitespace-nowrap px-1 py-2 text-xs ${t.id === tab ? 'text-fg shadow-[inset_0_-2px_var(--color-accent)]' : 'text-dim hover:text-fg'}`}
              onClick={() => r.setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
        <div className="px-4 py-3">
          {r.ready ? <Panel /> : <p className="text-dim">{r.status.phase === 'error' ? 'The viewer could not start.' : 'Loading…'}</p>}
        </div>
      </aside>
    </main>
  )
}
