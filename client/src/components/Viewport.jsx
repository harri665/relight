import { useEffect, useRef } from 'react'
import { useRelighter } from '../relight/useRelighter.js'

export default function Viewport() {
  const r = useRelighter()
  const stage = useRef(null)

  useEffect(() => {
    r.mount(stage.current)
    return () => r.unmount()
  }, [r])

  const { phase, message } = r.status
  return (
    <section className="flex min-h-0 flex-col items-center justify-center gap-2 p-4">
      <div ref={stage} className="relative aspect-square w-[min(100%,calc(100vh-110px))] bg-black max-[800px]:w-full">
        {phase !== 'ready' && (
          <div className={`absolute inset-0 z-10 flex items-center justify-center whitespace-pre-line p-5 text-center ${phase === 'error' ? 'text-danger' : 'text-dim'}`}>
            {message}
          </div>
        )}
      </div>
      <div className="min-h-[1.4em] text-xs text-dim">{r.hint}</div>
    </section>
  )
}
