import { useSyncExternalStore } from 'react'
import { getRelighter } from './relighter.js'

/** The viewer runtime; the calling component re-renders whenever it changes. */
export function useRelighter() {
  const r = getRelighter()
  useSyncExternalStore(r.subscribe, r.getVersion)
  return r
}
