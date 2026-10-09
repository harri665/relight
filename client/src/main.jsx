import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import App from './App.jsx'

// ?bench: no viewer, just window.__relightBench for perf/relight-bench.mjs (see relight/bench.js)
if (new URLSearchParams(location.search).has('bench')) {
  document.getElementById('root').textContent = 'relight bench: window.__relightBench'
  import('./relight/bench.js').then((m) => m.installBench())
} else {
  createRoot(document.getElementById('root')).render(
    <StrictMode>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </StrictMode>,
  )
}
