import { useEffect, useState } from 'react'

// What the browser offers the viewer: WebGPU (adapter, features, limits), WebGL2, and the device.
// Probing makes a WebGPU adapter and a WebGL context, so it only runs once it is unfolded.

const yes = (flag) => (flag ? 'yes' : 'no')

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return '—'
  if (bytes >= 1 << 30) return `${+(bytes / (1 << 30)).toFixed(1)} GB`
  if (bytes >= 1 << 20) return `${Math.round(bytes / (1 << 20))} MB`
  return `${Math.round(bytes / 1024)} KB`
}

async function probeWebGPU() {
  const out = { api: !!navigator.gpu, secure: window.isSecureContext }
  if (!out.api) return out
  try {
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) return out
    const info = adapter.info || {}
    out.adapter = [info.vendor, info.architecture, info.description || info.device].filter(Boolean).join(' · ') || 'unnamed adapter'
    out.fallback = adapter.isFallbackAdapter ?? info.isFallbackAdapter ?? false
    out.features = [...adapter.features].sort()
    out.limits = adapter.limits
  } catch (error) {
    out.error = error.message
  }
  return out
}

function probeWebGL() {
  const gl = document.createElement('canvas').getContext('webgl2')
  if (!gl) return { webgl2: false, webgl1: !!document.createElement('canvas').getContext('webgl') }
  const debug = gl.getExtension('WEBGL_debug_renderer_info')
  const out = {
    webgl2: true,
    renderer: String(gl.getParameter(debug ? debug.UNMASKED_RENDERER_WEBGL : gl.RENDERER)),
    version: String(gl.getParameter(gl.VERSION)),
    floatTargets: !!gl.getExtension('EXT_color_buffer_float'),
    halfTargets: !!gl.getExtension('EXT_color_buffer_half_float'),
    timer: !!gl.getExtension('EXT_disjoint_timer_query_webgl2'),
    maxTexture: gl.getParameter(gl.MAX_TEXTURE_SIZE),
    drawBuffers: gl.getParameter(gl.MAX_DRAW_BUFFERS),
    uniformBlock: gl.getParameter(gl.MAX_UNIFORM_BLOCK_SIZE),
  }
  gl.getExtension('WEBGL_lose_context')?.loseContext()
  return out
}

function probeDevice() {
  const c = navigator.connection
  return {
    memory: navigator.deviceMemory,
    cores: navigator.hardwareConcurrency,
    dpr: window.devicePixelRatio || 1,
    screen: `${window.screen.width} × ${window.screen.height}`,
    touch: matchMedia('(pointer: coarse)').matches,
    network: c?.effectiveType,
    saveData: !!c?.saveData,
  }
}

function Rows({ rows }) {
  return (
    <table className="w-full border-collapse text-xs">
      <tbody>
        {rows.map(([k, v, note]) => (
          <tr key={k} className="border-b border-line align-top">
            <td className="py-0.5 pr-2 text-dim">{k}</td>
            <td className="py-0.5 text-right">
              {v}
              {note && <div className="text-dim">{note}</div>}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Probe() {
  const [probe, setProbe] = useState(null)
  useEffect(() => {
    let cancelled = false
    const webgl = probeWebGL()
    const device = probeDevice()
    probeWebGPU().then((webgpu) => !cancelled && setProbe({ webgpu, webgl, device }))
    return () => { cancelled = true }
  }, [])
  if (!probe) return <p className="mt-2 text-xs text-dim">Asking your browser…</p>
  const { webgpu: g, webgl: w, device: d } = probe
  const has = (f) => !!g.features?.includes(f)

  const gpu = [['WebGPU', yes(g.api), g.api ? 'runs the network' : g.secure ? 'not in this browser' : 'needs HTTPS']]
  if (g.api) gpu.push(['Adapter', g.adapter || (g.error ? `none (${g.error})` : 'none offered for this GPU')])
  if (g.adapter) {
    gpu.push(
      ['Software fallback', yes(g.fallback)],
      ['shader-f16', yes(has('shader-f16')), 'runs the network in 16-bit floats'],
      ['timestamp-query', yes(has('timestamp-query')), 'times the network on the GPU'],
      ['Workgroup memory', formatBytes(g.limits.maxComputeWorkgroupStorageSize), 'the viewer needs 24 KB'],
      ['Threads per workgroup', String(g.limits.maxComputeInvocationsPerWorkgroup)],
      ['Largest storage buffer', formatBytes(g.limits.maxStorageBufferBindingSize)],
      ['Features', g.features.length ? g.features.join(', ') : 'none'],
    )
  }
  const gl = [['WebGL2', yes(w.webgl2), w.webgl2 ? 'runs the network where WebGPU can’t' : null]]
  if (w.webgl2) {
    gl.push(
      ['Renderer', w.renderer],
      ['Version', w.version],
      ['Float render targets', w.floatTargets ? '32-bit' : w.halfTargets ? '16-bit only' : 'no', 'needed by the WebGL network'],
      ['Timer queries', yes(w.timer), 'times the WebGL network on the GPU'],
      ['Largest texture', `${w.maxTexture} px`],
      ['Draw buffers', String(w.drawBuffers), 'layer outputs a pass'],
      ['Uniform block', formatBytes(w.uniformBlock), 'weights a pass'],
    )
  } else {
    gl.push(['WebGL 1', yes(w.webgl1)])
  }
  const device = [
    ['Memory', d.memory ? `${d.memory} GB or more` : 'not reported', 'the largest image needs 4 GB'],
    ['CPU threads', d.cores ? String(d.cores) : 'not reported'],
    ['Screen', `${d.screen} at ${d.dpr}×`],
    ['Touch screen', yes(d.touch), 'starts at the smallest image size'],
    ['Connection', d.network || 'not reported', d.saveData ? 'data saver is on: no larger sizes are fetched' : null],
  ]
  return (
    <div className="mt-2 space-y-3">
      <div><h4 className="mb-1 text-xs font-bold">WebGPU</h4><Rows rows={gpu} /></div>
      <div><h4 className="mb-1 text-xs font-bold">WebGL</h4><Rows rows={gl} /></div>
      <div><h4 className="mb-1 text-xs font-bold">This device</h4><Rows rows={device} /></div>
    </div>
  )
}

export default function BrowserSupport() {
  const [open, setOpen] = useState(false)
  return (
    <details className="mt-4" onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary className="cursor-pointer select-none text-xs text-dim hover:text-fg">What your browser supports</summary>
      {open && <Probe />}
    </details>
  )
}
