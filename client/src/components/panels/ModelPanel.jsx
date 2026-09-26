import { useRelighter } from '../../relight/useRelighter.js'
import { fmt } from '../../relight/color.js'

function Table({ rows }) {
  return (
    <table className="w-full border-collapse text-xs">
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k} className="border-b border-line">
            <td className="py-0.5 text-dim">{k}</td>
            <td className="py-0.5 text-right">{v}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export default function ModelPanel() {
  const r = useRelighter()
  const { engine, scene } = r
  const n = scene.network, t = scene.train
  const last = t.log?.[0]
  return (
    <>
      <h3 className="mb-1 font-bold">Network</h3>
      <Table
        rows={[
          ['MLP', `${n.width} × ${n.hidden}`],
          ['pixel encoding', `${n.grid_res.length} grids × ${n.feats} features, ${n.grid_res[0]}²–${n.grid_res.at(-1)}²`],
          ['geometric light inputs', n.geo ? 'yes' : 'no'],
          ['output head', n.head === 'mul' ? 'multiplicative (a·G + b)' : 'linear'],
          ['parameters', engine.paramCount.toLocaleString()],
          ['weights', `${fmt(engine.modelBytes / 1e6, 1)} MB`],
        ]}
      />
      <h3 className="mt-4 mb-1 font-bold">Training</h3>
      <Table
        rows={[
          ['iterations', t.iters.toLocaleString()],
          ['path samples', `${t.spp} spp, up to ${t.max_seg} segments`],
          ...(last
            ? [
                ['final loss (rel. MSE)', last.loss.toExponential(2)],
                ['PSNR vs. denoised target', `${fmt(last.psnr_den, 1)} dB`],
                ['training time', `${fmt(last.time / 60, 0)} min`],
              ]
            : []),
        ]}
      />
      <h3 className="mt-4 mb-1 font-bold">Viewer</h3>
      <Table
        rows={[
          ['image', `${r.W} × ${r.H}`],
          ['backend', engine.backend],
          ['time per light (full res)', r.stats.perLight ? `${fmt(r.stats.perLight, 1)} ms` : '–'],
        ]}
      />
    </>
  )
}
