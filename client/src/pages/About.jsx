function Section({ title, children }) {
  return (
    <section className="space-y-2">
      <h2 className="text-base font-bold text-fg">{title}</h2>
      {children}
    </section>
  )
}

const CONTROLS = [
  ['drag a light', 'move it parallel to the image plane'],
  ['mouse wheel', 'move the selected light towards / away from the camera'],
  ['shift + wheel', 'change its radius'],
  ['alt + wheel', 'change its intensity'],
  ['double-click a surface', 'place the selected light just in front of it'],
  ['Paint tab: drag', 'paint the lighting you want (ctrl or right-drag erases)'],
]

const MAPPING = [
  ['§3.1 Decoupled rendering', 'Light-agnostic camera paths are traced once in Mitsuba 3; a Triton kernel gathers light from virtual sphere lights along them to make training targets.'],
  ['§3.2 Linearity', 'Each light is evaluated by the network on its own and cached, so colour and intensity edits are free and moving a light re-evaluates only that light.'],
  ['§4.3 Network', 'Multi-resolution pixel grids plus albedo, normal and depth, and the light parameters, into a small MLP.'],
  ['§5.3 Inverse lighting', 'Adam over sigmoid / softplus reparameterised lights, with a hand-written WGSL backward pass and random pixel subsets.'],
  ['§6.2 / 6.3 Scribbles and targets', 'The Paint & Optimize tab: paint with a keep-mask weight, or load a target image.'],
]

export default function About() {
  return (
    <main className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-3xl space-y-8 px-4 py-8 text-dim">
        <header className="space-y-2">
          <h1 className="text-2xl font-bold text-fg">Neural Render Proxies on the web</h1>
          <p>
            A re-implementation of <em>Neural Render Proxies for Interactive and Differentiable Lighting</em>{' '}
            (Sancho et al., EGSR 2026). A small neural network stands in for a path tracer for one static scene
            and camera. Given a sphere light, it predicts that light's contribution to every pixel, fast enough to
            relight interactively in the browser, and differentiably, so lights can be optimized to match a
            painted or loaded target.
          </p>
        </header>

        <Section title="Controls">
          <table className="w-full border-collapse text-sm">
            <tbody>
              {CONTROLS.map(([k, v]) => (
                <tr key={k} className="border-b border-line">
                  <td className="py-1 pr-4 whitespace-nowrap text-fg">{k}</td>
                  <td className="py-1">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>

        <Section title="Backends">
          <p>
            The viewer uses WebGPU compute shaders when the browser has them and falls back to WebGL2, where the
            network runs as a chain of fragment-shader passes (about 2× slower per light, with finite differences
            for the light inputs when optimizing). Add <code className="text-fg">?backend=webgl</code> or{' '}
            <code className="text-fg">?backend=webgpu</code> to the URL, or use the switch in the header, to force one.
          </p>
          <p>
            On slow GPUs a moving light is evaluated on every 2nd, 4th or 8th pixel, chosen from measured timings,
            and upsampled along the scene geometry. About 150 ms after the last change it is re-evaluated at full
            resolution.
          </p>
        </Section>

        <Section title="How it maps to the paper">
          <table className="w-full border-collapse text-sm">
            <tbody>
              {MAPPING.map(([k, v]) => (
                <tr key={k} className="border-b border-line align-top">
                  <td className="py-1.5 pr-4 whitespace-nowrap text-fg">{k}</td>
                  <td className="py-1.5">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>

        <Section title="Limitations">
          <ul className="list-disc space-y-1 pl-5">
            <li>The proxy is specific to one static scene and camera. Lights are clamped to the training domain.</li>
            <li>Regions camera paths rarely reach (e.g. behind the tall box) are learned less accurately; test 6 in the Accuracy tab is such a case.</li>
            <li>Inverse lighting with several lights is non-convex. If it lands in a local minimum, press Undo, move a light roughly into place and optimize again.</li>
          </ul>
        </Section>
      </div>
    </main>
  )
}
