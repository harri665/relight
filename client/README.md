# client

The relighting viewer: React 19, React Router, Vite 7 and Tailwind CSS 4.

- `src/engine/` — WebGPU and WebGL2 runtimes for exported neural render proxies (plain ES modules, no React)
- `src/relight/relighter.js` — the viewer runtime (lights, render loop, viewport input, inverse lighting); a single
  instance per page that components subscribe to through `useRelighter()`
- `src/components/`, `src/pages/` — UI
- `public/scenes/` — exported models, written by `nrp/export.py`

`npm run dev`, `npm run dev:lan` (listen on the network), `npm run dev:https` (self-signed HTTPS, for WebGPU on
other devices), `npm run build`, `npm run lint`.
