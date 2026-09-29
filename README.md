# Neural Render Proxies on the web

A re-implementation of **Neural Render Proxies for Interactive and Differentiable Lighting**
(Sancho et al., EGSR 2026), with a WebGPU viewer (and a WebGL2 fallback) that runs the proxy in the browser. The viewer
supports interactive relighting and gradient-based inverse lighting ("paint the light you want").

```
nrp/                      Python: data generation + training (PyTorch, Mitsuba 3, Triton, OIDN)
  scenes.py               emitter-free scenes + light domain
  usd_scene.py            any USD stage as an emitter-free Mitsuba scene (see "Your own scene")
  sample_paths.py         SAMPLEPATHS: light-agnostic path dump (vertices + throughputs)
  gather.py               GATHERLIGHT: fused Triton kernel, sphere lights, segment sampling
  direct.py               analytic direct-view term (segment 0)
  denoise.py              OIDN 2 (CUDA, via ctypes) with CPU fallback
  model.py                grid encoding + MLP
  train.py                pool-based training with relative MSE
  export.py               export to client/public/scenes/<name>/
  evaluate.py             score models against high-spp references
  validate_gather.py      GATHERLIGHT vs a regular Mitsuba render with a real emitter
client/                   React 19 + Vite + Tailwind CSS 4 viewer
  public/scenes/          exported models (weights, per-pixel buffers, reference renders)
  src/engine/shaders.js   WGSL: precompute, fused MLP forward, composite, loss, backward
  src/engine/nrp.js       WebGPU engine + Adam light optimizer
  src/engine/nrp-gl.js    WebGL2 fallback engine (fragment-shader MLP passes)
  src/engine/engine-base.js  scene loading and camera helpers shared by both engines
  src/relight/relighter.js   viewer runtime: lights, render loop, viewport interaction, optimization
  src/components/, src/pages/  UI
server/                   Express API (health check); the viewer itself is static
```

## Run the viewer

Needs Node.js 20.19+ (Vite 7).

```
npm install
npm run install:all
npm run dev              # then open http://localhost:5173
npm run lan              # also reachable from a phone/tablet on the same network (WebGL2 there)
npm run https            # same over HTTPS with a self-signed certificate (WebGPU there too)
```

With `lan` or `https`, Vite prints the addresses to open on the other device. Windows Firewall
must allow Node on the network the phone uses, and Windows only does that for networks set to *Private*.
Browsers enable WebGPU only on HTTPS or localhost, so over plain `lan` other devices use the WebGL2
fallback. With `https`, accept the certificate warning once per device.

For production, `npm run build` writes the static site to `client/dist`, or `docker compose up --build`
serves it with nginx on port 80 (the API runs alongside on 3001).

The viewer uses WebGPU when the browser has it and falls back to WebGL2 otherwise (it needs
`EXT_color_buffer_float` or `EXT_color_buffer_half_float`, which nearly all WebGL2 devices have).
Add `?backend=webgl` or `?backend=webgpu` to the URL to force one. The status bar shows which is in use.
The **Auto / WebGPU / WebGL2** switch in the header does the same. It reloads the page but keeps your lights and
exposure, so you can compare both backends on identical lighting. Models exported at other image sizes
(`export.py --res N`) show up in a size menu next to the model menu (or use `?res=N`); smaller is faster on weak GPUs,
but there are no reference renders at those sizes.

The WebGL2 path evaluates the network as a chain of fragment-shader passes. It is about 2x slower
per light than WebGPU on an RTX 3080. For Paint & Optimize it uses central finite differences for the
network's 4 light inputs instead of the WGSL backward pass. Colour and the direct-view term still have
exact gradients.

On slow GPUs, both backends keep dragging responsive with previews. When evaluating a moving light at
full resolution would take longer than about 30 ms, the network runs on every 2nd, 4th or 8th pixel,
chosen from measured timings. The composite fills in the rest by bilinear interpolation weighted by
surface position, so light does not bleed across object edges, while the light's own disc stays exact.
About 150 ms after the last change, the image is re-evaluated at full resolution. Fast GPUs stay at
full resolution. To try the worst case (no GPU), start Chrome with `--use-angle=swiftshader
--enable-unsafe-swiftshader` and a separate `--user-data-dir` (plus `--no-first-run`), and open
`?backend=webgl`. Dragging there runs at about 7 fps instead of about one frame per 7 s.

## Rebuild from scratch

Big files (the venv and path dumps of about 2.5 GB per scene) live outside the project, in
`%USERPROFILE%\relight-work`. You can override this with `RELIGHT_WORK`.

```
python -m venv %USERPROFILE%\relight-work\venv
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install mitsuba numpy pillow "triton-windows<3.3" oidn usd-core
# optional but ~20x faster training-data generation: unpack the OIDN 2.x Windows release
# (github.com/RenderKit/oidn/releases) into %USERPROFILE%\relight-work, or set OIDN_DIR

cd nrp
python sample_paths.py --scene cornell --res 512 --spp 128   # ~1 min on an RTX 3080
python validate_gather.py                                    # optional sanity check
python train.py --geo --head mul --width 128 --hidden 4 --iters 100000 --name cornell_geo_128x4   # ~21 min on an RTX 3080
python export.py --run cornell_geo_128x4 --name cornell
python evaluate.py --runs cornell_geo_128x4                  # optional: score vs a 1024-spp reference
```

## Your own scene (USD)

Any USD stage (`.usd`, `.usda`, `.usdc`, `.usdz`) can replace the Cornell box. Pass the file to
`sample_paths.py`, and use the id it prints (the file name without extension, or `--name`) as the
scene for the other steps:

```
python sample_paths.py --scene D:\assets\kitchen.usdz --name kitchen [--camera /World/Cam]
python train.py --scene kitchen --geo --head mul --width 128 --hidden 4 --iters 100000
python export.py --run kitchen_128x4 --label "Kitchen · fast (128×4)"
```

The model then appears in the viewer's model menu (or open `?scene=kitchen`).

`nrp/examples/caustics_still_life.py` builds an example stage in code: glassware in a plaster niche, made to show off
caustics. Its docstring gives the settings it needs: more bounces for the wine glass, and smaller lights for sharper caustics.

What the import does (`usd_scene.py`):

- **Camera.** It uses the camera named with `--camera`, or else the first camera in the stage. Without a camera it frames all
  geometry from a three-quarter view. The image is square, so a wide camera is cropped to its smaller field of view.
- **Units and scale.** The stage is converted to Y-up, then moved and scaled so that the region the camera sees spans
  about [-1, 1]. The network's inputs, the light radius range and the path storage expect that scale. Light positions in the viewer are
  in these coordinates; `meta.json` records the transform (`normalize`) to map them back to the stage.
- **Light domain.** By default the lights live in the visible region. For enclosed scenes it is pulled in slightly, and when many camera rays
  escape (an object on a ground plane, say) it is grown by 0.3 so lights can go above and around things. The import prints it. Override it with
  `--light-bbox x0 y0 z0 x1 y1 z1` (normalised coordinates), `--light-margin` or `--radius-range`.
- **Geometry.** Meshes (polygons, holes, GeomSubsets with their own materials, vertex or faceVarying normals and UVs),
  the implicit Sphere, Cube, Cylinder, Cone, Capsule and Plane, native instances and PointInstancers are all imported. Subdivision surfaces
  render as their control cage. Invisible prims, guides and proxies are skipped, and so are curves, points and volumes.
- **Materials.** UsdPreviewSurface is converted in full: constant or textured colour, roughness, metallic and opacity, UsdTransform2d,
  normal maps, alpha cutouts (`opacityThreshold`), and glass (low opacity plus `ior`). MaterialX standard_surface / OpenPBR and OmniPBR / OmniGlass
  only get their constant values. Anything else falls back to `displayColor`. Emission is ignored, because the virtual lights are the only lights.
  Textures larger than `--max-texture` (2048) are downsampled.
- **Lights.** UsdLux lights don't take part in training, but sphere, disk, rect and cylinder lights become the viewer's starting lights
  (non-sphere ones as spheres of the same area). They keep their colours and relative intensities and are scaled for a good exposure.
  Without such lights, `export.py` picks a warm key and a cool fill that light the view well. It also picks the test lights for the Accuracy tab.

The import prints a summary and notes on anything it approximated or skipped. Check those first when something looks off.

The paper's limitations apply here too: one static stage, one camera, and sphere lights inside the light domain. Large, open or highly
detailed scenes spread the same network over more variation, so expect lower accuracy than on the Cornell box.
Keep the light domain tight, or train longer or wider.

## How it maps to the paper

| Paper | Here |
|---|---|
| §3.1 Decoupled rendering (SAMPLEPATHS / GATHERLIGHT) | `sample_paths.py` traces BSDF-sampled paths in Mitsuba 3 without NEE and stores fp16 vertices and throughputs. `gather.py` intersects all segments with virtual sphere lights in one Triton kernel. |
| §3.2 Linearity: Î = Σ E(v) N(px, F, v) | One network per light type (sphere). The viewer caches each light's contribution, so colour and intensity edits are free and moving a light re-evaluates only that light. |
| §4.3 Network: hash-grid pixel encoding + aux features (albedo, normal, depth) + light params | `model.py`. In 2D the hash grid is collision-free, so it is stored as dense multi-resolution grids. |
| §4.4 Training: pool of 300 denoised images, 2 replaced every 5 iterations; segment-based light sampling; relative MSE | `train.py`. A background thread gathers and denoises on the GPU (OIDN 2 CUDA). Lights are sampled half uniformly in the light box and half on recorded path segments. |
| §5.3 Inverse: Reinhard-tonemapped MSE, sigmoid/softplus reparameterisation, Adam lr 0.05, random pixel subsets | `client/src/engine/nrp.js` (`gradStep`, `LightOptimizer`) with a hand-written WGSL backward pass. |
| §6.2 / 6.3 Art-directed scribbles, generative targets | The *Paint & Optimize* tab: paint with a keep-mask weight, or load a target image. |

### Deliberate deviations

- **Segment 0 is analytic.** The camera seeing the light directly is exact and cheap to compute
  (ray–sphere test against the depth buffer, 4×4 supersampled). A sharp disc whose edge moves with
  the light parameters is very hard for an MLP, so the network learns segments ≥ 1 only
  (`--first-seg 0` restores the paper's setup). For optimization, a soft-edge version (coverage
  ramps across one pixel of angular distance) gives closed-form gradients with respect to light
  position and radius. Pixels on the disc rims are sampled as their own stratum. Without this, radius and
  intensity are ambiguous: a large dim light and a small bright light cast almost the same
  indirect light, and only the visible disc tells them apart.
- **Paint loss normalisation.** Painting optimizes `mean(painted) + keep × mean(unpainted)`, so the
  "keep" slider means the same thing for small and large strokes. The lights' own discs are left out of the
  painting loss, because the artist paints illumination, not light positions. Target images use the plain
  full-image mean, as in §5.3.
- **Geometric per-(pixel, light) inputs and a multiplicative output head.** These are not in the paper; see "Accuracy" below.
  They gave about +2 dB and 25% less error at negligible runtime cost.
- **Smaller networks (128×4 and 256×4 instead of 256×8)**, so they evaluate quickly in a browser. On an RTX 3080
  one light takes about 4.5 ms (128×4) or 16 ms (256×4) at 512².
- **Colour step size in the optimizer** scales with the magnitude of the unconstrained colour parameter, so bright
  lights (radiance around 50) change at a useful rate with the paper's lr of 0.05.

## Accuracy: what helps and what doesn't

`evaluate.py` scores models against a high-quality reference. It renders GATHERLIGHT on
1024 spp of freshly traced paths, streamed so nothing is stored, then denoises that and adds the exact direct term.
Images are exposure-normalised like the viewer, and the test set is 37 lights: the viewer's 6 test lights
plus 31 random ones (lights inside solid objects are skipped). It also scores the training targets
themselves (denoised 128-spp gathers), which shows how much error comes from supervision
and how much from the network.

```
python evaluate.py --runs cornell_128x4 cornell_geo_128x4
```

Findings (128×4 networks, 20k-iteration ablations unless noted):

| change | PSNR | rel. error | verdict |
|---|---|---|---|
| training targets (denoised 128 spp) | 52.4 dB | 1.5 % | supervision is not the bottleneck; the network is |
| baseline (paper inputs) | 39.6 dB | 7.2 % | |
| + world position as aux input | 39.8 dB | 7.5 % | no gain |
| **+ geometric features (`--geo`)** | **41.7 dB** | **5.7 %** | kept |
| **+ geo + multiplicative head (`--head mul`)** | **41.8 dB** | **5.4 %** | kept |
| + 3D grid encoding of the light position | 40.3 dB | 6.3 % | worse, dropped |
| 128×6 instead of 128×4 | 41.9 dB | 6.0 % | not worth 50 % more cost |
| error-driven light sampling (`--adapt 0.33`) | 40.5 dB | 5.7 % | +0.5 dB on the hard tests, −1.4 dB overall; off |

- **Geometric features.** For each pixel and light, the network gets the direction to the light, the cosine with the normal,
  the log distance, and the log solid angle of the sphere. It no longer has to synthesise 1/d² falloff and cosine terms
  from pixel coordinates and depth with ReLUs.
- **Multiplicative head.** The output is `a·G + b`, where `G = Ω·max(cos, 0)/π` is the unshadowed irradiance factor.
  The network then learns visibility and albedo (`a`) and everything else (`b`).
- **Cost in the browser.** These inputs cost about 4% at render time.

Width and training length remain the other big levers: 3× longer training and 128→256 width each gave about 2 dB.

Shipped models (full training, same evaluation):

| model | PSNR | rel. error | worst 5 lights | time per light (RTX 3080) |
|---|---|---|---|---|
| old fast 128×4, 60k iterations | 41.5 dB | 5.9 % | 30.6 dB | 4.5 ms |
| old quality 256×4, 100k iterations | 43.6 dB | 5.2 % | 31.7 dB | 15.9 ms |
| **fast 128×4 geo+mul, 100k iterations** (`cornell`) | **44.0 dB** | **4.4 %** | 32.5 dB | 4.8 ms |
| **quality 256×4 geo+mul, 150k iterations** (`cornell-hq`) | **46.2 dB** | **3.5 %** | **34.1 dB** | 17.1 ms |

Ideas not tried yet, roughly by expected value:

- **More path samples, only for the regions the camera rarely reaches.** The remaining worst cases (lights behind the tall box, tiny lights over the glass) also have the worst training targets.
- **The paper's full 256×8 network.** It would be too slow for interactive use in the browser, but could serve as a slower "final quality" option.
- **fp16 WebGPU kernels (`shader-f16`).** These should roughly halve the cost per light, which would make the quality model cheap enough to be the default.

## Notes and limitations

These match the paper's limitations:

- The proxy is specific to one static scene and camera. Lights are only valid inside the training domain (`light_bbox`,
  `radius_range`), and the viewer clamps them to it.
- Regions that camera paths rarely reach (e.g. behind the tall box against the wall) are learned less
  accurately. The *Accuracy* tab compares the proxy against held-out reference renders, and test 6 is
  deliberately such a case.
- Inverse lighting with several lights is non-convex. It can land in a local minimum; if it does, press Undo, move a light
  roughly into place, and optimize again.
