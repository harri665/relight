"""A rough underwater test scene: a giant gem by a shipwreck in a half-flooded grotto, as a USD stage.

    python examples/sunken_wreck.py [out_dir]

The grotto is flooded up to WATER_LEVEL, two thirds of its height, with an air pocket under the
dome. Sea water (customLayerData relight:medium, per metre) absorbs red within a couple of metres
and scatters forwards, so the visitor's light, a diver's lantern, has a halo while it is under
water; lifted into the air pocket, it shines down through the gently waving surface and throws
caustics over the sand and the wreck, and through the gem a red caustic of its own. The camera
is under water, with the surface's underside along the top of the frame. Rock above the waterline
is tagged relight:medium = "air", the water surface "boundary" (water inside, i.e. below it), and
the gem "submerged" (water outside, none inside). Contents:
  - the grotto: a rocky shell (displaced ellipsoid) with a sand floor (ripple normal map), a few
    half-buried rocks
  - the stern half of a broken ship, upright and half-sunk, seen end-on: planking broken off
    raggedly at the front, ribs receding as arches (deck beams tie every other one into a
    portal), a keel and a transom; almost symmetric, so it frames the gem
  - one giant brilliant-cut ruby, 60 cm across, centred just inside the hull: big, since light
    seen through small facets is what the network blurs most
  - the water surface: a spectral wind sea (ocean()), so the caustics are a rich, non-repeating
    net of big cells and fine filaments
  - kelp and fan coral as alpha-cutout cards
Units are centimetres, Y up.
"""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

rng = np.random.default_rng(7)
WATER_LEVEL = 110.0  # cm; None floods the grotto completely
# The water surface is a small wind sea (see ocean()). Its rms curvature sets how sharply the
# crests focus the light: with the lantern ~1.4 m above the surface, this focuses a little above
# the sand, which gives a caustic net. A lantern close to the surface barely focuses at all: real
# pool caustics come from a distant light, which is why the grotto is tall and the water shallow.
OCEAN = dict(n_waves=150, lam=(10.0, 70.0), wind_deg=35.0, spread=2.0, curvature=0.045, seed=11)
CEILING_GAP = False  # an opening for sunlight; only worth it with a sun layer


# ---------------------------------------------------------------- helpers

def ocean(x, z, n_waves, lam, wind_deg, spread, curvature, seed):
    """Height and slopes of a wind sea: n_waves sines with wavelengths log-uniform in `lam` (cm),
    directions spread around the wind as cos^spread (and a few running against it), amplitudes
    ~ k^-2.2 so long swells carry the height while every scale adds curvature (big caustic cells
    broken up by finer filaments, never repeating), scaled to an rms curvature of `curvature`."""
    r = np.random.default_rng(seed)
    k = 2 * np.pi / np.exp(r.uniform(np.log(lam[0]), np.log(lam[1]), n_waves))
    ang = []
    while len(ang) < n_waves:
        a = r.uniform(-np.pi / 2, np.pi / 2)
        if r.random() < np.cos(a) ** spread:
            ang.append(a)
    ang = np.radians(wind_deg) + np.array(ang) + np.pi * (r.random(n_waves) < 0.15)
    amp = k ** -2.2
    amp *= curvature / np.sqrt(np.sum((amp * k ** 2) ** 2) / 2)
    phase = r.uniform(0, 2 * np.pi, n_waves)
    h, dhx, dhz = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    for ki, ai, ang_i, ph in zip(k, amp, ang, phase):
        dx, dz = math.cos(ang_i), math.sin(ang_i)
        t = ki * (x * dx + z * dz) + ph
        h += ai * np.sin(t)
        c = ai * ki * np.cos(t)
        dhx += c * dx
        dhz += c * dz
    return h, dhx, dhz


def mesh(stage, path, P, faces, N=None, UV=None, uv_faceVarying=False, material=None):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(P, np.float32)))
    faces = np.asarray(faces)
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), faces.shape[1], np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel().astype(np.int32)))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    if N is not None:
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(N, np.float32)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if UV is not None:
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying if uv_faceVarying else UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(np.asarray(UV, np.float32)))
    if material:
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    return m


def grid_faces(rows, cols, wrap=False):
    """Quads of a (rows+1) x (cols+1) vertex grid, wound so d(row) x d(col) is the front."""
    i, j = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    a = i * (cols + 1) + j
    return np.stack([a, a + cols + 1, a + cols + 2, a + 1], -1).reshape(-1, 4)


def vertex_normals(P, faces):
    tri = np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]])
    fn = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
    vn = np.zeros_like(P)
    for k in range(3):
        np.add.at(vn, tri[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)


def noise3(p, octaves=6, base=1.0, seed=0):
    """Cheap smooth 3D noise: a sum of randomly oriented sine waves, [-1, 1]-ish."""
    r = np.random.default_rng(seed)
    out = np.zeros(len(p))
    amp, freq, total = 1.0, base, 0.0
    for _ in range(octaves):
        for _ in range(4):
            d = r.normal(size=3)
            d /= np.linalg.norm(d)
            out += amp * np.sin(p @ d * freq + r.uniform(0, 2 * np.pi))
            total += amp
        amp *= 0.5
        freq *= 2.0
    return out / math.sqrt(total)


def box(stage, path, size, at=(0, 0, 0), rot=(0, 0, 0), material=None, pivot=None):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    api = UsdGeom.XformCommonAPI(c)
    api.SetTranslate(Gf.Vec3d(*map(float, at)))
    api.SetRotate(Gf.Vec3f(*map(float, rot)))
    api.SetScale(Gf.Vec3f(*map(float, size)))
    if pivot is not None:
        api.SetPivot(Gf.Vec3f(*pivot))
    if material:
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(material)
    return c


def lathe(stage, path, prof, segs, at, material, smooth=True):
    prof = np.asarray(prof, float)
    d = np.gradient(prof, axis=0)
    n2 = np.stack([d[:, 1], -d[:, 0]], 1)
    n2 /= np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-9)
    th = np.linspace(0, 2 * np.pi, segs + 1)
    c, s = np.cos(th)[None], np.sin(th)[None]
    r, y = prof[:, :1], prof[:, 1:]
    P = np.stack([r * c, np.repeat(y, segs + 1, 1), r * s], -1).reshape(-1, 3) + np.array(at)
    N = np.stack([n2[:, :1] * c, np.repeat(n2[:, 1:], segs + 1, 1), n2[:, :1] * s], -1).reshape(-1, 3)
    return mesh(stage, path, P, grid_faces(len(prof) - 1, segs), N if smooth else None, material=material)


# ---------------------------------------------------------------- textures

def save(path, img):
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


def sand_textures(out, n=1024):
    y, x = np.mgrid[0:n, 0:n] / n
    warp = 0.08 * np.sin(2 * np.pi * (y * 1.3 + 0.2)) + 0.03 * np.sin(2 * np.pi * (x * 3.1 + y * 2.2))
    ripple = np.sin(2 * np.pi * 9 * (x + warp))  # 9 ripples per tile (tiled 4x over ~6 m)
    height = ripple + 0.15 * rng.normal(size=(n, n))
    gy, gx = np.gradient(height)
    nrm = np.stack([-gx * 6, -gy * 6, np.ones_like(gx)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    save(out / "sand_normal.png", nrm * 0.5 + 0.5)
    grain = rng.normal(size=(n, n)) * 0.05 + 0.04 * ripple
    alb = np.stack([0.74 + grain, 0.66 + grain, 0.50 + grain * 0.8], -1)
    save(out / "sand.png", alb)


def card_texture(path, kind, n=512):
    """Kelp blades or a fan coral on a transparent card (RGBA)."""
    y, x = np.mgrid[0:n, 0:n] / n
    a = np.zeros((n, n))
    if kind == "kelp":
        for k in range(5):
            cx = 0.15 + 0.7 * k / 4 + 0.05 * rng.normal()
            wig = 0.04 * np.sin(2 * np.pi * (y * (2 + k * 0.3)) + k)
            width = 0.045 * (0.4 + 0.6 * np.sin(np.pi * np.clip(1 - y, 0, 1) ** 0.7))
            top = 0.05 + 0.25 * rng.random()
            a = np.maximum(a, ((np.abs(x - cx - wig) < width) & (y > top)).astype(float))
        rgb = np.stack([0.22 + 0.1 * y, 0.32 + 0.05 * y, 0.08 + 0 * y], -1)
    else:  # fan coral: a lacy half-disc
        cx, cy = 0.5, 0.95
        r = np.hypot(x - cx, (y - cy) * 1.1)
        th = np.arctan2(cy - y, x - cx)
        lace = (np.sin(th * 60 + np.sin(r * 40) * 2) > 0.2) | (np.sin(r * 70) > 0.6)
        a = ((r < 0.85) & (th > 0.15) & (th < np.pi - 0.15) & lace).astype(float)
        a[(np.abs(x - cx) < 0.012) & (y > 0.6)] = 1
        rgb = np.stack([0.75 + 0 * y, 0.28 + 0.1 * r, 0.32 + 0 * y], -1)
    save(path, np.concatenate([rgb, a[..., None]], -1))


# ---------------------------------------------------------------- materials

def material(stage, path, color=(0.5, 0.5, 0.5), rough=0.5, metallic=0.0, opacity=1.0, ior=1.5,
             texture=None, normal_map=None, cutout=None, uv_reader=True):
    m = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior)
    st = None
    if texture or normal_map or cutout:
        rd = UsdShade.Shader.Define(stage, path + "/st")
        rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        st = rd.ConnectableAPI()

    def tex(name, file, raw=False, scale=None, bias=None):
        t = UsdShade.Shader.Define(stage, f"{path}/{name}")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(file)
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st, "result")
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw" if raw else "sRGB")
        if scale:
            t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
            t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
        t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        t.CreateOutput("a", Sdf.ValueTypeNames.Float)
        return t.ConnectableAPI()

    if texture or cutout:
        t = tex("tex", texture or cutout)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(t, "rgb")
        if cutout:
            sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(t, "a")
            sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    if not cutout:
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    if normal_map:
        n = tex("normal", normal_map, raw=True, scale=(2, 2, 2, 1), bias=(-1, -1, -1, 0))
        sh.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(n, "rgb")
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return m


# ---------------------------------------------------------------- scene

def build(out):
    out.mkdir(parents=True, exist_ok=True)
    sand_textures(out)
    card_texture(out / "kelp.png", "kelp")
    card_texture(out / "fan_coral.png", "fan")

    stage = Usd.Stage.CreateNew(str(out / "sunken_wreck.usdc"))  # binary: the water mesh is big
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    # Clear-ish sea water, per metre: red absorbed quickly, forward-scattering for lantern halos.
    stage.GetRootLayer().customLayerData = {"relight": {
        "medium": {"sigma_a": Gf.Vec3d(0.5, 0.1, 0.06), "sigma_s": Gf.Vec3d(0.07, 0.09, 0.11), "g": 0.85,
                   "default": "water", "camera": "water"},
        # Lights anywhere in the grotto, air pocket included (the camera sees little of it).
        "lightBox": Vt.Vec3dArray([Gf.Vec3d(-280, 5, -250), Gf.Vec3d(280, 290, 150)])}}

    L = "/World/Looks/"
    rock = material(stage, L + "Rock", (0.30, 0.28, 0.25), rough=0.9)
    sand = material(stage, L + "Sand", texture="./sand.png", normal_map="./sand_normal.png", rough=0.95)
    wood = material(stage, L + "WaterloggedWood", (0.19, 0.14, 0.09), rough=0.8)
    ruby = material(stage, L + "Ruby", (0.95, 0.14, 0.22), rough=0.0, opacity=0.0, ior=1.76 / 1.33)
    kelp = material(stage, L + "Kelp", cutout="./kelp.png", rough=0.6)
    fan = material(stage, L + "FanCoral", cutout="./fan_coral.png", rough=0.7)

    # Grotto shell: an ellipsoid 7 m across and 3.3 m tall, displaced into rock, seen from inside.
    R = np.array([350.0, 330.0, 330.0])
    rows, cols = 64, 128
    v = np.linspace(-0.25, 1, rows + 1)  # from below the sand (-0.25) to the top of the dome
    u = np.linspace(0, 2 * np.pi, cols + 1)
    V, U = np.meshgrid(v * np.pi / 2, u, indexing="ij")
    d = np.stack([np.cos(V) * np.cos(U), np.sin(V), np.cos(V) * np.sin(U)], -1).reshape(-1, 3)
    P = d * R
    disp = noise3(P / 100, octaves=5, base=1.2, seed=1) * 28 + noise3(P / 100, 3, 6, seed=2) * 6
    P = P * (1 + disp[:, None] / np.linalg.norm(P, axis=1, keepdims=True))
    faces = grid_faces(rows, cols)[:, ::-1]  # wound to face inwards
    # A gap in the ceiling, off to one side (for sunlight later).
    top = d[:, 1] > 0.9
    centre = np.array([0.25, 0.0, -0.15])
    hole = np.linalg.norm(d[:, [0, 2]] - centre[[0, 2]], axis=1) < 0.2
    keep = ~(hole[faces].all(1) & top[faces].all(1)) if CEILING_GAP else np.ones(len(faces), bool)
    faces = faces[keep]
    N = vertex_normals(P, faces)
    if WATER_LEVEL is None:
        mesh(stage, "/World/Grotto/Shell", P, faces, N, material=rock)
    else:
        above = P[faces].mean(1)[:, 1] > WATER_LEVEL
        mesh(stage, "/World/Grotto/Shell", P, faces[~above], N, material=rock)
        air = mesh(stage, "/World/Grotto/ShellAboveWater", P, faces[above], N, material=rock)
        air.GetPrim().CreateAttribute("relight:medium", Sdf.ValueTypeNames.String).Set("air")

        # The water surface: only its top is needed (rays never reach beyond the grotto), with
        # its normal up, so "inside" is below it.
        water = material(stage, L + "Water", (1, 1, 1), rough=0.0, opacity=0.0, ior=1.33)
        # ~1.2 cm spacing, fine enough for the shortest ripples.
        n = 700
        wx, wz = np.meshgrid(np.linspace(-420, 420, n + 1), np.linspace(-420, 420, n + 1), indexing="ij")
        h, dhx, dhz = ocean(wx, wz, **OCEAN)
        print(f"  water surface: height rms {h.std():.2f} cm, max slope {np.hypot(dhx, dhz).max():.2f}")
        P = np.stack([wx.ravel(), WATER_LEVEL + h.ravel(), wz.ravel()], -1)
        N = np.stack([-dhx.ravel(), np.ones(P.shape[0]), -dhz.ravel()], -1)
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        surf = mesh(stage, "/World/Water/Surface", P, grid_faces(n, n)[:, ::-1], N, material=water)
        surf.GetPrim().CreateAttribute("relight:medium", Sdf.ValueTypeNames.String).Set("boundary")

    # Sand floor, well past the shell's walls.
    n = 60
    gx, gz = np.meshgrid(np.linspace(-450, 450, n + 1), np.linspace(-450, 450, n + 1), indexing="ij")
    gy = 3 * noise3(np.stack([gx, np.zeros_like(gx), gz], -1).reshape(-1, 3) / 150, 3, 1, seed=3)
    P = np.stack([gx.ravel(), gy, gz.ravel()], -1)
    faces = grid_faces(n, n)[:, ::-1]
    UV = np.stack([(gx.ravel() + 450) / 900 * 4, (gz.ravel() + 450) / 900 * 4], -1)
    mesh(stage, "/World/Grotto/Sand", P, faces, vertex_normals(P, faces), UV, material=sand)

    # Half-buried rocks, roughly mirrored.
    for k, (x, z, r) in enumerate([(-215, -150, 60), (205, -140, 64), (-165, 75, 30), (172, 62, 26)]):
        rows, cols = 24, 48
        V, U = np.meshgrid(np.linspace(-np.pi / 2, np.pi / 2, rows + 1), np.linspace(0, 2 * np.pi, cols + 1),
                           indexing="ij")
        d = np.stack([np.cos(V) * np.cos(U), np.sin(V), np.cos(V) * np.sin(U)], -1).reshape(-1, 3)
        P = d * r * np.array([1.2, 0.8, 1.0])
        P *= (1 + 0.18 * noise3(d * 2, 4, 1, seed=10 + k))[:, None]
        P += [x, -r * 0.25, z]
        faces = grid_faces(rows, cols)
        mesh(stage, f"/World/Grotto/Rock{k}", P, faces, vertex_normals(P, faces), material=rock)

    # Shipwreck: the stern half of a hull that broke in two, upright and half-sunk in the sand,
    # seen end-on through its broken front. Its ribs recede as nested arches, every other one
    # tied across by a deck beam, and the open deck lets the light down onto the gem. Hull frame:
    # x across the beam, y up, z along the keel; left and right planking share their breakage,
    # with a little jitter, so the wreck is almost but not quite symmetric.
    hull = UsdGeom.Xform.Define(stage, "/World/Wreck")
    hx = UsdGeom.XformCommonAPI(hull)
    hx.SetTranslate(Gf.Vec3d(0, -22, 0))
    hx.SetRotate(Gf.Vec3f(-3, 0, 2.5))  # settled a little bow-down, with a slight list
    beam, depth, z_front, z_back = 78.0, 95.0, -20.0, -290.0

    def section(s):  # cross-section: s = -1 (port gunwale) .. 0 (keel) .. 1 (starboard gunwale)
        phi = s * np.pi / 2 * 0.95
        return np.array([beam * np.sin(phi), depth * (1 - np.cos(phi)), 0.0])

    edges = np.linspace(0, 1, 10)  # planks per side, from the keel up
    k = 0
    for sa, sb in zip(edges[:-1], edges[1:]):
        broken = rng.uniform(0, 10 + 40 * sb)       # the front edge is more broken higher up
        missing = sb > 0.55 and rng.random() < 0.25  # gaps in the upper planking let light in
        for side in (-1, 1):
            if missing and rng.random() < 0.8:
                continue
            a, b = section(side * sa), section(side * sb)
            c, d = (a + b) / 2, (b - a) * side
            z0 = z_front - broken - rng.uniform(0, 8)
            box(stage, f"/World/Wreck/Plank{k}", (np.linalg.norm(d) * 0.94, 3, z0 - z_back),
                (c[0], c[1], (z0 + z_back) / 2), (0, 0, math.degrees(math.atan2(d[1], d[0]))), wood)
            k += 1
    # Ribs (frames): square bars along the section just inside the planking.
    frames = np.linspace(-45, -275, 9)
    for i, z in enumerate(frames):
        lo, hi = -1.0, 1.0
        if i in (2, 6):  # two snapped frames, one on each side
            lo, hi = (-0.55, 1.0) if i == 2 else (-1.0, 0.6)
        C = np.array([section(t) for t in np.linspace(lo, hi, 48)])
        T = np.gradient(C, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        n_in = np.stack([-T[:, 1], T[:, 0], np.zeros(len(T))], 1)  # in-plane normal, pointing inwards
        C = C + n_in * 6 + np.array([0, 0, z])
        Z = np.array([0, 0, 1.0])
        P = np.concatenate([C + 3.5 * (u * Z + v * n_in) for u, v in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
        n = len(C)
        faces = [[q * n + j, q * n + j + 1, ((q + 1) % 4) * n + j + 1, ((q + 1) % 4) * n + j]
                 for q in range(4) for j in range(n - 1)]
        faces += [[3 * n, 2 * n, n, 0], [n - 1, 2 * n - 1, 3 * n - 1, 4 * n - 1]]
        mesh(stage, f"/World/Wreck/Frame{i}", P, faces, material=wood)
        if i % 2 == 0:  # deck beams tie every other frame into a portal; one has come loose
            top = section(1.0)[1] - 2
            if i == 4:
                box(stage, f"/World/Wreck/DeckBeam{i}", (2 * beam, 8, 9), (-12, top - 14, z), (0, 0, -12), wood)
            else:
                box(stage, f"/World/Wreck/DeckBeam{i}", (2 * beam + 12, 8, 9), (0, top, z), material=wood)
    box(stage, "/World/Wreck/Keel", (12, 12, z_front - z_back - 30), (0, -5, (z_front + z_back) / 2 - 15), material=wood)
    # The transom closes the far end, a dark backdrop for the gem; its top boards are gone.
    for j, y in enumerate(np.arange(8, 70, 10)):
        half = section(1.0)[0] * np.sin(np.arccos(1 - y / depth)) / np.sin(np.pi / 2 * 0.95) - 4
        box(stage, f"/World/Wreck/Transom{j}", (2 * half, 9, 4), (0, y, z_back + 2), material=wood)

    # Two loose planks in front, roughly mirrored.
    for i, (x, z, yaw) in enumerate([(-72, 38, 22), (66, 44, -27)]):
        box(stage, f"/World/LoosePlank{i}", (rng.uniform(80, 100), 3, 15), (x, 0.5, z),
            (0, yaw, rng.uniform(-3, 3)), wood)

    # The giant gem: a brilliant-cut ruby 60 cm across (16 facets around; pavilion, girdle,
    # crown, table), centred in the hull just inside its broken front. Glass in water: water
    # outside, none inside (relight:medium "submerged"), so its ior is relative to water (1.76 / 1.33).
    D = 60.0
    prof = [(0, 0), (D / 2, 0.43 * D), (D / 2, 0.455 * D), (0.41 * D, 0.53 * D), (0.29 * D, 0.61 * D), (0, 0.61 * D)]
    gem = lathe(stage, "/World/Gem", prof, 16, (0, 0, 0), ruby, smooth=False)
    gx = UsdGeom.XformCommonAPI(gem)
    gx.SetTranslate(Gf.Vec3d(0, 4, -60))
    gx.SetRotate(Gf.Vec3f(26, 0, 0))  # table tipped towards the camera, culet in the sand
    gem.GetPrim().CreateAttribute("relight:medium", Sdf.ValueTypeNames.String).Set("submerged")

    # Kelp and fan coral cards, upright, in mirrored pairs.
    def card(path, w, h, at, yaw, m):
        c = mesh(stage, path, [(-w / 2, 0, 0), (w / 2, 0, 0), (w / 2, h, 0), (-w / 2, h, 0)], [[0, 1, 2, 3]],
                 UV=[(0, 0), (1, 0), (1, 1), (0, 1)], material=m)
        api = UsdGeom.XformCommonAPI(c)
        api.SetTranslate(Gf.Vec3d(*at))
        api.SetRotate(Gf.Vec3f(0, yaw, 0))
    card("/World/Kelp/A", 55, 100, (-150, 0, -45), 20, kelp)  # all below the waterline
    card("/World/Kelp/B", 50, 94, (152, 0, -55), -22, kelp)
    card("/World/Kelp/C", 45, 85, (-125, 0, -230), 10, kelp)
    card("/World/Kelp/D", 48, 88, (128, 0, -222), -12, kelp)
    card("/World/Coral/A", 50, 42, (-118, 0, -95), 30, fan)
    card("/World/Coral/B", 46, 40, (121, 0, -102), -30, fan)

    # The lantern (the viewer's starting light), in the air pocket above and behind the gem, so the
    # first thing a visitor sees is caustics: the water's net inside the hull, and the gem's red
    # one in front of it. Small, for crisp caustics.
    lamp = UsdLux.SphereLight.Define(stage, "/World/Lantern")
    lamp.CreateRadiusAttr(3.5)
    lamp.CreateIntensityAttr(1.0)
    lamp.CreateColorAttr(Gf.Vec3f(1.0, 0.82, 0.58))
    UsdGeom.XformCommonAPI(lamp).SetTranslate(Gf.Vec3d(0, 245, -125) if WATER_LEVEL else Gf.Vec3d(-55, 75, 25))

    # Camera: a diver hovering on the hull's centreline, looking in through the broken front; the
    # top of the frame catches the underside of the water surface.
    cam = UsdGeom.Camera.Define(stage, "/World/Camera")
    cam.CreateFocalLengthAttr(24)
    cam.CreateHorizontalApertureAttr(36)
    cam.CreateVerticalApertureAttr(36)
    cam.CreateClippingRangeAttr(Gf.Vec2f(1, 10000))
    eye, target = np.array([0.0, 62.0, 150.0]), np.array([0.0, 30.0, -90.0])
    z = (eye - target) / np.linalg.norm(eye - target)
    xa = np.cross([0, 1, 0], z)
    xa /= np.linalg.norm(xa)
    cam.AddTransformOp().Set(Gf.Matrix4d(*xa, 0, *np.cross(z, xa), 0, *z, 0, *eye, 1))

    stage.GetRootLayer().Save()
    return out / "sunken_wreck.usdc"


if __name__ == "__main__":
    import os
    default = Path(os.environ.get("RELIGHT_WORK", Path.home() / "relight-work")) / "usd" / "sunken_wreck"
    print(build(Path(sys.argv[1]) if len(sys.argv) > 1 else default))
