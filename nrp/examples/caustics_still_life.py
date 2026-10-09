"""A rough caustics test scene: a glassware still life in a plaster niche, as a USD stage.

    python examples/caustics_still_life.py [out_dir]
    python sample_paths.py --scene <out_dir>/still_life.usda --spp 128 --max-seg 8 --radius-range 0.02 0.1

Everything is built procedurally (lathed profiles) so the layout is easy to change:
  - a round flask of water: a ball lens, throws a bright focused spot and ring on the cloth
  - a wine glass with red wine: a red caustic (light passes glass, wine, glass: up to 7 bounces,
    hence --max-seg 8)
  - a polished copper ring: the heart-shaped reflection caustic (cardioid) on the cloth inside it
  - an octagonal cut tumbler: flat facets that scatter sparkles (the hardest case)
The niche is enclosed apart from its front, which the camera looks through, and the tablecloth runs
out towards the camera, so no camera ray escapes. One small "Candle" sphere light sits behind the
glassware, so the caustics fall forwards onto the cloth, towards the viewer.
Units are centimetres, Y up.
"""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

SEGS = 128  # around the axis, for the smooth objects


# ---------------------------------------------------------------- lathe

def lathe(stage, path, runs, segs=SEGS, smooth=True, at=(0, 0, 0)):
    """A surface of revolution around +y. `runs` is a list of (r, y) polylines, walked so that
    the solid is on the left (e.g. from the axis at the bottom, up the outside, down the inside);
    consecutive runs meet at hard edges. With smooth=False the facets and edges are all hard."""
    rows, normals = [], []
    for run in runs:
        run = np.asarray(run, float)
        d = np.gradient(run, axis=0)
        n = np.stack([d[:, 1], -d[:, 0]], 1)  # tangent (dr, dy) turned outwards
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        rows.append(run)
        normals.append(n)
    prof, nprof = np.concatenate(rows), np.concatenate(normals)
    # A hard edge between runs: the shared point appears in both, with a zero-area band between.
    th = np.linspace(0, 2 * np.pi, segs + 1)
    c, s = np.cos(th)[None], np.sin(th)[None]
    r, y = prof[:, :1], prof[:, 1:]
    P = np.stack([r * c, np.repeat(y, segs + 1, 1), r * s], -1).reshape(-1, 3)
    N = np.stack([nprof[:, :1] * c, np.repeat(nprof[:, 1:], segs + 1, 1), nprof[:, :1] * s], -1).reshape(-1, 3)
    R, C = len(prof) - 1, segs
    i, j = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
    a = (i * (C + 1) + j).ravel()
    quads = np.stack([a, a + C + 1, a + C + 2, a + 1], 1)  # outward: d(profile) x d(angle)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in P])
    mesh.CreateFaceVertexCountsAttr([4] * len(quads))
    mesh.CreateFaceVertexIndicesAttr(quads.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    if smooth:
        mesh.CreateNormalsAttr([Gf.Vec3f(*n) for n in N])
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    UsdGeom.XformCommonAPI(mesh).SetTranslate(Gf.Vec3d(*at))
    return mesh


def curve(t, keys):
    """Smooth radius profile through (t, r) keys (monotone cubic-ish via dense interpolation)."""
    kt, kr = zip(*keys)
    dense = np.linspace(0, 1, 400)
    r = np.interp(dense, kt, kr)
    r = np.convolve(np.pad(r, 20, mode="edge"), np.ones(41) / 41, mode="valid")  # round the kinks
    return np.interp(t, dense, r)


# ---------------------------------------------------------------- materials

def material(stage, path, color, rough=0.5, metallic=0.0, opacity=1.0, ior=1.5, texture=None):
    m = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior)
    if texture:
        rd = UsdShade.Shader.Define(stage, path + "/st")
        rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tx = UsdShade.Shader.Define(stage, path + "/tex")
        tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
        tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return m


def bind(prim, m):
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(m)


def quad(stage, path, corners, uv_scale=1.0):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(*c) for c in corners])
    m.CreateFaceVertexCountsAttr([4])
    m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(uv_scale, 0), Gf.Vec2f(uv_scale, uv_scale), Gf.Vec2f(0, uv_scale)])
    return m


def linen(path, n=1024):
    """Off-white linen: a fine irregular weave with slub threads."""
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:n, 0:n]
    warp = 0.5 + 0.5 * np.sin(x * 2 * np.pi / 6 + rng.normal(0, 0.3, (1, n)))
    weft = 0.5 + 0.5 * np.sin(y * 2 * np.pi / 6 + rng.normal(0, 0.3, (n, 1)))
    slub = rng.normal(0, 1, (1, n)) * 0.04 + rng.normal(0, 1, (n, 1)) * 0.04
    v = 0.82 + 0.05 * (warp * weft - 0.25) + slub + rng.normal(0, 0.015, (n, n))
    img = np.stack([v, v * 0.975, v * 0.93], -1)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


# ---------------------------------------------------------------- scene

def build(out):
    out.mkdir(parents=True, exist_ok=True)
    linen(out / "linen.png")
    stage = Usd.Stage.CreateNew(str(out / "still_life.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())

    plaster = material(stage, "/World/Looks/Plaster", (0.78, 0.76, 0.72), rough=0.9)
    teal = material(stage, "/World/Looks/Teal", (0.07, 0.28, 0.31), rough=0.8)
    cloth = material(stage, "/World/Looks/Linen", None, rough=0.9, texture="./linen.png")
    glass = material(stage, "/World/Looks/Glass", (1, 1, 1), rough=0.0, opacity=0.0, ior=1.5)
    water = material(stage, "/World/Looks/Water", (0.97, 0.99, 1.0), rough=0.0, opacity=0.0, ior=1.36)
    # Transmittance per interface crossing; light crosses the wine twice, so this is ~its square root.
    wine = material(stage, "/World/Looks/Wine", (0.86, 0.2, 0.24), rough=0.0, opacity=0.0, ior=1.35)
    copper = material(stage, "/World/Looks/Copper", (0.95, 0.64, 0.54), rough=0.06, metallic=1.0)

    # Niche: 120 wide, 80 tall, 70 deep, open towards +z; the cloth runs out to z = 90.
    W, H, D0, D1, F = 60.0, 80.0, -35.0, 35.0, 90.0
    bind(quad(stage, "/World/Niche/Cloth", [(-W, 0, F), (W, 0, F), (W, 0, D0), (-W, 0, D0)], uv_scale=3), cloth)
    bind(quad(stage, "/World/Niche/Back", [(-W, 0, D0), (W, 0, D0), (W, H, D0), (-W, H, D0)]), plaster)
    bind(quad(stage, "/World/Niche/Left", [(-W, 0, D1), (-W, 0, D0), (-W, H, D0), (-W, H, D1)]), teal)
    bind(quad(stage, "/World/Niche/Right", [(W, 0, D0), (W, 0, D1), (W, H, D1), (W, H, D0)]), plaster)
    bind(quad(stage, "/World/Niche/Ceiling", [(-W, H, D0), (W, H, D0), (W, H, D1), (-W, H, D1)]), plaster)

    # Flask of water, treated as one solid of water (thin glass hardly changes its caustic).
    t = np.linspace(0, 1, 90)
    a = -math.pi / 2 + t * (math.pi - 0.2)  # sphere of radius 11 from its flattened bottom to the neck
    sphere = np.stack([11 * np.cos(a), 11 + 11 * np.sin(a)], 1)
    neck_y = sphere[-1, 1]
    r_base = math.sqrt(11 ** 2 - (11 - 0.35) ** 2)
    flask = [
        [(0, 0.35), (r_base, 0.35)],                                 # flat base
        [(r_base, 0.35)] + [tuple(p) for p in sphere[sphere[:, 1] > 0.35]] + [(2.2, neck_y + 0.6), (2.2, 33)],
        [(2.2, 33), (2.6, 33.4), (2.6, 34)],                         # lip
        [(2.6, 34), (0, 34)],                                        # water surface at the top
    ]
    bind(lathe(stage, "/World/Flask", flask, at=(-21, 0, -12)), water)

    # Wine glass: a thin shell, and a separate body of wine just inside it.
    tb = np.linspace(0, 1, 60)
    keys = [(0, 0.5), (0.12, 2.4), (0.35, 3.9), (0.6, 4.2), (0.85, 3.95), (1, 3.7)]
    bowl_y = 7.4 + 10.6 * tb
    outer = curve(tb, keys)
    inner = outer - 0.25
    inside = inner > 0.15
    wine_glass = [
        [(0, 0), (3.8, 0)],
        [(3.8, 0), (3.8, 0.3)],
        [(3.8, 0.3), (1.2, 0.6), (0.45, 1.3), (0.4, 4), (0.42, 7.0)] + list(zip(outer, bowl_y)),
        [(outer[-1], 18.0), (inner[-1], 18.0)],
        list(zip(inner[inside][::-1], bowl_y[inside][::-1])) + [(0, bowl_y[inside][0] - 0.1)],
    ]
    bind(lathe(stage, "/World/WineGlass/Glass", wine_glass, at=(1, 0, -15)), glass)
    level = 13.0
    wl = inside & (bowl_y <= level)
    wine_r = np.maximum(inner[wl] - 0.05, 0)
    r_top = float(np.interp(level, bowl_y[inside], inner[inside])) - 0.05
    wine_body = [
        [(0, bowl_y[wl][0] - 0.05)] + list(zip(wine_r, bowl_y[wl])) + [(r_top, level)],
        [(r_top, level), (0, level)],
    ]
    bind(lathe(stage, "/World/WineGlass/Wine", wine_body, at=(1, 0, -15)), wine)

    # Copper ring lying on the cloth: its inside wall throws the cardioid.
    ring = [[(8.2, 0.05), (8.8, 0.05)], [(8.8, 0.05), (8.8, 5.0)], [(8.8, 5.0), (8.2, 5.0)], [(8.2, 5.0), (8.2, 0.05)]]
    bind(lathe(stage, "/World/CopperRing", ring, at=(13, 0, 9)), copper)

    # Cut tumbler: eight flat facets, all edges hard.
    tumbler = [[(0, 0), (4.0, 0)], [(4.0, 0), (4.5, 9.5)], [(4.5, 9.5), (4.1, 9.5)],
               [(4.1, 9.5), (3.7, 2.0)], [(3.7, 2.0), (0, 2.0)]]
    bind(lathe(stage, "/World/Tumbler", tumbler, segs=8, smooth=False, at=(21, 0, -19)), glass)

    # The candle: small, low and behind the glassware, so caustics stretch towards the camera
    # (and the low angle lets the ring throw its cardioid).
    candle = UsdLux.SphereLight.Define(stage, "/World/Candle")
    candle.CreateRadiusAttr(1.6)
    candle.CreateIntensityAttr(1.0)
    candle.CreateColorAttr(Gf.Vec3f(1.0, 0.82, 0.6))
    UsdGeom.XformCommonAPI(candle).SetTranslate(Gf.Vec3d(-6, 21, -30))

    # Camera outside the niche, a little above the table, looking down onto the glassware.
    cam = UsdGeom.Camera.Define(stage, "/World/Camera")
    cam.CreateFocalLengthAttr(40)
    cam.CreateHorizontalApertureAttr(36)
    cam.CreateVerticalApertureAttr(36)
    cam.CreateClippingRangeAttr(Gf.Vec2f(1, 10000))
    eye, target = np.array([0.0, 50.0, 76.0]), np.array([0.0, 5.0, -3.0])
    z = (eye - target) / np.linalg.norm(eye - target)
    x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    M = Gf.Matrix4d(*x, 0, *y, 0, *z, 0, *eye, 1)
    cam.AddTransformOp().Set(M)

    stage.GetRootLayer().Save()
    return out / "still_life.usda"


if __name__ == "__main__":
    import os
    default = Path(os.environ.get("RELIGHT_WORK", Path.home() / "relight-work")) / "usd" / "still_life"
    print(build(Path(sys.argv[1]) if len(sys.argv) > 1 else default))
