"""A rough test scene: a giant gem in a sea cave full of beached voyaging canoes, as a USD stage.

    python examples/canoe_cave.py [out_dir]

A high, long cave with a sand floor. Double-hulled voyaging canoes are drawn up in a horseshoe
around a giant brilliant-cut ruby on a low rock in the middle. Their crab-claw sails, patterned
like bark cloth, are translucent (relight:translucency), so they glow when the light is behind
them. Dark rocks frame the view in the foreground. Two starting lights: a warm torch in front of
the gem, and a cool glow deep behind the canoes that lights the sails from behind and throws the
gem's red caustic towards the camera. The air is clear (no medium).
Units are centimetres, Y up.
"""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

rng = np.random.default_rng(5)


# ---------------------------------------------------------------- helpers

def mesh(stage, path, P, faces, N=None, UV=None, material=None):
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
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(np.asarray(UV, np.float32)))
    if material:
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    return m


def grid_faces(rows, cols):
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


def xform(prim, at=(0, 0, 0), rot=(0, 0, 0), scale=None):
    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate(Gf.Vec3d(*map(float, at)))
    api.SetRotate(Gf.Vec3f(*map(float, rot)))
    if scale is not None:
        api.SetScale(Gf.Vec3f(*map(float, scale)))


def box(stage, path, size, at=(0, 0, 0), rot=(0, 0, 0), material=None):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    xform(c, at, rot, size)
    if material:
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(material)
    return c


def tube(stage, path, C, r, material, sides=8):
    """A pole along the polyline C [n, 3] with radius r (scalar or per point)."""
    C = np.asarray(C, float)
    r = np.broadcast_to(np.asarray(r, float), (len(C),))
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    ref = np.where(np.abs(T[:, 1:2]) < 0.9, [[0, 1, 0]], [[1, 0, 0]])
    U = np.cross(T, ref)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    V = np.cross(T, U)
    th = np.linspace(0, 2 * np.pi, sides + 1)
    P = (C[:, None] + r[:, None, None] * (np.cos(th)[None, :, None] * U[:, None] + np.sin(th)[None, :, None] * V[:, None]))
    N = np.cos(th)[None, :, None] * U[:, None] + np.sin(th)[None, :, None] * V[:, None]
    return mesh(stage, path, P.reshape(-1, 3), grid_faces(len(C) - 1, sides), N.reshape(-1, 3), material=material)


def lathe(stage, path, prof, segs, material, smooth=True):
    prof = np.asarray(prof, float)
    d = np.gradient(prof, axis=0)
    n2 = np.stack([d[:, 1], -d[:, 0]], 1)
    n2 /= np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-9)
    th = np.linspace(0, 2 * np.pi, segs + 1)
    c, s = np.cos(th)[None], np.sin(th)[None]
    r, y = prof[:, :1], prof[:, 1:]
    P = np.stack([r * c, np.repeat(y, segs + 1, 1), r * s], -1).reshape(-1, 3)
    N = np.stack([n2[:, :1] * c, np.repeat(n2[:, 1:], segs + 1, 1), n2[:, :1] * s], -1).reshape(-1, 3)
    return mesh(stage, path, P, grid_faces(len(prof) - 1, segs), N if smooth else None, material=material)


def blob(stage, path, at, radii, material, seed, bumps=0.2, sink=0.25):
    """A rock: a noise-displaced ellipsoid, sunk into the sand by `sink` of its height."""
    rows, cols = 32, 64
    V, U = np.meshgrid(np.linspace(-np.pi / 2, np.pi / 2, rows + 1), np.linspace(0, 2 * np.pi, cols + 1), indexing="ij")
    d = np.stack([np.cos(V) * np.cos(U), np.sin(V), np.cos(V) * np.sin(U)], -1).reshape(-1, 3)
    P = d * np.array(radii) * (1 + bumps * noise3(d * 1.5, 5, 1, seed=seed))[:, None]
    P += np.array(at) - [0, radii[1] * sink, 0]
    faces = grid_faces(rows, cols)
    return mesh(stage, path, P, faces, vertex_normals(P, faces), material=material)


# ---------------------------------------------------------------- textures

def save(path, img):
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


def sand_textures(out, n=1024):
    y, x = np.mgrid[0:n, 0:n] / n
    warp = 0.08 * np.sin(2 * np.pi * (y * 1.3 + 0.2)) + 0.03 * np.sin(2 * np.pi * (x * 3.1 + y * 2.2))
    ripple = np.sin(2 * np.pi * 7 * (x + warp))
    height = 0.6 * ripple + 0.15 * rng.normal(size=(n, n))
    gy, gx = np.gradient(height)
    nrm = np.stack([-gx * 4, -gy * 4, np.ones_like(gx)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    save(out / "sand_normal.png", nrm * 0.5 + 0.5)
    grain = rng.normal(size=(n, n)) * 0.05 + 0.03 * ripple
    save(out / "sand.png", np.stack([0.80 + grain, 0.74 + grain, 0.62 + grain * 0.8], -1))


def bark_cloth(path, seed, n=1024):
    """A cream sail painted with bands of geometric motifs (triangles, diamonds, zigzags,
    chevrons, hatching) in dark brown, different for every seed."""
    r = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n
    ink = np.zeros((n, n), bool)
    edges = np.sort(np.concatenate([[0, 1], r.uniform(0.05, 0.95, r.integers(4, 7))]))
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        band = (y >= a) & (y < b)
        v = (y - a) / max(b - a, 1e-6)                     # 0..1 across the band
        reps = r.integers(6, 16)
        u = (x * reps) % 1.0                               # 0..1 within one repeat
        kind = r.integers(0, 6)
        if kind == 0:    # triangles, alternating up and down
            up = ((x * reps).astype(int) % 2) == 0
            m = np.where(up, v > np.abs(u - 0.5) * 2, v < 1 - np.abs(u - 0.5) * 2)
        elif kind == 1:  # diamonds
            m = np.abs(u - 0.5) + np.abs(v - 0.5) < 0.42
        elif kind == 2:  # zigzag line
            m = np.abs(v - (0.25 + 0.5 * np.abs(u - 0.5) * 2)) < 0.12
        elif kind == 3:  # chevrons
            m = ((v + np.abs(u - 0.5) * 1.2) * 3 % 1.0) < 0.4
        elif kind == 4:  # hatching between rules
            m = ((u + v) * 4 % 1.0 < 0.3) | (v < 0.08) | (v > 0.92)
        else:            # plain band
            m = np.ones_like(v, bool) if r.random() < 0.35 else (v < 0.1) | (v > 0.9)
        ink |= band & m
        ink |= (np.abs(y - a) < 0.004)                     # thin rule between bands
    cloth = np.stack([0.86 + 0 * x, 0.80 + 0 * x, 0.66 + 0 * x], -1)
    cloth *= (0.94 + 0.06 * r.normal(size=(n, n)))[..., None]  # fibre noise
    brown = np.array([0.24, 0.14, 0.08])
    img = np.where(ink[..., None], brown, cloth)
    save(path, img)


# ---------------------------------------------------------------- materials

def material(stage, path, color=(0.5, 0.5, 0.5), rough=0.5, metallic=0.0, opacity=1.0, ior=1.5,
             texture=None, normal_map=None, translucency=None):
    m = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior)
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    st = None
    if texture or normal_map:
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
        return t.ConnectableAPI()

    if texture:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex("tex", texture), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    if normal_map:
        n = tex("normal", normal_map, raw=True, scale=(2, 2, 2, 1), bias=(-1, -1, -1, 0))
        sh.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(n, "rgb")
    if translucency:
        m.GetPrim().CreateAttribute("relight:translucency", Sdf.ValueTypeNames.Float).Set(translucency)
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return m


# ---------------------------------------------------------------- canoes

def hull(stage, path, L, beam, depth, material):
    """A slender double-ended hull, open on top: lofted U sections with a rockered keel and
    ends that sweep up."""
    n, m = 64, 24
    t = np.linspace(0, 1, n + 1)
    w = beam / 2 * np.maximum(np.sin(np.pi * t) ** 0.55, 0.03)
    sheer = depth + 55 * (2 * t - 1) ** 6
    keel = 12 * (2 * t - 1) ** 2
    phi = np.linspace(-1, 1, m + 1) * np.pi / 2
    X = w[:, None] * np.sin(phi)[None]
    Y = keel[:, None] + (sheer - keel)[:, None] * (1 - np.cos(phi))[None]
    Z = np.repeat(((t - 0.5) * L)[:, None], m + 1, 1)
    P = np.stack([X, Y, Z], -1).reshape(-1, 3)
    faces = grid_faces(n, m)
    return mesh(stage, path, P, faces, vertex_normals(P, faces), material=material)


def bezier(p0, p1, p2, n=40):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def canoe(stage, path, at, yaw, looks, sail_tex, seed, scale=1.0):
    """A double-hulled voyaging canoe with a deck and a crab-claw sail. Canoe frame: z along the
    hulls (bow +z), x across, y up."""
    r = np.random.default_rng(seed)
    root = UsdGeom.Xform.Define(stage, path)
    xform(root, at, (r.uniform(-2, 2), yaw, r.uniform(-3, 3)), (scale,) * 3)
    L, beam, depth, gap = 820.0, 85.0, 70.0, 150.0
    for side in (-1, 1):
        h = hull(stage, f"{path}/Hull{'LR'[side > 0]}", L * (1 if side < 0 else 0.93), beam, depth, looks["hull"])
        xform(h, (side * gap, 0, 0))
    deck_y = depth + 12
    box(stage, f"{path}/Deck", (2 * gap + beam, 7, 360), (0, deck_y, -20), material=looks["deck"])
    for k, z in enumerate([-190, 150]):  # cross beams lashing the hulls together
        box(stage, f"{path}/Beam{k}", (2 * gap + beam + 40, 10, 12), (0, deck_y - 6, z), material=looks["deck"])

    # Crab-claw sail: two curved spars from a tack near the bow, opening upwards like a claw.
    tack = np.array([0.0, deck_y + 8, 150.0])
    A = bezier(tack, tack + [0, 520, 120], tack + [0, 900, 20] + r.uniform(-25, 25, 3))    # leading spar
    B = bezier(tack, tack + [0, 260, -300], tack + [0, 720, -520] + r.uniform(-25, 25, 3))  # trailing spar
    tube(stage, f"{path}/SparA", A, np.linspace(7, 4, len(A)), looks["spar"])
    tube(stage, f"{path}/SparB", B, np.linspace(7, 4, len(B)), looks["spar"])
    tube(stage, f"{path}/Mast", bezier(np.array([0, deck_y, 10.0]), np.array([0, deck_y + 200, 40.0]),
                                       np.array([0, deck_y + 380, 90.0]), 12), 7, looks["spar"])
    # The sail between the spars, billowing to one side, a little more in the middle.
    nu, nv = 40, 24
    u = np.linspace(0, 1, nu)[:, None]
    v = np.linspace(0, 1, nv)[None, :]
    billow = r.choice([-1, 1]) * 55 * np.sin(np.pi * v) * np.sin(np.pi * np.clip(u * 1.1, 0, 1)) ** 0.7
    S = A[:, None] * (1 - v[..., None]) + B[:, None] * v[..., None]
    S[..., 0] += billow
    # The loose edge between the spar tips sags inwards (the claw's notch).
    tip = (u ** 3) * np.sin(np.pi * v)
    S = S - tip[..., None] * (S - tack) * 0.18
    UV = np.stack([np.broadcast_to(v, (nu, nv)), np.broadcast_to(u, (nu, nv))], -1).reshape(-1, 2)
    P = S.reshape(-1, 3)
    faces = grid_faces(nu - 1, nv - 1)
    mesh(stage, f"{path}/Sail", P, faces, vertex_normals(P, faces), UV, material=sail_tex)
    # Paddles and a steering oar leaning on the hull, for scale.
    tube(stage, f"{path}/SteeringOar", bezier(np.array([gap, deck_y, -300.0]), np.array([gap + 20, 40, -420.0]),
                                              np.array([gap + 30, -10, -520.0]), 8), 4, looks["spar"])


# ---------------------------------------------------------------- scene

def build(out):
    out.mkdir(parents=True, exist_ok=True)
    sand_textures(out)
    for i in range(4):
        bark_cloth(out / f"sail{i}.png", seed=100 + i)

    stage = Usd.Stage.CreateNew(str(out / "canoe_cave.usdc"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    # Lights anywhere among the canoes, up into the sails, and in front of the gem.
    stage.GetRootLayer().customLayerData = {"relight": {
        "lightBox": Vt.Vec3dArray([Gf.Vec3d(-700, 20, -1100), Gf.Vec3d(700, 600, 380)])}}

    L = "/World/Looks/"
    rock = material(stage, L + "CaveRock", (0.09, 0.06, 0.045), rough=0.85)  # dark: the sails carry the image
    dark_rock = material(stage, L + "DarkRock", (0.06, 0.04, 0.03), rough=0.8)
    sand = material(stage, L + "Sand", texture="./sand.png", normal_map="./sand_normal.png", rough=0.95)
    looks = {
        "hull": material(stage, L + "HullWood", (0.36, 0.17, 0.09), rough=0.6),
        "deck": material(stage, L + "DeckWood", (0.42, 0.27, 0.15), rough=0.75),
        "spar": material(stage, L + "Spar", (0.45, 0.34, 0.22), rough=0.7),
    }
    sails = [material(stage, L + f"Sail{i}", texture=f"./sail{i}.png", rough=0.9, translucency=0.45)
             for i in range(4)]
    ruby = material(stage, L + "Ruby", (0.95, 0.12, 0.22), rough=0.0, opacity=0.0, ior=1.76)

    # The cave: a long, high vault (17 x 9 x 22 m), displaced into rock, seen from inside.
    R, C = np.array([850.0, 900.0, 1100.0]), np.array([0.0, 0.0, -250.0])
    rows, cols = 96, 192
    V, U = np.meshgrid(np.linspace(-0.2, 1, rows + 1) * np.pi / 2, np.linspace(0, 2 * np.pi, cols + 1), indexing="ij")
    d = np.stack([np.cos(V) * np.cos(U), np.sin(V), np.cos(V) * np.sin(U)], -1).reshape(-1, 3)
    P = d * R
    disp = noise3(P / 200, octaves=6, base=1.0, seed=1) * 70 + noise3(P / 200, 3, 8, seed=2) * 12
    P = P * (1 + disp[:, None] / np.linalg.norm(P, axis=1, keepdims=True)) + C
    faces = grid_faces(rows, cols)[:, ::-1]  # facing inwards
    mesh(stage, "/World/Cave/Shell", P, faces, vertex_normals(P, faces), material=rock)

    # Sand floor, gently uneven, running under the cave walls.
    n = 120
    gx, gz = np.meshgrid(np.linspace(-900, 900, n + 1), np.linspace(-1500, 800, n + 1), indexing="ij")
    gy = 6 * noise3(np.stack([gx, np.zeros_like(gx), gz], -1).reshape(-1, 3) / 300, 3, 1, seed=3)
    P = np.stack([gx.ravel(), gy, gz.ravel()], -1)
    faces = grid_faces(n, n)[:, ::-1]
    UV = np.stack([(gx.ravel() + 900) / 1800 * 8, (gz.ravel() + 1500) / 2300 * 10], -1)
    mesh(stage, "/World/Cave/Sand", P, faces, vertex_normals(P, faces), UV, material=sand)

    # Foreground rocks framing the view (left, right, and low in the middle), and a few more.
    blob(stage, "/World/Rocks/FrameLeft", (-385, 0, 170), (200, 360, 190), dark_rock, 11, 0.25, 0.15)
    blob(stage, "/World/Rocks/FrameRight", (395, 0, 150), (210, 400, 200), dark_rock, 12, 0.25, 0.15)
    blob(stage, "/World/Rocks/FrameLow", (50, 0, 250), (150, 55, 80), dark_rock, 13, 0.2, 0.3)
    blob(stage, "/World/Rocks/Pedestal", (0, 0, 0), (75, 38, 70), rock, 14, 0.12, 0.35)
    for k, (x, z, s) in enumerate([(-160, 60, 28), (180, 40, 24), (-120, -220, 22), (250, -120, 30)]):
        blob(stage, f"/World/Rocks/Small{k}", (x, 0, z), (s * 1.3, s, s * 1.1), rock, 20 + k)

    # Canoes (at 60 %, so the gem is giant beside them) drawn up in a horseshoe around it, broadside
    # to the camera with their bows to the walls, so every sail leans in over the gem.
    fleet = [(-420, -60, -90), (430, -90, 90), (-380, -520, -80), (390, -560, 80),
             (-200, -900, -100), (230, -940, 100), (30, -1180, -90)]
    for i, (x, z, yaw) in enumerate(fleet):
        canoe(stage, f"/World/Canoe{i}", (x, -8, z), yaw, looks, sails[i % 4], seed=200 + i, scale=0.6)

    # The giant gem: a brilliant-cut ruby 1.2 m across on its rock, table tipped to the camera.
    D = 120.0
    prof = [(0, 0), (D / 2, 0.43 * D), (D / 2, 0.455 * D), (0.41 * D, 0.53 * D), (0.29 * D, 0.61 * D), (0, 0.61 * D)]
    gem = lathe(stage, "/World/Gem", prof, 16, ruby, smooth=False)
    xform(gem, (0, 12, 0), (24, 11.25, 0))

    # Two starting lights: the torch (warm, in front of the gem) and a cool glow deep in the
    # cave behind the canoes, which backlights the sails and the gem.
    for name, pos, radius, color in [("Torch", (-110, 115, 120), 9, (1.0, 0.52, 0.22)),
                                     ("CaveGlow", (0, 470, -1000), 25, (0.45, 0.95, 0.85))]:
        l = UsdLux.SphereLight.Define(stage, f"/World/{name}")
        l.CreateRadiusAttr(radius)
        l.CreateIntensityAttr(1.0 if name == "Torch" else 2.0)
        l.CreateColorAttr(Gf.Vec3f(*color))
        xform(l, pos)

    # Camera: low, between the framing rocks, looking up past the gem into the sails.
    cam = UsdGeom.Camera.Define(stage, "/World/Camera")
    cam.CreateFocalLengthAttr(20)
    cam.CreateHorizontalApertureAttr(36)
    cam.CreateVerticalApertureAttr(36)
    cam.CreateClippingRangeAttr(Gf.Vec2f(1, 100000))
    eye, target = np.array([0.0, 95.0, 400.0]), np.array([0.0, 250.0, -400.0])
    z = (eye - target) / np.linalg.norm(eye - target)
    xa = np.cross([0, 1, 0], z)
    xa /= np.linalg.norm(xa)
    cam.AddTransformOp().Set(Gf.Matrix4d(*xa, 0, *np.cross(z, xa), 0, *z, 0, *eye, 1))

    stage.GetRootLayer().Save()
    return out / "canoe_cave.usdc"


if __name__ == "__main__":
    import os
    default = Path(os.environ.get("RELIGHT_WORK", Path.home() / "relight-work")) / "usd" / "canoe_cave"
    print(build(Path(sys.argv[1]) if len(sys.argv) > 1 else default))
