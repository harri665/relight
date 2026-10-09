"""A glowing giant gem in a flooded sea cave, ringed by a sunken fleet of voyaging canoes.

    python examples/flooded_canoe_cave.py [out_dir]

Merges canoe_cave.py (a dark vault, canoes with tall sails, the giant ruby) with sunken_wreck.py
(water whose waving surface throws caustics). Stylised rather than to scale: the water is only
WATER_LEVEL (3 m) deep, and the canoes are shrunk to fit under it, so the 1.2 m ruby towers over
them. The ruby, a round brilliant (57 facets, cut to real proportions so light bounces around
inside it), sits on a boulder in a shallow dish in the middle of the seabed, with a light inside
it: a glowing red heart that lights the fleet from the middle and throws red caustics around.
Around it, in a tight horseshoe, lie the canoes, sails still up: copies of the Smithsonian's CC0
scan of Queen Kapi'olani's wa'a (assets/kapiolani_canoe, see its LICENSE.txt), and canoe_cave's
double-hulled canoes with patterned crab-claw sails, which let light through
(relight:translucency). Rocks, the cave wall and the seabed are Poly Haven scans and photo
textures (CC0), fetched by assets/fetch_polyhaven.py into relight-work (not the repo).
The camera is under water between scanned boulders, close to the gem, aimed at its centre.
Starting lights, none of them in frame: the "Sun", 2 m above the water behind the camera, which
the short surface ripples focus into a crisp caustic net over everything; the gem's heart; and
a teal glow hidden behind a boulder at the back, which rims the fleet.
Media: water by default (the camera too); the water surface is tagged "boundary" (water below
it), the vault above the waterline "air", and the gem "submerged".
Units are centimetres, Y up.
"""
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canoe_cave import bark_cloth, canoe, grid_faces, material, mesh, noise3, vertex_normals, xform  # noqa: E402
from sunken_wreck import ocean  # noqa: E402
from assets.fetch_polyhaven import fetch  # noqa: E402

CANOE = Path(__file__).resolve().parent / "assets" / "kapiolani_canoe"
CANOE_KEEL = -19.0      # the scan's lowest point (its own coordinates)
WATER_LEVEL = 300.0     # cm above the seabed, over the tallest (shrunken) sail
SUN_HEIGHT = 200.0      # the Sun, above the water
BEACH = 22.0            # the seabed around the gem's dish
LAGOON = dict(centre=(0.0, 0.0), radii=(260.0, 220.0), drop=50.0)  # the dish the gem sits in
# The ripples on the surface (see sunken_wreck.ocean). For a light a distance u above the water,
# crests of curvature k focus at v below it, n / v = (n - 1) k - 1 / u (n = 1.33). With the Sun
# 2 m up and the seabed 3 m down, short waves (15-60 cm) with k = 0.035/cm give a crisp net of
# cells ~40-60 cm across (tested in isolation: longer swells give metre-wide blobs).
OCEAN = dict(n_waves=150, lam=(15.0, 60.0), wind_deg=35.0, spread=2.0, curvature=0.035, seed=11)
# Sunken canoes in a horseshoe around the gem: (kind, x, z, heading in degrees with 0 = bow
# along +z, scale). "scan" is the Smithsonian wa'a, "double" canoe_cave's canoe, both shrunk to
# fit under the water. At the sides they lie diagonally, bows into the cave, so their sails face
# the camera at an angle; at the back they lie broadside.
FLEET = [("scan", -300, 40, 150, 0.7), ("double", -330, -250, 118, 0.26), ("scan", -170, -470, 105, 0.68),
         ("double", 310, 20, 212, 0.27), ("scan", 320, -260, 238, 0.7), ("double", 180, -480, 250, 0.26),
         ("scan", 0, -620, 90, 0.7), ("double", -40, -760, 92, 0.26)]


def tag(prim, medium):
    prim.GetPrim().CreateAttribute("relight:medium", Sdf.ValueTypeNames.String).Set(medium)


def split_mesh(stage, path, P, F, N, UV, mat, below="water", above=None):
    """Writes a mesh as the faces under the waterline (tagged `below`) and those over it (tagged
    `above`, or untagged), split by face centroid."""
    over = P[F].mean(1)[:, 1] > WATER_LEVEL
    for part, sel, medium in (("Wet", ~over, below), ("Dry", over, above)):
        if sel.any():
            m = mesh(stage, f"{path}{part}", P, F[sel], N, UV, material=mat)
            if medium:
                tag(m, medium)


def sand_height(x, z):
    """The seabed, sloping down into the gem's bowl."""
    (cx, cz), (rx, rz) = LAGOON["centre"], LAGOON["radii"]
    r = np.sqrt(((x - cx) / rx) ** 2 + ((z - cz) / rz) ** 2)
    t = np.clip((1.0 - r) / 0.45, 0, 1)
    bowl = t * t * (3 - 2 * t)  # smoothstep: 0 at the rim (r = 1), 1 in the middle (r < 0.55)
    return BEACH - LAGOON["drop"] * bowl


def scan_canoe(out, stage):
    """The scan's parts (name, points, triangles, normals, uvs, material), with its textures
    copied next to the stage and materials defined in `stage`."""
    src = Usd.Stage.Open(str(CANOE / "canoe.usdc"))
    tex_dir = out / "canoe_textures"
    tex_dir.mkdir(exist_ok=True)
    parts = []
    for prim in src.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(prim)
        sh = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0].ComputeSurfaceSource()[0]
        files = {}
        for name in ("diffuseColor", "normal"):
            tex = sh.GetInput(name).GetConnectedSource()[0]
            f = Path(UsdShade.Shader(tex.GetPrim()).GetInput("file").Get().resolvedPath)
            shutil.copy(f, tex_dir / f.name)
            files[name] = f"./canoe_textures/{f.name}"
        # Wood, rope and pandanus: never metallic (the scan's glTF left metallic at its default 1).
        mat = material(stage, f"/World/Looks/Scan{prim.GetName()}",
                       rough=max(float(sh.GetInput("roughness").Get()), 0.6),
                       texture=files["diffuseColor"], normal_map=files["normal"],
                       translucency=0.3 if prim.GetName() == "Sail" else None)
        parts.append((prim.GetName(), np.array(m.GetPointsAttr().Get(), float),
                      np.array(m.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3),
                      np.array(m.GetNormalsAttr().Get(), float),
                      np.array(UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").Get(), float), mat))
    return parts


def textured(stage, path, maps, tile_tint=1.0, rough_scale=1.0, translucency=None):
    """UsdPreviewSurface from a Poly Haven texture set ({"diff", "nor_gl", "rough"} files):
    colour darkened by `tile_tint`, OpenGL normal map, roughness from its red channel."""
    m = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    rd = UsdShade.Shader.Define(stage, path + "/st")
    rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    def tex(name, file, raw, scale, bias=(0, 0, 0, 0)):
        t = UsdShade.Shader.Define(stage, f"{path}/{name}")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(file))
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw" if raw else "sRGB")
        t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
        t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
        t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        t.CreateOutput("r", Sdf.ValueTypeNames.Float)
        return t.ConnectableAPI()
    k = tile_tint
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex("diff", maps["diff"], False, (k, k, k, 1)), "rgb")
    sh.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
        tex("normal", maps["nor_gl"], True, (2, 2, 2, 1), (-1, -1, -1, 0)), "rgb")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
        tex("rough", maps["rough"], True, (rough_scale,) * 4), "r")
    if translucency:
        m.GetPrim().CreateAttribute("relight:translucency", Sdf.ValueTypeNames.Float).Set(translucency)
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return m


def scanned_rock(stage, path, usd, at, yaw=0.0, scale=1.0, material=None):
    """A Poly Haven rock (Z-up, metres) referenced in, turned Y-up and scaled to centimetres;
    `material` overrides the scan's own (a binding stronger than its descendants')."""
    x = UsdGeom.Xform.Define(stage, path)
    x.GetPrim().GetReferences().AddReference(str(usd))
    xform(x, at, (-90, yaw, 0), (100 * scale,) * 3)
    if material:
        UsdShade.MaterialBindingAPI.Apply(x.GetPrim()).Bind(material, UsdShade.Tokens.strongerThanDescendants)
    return x


def brilliant(stage, path, D, material):
    """A round brilliant, 57 facets, cut to classic proportions (table 53 %, crown angle 34.5 deg,
    pavilion angle 40.75 deg, which make light bounce around inside and come back out the top):
    crown = table, 8 stars, 8 bezels (kites), 16 upper girdle facets; pavilion = 8 mains (kites),
    16 lower girdle facets. Culet at the origin, table up; flat-shaded triangles."""
    R = D / 2
    rt, girdle = 0.53 * R, 0.03 * D
    crown = (R - rt) * math.tan(math.radians(34.5))
    pav = R * math.tan(math.radians(40.75))
    base = pav + girdle  # height of the girdle's top above the culet
    rs = rt + 0.5 * (R - rt)  # the stars reach halfway down the crown
    rl = 0.2 * R  # the lower girdle facets reach 80 % of the way to the culet
    ring = lambda r, y, n, off: [(r * math.cos(math.radians(off + 360 * i / n)), y,
                                  r * math.sin(math.radians(off + 360 * i / n))) for i in range(n)]
    T = ring(rt, base + crown, 8, 0)
    S = ring(rs, base + crown * (R - rs) / (R - rt), 8, 22.5)
    G = ring(R, base, 16, 0)
    Gb = ring(R, pav, 16, 0)
    M = ring(rl, pav * rl / R, 8, 22.5)
    C = [(0.0, 0.0, 0.0)]
    P = np.array(T + S + G + Gb + M + C)
    t, s_, g, gb, m, c = 0, 8, 16, 32, 48, 56
    tris = [(t, t + i, t + i + 1) for i in range(1, 7)]  # table
    for k in range(8):
        k1 = (k + 1) % 8
        tris.append((t + k, t + k1, s_ + k))  # star
        tris += [(t + k, s_ + (k - 1) % 8, g + 2 * k), (t + k, g + 2 * k, s_ + k)]  # bezel kite
        tris += [(s_ + k, g + 2 * k, g + 2 * k + 1), (s_ + k, g + 2 * k + 1, g + (2 * k + 2) % 16)]  # upper girdle
        tris += [(gb + 2 * k, m + (k - 1) % 8, c), (gb + 2 * k, c, m + k)]  # pavilion main kite
        tris += [(gb + 2 * k, gb + 2 * k + 1, m + k), (gb + 2 * k + 1, gb + (2 * k + 2) % 16, m + k)]  # lower girdle
    for j in range(16):  # the girdle band
        j1 = (j + 1) % 16
        tris += [(g + j, gb + j, gb + j1), (g + j, gb + j1, g + j1)]
    F = np.array(tris)
    # Wind every facet outwards (the stone is convex).
    mid = np.array([0, base * 0.8, 0])
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    flip = (n * (P[F].mean(1) - mid)).sum(1) < 0
    F[flip] = F[flip][:, ::-1]
    return mesh(stage, path, P, F, material=material), base + crown


def water_grid(lo, hi, fine_lo, fine_hi, fine, coarse):
    """Coordinates from lo to hi: `fine` spacing between fine_lo and fine_hi, `coarse` outside."""
    return np.unique(np.concatenate([np.arange(lo, fine_lo, coarse), np.arange(fine_lo, fine_hi, fine),
                                     np.arange(fine_hi, hi + coarse, coarse)]))


def build(out):
    out.mkdir(parents=True, exist_ok=True)
    ph = fetch()  # Poly Haven scans and textures (downloaded once)
    for i in range(4):
        bark_cloth(out / f"sail{i}.png", seed=100 + i)

    stage = Usd.Stage.CreateNew(str(out / "flooded_canoe_cave.usdc"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    stage.GetRootLayer().customLayerData = {"relight": {
        # Clear tropical water, per metre: red is gone within a few metres, the far end of the
        # cave fades into teal; a little forward scattering gives the lights a halo.
        "medium": {"sigma_a": Gf.Vec3d(0.28, 0.05, 0.035), "sigma_s": Gf.Vec3d(0.015, 0.02, 0.025), "g": 0.85,
                   "default": "water", "camera": "water"},
        # Lights anywhere in the fleet, and up above the water for caustics.
        "lightBox": Vt.Vec3dArray([Gf.Vec3d(-550, 25, -900), Gf.Vec3d(550, WATER_LEVEL + 260, 520)])}}

    L = "/World/Looks/"
    cave_rock = textured(stage, L + "CaveRock", ph["rock_face_03"], tile_tint=0.45)
    sand = textured(stage, L + "Sand", ph["coast_sand_01"], tile_tint=0.9)
    # Slightly frosted, so the light inside spreads through the whole stone (a glow, not a bulb).
    ruby = material(stage, L + "Ruby", (0.95, 0.12, 0.22), rough=0.12, opacity=0.0, ior=1.76 / 1.33)
    water = material(stage, L + "Water", (1, 1, 1), rough=0.0, opacity=0.0, ior=1.33)
    looks = {
        "hull": material(stage, L + "HullWood", (0.30, 0.15, 0.08), rough=0.6),
        "deck": material(stage, L + "DeckWood", (0.36, 0.23, 0.13), rough=0.75),
        "spar": material(stage, L + "Spar", (0.40, 0.30, 0.19), rough=0.7),
    }
    sails = [material(stage, L + f"Sail{i}", texture=f"./sail{i}.png", rough=0.9, translucency=0.45)
             for i in range(4)]

    # The cave: a vault 14 m wide, 17 m deep and ~8 m high, displaced into rock, seen from inside,
    # with the rock photo texture wrapped around it (~3 m per repeat).
    R, C = np.array([700.0, 800.0, 850.0]), np.array([0.0, 0.0, -150.0])
    rows, cols = 112, 192
    v, u = np.linspace(-0.2, 1, rows + 1) * np.pi / 2, np.linspace(0, 2 * np.pi, cols + 1)
    V, U = np.meshgrid(v, u, indexing="ij")
    d = np.stack([np.cos(V) * np.cos(U), np.sin(V), np.cos(V) * np.sin(U)], -1).reshape(-1, 3)
    P = d * R
    disp = noise3(P / 200, octaves=6, base=1.0, seed=1) * 70 + noise3(P / 200, 3, 8, seed=2) * 12
    P = P * (1 + disp[:, None] / np.linalg.norm(P, axis=1, keepdims=True)) + C
    UV = np.stack([U.ravel() * 780 / 300, V.ravel() * 800 / 300], -1)
    faces = grid_faces(rows, cols)[:, ::-1]  # facing inwards
    tri = np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]])
    split_mesh(stage, "/World/Cave/Shell", P, tri, vertex_normals(P, faces), UV, cave_rock, below=None, above="air")

    # Sand: the seabed and the gem's dish, with the sand photo texture (~2 m per repeat).
    xs, zs = np.linspace(-750, 750, 241), np.linspace(-1100, 750, 301)
    gx, gz = np.meshgrid(xs, zs, indexing="ij")
    ripple = 4 * noise3(np.stack([gx, np.zeros_like(gx), gz], -1).reshape(-1, 3) / 300, 3, 1, seed=3)
    P = np.stack([gx.ravel(), sand_height(gx, gz).ravel() + ripple, gz.ravel()], -1)
    faces = grid_faces(len(xs) - 1, len(zs) - 1)[:, ::-1]
    UV = np.stack([gx.ravel() / 200, gz.ravel() / 200], -1)
    tri = np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]])
    mesh(stage, "/World/Cave/Sand", P, tri, vertex_normals(P, faces), UV, material=sand)

    # The water surface, normal up ("boundary": water below it): 2 cm spacing over the cave's
    # floor, where the caustics come from, coarse towards the walls (which cut it off).
    (cx, cz), (rx, rz) = LAGOON["centre"], LAGOON["radii"]
    wx = water_grid(-750, 750, -500, 500, 2.0, 10.0)
    wz = water_grid(-1100, 750, -850, 560, 2.0, 10.0)
    WX, WZ = np.meshgrid(wx, wz, indexing="ij")
    h, dhx, dhz = ocean(WX, WZ, **OCEAN)
    print(f"  water surface: {WX.size / 1e6:.2f} M vertices, height rms {h.std():.2f} cm, "
          f"max slope {np.hypot(dhx, dhz).max():.2f}")
    P = np.stack([WX.ravel(), WATER_LEVEL + h.ravel(), WZ.ravel()], -1)
    N = np.stack([-dhx.ravel(), np.ones(P.shape[0]), -dhz.ravel()], -1)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    tag(mesh(stage, "/World/Water/Surface", P, grid_faces(len(wx) - 1, len(wz) - 1)[:, ::-1], N, material=water),
        "boundary")

    # Scanned rocks (their shapes, with the dark cave rock over their own pale granite): two
    # boulders framing the view, the gem's plinth, one at the back hiding the teal glow, and
    # pebbles (rock_09, scaled up) on the seabed.
    floor = float(sand_height(np.array(cx), np.array(cz)))
    dark = textured(stage, L + "ScannedRock", ph["rock_face_03"], tile_tint=0.3)
    scanned_rock(stage, "/World/Rocks/FrameLeft", ph["namaqualand_boulder_03"], (-175, -20, 175), 35, 0.7, dark)
    scanned_rock(stage, "/World/Rocks/FrameRight", ph["boulder_01"], (165, -20, 185), -50, 1.1, dark)
    scanned_rock(stage, "/World/Rocks/Plinth", ph["namaqualand_boulder_05"], (cx, floor - 14, cz), 100, 0.85, dark)
    scanned_rock(stage, "/World/Rocks/Back", ph["boulder_01"], (-20, -12, -700), 15, 1.6, dark)
    for k, (x, z, yaw, sc) in enumerate([(-190, 60, 20, 5.0), (175, 30, 80, 4.2), (-120, -200, 140, 3.6),
                                         (150, -170, 200, 4.6), (-70, 130, 60, 2.6), (95, 110, 300, 2.2)]):
        scanned_rock(stage, f"/World/Rocks/Pebble{k}", ph["rock_09"],
                     (x, float(sand_height(np.array(x), np.array(z))) - 2, z), yaw, sc, dark)

    # The giant gem: a round brilliant 1.2 m across on its plinth, table tipped towards the
    # camera. Glass in water: water outside, none inside, so its ior is relative to water (1.76 / 1.33).
    D, tilt = 120.0, 24.0
    gem, height = brilliant(stage, "/World/Gem", D, ruby)
    gem_at = np.array([cx, floor + 22, cz])
    xform(gem, gem_at, (tilt, 11.25, 0))
    tag(gem, "submerged")
    # Its middle, where the heart light goes (the culet is at the origin, the axis tipped by `tilt`).
    h_mid = 0.42 * height
    heart = gem_at + np.array([0, h_mid * math.cos(math.radians(tilt)), h_mid * math.sin(math.radians(tilt))])

    # The sunken canoes, resting on the seabed.
    parts = scan_canoe(out, stage)
    for i, (kind, x, z, heading, scale) in enumerate(FLEET):
        ground = float(sand_height(np.array(x), np.array(z)))
        if kind == "double":
            canoe(stage, f"/World/Canoe{i}", (x, ground - 6, z), heading, looks, sails[i % 4], seed=200 + i,
                  scale=scale)
            continue
        a, roll = math.radians(heading), math.radians(6 if x < 0 else -6)  # settled, leaning a little
        Rz = np.array([[math.cos(roll), math.sin(roll), 0], [-math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
        Ry = np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]])  # row vectors
        M = scale * Rz @ Ry
        offset = np.array([x, ground - CANOE_KEEL * scale - 4, z])
        UsdGeom.Xform.Define(stage, f"/World/Canoe{i}")
        for name, P, F, N, UV, mat in parts:
            mesh(stage, f"/World/Canoe{i}/{name}", P @ M + offset, F, N @ (Rz @ Ry), UV, material=mat)

    # Lights, none in frame: the Sun above the water behind the camera (the key: caustics over
    # everything), the gem's heart (big and dim, filling the stone, so it glows from within rather
    # than showing a hot spot), and a teal glow behind the back boulder (a rim on the fleet).
    # Power goes with radiance x area; the small Sun needs a high radiance to lead.
    for name, pos, radius, color, power in [("Sun", (0, WATER_LEVEL + SUN_HEIGHT, 470), 8, (1.0, 0.93, 0.8), 300.0),
                                            ("GemHeart", tuple(heart), 22, (1.0, 0.3, 0.25), 4.0),
                                            ("CaveGlow", (-20, 110, -820), 25, (0.45, 0.95, 0.85), 5.0)]:
        light = UsdLux.SphereLight.Define(stage, f"/World/{name}")
        light.CreateRadiusAttr(radius)
        light.CreateIntensityAttr(power)
        light.CreateColorAttr(Gf.Vec3f(*color))
        xform(light, pos)

    # Camera: under water between the framing boulders, close to the gem and aimed at its middle,
    # so it fills the centre of the frame with the fleet around it and the surface above.
    cam = UsdGeom.Camera.Define(stage, "/World/Camera")
    cam.CreateFocalLengthAttr(22)
    cam.CreateHorizontalApertureAttr(36)
    cam.CreateVerticalApertureAttr(36)
    cam.CreateClippingRangeAttr(Gf.Vec2f(1, 100000))
    eye, target = np.array([0.0, 130.0, 290.0]), heart
    z = (eye - target) / np.linalg.norm(eye - target)
    xa = np.cross([0, 1, 0], z)
    xa /= np.linalg.norm(xa)
    cam.AddTransformOp().Set(Gf.Matrix4d(*xa, 0, *np.cross(z, xa), 0, *z, 0, *eye, 1))

    stage.GetRootLayer().Save()
    return out / "flooded_canoe_cave.usdc"


if __name__ == "__main__":
    import os
    default = Path(os.environ.get("RELIGHT_WORK", Path.home() / "relight-work")) / "usd" / "flooded_canoe_cave"
    print(build(Path(sys.argv[1]) if len(sys.argv) > 1 else default))
