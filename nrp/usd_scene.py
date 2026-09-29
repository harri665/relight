"""USD stages as emitter-free Mitsuba scenes (used through scenes.py).

Converts the renderable geometry of a stage into triangle meshes, its materials into Mitsuba
BSDFs and one UsdGeom.Camera into the sensor. UsdLux lights are not part of the scene (the
proxy's lights are virtual), but sphere, disk, rect and cylinder lights become the viewer's
starting lights.

The stage is normalised: converted to Y-up, then translated and scaled so that the region the
camera sees spans about [-1, 1] on its longest axis. The network's geometric inputs, the light
radius range, the far distance given to escaping paths and the fp16 path storage all assume
roughly that scale, whatever units the stage was authored in. Light positions in the viewer are
in these normalised coordinates; meta.json keeps the transform ("normalize") to map them back:
    p_normalised = (p_world_y_up - center) * scale

Supported:
  - Mesh (polygons are fan-triangulated; holes, leftHanded orientation, GeomSubset materials,
    constant/uniform/vertex/faceVarying normals and UVs; subdivision surfaces render as their cage)
  - Sphere, Cube, Cylinder, Cone, Capsule, Plane (tessellated)
  - native instancing (flattened) and PointInstancer (merged into plain meshes; Mitsuba
    instances only above MERGE_INSTANCES_TRIS, as Mitsuba is slow with many small instances)
  - UsdPreviewSurface with constant or UsdUVTexture inputs (incl. UsdTransform2d, normal maps,
    opacity cutouts, glass via opacity + ior); constant-valued MaterialX standard_surface and
    OmniPBR / OmniGlass; everything else falls back to displayColor
Extensions (see the functions named):
  - customLayerData relight:medium, a homogeneous medium such as water, and prims tagged
    relight:medium for where it is (stage_medium)
  - customLayerData relight:lightBox, the light domain in stage units (stage_light_box)
  - relight:translucency on a Material: a thin sheet that lets light through (_translucency)
Ignored (with a note in the log): curves, points, UsdVol volumes, emission, displacement,
orthographic cameras (an error), dome and distant lights.
"""
import math
import os
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import mitsuba as mi

# Older files bind materials without applying MaterialBindingAPI; resolve those bindings too.
os.environ.setdefault("USD_SHADE_MATERIAL_BINDING_API_CHECK", "allowMissingAPI")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402

RADIUS_RANGE = [0.04, 0.3]  # in normalised units, as for the Cornell box
ROI_RES = 128               # resolution of the pass that finds what the camera sees
MERGE_INSTANCES_TRIS = 20_000_000  # point instancers up to this size become plain meshes

_notes = Counter()


def note(msg, n=1):
    _notes[msg] += n


def _mat(m):
    """Gf.Matrix4d -> numpy 4x4 (USD row-vector convention: p' = p @ M)."""
    return np.array(m, dtype=np.float64).reshape(4, 4)


def _xf_points(M, P):
    return P @ M[:3, :3] + M[3, :3]


def _xf_normals(M, N):
    N = N @ np.linalg.inv(M[:3, :3]).T
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def _up_matrix(stage):
    """Row-vector matrix taking the stage's world space to Y-up."""
    R = np.eye(4)
    if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z:
        R[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]  # (x, y, z) -> (x, z, -y)
    return R


# ---------------------------------------------------------------- stage

def open_stage(path, work_dir):
    path = Path(path).resolve()
    if path.suffix.lower() == ".usdz":
        # Mitsuba cannot read textures inside the package, so unpack it; the first USD file in a
        # .usdz is its root layer.
        dst = work_dir / "usdz"
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            z.extractall(dst)
        path = dst / next(n for n in names if n.lower().endswith((".usd", ".usda", ".usdc")))
    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise RuntimeError(f"could not open {path}")
    return stage


def default_time(stage):
    return Usd.TimeCode(stage.GetStartTimeCode()) if stage.HasAuthoredTimeCodeRange() else Usd.TimeCode.Default()


def traverse(root, time):
    """Visible, renderable prims under `root` (instance proxies included, guides/proxies skipped)."""
    it = iter(Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)))
    for prim in it:
        img = UsdGeom.Imageable(prim)
        if img:
            if img.GetVisibilityAttr().Get(time) == UsdGeom.Tokens.invisible or \
                    img.GetPurposeAttr().Get() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
                it.PruneChildren()
                continue
        yield prim, it


# ---------------------------------------------------------------- geometry
# A "surface" is a triangle list over corners: P [n, 3] points, tri [t, 3] corner indices,
# corner_point [c] point index of each corner, plus optional per-corner normals / uvs and the
# source face of each triangle (for GeomSubsets).

def _fan(counts):
    """Fan-triangulate polygons; returns (corner triangles [t, 3], face index per triangle)."""
    start = np.cumsum(counts) - counts
    ntri = np.maximum(counts - 2, 0)
    face = np.repeat(np.arange(len(counts)), ntri)
    k = np.arange(ntri.sum()) - np.repeat(np.cumsum(ntri) - ntri, ntri) + 1
    c0 = start[face]
    return np.stack([c0, c0 + k, c0 + k + 1], 1), face


def _primvar(prim, names, time):
    api = UsdGeom.PrimvarsAPI(prim)
    for n in names:
        pv = api.GetPrimvar(n)
        if pv and pv.HasValue():
            v = pv.ComputeFlattened(time)
            if v is not None and len(v):
                return np.array(v, dtype=np.float64), pv.GetInterpolation()
    return None


def _uv_primvar(prim, varname, time):
    names = ([varname] if varname else []) + ["st", "st0", "UVMap", "uv", "map1"]
    got = _primvar(prim, names, time)
    if got is None:
        for pv in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
            if pv.GetTypeName() in (Sdf.ValueTypeNames.TexCoord2fArray, Sdf.ValueTypeNames.Float2Array):
                got = _primvar(prim, [pv.GetPrimvarName()], time)
                if got is not None:
                    break
    return got


def _to_corners(vals, interp, corner_point, corner_face, npts, nfaces, what):
    C = len(corner_point)
    need = {"constant": 1, "uniform": nfaces, "vertex": npts, "varying": npts, "faceVarying": C}.get(interp)
    if need is None or len(vals) < need:
        note(f"ignored {what} with {interp} interpolation and the wrong number of values")
        return None
    if interp == "constant":
        return np.repeat(vals[:1], C, 0)
    if interp == "uniform":
        return vals[corner_face]
    if interp == "faceVarying":
        return vals[:C]
    return vals[corner_point]


def _smooth_normals(P, tri_pts):
    fn = np.cross(P[tri_pts[:, 1]] - P[tri_pts[:, 0]], P[tri_pts[:, 2]] - P[tri_pts[:, 0]])
    vn = np.zeros_like(P)
    for k in range(3):
        np.add.at(vn, tri_pts[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def mesh_surface(prim, time, uv_name):
    m = UsdGeom.Mesh(prim)
    pts, counts, idx = (m.GetPointsAttr().Get(time), m.GetFaceVertexCountsAttr().Get(time),
                        m.GetFaceVertexIndicesAttr().Get(time))
    if not pts or not counts or not idx:
        return None
    P = np.array(pts, dtype=np.float64)
    counts = np.array(counts, dtype=np.int64)
    corner_point = np.array(idx, dtype=np.int64)
    if counts.sum() != len(corner_point) or corner_point.max() >= len(P):
        note("skipped meshes with inconsistent topology")
        return None
    tri, face = _fan(counts)
    holes = m.GetHoleIndicesAttr().Get(time)
    if holes:
        keep = ~np.isin(face, np.array(holes))
        tri, face = tri[keep], face[keep]
    if m.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
        tri = tri[:, [0, 2, 1]]
    corner_face = np.repeat(np.arange(len(counts)), counts)
    spread = lambda got, what: None if got is None else \
        _to_corners(got[0], got[1], corner_point, corner_face, len(P), len(counts), what)

    N = spread(_primvar(prim, ["normals"], time), "normals")
    if N is None and m.GetNormalsAttr().HasAuthoredValue():
        N = spread((np.array(m.GetNormalsAttr().Get(time), dtype=np.float64), m.GetNormalsInterpolation()), "normals")
    scheme = m.GetSubdivisionSchemeAttr().Get()
    if scheme != UsdGeom.Tokens.none:
        note(f"rendered {scheme} subdivision surfaces as their control cage")
        if N is None:  # a subdivision surface is smooth; a plain polygon mesh without normals is faceted
            N = _smooth_normals(P, corner_point[tri])[corner_point]
    UV = spread(_uv_primvar(prim, uv_name, time), "texture coordinates")
    if UV is not None:
        UV = UV[:, :2]
    return {"P": P, "tri": tri, "face": face, "corner_point": corner_point, "N": N, "UV": UV}


def _grid_surface(P, N, UV, rows, cols):
    """Surface from a (rows+1) x (cols+1) vertex grid."""
    i, j = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    a = (i * (cols + 1) + j).ravel()
    b, c, d = a + 1, a + cols + 1, a + cols + 2
    tri = np.concatenate([np.stack([a, d, c], 1), np.stack([a, b, d], 1)])  # outward for rows along +z
    return {"P": P, "tri": tri, "corner_point": np.arange(len(P)), "N": N, "UV": UV}


def _lathe(profile, normals, segs=96):
    """Revolve (radius, z) profile points around the z axis."""
    th = np.linspace(0, 2 * np.pi, segs + 1)
    r, z = profile[:, :1], profile[:, 1:2]
    nr, nz = normals[:, :1], normals[:, 1:2]
    c, s = np.cos(th)[None], np.sin(th)[None]
    P = np.stack([r * c, r * s, np.repeat(z, segs + 1, 1)], -1).reshape(-1, 3)
    N = np.stack([nr * c, nr * s, np.repeat(nz, segs + 1, 1)], -1).reshape(-1, 3)
    u, v = np.meshgrid(th / (2 * np.pi), np.linspace(0, 1, len(profile)))
    return _grid_surface(P, N, np.stack([u.ravel(), v.ravel()], 1), len(profile) - 1, segs)


def _disc(r, z, up, segs=96):
    th = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    P = np.concatenate([[[0, 0, z]], np.stack([r * np.cos(th), r * np.sin(th), np.full(segs, z)], 1)])
    k = np.arange(segs)
    tri = np.stack([np.zeros(segs, int), 1 + k, 1 + (k + 1) % segs], 1)
    if not up:
        tri = tri[:, [0, 2, 1]]
    N = np.tile([0, 0, 1 if up else -1], (len(P), 1)).astype(np.float64)
    UV = 0.5 + 0.5 * P[:, :2] / max(r, 1e-12)
    return {"P": P, "tri": tri, "corner_point": np.arange(len(P)), "N": N, "UV": UV}


def _merge(*surfs):
    out, off = {k: [] for k in ("P", "tri", "N", "UV")}, 0
    for s in surfs:
        for k in ("P", "N", "UV"):
            out[k].append(s[k])
        out["tri"].append(s["tri"] + off)
        off += len(s["P"])
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["corner_point"] = np.arange(len(out["P"]))
    return out


def _sphere_profile(r, h, n=48):
    """Profile of a capsule of spine length h (h = 0: a sphere), from the bottom pole up."""
    a = np.linspace(-np.pi / 2, np.pi / 2, n + 1)
    z = r * np.sin(a) + np.where(a < 0, -h / 2, h / 2)
    if h > 0:  # duplicate the equator so the cylinder part gets its own ring
        mid = n // 2
        a = np.insert(a, mid + 1, a[mid])
        z = np.insert(z, mid + 1, h / 2)
        z[mid] = -h / 2
    return np.stack([r * np.cos(a), z], 1), np.stack([np.cos(a), np.sin(a)], 1)


def implicit_surface(prim, time):
    """Tessellates a UsdGeom implicit prim, built along +z and then turned to its spine axis."""
    g = lambda name: prim.GetAttribute(name).Get(time)
    axis = "Z"
    if prim.IsA(UsdGeom.Sphere):
        s = _lathe(*_sphere_profile(g("radius"), 0.0))
    elif prim.IsA(UsdGeom.Capsule):
        s = _lathe(*_sphere_profile(g("radius"), g("height")))
        axis = g("axis")
    elif prim.IsA(UsdGeom.Cylinder):
        r, h = g("radius"), g("height")
        side = _lathe(np.array([[r, -h / 2], [r, h / 2]]), np.array([[1.0, 0], [1.0, 0]]))
        s = _merge(side, _disc(r, -h / 2, False), _disc(r, h / 2, True))
        axis = g("axis")
    elif prim.IsA(UsdGeom.Cone):
        r, h = g("radius"), g("height")
        n = np.array([h, r]) / math.hypot(h, r)
        side = _lathe(np.array([[r, -h / 2], [0, h / 2]]), np.array([n, n]))
        s = _merge(side, _disc(r, -h / 2, False))
        axis = g("axis")
    elif prim.IsA(UsdGeom.Cube):
        e = g("size") / 2
        faces = []
        for k in range(3):
            for sgn in (-1, 1):
                n = np.zeros(3); n[k] = sgn
                u = np.zeros(3); u[(k + 1) % 3] = 1
                v = np.cross(n, u)
                c = np.array([n - u - v, n + u - v, n + u + v, n - u + v]) * e
                faces.append({"P": c, "tri": np.array([[0, 1, 2], [0, 2, 3]]), "N": np.tile(n, (4, 1)),
                              "UV": np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]])})
        s = _merge(*faces)
    elif prim.IsA(UsdGeom.Plane):
        w, l = g("width") / 2, g("length") / 2
        s = {"P": np.array([[-w, -l, 0], [w, -l, 0], [w, l, 0], [-w, l, 0.0]]),
             "tri": np.array([[0, 1, 2], [0, 2, 3]]), "N": np.tile([0, 0, 1.0], (4, 1)),
             "UV": np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]]), "corner_point": np.arange(4)}
        axis = g("axis")
    else:
        return None
    # Cyclic permutations (det +1) that take +z to the requested axis.
    perm = {"X": [2, 0, 1], "Y": [1, 2, 0], "Z": [0, 1, 2]}[axis or "Z"]
    s["P"], s["N"] = s["P"][:, perm], s["N"][:, perm]
    s["corner_point"] = np.arange(len(s["P"]))
    s["face"] = np.arange(len(s["tri"]))
    return s


def finalize(s, M, tris=None):
    """Bakes the row-vector transform M into a surface (optionally a subset of its triangles) and
    merges corners with identical point / normal / uv into Mitsuba vertices."""
    tri = s["tri"] if tris is None else s["tri"][tris]
    used = np.unique(tri)
    cols = [s["corner_point"][used, None].astype(np.float64)]
    cols += [s[k][used] for k in ("N", "UV") if s.get(k) is not None]
    key = np.ascontiguousarray(np.concatenate(cols, 1))
    _, first, inv = np.unique(key.view(np.dtype((np.void, key.dtype.itemsize * key.shape[1]))).ravel(),
                              return_index=True, return_inverse=True)
    corner = used[first]  # one representative corner per vertex
    remap = np.empty(tri.max() + 1, np.int64)
    remap[used] = inv.ravel()
    F = remap[tri]
    if np.linalg.det(M[:3, :3]) < 0:
        F = F[:, [0, 2, 1]]
    P = _xf_points(M, s["P"][s["corner_point"][corner]])
    N = _xf_normals(M, s["N"][corner]) if s.get("N") is not None else None
    UV = s["UV"][corner] if s.get("UV") is not None else None
    return {"P": P, "F": F, "N": N, "UV": UV}


# ---------------------------------------------------------------- materials

def _value(shader, name):
    """(kind, payload) for a shader input: ("out", (shader, output name)) when it is driven by
    another shader, ("val", value) when it has a value, None when it is unset."""
    inp = shader.GetInput(name)
    if not inp:
        return None
    attrs = inp.GetValueProducingAttributes()
    if not attrs:
        return None
    a = attrs[0]
    if UsdShade.Output.IsOutput(a):
        return "out", (UsdShade.Shader(a.GetPrim()), UsdShade.Output(a).GetBaseName())
    v = a.Get()
    return None if v is None else ("val", v)


def _const(shader, name, default):
    v = _value(shader, name)
    return v[1] if v and v[0] == "val" else default


def _shader_id(shader):
    return shader.GetShaderId() or ""


def _lum(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


class Materials:
    def __init__(self, stage, time, max_texture):
        self.stage, self.time, self.max_texture = stage, time, max_texture
        self.bsdfs, self.images, self.textures = {}, {}, {}
        self.stage_dir = Path(stage.GetRootLayer().realPath).parent

    # --- textures
    def _read(self, path):
        if path not in self.images:
            try:
                bm = mi.Bitmap(path)
                a = np.array(bm)
            except Exception:
                from PIL import Image
                a = np.array(Image.open(path))
            if a.ndim == 2:
                a = a[..., None]
            ldr = a.dtype.kind in "ui"
            a = a.astype(np.float32) / (np.iinfo(a.dtype).max if ldr else 1)
            while max(a.shape[:2]) > self.max_texture:  # box-filter down to the size limit
                h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
                a = a[:h, :w]
                a = 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])
            self.images[path] = (a, ldr)
        return self.images[path]

    def _file(self, shader):
        v = _value(shader, "file")
        if not v or v[0] != "val":
            return None
        asset = v[1]
        path = asset.resolvedPath or (str(self.stage_dir / asset.path) if asset.path else "")
        if "<UDIM>" in (asset.path or ""):
            path = str(self.stage_dir / asset.path.replace("<UDIM>", "1001"))
            note("used only tile 1001 of UDIM textures")
        if not path or not Path(path).exists():
            note(f"missing texture files (e.g. {asset.path})")
            return None
        return path

    def _st(self, shader):
        """(uv primvar name, 3x3 uv transform) feeding a UsdUVTexture's st input."""
        name, T = None, np.eye(3)
        v = _value(shader, "st")
        while v and v[0] == "out":
            src = v[1][0]
            sid = _shader_id(src)
            if sid == "UsdTransform2d":
                sx, sy = _const(src, "scale", (1, 1))
                a = math.radians(_const(src, "rotation", 0.0))
                tx, ty = _const(src, "translation", (0, 0))
                S = np.diag([sx, sy, 1.0])
                R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
                Tr = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1.0]])
                T = T @ Tr @ R @ S
                v = _value(src, "in")
            elif sid.startswith("UsdPrimvarReader"):
                name = str(_const(src, "varname", "") or "") or None
                break
            else:
                break
        return name, T

    def texture(self, shader, channels, threshold=None, normal=False):
        """(Mitsuba bitmap texture, uv primvar name) for a UsdUVTexture output ("rgb", "r", "g",
        "b" or "a")."""
        path = self._file(shader)
        if path is None:
            return None
        space = str(_const(shader, "sourceColorSpace", "auto"))
        scale = tuple(float(x) for x in _const(shader, "scale", (1, 1, 1, 1)))
        bias = tuple(float(x) for x in _const(shader, "bias", (0, 0, 0, 0)))
        wrap = str(_const(shader, "wrapS", "repeat"))
        uv_name, T = self._st(shader)
        tex = {"type": "bitmap", "bitmap": self.bitmap(path, channels, space, scale, bias, threshold, normal),
               "raw": True, "filter_type": "bilinear",
               "wrap_mode": {"mirror": "mirror", "clamp": "clamp", "black": "clamp"}.get(wrap, "repeat")}
        if not np.allclose(T, np.eye(3)):
            tex["to_uv"] = mi.ScalarTransform3f(T.tolist())
        return tex, uv_name

    def bitmap(self, path, channels="rgb", space="auto", scale=(1, 1, 1, 1), bias=(0, 0, 0, 0),
               threshold=None, normal=False):
        """Linear float mi.Bitmap of the requested channels, with scale/bias applied."""
        key = (path, channels, space, scale, bias, threshold, normal)
        if key not in self.textures:
            a, ldr = self._read(path)
            nc = a.shape[2]
            a = a.copy()
            # Like Hydra: "auto" means sRGB for 8/16-bit colour images. Alpha is always linear.
            if not normal and (space == "sRGB" or (space == "auto" and ldr and nc >= 3)):
                a[..., :min(nc, 3)] = _srgb_to_linear(a[..., :min(nc, 3)])
            rgb = a[..., :3] if nc >= 3 else np.repeat(a[..., :1], 3, -1)
            scale, bias = np.array(scale, np.float32), np.array(bias, np.float32)
            if channels == "rgb":
                img = rgb * scale[:3] + bias[:3]
            elif channels == "a":
                img = (a[..., -1:] if nc in (2, 4) else np.ones_like(a[..., :1])) * scale[3] + bias[3]
            else:
                k = "rgb".index(channels)
                img = rgb[..., k:k + 1] * scale[k] + bias[k]
            if threshold is not None:
                img = (img >= threshold).astype(np.float32)
            if normal:  # Mitsuba's normalmap wants [0, 1] and maps it to [-1, 1] itself
                img = img * 0.5 + 0.5
            # USD's st origin is the image's bottom-left corner, Mitsuba's uv origin its first row.
            self.textures[key] = mi.Bitmap(np.ascontiguousarray(np.flipud(img), dtype=np.float32))
        return self.textures[key]

    # --- inputs
    def input(self, shader, name, default, color=False, **kw):
        """Mitsuba value for a shader input: a float / {"type": "rgb"} / bitmap dict, plus the uv name."""
        v = _value(shader, name)
        if v and v[0] == "out":
            src, out = v[1]
            if _shader_id(src) == "UsdUVTexture":
                got = self.texture(src, "rgb" if color else (out if out in ("r", "g", "b", "a") else "r"), **kw)
                if got:
                    return got
            else:
                note(f"used defaults for inputs driven by {_shader_id(src) or 'unknown'} shaders")
            v = None
        val = v[1] if v else default
        if color:
            return {"type": "rgb", "value": [float(x) for x in val]}, None
        return float(val), None

    # --- materials
    def bsdf(self, material, prim, has_uv):
        """(Mitsuba BSDF object, uv primvar name) for a bound material (or None)."""
        key = (str(material.GetPath()) if material else "", has_uv, "" if material else str(prim.GetPath()))
        if key not in self.bsdfs:
            self.bsdfs[key] = self._make(material, prim, has_uv)
        return self.bsdfs[key]

    def _make(self, material, prim, has_uv):
        if material:
            for ctx in ("", "mtlx", "mdl"):
                shader, _, _ = material.ComputeSurfaceSource(ctx)
                if not shader:
                    continue
                sid = _shader_id(shader)
                if sid in ("UsdPreviewSurface", "ND_UsdPreviewSurface_surfaceshader"):
                    d, uv = self._preview_surface(shader, has_uv, _translucency(material))
                    return mi.load_dict(d), uv
                if sid.startswith("ND_standard_surface") or sid.startswith("ND_open_pbr"):
                    return mi.load_dict(self._standard_surface(shader)), None
                asset = shader.GetSourceAsset("mdl")
                mdl = Path(asset.path).stem if asset else ""
                if mdl:
                    return mi.load_dict(self._omni(shader, mdl)), None
                note(f"approximated {sid or 'unknown'} materials by displayColor")
                break
        # Fallback: displayColor, like Hydra when there is no usable material.
        c = (0.5, 0.5, 0.5)
        got = _primvar(prim, ["displayColor"], self.time)
        if got is not None:
            c = tuple(np.clip(got[0].reshape(-1, 3).mean(0), 0, 1))
        return mi.load_dict({"type": "twosided", "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": list(c)}}}), None

    def _preview_surface(self, sh, has_uv, translucency=0.0):
        uvs = []
        def inp(name, default, **kw):
            v, uv = self.input(sh, name, default, **kw)
            uvs.append(uv)
            return v
        base = inp("diffuseColor", (0.18, 0.18, 0.18), color=True)
        rough = inp("roughness", 0.5)
        metallic = inp("metallic", 0.0)
        ior = float(_const(sh, "ior", 1.5))
        thr = float(_const(sh, "opacityThreshold", 0.0))
        opacity = inp("opacity", 1.0, threshold=thr if thr > 0 else None)
        cc, ccr = float(_const(sh, "clearcoat", 0.0)), float(_const(sh, "clearcoatRoughness", 0.01))
        emissive = _value(sh, "emissiveColor")
        if emissive and (emissive[0] == "out" or max(emissive[1]) > 0):
            note("ignored emissive materials (the proxy's only lights are the virtual sphere lights)")

        b = {"type": "principled", "base_color": base, "roughness": rough, "metallic": metallic}
        if int(_const(sh, "useSpecularWorkflow", 0)):
            spec = _const(sh, "specularColor", (0, 0, 0))
            b["metallic"] = 0.0
            b["specular"] = float(np.clip(_lum(spec) / 0.08, 0, 1))
        else:
            b["eta"] = max(ior, 1.01)
        if cc > 0:
            b["clearcoat"], b["clearcoat_gloss"] = cc, 1 - ccr

        if translucency > 0:  # thin sheet that lets light through (sails, paper, leaves)
            b = {"type": "principledthin", "base_color": base, "roughness": rough,
                 "diff_trans": float(np.clip(2 * translucency, 0, 2))}

        cutout = isinstance(opacity, dict) or thr > 0
        if thr > 0 and not isinstance(opacity, dict):
            opacity = 1.0 if opacity >= thr else 0.0
        if not cutout and opacity < 0.999:  # transparent: treat as glass
            if isinstance(rough, float) and rough < 0.05 and opacity < 0.1 and metallic == 0:
                tint = base if not isinstance(base, dict) or base.get("type") == "rgb" else 1.0
                b = {"type": "dielectric", "int_ior": max(ior, 1.01), "specular_transmittance": tint}
            else:
                b["spec_trans"] = 1 - opacity
        elif b["type"] == "principled":
            normal = _value(sh, "normal")
            if has_uv and normal and normal[0] == "out" and _shader_id(normal[1][0]) == "UsdUVTexture":
                got = self.texture(normal[1][0], "rgb", normal=True)
                if got:
                    b = {"type": "normalmap", "normalmap": got[0], "bsdf": b}
                    uvs.append(got[1])
            b = {"type": "twosided", "bsdf": b}
        if cutout:
            b = {"type": "mask", "opacity": opacity, "bsdf": b}
        return b, next((u for u in uvs if u), None)

    def _standard_surface(self, sh):
        c = lambda n, d: _const(sh, n, d)
        if sh.GetShaderId().startswith("ND_open_pbr"):
            base = np.array(c("base_color", (0.8, 0.8, 0.8))) * c("base_weight", 1.0)
            rough, metal, trans, ior = c("specular_roughness", 0.3), c("base_metalness", 0.0), \
                c("transmission_weight", 0.0), c("specular_ior", 1.5)
        else:
            base = np.array(c("base_color", (0.8, 0.8, 0.8))) * c("base", 1.0)
            rough, metal, trans, ior = c("specular_roughness", 0.2), c("metalness", 0.0), \
                c("transmission", 0.0), c("specular_IOR", 1.5)
        note("used constant values of MaterialX materials (textures in their node graphs are not read)")
        if trans > 0.5:
            return {"type": "dielectric", "int_ior": max(ior, 1.01)}
        return {"type": "twosided", "bsdf": {"type": "principled", "base_color": {"type": "rgb", "value": base.tolist()},
                                             "roughness": float(rough), "metallic": float(metal), "eta": max(ior, 1.01)}}

    def _omni(self, sh, mdl):
        if "glass" in mdl.lower():
            return {"type": "dielectric", "int_ior": float(_const(sh, "glass_ior", 1.5))}
        base = {"type": "rgb", "value": list(_const(sh, "diffuse_color_constant", (0.2, 0.2, 0.2)))}
        tex = _value(sh, "diffuse_texture")
        if tex and tex[0] == "val" and tex[1].resolvedPath:
            base = {"type": "bitmap", "bitmap": self.bitmap(tex[1].resolvedPath), "raw": True}
        if mdl != "OmniPBR":
            note(f"approximated MDL materials other than OmniPBR/OmniGlass as OmniPBR")
        return {"type": "twosided", "bsdf": {"type": "principled", "base_color": base,
                                             "roughness": float(_const(sh, "reflection_roughness_constant", 0.5)),
                                             "metallic": float(_const(sh, "metallic_constant", 0.0))}}


def _prim_medium(prim):
    """The prim's `relight:medium` tag ("water", "air" or "boundary"), inherited from ancestors."""
    while prim:
        a = prim.GetAttribute("relight:medium")
        if a and a.HasAuthoredValue():
            return str(a.Get())
        prim = prim.GetParent()
    return None


def _translucency(material):
    """A material's `relight:translucency` (0..1, the share of diffuse light passed through the
    surface, as for a thin sail or paper), an extension UsdPreviewSurface has no input for."""
    a = material.GetPrim().GetAttribute("relight:translucency")
    return float(a.Get()) if a and a.HasAuthoredValue() else 0.0


def _binding(prim):
    return UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]


def _material_uv_name(material):
    if not material:
        return None
    for p in Usd.PrimRange(material.GetPrim()):
        sh = UsdShade.Shader(p)
        if sh and _shader_id(sh).startswith("UsdPrimvarReader"):
            name = _const(sh, "varname", None)
            if name:
                return str(name)
    return None


# ---------------------------------------------------------------- stage -> parts

class Converter:
    def __init__(self, stage, time, max_texture):
        self.stage, self.time = stage, time
        self.xc = UsdGeom.XformCache(time)
        self.mats = Materials(stage, time, max_texture)
        self.R = _up_matrix(stage)
        self.tris = 0

    def world(self, prim):
        return _mat(self.xc.GetLocalToWorldTransform(prim))

    def gprim_parts(self, prim, M):
        """Meshes (world space via row matrix M) with their BSDFs for one gprim."""
        material = _binding(prim)
        if prim.IsA(UsdGeom.Mesh):
            s = mesh_surface(prim, self.time, _material_uv_name(material))
        elif prim.IsA(UsdGeom.Sphere) or prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Cylinder) or \
                prim.IsA(UsdGeom.Cone) or prim.IsA(UsdGeom.Capsule) or prim.IsA(UsdGeom.Plane):
            s = implicit_surface(prim, self.time)
        else:
            note(f"skipped {prim.GetTypeName()} prims (not surfaces)")
            return []
        if s is None or not len(s["tri"]):
            return []
        groups = []  # (triangle mask or None, material)
        subsets = UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets() if prim.IsA(UsdGeom.Mesh) else []
        rest = np.ones(len(s["tri"]), bool)
        for sub in subsets:
            faces = sub.GetIndicesAttr().Get(self.time)
            if not faces:
                continue
            sel = np.isin(s["face"], np.array(faces)) & rest
            if sel.any():
                groups.append((sel, _binding(sub.GetPrim()) or material))
                rest &= ~sel
        if rest.any():
            groups.append((None if rest.all() else rest, material))
        parts = []
        for sel, mat in groups:
            bsdf, uv_name = self.mats.bsdf(mat, prim, s.get("UV") is not None)
            g = finalize(s, M, sel)
            self.tris += len(g["F"])
            parts.append({"name": str(prim.GetPath()), "bsdf": bsdf, "medium": _prim_medium(prim), **g})
        return parts

    def collect(self):
        """All surfaces (world space, Y-up) and point instancers of the stage."""
        parts, instancers = [], []
        for prim, it in traverse(self.stage.GetPseudoRoot(), self.time):
            if prim.IsA(UsdGeom.PointInstancer):
                it.PruneChildren()
                inst = self.point_instancer(prim)
                if inst:
                    instancers.append(inst)
            elif prim.IsA(UsdGeom.Gprim):
                parts += self.gprim_parts(prim, self.world(prim) @ self.R)
        return parts, instancers

    def point_instancer(self, prim):
        pi = UsdGeom.PointInstancer(prim)
        protos = pi.GetPrototypesRel().GetTargets()
        xforms = pi.ComputeInstanceTransformsAtTime(self.time, self.time)
        ids = pi.GetProtoIndicesAttr().Get(self.time)
        if not protos or xforms is None or ids is None:
            return None
        W = self.world(prim) @ self.R
        out = []
        for k, path in enumerate(protos):
            proto = self.stage.GetPrimAtPath(path)
            # Gprims relative to the prototype root's parent, so the root's own transform is kept.
            base = np.linalg.inv(self.world(proto.GetParent()))
            geo = [p for g, _ in traverse(proto, self.time) if g.IsA(UsdGeom.Gprim)
                   for p in self.gprim_parts(g, self.world(g) @ base)]
            mats = [_mat(xforms[i]) @ W for i in range(len(ids)) if ids[i] == k]
            if geo and mats:
                self.tris += sum(len(g["F"]) for g in geo) * (len(mats) - 1)
                out.append({"name": str(path), "parts": geo, "instances": mats})
        return out

    # --- camera and lights
    def camera(self, path):
        cams = [p for p, _ in traverse(self.stage.GetPseudoRoot(), self.time) if p.IsA(UsdGeom.Camera)]
        if path:
            prim = self.stage.GetPrimAtPath(path)
            if not prim or not prim.IsA(UsdGeom.Camera):
                raise ValueError(f"{path} is not a camera; cameras: {[str(c.GetPath()) for c in cams]}")
        elif cams:
            prim = sorted(cams, key=lambda p: str(p.GetPath()))[0]
            if len(cams) > 1:
                print(f"  {len(cams)} cameras, using {prim.GetPath()} (pick another with --camera): "
                      + ", ".join(str(c.GetPath()) for c in cams))
        else:
            return None
        gc = UsdGeom.Camera(prim).GetCamera(self.time)
        if gc.projection != Gf.Camera.Perspective:
            raise ValueError(f"{prim.GetPath()} is orthographic; the proxy needs a perspective camera")
        M = _mat(gc.transform) @ self.R
        # The image is square, so crop to the smaller field of view.
        fov = min(gc.GetFieldOfView(Gf.Camera.FOVHorizontal), gc.GetFieldOfView(Gf.Camera.FOVVertical))
        return {"path": str(prim.GetPath()), "M": M, "fov": fov, "near": gc.clippingRange.min}

    def lights(self):
        out = []
        for prim, _ in traverse(self.stage.GetPseudoRoot(), self.time):
            if not prim.HasAPI(UsdLux.LightAPI):
                continue
            M = self.world(prim) @ self.R
            s = abs(np.linalg.det(M[:3, :3])) ** (1 / 3)
            g = lambda n, d: (prim.GetAttribute(n).Get(self.time) if prim.GetAttribute(n).HasValue() else d)
            if prim.IsA(UsdLux.SphereLight):
                r = 0.0 if g("treatAsPoint", False) else g("inputs:radius", 0.5)
            elif prim.IsA(UsdLux.DiskLight):
                r = g("inputs:radius", 0.5) / 2  # same area
            elif prim.IsA(UsdLux.RectLight):
                r = math.sqrt(g("inputs:width", 1.0) * g("inputs:height", 1.0) / (4 * math.pi))
            elif prim.IsA(UsdLux.CylinderLight):
                r = math.sqrt(g("inputs:radius", 0.5) * g("inputs:length", 1.0) / 2)
            else:
                note(f"skipped {prim.GetTypeName()} lights (only sphere lights can be represented)")
                continue
            if not prim.IsA(UsdLux.SphereLight):
                note(f"approximated {prim.GetTypeName()} lights as sphere lights of the same area")
            lin = g("inputs:intensity", 1.0) * 2 ** g("inputs:exposure", 0.0)
            color = list(g("inputs:color", (1, 1, 1)))
            m = max(color) or 1.0
            out.append({"name": prim.GetName(), "pos": M[3, :3], "radius": r * s,
                        "color": [c / m for c in color], "intensity": lin * m})
        return out


# ---------------------------------------------------------------- Mitsuba scene

def _mi_mesh(name, part, N, medium=None, default="water"):
    """Mitsuba mesh from a part, with the normalisation (scale s, center c) baked in. `medium` (a
    Mitsuba medium) goes where the part's tag (or `default`) says: "water" on both sides, so a
    surface in the water keeps it on any bounce; "air" on neither; "boundary" inside only (the
    closed surface of a body of water); "submerged" outside only (glass in the water)."""
    c, s = N
    props = mi.Properties()
    if part.get("bsdf") is not None:
        props["bsdf"] = part["bsdf"]
    mode = part.get("medium") or default
    if medium is not None:
        if mode in ("water", "boundary"):
            props["interior"] = medium
        if mode in ("water", "submerged"):
            props["exterior"] = medium
    m = mi.Mesh(name, len(part["P"]), len(part["F"]), has_vertex_normals=part["N"] is not None,
                has_vertex_texcoords=part["UV"] is not None, props=props)
    p = mi.traverse(m)
    p["vertex_positions"] = mi.Float(((part["P"] - c) * s).astype(np.float32).ravel())
    p["faces"] = mi.UInt32(part["F"].astype(np.uint32).ravel())
    if part["N"] is not None:
        p["vertex_normals"] = mi.Float(part["N"].astype(np.float32).ravel())
    if part["UV"] is not None:
        p["vertex_texcoords"] = mi.Float(part["UV"].astype(np.float32).ravel())
    p.update()
    return m


def _sensor(cam, res, norm):
    c, s = norm
    M = cam["M"]
    o = (M[3, :3] - c) * s
    fwd = -M[2, :3] / np.linalg.norm(M[2, :3])
    up = M[1, :3] / np.linalg.norm(M[1, :3])
    return {
        "type": "perspective", "fov": float(cam["fov"]), "fov_axis": "x",
        "near_clip": float(max(cam["near"] * s, 1e-3)), "far_clip": 1e5,
        "to_world": mi.ScalarTransform4f().look_at(origin=o.tolist(), target=(o + fwd).tolist(), up=up.tolist()),
        "film": {"type": "hdrfilm", "width": res, "height": res, "rfilter": {"type": "box"}, "pixel_format": "rgb"},
        "sampler": {"type": "independent"},
    }


def _merge_instances(part, mats):
    """One part copied to every instance transform (row-vector matrices), as a single mesh."""
    n = len(part["P"])
    P = np.concatenate([_xf_points(M, part["P"]) for M in mats])
    F = np.concatenate([(part["F"][:, [0, 2, 1]] if np.linalg.det(M[:3, :3]) < 0 else part["F"]) + i * n
                        for i, M in enumerate(mats)])
    N = np.concatenate([_xf_normals(M, part["N"]) for M in mats]) if part["N"] is not None else None
    UV = np.tile(part["UV"], (len(mats), 1)) if part["UV"] is not None else None
    return {**part, "P": P, "F": F, "N": N, "UV": UV}


def mi_medium(medium):
    """Mitsuba homogeneous medium for a config["medium"] (normalised units)."""
    top = max(max(medium["sigma_t"]), 1e-12)  # Mitsuba's RGB values must be <= 1; the rest goes in scale
    return mi.load_dict({"type": "homogeneous", "scale": top,
                         "sigma_t": {"type": "rgb", "value": [x / top for x in medium["sigma_t"]]},
                         "albedo": {"type": "rgb", "value": medium["albedo"]},
                         "phase": {"type": "hg", "g": medium["g"]}})


def scene_dict(parts, instancers, cam, res, norm, medium=None):
    """`medium` (config["medium"]) fills the scene for Mitsuba's own volumetric integrators
    (previews, validate_gather); sample_paths.py traces the medium itself."""
    d = {"type": "scene", "sensor": _sensor(cam, res, norm)}
    med = mi_medium(medium) if medium else None
    default = medium["default"] if medium else "air"
    if med is not None and medium["camera"] == "water":
        d["sensor"]["medium"] = med
    for i, part in enumerate(parts):
        d[f"shape_{i}"] = _mi_mesh(part["name"], part, norm, med, default)
    c, s = norm
    Nm = np.eye(4)
    Nm[:3, :3] *= s
    Nm[3, :3] = -c * s
    for k, inst in enumerate(instancers):
        n_tris = sum(len(p["F"]) for p in inst["parts"]) * len(inst["instances"])
        if n_tris <= MERGE_INSTANCES_TRIS:
            # Mitsuba dispatches per instance, which gets very slow with thousands of small
            # instances (coins, pebbles), so modest instancers become one plain mesh per part.
            for i, part in enumerate(inst["parts"]):
                d[f"instanced_{k}_{i}"] = _mi_mesh(part["name"], _merge_instances(part, inst["instances"]), norm,
                                                   med, default)
            continue
        group = {"type": "shapegroup"}
        for i, part in enumerate(inst["parts"]):
            group[f"shape_{i}"] = _mi_mesh(part["name"], part, (np.zeros(3), 1.0), med, default)
        group = d[f"shapegroup_{k}"] = mi.load_dict(group)  # the scene must hold the group itself too
        for j, M in enumerate(inst["instances"]):
            # Row-vector matrices -> Mitsuba's column-vector to_world.
            d[f"instance_{k}_{j}"] = {"type": "instance", "shapegroup": group,
                                      "to_world": mi.ScalarTransform4f((M @ Nm).T.tolist())}
    return d


def _auto_camera(parts):
    """Three-quarter view that frames all geometry, for stages without a camera."""
    P = np.concatenate([p["P"] for p in parts])
    lo, hi = P.min(0), P.max(0)
    c, R = (lo + hi) / 2, np.linalg.norm(hi - lo) / 2
    fov = 40.0
    d = np.array([0.45, 0.35, 1.0])
    d /= np.linalg.norm(d)
    o = c + d * R / math.sin(math.radians(fov / 2)) * 1.05
    z = (o - c) / np.linalg.norm(o - c)  # USD cameras look down -z
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3], M[3, :3] = x, np.cross(z, x), z, o
    print("  no camera in the stage: using an automatic three-quarter view")
    return {"path": None, "M": M, "fov": fov, "near": R * 1e-3}


def stage_medium(stage, scale):
    """A homogeneous medium (water, haze), declared on the stage as
        customLayerData = {"relight": {"medium": {"sigma_a": [r, g, b], "sigma_s": [r, g, b], "g": 0.8,
                                                  "default": "water", "camera": "water"}}}
    with absorption / scattering coefficients per metre and the Henyey-Greenstein anisotropy g.
    By default it fills the scene. For a scene that is partly under water, tag prims with a string
    attribute relight:medium ("water", "air", "boundary" for the closed surface of the water body,
    whose inside is water, or "submerged" for glass in the water, water outside and none inside);
    untagged prims get "default", and "camera" says where the camera is.
    Returned in normalised units: {"sigma_t", "albedo", "g", "default", "camera"}."""
    d = dict(stage.GetRootLayer().customLayerData.get("relight", {})).get("medium")
    if not d:
        return None
    k = UsdGeom.GetStageMetersPerUnit(stage) / scale  # metres per normalised unit
    sa = np.array(d.get("sigma_a", (0, 0, 0)), float) * k
    ss = np.array(d.get("sigma_s", (0, 0, 0)), float) * k
    st = sa + ss
    default = str(d.get("default", "water"))
    return {"sigma_t": st.tolist(), "albedo": (ss / np.maximum(st, 1e-12)).tolist(), "g": float(d.get("g", 0.0)),
            "default": default, "camera": str(d.get("camera", default))}


def stage_light_box(stage, R, center, scale):
    """customLayerData relight:lightBox [[x0, y0, z0], [x1, y1, z1]] (stage units and axes), in
    normalised coordinates; None if the stage doesn't declare one."""
    box = dict(stage.GetRootLayer().customLayerData.get("relight", {})).get("lightBox")
    if box is None:
        return None
    lo, hi = np.array(box[0], float), np.array(box[1], float)
    corners = np.array([[(lo, hi)[i][0], (lo, hi)[j][1], (lo, hi)[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    p = (_xf_points(R, corners) - center) * scale
    return [np.round(p.min(0), 3).tolist(), np.round(p.max(0), 3).tolist()]


def region_of_interest(parts, instancers, cam):
    """Bounds of what the camera sees (2nd-98th percentile of visible points) and the fraction
    of camera rays that escape, from a quick position render."""
    scene = mi.load_dict(scene_dict(parts, instancers, cam, ROI_RES, (np.zeros(3), 1.0)))
    integ = mi.load_dict({"type": "aov", "aovs": "pp:position,dd:depth"})
    img = np.array(mi.render(scene, integrator=integ, spp=1))[..., -4:].reshape(-1, 4)
    hit = img[:, 3] > 0
    if hit.mean() < 0.01:
        raise ValueError("the camera sees (almost) no geometry; pick another camera with --camera")
    pts = img[hit, :3].astype(np.float64)
    return np.percentile(pts, 2, 0), np.percentile(pts, 98, 0), 1 - hit.mean()


def light_domain(lo, hi, escaped, margin=None):
    """Default light box in normalised coordinates: the visible region, pulled in slightly for
    enclosed scenes (lights behind walls light nothing) and grown for open ones (so lights can go
    above and around the objects). Thin axes are widened, upwards for y."""
    if margin is None:
        margin = 0.3 if escaped > 0.05 else -0.03
    lo, hi = lo - margin, hi + margin
    for k in range(3):
        ext = hi[k] - lo[k]
        if ext < 1.0:
            if k == 1:
                hi[k] = lo[k] + 1.0
            else:
                lo[k] -= (1 - ext) / 2
                hi[k] += (1 - ext) / 2
    return [np.round(lo, 3).tolist(), np.round(hi, 3).tolist()]


def build(path, res, work_dir, camera=None, time=None, max_texture=2048, normalize=None,
          light_bbox=None, radius_range=None, light_margin=None):
    """Returns (Mitsuba scene dict, config) for a USD file. `normalize` (from an earlier meta.json)
    skips the region-of-interest pass so a reload uses exactly the same coordinates."""
    _notes.clear()
    stage = open_stage(path, work_dir)
    t = Usd.TimeCode(time) if time is not None else default_time(stage)
    conv = Converter(stage, t, max_texture)
    parts, instancers = conv.collect()
    instancers = [i for group in instancers for i in group]
    if not parts and not instancers:
        raise ValueError(f"{path}: no renderable surfaces")
    cam = conv.camera(camera) or _auto_camera(parts or [p for i in instancers for p in i["parts"]])

    if normalize:
        center, scale = np.array(normalize["center"]), normalize["scale"]
    else:
        lo, hi, escaped = region_of_interest(parts, instancers, cam)
        center, scale = (lo + hi) / 2, 1.0 / max((hi - lo).max() / 2, 1e-9)
        if light_bbox is None:
            light_bbox = stage_light_box(stage, conv.R, center, scale) or                 light_domain((lo - center) * scale, (hi - center) * scale, escaped, light_margin)
        print(f"  visible region {np.round(lo, 3).tolist()} .. {np.round(hi, 3).tolist()} (stage units, Y-up), "
              f"{100 * escaped:.0f}% of camera rays escape")
    norm = (center, scale)
    medium = stage_medium(stage, scale)
    d = scene_dict(parts, instancers, cam, res, norm, medium)

    lights = []
    for l in conv.lights():
        lights.append({"name": l["name"], "pos": ((l["pos"] - center) * scale).round(4).tolist(),
                       "radius": round(float(l["radius"] * scale), 4), "color": l["color"],
                       "intensity": float(l["intensity"]), "auto_intensity": True})
    n_inst = sum(len(i["instances"]) for i in instancers)
    print(f"  {len(parts)} meshes" + (f", {n_inst} instances" if n_inst else "")
          + f", {conv.tris / 1e6:.2f} M triangles, {len(conv.mats.bsdfs)} materials, "
          f"{len(conv.mats.textures)} textures, {len(lights)} lights -> viewer defaults")
    for msg, n in _notes.items():
        print(f"  note: {msg}" + (f" ({n}x)" if n > 1 else ""))

    cfg = {
        "source": {"usd": str(Path(path).resolve()), "camera": cam["path"], "time": time, "max_texture": max_texture},
        "normalize": {"up": "Y", "center": np.asarray(center).tolist(), "scale": float(scale),
                      "stage_up": UsdGeom.GetStageUpAxis(stage)},
        "light_bbox": light_bbox,
        "radius_range": radius_range or RADIUS_RANGE,
    }
    if lights:
        cfg["default_lights"] = lights
    if medium:
        cfg["medium"] = medium
        print(f"  medium: extinction {np.round(medium['sigma_t'], 3).tolist()} per normalised unit, "
              f"albedo {np.round(medium['albedo'], 3).tolist()}, g {medium['g']}, "
              f"untagged prims in {medium['default']}, camera in {medium['camera']}")
    return d, cfg
