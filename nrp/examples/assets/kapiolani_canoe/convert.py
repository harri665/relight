"""Converts the Smithsonian's 3D scan of Queen Kapi'olani's wa'a (outrigger canoe) to canoe.usdc.

    pip install numpy Pillow usd-core DracoPy
    python examples/assets/kapiolani_canoe/convert.py [download_dir]

Only needed to rebuild the asset; the example scenes read the converted canoe.usdc (see
LICENSE.txt for the source). The scan comes as five Draco-compressed glTF parts ("AR" level of
detail, about 60k triangles in all), each with its own 2048^2 colour, occlusion and normal
textures. This writes them as five meshes under /Canoe, each with a UsdPreviewSurface that has
the colour and normal textures, downscaled to TEX_SIZE (occlusion is dropped).

canoe.usdc: centimetres, Y up (metersPerUnit 0.01), the scan's own origin and axes: z along the
hull (5.8 m), x across it with the outrigger float on the -x side (1.56 m out), y up from the
keel (-19 cm) to the top of the sprit (3.2 m). The hull sits in the water at about y = -7.
"""
import io
import json
import struct
import sys
import urllib.request
from pathlib import Path

import DracoPy
import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

SOURCE = "https://3d-api.si.edu/content/document/3d_package:e6b9bb11-8297-4883-9b6f-101233999337/"
# scan part -> (prim name, what it is)
PARTS = {
    "canoe-body-35k-ar.glb": ("Hull", "the dugout hull"),
    "canoe-moos-15k-ar.glb": ("Gunwales", "the mo'o, strakes along the top of the hull, and the end pieces"),
    "canoe-paletos-15k-ar.glb": ("Outrigger", "the 'iako booms, the ama float, the mast and seats"),
    "canoe-ropes-15k-ar.glb": ("Rigging", "the sprit and boom of the sail, and the lashings"),
    "canoe-sail-20k-ar.glb": ("Sail", "the sail, plaited pandanus"),
}
TEX_SIZE = 1024
MM_TO_CM = 0.1


def read_glb(path):
    b = path.read_bytes()
    n = struct.unpack("<I", b[12:16])[0]
    gltf = json.loads(b[20:20 + n])
    off = 20 + n
    size = struct.unpack("<I", b[off:off + 4])[0]
    blob = b[off + 8:off + 8 + size]

    def view(i):
        v = gltf["bufferViews"][i]
        o = v.get("byteOffset", 0)
        return blob[o:o + v["byteLength"]]
    return gltf, view


def decode(path):
    """Points (cm), triangles, normals, UVs (USD's v-up convention) and the colour / normal
    texture images of a single-primitive Draco GLB."""
    gltf, view = read_glb(path)
    assert len(gltf["meshes"]) == 1 and len(gltf["meshes"][0]["primitives"]) == 1
    assert all(set(n) <= {"mesh", "name"} for n in gltf["nodes"]), "node transforms are not handled"
    prim = gltf["meshes"][0]["primitives"][0]
    ext = prim["extensions"]["KHR_draco_mesh_compression"]
    mesh = DracoPy.decode(view(ext["bufferView"]))
    attr = {k: np.asarray(mesh.get_attribute_by_unique_id(i)["data"], np.float64) for k, i in ext["attributes"].items()}
    P = attr["POSITION"] * MM_TO_CM  # the scan is in millimetres
    N = attr["NORMAL"] / np.linalg.norm(attr["NORMAL"], axis=1, keepdims=True)
    UV = attr["TEXCOORD_0"] * [1, -1] + [0, 1]  # glTF's v runs down
    F = np.asarray(mesh.faces, np.int64).reshape(-1, 3)
    mat = gltf["materials"][prim["material"]]

    def image(tex):
        src = gltf["images"][gltf["textures"][tex["index"]]["source"]]
        return Image.open(io.BytesIO(view(src["bufferView"])))
    color = image(mat["pbrMetallicRoughness"]["baseColorTexture"])
    normal = image(mat["normalTexture"])
    pbr = mat["pbrMetallicRoughness"]
    return P, F, N, UV, color, normal, pbr.get("roughnessFactor", 1.0), pbr.get("metallicFactor", 1.0)


def preview_material(stage, path, color_file, normal_file, rough, metallic):
    m = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    rd = UsdShade.Shader.Define(stage, path + "/st")
    rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    def tex(name, file, raw):
        t = UsdShade.Shader.Define(stage, f"{path}/{name}")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(file)
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw" if raw else "sRGB")
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        if raw:
            t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
            t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        return t.ConnectableAPI()
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex("color", color_file, False), "rgb")
    sh.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(tex("normal", normal_file, True), "rgb")
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return m


def main(download):
    here = Path(__file__).resolve().parent
    download.mkdir(parents=True, exist_ok=True)
    (here / "textures").mkdir(exist_ok=True)
    stage = Usd.Stage.CreateNew(str(here / "canoe.usdc"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    root = UsdGeom.Xform.Define(stage, "/Canoe")
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().documentation = (
        "Queen Kapi'olani's wa'a (outrigger canoe), NMNH, Smithsonian Institution; 3D scan by the "
        "University of South Florida CDHGI; CC0. Converted by convert.py.")
    total = 0
    for file, (name, what) in PARTS.items():
        glb = download / file
        if not glb.exists():
            print("  downloading", file)
            urllib.request.urlretrieve(SOURCE + file, glb)
        P, F, N, UV, color, normal, rough, metallic = decode(glb)
        stem = name.lower()
        color.convert("RGB").resize((TEX_SIZE, TEX_SIZE), Image.LANCZOS).save(
            here / "textures" / f"{stem}_color.jpg", quality=90)
        normal.convert("RGB").resize((TEX_SIZE, TEX_SIZE), Image.LANCZOS).save(
            here / "textures" / f"{stem}_normal.jpg", quality=92)
        mat = preview_material(stage, f"/Canoe/Looks/{name}", f"./textures/{stem}_color.jpg",
                               f"./textures/{stem}_normal.jpg", rough, metallic)
        m = UsdGeom.Mesh.Define(stage, f"/Canoe/{name}")
        m.GetPrim().SetDocumentation(what)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P.astype(np.float32)))
        m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.ravel().astype(np.int32)))
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        m.CreateDoubleSidedAttr(True)  # a scan: some thin parts (the sail) are single sheets
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(N.astype(np.float32)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(UV.astype(np.float32)))
        m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*P.min(0)), Gf.Vec3f(*P.max(0))]))
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
        total += len(F)
        print(f"  {name:10s} {len(F):6d} triangles, bounds {np.round(P.min(0), 1)} .. {np.round(P.max(0), 1)} cm")
    stage.GetRootLayer().Save()
    print(f"  {total} triangles -> {here / 'canoe.usdc'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "relight-work" / "downloads" / "kapiolani_canoe")
