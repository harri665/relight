"""Scene definitions for the light-agnostic path sampling pass.

Scenes contain no emitters: all illumination comes from virtual lights that are
added after the fact by GATHERLIGHT / the neural render proxy.

Each scene also defines the domain from which training lights are drawn
(`light_bbox`, `radius_range`); the web viewer clamps lights to the same domain,
because the network is only trained inside it. Optional `test_lights` ([x, y, z, r]) get
reference renders in the viewer's Accuracy tab, and `default_lights` are the viewer's
starting lights.

Besides the built-in scenes, any USD file works (see usd_scene.py):
    sample_paths.py --scene path/to/stage.usd [--name id]
imports it and records its source in meta.json, so later steps refer to it by id.
"""
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import mitsuba as mi

from common import WORK_DIR, scene_dir

USD_SUFFIXES = (".usd", ".usda", ".usdc", ".usdz")


def _bitterli_xml(name):
    """Mitsuba 3 XML of a scene from benedikt-bitterli.me/resources, downloaded on first use,
    with every emitter removed. Shapes that carried an area emitter stay (as plain geometry):
    without them, pixels where some samples escape (e.g. between blind slats in front of the
    window light) get aux positions averaged with zeros, which breaks the geometric features."""
    root = WORK_DIR / "assets" / name
    src = root / name / "scene_v3.xml"
    if not src.exists():
        url = f"https://benedikt-bitterli.me/resources/mitsuba/{name}.zip"
        print(f"downloading {url}", flush=True)
        with urllib.request.urlopen(url) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(root)
    out = src.with_name("scene_noemit.xml")
    xml = re.sub(r"\s*<emitter [^>]*/>|\s*<emitter .*?</emitter>", "", src.read_text(), flags=re.S)
    if not out.exists() or out.read_text() != xml:
        out.write_text(xml)
    return out


def _bathroom(res):
    return _bitterli_xml("bathroom")


def _cornell(res):
    T = mi.ScalarTransform4f
    white = {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.8, 0.8, 0.8]}}
    return {
        "type": "scene",
        "sensor": {
            "type": "perspective",
            "fov": 39.3077,
            "fov_axis": "smaller",
            "to_world": T().look_at(origin=[0, 0, 3.9], target=[0, 0, 0], up=[0, 1, 0]),
            "film": {"type": "hdrfilm", "width": res, "height": res,
                     "rfilter": {"type": "box"}, "pixel_format": "rgb"},
            "sampler": {"type": "independent"},
        },
        "white": white,
        "red": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.63, 0.065, 0.05]}},
        "green": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.14, 0.45, 0.091]}},
        "glossy-floor": {"type": "roughplastic", "distribution": "ggx", "alpha": 0.12,
                         "diffuse_reflectance": {"type": "rgb", "value": [0.75, 0.73, 0.70]}},
        "floor": {"type": "rectangle", "bsdf": {"type": "ref", "id": "glossy-floor"},
                  "to_world": T().translate([0, -1, 0]).rotate([1, 0, 0], -90)},
        "ceiling": {"type": "rectangle", "bsdf": {"type": "ref", "id": "white"},
                    "to_world": T().translate([0, 1, 0]).rotate([1, 0, 0], 90)},
        "back": {"type": "rectangle", "bsdf": {"type": "ref", "id": "white"},
                 "to_world": T().translate([0, 0, -1])},
        "green-wall": {"type": "rectangle", "bsdf": {"type": "ref", "id": "green"},
                       "to_world": T().translate([1, 0, 0]).rotate([0, 1, 0], -90)},
        "red-wall": {"type": "rectangle", "bsdf": {"type": "ref", "id": "red"},
                     "to_world": T().translate([-1, 0, 0]).rotate([0, 1, 0], 90)},
        "tall-box": {"type": "cube", "bsdf": {"type": "ref", "id": "white"},
                     "to_world": T().translate([-0.33, -0.4, -0.28]).rotate([0, 1, 0], 18.25)
                     .scale([0.3, 0.61, 0.3])},
        "glass-ball": {"type": "sphere", "center": [0.38, -0.62, 0.3], "radius": 0.38,
                       "bsdf": {"type": "dielectric", "int_ior": 1.5}},
        "copper-ball": {"type": "sphere", "center": [0.55, -0.82, -0.55], "radius": 0.18,
                        "bsdf": {"type": "roughconductor", "material": "Cu", "alpha": 0.15}},
    }


SCENES = {
    "cornell": {
        "build": _cornell,
        # Lights live inside the box, plus a slab in front of the opening.
        "light_bbox": [[-0.97, -0.97, -0.97], [0.97, 0.97, 2.0]],
        "radius_range": [0.04, 0.3],
        # Reference lights shown in the viewer's Accuracy tab and scored by evaluate.py.
        "test_lights": [
            [0.0, 0.75, 0.0, 0.15],
            [-0.55, 0.1, 0.45, 0.12],
            [0.6, -0.1, 0.6, 0.08],
            [0.0, 0.2, 1.6, 0.3],
            [0.35, -0.2, 0.3, 0.07],   # just above the glass ball
            [-0.1, -0.85, -0.6, 0.1],  # behind the tall box, near the floor
        ],
        # The viewer's starting lights.
        "default_lights": [
            {"name": "Key", "pos": [0.1, 0.8, 0.1], "radius": 0.12, "color": [1, 0.86, 0.68], "intensity": 18},
            {"name": "Fill", "pos": [-0.72, 0.15, 0.75], "radius": 0.09, "color": [0.6, 0.75, 1], "intensity": 10},
        ],
    },
    # 'Contemporary Bathroom' by Mareck (CC0), Mitsuba port by Benedikt Bitterli. The camera stands
    # near the z = -0.05 wall looking at the vanity wall (z = -2.43) and the picture wall (x = -2.5);
    # the tub is in front of the window, and the room continues behind the camera to x = 2.5.
    "bathroom": {
        "build": _bathroom,
        # The visible half of the room plus about a metre behind the camera.
        "light_bbox": [[-2.45, 0.05, -2.38], [0.8, 2.6, -0.12]],
        "radius_range": [0.05, 0.35],
        # The viewer's "random lights" stay in front of the camera, where they are visible.
        "random_bbox": [[-2.3, 0.3, -2.2], [-0.4, 2.4, -0.3]],
        "test_lights": [
            [-1.3, 2.1, -1.9, 0.12],   # among the pendant bulbs above the vanity
            [-1.4, 1.2, -0.55, 0.15],  # over the tub, in front of the window
            [-1.0, 1.4, -1.2, 0.2],    # middle of the room
            [-1.0, 0.25, -1.5, 0.08],  # low, in front of the vanity
            [0.3, 1.6, -0.4, 0.25],    # behind the camera (only indirect light on screen)
            [-1.6, 1.3, -2.2, 0.06],   # small, close to the mirror: sharp reflection
        ],
        "default_lights": [
            {"name": "Pendant", "pos": [-1.3, 1.9, -1.95], "radius": 0.1, "color": [1, 0.8, 0.55], "intensity": 60},
            {"name": "Window fill", "pos": [-2.15, 1.6, -0.9], "radius": 0.2, "color": [0.65, 0.78, 1], "intensity": 16},
        ],
    },
}


def is_usd(spec):
    return str(spec).lower().endswith(USD_SUFFIXES)


def scene_id(spec, name=None):
    """Cache id of a scene: the built-in name, or the USD file's stem unless `name` is given."""
    return name or (Path(spec).stem if is_usd(spec) else spec)


def _cached_meta(name):
    p = WORK_DIR / "cache" / name / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def build(spec, res, name=None, **usd_opts):
    """(Mitsuba scene dict or XML path, config) for a built-in scene, a USD file, or the id of a USD scene
    imported before (whose meta.json records the file, camera and normalisation)."""
    if spec in SCENES:
        cfg = {k: v for k, v in SCENES[spec].items() if k != "build"}
        return SCENES[spec]["build"](res), cfg
    import usd_scene
    if is_usd(spec):
        return usd_scene.build(spec, res, scene_dir(scene_id(spec, name)), **usd_opts)
    meta = _cached_meta(spec)
    if meta and "source" in meta:
        src = meta["source"]
        d, cfg = usd_scene.build(src["usd"], res, scene_dir(spec), camera=src.get("camera"), time=src.get("time"),
                                 max_texture=src.get("max_texture", 2048), normalize=meta["normalize"])
        cfg.update(light_bbox=meta["light_bbox"], radius_range=meta["radius_range"])
        return d, cfg
    raise KeyError(f"unknown scene {spec!r}: not built in ({', '.join(SCENES)}), not a USD file "
                   f"({', '.join(USD_SUFFIXES)}), and not imported before")


def load(spec, res, **kw):
    d, cfg = build(spec, res, **kw)
    if isinstance(d, dict):
        return mi.load_dict(d), cfg
    # XML scenes: square film, box filter like the dict scenes.
    return mi.load_file(str(d), resx=res, resy=res), cfg


def config(meta):
    """Per-scene extras (test and default lights, the viewer's random-light box) for a meta.json: from the meta itself, or from
    the built-in definition for dumps made before these were recorded there."""
    builtin = SCENES.get(meta["scene"], {})
    return {k: meta.get(k, builtin.get(k)) for k in ("test_lights", "default_lights", "random_bbox")}
