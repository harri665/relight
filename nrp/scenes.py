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
import json
from pathlib import Path

import mitsuba as mi

from common import WORK_DIR, scene_dir

USD_SUFFIXES = (".usd", ".usda", ".usdc", ".usdz")


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
        "test_lights": [
            [0.0, 0.75, 0.0, 0.15],
            [-0.55, 0.1, 0.45, 0.12],
            [0.6, -0.1, 0.6, 0.08],
            [0.0, 0.2, 1.6, 0.3],
            [0.35, -0.2, 0.3, 0.07],   # just above the glass ball
            [-0.1, -0.85, -0.6, 0.1],  # behind the tall box, near the floor
        ],
        "default_lights": [
            {"name": "Key", "pos": [0.1, 0.8, 0.1], "radius": 0.12, "color": [1, 0.86, 0.68], "intensity": 18},
            {"name": "Fill", "pos": [-0.72, 0.15, 0.75], "radius": 0.09, "color": [0.6, 0.75, 1], "intensity": 10},
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
    """(Mitsuba scene dict, config) for a built-in scene, a USD file, or the id of a USD scene
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
    return mi.load_dict(d), cfg


def config(meta):
    """Per-scene extras (test and default lights) for a meta.json: from the meta itself, or from
    the built-in definition for dumps made before these were recorded there."""
    builtin = SCENES.get(meta["scene"], {})
    return {k: meta.get(k, builtin.get(k)) for k in ("test_lights", "default_lights")}
