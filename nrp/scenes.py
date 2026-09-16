"""Scene definitions for the light-agnostic path sampling pass.

Scenes contain no emitters: all illumination comes from virtual lights that are
added after the fact by GATHERLIGHT / the neural render proxy.

Each scene also defines the domain from which training lights are drawn
(`light_bbox`, `radius_range`); the web viewer clamps lights to the same domain,
because the network is only trained inside it.
"""
import mitsuba as mi


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
    },
}


def load(name, res):
    cfg = SCENES[name]
    return mi.load_dict(cfg["build"](res)), cfg
