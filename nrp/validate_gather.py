"""Sanity check: GATHERLIGHT on the light-agnostic path dump vs. a regular Mitsuba
render (with NEE + MIS) of the same scene containing a real sphere emitter."""
import argparse

import numpy as np
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")
import torch  # noqa: E402

import scenes  # noqa: E402
from common import WORK_DIR, load_meta  # noqa: E402
from gather import PathData  # noqa: E402
from denoise import Denoiser  # noqa: E402


def reference(scene_name, res, light, max_depth, spp):
    d = scenes.SCENES[scene_name]["build"](res)
    if not isinstance(d, dict):
        # XML scene: include it next to the extra light, so its relative asset paths still resolve.
        x, y, z, r = light
        wrapper = d.with_name("validate_tmp.xml")
        wrapper.write_text(f"""<scene version="3.0.0">
  <include filename="{d.name}"/>
  <shape type="sphere"><point name="center" x="{x}" y="{y}" z="{z}"/><float name="radius" value="{r}"/>
    <bsdf type="diffuse"><rgb name="reflectance" value="0"/></bsdf>
    <emitter type="area"><rgb name="radiance" value="1"/></emitter></shape>
</scene>""")
        integ = mi.load_dict({"type": "path", "max_depth": max_depth, "rr_depth": 1000})
        return np.array(mi.render(mi.load_file(str(wrapper), resx=res, resy=res), integrator=integ, spp=spp))
    d["integrator"] = {"type": "path", "max_depth": max_depth, "rr_depth": 1000}
    d["virtual-light"] = {
        "type": "sphere", "center": light[:3], "radius": light[3],
        "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": 0.0}},
        "emitter": {"type": "area", "radiance": {"type": "rgb", "value": 1.0}},
    }
    return np.array(mi.render(mi.load_dict(d), spp=spp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="cornell")
    ap.add_argument("--spp", type=int, default=2048)
    args = ap.parse_args()
    meta = load_meta(args.scene)
    pd = PathData(args.scene)
    den = Denoiser(pd.aux[..., 0:3].cpu().numpy(), pd.aux[..., 3:6].cpu().numpy())
    out = WORK_DIR / "validate" / args.scene
    out.mkdir(exist_ok=True, parents=True)
    lights = scenes.SCENES[args.scene]["test_lights"][:3]
    if args.scene == "cornell":  # the lights this check was originally run with
        lights = [[0.0, 0.6, 0.0, 0.2], [-0.5, -0.2, 0.5, 0.1], [0.3, 0.3, 1.5, 0.3]]
    for i, light in enumerate(lights):
        g = pd.gather(torch.tensor(light))[0].cpu().numpy()
        ref = reference(args.scene, meta["width"], light, meta["max_seg"], args.spp)
        dn = den(g)
        rel = lambda a: np.abs(a - ref).mean() / ref.mean()
        print(f"light {light}: mean gather {g.mean():.4f} ref {ref.mean():.4f} | "
              f"rel. MAE raw {rel(g):.3f} denoised {rel(dn):.3f}")
        tm = lambda x: (np.clip(x / (1 + x), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
        from PIL import Image
        Image.fromarray(np.concatenate([tm(g), tm(dn), tm(ref)], 1)).save(out / f"light{i}.png")


if __name__ == "__main__":
    main()
