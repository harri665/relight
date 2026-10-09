"""Export a trained NRP for the web viewer.

Writes client/public/scenes/<name>/scene.json plus binary blobs:
  model.bin  grid levels (f16, [R, R, F] row-major) + MLP layers (f32, [out, in] row-major, bias)
  pixels.bin per-pixel network aux features (f16) and surface positions (u16 over the scene's extent,
             0 = miss), packed so they compress well (pixel format 2, see pixels.py)
  refs.bin   f16 [n, H, W, 3] reference images (denoised gather + analytic direct term) for n test lights

Test lights and the viewer's starting lights come from the scene (scenes.py / the USD stage's
lights). Where a scene has none, they are picked from random lights that light the view well;
imported USD lights keep their colours and relative intensities but are scaled to a good exposure.

With --res N (other than the resolution the paths were traced at) it exports the same network at
another image size instead: the per-pixel buffers are rendered afresh at N x N (the network's pixel
input is a continuous uv, so nothing is retrained), there are no reference images (those need the
path dump), and it writes scene-N.json and pixels-N.bin next to the usual files, which it leaves be.
The size is listed under "tiers" in the scene index so the viewer offers it.
"""
import argparse
import json

import numpy as np
import torch

import scenes
from common import WEB_SCENES_DIR, WORK_DIR
from direct import direct_view
from model import NRP, PixelBuffers
from pixels import pack

LUM = [0.2126, 0.7152, 0.0722]
WARM, COOL = [1, 0.86, 0.68], [0.6, 0.75, 1]


class Blob:
    def __init__(self):
        self.parts, self.size = [], 0

    def add(self, arr, dtype):
        a = np.ascontiguousarray(arr, dtype=dtype)
        pad = (-self.size) % 16
        if pad:
            self.parts.append(b"\0" * pad)
            self.size += pad
        entry = {"offset": self.size, "dtype": np.dtype(dtype).name, "shape": list(a.shape)}
        self.parts.append(a.tobytes())
        self.size += a.nbytes
        return entry

    def write(self, path):
        path.write_bytes(b"".join(self.parts))


def random_lights(meta, n, seed, upper=0.0):
    """n random lights in the light domain; `upper` > 0 keeps them in the top part of it (in y)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    lo, hi = torch.tensor(meta["light_bbox"], device="cuda")
    lo = lo.clone()
    lo[1] += upper * (hi[1] - lo[1])
    rmin, rmax = meta["radius_range"]
    c = lo + (hi - lo) * torch.rand(n, 3, device="cuda", generator=g)
    r = rmin + (rmax - rmin) * (0.1 + 0.4 * torch.rand(n, 1, device="cuda", generator=g))
    return torch.cat([c, r], -1)


@torch.no_grad()
def light_scores(pd, meta, aux, lights):
    """Mean luminance each light puts on the image (per unit radiance, camera-visible disc
    excluded) and the fraction of the image its disc covers."""
    lum = torch.tensor(LUM, device="cuda")
    m = torch.cat([(pd.gather(lights[i:i + 16], first_seg=1) * lum).sum(-1).flatten(1).mean(1)
                   for i in range(0, len(lights), 16)])
    cover = torch.stack([direct_view(meta, aux, l, ss=1).mean() for l in lights])
    return m, cover


def clamp_light(meta, l):
    lo, hi = meta["light_bbox"]
    pos = [min(max(x, a), b) for x, a, b in zip(l["pos"], lo, hi)]
    radius = min(max(l["radius"], meta["radius_range"][0]), meta["radius_range"][1])
    return {**l, "pos": pos, "radius": radius}


def default_lights(pd, meta, aux, given):
    """Viewer starting lights: the scene's own (built-in ones as they are, imported ones scaled so
    the image is well exposed), or a warm key and a cool fill picked from random candidates."""
    if given and not any(l.get("auto_intensity") for l in given):
        return given
    lum = lambda c: sum(a * b for a, b in zip(LUM, c))
    if given:
        lights = [clamp_light(meta, l) for l in given]
        m, _ = light_scores(pd, meta, aux, torch.tensor([l["pos"] + [l["radius"]] for l in lights], device="cuda"))
        total = sum(l["intensity"] * lum(l["color"]) * float(s) for l, s in zip(lights, m))
        k = 0.14 / max(total, 1e-12)
        return [{k2: v for k2, v in {**l, "intensity": round(l["intensity"] * k, 4)}.items() if k2 != "auto_intensity"}
                for l in lights]
    cand = random_lights(meta, 96, seed=7, upper=0.35)
    m, cover = light_scores(pd, meta, aux, cand)
    score = torch.where(cover < 0.01, m, torch.zeros_like(m))  # don't start with a light in the viewer's face
    key = int(score.argmax())
    size = torch.tensor(meta["light_bbox"], device="cuda").diff(dim=0).norm()
    far = (cand[:, :3] - cand[key, :3]).norm(dim=-1) > 0.3 * size
    fill = int(torch.where(far, score, torch.zeros_like(score)).argmax())
    out = []
    for name, i, color, target in [("Key", key, WARM, 0.10), ("Fill", fill, COOL, 0.04)]:
        if score[i] <= 0:
            continue
        L = cand[i].tolist()
        out.append({"name": name, "pos": [round(x, 3) for x in L[:3]], "radius": round(L[3], 3), "color": color,
                    "intensity": round(target / (float(m[i]) * lum(color)), 3)})
    return out


def test_lights(pd, meta, aux, n):
    """The scene's test lights, or n random ones that light a fair part of the image."""
    given = scenes.config(meta).get("test_lights")
    if given:
        return torch.tensor(given, device="cuda")[:n]
    cand = random_lights(meta, 16 * n, seed=11)
    m, cover = light_scores(pd, meta, aux, cand)
    ok = (m > 0.05 * m.max()) & (cover < 0.05)
    return cand[ok][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cornell_128x4")
    ap.add_argument("--name", default=None)
    ap.add_argument("--refs", type=int, default=6)
    ap.add_argument("--label", default=None, help="name shown in the viewer's model picker")
    ap.add_argument("--res", type=int, default=None, help="export at this image size (see above)")
    args = ap.parse_args()
    ck = torch.load(WORK_DIR / "runs" / args.run / "model.pt", weights_only=False)
    meta, cfg = ck["meta"], ck["cfg"]
    name = args.name or meta["scene"]
    out = WEB_SCENES_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    model = NRP(**cfg)
    model.load_state_dict(ck["state"])
    tier = args.res if args.res and args.res != meta["width"] else None
    if tier:
        meta = {**meta, "width": tier, "height": tier}
    H, W = meta["height"], meta["width"]
    suffix = f"-{tier}" if tier else ""

    mb = Blob()
    grids = [mb.add(g.detach()[0].permute(1, 2, 0).numpy(), np.float16) for g in model.enc.grids]
    layers = [{"weight": mb.add(l.weight.detach().numpy(), np.float32),
               "bias": mb.add(l.bias.detach().numpy(), np.float32)} for l in model.layers]
    # 3D light-position grids, [R, R, R, F] with index order (z, y, x) like grid_sample's (D, H, W).
    lgrids = [mb.add(g.detach()[0].permute(1, 2, 3, 0).numpy(), np.float32) for g in model.lenc.grids]         if model.lenc else []
    if not tier:
        mb.write(out / "model.bin")

    # Per-pixel buffers.
    if tier:
        from sample_paths import render_aux
        albedo, normal, pos, dist = render_aux(scenes.load(meta["scene"], tier)[0], spp=64)
        aux = torch.from_numpy(np.concatenate([albedo, normal, pos, dist], -1).astype(np.float32)).cuda()
    else:
        from gather import PathData
        pd = PathData(meta["scene"])
        aux = pd.aux
    bufs = PixelBuffers(aux, meta, aux_dim=cfg.get("aux_dim", 7))
    feats = bufs.aux.reshape(H, W, -1)
    # The viewer derives the camera distance from the position and the normals from the aux features.
    data, pix = pack(feats.cpu().numpy().astype(np.float16), aux[..., 6:9].float().cpu().numpy(),
                     (aux[..., 9] > 0).cpu().numpy())
    (out / f"pixels{suffix}.bin").write_bytes(data)

    # Reference renders for a few test lights, and the viewer's starting lights.
    tests, starts = [], None
    if tier and (out / "scene.json").exists():  # same lights as the native-size export
        starts = json.loads((out / "scene.json").read_text()).get("default_lights")
    if not tier:
        from denoise import make_denoiser
        den = make_denoiser(aux[..., 0:3], aux[..., 3:6])
        tests = test_lights(pd, meta, aux, args.refs) if args.refs else []
        starts = default_lights(pd, meta, aux, scenes.config(meta).get("default_lights"))
    rb = Blob()
    fs = meta.get("first_seg", 0)
    refs = []
    for l in tests:
        img = pd.gather(l, first_seg=fs)[0]
        img = den(img).reshape(-1, 3)
        if fs:
            img = img + direct_view(meta, aux, l)
        refs.append({"light": l.tolist(), "image": rb.add(img.reshape(H, W, 3).cpu().numpy(), np.float16)})
    if not tier:
        rb.write(out / "refs.bin")

    # Where the viewer's "random lights" go, if the scene narrows it down from light_bbox.
    random_box = scenes.config(meta).get("random_bbox")
    scene = {
        "name": name, "source_scene": meta["scene"], "width": W, "height": H,
        "camera": meta["camera"],
        "light_bbox": meta["light_bbox"], "radius_range": meta["radius_range"],
        **({"random_bbox": random_box} if random_box else {}),
        "first_seg": fs,
        "network": {"width": cfg["width"], "hidden": cfg["hidden"], "feats": cfg["feats"],
                    "grid_res": model.enc.res, "aux_dim": cfg.get("aux_dim", 7), "light_dim": 4,
                    "geo": bool(cfg.get("geo", False)), "head": cfg.get("head", "linear"),
                    "light_grid": cfg.get("light_grid", 0),
                    "light_grid_res": model.lenc.res if model.lenc else [],
                    "light_grid_feats": model.lenc.feats if model.lenc else 0},
        "model": {"grids": grids, "layers": layers, "light_grids": lgrids},
        "pixels": pix,
        "refs": refs,
        **({"default_lights": starts} if starts else {}),
        "train": {"spp": meta["spp"], "max_seg": meta["max_seg"], "iters": ck["args"]["iters"],
                  "log": ck.get("log", [])[-1:] },
    }
    (out / f"scene{suffix}.json").write_text(json.dumps(scene, indent=1))
    index_path = WEB_SCENES_DIR / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8-sig")) if index_path.exists() else []
    if tier:
        for e in index:
            if e["name"] == name:
                e["tiers"] = sorted(set(e.get("tiers", [])) | {tier})
        index_path.write_text(json.dumps(index, indent=1))
        print(f"exported {out}: scene{suffix}.json, pixels{suffix}.bin {(out / f'pixels{suffix}.bin').stat().st_size / 1e6:.1f} MB")
        return
    # Register in the viewer's scene index (keeping any other exported sizes).
    tiers = next((e.get("tiers") for e in index if e["name"] == name), None)
    index = [e for e in index if e["name"] != name]
    # family: exports of the same scene (the viewer switches between their networks); size: the
    # native image size, beside the other sizes in tiers
    index.append({"name": name, "label": args.label or f"{meta['scene']} · {cfg['width']}×{cfg['hidden']}",
                  "network": f"{cfg['width']}x{cfg['hidden']}", "family": meta["scene"], "size": W,
                  **({"tiers": tiers} if tiers else {})})
    index_path.write_text(json.dumps(sorted(index, key=lambda e: e["name"]), indent=1))
    sizes = {f: (out / f).stat().st_size / 1e6 for f in ["model.bin", "pixels.bin", "refs.bin"]}
    print(f"exported {out}: " + ", ".join(f"{k} {v:.1f} MB" for k, v in sizes.items()))


if __name__ == "__main__":
    main()
