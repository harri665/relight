"""Evaluate trained proxies against a high-quality reference.

The reference is GATHERLIGHT on freshly traced light-agnostic paths at --ref-spp
(streamed in chunks, never stored), denoised, plus the exact direct-view term.
Also reports how good the training targets themselves are (denoised gather of
the stored --spp dump), i.e. the floor that supervision quality puts on a model.

Images are exposure-normalised like the viewer (reference mean luminance 0.15),
then compared in Reinhard-tonemapped space (PSNR) and linear space (relative MAE).
The average hides dark regions (shadows, indirect light, dim lights), so it also reports
relative MAE over the darkest quarter of each reference and PSNR at +3 EV (exposure x 8).

    python evaluate.py --runs cornell_128x4 cornell_256x4
"""
import argparse
import json
import time

import numpy as np
import torch

import sample_paths  # sets the Mitsuba variant
import scenes
from common import WORK_DIR, load_meta, scene_dir
from denoise import make_denoiser
from direct import direct_view
from gather import PathData, gather_arrays, medium_delta
from model import NRP, PixelBuffers, predict_image

def eval_lights(meta, n, seed):
    """The scene's viewer test lights (if it defines any) followed by n random ones."""
    g = torch.Generator().manual_seed(seed)
    lo, hi = torch.tensor(meta["light_bbox"][0]), torch.tensor(meta["light_bbox"][1])
    rmin, rmax = meta["radius_range"]
    c = lo + (hi - lo) * torch.rand(n, 3, generator=g)
    r = rmin + (rmax - rmin) * torch.rand(n, 1, generator=g)
    tests = torch.tensor(scenes.config(meta).get("test_lights") or [], dtype=torch.float32).reshape(-1, 4)
    return torch.cat([tests, torch.cat([c, r], -1)]).cuda(), len(tests)


@torch.no_grad()
def reference(scene_name, meta, lights, spp, chunk=16):
    """Indirect (segments >= 1) gather at `spp`, streamed."""
    scene, _ = scenes.load(scene_name, meta["width"])
    sensor = scene.sensors()[0]
    W, H, D = meta["width"], meta["height"], meta["max_seg"]
    NP = W * H
    acc = torch.zeros(len(lights), NP, 3, device="cuda")
    t0 = time.time()
    for s0 in range(0, spp, chunk):
        v, t = sample_paths.trace_chunk(scene, sensor, (W, H), chunk, seed=10_000_000 + s0, max_seg=D,
                                        medium=meta.get("medium"))
        V = torch.from_numpy(v).cuda().half()
        T = torch.from_numpy(np.clip(t, -6e4, 6e4)).cuda().half()
        acc += gather_arrays(V, T, NP, chunk, lights, first_seg=1, delta=medium_delta(meta.get("medium")))
        if (s0 // chunk) % 16 == 0:
            print(f"  reference spp {s0 + chunk}/{spp} ({time.time() - t0:.0f}s)", flush=True)
    return acc / (spp // chunk)


def metrics(img, ref):
    """img, ref: [NP, 3] linear. Exposure-normalised tonemapped PSNR, relative MAE, relative MAE
    over the darkest 25 % of the reference's pixels, and tonemapped PSNR at +3 EV."""
    Y = lambda x: (x * torch.tensor([0.2126, 0.7152, 0.0722], device=x.device)).sum(-1)
    s = 0.15 / Y(ref).mean().clamp_min(1e-8)
    tm = lambda x, k: (x * s * k).clamp_min(0) / (1 + (x * s * k).clamp_min(0))
    psnr = lambda k: (-10 * torch.log10(((tm(img, k) - tm(ref, k)) ** 2).mean())).item()
    rel = lambda m: ((img[m] - ref[m]).abs().sum() / ref[m].abs().sum().clamp_min(1e-12)).item()
    yr = Y(ref)
    dark = yr <= torch.quantile(yr, 0.25)
    return psnr(1), rel(slice(None)), rel(dark), psnr(8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="cornell")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=32, help="random lights (plus the scene's viewer tests)")
    ap.add_argument("--ref-spp", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    meta = load_meta(args.scene)
    aux = torch.from_numpy(np.load(scene_dir(args.scene) / "aux.npy")).cuda()
    meta["depth_scale"] = min(float(aux[..., 9].max()), meta.get("far", 40.0))
    lights, n_tests = eval_lights(meta, args.n, args.seed)
    den = make_denoiser(aux[..., 0:3], aux[..., 3:6])
    H, W = meta["height"], meta["width"]
    NP = H * W
    directs = [direct_view(meta, aux, l) for l in lights]

    cache = WORK_DIR / "cache" / args.scene / f"eval_ref_{args.n}_{args.seed}_{args.ref_spp}.pt"
    if cache.exists():
        ref = torch.load(cache).cuda()
    else:
        raw = reference(args.scene, meta, lights, args.ref_spp)
        ref = torch.stack([den(r.view(H, W, 3)).reshape(NP, 3) for r in raw])
        torch.save(ref.cpu(), cache)
    ref = ref + torch.stack(directs)
    # Lights inside solid objects (tall box, balls) light nothing; they are not meaningful tests.
    keep = [i for i in range(len(lights)) if ref[i].mean() > 1e-5]
    print(f"{len(lights) - len(keep)} of {len(lights)} lights are inside geometry and skipped")
    lights, ref, directs = lights[keep], ref[keep], [directs[i] for i in keep]
    n_tests = sum(i < n_tests for i in keep)

    results = {}
    # Supervision quality: denoised gather of the stored dump (what training sees).
    pd = PathData(args.scene)
    part = pd.gather(lights, first_seg=1)
    tgt = [den(im).reshape(NP, 3) + d for im, d in zip(part, directs)]
    results[f"targets ({meta['spp']} spp, denoised)"] = [metrics(t, r) for t, r in zip(tgt, ref)]
    del pd, part
    torch.cuda.empty_cache()

    for run in args.runs:
        ck = torch.load(WORK_DIR / "runs" / run / "model.pt", weights_only=False)
        model = NRP(**ck["cfg"]).cuda().eval()
        model.load_state_dict(ck["state"])
        bufs = PixelBuffers(aux, ck["meta"], aux_dim=ck["cfg"].get("aux_dim", 7))
        results[run] = [metrics(predict_image(model, bufs, l) + d, r) for l, d, r in zip(lights, directs, ref)]

    head = f"viewer tests 1..{n_tests} (dB)" if n_tests else ""
    print(f"\n{'':44s} {'PSNR all':>9s} {head:>40s} {'rel.MAE':>8s} {'worst 5 mean':>13s}"
          f" {'dark rel.MAE':>13s} {'PSNR +3EV':>10s}")
    summary = {}
    for name, m in results.items():
        p = np.array([x[0] for x in m]); e = np.array([x[1] for x in m])
        ed = np.array([x[2] for x in m]); p8 = np.array([x[3] for x in m])
        tests = " ".join(f"{x:5.1f}" for x in p[:n_tests])
        worst = np.sort(p)[:5].mean()
        print(f"{name:44s} {p.mean():8.2f}  {tests:>40s} {100 * e.mean():7.1f}% {worst:12.2f}"
              f" {100 * ed.mean():12.1f}% {p8.mean():10.2f}")
        summary[name] = {"psnr_mean": float(p.mean()), "psnr": p.tolist(), "rel_mae_mean": float(e.mean()), "worst5": float(worst),
                         "dark_rel_mae_mean": float(ed.mean()), "psnr_3ev_mean": float(p8.mean())}
    (WORK_DIR / "eval.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
