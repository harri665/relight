"""SAMPLEPATHS (paper Sec. 3.1 / Fig. 3b).

Traces camera paths with pure BSDF sampling (no NEE: the lights are unknown)
and stores, per path, the vertices x_0..x_D and throughputs T_0..T_{D-1}.
Segment j runs x_j -> x_{j+1} and carries weight T_j. Paths that escape get a
far-away end vertex so the escaping segment can still hit a virtual light;
segments after termination have zero weight.

Output layout (structure-of-arrays, fp16, path index = pixel * spp + sample):
    verts.npy  [D+1, 3, P*S]
    thr.npy    [D,   3, P*S]
    aux.npy    [H, W, 10]  albedo(3) normal(3) position(3) distance(1)
    meta.json
"""
import argparse
import json
import time

import numpy as np
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")
import drjit as dr  # noqa: E402

from common import scene_dir  # noqa: E402
import scenes  # noqa: E402

FAR = 40.0


def to_np(v):
    """Dr.Jit 3-vector -> numpy [3, n]."""
    a = np.array(v)
    return a if a.shape[0] == 3 else a.T


def trace_chunk(scene, sensor, res, k, seed, max_seg):
    W, H = res
    n = W * H * k
    sampler = mi.load_dict({"type": "independent"})
    sampler.seed(seed, n)
    idx = dr.arange(mi.UInt32, n)
    pix = idx // k
    px = mi.Point2f(mi.Float(pix % W), mi.Float(pix // W))
    film_pos = (px + sampler.next_2d()) / mi.ScalarVector2f(W, H)
    ray, _ = sensor.sample_ray(0.0, sampler.next_1d(), film_pos, sampler.next_2d())

    ctx = mi.BSDFContext()
    T = dr.full(mi.Color3f, 1.0, n)
    active = dr.full(mi.Bool, True, n)
    prev = mi.Point3f(ray.o)
    verts, thr = [to_np(prev)], []
    for _ in range(max_seg):
        si = scene.ray_intersect(ray, active)
        hit = active & si.is_valid()
        end = dr.select(hit, si.p, dr.select(active, ray.o + ray.d * FAR, prev))
        seg_T = dr.select(active, T, 0.0)
        bsdf = si.bsdf(ray)
        bs, w = bsdf.sample(ctx, si, sampler.next_1d(hit), sampler.next_2d(hit), hit)
        T = dr.select(hit, T * w, 0.0)
        active = hit & ((T[0] > 0) | (T[1] > 0) | (T[2] > 0))
        ray = si.spawn_ray(si.to_world(bs.wo))
        prev = end
        sampler.schedule_state()
        dr.eval(T, active, ray, prev, seg_T)
        verts.append(to_np(end))
        thr.append(to_np(seg_T))
    return np.stack(verts), np.stack(thr)  # [D+1,3,n], [D,3,n]


def render_aux(scene, spp):
    integ = mi.load_dict({"type": "aov", "aovs": "albedo:albedo,nn:sh_normal,pp:position,dd:depth"})
    img = np.array(mi.render(scene, integrator=integ, spp=spp))
    names = integ.aov_names()
    # The aov integrator prepends RGB channels from its (empty) inner integrator.
    off = img.shape[-1] - len(names)
    ch = {n: off + i for i, n in enumerate(names)}
    pick = lambda pre: img[..., [ch[n] for n in names if n.startswith(pre + ".")]]
    return pick("albedo"), pick("nn"), pick("pp"), pick("dd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="cornell")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--spp", type=int, default=128)
    ap.add_argument("--max-seg", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=16, help="spp traced per launch")
    args = ap.parse_args()

    scene, cfg = scenes.load(args.scene, args.res)
    sensor = scene.sensors()[0]
    W = H = args.res
    P, S, D = W * H, args.spp, args.max_seg
    out = scene_dir(args.scene)

    t0 = time.time()
    verts = np.lib.format.open_memmap(out / "verts.npy", "w+", np.float16, (D + 1, 3, P, S))
    thr = np.lib.format.open_memmap(out / "thr.npy", "w+", np.float16, (D, 3, P, S))
    for s0 in range(0, S, args.chunk):
        k = min(args.chunk, S - s0)
        v, t = trace_chunk(scene, sensor, (W, H), k, seed=1000 + s0, max_seg=D)
        verts[..., s0:s0 + k] = v.reshape(D + 1, 3, P, k)
        thr[..., s0:s0 + k] = np.minimum(t, 6e4).reshape(D, 3, P, k)
        print(f"  traced spp {s0 + k}/{S}  ({time.time() - t0:.1f}s)", flush=True)
    verts.flush(); thr.flush()
    del verts, thr

    albedo, normal, pos, dist = render_aux(scene, spp=64)
    aux = np.concatenate([albedo, normal, pos, dist], -1).astype(np.float32)
    np.save(out / "aux.npy", aux)

    params = mi.traverse(sensor)
    meta = {
        "scene": args.scene, "width": W, "height": H, "spp": S, "max_seg": D, "far": FAR,
        "camera": {
            "to_world": np.array(sensor.world_transform().matrix).reshape(4, 4).tolist(),
            "fov": float(params["x_fov"][0]) if "x_fov" in params else 39.3077,
            "fov_axis": "x",
        },
        "light_bbox": cfg["light_bbox"],
        "radius_range": cfg["radius_range"],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"done in {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
