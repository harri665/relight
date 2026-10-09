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

--scene takes a built-in scene name or a USD file (see usd_scene.py); a USD scene's dump goes to
cache/<file stem or --name>/ and later steps use that id as their --scene.
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
        # Hits beyond FAR count as escapes: light from there is negligible for lights near the
        # scene, and it keeps vertices within fp16 range for stages with huge ground planes.
        hit = active & si.is_valid() & (si.t < FAR)
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


def render_aux(scene, spp, chunk=16):
    """Per-pixel albedo, shading normal, position and camera distance (box-filtered over spp).
    Like Mitsuba's aov integrator, but it looks through pass-through (null) interactions such as
    alpha cutouts, so leaves and fences get the aux values of whatever is visible through them."""
    sensor = scene.sensors()[0]
    W, H = sensor.film().crop_size()
    ctx = mi.BSDFContext()
    acc = np.zeros((H * W, 10))
    for s0 in range(0, spp, chunk):
        k = min(chunk, spp - s0)
        n = W * H * k
        sampler = mi.load_dict({"type": "independent"})
        sampler.seed(77 + s0, n)
        pix = dr.arange(mi.UInt32, n) // k
        px = mi.Point2f(mi.Float(pix % W), mi.Float(pix // W))
        ray, _ = sensor.sample_ray(0.0, sampler.next_1d(), (px + sampler.next_2d()) / mi.ScalarVector2f(W, H),
                                   sampler.next_2d())
        o = mi.Point3f(ray.o)
        albedo, normal, pos = mi.Color3f(0.0), mi.Vector3f(0.0), mi.Point3f(0.0)
        dist = mi.Float(0.0)
        active = dr.full(mi.Bool, True, n)
        for _ in range(16):
            si = scene.ray_intersect(ray, active)
            hit = active & si.is_valid()
            bsdf = si.bsdf(ray)
            bs, _ = bsdf.sample(ctx, si, sampler.next_1d(hit), sampler.next_2d(hit), hit)
            through = hit & mi.has_flag(bs.sampled_type, mi.BSDFFlags.Null)
            done = hit & ~through
            albedo = dr.select(done, bsdf.eval_diffuse_reflectance(si, done), albedo)
            normal = dr.select(done, si.sh_frame.n, normal)
            pos = dr.select(done, si.p, pos)
            dist = dr.select(done, dr.norm(si.p - o), dist)
            active = through
            ray = si.spawn_ray(ray.d)
            sampler.schedule_state()
            dr.eval(albedo, normal, pos, dist, active, ray)
            if not dr.any(active):
                break
        a = np.concatenate([to_np(albedo).T, to_np(normal).T, to_np(pos).T, np.array(dist)[:, None]], 1)
        acc += a.reshape(H * W, k, 10).sum(1)
    img = (acc / spp).reshape(H, W, 10)
    return img[..., 0:3], img[..., 3:6], img[..., 6:9], img[..., 9:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="cornell", help="built-in scene name or USD file")
    ap.add_argument("--name", default=None, help="id for a USD scene (default: file name without extension)")
    ap.add_argument("--camera", default=None, help="USD: camera prim path (default: the first camera)")
    ap.add_argument("--time", type=float, default=None, help="USD: time code (default: start of the stage)")
    ap.add_argument("--light-bbox", type=float, nargs=6, default=None, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
                    help="USD: light domain in normalised coordinates (default: from what the camera sees)")
    ap.add_argument("--light-margin", type=float, default=None,
                    help="USD: grow the default light domain by this much (default: 0.3 open, -0.03 enclosed)")
    ap.add_argument("--radius-range", type=float, nargs=2, default=None, help="USD: light radius range")
    ap.add_argument("--max-texture", type=int, default=2048, help="USD: downsample larger textures")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--spp", type=int, default=128)
    ap.add_argument("--max-seg", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=16, help="spp traced per launch")
    args = ap.parse_args()
    if args.spp & (args.spp - 1):
        ap.error("--spp must be a power of two (the gather kernel sums each pixel's samples as a block)")

    name = scenes.scene_id(args.scene, args.name)
    usd = {}
    if scenes.is_usd(args.scene):
        usd = dict(name=name, camera=args.camera, time=args.time, max_texture=args.max_texture,
                   light_margin=args.light_margin, radius_range=args.radius_range,
                   light_bbox=[args.light_bbox[:3], args.light_bbox[3:]] if args.light_bbox else None)
        print(f"importing {args.scene} as '{name}'")
    scene, cfg = scenes.load(args.scene, args.res, **usd)
    sensor = scene.sensors()[0]
    W = H = args.res
    P, S, D = W * H, args.spp, args.max_seg
    out = scene_dir(name)
    print(f"light domain {cfg['light_bbox']}, radius {cfg['radius_range']}")

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
        "scene": name, "width": W, "height": H, "spp": S, "max_seg": D, "far": FAR,
        "camera": {
            "to_world": np.array(sensor.world_transform().matrix).reshape(4, 4).tolist(),
            "fov": float(params["x_fov"][0]) if "x_fov" in params else 39.3077,
            "fov_axis": "x",
        },
        "light_bbox": cfg["light_bbox"],
        "radius_range": cfg["radius_range"],
        **{k: cfg[k] for k in ("source", "normalize", "test_lights", "default_lights", "random_bbox") if k in cfg},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"done in {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
