"""Train a neural render proxy for sphere lights (paper Sec. 4.4).

For each batch every sample picks a random pixel and a random image from a pool
of denoised GATHERLIGHT reconstructions, each rendered with its own random light.
A background thread keeps producing fresh pool images (gather on GPU, denoise on
CPU); the main loop swaps them in (default: 2 images every 5 iterations).
"""
import argparse
import json
import queue
import threading
import time

import numpy as np
import torch

from common import WORK_DIR, scene_dir
from denoise import make_denoiser
from direct import direct_view
from gather import PathData
from model import NRP, PixelBuffers, predict_image, run_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def tonemap(x):
    x = x.clamp_min(0)
    return x / (1 + x)


def psnr_tm(pred, ref):
    return (-10 * torch.log10(((tonemap(pred) - tonemap(ref)) ** 2).mean())).item()


class LightSampler:
    """Light positions: uniform in the light domain and on recorded path segments (Sec. 4.4),
    optionally mixed with error-driven sampling: `cell_w` holds weights over a G^3 grid of
    the light domain (set from the running training loss per cell), and a fraction `adapt`
    of lights is drawn from cells in proportion to it."""

    def __init__(self, pd, meta, seed=0, adapt=0.0, grid=8):
        self.pd = pd
        self.lo = torch.tensor(meta["light_bbox"][0], device="cuda")
        self.hi = torch.tensor(meta["light_bbox"][1], device="cuda")
        self.rmin, self.rmax = meta["radius_range"]
        self.gen = torch.Generator(device="cuda").manual_seed(seed)
        self.adapt, self.grid = adapt, grid
        self.cell_w = None

    def cell_index(self, c):
        g = ((c - self.lo) / (self.hi - self.lo) * self.grid).long().clamp(0, self.grid - 1)
        return (g[:, 0] * self.grid + g[:, 1]) * self.grid + g[:, 2]

    def __call__(self, n):
        w = self.cell_w
        p_ad = self.adapt if w is not None else 0.0
        kind = torch.multinomial(torch.tensor([(1 - p_ad) / 2, (1 - p_ad) / 2, p_ad], device="cuda"), n,
                                 replacement=True, generator=self.gen)
        c = self.lo + (self.hi - self.lo) * torch.rand(n, 3, device="cuda", generator=self.gen)
        n_seg = int((kind == 1).sum())
        if n_seg:
            c[kind == 1] = self.pd.sample_segment_points(n_seg, self.lo, self.hi, gen=self.gen)
        n_ad = int((kind == 2).sum())
        if n_ad:
            G = self.grid
            cell = torch.multinomial(w, n_ad, replacement=True, generator=self.gen)
            ijk = torch.stack([cell // (G * G), (cell // G) % G, cell % G], -1).float()
            u = (ijk + torch.rand(n_ad, 3, device="cuda", generator=self.gen)) / G
            c[kind == 2] = self.lo + (self.hi - self.lo) * u
        r = self.rmin + (self.rmax - self.rmin) * torch.rand(n, 1, device="cuda", generator=self.gen)
        return torch.cat([c, r], -1)


class Producer(threading.Thread):
    """Generates (light, denoised image) pairs in the background."""

    def __init__(self, pd, sampler, denoiser, first_seg, batch=2, maxsize=16):
        super().__init__(daemon=True)
        self.pd, self.sampler, self.den, self.batch = pd, sampler, denoiser, batch
        self.first_seg = first_seg
        self.q = queue.Queue(maxsize=maxsize)
        self.stop = threading.Event()
        self.stream = torch.cuda.Stream()

    def make(self, n):
        with torch.cuda.stream(self.stream):
            # Lights inside solid objects light nothing; don't waste pool slots on them.
            while True:
                lights = self.sampler(n)
                imgs = self.pd.gather(lights, first_seg=self.first_seg)
                ok = imgs.flatten(1).mean(1) > 1e-6
                if ok.any():
                    lights, imgs = lights[ok], imgs[ok]
                    break
            if self.den:
                imgs = torch.stack([self.den(im) for im in imgs])
            imgs = imgs.half()
            self.stream.synchronize()
        return lights, imgs

    def run(self):
        while not self.stop.is_set():
            item = self.make(self.batch)
            while not self.stop.is_set():
                try:
                    self.q.put(item, timeout=0.5)
                    break
                except queue.Full:
                    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="cornell")
    ap.add_argument("--name", default=None)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=4)
    ap.add_argument("--levels", type=int, default=16)
    ap.add_argument("--iters", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=1 << 17)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument("--replace", type=int, default=2, help="pool images replaced per --replace-every iters")
    ap.add_argument("--replace-every", type=int, default=5)
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--no-aux", action="store_true")
    ap.add_argument("--aux-pos", action="store_true", help="add world position to the aux features")
    ap.add_argument("--geo", action="store_true", help="per (pixel, light) geometric features")
    ap.add_argument("--head", default="linear", choices=["linear", "mul"], help="mul: out = a*G + b")
    ap.add_argument("--light-grid", type=int, default=0, help="levels of a 3D grid encoding of the light position")
    ap.add_argument("--adapt", type=float, default=0.0,
                    help="fraction of training lights drawn in proportion to the running loss per light-domain cell")
    ap.add_argument("--rel-eps", type=float, default=0.01,
                    help="relative MSE: (pred - target)^2 / (pred^2 + rel_eps). Targets are transport per unit "
                         "radiance (median ~1e-3), so 0.01 makes this plain MSE for nearly every pixel")
    ap.add_argument("--first-seg", type=int, default=1,
                    help="1: network skips segment 0 (direct view, added analytically); 0: paper setup")
    args = ap.parse_args()
    name = args.name or f"{args.scene}_{args.width}x{args.hidden}"
    out_dir = WORK_DIR / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    pd = PathData(args.scene)
    meta = dict(pd.meta)
    H, W, NP = pd.H, pd.W, pd.NP
    # Capped so a far horizon does not squash the depth input of the actual scene.
    meta["depth_scale"] = min(float(pd.aux[..., 9].max()), meta.get("far", 40.0))
    bufs = PixelBuffers(pd.aux, meta, aux_dim=10 if args.aux_pos else 7)
    if args.no_aux:
        bufs.aux = torch.zeros_like(bufs.aux)

    meta["first_seg"] = args.first_seg
    denoiser = None if args.no_denoise else make_denoiser(pd.aux[..., 0:3], pd.aux[..., 3:6])
    sampler = LightSampler(pd, meta, seed=1, adapt=args.adapt)
    G3 = sampler.grid ** 3
    cell_loss = torch.zeros(G3, device="cuda")
    cell_sum = torch.zeros(G3, device="cuda")
    cell_cnt = torch.zeros(G3, device="cuda")
    producer = Producer(pd, sampler, denoiser, args.first_seg, batch=args.replace)

    # Validation set: fixed lights. References are full images (all segments):
    # "denoised" = denoised network target + analytic direct term, "raw" = plain gather.
    val_lights = LightSampler(pd, meta, seed=12345)(16)
    add_direct = lambda img, l: img + direct_view(meta, pd.aux, l) if args.first_seg else img
    val_raw = pd.gather(val_lights).reshape(16, NP, 3)
    val_part = pd.gather(val_lights, first_seg=args.first_seg)
    val_den = torch.stack([add_direct((denoiser(im) if denoiser else im).reshape(NP, 3), l)
                           for im, l in zip(val_part, val_lights)])

    # Initial pool.
    t0 = time.time()
    pool = torch.empty(args.pool, NP, 3, dtype=torch.float16, device="cuda")
    pool_lights = torch.empty(args.pool, 4, device="cuda")
    for i in range(0, args.pool, 10):
        l, im = producer.make(min(10, args.pool - i))
        pool[i:i + len(l)] = im.reshape(len(l), NP, 3)
        pool_lights[i:i + len(l)] = l
    print(f"initial pool of {args.pool} built in {time.time() - t0:.1f}s", flush=True)
    producer.start()

    model = NRP(args.width, args.hidden, levels=args.levels, max_res=max(W, H),
                aux_dim=bufs.aux.shape[1], geo=args.geo, head=args.head, light_grid=args.light_grid).cuda()
    opt = torch.optim.Adam([
        {"params": list(model.enc.parameters()) + (list(model.lenc.parameters()) if model.lenc else []),
         "lr": args.lr * 5, "eps": 1e-15},
        {"params": model.layers.parameters(), "lr": args.lr, "weight_decay": 0},
    ], betas=(0.9, 0.99), eps=1e-15)
    base_lrs = [g["lr"] for g in opt.param_groups]

    def predict(light):
        return add_direct(predict_image(model, bufs, light), light)

    def validate():
        p_den, p_raw = [], []
        for k in range(16):
            pr = predict(val_lights[k])
            p_den.append(psnr_tm(pr, val_den[k]))
            p_raw.append(psnr_tm(pr, val_raw[k]))
        return float(np.mean(p_den)), float(np.mean(p_raw))

    log = []
    t0 = time.time()
    swapped = 0
    for it in range(1, args.iters + 1):
        # Cosine decay to 5% of the base learning rate.
        f = 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * it / args.iters))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f

        if it % args.replace_every == 0:
            try:
                l, im = producer.q.get_nowait()
                slots = torch.randint(0, args.pool, (len(l),), device="cuda")
                pool[slots] = im.reshape(len(l), NP, 3)
                pool_lights[slots] = l
                swapped += len(l)
            except queue.Empty:
                pass

        pix = torch.randint(0, NP, (args.batch,), device="cuda")
        k = torch.randint(0, args.pool, (args.batch,), device="cuda")
        target = pool[k, pix].float()
        pred = run_model(model, bufs, pix, pool_lights[k])
        per = (pred - target) ** 2 / (pred.detach() ** 2 + args.rel_eps)
        loss = per.mean()
        if args.adapt > 0:
            with torch.no_grad():
                cell = sampler.cell_index(pool_lights[k][:, :3])
                cell_sum.index_add_(0, cell, per.mean(-1))
                cell_cnt.index_add_(0, cell, torch.ones_like(cell, dtype=torch.float32))
                if it % 200 == 0:
                    seen = cell_cnt > 0
                    avg = torch.where(seen, cell_sum / cell_cnt.clamp_min(1), cell_loss)
                    cell_loss = torch.where(cell_loss > 0, 0.8 * cell_loss + 0.2 * avg, avg)
                    cell_sum.zero_(); cell_cnt.zero_()
                    # unseen cells keep a floor so they are still explored
                    w = cell_loss + 0.1 * cell_loss[cell_loss > 0].mean().clamp_min(1e-8)
                    sampler.cell_w = (w / w.sum()).clone()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if it % 500 == 0 or it == args.iters:
            torch.cuda.synchronize()
            msg = f"it {it:6d}  loss {loss.item():.4f}  {(time.time() - t0) / it * 1000:.1f} ms/it  swapped {swapped}"
            if it % 2500 == 0 or it == args.iters:
                pd_, pr_ = validate()
                msg += f"  | val PSNR(tm) vs denoised {pd_:.2f} dB, vs raw {pr_:.2f} dB"
                log.append({"it": it, "loss": loss.item(), "psnr_den": pd_, "psnr_raw": pr_,
                            "time": time.time() - t0})
            print(msg, flush=True)

    producer.stop.set()
    torch.save({"state": model.state_dict(), "cfg": model.cfg, "meta": meta, "args": vars(args), "log": log},
               out_dir / "model.pt")

    # Preview strip: prediction vs denoised vs raw for a few validation lights.
    from PIL import Image
    rows = []
    for k in range(4):
        imgs = [predict(val_lights[k]), val_den[k], val_raw[k]]
        rows.append(np.concatenate([(tonemap(x).view(H, W, 3).cpu().numpy() ** (1 / 2.2) * 255).astype(np.uint8)
                                    for x in imgs], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(out_dir / "preview.png")
    (out_dir / "log.json").write_text(json.dumps(log, indent=1))
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
