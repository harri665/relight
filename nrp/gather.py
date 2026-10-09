"""GATHERLIGHT (paper Sec. 3.1 / 4.2): relight stored path data with virtual sphere lights.

A single fused Triton kernel streams every path segment once per light, tests it
against the sphere, accumulates T_j on a hit and reduces the S samples of each
pixel in-register. Virtual lights do not occlude: the path continues unaltered.

Segments through a participating medium (water) carry negated throughputs (sample_paths.py):
their free flights were sampled with a grey extinction, so a light hit at distance t along
such a segment is weighted by exp(-delta * t) per channel, delta = sigma_t - mean(sigma_t).
"""
import threading

import numpy as np
import torch
import triton
import triton.language as tl

from common import load_meta, scene_dir


def medium_delta(medium):
    """Per-channel extinction minus the grey coefficient paths sample their free flights with
    (normalised units, as in meta.json's "medium"); zeros without a medium."""
    if not medium:
        return (0.0, 0.0, 0.0)
    st = np.asarray(medium["sigma_t"], float)
    return tuple(float(x) for x in st - st.mean())


@triton.jit
def _gather_sphere_kernel(V, T, lights, out, N, NP, dr_, dg_, db_,
                          S: tl.constexpr, D: tl.constexpr, J0: tl.constexpr, BLOCK: tl.constexpr,
                          MEDIUM: tl.constexpr):
    pid = tl.program_id(0)
    lid = tl.program_id(1)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N

    cx = tl.load(lights + lid * 4 + 0)
    cy = tl.load(lights + lid * 4 + 1)
    cz = tl.load(lights + lid * 4 + 2)
    r = tl.load(lights + lid * 4 + 3)
    r2 = r * r

    ox = tl.load(V + 0 * N + offs, mask=m, other=0.0).to(tl.float32) - cx
    oy = tl.load(V + 1 * N + offs, mask=m, other=0.0).to(tl.float32) - cy
    oz = tl.load(V + 2 * N + offs, mask=m, other=0.0).to(tl.float32) - cz
    acc_r = tl.zeros([BLOCK], tl.float32)
    acc_g = tl.zeros([BLOCK], tl.float32)
    acc_b = tl.zeros([BLOCK], tl.float32)
    for j in tl.static_range(D):
        ex = tl.load(V + ((j + 1) * 3 + 0) * N + offs, mask=m, other=0.0).to(tl.float32) - cx
        ey = tl.load(V + ((j + 1) * 3 + 1) * N + offs, mask=m, other=0.0).to(tl.float32) - cy
        ez = tl.load(V + ((j + 1) * 3 + 2) * N + offs, mask=m, other=0.0).to(tl.float32) - cz
        dx = ex - ox
        dy = ey - oy
        dz = ez - oz
        a = dx * dx + dy * dy + dz * dz
        b = ox * dx + oy * dy + oz * dz
        c = ox * ox + oy * oy + oz * oz - r2
        disc = b * b - a * c
        # Entering hit with t in (0, 1]: origin outside, moving closer, first root before the end.
        root = -b - tl.sqrt(tl.maximum(disc, 0.0))
        hit = (c > 0) & (b < 0) & (disc >= 0) & (root <= a)
        if j >= J0:
            tr = tl.load(T + (j * 3 + 0) * N + offs, mask=m, other=0.0).to(tl.float32)
            tg = tl.load(T + (j * 3 + 1) * N + offs, mask=m, other=0.0).to(tl.float32)
            tb = tl.load(T + (j * 3 + 2) * N + offs, mask=m, other=0.0).to(tl.float32)
            if MEDIUM:
                # In the medium (negated): magnitude times the extinction left over from the grey
                # free-flight sampling, over the distance to where the sphere is entered.
                wet = tr < 0
                dist = tl.where(wet, root / tl.sqrt(tl.maximum(a, 1e-20)), 0.0)
                tr = tl.abs(tr) * tl.exp(-dr_ * dist)
                tg = tl.abs(tg) * tl.exp(-dg_ * dist)
                tb = tl.abs(tb) * tl.exp(-db_ * dist)
            acc_r += tl.where(hit, tr, 0.0)
            acc_g += tl.where(hit, tg, 0.0)
            acc_b += tl.where(hit, tb, 0.0)
        ox = ex
        oy = ey
        oz = ez

    PB: tl.constexpr = BLOCK // S
    pix = pid.to(tl.int64) * PB + tl.arange(0, PB)
    pm = pix < NP
    base = out + lid.to(tl.int64) * NP * 3 + pix * 3
    tl.store(base + 0, tl.sum(tl.reshape(acc_r, (PB, S)), axis=1) / S, mask=pm)
    tl.store(base + 1, tl.sum(tl.reshape(acc_g, (PB, S)), axis=1) / S, mask=pm)
    tl.store(base + 2, tl.sum(tl.reshape(acc_b, (PB, S)), axis=1) / S, mask=pm)


def gather_arrays(V, T, NP, S, lights, first_seg=0, delta=(0.0, 0.0, 0.0)):
    """GATHERLIGHT on raw arrays. V: [D+1, 3, NP*S] fp16, T: [D, 3, NP*S] fp16 with
    path index = pixel * S + sample. `delta`: see medium_delta(). Returns [L, NP, 3] per unit
    radiance."""
    lights = torch.as_tensor(lights, dtype=torch.float32, device=V.device).reshape(-1, 4).contiguous()
    L, D, N = lights.shape[0], T.shape[0], NP * S
    out = torch.empty(L, NP, 3, dtype=torch.float32, device=V.device)
    BLOCK = max(1024, S)
    _gather_sphere_kernel[(triton.cdiv(N, BLOCK), L)](V, T, lights, out, N, NP, *map(float, delta),
                                                      S=S, D=D, J0=first_seg, BLOCK=BLOCK,
                                                      MEDIUM=any(x != 0 for x in delta))
    return out


class PathData:
    """Path dump of one scene resident on the GPU."""

    def __init__(self, scene, device="cuda"):
        self.meta = load_meta(scene)
        d = scene_dir(scene)
        self.W, self.H = self.meta["width"], self.meta["height"]
        self.S, self.D = self.meta["spp"], self.meta["max_seg"]
        self.NP = self.W * self.H
        self.N = self.NP * self.S
        self.V = self._upload(d / "verts.npy", device)
        self.T = self._upload(d / "thr.npy", device)
        self.aux = torch.from_numpy(np.load(d / "aux.npy")).to(device)
        self.delta = medium_delta(self.meta.get("medium"))
        self.lock = threading.Lock()  # held while the dump is read or re-traced (train.py --refresh)

    def _upload(self, path, device):
        arr = np.load(path, mmap_mode="r")
        flat = arr.reshape(arr.shape[0], 3, -1)
        out = torch.empty(flat.shape, dtype=torch.float16, device=device)
        for i in range(flat.shape[0]):  # stream to avoid a full host copy
            out[i] = torch.from_numpy(np.array(flat[i])).to(device)
        return out

    @torch.no_grad()
    def gather(self, lights, first_seg=0):
        """lights: [L, 4] (cx, cy, cz, radius). Returns [L, H, W, 3] per unit radiance.
        first_seg=1 drops segment 0 (light seen directly by the camera), which the
        viewer evaluates analytically (see direct.py)."""
        with self.lock:
            out = gather_arrays(self.V, self.T, self.NP, self.S, lights, first_seg, self.delta)
            torch.cuda.current_stream().synchronize()
        return out.view(-1, self.H, self.W, 3)

    @torch.no_grad()
    def replace_samples(self, s0, V, T):
        """Overwrites samples s0 .. s0+k of every pixel with freshly traced paths
        (V: [D+1, 3, NP*k], T: [D, 3, NP*k], path index = pixel * k + sample)."""
        k = V.shape[-1] // self.NP
        with self.lock:
            self.V.view(self.D + 1, 3, self.NP, self.S)[..., s0:s0 + k] = V.view(self.D + 1, 3, self.NP, k)
            self.T.view(self.D, 3, self.NP, self.S)[..., s0:s0 + k] = T.view(self.D, 3, self.NP, k)
            torch.cuda.synchronize()

    @torch.no_grad()
    def sample_segment_points(self, n, lo, hi, gen=None):
        """Light positions drawn uniformly on random path segments (paper Sec. 4.4),
        restricted to the light domain [lo, hi]. Implicitly importance-samples
        locations that are well covered by camera paths."""
        with self.lock:
            return self._sample_segment_points(n, lo, hi, gen)

    def _sample_segment_points(self, n, lo, hi, gen):
        dev = self.V.device
        lo = torch.as_tensor(lo, device=dev)
        hi = torch.as_tensor(hi, device=dev)
        got = []
        while sum(g.shape[0] for g in got) < n:
            m = 4 * n
            idx = torch.randint(0, self.N, (m,), device=dev, generator=gen)
            j = torch.randint(0, self.D, (m,), device=dev, generator=gen)
            a = self.V[j, :, idx].float()
            b = self.V[j + 1, :, idx].float()
            w = self.T[j, :, idx].float().abs().amax(-1)  # negated in a medium
            t = torch.rand(m, 1, device=dev, generator=gen)
            p = a + t * (b - a)
            ok = (w > 0) & ((p >= lo) & (p <= hi)).all(-1)
            got.append(p[ok])
        return torch.cat(got)[:n]
