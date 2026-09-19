"""Neural render proxy N_sphere(px, F_px, light) (paper Sec. 3.2 / 4.3).

Inputs: pixel coordinates (multi-resolution grid encoding), auxiliary features
(albedo 3, normal 3, depth 1, optionally world position 3) and sphere light
parameters (center 3, radius 1), all normalized to roughly [-1, 1].
Output: RGB transport per unit radiance.

Optional extensions beyond the paper (see README):
  geo       per (pixel, light) geometric features: direction to the light, cosine,
            log distance, log solid angle of the sphere.
  head=mul  output = a * G + b with G = Omega * max(cos, 0) / pi, the unshadowed
            irradiance factor; the network learns visibility/albedo (a) and the rest (b).

The grid is the 2D multi-resolution hash encoding of Instant-NGP. With 2D
inputs and tables >= res^2, every level is collision-free, i.e. a dense grid,
so we store it densely (this also maps directly to the WebGPU viewer).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

GEO_DIM = 6


class GridEncoding2D(nn.Module):
    def __init__(self, levels=16, feats=2, base_res=16, max_res=512):
        super().__init__()
        b = math.exp((math.log(max_res) - math.log(base_res)) / (levels - 1))
        self.res = [int(math.floor(base_res * b ** l)) for l in range(levels)]
        self.res[-1] = max_res
        self.feats = feats
        self.grids = nn.ParameterList(
            [nn.Parameter(torch.empty(1, feats, r, r).uniform_(-1e-4, 1e-4)) for r in self.res])

    @property
    def out_dim(self):
        return len(self.res) * self.feats

    def forward(self, uv):
        """uv: [B, 2] in [0, 1] (x right, y down)."""
        g = (uv * 2 - 1).view(1, 1, -1, 2)
        outs = [F.grid_sample(grid, g, mode="bilinear", align_corners=True).view(self.feats, -1)
                for grid in self.grids]
        return torch.cat(outs, 0).t()


class GridEncoding3D(nn.Module):
    """Dense multi-resolution grid over the (normalised) light position. Evaluated once per
    light, so it costs nothing per pixel at render time."""

    def __init__(self, levels=6, feats=4, base_res=4, max_res=32):
        super().__init__()
        b = math.exp((math.log(max_res) - math.log(base_res)) / max(levels - 1, 1))
        self.res = [int(round(base_res * b ** l)) for l in range(levels)]
        self.feats = feats
        self.grids = nn.ParameterList(
            [nn.Parameter(torch.empty(1, feats, r, r, r).uniform_(-1e-4, 1e-4)) for r in self.res])

    @property
    def out_dim(self):
        return len(self.res) * self.feats

    def forward(self, x):
        """x: [B, 3] in [-1, 1] (x, y, z)."""
        g = x.view(1, 1, 1, -1, 3)
        return torch.cat([F.grid_sample(grid, g, mode="bilinear", align_corners=True).view(self.feats, -1)
                          for grid in self.grids], 0).t()


class NRP(nn.Module):
    LIGHT_DIM = 4

    def __init__(self, width=128, hidden=4, levels=16, feats=2, base_res=16, max_res=512,
                 aux_dim=7, geo=False, head="linear", light_grid=0):
        super().__init__()
        self.cfg = dict(width=width, hidden=hidden, levels=levels, feats=feats, base_res=base_res,
                        max_res=max_res, aux_dim=aux_dim, geo=geo, head=head, light_grid=light_grid)
        self.geo, self.head = geo, head
        self.enc = GridEncoding2D(levels, feats, base_res, max_res)
        self.lenc = GridEncoding3D(levels=light_grid) if light_grid else None
        in_dim = (self.enc.out_dim + aux_dim + self.LIGHT_DIM + (GEO_DIM if geo else 0)
                  + (self.lenc.out_dim if self.lenc else 0))
        out_dim = 6 if head == "mul" else 3
        dims = [in_dim] + [width] * hidden + [out_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, uv, aux, light, geo=None, G=None):
        # Column order of the first layer: [pixel grid, aux, light, geo, light grid].
        parts = [self.enc(uv), aux, light] + ([geo] if self.geo else []) +             ([self.lenc(light[:, :3])] if self.lenc else [])
        h = torch.cat(parts, -1)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
        if self.head == "mul":
            return h[:, :3] * G + h[:, 3:]
        return h


class Normalizer:
    """Maps raw light parameters / aux buffers to network inputs."""

    def __init__(self, meta):
        self.lo = torch.tensor(meta["light_bbox"][0])
        self.hi = torch.tensor(meta["light_bbox"][1])
        self.rmin, self.rmax = meta["radius_range"]

    def light(self, p):
        """p: [..., 4] (center, radius) -> [-1, 1]^4."""
        lo, hi = self.lo.to(p.device), self.hi.to(p.device)
        c = 2 * (p[..., :3] - lo) / (hi - lo) - 1
        r = 2 * (p[..., 3:4] - self.rmin) / (self.rmax - self.rmin) - 1
        return torch.cat([c, r], -1)

    def position(self, x):
        lo, hi = self.lo.to(x.device), self.hi.to(x.device)
        return 2 * (x - lo) / (hi - lo) - 1


class PixelBuffers:
    """Per-pixel network inputs derived from the aux render."""

    def __init__(self, aux, meta, aux_dim=7):
        H, W = aux.shape[:2]
        self.norm = Normalizer(meta)
        self.pos = aux[..., 6:9].reshape(-1, 3)
        self.nrm = aux[..., 3:6].reshape(-1, 3)
        self.valid = (aux[..., 9] > 0).reshape(-1, 1).float()
        albedo = aux[..., 0:3].clamp(0, 1).reshape(-1, 3)
        depth = (aux[..., 9:10] / meta["depth_scale"]).reshape(-1, 1)
        feats = [albedo, self.nrm, depth]
        if aux_dim == 10:
            feats.append(self.norm.position(self.pos) * self.valid)
        self.aux = torch.cat(feats, -1)
        ys, xs = torch.meshgrid(torch.arange(H, device=aux.device), torch.arange(W, device=aux.device), indexing="ij")
        self.uv = torch.stack([(xs + 0.5) / W, (ys + 0.5) / H], -1).reshape(-1, 2).float()


def geo_features(pos, nrm, valid, light):
    """Per (pixel, light) features. pos/nrm/valid: [B, 3/3/1], light: [B, 4] raw.
    Returns (features [B, 6], G [B, 1]) where G = Omega * max(cos, 0) / pi."""
    v = light[:, :3] - pos
    d = v.norm(dim=-1, keepdim=True).clamp_min(1e-4)
    l = v / d
    cos = (nrm * l).sum(-1, keepdim=True)
    s = (light[:, 3:4] / d).clamp(max=1.0)
    # Solid angle of the sphere, 2 pi (1 - sqrt(1 - s^2)), in cancellation-free form.
    omega = 2 * math.pi * s * s / (1 + torch.sqrt((1 - s * s).clamp_min(0)))
    G = omega * cos.clamp_min(0) / math.pi * valid
    feats = torch.cat([l, cos, 0.5 * torch.log(d), 0.25 * torch.log(omega + 1e-6)], -1) * valid
    return feats, G


def run_model(model, bufs, pix, light_raw):
    """Evaluate the network for pixel indices `pix` and raw lights [B, 4] (or [4])."""
    if light_raw.dim() == 1:
        light_raw = light_raw.view(1, 4).expand(pix.shape[0], 4)
    geo = G = None
    if model.geo or model.head == "mul":
        geo, G = geo_features(bufs.pos[pix], bufs.nrm[pix], bufs.valid[pix], light_raw)
    return model(bufs.uv[pix], bufs.aux[pix], bufs.norm.light(light_raw), geo, G)


@torch.no_grad()
def predict_image(model, bufs, light_raw, chunk=1 << 18):
    NP = bufs.uv.shape[0]
    idx = torch.arange(NP, device=bufs.uv.device)
    return torch.cat([run_model(model, bufs, idx[i:i + chunk], light_raw) for i in range(0, NP, chunk)]).clamp_min(0)
