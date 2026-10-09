"""Analytic direct-view term (segment 0 of every path).

The camera sees a virtual sphere light wherever the primary ray hits the sphere
before the first surface. This term is exact and trivially cheap, but a sharp
disc whose edge moves with the light parameters is hard for an MLP to learn, so
the network is trained on segments >= 1 and this term is added at render time
(the WebGPU viewer implements the same function in WGSL).
"""
import math

import torch


def camera_rays(meta, uv):
    """uv: [..., 2] film coordinates in [0, 1] (y down). Returns (origin [3], dirs [..., 3])."""
    M = torch.tensor(meta["camera"]["to_world"], dtype=torch.float32, device=uv.device)
    t = math.tan(math.radians(meta["camera"]["fov"]) / 2)
    ty = t * meta["height"] / meta["width"]
    d = torch.stack([(0.5 - uv[..., 0]) * 2 * t, (0.5 - uv[..., 1]) * 2 * ty, torch.ones_like(uv[..., 0])], -1)
    d = d @ M[:3, :3].T
    return M[:3, 3], d / d.norm(dim=-1, keepdim=True)


def direct_view(meta, aux, light, ss=4):
    """Coverage of the light disc per pixel, [H*W, 1], averaged over ss x ss subpixels.
    aux: [H, W, 10]; pixels without geometry (distance 0) count as unoccluded. With the camera
    in a medium (meta["medium"]["camera"] == "water") the disc is dimmed by exp(-sigma_t * t)
    per channel, and the result is [H*W, 3]."""
    H, W = meta["height"], meta["width"]
    dev = aux.device
    ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    o = (torch.arange(ss, device=dev) + 0.5) / ss
    sy, sx = torch.meshgrid(o, o, indexing="ij")
    uv = torch.stack([(xs[..., None, None] + sx) / W, (ys[..., None, None] + sy) / H], -1)  # H W ss ss 2
    org, d = camera_rays(meta, uv)
    c, r = light[:3].to(dev), light[3].item()
    oc = org - c
    b = (d * oc).sum(-1)
    disc = b * b - (oc * oc).sum() + r * r
    t0 = -b - disc.clamp_min(0).sqrt()
    dist = torch.where(aux[..., 9] > 0, (aux[..., 6:9] - org).norm(dim=-1), torch.zeros_like(aux[..., 9]))
    surf = torch.where(dist > 0, dist, torch.full_like(dist, 1e9))[..., None, None]
    hit = (disc >= 0) & (t0 > 0) & (t0 < surf)
    medium = meta.get("medium")
    if medium and medium.get("camera") == "water":
        st = torch.tensor(medium["sigma_t"], dtype=torch.float32, device=dev)
        att = torch.exp(-st * t0.clamp_min(0)[..., None]) * hit[..., None]  # H W ss ss 3
        return att.mean((2, 3)).reshape(-1, 3)
    return hit.float().mean((-1, -2)).reshape(-1, 1)
