"""Per-pixel buffers of an export, packed so they download small (pixel format 2).

The viewer needs, per pixel, the network's aux features (f16: albedo, normal, depth) and the
surface position. Format 1 stored those as f16 aux, f32 position + camera distance, and f16
normals again, which compresses poorly. Format 2 stores:

  aux  the aux features' f16 bits, unchanged
  pos  each position as 3 x u16 over `range` (steps of (hi - lo) / 65534), 0 where the pixel sees
       nothing; the camera distance follows from it, the normals from aux

each channel as a plane of its own, each value as its difference from the one to its left, and the
low bytes of a plane before its high bytes. Surfaces are smooth, so the differences are small and
their high bytes mostly zero, which gzip (nginx serves the files gzipped) compresses several times
better than the values. engine-base.js unpacks it. Ported from the portfolio's relight-pack.mjs.

    python pixels.py ../client/public/scenes/cornell     # repack an existing export in place
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

FORMAT = 2


def _planes(values):
    """values: [H, W, C] uint16 -> bytes: per channel, row differences, low bytes then high bytes."""
    v = values.astype(np.int32)
    d = (np.diff(v, axis=1, prepend=0) & 0xFFFF).astype(np.uint16)
    out = []
    for k in range(d.shape[2]):
        c = d[..., k].reshape(-1)
        out += [(c & 0xFF).astype(np.uint8).tobytes(), (c >> 8).astype(np.uint8).tobytes()]
    return b"".join(out)


def _unplanes(buf, H, W, C):
    """Inverse of _planes -> [H, W, C] uint16."""
    NP = H * W
    a = np.frombuffer(buf, np.uint8).reshape(C, 2, NP).astype(np.uint32)
    d = (a[:, 0] | (a[:, 1] << 8)).reshape(C, H, W)
    return (np.cumsum(d, axis=2) & 0xFFFF).astype(np.uint16).transpose(1, 2, 0)


def pack(aux, pos, valid):
    """aux: [H, W, C] float16 network features; pos: [H, W, 3] float32 surface positions;
    valid: [H, W] bool (the pixel sees a surface). Returns (bytes, the scene.json "pixels" entry)."""
    H, W, C = aux.shape
    p = pos[valid]
    lo, hi = (float(p.min()) - 1e-3, float(p.max()) + 1e-3) if len(p) else (-1.0, 1.0)
    q = np.zeros((H, W, 3), np.uint16)
    q[valid] = 1 + np.rint((np.clip(pos[valid], lo, hi) - lo) / (hi - lo) * 65534).astype(np.uint16)
    parts, entry, size = [], {"format": FORMAT}, 0
    for name, data, extra in [("aux", _planes(aux.view(np.uint16)), {"channels": C}),
                              ("pos", _planes(q), {"channels": 3, "range": [lo, hi]})]:
        pad = -size % 16
        parts.append(b"\0" * pad)
        size += pad
        entry[name] = {"offset": size, "bytes": len(data), **extra}
        parts.append(data)
        size += len(data)
    return b"".join(parts), entry


def unpack(buf, entry, H, W):
    """-> (aux [H, W, C] float16, pos [H, W, 3] float32, valid [H, W] bool); for checking."""
    a, p = entry["aux"], entry["pos"]
    aux = _unplanes(buf[a["offset"]:a["offset"] + a["bytes"]], H, W, a["channels"]).view(np.float16)
    q = _unplanes(buf[p["offset"]:p["offset"] + p["bytes"]], H, W, 3)
    lo, hi = p["range"]
    valid = q[..., 0] > 0
    pos = np.where(valid[..., None], lo + (q.astype(np.float32) - 1) * ((hi - lo) / 65534), 0).astype(np.float32)
    return aux, pos, valid


def repack(folder):
    """Converts every scene*.json / pixels*.bin of an export folder still in format 1."""
    folder = Path(folder)
    for sj in sorted(folder.glob("scene*.json")):
        scene = json.loads(sj.read_text(encoding="utf-8"))
        if scene["pixels"].get("format") == FORMAT:
            print(f"{sj.name}: already packed")
            continue
        suffix = sj.stem[len("scene"):]
        pb = folder / f"pixels{suffix}.bin"
        raw = pb.read_bytes()
        H, W = scene["height"], scene["width"]
        ae, ge = scene["pixels"]["aux"], scene["pixels"]["geom"]
        assert ae["dtype"] == "float16" and ge["dtype"] == "float32", "unexpected pixel layout"
        aux = np.frombuffer(raw, np.float16, int(np.prod(ae["shape"])), ae["offset"]).reshape(H, W, -1)
        geom = np.frombuffer(raw, np.float32, H * W * 4, ge["offset"]).reshape(H, W, 4)
        valid = geom[..., 3] > 0
        data, entry = pack(aux, geom[..., :3], valid)
        a2, p2, v2 = unpack(data, entry, H, W)
        assert (a2.view(np.uint16) == aux.view(np.uint16)).all() and (v2 == valid).all()
        err = float(np.abs(p2 - geom[..., :3])[valid].max()) if valid.any() else 0.0
        pb.write_bytes(data)
        scene["pixels"] = entry
        sj.write_text(json.dumps(scene, indent=1))
        gz = lambda b: len(gzip.compress(b, 6))
        print(f"{sj.name}: {W}x{H}, {len(raw) / 1e6:.1f} MB -> {len(data) / 1e6:.1f} MB, "
              f"gzipped {gz(raw) / 1e6:.2f} -> {gz(data) / 1e6:.2f} MB, worst position error {err:.1e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", help="export folders (web scenes) to repack in place")
    for f in ap.parse_args().folders:
        repack(f)
