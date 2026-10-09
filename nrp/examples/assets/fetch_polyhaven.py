"""Downloads the Poly Haven assets the example scenes use (all CC0, https://polyhaven.com/license).

    python examples/assets/fetch_polyhaven.py [dest]

Models come as USD with their textures, textures as JPGs (diffuse, OpenGL normal map, roughness),
at 2k, into `dest` (default $RELIGHT_WORK/assets/polyhaven, else ~/relight-work/assets/polyhaven),
one folder per asset. They are 5-30 MB each, so they live next to the path dumps, not in the
repo; the generators call fetch() and only download what is missing.
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

MODELS = ["boulder_01", "namaqualand_boulder_03", "namaqualand_boulder_05", "rock_09"]
TEXTURES = ["rock_face_03", "coast_sand_01"]
RES = "2k"
API = "https://api.polyhaven.com/files/"


def default_dest():
    return Path(os.environ.get("RELIGHT_WORK", Path.home() / "relight-work")) / "assets" / "polyhaven"


def _get(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {path.name}")
        req = urllib.request.Request(url, headers={"User-Agent": "relight-web example scenes"})
        with urllib.request.urlopen(req) as r:
            path.write_bytes(r.read())
    return path


def _files(asset):
    req = urllib.request.Request(API + asset, headers={"User-Agent": "relight-web example scenes"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def fetch(dest=None):
    """Downloads what is missing; returns {asset: folder}."""
    dest = Path(dest) if dest else default_dest()
    out = {}
    for asset in MODELS:
        folder = dest / asset
        usd = folder / f"{asset}_{RES}.usd"
        if not usd.exists():
            entry = _files(asset)["usd"][RES]["usd"]
            for rel, f in entry.get("include", {}).items():
                _get(f["url"], folder / rel)
            _get(entry["url"], usd)
        out[asset] = usd
    for asset in TEXTURES:
        folder = dest / asset
        maps = {"diff": "Diffuse", "nor_gl": "nor_gl", "rough": "Rough"}
        paths = {k: folder / f"{asset}_{k}_{RES}.jpg" for k in maps}
        if not all(p.exists() for p in paths.values()):
            files = _files(asset)
            for k, key in maps.items():
                _get(files[key][RES]["jpg"]["url"], paths[k])
        out[asset] = paths
    (dest / "LICENSE.txt").write_text(
        "Assets from Poly Haven (https://polyhaven.com), released under CC0 1.0 "
        "(https://polyhaven.com/license).\n" + "\n".join(f"  {a}: https://polyhaven.com/a/{a}" for a in MODELS + TEXTURES) + "\n")
    return out


if __name__ == "__main__":
    for k, v in fetch(sys.argv[1] if len(sys.argv) > 1 else None).items():
        print(k, v)
