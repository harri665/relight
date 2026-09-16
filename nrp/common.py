import json
import os
from pathlib import Path

# Path dumps are multiple GB; keep them outside the (synced) project folder.
WORK_DIR = Path(os.environ.get("RELIGHT_WORK", Path.home() / "relight-work"))
PROJECT_DIR = Path(__file__).resolve().parent.parent
WEB_SCENES_DIR = PROJECT_DIR / "client" / "public" / "scenes"


def scene_dir(name):
    d = WORK_DIR / "cache" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_meta(name):
    return json.loads((scene_dir(name) / "meta.json").read_text())
