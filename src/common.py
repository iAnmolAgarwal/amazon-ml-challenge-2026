import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    p = Path(path or os.environ.get("AML_CONFIG", ROOT / "config.yaml"))
    cfg = yaml.safe_load(p.read_text())
    base = ROOT if path is None else p.parent
    for k in ("data_dir", "cache_dir"):
        cfg[k] = str((base / cfg[k]).resolve()) if not Path(cfg[k]).is_absolute() else cfg[k]
    Path(cfg["cache_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


def load_attempt(attempt_id):
    p = ROOT / "attempts" / attempt_id / "train.py"
    spec = importlib.util.spec_from_file_location(f"attempt_{attempt_id}", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get("AML_GIT_COMMIT", "nogit")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
