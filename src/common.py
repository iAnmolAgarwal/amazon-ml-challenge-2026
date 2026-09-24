import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    p = Path(path or os.environ.get("AML_CONFIG", ROOT / "config.yaml"))
    cfg = yaml.safe_load(p.read_text())
    for k in ("data_dir", "cache_dir"):
        cfg[k] = str((p.parent / cfg[k]).resolve()) if not Path(cfg[k]).is_absolute() else cfg[k]
    Path(cfg["cache_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


def get_device(cfg=None):
    import torch

    want = (cfg or {}).get("device", "mps")
    if want == "cuda" and torch.cuda.is_available():
        return "cuda"
    if want in ("mps", "cuda") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_table(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path, infer_schema_length=10000)


def load_train(cfg):
    return read_table(Path(cfg["data_dir"]) / cfg["train_file"])


def load_test(cfg):
    return read_table(Path(cfg["data_dir"]) / cfg["test_file"])


def load_folds(cfg):
    return pl.read_csv(Path(cfg["data_dir"]) / "folds.csv")


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
        return "nogit"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def _as_sets(y):
    return [set(v) if isinstance(v, (list, tuple, set, np.ndarray)) else {v} for v in y]


def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    d = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(np.where(d == 0, 0.0, 2 * np.abs(y_true - y_pred) / np.where(d == 0, 1, d))) * 100)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def f1_macro(y_true, y_pred):
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro"))


def f1_micro(y_true, y_pred):
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="micro"))


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def map_at_k(y_true, y_pred, k=10):
    truth = _as_sets(y_true)
    scores = []
    for t, ranked in zip(truth, y_pred):
        hits, s = 0, 0.0
        for i, p in enumerate(list(ranked)[:k]):
            if p in t:
                hits += 1
                s += hits / (i + 1)
        scores.append(s / min(len(t), k) if t else 0.0)
    return float(np.mean(scores))


def recall_at_k(y_true, y_pred, k=10):
    truth = _as_sets(y_true)
    return float(np.mean([len(t & set(list(r)[:k])) / len(t) if t else 0.0 for t, r in zip(truth, y_pred)]))


HIGHER_IS_BETTER = {"smape": False, "mae": False, "rmse": False, "f1_macro": True, "f1_micro": True, "accuracy": True, "map@k": True, "recall@k": True}


def get_metric(name):
    base, _, k = name.partition("@")
    if base in ("map", "recall"):
        kk = int(k) if k and k != "k" else 10
        fn = map_at_k if base == "map" else recall_at_k
        return (lambda yt, yp: fn(yt, yp, kk)), True
    fns = {"smape": smape, "mae": mae, "rmse": rmse, "f1_macro": f1_macro, "f1_micro": f1_micro, "accuracy": accuracy}
    return fns[base], HIGHER_IS_BETTER[base]
