"""Generic harness helpers (metrics, template loaders, CV comparison) for the research tooling; not used by the pipeline."""
import json
from pathlib import Path

import numpy as np
import polars as pl


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


HIGHER_IS_BETTER = {"f05_er": True, "smape": False, "mae": False, "rmse": False, "f1_macro": True, "f1_micro": True, "accuracy": True, "map@k": True, "recall@k": True}


def get_metric(name):
    base, _, k = name.partition("@")
    if base in ("map", "recall"):
        kk = int(k) if k and k != "k" else 10
        fn = map_at_k if base == "map" else recall_at_k
        return (lambda yt, yp: fn(yt, yp, kk)), True
    if base == "f05_er":
        from er import f05_macro

        return f05_macro, True
    fns = {"smape": smape, "mae": mae, "rmse": rmse, "f1_macro": f1_macro, "f1_micro": f1_micro, "accuracy": accuracy}
    return fns[base], HIGHER_IS_BETTER[base]


def paired_gain(child, parent):
    a, b = child.get("fold_scores"), parent.get("fold_scores")
    if not a or not b or child.get("folds") != parent.get("folds") or len(a) < 2:
        return None
    sign = 1 if child.get("higher_is_better", True) else -1
    d = sign * (np.asarray(a) - np.asarray(b))
    se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
    wins = int((d > 0).sum())
    sig = bool(d.mean() > 2 * se and wins >= len(d) - 1 and d.mean() > 0)
    worse = bool(d.mean() < -2 * se and (d < 0).sum() >= len(d) - 1)
    return {"mean": float(d.mean()), "se": se, "wins": wins, "n": len(d), "significant": sig, "worse": worse}
