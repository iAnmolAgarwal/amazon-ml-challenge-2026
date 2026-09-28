"""Generic harness helpers (metrics, template loaders, CV comparison) for the research tooling; not used by the pipeline."""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import polars as pl

from er import raw_dir


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


def read_json(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0, missing_utf8_is_empty_string=True).with_columns(pl.all().fill_null(""))


def load_split(cfg, split):
    d = raw_dir(cfg) / split
    s = [read_tsv(d / f"{split}_source{i}.tsv") for i in (1, 2, 3)]
    pool = pl.concat([s[1], s[2]]).with_columns(pl.col("entity_id").str.slice(0, 2).alias("source"))
    return s[0], pool


def load_pool(cfg):
    return load_split(cfg, cfg.get("pool", "train"))[1]


def split_ids(s):
    return [x for x in (s or "").split(",") if x]


def f05(pred, gold):
    pred, gold = set(pred), set(gold)
    if not gold:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 1.25 * p * r / (0.25 * p + r)


def f05_macro(y_true, y_pred):
    return float(np.mean([f05(p, split_ids(t) if isinstance(t, str) else t) for t, p in zip(y_true, y_pred)]))


def candidate_stats(y_true, cands):
    gold = [set(split_ids(t)) for t in y_true]
    n_gold = sum(len(g) for g in gold)
    hit = sum(len(g & set(c)) for g, c in zip(gold, cands))
    oracle = f05_macro(y_true, [list(g & set(c)) for g, c in zip(gold, cands)])
    return {"pair_recall": hit / max(n_gold, 1), "oracle_f05": oracle, "avg_candidates": float(np.mean([len(c) for c in cands]))}


ABBR = {
    "&": "and", "corp": "corporation", "co": "company", "inc": "incorporated", "ltd": "limited", "pvt": "private",
    "llc": "llc", "intl": "international", "mfg": "manufacturing", "svcs": "services", "svc": "service", "bros": "brothers",
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard", "bd": "boulevard", "dr": "drive",
    "ln": "lane", "hwy": "highway", "pkwy": "parkway", "ct": "court", "pl": "place", "sq": "square", "ste": "suite",
    "fl": "floor", "apt": "apartment", "bldg": "building", "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest", "mt": "mount", "ft": "fort",
    "nr": "near", "opp": "opposite", "mg": "mahatma gandhi", "marg": "marg", "ngr": "nagar", "clny": "colony",
    "sect": "sector", "sec": "sector", "rte": "route", "chem": "chemin", "imp": "impasse", "fbg": "faubourg",
    "no": "number", "num": "number",
}
LEGAL = {
    "corporation", "company", "incorporated", "limited", "private", "llc", "llp", "lp", "plc", "pllc", "the",
    "sarl", "sas", "sa", "eurl", "sasu", "sci", "snc", "gmbh", "ag", "bv", "nv", "pte", "pty", "opc",
}


def norm_text(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("&", " and ").replace("'", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(ABBR.get(t, t) for t in s.split())


def core_name(s):
    return " ".join(t for t in norm_text(s).split() if t not in LEGAL and t != "and") or norm_text(s)


def add_norm(df):
    return df.with_columns(
        pl.col("business_name").map_elements(norm_text, return_dtype=pl.Utf8).alias("name_n"),
        pl.col("business_name").map_elements(core_name, return_dtype=pl.Utf8).alias("name_c"),
        pl.col("business_address").map_elements(norm_text, return_dtype=pl.Utf8).alias("addr_n"),
        pl.col("country").str.strip_chars().str.to_lowercase().alias("country_n"),
    )
