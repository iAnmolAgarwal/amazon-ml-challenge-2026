import re
import unicodedata
from pathlib import Path

import numpy as np
import polars as pl

from common import load_config

COLS = ["entity_id", "business_name", "business_address", "country"]


def raw_dir(cfg):
    base = Path(cfg["data_dir"])
    hits = sorted(base.rglob("train_source1.tsv"), key=lambda p: len(p.parts))
    if not hits:
        raise SystemExit(f"train_source1.tsv not found under {base}; unzip the dataset into data/")
    return hits[0].parent.parent


def validator_path(cfg):
    hits = sorted(Path(cfg["data_dir"]).rglob("validate_submission.py"), key=lambda p: len(p.parts))
    return hits[0] if hits else None


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
