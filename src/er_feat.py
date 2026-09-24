"""Pair features for candidate (S1, pool) pairs.

Reads cache/cand/{split}_cand{tag}.parquet and cache/norm/{split}_{s1,pool}.parquet, adds graph
features (ranks and gaps from both sides, candidate counts) over the whole candidate set, then
string/number features in batches with rapidfuzz, and writes cache/feat/{split}_feat{tag}.parquet.
No ids, row positions or country labels are used as features.
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from common import load_config

NONLATIN = re.compile(r"[ऀ-෿]")
COLS = ["name_core", "name_join", "initials", "name_alt", "addr_norm", "house", "street1", "postal", "nums", "is_web"]


def graph_features(c):
    """Rank/gap features computed on the full candidate graph, from the S1 side and the pool side."""
    return c.with_columns(
        n_cand=pl.len().over("s1_idx").cast(pl.Float32),
        n_rival=pl.len().over("pool_idx").cast(pl.Float32),
        word_rank=pl.col("word_sim").rank("min", descending=True).over("s1_idx").cast(pl.Float32),
        word_gap=(pl.col("word_sim").max().over("s1_idx") - pl.col("word_sim")).cast(pl.Float32),
        rev_rank=pl.col("word_sim").rank("min", descending=True).over("pool_idx").cast(pl.Float32),
        rev_gap=(pl.col("word_sim").max().over("pool_idx") - pl.col("word_sim")).cast(pl.Float32),
        n_keys=pl.sum_horizontal(pl.col("^k_.*$")).cast(pl.Float32),
    )


def score(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)


def string_features(p):
    """Name, address and number features for a batch of joined pairs."""
    n1, n2 = p["name_core"].to_list(), p["name_core_p"].to_list()
    j1, j2 = p["name_join"].to_list(), p["name_join_p"].to_list()
    a1, a2 = p["addr_norm"].to_list(), p["addr_norm_p"].to_list()
    alt = p["name_alt_p"].to_list()
    f = {
        "n_ratio": score(n1, n2, fuzz.ratio),
        "n_tset": score(n1, n2, fuzz.token_set_ratio),
        "n_tsort": score(n1, n2, fuzz.token_sort_ratio),
        "n_partial": score(n1, n2, fuzz.partial_ratio),
        "n_jw": score(n1, n2, JaroWinkler.normalized_similarity),
        "j_ratio": score(j1, j2, fuzz.ratio),
        "j_partial": score(j1, j2, fuzz.partial_ratio),
        "alt_ratio": np.where(np.array([bool(x) for x in alt]), score(n1, alt, fuzz.token_set_ratio), -1).astype(np.float32),
        "a_tset": score(a1, a2, fuzz.token_set_ratio),
        "a_ratio": score(a1, a2, fuzz.ratio),
        "a_partial": score(a1, a2, fuzz.partial_ratio),
    }
    h1, h2 = p["house"].to_list(), p["house_p"].to_list()
    s1s, s2s = p["street1"].to_list(), p["street1_p"].to_list()
    z1, z2 = p["postal"].to_list(), p["postal_p"].to_list()
    u1, u2 = p["nums"].to_list(), p["nums_p"].to_list()
    house_eq, house_pre, street_eq, postal_eq, postal_bad, num_j, nl_n, nl_a = ([] for _ in range(8))
    for x1, x2, y1, y2, q1, q2, m1, m2, nm2, ad2 in zip(h1, h2, s1s, s2s, z1, z2, u1, u2, n2, a2):
        both = bool(x1) and bool(x2)
        house_eq.append(float(both and x1 == x2) if both else -1.0)
        house_pre.append(float(both and x1 != x2 and (x1.startswith(x2) or x2.startswith(x1))))
        street_eq.append(float(y1 == y2) if y1 and y2 else -1.0)
        postal_eq.append(float(q1 == q2) if q1 and q2 else -1.0)
        postal_bad.append(float(bool(q1) and bool(q2) and q1 != q2))
        su, sv = set(m1.split()), set(m2.split())
        num_j.append(len(su & sv) / len(su | sv) if su | sv else -1.0)
        nl_n.append(float(bool(NONLATIN.search(nm2))))
        nl_a.append(float(bool(NONLATIN.search(ad2))))
    f.update({"house_eq": house_eq, "house_prefix": house_pre, "street_eq": street_eq, "postal_eq": postal_eq, "postal_conflict": postal_bad, "num_jacc": num_j, "nonlatin_name_p": nl_n, "nonlatin_addr_p": nl_a})
    return pl.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})


def build(split, tag, batch=4_000_000):
    """Compute and cache all features for one split."""
    cfg = load_config()
    cache = Path(cfg["cache_dir"])
    cand = pl.read_parquet(cache / "cand" / f"{split}_cand{tag}.parquet")
    s1 = pl.read_parquet(cache / "norm" / f"{split}_s1.parquet", columns=COLS).with_row_index("s1_idx")
    pool = pl.read_parquet(cache / "norm" / f"{split}_pool.parquet", columns=COLS + ["entity_id"]).with_row_index("pool_idx")
    pool = pool.with_columns(is_s3=(pl.col("entity_id").str.slice(0, 3) == "S3-").cast(pl.Float32)).drop("entity_id")
    t = time.time()
    cand = graph_features(cand).sort("s1_idx")
    print(f"graph features for {len(cand):,} pairs in {time.time() - t:.0f}s", flush=True)
    s1 = s1.with_columns(pl.col("s1_idx").cast(pl.UInt32), a_empty=(pl.col("addr_norm") == "").cast(pl.Float32), n_tok=pl.col("name_core").str.count_matches(" ").cast(pl.Float32) + 1)
    pool = pool.with_columns(pl.col("pool_idx").cast(pl.UInt32), a_empty_p=(pl.col("addr_norm") == "").cast(pl.Float32), n_tok_p=pl.col("name_core").str.count_matches(" ").cast(pl.Float32) + 1, is_web_p=pl.col("is_web").cast(pl.Float32)).drop("is_web")
    pool = pool.rename({c: c + "_p" for c in COLS if c != "is_web"})
    out_dir = cache / "feat"
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for s in range(0, len(cand), batch):
        tb = time.time()
        c = cand.slice(s, batch)
        p = c.join(s1, on="s1_idx", how="left").join(pool, on="pool_idx", how="left")
        sf = string_features(p)
        base = p.select(["s1_idx", "pool_idx", "word_sim", "n_cand", "n_rival", "word_rank", "word_gap", "rev_rank", "rev_gap", "n_keys", "is_s3", "a_empty", "a_empty_p", "n_tok", "n_tok_p", "is_web_p"] + [x for x in c.columns if x.startswith("k_")])
        base = base.with_columns([pl.col(x).cast(pl.Float32) for x in base.columns if x.startswith("k_")])
        acr = (p["initials"] == p["name_join_p"]).cast(pl.Float32).alias("acronym")
        je = (p["name_join"] == p["name_join_p"]).cast(pl.Float32).alias("join_eq")
        parts.append(pl.concat([base, sf], how="horizontal").with_columns(acr, je))
        print(f"  batch {s // batch + 1}/{(len(cand) + batch - 1) // batch}: {len(c):,} pairs in {time.time() - tb:.0f}s", flush=True)
    feat = pl.concat(parts)
    feat.write_parquet(out_dir / f"{split}_feat{tag}.parquet")
    print(f"wrote {split}_feat{tag}.parquet {feat.shape} in {time.time() - t:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    build(a.split, a.tag)


if __name__ == "__main__":
    main()
