"""Side-car pair features (feat2): label-free name frequencies, conservative-name similarity, gated address scores.

usage: python src/er_feat2.py --split train|test [--tag ""]

Adds, for every candidate pair in cache/cand/{split}_cand{tag}.parquet:
- nf_s1_s1 / nf_s1_pool: how many S1 / pool records in the same country share the S1's space-free name;
  nf_p_s1 / nf_p_pool: the same for the pool record's space-free name (log1p counts; label-free, per split).
- c_ratio / c_tset / c_jw: rapidfuzz scores on the conservatively normalized raw names
  (NFKC, lowercase, accents folded, punctuation -> space; no legal-word removal, no digit substitution).
- a_ratio_g / a_partial_g: address ratio / partial ratio, -1 when either address is empty.
Writes cache/feat/{split}_feat2{tag}.parquet keyed by (s1_idx, pool_idx). No ids or labels used as features.
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
from er_norm import fold

PUNCT = re.compile(r"[^\wऀ-෿]+")


def conservative(s):
    """Light normalization that keeps every word: fold case/accents, punctuation to spaces."""
    return " ".join(PUNCT.sub(" ", fold(s or "")).split())


def freq(df, key, col):
    return df.group_by(["country_n", key]).agg(pl.len().alias(col))


def build(split, tag, batch=5_000_000):
    """Compute and cache feat2 for one split."""
    cfg = load_config()
    cache = Path(cfg["cache_dir"])
    t = time.time()
    s1 = pl.read_parquet(cache / "norm" / f"{split}_s1.parquet", columns=["country_n", "name_join", "addr_norm"]).with_row_index("s1_idx")
    pool = pl.read_parquet(cache / "norm" / f"{split}_pool.parquet", columns=["country_n", "name_join", "addr_norm"]).with_row_index("pool_idx")
    f_s1 = freq(s1, "name_join", "n_s1")
    f_pool = freq(pool, "name_join", "n_pool")
    counts = f_s1.join(f_pool, on=["country_n", "name_join"], how="full", coalesce=True).fill_null(0)
    s1 = s1.join(counts, on=["country_n", "name_join"], how="left").select(
        pl.col("s1_idx").cast(pl.UInt32), "addr_norm", pl.col("n_s1").log1p().cast(pl.Float32).alias("nf_s1_s1"), pl.col("n_pool").log1p().cast(pl.Float32).alias("nf_s1_pool")
    ).sort("s1_idx")
    pool = pool.join(counts, on=["country_n", "name_join"], how="left").select(
        pl.col("pool_idx").cast(pl.UInt32), pl.col("addr_norm").alias("addr_p"), pl.col("n_s1").log1p().cast(pl.Float32).alias("nf_p_s1"), pl.col("n_pool").log1p().cast(pl.Float32).alias("nf_p_pool")
    ).sort("pool_idx")
    raw1 = pl.read_parquet(cache / "raw" / f"{split}_source1.parquet", columns=["business_name"])
    rawp = pl.concat([pl.read_parquet(cache / "raw" / f"{split}_source{i}.parquet", columns=["business_name"]) for i in (2, 3)])
    s1 = s1.with_columns(cname=pl.Series([conservative(x) for x in raw1["business_name"].to_list()]))
    pool = pool.with_columns(cname_p=pl.Series([conservative(x) for x in rawp["business_name"].to_list()]))
    del raw1, rawp
    print(f"lookups ready in {time.time() - t:.0f}s", flush=True)
    cand = pl.read_parquet(cache / "cand" / f"{split}_cand{tag}.parquet", columns=["s1_idx", "pool_idx"])
    out_dir = cache / "feat"
    parts = []
    for s in range(0, len(cand), batch):
        tb = time.time()
        p = cand.slice(s, batch).join(s1, on="s1_idx", how="left").join(pool, on="pool_idx", how="left")
        c1, c2 = p["cname"].to_list(), p["cname_p"].to_list()
        a1, a2 = p["addr_norm"].to_list(), p["addr_p"].to_list()
        empty = (p["addr_norm"] == "") | (p["addr_p"] == "")
        ar = cpdist(a1, a2, scorer=fuzz.ratio, workers=-1).astype(np.float32)
        ap = cpdist(a1, a2, scorer=fuzz.partial_ratio, workers=-1).astype(np.float32)
        out = p.select("s1_idx", "pool_idx", "nf_s1_s1", "nf_s1_pool", "nf_p_s1", "nf_p_pool").with_columns(
            c_ratio=pl.Series(cpdist(c1, c2, scorer=fuzz.ratio, workers=-1).astype(np.float32)),
            c_tset=pl.Series(cpdist(c1, c2, scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32)),
            c_jw=pl.Series(cpdist(c1, c2, scorer=JaroWinkler.normalized_similarity, workers=-1).astype(np.float32)),
            a_ratio_g=pl.Series(np.where(empty.to_numpy(), -1.0, ar).astype(np.float32)),
            a_partial_g=pl.Series(np.where(empty.to_numpy(), -1.0, ap).astype(np.float32)),
        )
        part = out_dir / f"{split}_feat2{tag}.part{s // batch:04d}.parquet"
        out.write_parquet(part)
        parts.append(part)
        print(f"  batch {s // batch + 1}/{(len(cand) + batch - 1) // batch} in {time.time() - tb:.0f}s", flush=True)
    final = out_dir / f"{split}_feat2{tag}.parquet"
    pl.scan_parquet(parts).sink_parquet(final)
    for part in parts:
        part.unlink()
    print(f"wrote {final.name} ({pl.scan_parquet(final).select(pl.len()).collect().item():,} rows) in {time.time() - t:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    build(a.split, a.tag)


if __name__ == "__main__":
    main()
