"""Side-car competition features (feat4) from stage-1 probabilities, for a second-stage model.

usage: python src/er_feat4.py --split train|test --src <stage-1 attempt> [--tag ""]

Inputs: train uses the stage-1 out-of-fold probabilities (cache/oof/<src>.parquet, every pair scored by a model
that never saw its S1); test uses the stage-1 test probabilities (cache/oof/test_<src>.parquet).
For each candidate pair (s, x) with stage-1 probability p:
- pool side (the S2/S3 record x against all S1 that retrieved it): rank of p among them, best probability of any
  other S1, margin p - best other, number of other S1 with p >= 0.1, sum of p over all S1
- S1 side (s against all its candidates): rank of p, gap to the best candidate, best other candidate, number of
  candidates with p >= 0.5, sum of p (expected number of matches)
- p itself.
Writes cache/feat/{split}_feat4{tag}.parquet keyed by (s1_idx, pool_idx).
"""
import argparse
import time
from pathlib import Path

import polars as pl

from common import load_config


def competition(pairs):
    p = pl.col("prob")
    pairs = pairs.with_columns(
        st1_p=p,
        cp_rank=p.rank("min", descending=True).over("pool_idx").cast(pl.Float32),
        cp_sum=p.sum().over("pool_idx"),
        cp_n=(p >= 0.1).sum().over("pool_idx").cast(pl.Float32),
        cs_rank=p.rank("min", descending=True).over("s1_idx").cast(pl.Float32),
        cs_sum=p.sum().over("s1_idx"),
        cs_n=(p >= 0.5).sum().over("s1_idx").cast(pl.Float32),
        cs_top=p.max().over("s1_idx"),
    )
    pairs = pairs.with_columns(
        cp_best_other=pl.when(pl.col("cp_rank") == 1).then(p.sort(descending=True).get(1, null_on_oob=True).over("pool_idx")).otherwise(p.max().over("pool_idx")).fill_null(0.0),
        cs_best_other=pl.when(pl.col("cs_rank") == 1).then(p.sort(descending=True).get(1, null_on_oob=True).over("s1_idx")).otherwise(pl.col("cs_top")).fill_null(0.0),
        cp_n=pl.col("cp_n") - (p >= 0.1).cast(pl.Float32),
    )
    return pairs.select(
        "s1_idx", "pool_idx", "st1_p", "cp_rank", "cp_best_other", (p - pl.col("cp_best_other")).alias("cp_margin"), "cp_n", "cp_sum",
        "cs_rank", (pl.col("cs_top") - p).alias("cs_gap"), "cs_best_other", "cs_n", "cs_sum",
    ).with_columns(pl.exclude("s1_idx", "pool_idx").cast(pl.Float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cache = Path(load_config()["cache_dir"])
    t = time.time()
    name = a.src if a.split == "train" else f"test_{a.src}"
    pairs = pl.read_parquet(cache / "oof" / f"{name}{a.tag}.parquet", columns=["s1_idx", "pool_idx", "prob"])
    out = competition(pairs)
    path = cache / "feat" / f"{a.split}_feat4{a.tag}.parquet"
    out.write_parquet(path)
    print(f"wrote {path.name}: {len(out):,} rows from {name} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
