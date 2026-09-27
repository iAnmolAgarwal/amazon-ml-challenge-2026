"""Side-car coverage by pair source and main-probability band, train vs test (per S1 counts and share with the column)."""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
import er_meta as E
from common import load_attempt, load_config

cfg = load_config()
mod = load_attempt(sys.argv[1])
col = sys.argv[2]
cache = Path(cfg["cache_dir"])
for split in ["train", "test"]:
    u, _ = E.build(cache, split, mod)
    u = u.filter(pl.col("country_n") != "france").with_columns(src=pl.when(pl.col("pm").is_null()).then(pl.lit("stack")).when(pl.col("ps").is_null()).then(pl.lit("main")).otherwise(pl.lit("both")), band=pl.col("pm").cut([0.01, 0.1, 0.5, 0.9, 0.99]))
    ns = u.group_by("country_n").agg(pl.col("s1_idx").n_unique().alias("ns"))
    r = u.group_by("country_n", "src", "band").agg(n=pl.len(), miss=pl.col(col).is_null().sum()).join(ns, on="country_n").with_columns(per_s1=pl.col("n") / pl.col("ns"), miss_per_s1=pl.col("miss") / pl.col("ns"), cov=1 - pl.col("miss") / pl.col("n")).drop("ns", "n", "miss").sort("country_n", "src", "band")
    with pl.Config(tbl_rows=60):
        print(split, r, flush=True)
