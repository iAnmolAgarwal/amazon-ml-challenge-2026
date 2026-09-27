"""Coverage of one side-car column on the meta union as built by er_meta, train vs test, by pair source and country."""
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
    u = u.filter(pl.col("country_n") != "france").with_columns(src=pl.when(pl.col("pm").is_null()).then(pl.lit("stack")).when(pl.col("ps").is_null()).then(pl.lit("main")).otherwise(pl.lit("both")))
    ns = u.group_by("country_n").agg(pl.col("s1_idx").n_unique().alias("ns"))
    r = u.group_by("country_n", "src").agg(n=pl.len(), cov=pl.col(col).is_not_null().mean(), mean=pl.col(col).mean()).join(ns, on="country_n").with_columns(per_s1=pl.col("n") / pl.col("ns")).sort("country_n", "src")
    print(split, r, flush=True)
