"""Where the in-band pairs without the side-car column come from: blocking candidates, retrieval route, other (key/reverse-text)."""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
import er_meta as E
from common import load_attempt, load_config

cfg = load_config()
mod = load_attempt(sys.argv[1]); col = sys.argv[2]
cache = Path(cfg["cache_dir"])
for split, files in [("train", ["train_feat5"]), ("test", ["test_feat5p", "test_feat5_new"])]:
    u, _ = E.build(cache, split, mod)
    u = u.filter((pl.col("country_n") != "france") & pl.col("pm").is_not_null()).with_columns(inb=(pl.col("pm") > 0.1) & (pl.col("pm") < 0.99), miss=pl.col(col).is_null(), src=pl.lit("other"))
    for f in files[::-1]:
        k = pl.read_parquet(cache / "feat" / f"{f}.parquet", columns=E.KEYS).join(u.select(E.KEYS), on=E.KEYS, how="semi").with_columns(t=pl.lit(f))
        u = u.join(k, on=E.KEYS, how="left").with_columns(src=pl.coalesce("t", "src")).drop("t")
    ns = u.group_by("country_n").agg(pl.col("s1_idx").n_unique().alias("ns"))
    r = u.filter("inb").group_by("country_n", "src", "miss").agg(n=pl.len()).join(ns, on="country_n").with_columns(per_s1=pl.col("n") / pl.col("ns")).drop("ns").sort("country_n", "src", "miss")
    with pl.Config(tbl_rows=40):
        print(split, r, flush=True)
