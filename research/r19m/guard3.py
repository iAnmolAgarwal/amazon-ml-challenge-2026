"""Test emitted-pair changes B vs A by pair source (blocking / route / other) with house-conflict share; train by blocking / other."""
import sys

import polars as pl

sys.path.insert(0, "src")
import er_model as M
from common import load_config

cfg = load_config()
K = ["s1_idx", "pool_idx"]
a, b = sys.argv[1], sys.argv[2]
for split, pre, files in [("train", "", ["train_feat5"]), ("test", "test_", ["test_feat5p", "test_feat5_new"])]:
    s1 = pl.read_parquet(f"cache/norm/{split}_s1.parquet", columns=["country_n", "house"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
    pool = pl.read_parquet(f"cache/norm/{split}_pool.parquet", columns=["house"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32)).rename({"house": "house_p"})
    d = {n: M.decide(pl.read_parquet(f"cache/oof/{pre}{n}.parquet", columns=K + ["prob"]), cfg, floor=0.6).select(K) for n in (a, b)}
    ch = pl.concat([d[b].join(d[a], on=K, how="anti").with_columns(kind=pl.lit("added")), d[a].join(d[b], on=K, how="anti").with_columns(kind=pl.lit("dropped"))]).with_columns(src=pl.lit("other"))
    for f in files[::-1]:
        k = pl.read_parquet(f"cache/feat/{f}.parquet", columns=K).join(ch.select(K), on=K, how="semi").with_columns(t=pl.lit(f))
        ch = ch.join(k, on=K, how="left").with_columns(src=pl.coalesce("t", "src")).drop("t")
    ch = ch.join(s1, on="s1_idx").filter(pl.col("country_n") != "france").join(pool, on="pool_idx").with_columns(conf=(pl.col("house").fill_null("") != "") & (pl.col("house_p").fill_null("") != "") & (pl.col("house") != pl.col("house_p")))
    ns = s1.group_by("country_n").len().rename({"len": "ns"})
    r = ch.group_by("country_n", "kind", "src").agg(n=pl.len(), conf=pl.col("conf").mean()).join(ns, on="country_n").with_columns(per_s1=pl.col("n") / pl.col("ns")).drop("ns").sort("country_n", "kind", "src")
    with pl.Config(tbl_rows=40):
        print(split, r, flush=True)
