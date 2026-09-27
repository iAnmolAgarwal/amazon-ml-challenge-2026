"""Emitted-pair changes B vs A on train OOF and test: added / dropped per S1 and their house-conflict share, per country."""
import sys

import polars as pl

sys.path.insert(0, "src")
import er_model as M
from common import load_config

cfg = load_config()
K = ["s1_idx", "pool_idx"]
a, b = sys.argv[1], sys.argv[2]
for split, pre in [("train", ""), ("test", "test_")]:
    s1 = pl.read_parquet(f"cache/norm/{split}_s1.parquet", columns=["country_n", "house"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
    pool = pl.read_parquet(f"cache/norm/{split}_pool.parquet", columns=["house"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32)).rename({"house": "house_p"})
    d = {n: M.decide(pl.read_parquet(f"cache/oof/{pre}{n}.parquet", columns=K + ["prob"]), cfg, floor=0.6).select(K) for n in (a, b)}
    out = {}
    for c in ("us", "india"):
        ns = s1.filter(pl.col("country_n") == c).height
        r = []
        for x in (d[b].join(d[a], on=K, how="anti"), d[a].join(d[b], on=K, how="anti"), d[b]):
            x = x.join(s1, on="s1_idx").filter(pl.col("country_n") == c).join(pool, on="pool_idx")
            conf = x.filter((pl.col("house").fill_null("") != "") & (pl.col("house_p").fill_null("") != "") & (pl.col("house") != pl.col("house_p"))).height
            r.append((round(x.height / ns, 4), round(conf / max(x.height, 1), 3)))
        out[c] = {"added": r[0], "dropped": r[1], "emitted": r[2]}
    print(split, b, "vs", a, out, flush=True)
