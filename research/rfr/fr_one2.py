import sys
import polars as pl
sys.path.insert(0, "src")
import er_model as M
from common import load_config
cfg = load_config(); K = ["s1_idx", "pool_idx"]
for split, fn in [("train", "030p_fix"), ("test", "test_029f7_final")]:
    s = pl.read_parquet(f"cache/norm/{split}_s1.parquet", columns=["country_n", "house"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32)).rename({"house": "hs"})
    p = pl.read_parquet(f"cache/norm/{split}_pool.parquet", columns=["house"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32)).rename({"house": "hp"})
    e = M.decide(pl.read_parquet(f"cache/oof/{fn}.parquet", columns=K + ["prob"]), cfg, floor=0.6).join(s, on="s1_idx").join(p, on="pool_idx")
    c = e.filter((pl.col("hs").fill_null("") != "") & (pl.col("hp").fill_null("") != "") & (pl.col("hs") != pl.col("hp")))
    print(split, c.group_by("country_n").agg(pl.len(), hp1=(pl.col("hp") == "1").mean(), hs1=(pl.col("hs") == "1").mean()).sort("country_n"))
    print(split, "all emitted", e.group_by("country_n").agg(hp1=(pl.col("hp") == "1").mean(), hs1=(pl.col("hs") == "1").mean()).sort("country_n"))
