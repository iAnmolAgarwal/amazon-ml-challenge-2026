import sys
import polars as pl
sys.path.insert(0, "src")
import er_model as M
from common import load_config
cfg = load_config(); K = ["s1_idx", "pool_idx"]
s = pl.read_parquet("cache/norm/train_s1.parquet", columns=["country_n", "house"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32)).rename({"house": "hs"})
p = pl.read_parquet("cache/norm/train_pool.parquet", columns=["house"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32)).rename({"house": "hp"})
oof = pl.read_parquet("cache/oof/030p_fix.parquet", columns=K + ["prob"])
s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1)
e = oof.filter(pl.col("prob") > 0.05).join(gold.with_columns(y=pl.lit(1)), on=K, how="left").with_columns(pl.col("y").fill_null(0)).join(s, on="s1_idx").join(p, on="pool_idx")
c = e.filter((pl.col("hs").fill_null("") != "") & (pl.col("hp").fill_null("") != "") & (pl.col("hs") != pl.col("hp")))
print(c.group_by("country_n", "y").agg(pl.len(), hp1=(pl.col("hp") == "1").mean(), dh1=((pl.col("hp").cast(pl.Int64, strict=False) - pl.col("hs").cast(pl.Int64, strict=False)).abs() <= 2).mean()).sort("country_n", "y"))
print("gold pairs hp==1 share by country", e.filter(pl.col("y") == 1).group_by("country_n").agg(hp1=(pl.col("hp") == "1").mean(), hs1=(pl.col("hs") == "1").mean()))
f = pl.read_parquet("research/rfr/fr_emitted.parquet")
print("FR emitted hp==1", (f["hp"] == "1").mean(), "hs==1", (f["hs"] == "1").mean())
fc = f.filter((pl.col("hs").fill_null("") != "") & (pl.col("hp").fill_null("") != "") & (pl.col("hs") != pl.col("hp")))
print("FR conflict hp==1", (fc["hp"] == "1").mean(), (fc["hs"] == "1").mean(), fc.height)
