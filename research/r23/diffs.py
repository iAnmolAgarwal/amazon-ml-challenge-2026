import sys
sys.path.insert(0, "src")
import polars as pl
import er_model as M
from common import load_config
cfg = load_config(); K = ["s1_idx", "pool_idx"]
def hh(split):
    s = pl.read_parquet(f"cache/norm/{split}_s1.parquet", columns=["house", "country_n"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32), hs=pl.col("house").str.extract(r"(\d+)").cast(pl.Int64, strict=False)).drop("house")
    p = pl.read_parquet(f"cache/norm/{split}_pool.parquet", columns=["house"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32), hq=pl.col("house").str.extract(r"(\d+)").cast(pl.Int64, strict=False)).drop("house")
    return s, p
def hist(d, tag):
    x = d.filter(pl.col("hs").is_not_null() & pl.col("hq").is_not_null() & (pl.col("hs") != pl.col("hq"))).with_columns(ad=(pl.col("hq") - pl.col("hs")).abs())
    v = x["ad"].value_counts().sort("count", descending=True).head(14)
    print(tag, "n", x.height, "top:", " ".join(f"{a}:{c / x.height:.3f}" for a, c in v.iter_rows()), flush=True)
s, p = hh("train")
s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1)
o = pl.read_parquet("cache/oof/030p_fix.parquet").join(s, on="s1_idx").join(p, on="pool_idx").join(gold.with_columns(y=pl.lit(1)), on=K, how="left").with_columns(pl.col("y").fill_null(0))
for c in ["us", "india"]:
    e = o.filter(pl.col("country_n") == c)
    hist(e.filter(pl.col("y") == 1), f"train {c} TP")
    hist(e.filter((pl.col("y") == 0) & (pl.col("prob") > 0.01)), f"train {c} neg p>.01")
s, p = hh("test")
t = pl.read_parquet("cache/oof/test_029f7_final.parquet").join(s.filter(pl.col("country_n") == "france"), on="s1_idx").join(p, on="pool_idx")
for lo, hi in [(0.999, 2), (0.6, 0.999), (0.1, 0.6), (0.01, 0.1)]:
    hist(t.filter((pl.col("prob") >= lo) & (pl.col("prob") < hi)), f"france p[{lo},{hi})")
