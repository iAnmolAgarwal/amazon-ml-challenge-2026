import sys
sys.path.insert(0, "src")
import polars as pl
import er_model as M
from common import load_config
cfg = load_config(); K = ["s1_idx", "pool_idx"]
STEPS = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
def hh(split):
    s = pl.read_parquet(f"cache/norm/{split}_s1.parquet", columns=["house", "country_n", "nums"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32), hs=pl.col("house").str.extract(r"(\d+)").cast(pl.Int64, strict=False)).drop("house")
    p = pl.read_parquet(f"cache/norm/{split}_pool.parquet", columns=["house", "nums"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32), hq=pl.col("house").str.extract(r"(\d+)").cast(pl.Int64, strict=False)).drop("house").rename({"nums": "nums_p"})
    return s, p
def cls(d):
    d = d.filter(pl.col("hs").is_not_null() & pl.col("hq").is_not_null() & (pl.col("hs") != pl.col("hq"))).with_columns(ss=pl.col("hs").cast(pl.String), sq=pl.col("hq").cast(pl.String))
    noise = pl.col("ss").str.ends_with(pl.col("sq")) | pl.col("sq").str.ends_with(pl.col("ss")) | pl.col("ss").str.starts_with(pl.col("sq")) | pl.col("sq").str.starts_with(pl.col("ss")) | (pl.col("ss").str.split("").list.sort().list.join("") == pl.col("sq").str.split("").list.sort().list.join(""))
    cross = pl.col("nums_p").fill_null("").str.split(" ").list.contains(pl.col("ss")) | pl.col("nums").fill_null("").str.split(" ").list.contains(pl.col("sq"))
    return d.with_columns(noise=noise, cross=cross, step=(pl.col("hq") - pl.col("hs")).abs().is_in(STEPS) & ~noise)
def row(d, tag):
    d = cls(d); print(f"{tag}: n {d.height} noise {d['noise'].mean():.3f} cross {d['cross'].mean():.3f} step {d['step'].mean():.3f} other {(~d['noise'] & ~d['step'] & ~d['cross']).mean():.3f}", flush=True)
s, p = hh("train")
s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1)
o = pl.read_parquet("cache/oof/030p_fix.parquet").join(s, on="s1_idx").join(p, on="pool_idx").join(gold.with_columns(y=pl.lit(1)), on=K, how="left").with_columns(pl.col("y").fill_null(0))
for c in ["us", "india"]:
    e = o.filter(pl.col("country_n") == c)
    row(e.filter(pl.col("y") == 1), f"train {c} TP")
    row(e.filter((pl.col("y") == 0) & (pl.col("prob") > 0.01)), f"train {c} neg p>.01")
    row(e.filter((pl.col("y") == 0) & (pl.col("prob") > 0.3)), f"train {c} neg p>.3")
s, p = hh("test")
t = pl.read_parquet("cache/oof/test_029f7_final.parquet").join(s, on="s1_idx").join(p, on="pool_idx")
for c in ["france", "us"]:
    for lo, hi in [(0.999, 2), (0.99, 0.999), (0.6, 0.99), (0.3, 0.6), (0.1, 0.3), (0.01, 0.1)]:
        row(t.filter((pl.col("country_n") == c) & (pl.col("prob") >= lo) & (pl.col("prob") < hi)), f"test {c} p[{lo},{hi})")
