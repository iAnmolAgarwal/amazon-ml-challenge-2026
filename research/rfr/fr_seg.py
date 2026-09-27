import sys
import polars as pl
sys.path.insert(0, "src")
import er_model as M
from common import load_config
cfg = load_config(); K = ["s1_idx", "pool_idx"]
def hv(split):
    s = pl.read_parquet(f"cache/norm/{split}_s1.parquet", columns=["country_n", "house"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32)).rename({"house": "hs"})
    p = pl.read_parquet(f"cache/norm/{split}_pool.parquet", columns=["house"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32)).rename({"house": "hp"})
    return s, p
both = (pl.col("hs").fill_null("") != "") & (pl.col("hp").fill_null("") != "")
agg = [pl.len().alias("n"), both.mean().alias("both"), (both & (pl.col("hs") != pl.col("hp"))).sum().alias("nc"), ((pl.col("hp").cast(pl.Int64, strict=False) - pl.col("hs").cast(pl.Int64, strict=False)).abs().is_in([1, 2, 3, 4, 5, 7, 9, 11, 13, 21])).mean().alias("sh")]
fin = lambda g: g.with_columns(hcb=pl.col("nc") / (pl.col("n") * pl.col("both")))
s, p = hv("train")
oof = pl.read_parquet("cache/oof/030p_fix.parquet", columns=K + ["prob"])
s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1)
e = M.decide(oof, cfg, floor=0.6).select(K + ["prob"]).join(gold.with_columns(y=pl.lit(1)), on=K, how="left").with_columns(pl.col("y").fill_null(0)).join(s, on="s1_idx").join(p, on="pool_idx")
print("TRAIN emitted by country x y"); print(fin(e.group_by("country_n", "y").agg(agg)).sort("country_n", "y"))
e = e.with_columns(pb=pl.col("prob").cut([0.7, 0.8, 0.9, 0.95, 0.99]))
print(fin(e.group_by("country_n", "pb").agg(agg + [pl.col("y").mean().alias("prec")])).sort("country_n", "pb"))
s, p = hv("test")
base = pl.read_parquet("cache/oof/test_029f7_final.parquet", columns=K + ["prob"])
band = pl.read_parquet("cache/oof/test_006_stage2.parquet", columns=K + ["prob"]).rename({"prob": "pb0"})
base = base.join(band, on=K, how="left").filter(pl.col("pb0").is_null() | (pl.col("pb0") >= 0.001)).drop("pb0")
e = M.decide(base, cfg, floor=0.6).join(s, on="s1_idx").join(p, on="pool_idx")
fr = s.filter(pl.col("country_n") == "france").with_row_index("i")
ent = pl.read_parquet("cache/norm/test_s1.parquet", columns=["entity_id"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
pent = pl.read_parquet("cache/norm/test_pool.parquet", columns=["entity_id"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32))
st = pl.read_parquet("cache/stack/p_test_France.parquet").join(ent.rename({"entity_id": "a_id"}), on="a_id").join(pent.rename({"entity_id": "r_id"}), on="r_id").select(K + [pl.col("p").alias("pS")])
e = e.join(st, on=K, how="left").with_columns(pb=pl.col("prob").cut([0.7, 0.8, 0.9, 0.95, 0.99]), sb=pl.when(pl.col("pS").is_null()).then(pl.lit("null")).otherwise(pl.col("pS").cut([0.1, 0.3, 0.5, 0.8]).cast(pl.Utf8)))
print("TEST emitted by country"); print(fin(e.group_by("country_n").agg(agg)).sort("country_n"))
print(fin(e.group_by("country_n", "pb").agg(agg)).sort("country_n", "pb"))
f = e.filter(pl.col("country_n") == "france")
print("FR by stack band"); print(fin(f.group_by("sb").agg(agg + [pl.col("prob").mean().alias("pm")])).sort("sb"))
print("FR null-stack by prob band"); print(fin(f.filter(pl.col("pS").is_null()).group_by("pb").agg(agg)).sort("pb"))
print("FR pS<0.3 by prob band"); print(fin(f.filter(pl.col("pS") < 0.3).group_by("pb").agg(agg)).sort("pb"))
f.select(K + ["prob", "pS", "hs", "hp"]).write_parquet("research/rfr/fr_emitted.parquet")
