import sys
sys.path.insert(0, "research/r23")
from ladder import *
t = pl.read_parquet("cache/oof/test_029f7_final.parquet").join(fr, on="s1_idx").join(bb, on=K, how="left").filter(pl.col("pb").is_null() | (pl.col("pb") >= 0.001)).drop("pb")
d = M.decide(t, cfg, floor=0.6).select(K + ["prob"]).join(s1, on="s1_idx").join(pool, on="pool_idx")
hs, hq = pl.col("house").str.extract(r"(\d+)").cast(pl.Int64, strict=False), pl.col("hp").str.extract(r"(\d+)").cast(pl.Int64, strict=False)
d = d.with_columns(hs=hs, hq=hq).filter((pl.col("house").fill_null("") != "") & (pl.col("hp").fill_null("") != "") & (pl.col("house") != pl.col("hp")))
print("conf", d.height, "numeric-equal", d.filter(pl.col("hs") == pl.col("hq")).height, "numeric-null", d.filter(pl.col("hs").is_null() | pl.col("hq").is_null()).height)
x = d.filter(pl.col("hs") != pl.col("hq")).with_columns(ad=(pl.col("hq") - pl.col("hs")).abs())
print(x["ad"].value_counts().sort("count", descending=True).head(25))
r1 = pl.read_parquet("cache/raw/test_source1.parquet", columns=["business_name", "business_address"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
nm = pl.read_parquet("cache/norm/test_s1.parquet", columns=["entity_id"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
r1 = nm.join(pl.read_parquet("cache/raw/test_source1.parquet", columns=["entity_id", "business_name", "business_address"]), on="entity_id").drop("entity_id")
pn = pl.read_parquet("cache/norm/test_pool.parquet", columns=["entity_id"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32))
r2 = pn.join(pl.concat([pl.read_parquet(f"cache/raw/test_source{i}.parquet", columns=["entity_id", "business_name", "business_address"]) for i in (2, 3)]), on="entity_id").drop("entity_id")
pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(40)
for flt in [pl.col("ad").is_in(STEPS), ~pl.col("ad").is_in(STEPS)]:
    e = x.filter(flt).sample(12, seed=1).join(r1, on="s1_idx").join(r2, on="pool_idx", suffix="_p")
    print(e.select("prob", "business_name", "business_address", "business_name_p", "business_address_p"))
