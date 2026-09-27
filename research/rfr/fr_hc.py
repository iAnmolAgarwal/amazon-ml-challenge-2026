import polars as pl
K = ["s1_idx", "pool_idx"]
f = pl.read_parquet("research/rfr/fr_emitted.parquet")
c = f.filter((pl.col("hs").fill_null("") != "") & (pl.col("hp").fill_null("") != "") & (pl.col("hs") != pl.col("hp")))
c = c.with_columns(dh=(pl.col("hp").cast(pl.Int64, strict=False) - pl.col("hs").cast(pl.Int64, strict=False)).abs())
print(c.height, c["s1_idx"].n_unique()); print(c["dh"].value_counts().sort("count", descending=True).head(15))
k = f.group_by("s1_idx").agg(k=pl.len())
cc = c.join(k, on="s1_idx")
print("k of S1 with conflict pair", cc["k"].value_counts().sort("k").head(8))
s1 = pl.read_parquet("cache/raw/test_source1.parquet", columns=["business_name", "business_address"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
pool = pl.concat([pl.read_parquet(f"cache/raw/test_source{i}.parquet", columns=["business_name", "business_address"]) for i in (2, 3)]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32))
x = c.sample(25, seed=1).join(s1, on="s1_idx").join(pool, on="pool_idx", suffix="_p")
with pl.Config(fmt_str_lengths=45, tbl_width_chars=260, tbl_rows=30):
    print(x.select("prob", "pS", "hs", "hp", "business_name", "business_name_p", "business_address", "business_address_p"))
# same-S1 sibling: does the S1 have another emitted pair with same house?
same = f.filter((pl.col("hs") == pl.col("hp"))).group_by("s1_idx").len().rename({"len": "nsame"})
cc = cc.join(same, on="s1_idx", how="left").with_columns(pl.col("nsame").fill_null(0))
print("conflict pairs whose S1 also has same-house emitted pair:", (cc["nsame"] > 0).mean())
