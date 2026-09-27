import sys
import polars as pl
split = sys.argv[1]
def toks(c):
    return pl.col(c).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_lowercase().str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars().str.split(" ")
s = pl.read_parquet(f"cache/raw/{split}_source1.parquet", columns=["business_address"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32), t=toks("business_address")).drop("business_address")
cn = pl.read_parquet(f"cache/norm/{split}_s1.parquet", columns=["country_n"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
s = s.join(cn, on="s1_idx")
fq = s.select("country_n", w="t").explode("w").group_by("country_n", "w").len().sort("len", descending=True).group_by("country_n").head(150)
s = s.explode("t").join(fq.select("country_n", t="w", stop=pl.lit(1)), on=["country_n", "t"], how="left")
num = s.filter(pl.col("t").str.contains(r"^\d+$")).group_by("s1_idx", maintain_order=True).agg(h=pl.col("t").first().str.replace(r"^0+", ""))
sw = s.filter(pl.col("stop").is_null() & ~pl.col("t").str.contains(r"^\d") & (pl.col("t").str.len_chars() > 2)).group_by("s1_idx").agg(sw=pl.col("t").unique().sort().str.join(" "))
key = cn.join(num, on="s1_idx", how="left").join(sw, on="s1_idx", how="left").filter(pl.col("h").is_not_null() & pl.col("sw").is_not_null())
key = key.with_columns(nk=pl.len().over("country_n", "h", "sw"))
d = pl.read_parquet(f"research/r22/st_{split}.parquet").filter(pl.col("sr").is_in(["ov", "typo"])).join(key.select("s1_idx", "nk"), on="s1_idx", how="left")
d = d.with_columns(nkb=pl.col("nk").fill_null(0).clip(0, 3))
agg = [pl.len().alias("n"), pl.col("em").mean().alias("em"), pl.col("p").fill_null(0).mean().alias("mp"), pl.col("em").sum().alias("n_em")] + ([pl.col("y").mean().alias("y"), pl.col("y").sum().alias("ny")] if split == "train" else [pl.col("pA").mean().alias("mA"), pl.col("pS").mean().alias("mS"), pl.col("pS").is_null().mean().alias("nS")])
t = d.filter(pl.col("nm").is_in(["pseudo", "swap1", "same"])).group_by("country_n", "nm", "nkb").agg(agg).sort("country_n", "nm", "nkb").with_columns(pl.col(pl.Float64, pl.Float32).round(3))
for r in t.iter_rows(): print(" ".join(str(x) for x in r))
d.write_parquet(f"research/r22/ps_{split}.parquet")
