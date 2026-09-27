import sys
import polars as pl
sys.path.insert(0, "src")
import er_model as M
from common import load_config
cfg = load_config(); K = ["s1_idx", "pool_idx"]
nums = lambda c: pl.col(c).fill_null("").str.to_lowercase().str.replace_all(r"\b\d{1,4}\s*(er|e|eme|ème|ere|re)\b", " ").str.extract_all(r"\b0*\d{1,4}(?:\s?(?:bis|ter|b|t))?\b").list.eval(pl.element().str.extract(r"0*(\d+)", 1)).list.unique()
s1 = pl.read_parquet("cache/raw/test_source1.parquet", columns=["business_address"]).with_row_index("s1_idx").select(pl.col("s1_idx").cast(pl.UInt32), ns=nums("business_address"))
pool = pl.concat([pl.read_parquet(f"cache/raw/test_source{i}.parquet", columns=["business_address"]) for i in (2, 3)]).with_row_index("pool_idx").select(pl.col("pool_idx").cast(pl.UInt32), np_=nums("business_address"))
f = pl.read_parquet("research/rfr/fr_emitted.parquet").join(s1, on="s1_idx").join(pool, on="pool_idx")
f = f.with_columns(rc=(pl.col("ns").list.len() > 0) & (pl.col("np_").list.len() > 0) & (pl.col("ns").list.set_intersection("np_").list.len() == 0),
                   hc=(pl.col("hs").fill_null("") != "") & (pl.col("hp").fill_null("") != "") & (pl.col("hs") != pl.col("hp")))
same = f.filter(~pl.col("rc")).group_by("s1_idx").len().rename({"len": "nok"})
f = f.join(same, on="s1_idx", how="left").with_columns(pl.col("nok").fill_null(0))
print("emitted", f.height, "hc", f["hc"].sum(), "rc", f["rc"].sum(), "rc&hc", (f["rc"] & f["hc"]).sum())
print(f.filter(pl.col("rc")).with_columns(pb=pl.col("prob").cut([0.8, 0.9, 0.95, 0.99, 0.999])).group_by("pb").agg(pl.len(), sib=(pl.col("nok") > 0).mean(), hc=pl.col("hc").mean()).sort("pb"))
print("rc pairs' S1 with a non-conflict sibling:", f.filter(pl.col("rc"))["nok"].gt(0).mean(), "S1s:", f.filter(pl.col("rc"))["s1_idx"].n_unique())
f.filter(pl.col("rc")).select(K + ["prob", "pS", "nok"]).write_parquet("research/rfr/fr_rc.parquet")
s1r = pl.read_parquet("cache/raw/test_source1.parquet", columns=["business_name", "business_address"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
pr = pl.concat([pl.read_parquet(f"cache/raw/test_source{i}.parquet", columns=["business_name", "business_address"]) for i in (2, 3)]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32))
x = f.filter(pl.col("rc") & (pl.col("prob") > 0.99)).sample(15, seed=3).join(s1r, on="s1_idx").join(pr, on="pool_idx", suffix="_p")
with pl.Config(fmt_str_lengths=50, tbl_width_chars=240, tbl_rows=20):
    print(x.select("prob", "nok", "business_name", "business_name_p", "business_address", "business_address_p"))
x = f.filter(pl.col("rc") & (pl.col("nok") == 0)).sample(12, seed=4).join(s1r, on="s1_idx").join(pr, on="pool_idx", suffix="_p")
with pl.Config(fmt_str_lengths=50, tbl_width_chars=240, tbl_rows=20):
    print(x.select("prob", "nok", "business_name", "business_name_p", "business_address", "business_address_p"))
