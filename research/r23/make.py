import sys
sys.path.insert(0, "research/r23")
from ladder import *
cols = ["house", "nums"]
sa = pl.read_parquet("cache/norm/test_s1.parquet", columns=cols).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
pa = pl.read_parquet("cache/norm/test_pool.parquet", columns=cols).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32))
full = pl.read_parquet("cache/oof/test_029f7_final.parquet")
f = full.join(fr, on="s1_idx").select(K).join(sa, on="s1_idx", how="left").join(pa, on="pool_idx", how="left", suffix="_p")
hs, hq = pl.col("house").str.extract(r"(\d+)").cast(pl.Int64, strict=False), pl.col("house_p").str.extract(r"(\d+)").cast(pl.Int64, strict=False)
f = f.with_columns(hs=hs, hq=hq, ns=pl.col("nums").fill_null("").str.split(" "), nq=pl.col("nums_p").fill_null("").str.split(" "))
f = f.with_columns(ss=pl.col("hs").cast(pl.String), sq=pl.col("hq").cast(pl.String))
conf = pl.col("hs").is_not_null() & pl.col("hq").is_not_null() & (pl.col("hs") != pl.col("hq"))
cross = pl.col("nq").list.contains(pl.col("ss")) | pl.col("ns").list.contains(pl.col("sq"))
noise = pl.col("ss").str.ends_with(pl.col("sq")) | pl.col("sq").str.ends_with(pl.col("ss")) | pl.col("ss").str.starts_with(pl.col("sq")) | pl.col("sq").str.starts_with(pl.col("ss")) | (pl.col("ss").str.split("").list.sort().list.join("") == pl.col("sq").str.split("").list.sort().list.join(""))
f = f.with_columns(c_any=conf.fill_null(False), c_strict=(conf & ~cross).fill_null(False), c_clean=(conf & ~cross & ~noise).fill_null(False)).select(K + ["c_any", "c_strict", "c_clean"])
full = full.join(f, on=K, how="left", maintain_order="left").with_columns([pl.col(c).fill_null(False) for c in ["c_any", "c_strict", "c_clean"]])
print({c: int(full[c].sum()) for c in ["c_any", "c_strict", "c_clean"]}, flush=True)
V = {"a": (pl.col("c_clean") & (pl.col("prob") < 0.999)), "b": pl.col("c_strict"), "c": pl.col("c_any")}
for k, e in V.items():
    out = full.with_columns(prob=pl.when(e).then(pl.lit(1e-6, pl.Float32)).otherwise(pl.col("prob")).cast(full.schema["prob"])).select(K + ["prob"])
    out.write_parquet(f"cache/oof/test_029f8{k}_final.parquet")
    report(f"029f8{k}", load(f"029f8{k}_final"))
