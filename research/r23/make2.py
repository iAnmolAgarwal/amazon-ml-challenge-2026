import sys
sys.path.insert(0, "research/r23")
from ladder import *
cols = ["house", "nums"]
sa = pl.read_parquet("cache/norm/test_s1.parquet", columns=cols).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
pa = pl.read_parquet("cache/norm/test_pool.parquet", columns=cols).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32))
def flags(full):
    f = full.join(fr, on="s1_idx").select(K).join(sa, on="s1_idx", how="left").join(pa, on="pool_idx", how="left", suffix="_p")
    f = f.with_columns(hs=pl.col("house").str.extract(r"(\d+)").cast(pl.Int64, strict=False), hq=pl.col("house_p").str.extract(r"(\d+)").cast(pl.Int64, strict=False), ns=pl.col("nums").fill_null("").str.split(" "), nq=pl.col("nums_p").fill_null("").str.split(" "))
    f = f.with_columns(ss=pl.col("hs").cast(pl.String), sq=pl.col("hq").cast(pl.String))
    conf = pl.col("hs").is_not_null() & pl.col("hq").is_not_null() & (pl.col("hs") != pl.col("hq"))
    cross = pl.col("nq").list.contains(pl.col("ss")) | pl.col("ns").list.contains(pl.col("sq"))
    noise = pl.col("ss").str.ends_with(pl.col("sq")) | pl.col("sq").str.ends_with(pl.col("ss")) | pl.col("ss").str.starts_with(pl.col("sq")) | pl.col("sq").str.starts_with(pl.col("ss")) | (pl.col("ss").str.split("").list.sort().list.join("") == pl.col("sq").str.split("").list.sort().list.join(""))
    f = f.with_columns(conf=conf.fill_null(False), strict=(conf & ~cross).fill_null(False), clean=(conf & ~cross & ~noise).fill_null(False)).select(K + ["conf", "strict", "clean"])
    return full.join(f, on=K, how="left", maintain_order="left").with_columns([pl.col(c).fill_null(False) for c in ["conf", "strict", "clean"]])
p = pl.col("prob")
V = {"a": pl.col("clean") & (p < 0.999), "b": pl.col("clean") | (pl.col("strict") & (p < 0.99)), "c": pl.col("clean") | (pl.col("conf") & (p < 0.99))}
f7 = flags(pl.read_parquet("cache/oof/test_029f7_final.parquet"))
q = pl.read_parquet("cache/oof/test_030q_fix.parquet").join(fr.with_columns(fr_=pl.lit(True)), on="s1_idx", how="left").with_columns(prob=pl.when(pl.col("fr_")).then(1 / (1 + (-lg(pl.col("prob")) / 2.0).exp())).otherwise(pl.col("prob")).cast(pl.Float32)).drop("fr_")
f2 = flags(q)
for k, src, e in [("a", f7, V["a"]), ("b", f7, V["b"]), ("c", f2, V["c"])]:
    out = src.with_columns(prob=pl.when(e).then(pl.lit(1e-6)).otherwise(p).cast(pl.Float32)).select(K + ["prob"])
    out.write_parquet(f"cache/oof/test_029f8{k}_final.parquet")
    print(k, "vetoed rows", int(src.select(e.sum()).item()), flush=True)
    report(f"029f8{k}", load(f"029f8{k}_final"))
