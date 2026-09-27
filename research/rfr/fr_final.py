import sys
import polars as pl
sys.path.insert(0, "src")
import er_model as M
from common import load_config
cfg = load_config(); K = ["s1_idx", "pool_idx"]
MODE = sys.argv[1]
nums = lambda c: pl.col(c).fill_null("").str.to_lowercase().str.replace_all(r"\b\d{1,4}\s*(er|e|eme|ème|ere|re)\b", " ").str.extract_all(r"\b0*\d{1,4}(?:\s?(?:bis|ter|b|t))?\b").list.eval(pl.element().str.extract(r"0*(\d+)", 1)).list.unique()
S = pl.read_parquet("cache/norm/test_s1.parquet", columns=["country_n", "house"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
frs = S.filter(pl.col("country_n") == "france").select("s1_idx")
s1 = pl.read_parquet("cache/raw/test_source1.parquet", columns=["business_address"]).with_row_index("s1_idx").select(pl.col("s1_idx").cast(pl.UInt32), ns=nums("business_address")).join(frs, on="s1_idx")
pool = pl.concat([pl.read_parquet(f"cache/raw/test_source{i}.parquet", columns=["business_address"]) for i in (2, 3)]).with_row_index("pool_idx").select(pl.col("pool_idx").cast(pl.UInt32), np_=nums("business_address"))
hp = pl.read_parquet("cache/norm/test_pool.parquet", columns=["house"]).with_row_index("pool_idx").select(pl.col("pool_idx").cast(pl.UInt32), hp="house")
hs = S.select("s1_idx", "country_n", hs="house")
band = pl.read_parquet("cache/oof/test_006_stage2.parquet", columns=K + ["prob"]).rename({"prob": "pb"})
base = pl.read_parquet("cache/oof/test_029f7_final.parquet", columns=K + ["prob"])
dec = lambda p: M.decide(p.join(band, on=K, how="left").filter(pl.col("pb").is_null() | (pl.col("pb") >= 0.001)).drop("pb"), cfg, floor=0.6).select(K)
e0 = dec(base)
f = base.join(frs, on="s1_idx").filter(pl.col("prob") > 0.05).join(s1, on="s1_idx").join(pool, on="pool_idx").join(hp, on="pool_idx")
f = f.with_columns(rc=(pl.col("ns").list.len() > 0) & (pl.col("np_").list.len() > 0) & (pl.col("ns").list.set_intersection("np_").list.len() == 0)).drop("ns", "np_")
sib = f.join(e0, on=K, how="semi").filter(~pl.col("rc")).select("s1_idx").unique().with_columns(sib=pl.lit(True))
f = f.join(sib, on="s1_idx", how="left").with_columns(pl.col("sib").fill_null(False))
em = f.join(e0, on=K, how="semi").filter("rc")
print("emitted rc", em.height, "hp==1", (em["hp"] == "1").mean(), "sib rate hp1", em.filter(pl.col("hp") == "1")["sib"].mean(), "sib rate other", em.filter(pl.col("hp") != "1")["sib"].mean())
cond = pl.col("rc") & pl.col("sib") & (pl.col("hp").fill_null("") != "1") if MODE in ("veto", "soft") else pl.col("rc") & (pl.col("hp").fill_null("") != "1")
v = f.filter(cond).select(K + ["prob"])
print(MODE, "veto set (cand prob>0.05)", v.height, "of which emitted", v.join(e0, on=K, how="semi").height)
new = v.with_columns(nprob=(pl.col("prob") * 0.15) if MODE == "soft" else pl.lit(0.0)).select(K + ["nprob"])
out = base.join(new, on=K, how="left").with_columns(prob=pl.coalesce("nprob", "prob").cast(pl.Float32)).drop("nprob")
e1 = dec(out)
g = lambda e: e.join(hs, on="s1_idx").join(hp, on="pool_idx")
hc = ((pl.col("hs").fill_null("") != "") & (pl.col("hp").fill_null("") != "") & (pl.col("hs") != pl.col("hp"))).mean()
sh = ((pl.col("hp").cast(pl.Int64, strict=False) - pl.col("hs").cast(pl.Int64, strict=False)).abs().is_in([1, 2, 3, 4, 5, 7, 9, 11, 13, 21])).mean()
nfr = frs.height
def prof(e):
    e = g(e).filter(pl.col("country_n") == "france")
    return dict(k=e.height, perS1=round(e.height / nfr, 4), empty=round(1 - e["s1_idx"].n_unique() / nfr, 4), hc=round(e.select(hc).item(), 4), sh=round(e.select(sh).item(), 4))
add = e1.join(e0, on=K, how="anti"); drop = e0.join(e1, on=K, how="anti")
print("029f7", prof(e0)); print("new", prof(e1)); print("added", prof(add) if add.height else 0); print("dropped", prof(drop) if drop.height else 0)
print("US/India unchanged:", out.join(frs, on="s1_idx", how="anti").sort(K).equals(base.join(frs, on="s1_idx", how="anti").sort(K)), "rows", out.height, base.height)
out.sort(K).write_parquet(f"research/rfr/test_fr_{MODE}.parquet")
