import sys, time
import numpy as np
import polars as pl
sys.path.insert(0, "src")
import er_model as M
from common import load_config
cfg = load_config(); K = ["s1_idx", "pool_idx"]; t0 = time.time()
s1 = pl.read_parquet("cache/norm/test_s1.parquet", columns=["entity_id", "country_n", "house"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
pool = pl.read_parquet("cache/norm/test_pool.parquet", columns=["entity_id", "house"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32))
fr = s1.filter(pl.col("country_n") == "france").select("s1_idx", a_id="entity_id")
band = pl.read_parquet("cache/oof/test_006_stage2.parquet", columns=K + ["prob"]).rename({"prob": "pb"})
base = pl.read_parquet("cache/oof/test_029f7_final.parquet", columns=K + ["prob"])
bf = base.join(fr.select("s1_idx"), on="s1_idx")
st = pl.read_parquet("cache/stack/p_test_France.parquet").join(fr, on="a_id").join(pool.select("pool_idx", r_id="entity_id"), on="r_id").select(K + [pl.col("p").alias("pS")])
d = bf.join(st, on=K, how="full", coalesce=True)
print("base FR pairs", bf.height, "stack pairs", st.height, "both", d.filter(pl.col("prob").is_not_null() & pl.col("pS").is_not_null()).height, "stack-only", d.filter(pl.col("prob").is_null()).height, "base-only", d.filter(pl.col("pS").is_null()).height)
lg = lambda e: (e.clip(1e-6, 1 - 1e-6) / (1 - e.clip(1e-6, 1 - 1e-6))).log()
sg = lambda z: 1 / (1 + (-z).exp())
hs = s1.select("s1_idx", "country_n", hs="house"); hp = pool.select("pool_idx", hp="house")
def dec(p):
    p = p.join(band, on=K, how="left").filter(pl.col("pb").is_null() | (pl.col("pb") >= 0.001)).drop("pb")
    return M.decide(p.select(K + ["prob"]), cfg, floor=0.6).select(K).join(hs, on="s1_idx").join(hp, on="pool_idx")
hc = ((pl.col("hs").fill_null("") != "") & (pl.col("hp").fill_null("") != "") & (pl.col("hs") != pl.col("hp"))).mean()
dh = (pl.col("hp").cast(pl.Int64, strict=False) - pl.col("hs").cast(pl.Int64, strict=False)).abs()
sh = dh.is_in([1, 2, 3, 4, 5, 7, 9, 11, 13, 21]).mean()
nfr = fr.height
def prof(e):
    return dict(k=e.height, perS1=round(e.height / nfr, 4), empty=round(1 - e["s1_idx"].n_unique() / nfr, 4), hc=round(e.select(hc).item(), 4), sh=round(e.select(sh).item(), 4))
e_all = M.decide(base.join(band, on=K, how="left").filter(pl.col("pb").is_null() | (pl.col("pb") >= 0.001)).drop("pb"), cfg, floor=0.6).select(K).join(hs, on="s1_idx").join(hp, on="pool_idx")
ns = s1.group_by("country_n").len()
print(e_all.group_by("country_n").agg(k=pl.len(), ne=pl.col("s1_idx").n_unique(), hc=hc, sh=sh).join(ns, on="country_n").with_columns(perS1=pl.col("k") / pl.col("len"), empty=1 - pl.col("ne") / pl.col("len")))
e0 = dec(bf)
print("BASE 029f7", prof(e0))
has = pl.col("pS").is_not_null() & pl.col("prob").is_not_null() & (pl.col("prob") > 1e-4)
V = {}
for a in [0.7, 0.5, 0.3]:
    V[f"shift{a}"] = d.with_columns(prob=pl.when(has).then(sg(a * lg(pl.col("prob")) + (1 - a) * lg(pl.col("pS")) / 1.5)).otherwise(pl.col("prob")).fill_null(0.0))
for t in [0.1, 0.3, 0.5]:
    V[f"cons{t}"] = d.with_columns(prob=pl.when(pl.col("pS").fill_null(0) >= t).then(pl.col("prob")).otherwise(pl.col("prob") * 0.2).fill_null(0.0))
V["consNull0.3"] = d.with_columns(prob=pl.when(pl.col("pS").is_null() | (pl.col("pS") >= 0.3)).then(pl.col("prob")).otherwise(pl.col("prob") * 0.2).fill_null(0.0))
V["stackT2"] = d.with_columns(prob=sg(lg(pl.col("pS").fill_null(0.0)) / 2.0))
V["min"] = d.with_columns(prob=pl.when(has).then(pl.min_horizontal("prob", sg(lg(pl.col("pS")) / 1.5))).otherwise(pl.col("prob")).fill_null(0.0))
for n, v in V.items():
    e = dec(v.select(K + ["prob"]))
    add = e.join(e0, on=K, how="anti"); drop = e0.join(e, on=K, how="anti")
    print(n, prof(e), "| added", prof(add) if add.height else 0, "| dropped", prof(drop) if drop.height else 0, flush=True)
    v.select(K + ["prob"]).write_parquet(f"research/rfr/fr_{n}.parquet")
print(time.time() - t0)
