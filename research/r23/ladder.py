import sys
sys.path.insert(0, "src")
import numpy as np
import polars as pl
import er_model as M
from common import load_config
cfg = load_config(); K = ["s1_idx", "pool_idx"]
STEPS = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
s1 = pl.read_parquet("cache/norm/test_s1.parquet", columns=["country_n", "house"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
pool = pl.read_parquet("cache/norm/test_pool.parquet", columns=["house"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32)).rename({"house": "hp"})
fr = s1.filter(pl.col("country_n") == "france").select("s1_idx"); n = fr.height
bb = pl.read_parquet("cache/oof/test_006_stage2.parquet", columns=K + ["prob"]).rename({"prob": "pb"})
def load(f):
    q = pl.read_parquet(f"cache/oof/test_{f}.parquet").join(fr, on="s1_idx")
    return q.join(bb, on=K, how="left").filter(pl.col("pb").is_null() | (pl.col("pb") >= 0.001)).drop("pb")
def prof(x):
    if x.height == 0: return (0, 0, 0)
    x = x.join(s1, on="s1_idx").join(pool, on="pool_idx")
    hs, hq = pl.col("house").str.extract(r"(\d+)").cast(pl.Int64, strict=False), pl.col("hp").str.extract(r"(\d+)").cast(pl.Int64, strict=False)
    c = (pl.col("house").fill_null("") != "") & (pl.col("hp").fill_null("") != "") & (pl.col("house") != pl.col("hp"))
    return x.height, round(x.select(c.mean()).item(), 4), round(x.select((c & (hq - hs).abs().is_in(STEPS)).fill_null(False).mean()).item(), 4)
lg = lambda e: (e.clip(1e-6, 1 - 1e-6) / (1 - e.clip(1e-6, 1 - 1e-6))).log()
def temp(q, T, sh=0.0): return q.with_columns(prob=1 / (1 + (-(lg(pl.col("prob")) / T - sh)).exp()))
ref = M.decide(load("029f7_final"), cfg, floor=0.6).select(K)
def report(name, q, fl=0.6):
    d = M.decide(q, cfg, floor=fl).select(K)
    add, drop = d.join(ref, on=K, how="anti"), ref.join(d, on=K, how="anti")
    print(f"{name}: pairs/S1 {d.height / n:.4f} empty {1 - d['s1_idx'].n_unique() / n:.4f} | all {prof(d)} | dropped vs 029f7 {prof(drop)} | added {prof(add)}", flush=True)
    return d
if __name__ == "__main__":
    report("029f7", load("029f7_final"))
    for f in sys.argv[1:]:
        q = load(f)
        for T in [1.0, 1.5, 2.0, 3.0]:
            report(f"{f} T{T}", temp(q, T))
