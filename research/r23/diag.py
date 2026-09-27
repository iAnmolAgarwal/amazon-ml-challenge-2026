import sys
sys.path.insert(0, "src")
import polars as pl
import er_model as M
from common import load_config
cfg = load_config(); K = ["s1_idx", "pool_idx"]
STEPS = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
def recs(split):
    s = pl.read_parquet(f"cache/norm/{split}_s1.parquet", columns=["country_n", "house"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
    p = pl.read_parquet(f"cache/norm/{split}_pool.parquet", columns=["house"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32)).rename({"house": "hp"})
    return s, p
def tag(x, cand, s, p):
    x = x.join(s, on="s1_idx").join(p, on="pool_idx")
    c = cand.join(p, on="pool_idx").filter(pl.col("hp").fill_null("") != "").group_by("s1_idx", "hp").agg(nh=pl.len())
    x = x.join(c, on=["s1_idx", "hp"], how="left").with_columns(pl.col("nh").fill_null(0))
    hs, hq = pl.col("house").str.extract(r"(\d+)").cast(pl.Int64, strict=False), pl.col("hp").str.extract(r"(\d+)").cast(pl.Int64, strict=False)
    conf = (pl.col("house").fill_null("") != "") & (pl.col("hp").fill_null("") != "") & (pl.col("house") != pl.col("hp"))
    return x.with_columns(conf=conf, shift=conf & (hq - hs).abs().is_in(STEPS).fill_null(False), clus=conf & (pl.col("nh") >= 2))
if sys.argv[1] == "train":
    s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1)
    s, p = recs("train")
    o = pl.read_parquet("cache/oof/030p_fix.parquet")
    cand = o.filter(pl.col("prob") >= 0.001).select(K)
    d = M.decide(o, cfg, floor=0.6).select(K + ["prob"])
    d = tag(d, cand, s, p).join(gold.with_columns(y=pl.lit(1)), on=K, how="left").with_columns(pl.col("y").fill_null(0))
    for c in ["us", "india"]:
        e = d.filter(pl.col("country_n") == c)
        print(c, "pairs", e.height, "prec", round(e["y"].mean(), 4))
        for f in ["conf", "shift", "clus"]:
            g = e.filter(pl.col(f))
            print("  ", f, "share", round(g.height / e.height, 5), "prec", round(g["y"].mean(), 4) if g.height else None, "n", g.height)
        g = e.filter(pl.col("shift")).with_columns(pb=(pl.col("prob") * 10).floor())
        print(g.group_by("pb").agg(n=pl.len(), prec=pl.col("y").mean()).sort("pb"))
else:
    s, p = recs("test")
    fr = s.filter(pl.col("country_n") == "france").select("s1_idx")
    b = pl.read_parquet("cache/oof/test_006_stage2.parquet", columns=K + ["prob"]).rename({"prob": "pb"})
    for f in sys.argv[2:]:
        q = pl.read_parquet(f"cache/oof/test_{f}.parquet").join(fr, on="s1_idx")
        q = q.join(b, on=K, how="left").filter(pl.col("pb").is_null() | (pl.col("pb") >= 0.001)).drop("pb")
        cand = q.filter(pl.col("prob") >= 0.001).select(K)
        d = tag(M.decide(q, cfg, floor=0.6).select(K + ["prob"]), cand, s, p)
        print(f, "pairs/S1", round(d.height / fr.height, 4), "empty", round(1 - d["s1_idx"].n_unique() / fr.height, 4), {k: round(d[k].mean(), 5) for k in ["conf", "shift", "clus"]})
