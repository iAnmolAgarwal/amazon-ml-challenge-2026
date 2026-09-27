import sys
from pathlib import Path
import polars as pl
sys.path.insert(0, "src"); sys.path.insert(0, "research/r21")
import er_model as M
import er_fix2 as F2
from common import load_config
K = ["s1_idx", "pool_idx"]


def decide_et(pairs, et, floor=0.6):
    d = pairs.select(K + ["prob"]).sort(["pool_idx", "prob", "s1_idx"], descending=[False, True, False]).unique("pool_idx", keep="first", maintain_order=True)
    d = d.join(et, on="s1_idx", how="left").sort(["s1_idx", "prob", "pool_idx"], descending=[False, True, False]).with_columns(
        k=pl.int_range(1, pl.len() + 1).over("s1_idx"), cum=pl.col("prob").cum_sum().over("s1_idx"),
        et0=pl.col("prob").sum().over("s1_idx"), p0=(1 - pl.col("prob")).log().sum().over("s1_idx").exp())
    d = d.with_columns(et=pl.coalesce("et", "et0")).with_columns(score=5 * pl.col("cum") / (pl.col("et") + 4 * pl.col("k")))
    best = d.sort(["s1_idx", "score", "k"], descending=[False, True, False]).group_by("s1_idx", maintain_order=True).agg(pl.col("score").first().alias("best"), pl.col("k").first().alias("kbest"), pl.col("p0").first())
    keep = best.filter(pl.col("best") > pl.col("p0")).select("s1_idx", "kbest")
    return d.join(keep, on="s1_idx").filter((pl.col("k") <= pl.col("kbest")) & (pl.col("prob") > floor)).select(K)



cfg = load_config()
s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1); fo = M.folds(cfg, s1, gold)
ents = s1.select("s1_idx", "country_n").join(fo.select("s1_idx", "fold"), on="s1_idx")
th = F2.count_prior(cfg, Path(cfg["cache_dir"]))["train"]
p = pl.read_parquet(f"cache/oof/{sys.argv[1]}.parquet", columns=K + ["prob"])
base = M.per_entity_f05(M.decide(p, cfg, floor=0.6).select(K), gold, ents).select("s1_idx", "fold", x="f05")
sp = p.sort(["pool_idx", "prob", "s1_idx"], descending=[False, True, False]).unique("pool_idx", keep="first").group_by("s1_idx").agg(sp=pl.col("prob").sum())
for name, et in [("t_hat", th.select("s1_idx", et=pl.col("t_hat").cast(pl.Float64))), ("max", th.join(sp, on="s1_idx", how="left").select("s1_idx", et=pl.max_horizontal(pl.col("t_hat").cast(pl.Float64), pl.col("sp").fill_null(0)))), ("mean", th.join(sp, on="s1_idx", how="left").select("s1_idx", et=(pl.col("t_hat").cast(pl.Float64) + pl.col("sp").fill_null(0)) / 2))]:
    r = M.per_entity_f05(decide_et(p, et), gold, ents).select("s1_idx", z="f05").join(base, on="s1_idx")
    d = r.group_by("fold").agg((pl.col("z") - pl.col("x")).mean()).sort("fold")["z"].to_numpy()
    print(name, r["z"].mean(), d.mean(), d.std(ddof=1) / 5 ** 0.5, (d > 0).sum(), flush=True)
