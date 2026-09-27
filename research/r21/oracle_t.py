import sys
import numpy as np
import polars as pl
sys.path.insert(0, "src")
import er_model as M
from common import load_config

cfg = load_config()
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


if __name__ == "__main__":
  pass
s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1); fo = M.folds(cfg, s1, gold)
ents = s1.select("s1_idx", "country_n").join(fo.select("s1_idx", "fold"), on="s1_idx")
p = pl.read_parquet(f"cache/oof/{sys.argv[1]}.parquet", columns=K + ["prob"])
t = s1.select("s1_idx").join(gold.group_by("s1_idx").len(), on="s1_idx", how="left").fill_null(0).rename({"len": "t"})
base = M.per_entity_f05(M.decide(p, cfg, floor=0.6).select(K), gold, ents)["f05"].mean()
print("base", base)
for name, et in [("sum_p", None), ("oracle_T", t.select("s1_idx", et=pl.col("t").cast(pl.Float64))), ("oracle_T_min1", t.select("s1_idx", et=pl.col("t").clip(1).cast(pl.Float64)))]:
    e = et if et is not None else pl.DataFrame({"s1_idx": pl.Series("s1_idx", [], pl.UInt32), "et": pl.Series("et", [], pl.Float64)})
    r = M.per_entity_f05(decide_et(p, e), gold, ents)
    print(name, r["f05"].mean(), r.group_by("country_n").agg(pl.col("f05").mean()).rows(), flush=True)
