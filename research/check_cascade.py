import sys
import numpy as np
import polars as pl
sys.path.insert(0, "src")
import er_model as M
from common import load_config
cfg = load_config()
K = ["s1_idx", "pool_idx"]
name = sys.argv[1]
s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1); fo = M.folds(cfg, s1, gold)
ents = s1.select("s1_idx", "country_n").join(fo.select("s1_idx", "fold"), on="s1_idx")
p = pl.read_parquet(f"cache/oof/{name}.parquet", columns=K + ["prob"])
base = M.per_entity_f05(M.decide(p, cfg, floor=0.6).select(K), gold, ents)
print(f"full: {len(p) / len(s1):.2f} cand/S1, F0.5 {base['f05'].mean():.6f}", flush=True)
for minp, topk in [(0.001, None), (0.005, None), (0.01, None), (0.02, None), (0.01, 10), (0.02, 10), (0.05, 10)]:
    c = p.filter(pl.col("prob") > minp)
    if topk:
        c = c.sort(["s1_idx", "prob"], descending=[False, True]).with_columns(r=pl.int_range(pl.len()).over("s1_idx")).filter(pl.col("r") < topk).drop("r")
    e = M.per_entity_f05(M.decide(c, cfg, floor=0.6).select(K), gold, ents)
    d = e.select("s1_idx", "fold", z="f05").join(base.select("s1_idx", x="f05"), on="s1_idx").with_columns(d=pl.col("z") - pl.col("x")).group_by("fold").agg(pl.col("d").mean())["d"].to_numpy()
    print(f"p > {minp}, top {topk}: {len(c) / len(s1):.2f} cand/S1, F0.5 {e['f05'].mean():.6f}, diff {d.mean():+.6f} (folds + {(d > 0).sum()}, - {(d < 0).sum()})", flush=True)
