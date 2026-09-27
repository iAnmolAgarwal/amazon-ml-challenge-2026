import sys
import numpy as np
import polars as pl
sys.path.insert(0, "src")
import er_model as M
from common import load_config
cfg = load_config(); K = ["s1_idx", "pool_idx"]
d = pl.read_parquet("research/r22/loco_preds2.parquet")
ents = d.select("s1_idx", "country_n", "fold").unique("s1_idx")
s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1).join(ents.select("s1_idx"), on="s1_idx")
lg = lambda e: (e.clip(1e-6, 1 - 1e-6) / (1 - e.clip(1e-6, 1 - 1e-6))).log()
arms = [("f", f, 1.0, 1.0) for f in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7)] + [("T", 0.6, t, 1.0) for t in (0.7, 0.85, 1.2, 1.5)] + [("E", 0.6, 1.0, e) for e in (0.8, 0.9, 1.1, 1.25)]
out = []
for st in ("ic", "lo"):
    z = 0.5 * lg(pl.col(f"pA_{st}")) + 0.5 * lg(pl.col(f"pB2_{st}"))
    res = {}
    for kind, fl, T, er in arms:
        pr = d.select(K + [(1 / (1 + (-z / T).exp())).alias("prob")])
        e = M.per_entity_f05(M.decide(pr, cfg, floor=fl, empty_ratio=er).select(K), gold, ents)
        res[(fl, T, er)] = e.select("s1_idx", "fold", "country_n", "f05")
    b = res[(0.6, 1.0, 1.0)]
    for k, v in res.items():
        j = b.join(v.select("s1_idx", z="f05"), on="s1_idx").with_columns(dd=pl.col("z") - pl.col("f05"))
        for c in ("us", "india"):
            x = j.filter(pl.col("country_n") == c).group_by("fold").agg(pl.col("dd").mean()).sort("fold")["dd"].to_numpy()
            se = x.std(ddof=1) / np.sqrt(5)
            print(f"{st} floor={k[0]} T={k[1]} er={k[2]} [{c} held]: {x.mean():+.5f} SE {se:.5f} {(x > 0).sum()}/5 {'PASS' if x.mean() > 2 * se and (x > 0).sum() >= 4 else ''}", flush=True)
