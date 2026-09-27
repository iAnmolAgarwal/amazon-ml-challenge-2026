import sys
import numpy as np
import polars as pl
sys.path.insert(0, "src")
import er_model as M
from common import load_config
cfg = load_config(); K = ["s1_idx", "pool_idx"]
s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1); fo = M.folds(cfg, s1, gold)
ents = s1.select("s1_idx", "country_n").join(fo.select("s1_idx", "fold"), on="s1_idx")
u = pl.read_parquet("research/r22/cls_train.parquet").filter(pl.col("pA").is_not_null() | pl.col("pS").is_not_null()).select(K + ["pA", "pS"])
r = pl.read_parquet("research/r22/ps_train.parquet").filter((pl.col("nm") == "pseudo") & (pl.col("nk") == 1)).select(K, R=pl.lit(True))
u = u.join(r, on=K, how="left").with_columns(pl.col("R").fill_null(False))
lg = lambda e: (e.clip(1e-6, 1 - 1e-6) / (1 - e.clip(1e-6, 1 - 1e-6))).log()
both = pl.col("pA").is_not_null() & pl.col("pS").is_not_null()
u = u.with_columns(base=pl.when(both).then(1 / (1 + (-(0.5 * lg(pl.col("pA")) + 0.5 * lg(pl.col("pS")))).exp())).otherwise(pl.coalesce("pA", "pS")))
arms = {"base": u["base"], "R_pS": u.select(pl.when(pl.col("R") & pl.col("pS").is_not_null()).then(pl.col("pS")).otherwise(pl.col("base")))[:, 0],
        "R_max": u.select(pl.when(pl.col("R") & pl.col("pS").is_not_null()).then(pl.max_horizontal("base", "pS")).otherwise(pl.col("base")))[:, 0],
        "R_w02": u.select(pl.when(pl.col("R") & both).then(1 / (1 + (-(0.2 * lg(pl.col("pA")) + 0.8 * lg(pl.col("pS")))).exp())).otherwise(pl.col("base")))[:, 0]}
print("R pairs", u["R"].sum(), "R with pS", u.filter(pl.col("R") & pl.col("pS").is_not_null()).height)
res = {}
for n, pr in arms.items():
    d = M.decide(u.select(K).with_columns(prob=pr), cfg, floor=0.6).select(K)
    e = M.per_entity_f05(d, gold, ents)
    res[n] = e.select("s1_idx", "fold", "country_n", "f05")
    print(n, d.height, {k: round(v, 5) for k, v in e.group_by("country_n").agg(pl.col("f05").mean()).sort("country_n").iter_rows()}, flush=True)
for b in [k for k in res if k != "base"]:
    j = res["base"].join(res[b].select("s1_idx", z="f05"), on="s1_idx").with_columns(d=pl.col("z") - pl.col("f05"))
    for c in ("us", "india"):
        d = j.filter(pl.col("country_n") == c).group_by("fold").agg(pl.col("d").mean()).sort("fold")["d"].to_numpy()
        se = d.std(ddof=1) / np.sqrt(5)
        print(f"PAIRED {b} vs base [{c}]: {d.mean():+.6f} SE {se:.6f} folds+ {(d > 0).sum()}/5 {'PASS' if d.mean() > 2 * se and (d > 0).sum() >= 4 else 'FAIL'}", flush=True)
