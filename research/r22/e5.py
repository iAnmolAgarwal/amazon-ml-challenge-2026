import sys
import numpy as np
import polars as pl
sys.path.insert(0, "src"); sys.path.insert(0, "research/r22")
import er_model as M
from common import load_config
from rcls import views, flag
cfg = load_config(); K = ["s1_idx", "pool_idx"]
d = pl.read_parquet("research/r22/loco_preds2.parquet")
d = flag(d, *views("train"))
ents = d.select("s1_idx", "country_n", "fold").unique("s1_idx")
s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1).join(ents.select("s1_idx"), on="s1_idx")
print(d.filter("R").group_by("country_n").agg(n=pl.len(), y=pl.col("y").mean(), A_ic=pl.col("pA_ic").mean(), A_lo=pl.col("pA_lo").mean(), B_ic=pl.col("pB2_ic").mean(), B_lo=pl.col("pB2_lo").mean(), Alo_low=(pl.col("pA_lo") < 0.5).mean(), Blo_hi=(pl.col("pB2_lo") > 0.9).mean()), flush=True)
lg = lambda e: (e.clip(1e-6, 1 - 1e-6) / (1 - e.clip(1e-6, 1 - 1e-6))).log()
for st in ("ic", "lo"):
    z = 0.5 * lg(pl.col(f"pA_{st}")) + 0.5 * lg(pl.col(f"pB2_{st}"))
    b = 1 / (1 + (-z).exp())
    pa, pb = pl.col(f"pA_{st}"), pl.col(f"pB2_{st}")
    arms = {"base": b, "max_B": pl.when("R").then(pl.max_horizontal(b, pb)).otherwise(b), "max_A": pl.when("R").then(pl.max_horizontal(b, pa)).otherwise(b),
            "fl061_B09": pl.when(pl.col("R") & (pb > 0.9)).then(pl.max_horizontal(b, pl.lit(0.61))).otherwise(b), "fl061_any09": pl.when(pl.col("R") & (pl.max_horizontal(pa, pb) > 0.9)).then(pl.max_horizontal(b, pl.lit(0.61))).otherwise(b)}
    res = {}
    for n, e in arms.items():
        pr = d.select(K + [e.alias("prob")])
        res[n] = M.per_entity_f05(M.decide(pr, cfg, floor=0.6).select(K), gold, ents).select("s1_idx", "fold", "country_n", "f05")
    for n in arms:
        if n == "base":
            continue
        j = res["base"].join(res[n].select("s1_idx", z="f05"), on="s1_idx").with_columns(dd=pl.col("z") - pl.col("f05"))
        for c in ("us", "india"):
            x = j.filter(pl.col("country_n") == c).group_by("fold").agg(pl.col("dd").mean()).sort("fold")["dd"].to_numpy()
            se = x.std(ddof=1) / np.sqrt(5)
            print(f"{st} {n} [{c} held]: {x.mean():+.6f} SE {se:.6f} {(x > 0).sum()}/5 {'PASS' if x.mean() > 2 * se and (x > 0).sum() >= 4 else ''}", flush=True)
