import sys
import time
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "src")
import er_model as M
from common import load_config

cfg = load_config(); K = ["s1_idx", "pool_idx"]; t = time.time()
s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1); fo = M.folds(cfg, s1, gold)
ents = s1.select("s1_idx", "country_n").join(fo.select("s1_idx", "fold"), on="s1_idx")
u = pl.read_parquet("research/r22/cls_train.parquet").filter(pl.col("pA").is_not_null() | pl.col("pS").is_not_null())
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
sg = lambda z: 1 / (1 + np.exp(-z))
pa, ps = u["pA"].to_numpy().astype(float), u["pS"].to_numpy().astype(float)
both = ~np.isnan(pa) & ~np.isnan(ps); aon = ~np.isnan(pa) & np.isnan(ps); son = np.isnan(pa) & ~np.isnan(ps)
za, zs = lg(np.nan_to_num(pa, nan=0.5)), lg(np.nan_to_num(ps, nan=0.5))
cn = u["country_n"].to_numpy(); y = u["y"].to_numpy()
base = np.where(both, sg(0.5 * za + 0.5 * zs), np.where(aon, np.nan_to_num(pa), np.nan_to_num(ps)))
arms = {"base": base}
for tag, grp, z in (("calA", aon, za), ("calS", son, zs)):
    v = base.copy()
    for tr_c, te_c in (("india", "us"), ("us", "india")):
        m = grp & (cn == tr_c) & (np.abs(z) < 9)
        X = np.c_[z, np.maximum(z, 0)]
        r = LogisticRegression(C=1e4, max_iter=500).fit(X[m], y[m])
        print(tag, tr_c, "->", te_c, r.coef_, r.intercept_, flush=True)
        te = grp & (cn == te_c)
        v[te] = sg(X[te] @ r.coef_[0] + r.intercept_[0])
    arms[tag] = v
arms["calAS"] = np.where(aon, arms["calA"], np.where(son, arms["calS"], base))
np.save("research/r22/e2_arms.npy", np.c_[arms["base"], arms["calA"], arms["calS"]])
res = {}
for n, pr in arms.items():
    d = M.decide(u.select(K).with_columns(prob=pl.Series(pr)), cfg, floor=0.6).select(K)
    e = M.per_entity_f05(d, gold, ents)
    res[n] = e.select("s1_idx", "fold", "country_n", "f05")
    print(n, d.height, {k: round(v, 5) for k, v in e.group_by("country_n").agg(pl.col("f05").mean()).sort("country_n").iter_rows()}, f"{time.time() - t:.0f}s", flush=True)
for b in [k for k in res if k != "base"]:
    j = res["base"].join(res[b].select("s1_idx", z="f05"), on="s1_idx").with_columns(d=pl.col("z") - pl.col("f05"))
    for c in ("us", "india"):
        d = j.filter(pl.col("country_n") == c).group_by("fold").agg(pl.col("d").mean()).sort("fold")["d"].to_numpy()
        se = d.std(ddof=1) / np.sqrt(5)
        print(f"PAIRED {b} vs base [{c} held out]: {d.mean():+.5f} SE {se:.5f} folds+ {(d > 0).sum()}/5 {'PASS' if d.mean() > 2 * se and (d > 0).sum() >= 4 else 'FAIL'}", flush=True)
