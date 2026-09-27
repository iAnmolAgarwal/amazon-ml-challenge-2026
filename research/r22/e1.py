import sys
import time
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "src")
import er_meta as E
import er_model as M
from common import load_config

cfg = load_config(); cache = Path(cfg["cache_dir"]); K = ["s1_idx", "pool_idx"]; t = time.time()
A = sys.argv[1] if len(sys.argv) > 1 else "024_keys"
s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1); fo = M.folds(cfg, s1, gold)
ents = s1.select("s1_idx", "country_n").join(fo.select("s1_idx", "fold"), on="s1_idx")
s, p = E.ids(cache, "train")
sp = E.stack(Path("cache/stack"), "train", s, p, (0.02, 10), "none").rename({"ps": "pS"})
a = pl.read_parquet(cache / "oof" / f"{A}.parquet", columns=K + ["prob"]).filter(pl.col("prob") >= 0.001).rename({"prob": "pA"})
u = a.join(sp, on=K, how="full", coalesce=True).join(ents.select("s1_idx", "country_n"), on="s1_idx")
u = u.join(gold.with_columns(y=pl.lit(1, pl.Int8)), on=K, how="left").with_columns(pl.col("y").fill_null(0))
print(u.height, f"{time.time() - t:.0f}s", flush=True)
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
sg = lambda z: 1 / (1 + np.exp(-z))
pa, ps = u["pA"].to_numpy(), u["pS"].to_numpy()
both = ~np.isnan(pa) & ~np.isnan(ps)
za, zs = lg(np.nan_to_num(pa, nan=0.5)), lg(np.nan_to_num(ps, nan=0.5))
single = np.where(np.isnan(pa), ps, pa)
cn = u["country_n"].to_numpy(); y = u["y"].to_numpy()
avg = 0.5 * za + 0.5 * zs
arms = {"base": np.where(both, sg(avg), single)}
for c in (1.2, 1.5, 2.0):
    arms[f"sh{c}"] = np.where(both, sg(c * avg), single)
for tr_c, te_c in (("india", "us"), ("us", "india")):
    m = both & (cn == tr_c)
    for tag, X in (("lr", np.c_[za, zs]), ("lrsym", avg[:, None])):
        r = LogisticRegression(C=1e4, max_iter=200).fit(X[m], y[m])
        print(tag, tr_c, "->", te_c, r.coef_, r.intercept_, flush=True)
        z = X @ r.coef_[0] + r.intercept_[0]
        v = arms.setdefault(tag, arms["base"].copy())
        te = cn == te_c
        v[te] = np.where(both[te], sg(z[te]), single[te])
print("corr", {c: np.corrcoef(za[both & (cn == c)], zs[both & (cn == c)])[0, 1] for c in ("us", "india")}, {c: zs[both & (cn == c)].std() / za[both & (cn == c)].std() for c in ("us", "india")})
res = {}
for n, pr in arms.items():
    for fl in ((0.55, 0.6, 0.65) if n == "base" else (0.6,)):
        d = M.decide(u.select(K).with_columns(prob=pl.Series(pr)), cfg, floor=fl).select(K)
        e = M.per_entity_f05(d, gold, ents)
        nm = n if fl == 0.6 else f"{n}_f{fl}"
        res[nm] = e.select("s1_idx", "fold", "country_n", "f05")
        print(nm, {k: round(v, 5) for k, v in e.group_by("country_n").agg(pl.col("f05").mean()).sort("country_n").iter_rows()}, f"{time.time() - t:.0f}s", flush=True)
for b in [k for k in res if k != "base"]:
    j = res["base"].join(res[b].select("s1_idx", z="f05"), on="s1_idx").with_columns(d=pl.col("z") - pl.col("f05"))
    for c in ("us", "india"):
        d = j.filter(pl.col("country_n") == c).group_by("fold").agg(pl.col("d").mean()).sort("fold")["d"].to_numpy()
        se = d.std(ddof=1) / np.sqrt(5)
        print(f"PAIRED {b} vs base [{c}]: {d.mean():+.5f} SE {se:.5f} folds+ {(d > 0).sum()}/5 {'PASS' if d.mean() > 2 * se and (d > 0).sum() >= 4 else 'FAIL'}", flush=True)
