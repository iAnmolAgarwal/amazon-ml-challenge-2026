"""Meta capacity sweep: one cross-fit per PARAMS override, OOF saved at several round counts.

usage: python research/r19/cap.py <base_attempt> <tag> <rounds,comma> key=value ...
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "src")
import er_meta as E
import er_model as M
from common import load_attempt, load_config

base, tag, rs = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
cfg = load_config()
mod = load_attempt(base)
P = dict(mod.PARAMS)
for kv in sys.argv[4:]:
    k, v = kv.split("=")
    P[k] = type(P.get(k, 1.0))(float(v)) if k in P else float(v)
cache = Path(cfg["cache_dir"])
s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1); fo = M.folds(cfg, s1, gold)
u, _ = E.build(cache, "train", mod)
u = u.join(fo.select("s1_idx", "fold"), on="s1_idx", how="left", maintain_order="left").join(gold.with_columns(y=pl.lit(1, pl.Int8)), on=E.KEYS, how="left", maintain_order="left")
names = [c for c in u.columns if c not in E.KEYS + ["pm", "ps", "country_n", "fold", "y"]]
X = u.select(names).to_numpy().astype(np.float32)
y, fold = u["y"].fill_null(0).to_numpy(), u["fold"].to_numpy()
print(tag, P, names, flush=True)
oof = {r: np.empty(len(y)) for r in rs}
for f in range(5):
    tr, va = fold != f, fold == f
    m = lgb.train(P, lgb.Dataset(X[tr], y[tr], feature_name=names, free_raw_data=True), max(rs))
    for r in rs:
        oof[r][va] = m.predict(X[va], raw_score=True, num_iteration=r)
    print("fold", f, flush=True)
    if f == 0:
        imp = sorted(zip(m.feature_importance("gain"), names), reverse=True)[:25]
        print([(n, round(g / 1e3)) for g, n in imp], flush=True)
for r in rs:
    u.select(E.KEYS).with_columns(prob=pl.Series((1 / (1 + np.exp(-oof[r]))).astype(np.float32))).write_parquet(cache / "oof" / f"m19_{tag}_{r}.parquet")
print("done", flush=True)
