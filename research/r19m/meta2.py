"""Second pass: meta inputs plus the first-pass OOF logit and its pool-side competition, cross-fit on the same folds."""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "src")
import er_meta as E
import er_model as M
from common import load_attempt, load_config

att, first, tag = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = load_config()
mod = load_attempt(att)
cache = Path(cfg["cache_dir"])
s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1); fo = M.folds(cfg, s1, gold)
u, _ = E.build(cache, "train", mod)
u = u.join(pl.read_parquet(cache / "oof" / f"{first}.parquet").rename({"prob": "p1"}), on=E.KEYS, how="left")
u = E.pool_side(u, "p1").with_columns(l1=E.lg(pl.col("p1")).cast(pl.Float32))
u = u.join(fo.select("s1_idx", "fold"), on="s1_idx", how="left", maintain_order="left").join(gold.with_columns(y=pl.lit(1, pl.Int8)), on=E.KEYS, how="left", maintain_order="left")
names = [c for c in u.columns if c not in E.KEYS + ["pm", "ps", "p1", "country_n", "fold", "y"]]
X = u.select(names).to_numpy().astype(np.float32)
y, fold = u["y"].fill_null(0).to_numpy(), u["fold"].to_numpy()
oof = np.empty(len(y))
for f in range(5):
    tr, va = fold != f, fold == f
    m = lgb.train(mod.PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=names, free_raw_data=True), mod.ROUNDS)
    oof[va] = m.predict(X[va], raw_score=True)
    print("fold", f, flush=True)
u.select(E.KEYS).with_columns(prob=pl.Series((1 / (1 + np.exp(-oof))).astype(np.float32))).write_parquet(cache / "oof" / f"m19_{tag}.parquet")
print("done", flush=True)
