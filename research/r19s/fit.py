import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import er_model as M
from common import load_config

K = ["s1_idx", "pool_idx"]
cfg = load_config()
cache = Path(cfg["cache_dir"])
d = Path(__import__("os").environ.get("R19D", "cache/r19s"))
name, learner, drop, rounds = sys.argv[1], sys.argv[2], sys.argv[3].split(",") if sys.argv[3] != "-" else [], int(sys.argv[4])
extra = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
names = json.load(open(d / "names.json"))


def group(c):
    if c.endswith("_gen"):
        return "gen"
    for g in ["gen", "ce", "feat4"]:
        if c in gcols[g]:
            return g
    if c.startswith("pr_") or c.startswith("pc_"):
        return "comp"
    return "p006" if c == "p006" else "base"


gp = d / "groups.json"
if not gp.exists():
    json.dump({g: sorted(c + "_" + g if c in ("nA", "nB", "acronym") else c for c in pl.read_parquet_schema(cache / "feat" / f"train_{g}.parquet") if c not in K) for g in ["gen", "ce", "feat4"]}, open(gp, "w"))
gcols = {g: set(v) for g, v in json.load(open(gp)).items()}
drop = set(drop) | {"p006"}
keep = extra.pop("cols", None)
use = [j for j, c in enumerate(names) if (c in keep if keep else group(c) not in drop)]
cols = [names[j] for j in use]
print(name, learner, len(cols), "features", {g: sum(group(c) == g for c in cols) for g in ["base", "feat4", "comp", "gen", "ce"]}, flush=True)
t = time.time()
Xa = np.load(d / "train_X.npy", mmap_mode="r")
X = np.ascontiguousarray(Xa[:, use]) if len(use) < len(names) else np.asarray(Xa)
keys = pl.read_parquet(d / "train_keys.parquet")
s1 = M.load_s1(cfg, "train")
gold = M.load_gold(cfg, s1)
fo = M.folds(cfg, s1, gold)
lab = keys.join(fo.select("s1_idx", "fold"), on="s1_idx", how="left", maintain_order="left").join(gold.with_columns(y=pl.lit(1, pl.Int8)), on=K, how="left", maintain_order="left")
y, fold = lab["y"].fill_null(0).to_numpy().astype(np.float32), lab["fold"].to_numpy()
print(f"loaded {X.shape} {time.time() - t:.0f}s", flush=True)
oof = np.empty(len(y))
models = []
if learner == "lgb":
    import lightgbm as lgb
    P = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, seed=42, num_threads=12, verbose=-1, max_bin=255)
    P.update(extra)
    full = lgb.Dataset(X, y, feature_name=cols, free_raw_data=False, params={"max_bin": P["max_bin"]}).construct()
    for f in range(5):
        tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        m = lgb.train(P, full.subset(tr), rounds)
        oof[va] = m.predict(X[va], raw_score=True, num_threads=12)
        models.append(m)
        print(f"fold {f} {time.time() - t:.0f}s", flush=True)
    imp = sorted(zip(cols, models[0].feature_importance("gain")), key=lambda x: -x[1])
    print("top gain", [(c, round(g / sum(v for _, v in imp), 4)) for c, g in imp[:25]], flush=True)
elif learner == "xgb":
    import xgboost as xgb
    P = dict(objective="binary:logistic", eta=0.08, max_depth=10, min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, tree_method="hist", max_bin=128, nthread=12, eval_metric="logloss", grow_policy="lossguide", max_leaves=255)
    P.update(extra)
    for f in range(5):
        tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        dm = xgb.QuantileDMatrix(X[tr], y[tr], max_bin=P["max_bin"])
        m = xgb.train(P, dm, rounds)
        del dm
        oof[va] = m.predict(xgb.DMatrix(X[va]), output_margin=True)
        models.append(m)
        print(f"fold {f} {time.time() - t:.0f}s", flush=True)
elif learner == "logit":
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import KBinsDiscretizer
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    for f in range(5):
        tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        sub = tr[np.random.default_rng(f).random(len(tr)) < extra.get("frac", 0.3)]
        m = make_pipeline(SimpleImputer(strategy="constant", fill_value=-9, add_indicator=True), KBinsDiscretizer(n_bins=extra.get("bins", 16), encode="onehot", strategy="quantile", subsample=200000), LogisticRegression(C=extra.get("C", 1.0), max_iter=300, solver="lbfgs"))
        m.fit(X[sub], y[sub])
        oof[va] = m.decision_function(X[va])
        models.append(m)
        print(f"fold {f} {time.time() - t:.0f}s", flush=True)
np.save(d / f"oof_{name}.npy", oof.astype(np.float32))
band = keys.with_columns(p3=pl.Series((1 / (1 + np.exp(-oof))).astype(np.float32)))
base = pl.read_parquet(cache / "oof" / (__import__("os").environ.get("R19B", "006_stage2") + ".parquet"), columns=K + ["prob"])
base.join(band, on=K, how="left", maintain_order="left").with_columns(prob=pl.coalesce("p3", "prob").cast(pl.Float32)).select(K + ["prob"]).write_parquet(cache / "oof" / f"r19_{name}.parquet")
ents = s1.select("s1_idx", "country_n").join(fo.select("s1_idx", "fold"), on="s1_idx")
pr = pl.read_parquet(cache / "oof" / f"r19_{name}.parquet")
e = M.per_entity_f05(M.decide(pr, cfg, floor=0.6).select(K), gold, ents)
print(f"CV {name}: {e['f05'].mean():.5f}", {k: round(v, 5) for k, v in e.group_by("country_n").agg(pl.col("f05").mean()).iter_rows()}, f"{time.time() - t:.0f}s", flush=True)
del X, Xa
full = None
Ta = np.load(d / "test_X.npy", mmap_mode="r")
TX = np.ascontiguousarray(Ta[:, use])
if learner == "lgb":
    z = np.mean([m.predict(TX, raw_score=True, num_threads=12) for m in models], axis=0)
elif learner == "xgb":
    import xgboost as xgb
    dt = xgb.DMatrix(TX)
    z = np.mean([m.predict(dt, output_margin=True) for m in models], axis=0)
else:
    z = np.mean([m.decision_function(TX) for m in models], axis=0)
np.save(d / f"test_{name}.npy", z.astype(np.float32))
tk = pl.read_parquet(d / "test_keys.parquet").with_columns(p3=pl.Series((1 / (1 + np.exp(-z))).astype(np.float32)))
tb = pl.read_parquet(cache / "oof" / ("test_" + __import__("os").environ.get("R19T", "006_stage2") + ".parquet"), columns=K + ["prob"])
tb.join(tk, on=K, how="left", maintain_order="left").with_columns(prob=pl.coalesce("p3", "prob").cast(pl.Float32)).select(K + ["prob"]).write_parquet(cache / "oof" / f"test_r19_{name}.parquet")
print(f"done {name} {time.time() - t:.0f}s", flush=True)
