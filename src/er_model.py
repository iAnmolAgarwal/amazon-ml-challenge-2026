"""Model, decision rule and scoring shared by er_eval.py (CV) and er_predict.py (test inference).

Attempts (attempts/<id>/train.py) may override any of: FEATURES, PARAMS, ROUNDS, TRAIN_FRAC,
fit(X, y, cfg) -> model, predict(model, X) -> probs, decide(pairs, cfg) -> pairs kept.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from common import load_config

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=42)
ROUNDS = 600
TRAIN_FRAC = 0.2
FLOOR = 0.0
EMPTY_RATIO = 1.0
ID_COLS = ("s1_idx", "pool_idx", "y", "fold", "country_n", "prob")


def feature_cols(df, wanted=None):
    """All numeric feature columns (never ids, labels, folds or country)."""
    cols = [c for c in df.columns if c not in ID_COLS]
    return [c for c in (wanted or cols) if c in cols]


def to_matrix(df, cols):
    """Build a C-contiguous float32 matrix column by column, dropping each column from df as it goes (bounded peak memory)."""
    X = np.empty((len(df), len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        X[:, j] = df[c].cast(pl.Float32).to_numpy()
        df = df.drop(c)
    return X


def full_dataset(X, y, params=None):
    """Bin all rows once (label-free feature quantiles); folds then train on zero-copy subsets (free_raw_data=True: subsets never copy raw rows)."""
    return lgb.Dataset(X, y, params=dict(params or PARAMS), free_raw_data=True).construct()


def fit_subset(full, idx, params=None, rounds=None):
    return lgb.train(params or PARAMS, full.subset(np.sort(idx).astype(np.int32)), rounds or ROUNDS)


def predict_chunked(model, X, idx, chunk=4_000_000):
    out = np.empty(len(idx), dtype=np.float32)
    for s in range(0, len(idx), chunk):
        out[s : s + chunk] = model.predict(X[idx[s : s + chunk]], num_threads=8)
    return out


def fit(X, y, cfg, params=None, rounds=None):
    return lgb.train(params or PARAMS, lgb.Dataset(X, y, free_raw_data=True), rounds or ROUNDS)


def predict(model, X):
    return model.predict(X, num_threads=8).astype(np.float32)


def decide(pairs, cfg=None, floor=0.0, empty_ratio=1.0):
    """Assignment + per-entity k selection maximizing expected F0.5, then pairs with prob <= floor are dropped.

    An S1 gets any prediction only if its best expected F0.5 exceeds empty_ratio x the probability that it has no match.

    pairs: s1_idx, pool_idx, prob. Each pool record is kept only for its highest-probability S1
    (ties -> lowest s1_idx). For each S1 with kept probs p1>=...>=pn, E[T] = sum(p);
    k>=1 scores 5*sum(p_1..p_k)/(E[T]+4k); k=0 scores prod(1-p). The best k is chosen.
    """
    d = pairs.select("s1_idx", "pool_idx", "prob").sort(["pool_idx", "prob", "s1_idx"], descending=[False, True, False])
    d = d.unique("pool_idx", keep="first", maintain_order=True)
    d = d.sort(["s1_idx", "prob", "pool_idx"], descending=[False, True, False]).with_columns(
        k=pl.int_range(1, pl.len() + 1).over("s1_idx"),
        cum=pl.col("prob").cum_sum().over("s1_idx"),
        et=pl.col("prob").sum().over("s1_idx"),
        p0=(1 - pl.col("prob")).log().sum().over("s1_idx").exp(),
    ).with_columns(score=5 * pl.col("cum") / (pl.col("et") + 4 * pl.col("k")))
    best = d.sort(["s1_idx", "score", "k"], descending=[False, True, False]).group_by("s1_idx", maintain_order=True).agg(pl.col("score").first().alias("best"), pl.col("k").first().alias("kbest"), pl.col("p0").first())
    keep = best.filter(pl.col("best") > empty_ratio * pl.col("p0")).select("s1_idx", "kbest")
    return d.join(keep, on="s1_idx").filter((pl.col("k") <= pl.col("kbest")) & (pl.col("prob") > floor)).select("s1_idx", "pool_idx", "prob")


def per_entity_f05(pred, gold, entities):
    """Macro F0.5 inputs: per-entity score for every S1 in `entities` (s1_idx), given predicted and gold pairs."""
    tp = pred.join(gold, on=["s1_idx", "pool_idx"], how="semi").group_by("s1_idx").len().rename({"len": "tp"})
    kp = pred.group_by("s1_idx").len().rename({"len": "k"})
    tt = gold.group_by("s1_idx").len().rename({"len": "t"})
    e = entities.join(tt, on="s1_idx", how="left").join(kp, on="s1_idx", how="left").join(tp, on="s1_idx", how="left").fill_null(0)
    return e.with_columns(
        f05=pl.when(pl.col("t") == 0).then((pl.col("k") == 0).cast(pl.Float64)).when((pl.col("k") == 0) | (pl.col("tp") == 0)).then(0.0).otherwise(5 * pl.col("tp") / (pl.col("t") + 4 * pl.col("k")))
    )


def load_gold(cfg, s1):
    """Gold (s1_idx, pool_idx) pairs for the train split, using the cached normalized id order."""
    cache = Path(cfg["cache_dir"])
    pool = pl.read_parquet(cache / "norm" / "train_pool.parquet", columns=["entity_id"]).with_row_index("pool_idx")
    gt = pl.read_parquet(cache / "raw" / "train_ground_truth.parquet")
    g = gt.select(pl.col("source1_entity_id").alias("entity_id"), pl.col("matched_entity_ids").fill_null("").str.split(",").alias("m")).explode("m").filter(pl.col("m") != "")
    g = g.join(s1.select("entity_id", "s1_idx"), on="entity_id").join(pool.select(pl.col("entity_id").alias("m"), "pool_idx"), on="m")
    return g.select(pl.col("s1_idx").cast(pl.UInt32), pl.col("pool_idx").cast(pl.UInt32))


def load_s1(cfg, split):
    return pl.read_parquet(Path(cfg["cache_dir"]) / "norm" / f"{split}_s1.parquet", columns=["entity_id", "country_n"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))


def folds(cfg, s1, gold, n=5, seed=42):
    """Stratified folds over S1 entities by country x match count (0,1,2,3,4+); cached in data/folds.parquet."""
    path = Path(cfg["data_dir"]) / "folds.parquet"
    if path.exists():
        f = pl.read_parquet(path)
        if len(f) == len(s1) and f["entity_id"].equals(s1.sort("s1_idx")["entity_id"]):
            return f
        print("folds.parquet does not match the current S1 table: rebuilding", flush=True)
    from sklearn.model_selection import StratifiedKFold

    cnt = s1.join(gold.group_by("s1_idx").len(), on="s1_idx", how="left").fill_null(0)
    strat = (cnt["country_n"] + "_" + cnt["len"].clip(0, 4).cast(pl.Utf8)).to_numpy()
    f = np.zeros(len(cnt), dtype=np.int8)
    for i, (_, va) in enumerate(StratifiedKFold(n, shuffle=True, random_state=seed).split(strat, strat)):
        f[va] = i
    out = cnt.select("s1_idx", "entity_id").with_columns(fold=pl.Series(f))
    out.write_parquet(path)
    return out
