"""Cross-encoder correction: boost band pairs from [logit(parent p), cross-encoder logits]; other pairs keep p.

usage: python src/er_ce.py <attempt_id> [--no-cv]

The attempt sets PARENT (train OOF and default test parent), TEST_PARENT (optional, e.g. a test-only extension of the
parent), CE_COLS (columns of cache/feat/{split}_ce.parquet), PARAMS and ROUNDS. The band is the key set of
cache/feat/{split}_ce.parquet (BAND_SRC prob in BAND, written by src/ce/ce_export.py band + collect); band pairs missing
from the parent are ignored. Train: 5-fold out-of-fold on the harness folds -> cache/oof/<attempt>.parquet;
test: one model on all train band pairs -> cache/oof/test_<attempt>.parquet. Rows and order equal the parent's.
"""
import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

import er_model as M
from common import load_attempt, load_config

K = ["s1_idx", "pool_idx"]


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def X(d, cols):
    return np.column_stack([lg(d["prob"].to_numpy())] + [d[c].to_numpy() for c in cols]).astype(np.float32)


def merge(parent, band, p):
    return parent.join(band.select(*K, pl.Series("pn", p.astype(np.float32))), on=K, how="left").select(*K, pl.coalesce("pn", "prob").cast(pl.Float32).alias("prob"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attempt_id")
    ap.add_argument("--no-cv", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    mod = load_attempt(a.attempt_id)
    cache = Path(cfg["cache_dir"])
    t = time.time()
    s1 = M.load_s1(cfg, "train")
    gold = M.load_gold(cfg, s1)
    fo = M.folds(cfg, s1, gold).select(pl.col("s1_idx").cast(pl.UInt32), "fold")
    par = pl.read_parquet(cache / "oof" / f"{mod.PARENT}.parquet", columns=K + ["prob"])
    b = pl.read_parquet(cache / "feat" / "train_ce.parquet").join(par, on=K).join(gold.with_columns(y=pl.lit(1, pl.Int8)), on=K, how="left").with_columns(pl.col("y").fill_null(0)).join(fo, on="s1_idx")
    Xa, y, f = X(b, mod.CE_COLS), b["y"].to_numpy(), b["fold"].to_numpy()
    print(f"train: {b.height:,} band pairs of {par.height:,}, {int(y.sum()):,} positives, features lp + {mod.CE_COLS}, {time.time() - t:.0f}s", flush=True)
    if not a.no_cv:
        out = np.zeros(b.height, dtype=np.float32)
        for i in range(5):
            m = lgb.train(mod.PARAMS, lgb.Dataset(Xa[f != i], y[f != i]), mod.ROUNDS)
            out[f == i] = m.predict(Xa[f == i])
            print(f"fold {i}: {int((f != i).sum()):,} train pairs, {time.time() - t:.0f}s", flush=True)
        merge(par, b, out).write_parquet(cache / "oof" / f"{a.attempt_id}.parquet")
    m = lgb.train(mod.PARAMS, lgb.Dataset(Xa, y), mod.ROUNDS)
    tpar = pl.read_parquet(cache / "oof" / f"test_{getattr(mod, 'TEST_PARENT', mod.PARENT)}.parquet", columns=K + ["prob"])
    tb = pl.read_parquet(cache / "feat" / "test_ce.parquet").join(tpar, on=K)
    merge(tpar, tb, m.predict(X(tb, mod.CE_COLS))).write_parquet(cache / "oof" / f"test_{a.attempt_id}.parquet")
    print(f"test: {tb.height:,} of {tpar.height:,} pairs re-scored; {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
