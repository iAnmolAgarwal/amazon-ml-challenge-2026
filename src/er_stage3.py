"""Stage 3: a correction of a parent attempt's probabilities with extra pair features.

usage: python src/er_stage3.py <attempt_id> [--tag ""] [--no-cv]

The attempt sets PARENT (attempt whose probabilities are corrected), FEATS3 (side-car feature files, e.g. ["feat5"]),
BAND (only pairs with parent probability >= BAND are re-scored; the rest keep it), PARAMS3 and ROUNDS3.
The correction is a LightGBM boosted from logit(parent p) (init_score) on [logit p] + the side-car features.
Optional: INIT = False trains without the init score, USE_LOGIT = False drops the logit p column (together: a fresh
model on the band, e.g. a band stage 2 with every feature set), EXCLUDE3 lists feature columns to leave out.
- train: 5-fold out-of-fold on the harness folds (every pair scored by a model that never saw its S1), using the
  parent's out-of-fold probabilities cache/oof/<PARENT>.parquet -> cache/oof/<attempt>.parquet
- test: one model on all train pairs, applied to cache/oof/test_<PARENT>.parquet -> cache/oof/test_<attempt>.parquet
Then `er_eval.py <attempt> --from-oof <attempt>` scores the CV and `er_predict.py <attempt> --from-probs <attempt>`
writes the submission files.
"""
import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

import er_model as M
from common import load_attempt, load_config

KEYS = ["s1_idx", "pool_idx"]


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def band_matrix(cache, split, pairs, feats, band, tag, exclude=(), use_logit=True):
    """Pairs with p >= band, their logit and [logit, side-car features] as float32, in band order."""
    b = pairs.filter(pl.col("prob") >= band)
    cols = []
    for f in feats:
        x = pl.scan_parquet(cache / "feat" / f"{split}_{f}{tag}.parquet").join(b.lazy().select(KEYS), on=KEYS, how="semi").collect()
        x = x.drop([c for c in exclude if c in x.columns])
        cols += [c for c in x.columns if c not in KEYS]
        b = b.join(x, on=KEYS, how="left")
    assert b.select(pl.col(cols[0]).null_count()).item() == 0, f"{split}: band pairs missing side-car features"
    lp = logit(b["prob"].to_numpy())
    o = 1 if use_logit else 0
    X = np.empty((len(b), o + len(cols)), dtype=np.float32)
    if use_logit:
        X[:, 0] = lp
    for j, c in enumerate(cols):
        X[:, j + o] = b[c].cast(pl.Float32).to_numpy()
    return b.select(KEYS), lp, X, (["logit_p"] if use_logit else []) + cols


def merge(pairs, keys, p):
    upd = keys.with_columns(p3=pl.Series(np.asarray(p, np.float32)))
    return pairs.join(upd, on=KEYS, how="left", maintain_order="left").with_columns(prob=pl.coalesce("p3", "prob").cast(pl.Float32)).select(KEYS + ["prob"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attempt_id")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-cv", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    mod = load_attempt(a.attempt_id)
    cache = Path(cfg["cache_dir"])
    t = time.time()
    s1 = M.load_s1(cfg, "train")
    gold = M.load_gold(cfg, s1)
    fo = M.folds(cfg, s1, gold)
    oof = pl.read_parquet(cache / "oof" / f"{mod.PARENT}{a.tag}.parquet", columns=KEYS + ["prob"])
    init, ul, ex = getattr(mod, "INIT", True), getattr(mod, "USE_LOGIT", True), getattr(mod, "EXCLUDE3", [])
    keys, lp, X, names = band_matrix(cache, "train", oof, mod.FEATS3, mod.BAND, a.tag, ex, ul)
    lab = keys.join(fo.select("s1_idx", "fold"), on="s1_idx", how="left", maintain_order="left").join(gold.with_columns(y=pl.lit(1, pl.Int8)), on=KEYS, how="left", maintain_order="left")
    y, fold = lab["y"].fill_null(0).to_numpy(), lab["fold"].to_numpy()
    print(f"train: {len(keys):,} of {len(oof):,} pairs in the band (p >= {mod.BAND}), {int(y.sum()):,} positives, {len(names)} features, {time.time() - t:.0f}s", flush=True)
    if not a.no_cv:
        out = np.empty(len(y))
        for f in sorted(np.unique(fold)):
            tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            m = lgb.train(mod.PARAMS3, lgb.Dataset(X[tr], y[tr], init_score=lp[tr] if init else None, feature_name=names, free_raw_data=True), mod.ROUNDS3)
            out[va] = 1 / (1 + np.exp(-((lp[va] if init else 0) + m.predict(X[va], raw_score=True))))
            print(f"fold {f}: {len(tr):,} train pairs, {time.time() - t:.0f}s", flush=True)
        merge(oof, keys, out).write_parquet(cache / "oof" / f"{a.attempt_id}{a.tag}.parquet")
    m = lgb.train(mod.PARAMS3, lgb.Dataset(X, y, init_score=lp if init else None, feature_name=names, free_raw_data=True), mod.ROUNDS3)
    del X
    tst = pl.read_parquet(cache / "oof" / f"test_{mod.PARENT}{a.tag}.parquet", columns=KEYS + ["prob"])
    tkeys, tlp, TX, tnames = band_matrix(cache, "test", tst, mod.FEATS3, mod.BAND, a.tag, ex, ul)
    assert tnames == names
    tp = 1 / (1 + np.exp(-((tlp if init else 0) + m.predict(TX, raw_score=True))))
    merge(tst, tkeys, tp).write_parquet(cache / "oof" / f"test_{a.attempt_id}{a.tag}.parquet")
    imp = sorted(zip(names, m.feature_importance("gain")), key=lambda x: -x[1])[:10]
    print(f"test: {len(tkeys):,} of {len(tst):,} pairs re-scored; top gain: " + ", ".join(f"{n} {g:.0f}" for n, g in imp) + f"; {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
