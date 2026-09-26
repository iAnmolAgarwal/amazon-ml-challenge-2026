"""Combined final correction: base band features + generator-operator features (er_gen.py) + pool-side competition
features (er_pool.py), boosted from logit(parent p), for US/India pairs with lo <= p <= hi.

usage: python src/er_comb.py <attempt_id> [--no-cv]

The attempt sets PARENT, BAND (lo, hi), FEATSC (side-car feature files with base features), EXCLUDEC, PC_COLS,
PARAMS_A / ROUNDS_A (model on pairs with base features) and PARAMS_B / ROUNDS_B (model without base features, used for
band pairs that have none, i.e. pairs added by the retrieval route). France and pairs outside the band keep p.
Needs cache/feat/{split}_gen.parquet built by `er_gen.py --split <split> --parent <PARENT>`.
Train: 5-fold out-of-fold on the harness folds -> cache/oof/<attempt>.parquet; test -> cache/oof/test_<attempt>.parquet.
"""
import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

import er_model as M
from common import load_attempt, load_config
from er_pool import KEYS, PR, contrasts, merge, pool_side


def build(cache, split, pairs, mod, country):
    lo, hi = mod.BAND
    b = pool_side(pairs).filter((pl.col("prob") >= lo) & (pl.col("prob") <= hi))
    b = b.join(country, on="s1_idx", how="left").filter(pl.col("country_n") != "france").with_columns(india=(pl.col("country_n") == "india").cast(pl.Float32))
    gen = pl.read_parquet(cache / "feat" / f"{split}_gen.parquet").join(b.select(KEYS), on=KEYS, how="semi")
    b = b.join(gen, on=KEYS, how="left")
    assert b["nA"].null_count() == 0, f"{split}: band pairs missing generator features (rebuild er_gen for this parent)"
    gcols = [c for c in gen.columns if c not in KEYS]
    base = []
    for f in mod.FEATSC:
        x = pl.scan_parquet(cache / "feat" / f"{split}_{f}.parquet").join(b.lazy().select(KEYS), on=KEYS, how="semi").collect()
        x = x.drop([c for c in mod.EXCLUDEC if c in x.columns]).with_columns(pl.lit(1, pl.Int8).alias(f"_has_{f}"))
        base += [c for c in x.columns if c not in KEYS and not c.startswith("_has_")]
        b = b.join(x, on=KEYS, how="left")
    has = pl.all_horizontal([pl.col(f"_has_{f}").is_not_null() for f in mod.FEATSC])
    b = b.with_columns(has=has).drop([f"_has_{f}" for f in mod.FEATSC])
    pc = contrasts(cache, split, mod.FEATSC, mod.PC_COLS, b.filter(pl.col("has")).select(KEYS))
    b = b.join(pc, on=KEYS, how="left")
    pcols = [c for c in pc.columns if c not in KEYS]
    p = np.clip(b["prob"].to_numpy().astype(np.float64), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    cols_b = PR + ["india"] + gcols
    cols_a = cols_b + base + pcols
    XA = np.column_stack([lp] + [b[c].cast(pl.Float32).fill_null(np.nan).to_numpy() for c in cols_a]).astype(np.float32)
    return b.select(KEYS + ["has"]), lp, XA, ["lp"] + cols_a, len(cols_b) + 1


def fit_predict(params, rounds, X, y, lp, names, tr, va):
    m = lgb.train(params, lgb.Dataset(X[tr], y[tr], init_score=lp[tr], feature_name=names, free_raw_data=True), rounds)
    return m, 1 / (1 + np.exp(-(lp[va] + m.predict(X[va], raw_score=True))))


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
    fo = M.folds(cfg, s1, gold)
    oof = pl.read_parquet(cache / "oof" / f"{mod.PARENT}.parquet", columns=KEYS + ["prob"])
    keys, lp, X, names, nb = build(cache, "train", oof, mod, s1.select("s1_idx", "country_n"))
    lab = keys.join(fo.select("s1_idx", "fold"), on="s1_idx", how="left", maintain_order="left").join(gold.with_columns(y=pl.lit(1, pl.Int8)), on=KEYS, how="left", maintain_order="left")
    y, fold, has = lab["y"].fill_null(0).to_numpy(), lab["fold"].to_numpy(), keys["has"].to_numpy()
    XB, nB = X[:, :nb], names[:nb]
    print(f"train: {len(keys):,} band pairs ({has.sum():,} with base features), {int(y.sum()):,} positives, {len(names)} / {nb} features, {time.time() - t:.0f}s", flush=True)
    if not a.no_cv:
        out = np.empty(len(y))
        for f in sorted(np.unique(fold)):
            trA, vaA = np.flatnonzero((fold != f) & has), np.flatnonzero((fold == f) & has)
            trB, vaB = np.flatnonzero(fold != f), np.flatnonzero((fold == f) & ~has)
            _, out[vaA] = fit_predict(mod.PARAMS_A, mod.ROUNDS_A, X, y, lp, names, trA, vaA)
            if len(vaB):
                _, out[vaB] = fit_predict(mod.PARAMS_B, mod.ROUNDS_B, XB, y, lp, nB, trB, vaB)
            print(f"fold {f}: {len(trA):,} / {len(trB):,} train pairs, {time.time() - t:.0f}s", flush=True)
        merge(oof, keys, out).write_parquet(cache / "oof" / f"{a.attempt_id}.parquet")
    allA, allB = np.flatnonzero(has), np.arange(len(y))
    mA = lgb.train(mod.PARAMS_A, lgb.Dataset(X[allA], y[allA], init_score=lp[allA], feature_name=names, free_raw_data=True), mod.ROUNDS_A)
    mB = lgb.train(mod.PARAMS_B, lgb.Dataset(XB, y, init_score=lp, feature_name=nB, free_raw_data=True), mod.ROUNDS_B)
    del X, XB
    tst = pl.read_parquet(cache / "oof" / f"test_{mod.PARENT}.parquet", columns=KEYS + ["prob"])
    tkeys, tlp, TX, tnames, tnb = build(cache, "test", tst, mod, M.load_s1(cfg, "test").select("s1_idx", "country_n"))
    assert tnames == names and tnb == nb
    th = tkeys["has"].to_numpy()
    tp = np.empty(len(tlp))
    tp[th] = 1 / (1 + np.exp(-(tlp[th] + mA.predict(TX[th], raw_score=True))))
    tp[~th] = 1 / (1 + np.exp(-(tlp[~th] + mB.predict(TX[~th][:, :nb], raw_score=True))))
    merge(tst, tkeys, tp).write_parquet(cache / "oof" / f"test_{a.attempt_id}.parquet")
    imp = sorted(zip(names, mA.feature_importance("gain")), key=lambda x: -x[1])[:12]
    print(f"test: {len(tkeys):,} of {len(tst):,} pairs re-scored ({th.sum():,} with base features); top gain: " + ", ".join(f"{n} {g:.0f}" for n, g in imp) + f"; {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
