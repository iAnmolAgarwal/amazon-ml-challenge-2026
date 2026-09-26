"""Pool-side second round: a correction of a parent attempt's probabilities with competition features recomputed from
those probabilities and pool-side feature contrasts.

usage: python src/er_pool.py <attempt_id> [--no-cv]

The attempt sets PARENT, BAND (lo, hi), FEATSP (side-car feature files), EXCLUDEP, PC_COLS, PARAMSP, ROUNDSP, FRANCE.
For every pair (s, x) with parent probability p, over all pairs of the split:
- pr_rank (rank of p among the S1 that claim pool record x, capped at 20), pr_other (best p of another S1 for x, 0 if
  none), pr_margin = p - pr_other, pr_sum (sum of p of the other S1), pr_n01 / pr_n50 (S1 of x with p > 0.01 / 0.5);
- pc_c = c - (best c among the other S1 of x) for each column c in PC_COLS (missing values count as -1).
Only the pool side is used: nothing compares a pair with the S1's other candidates.
A LightGBM boosted from logit(p) re-scores pairs with lo <= p <= hi that have rows in the feature files; other pairs
keep p. Train: 5-fold out-of-fold on the harness folds -> cache/oof/<attempt>.parquet. Test: one model on all train
band pairs -> cache/oof/test_<attempt>.parquet; France pairs take min(p, p_new) when FRANCE == "min".
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
PR = ["pr_rank", "pr_other", "pr_margin", "pr_sum", "pr_n01", "pr_n50"]


def other_best(df, col):
    top = pl.col(col).max().over("pool_idx")
    ntop = (pl.col(col) == top).sum().over("pool_idx")
    second = pl.when(pl.col(col) < top).then(pl.col(col)).otherwise(None).max().over("pool_idx")
    return pl.when((pl.col(col) == top) & (ntop == 1)).then(second).otherwise(top)


def pool_side(pairs):
    p = pl.col("prob")
    return pairs.with_columns(
        pr_rank=p.rank("min", descending=True).over("pool_idx").clip(upper_bound=20).cast(pl.Float32),
        pr_other=other_best(pairs, "prob").fill_null(0.0).cast(pl.Float32),
        pr_sum=(p.sum().over("pool_idx") - p).cast(pl.Float32),
        pr_n01=(p > 0.01).sum().over("pool_idx").cast(pl.Float32),
        pr_n50=(p > 0.5).sum().over("pool_idx").cast(pl.Float32),
    ).with_columns(pr_margin=(p - pl.col("pr_other")).cast(pl.Float32))


def contrasts(cache, split, feats, cols, band_keys):
    have = {}
    for f in feats:
        names = [c for c in pl.read_parquet_schema(cache / "feat" / f"{split}_{f}.parquet") if c in cols]
        if names:
            have[f] = names
    out = band_keys
    for f, names in have.items():
        x = pl.read_parquet(cache / "feat" / f"{split}_{f}.parquet", columns=KEYS + names).with_columns([pl.col(c).cast(pl.Float32).fill_null(-1.0) for c in names])
        x = x.with_columns([(pl.col(c) - other_best(x, c).fill_null(-1.0)).alias("pc_" + c) for c in names]).select(KEYS + ["pc_" + c for c in names])
        out = out.join(x.join(band_keys, on=KEYS, how="semi"), on=KEYS, how="left")
        del x
    return out


def matrix(cache, split, pairs, mod, country):
    lo, hi = mod.BAND
    b = pool_side(pairs).filter((pl.col("prob") >= lo) & (pl.col("prob") <= hi))
    b = b.join(pl.scan_parquet(cache / "feat" / f"{split}_{mod.FEATSP[0]}.parquet").select(KEYS).collect(), on=KEYS, how="semi")
    b = b.join(country, on="s1_idx", how="left").with_columns(india=(pl.col("country_n") == "india").cast(pl.Float32))
    cols = PR + ["india"]
    for f in mod.FEATSP:
        x = pl.scan_parquet(cache / "feat" / f"{split}_{f}.parquet").join(b.lazy().select(KEYS), on=KEYS, how="semi").collect()
        x = x.drop([c for c in mod.EXCLUDEP if c in x.columns])
        cols += [c for c in x.columns if c not in KEYS]
        b = b.join(x, on=KEYS, how="left")
    pc = contrasts(cache, split, mod.FEATSP, mod.PC_COLS, b.select(KEYS))
    b = b.join(pc, on=KEYS, how="left")
    cols += [c for c in pc.columns if c not in KEYS]
    lp = np.log(np.clip(b["prob"].to_numpy().astype(np.float64), 1e-7, 1 - 1e-7) / (1 - np.clip(b["prob"].to_numpy().astype(np.float64), 1e-7, 1 - 1e-7)))
    X = np.empty((len(b), 1 + len(cols)), dtype=np.float32)
    X[:, 0] = lp
    for j, c in enumerate(cols):
        X[:, j + 1] = b[c].cast(pl.Float32).to_numpy()
    return b.select(KEYS + ["country_n"]), lp, X, ["lp"] + cols


def merge(pairs, keys, p):
    upd = keys.select(KEYS).with_columns(pn=pl.Series(np.asarray(p, np.float32)))
    return pairs.join(upd, on=KEYS, how="left", maintain_order="left").with_columns(prob=pl.coalesce("pn", "prob").cast(pl.Float32)).select(KEYS + ["prob"])


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
    ctr = s1.select("s1_idx", "country_n")
    oof = pl.read_parquet(cache / "oof" / f"{mod.PARENT}.parquet", columns=KEYS + ["prob"])
    keys, lp, X, names = matrix(cache, "train", oof, mod, ctr)
    lab = keys.join(fo.select("s1_idx", "fold"), on="s1_idx", how="left", maintain_order="left").join(gold.with_columns(y=pl.lit(1, pl.Int8)), on=KEYS, how="left", maintain_order="left")
    y, fold = lab["y"].fill_null(0).to_numpy(), lab["fold"].to_numpy()
    print(f"train: {len(keys):,} band pairs, {int(y.sum()):,} positives, {len(names)} features, {time.time() - t:.0f}s", flush=True)
    if not a.no_cv:
        out = np.empty(len(y))
        for f in sorted(np.unique(fold)):
            tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            m = lgb.train(mod.PARAMSP, lgb.Dataset(X[tr], y[tr], init_score=lp[tr], feature_name=names, free_raw_data=True), mod.ROUNDSP)
            out[va] = 1 / (1 + np.exp(-(lp[va] + m.predict(X[va], raw_score=True))))
            print(f"fold {f}: {len(tr):,} train pairs, {time.time() - t:.0f}s", flush=True)
        merge(oof, keys, out).write_parquet(cache / "oof" / f"{a.attempt_id}.parquet")
    m = lgb.train(mod.PARAMSP, lgb.Dataset(X, y, init_score=lp, feature_name=names, free_raw_data=True), mod.ROUNDSP)
    del X
    tst = pl.read_parquet(cache / "oof" / f"test_{mod.PARENT}.parquet", columns=KEYS + ["prob"])
    tctr = M.load_s1(cfg, "test").select("s1_idx", "country_n")
    tkeys, tlp, TX, tnames = matrix(cache, "test", tst, mod, tctr)
    assert tnames == names
    tp = 1 / (1 + np.exp(-(tlp + m.predict(TX, raw_score=True))))
    if getattr(mod, "FRANCE", "min") == "min":
        fr = (tkeys["country_n"] == "france").to_numpy()
        tp[fr] = np.minimum(tp[fr], 1 / (1 + np.exp(-tlp[fr])))
    merge(tst, tkeys, tp).write_parquet(cache / "oof" / f"test_{a.attempt_id}.parquet")
    imp = sorted(zip(names, m.feature_importance("gain")), key=lambda x: -x[1])[:10]
    print(f"test: {len(tkeys):,} of {len(tst):,} pairs re-scored; top gain: " + ", ".join(f"{n} {g:.0f}" for n, g in imp) + f"; {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
