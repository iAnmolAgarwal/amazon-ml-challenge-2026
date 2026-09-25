"""Retrieval route: extra candidates from character retrieval (er_retrieve.py), scored by a separate pair classifier.

usage:
  python src/er_newpairs.py union <attempt_id> --split train|test   # new candidate file cache/cand/{split}_cand{NEW_TAG}.parquet
  python src/er_newpairs.py fit <attempt_id>                        # new-pair classifiers (5 fold models + 1 full model)
  python src/er_newpairs.py score <attempt_id> --split train|test   # cache/oof/[test_]<attempt_id>.parquet

The attempt sets OLD (attempt whose probabilities score the existing candidate graph), NEW_TAG and RET_TAG (per split:
new candidate-file tag and retrieval-file tag), NEW_BASE + NEW_R5 (feature names, read from feat / feat2 / feat5 side-cars built on the new candidate file with
the unchanged er_feat.py, er_feat2.py, er_feat5.py and --tag NEW_TAG), NEW_PARAMS, NEW_ROUNDS, NEW_FILTER, NEW_SAMPLE.
- union: pairs retrieved by the name or the address view that are not existing candidates, in the candidate-file
  format (word_sim = best retrieval similarity, key flags 0).
- fit: trained on EXISTING train candidate pairs of a label-independent owner sample (s1_idx hash, NEW_SAMPLE of S1)
  with their gold labels; no probability, rank or graph input. Fold models exclude their fold's owners.
- score: new pairs are scored by the classifier (train: the fold model that never saw the owner; test: the full
  model); pairs with p > NEW_FILTER are appended to OLD's probabilities, whose values are unchanged. The candidate set
  of the attempt is therefore OLD's candidates plus the kept new pairs.
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
KFLAGS = ["k_hs", "k_nj", "k_ns", "k_njh", "k_ih", "k_alt"]


def sample_mask(col, frac):
    return ((pl.col(col).cast(pl.UInt64) * 2654435761) % 4294967296) % 1000 < int(frac * 1000)


def union(cache, mod, split):
    t = time.time()
    views = []
    for v in ("name", "address"):
        files = sorted((cache / "cand").glob(f"{split}_ret_{v}{mod.RET_TAG[split]}*.parquet"))
        assert files, f"no {split} {v} retrieval files"
        views.append(pl.concat([pl.read_parquet(f, columns=KEYS + ["similarity"]) for f in files]))
    e = pl.concat(views).group_by(KEYS).agg(pl.col("similarity").max().alias("word_sim"))
    old = pl.scan_parquet(cache / "cand" / f"{split}_cand.parquet").select(KEYS)
    new = e.lazy().join(old, on=KEYS, how="anti").collect()
    s1c = pl.read_parquet(cache / "norm" / f"{split}_s1.parquet", columns=["country_n"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
    new = new.join(s1c, on="s1_idx", how="left").with_columns([pl.lit(0, pl.UInt8).alias(k) for k in KFLAGS]).with_columns(pl.col("word_sim").cast(pl.Float32))
    new = new.select(KEYS + ["word_sim"] + KFLAGS + ["country_n"]).sort(KEYS)
    path = cache / "cand" / f"{split}_cand{mod.NEW_TAG[split]}.parquet"
    new.write_parquet(path)
    print(f"{split}: {len(e):,} retrieved pairs, {len(new):,} new ({len(new) / max(new['s1_idx'].n_unique(), 1):.1f} per S1 with new pairs) -> {path.name}, {time.time() - t:.0f}s")


def matrix(cache, split, tag, cols_base, cols_r5, keys=None, rng=None):
    b = pl.scan_parquet(cache / "feat" / f"{split}_feat{tag}.parquet").select(KEYS + [c for c in cols_base if c not in ("nf_s1_s1", "nf_s1_pool", "nf_p_s1", "nf_p_pool", "c_ratio", "c_tset", "c_jw", "a_ratio_g", "a_partial_g")])
    b2 = pl.scan_parquet(cache / "feat" / f"{split}_feat2{tag}.parquet").select(KEYS + [c for c in cols_base if c in ("nf_s1_s1", "nf_s1_pool", "nf_p_s1", "nf_p_pool", "c_ratio", "c_tset", "c_jw", "a_ratio_g", "a_partial_g")])
    b5 = pl.scan_parquet(cache / "feat" / f"{split}_feat5{tag}.parquet").select(KEYS + cols_r5)
    if keys is not None:
        b, b2, b5 = (x.join(keys.lazy(), on=KEYS, how="semi") for x in (b, b2, b5))
    if rng is not None:
        b, b2, b5 = (x.filter(pl.col("s1_idx").is_between(rng[0], rng[1], closed="left")) for x in (b, b2, b5))
    d = b.collect(engine="streaming").join(b2.collect(engine="streaming"), on=KEYS, how="left").join(b5.collect(engine="streaming"), on=KEYS, how="left").sort(KEYS)
    cols = cols_base + cols_r5
    assert d.select(pl.any_horizontal(pl.col(cols).is_null()).any()).item() is False, "missing new-pair features"
    X = np.empty((len(d), len(cols)), np.float32)
    for j, c in enumerate(cols):
        X[:, j] = d[c].cast(pl.Float32).to_numpy()
    return d.select(KEYS), X, cols


def fit(cache, cfg, mod, attempt):
    t = time.time()
    s1 = M.load_s1(cfg, "train")
    gold = M.load_gold(cfg, s1)
    fo = M.folds(cfg, s1, gold)
    owners = fo.filter(sample_mask("s1_idx", mod.NEW_SAMPLE)).select("s1_idx", "fold")
    keys = pl.scan_parquet(cache / "cand" / "train_cand.parquet").select(KEYS).join(owners.lazy().select("s1_idx"), on="s1_idx", how="semi").collect()
    k, X, cols = matrix(cache, "train", "", mod.NEW_BASE, mod.NEW_R5, keys)
    lab = k.join(owners, on="s1_idx", how="left", maintain_order="left").join(gold.with_columns(y=pl.lit(1, pl.Int8)), on=KEYS, how="left", maintain_order="left")
    y, fold = lab["y"].fill_null(0).to_numpy(), lab["fold"].to_numpy()
    print(f"fit: {len(y):,} existing pairs of {owners.height:,} sampled owners, {int(y.sum()):,} positives, {len(cols)} features, {time.time() - t:.0f}s", flush=True)
    out = cache / "models" / attempt
    out.mkdir(parents=True, exist_ok=True)
    for f in [*sorted(np.unique(fold)), "all"]:
        tr = np.arange(len(y)) if f == "all" else np.flatnonzero(fold != f)
        m = lgb.train(mod.NEW_PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=cols, free_raw_data=True), mod.NEW_ROUNDS)
        m.save_model(out / f"newpair_{f}.txt")
        print(f"  model {f}: {len(tr):,} pairs, {time.time() - t:.0f}s", flush=True)


def score(cache, cfg, mod, attempt, split, blocks=12):
    t = time.time()
    mdir = cache / "models" / attempt
    fold_of = None
    if split == "train":
        s1 = M.load_s1(cfg, "train")
        fold_of = M.folds(cfg, s1, M.load_gold(cfg, s1)).select(pl.col("s1_idx").cast(pl.UInt32), "fold")
    models = {f: lgb.Booster(model_file=str(mdir / f"newpair_{f}.txt")) for f in ([0, 1, 2, 3, 4] if split == "train" else ["all"])}
    n_s1 = pl.scan_parquet(cache / "norm" / f"{split}_s1.parquet").select(pl.len()).collect().item()
    edges = np.linspace(0, n_s1, blocks + 1).astype(np.int64)
    kept, scored = [], 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        k, X, cols = matrix(cache, split, mod.NEW_TAG[split], mod.NEW_BASE, mod.NEW_R5, rng=(int(lo), int(hi)))
        p = np.empty(len(k), np.float32)
        if split == "test":
            p[:] = models["all"].predict(X)
        else:
            fold = k.join(fold_of, on="s1_idx", how="left", maintain_order="left")["fold"].to_numpy()
            for f in range(5):
                i = np.flatnonzero(fold == f)
                if len(i):
                    p[i] = models[f].predict(X[i])
        scored += len(k)
        kept.append(k.with_columns(prob=pl.Series(p)).filter(pl.col("prob") > mod.NEW_FILTER))
        del X
        print(f"  S1 {lo:,}-{hi:,}: {len(k):,} new pairs, {time.time() - t:.0f}s", flush=True)
    new = pl.concat(kept)
    k = new
    old_name = mod.OLD if split == "train" else f"test_{mod.OLD}"
    old = pl.read_parquet(cache / "oof" / f"{old_name}.parquet", columns=KEYS + ["prob"])
    assert new.join(old, on=KEYS, how="semi").is_empty(), "new pairs overlap the existing candidates"
    out = pl.concat([old.with_columns(pl.col("prob").cast(pl.Float32)), new.with_columns(pl.col("prob").cast(pl.Float32))]).sort(KEYS)
    name = attempt if split == "train" else f"test_{attempt}"
    out.write_parquet(cache / "oof" / f"{name}.parquet")
    print(f"{split}: scored {scored:,} new pairs, kept {len(new):,} with p > {mod.NEW_FILTER} ({len(new) / out['s1_idx'].n_unique():.2f} per S1), above 0.6: {int((new['prob'] > 0.6).sum()):,}; wrote {name}.parquet ({len(out):,} pairs), {time.time() - t:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["union", "fit", "score"])
    ap.add_argument("attempt_id")
    ap.add_argument("--split", choices=["train", "test"], default="test")
    a = ap.parse_args()
    cfg = load_config()
    cache = Path(cfg["cache_dir"])
    mod = load_attempt(a.attempt_id)
    {"union": lambda: union(cache, mod, a.split), "fit": lambda: fit(cache, cfg, mod, a.attempt_id), "score": lambda: score(cache, cfg, mod, a.attempt_id, a.split)}[a.mode]()


if __name__ == "__main__":
    main()
