"""Two-stage LightGBM matcher trained with grouped out-of-fold predictions, memory-bounded (streams parquet row groups).

Stage 1: pairwise features -> p1.
Stage 2: pairwise features + competition/context features of p1 (within the S2/S3 record and within the S1 entity)
         -> p2. Context features are computed from OUT-OF-FOLD p1 on train, from the full-train model on test.
Folds are assigned per S1 entity (stable hash of the entity id), so an entity's pairs never straddle folds.
"""
import os
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

META = ["a_id", "r_id", "a", "r", "y"]
EXCLUDE = set(META)
N_FOLDS = 5


def fold_of(ids, n=N_FOLDS, salt=b""):
    return np.fromiter((zlib.crc32(salt + s.encode()) % n for s in ids), dtype=np.int8, count=len(ids))


def feature_cols(path):
    return [c for c in pq.ParquetFile(path).schema_arrow.names if c not in EXCLUDE]


def load_keys(path, cols=("a", "r", "y")):
    t = pq.read_table(path, columns=[c for c in cols if c in pq.ParquetFile(path).schema_arrow.names])
    return {c: t.column(c).to_numpy() for c in t.column_names}


def iter_groups(path, cols):
    pf = pq.ParquetFile(path)
    for i in range(pf.num_row_groups):
        yield pf.read_row_group(i, columns=cols).to_pandas()


def sample_rows(a, a_fold, rate, seed=0):
    """Boolean mask of rows whose S1 entity is in the training sample (entity-level sampling, deterministic)."""
    rng = np.random.RandomState(seed)
    keep_ent = rng.rand(a_fold.shape[0]) < rate
    return keep_ent[a]


# Per-block count -> rate rescaling (config rate_features): path -> factor (1e6 / S1 entities of the block). Absolute
# chain/frequency counts grow with the size of a country's sample, so a model trained on large blocks reads the counts
# of a small unseen country (France) as "rare name"; rates per million S1 are comparable across countries.
SCALE = {}
RATE_COLS = ("s1_name_chain", "s1_skel_chain", "r_name_s1count", "r_name_recfreq")


def apply_scale(d, path):
    """Rescale count columns of one row group in place when the block has a rate factor (see SCALE)."""
    f = SCALE.get(path)
    if f is None:
        return d
    for c in RATE_COLS:
        if c in d:
            d[c] = (d[c] * f).astype(np.float32)
    if "r_empty_x_chain" in d and "s1_name_chain" in d and "r_addr_empty" in d:
        d["r_empty_x_chain"] = (d["r_addr_empty"] * np.log1p(d["s1_name_chain"])).astype(np.float32)
    return d


def gather(path, cols, mask, extra=None):
    """Load feature rows where mask is True (streamed by row group). extra: dict name->full-length array to attach."""
    out, off = [], 0
    for d in iter_groups(path, cols):
        m = mask[off:off + len(d)]
        if m.any():
            d = apply_scale(d.loc[m].astype(np.float32), path)
            if extra:
                for k, v in extra.items():
                    d[k] = v[off:off + len(m)][m]
            out.append(d)
        off += len(m)
    return pd.concat(out, ignore_index=True)


PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=127, verbose=-1, num_threads=6, seed=0)


def train_lgb(X, y, Xv=None, yv=None, rounds=600, params=None):
    p = dict(PARAMS, **(params or {}))
    dtr = lgb.Dataset(X, y, free_raw_data=True)
    if Xv is not None:
        dv = lgb.Dataset(Xv, yv, reference=dtr)
        m = lgb.train(p, dtr, rounds, valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        m = lgb.train(p, dtr, rounds)
    return m


def predict_stream(path, cols, models, row_fold=None, extra=None):
    """Predict every row. models: single model, or list indexed by row_fold (OOF)."""
    preds, off = [], 0
    for d in iter_groups(path, cols):
        n = len(d)
        X = apply_scale(d.astype(np.float32), path)
        if extra:
            for k, v in extra.items():
                X[k] = v[off:off + n]
        if isinstance(models, list):
            p = np.empty(n, np.float32)
            f = row_fold[off:off + n]
            for k, m in enumerate(models):
                sel = f == k
                if sel.any():
                    p[sel] = m.predict(X.loc[sel], num_threads=6)
        else:
            p = models.predict(X, num_threads=6).astype(np.float32)
        preds.append(p)
        off += n
    return np.concatenate(preds)


def _group_rank(g, s):
    """Descending rank of s within groups g (1 = best), group max, second best, group size (int32 temporaries)."""
    n = len(s)
    o = np.lexsort((-s, g))
    gs = g[o]
    start = np.empty(n, bool)
    start[0] = True
    np.not_equal(gs[1:], gs[:-1], out=start[1:])
    del gs
    first_idx = np.flatnonzero(start).astype(np.int32)
    gid = (np.cumsum(start, dtype=np.int32) - 1)
    del start
    ss = s[o]
    size = np.diff(np.r_[first_idx, n]).astype(np.int32)
    gmax = ss[first_idx]
    second = np.where(size > 1, ss[np.minimum(first_idx + 1, n - 1)], -1.0).astype(np.float32)
    del ss

    def scatter(v):
        out = np.empty(n, np.float32)
        out[o] = v
        return out

    rank = scatter(np.arange(n, dtype=np.int32) - first_idx[gid] + 1)
    out = rank, scatter(gmax[gid]), scatter(second[gid]), scatter(size[gid])
    return out


def context(a, r, p, prefix="c", store=None):
    """Competition features of score p within record r and within S1 entity a.
    With `store` (a directory), each array is written to disk and returned memory-mapped."""
    out = {}

    def put(name, v):
        v = np.asarray(v, dtype=np.float32)
        if store is not None:
            fn = os.path.join(store, f"ctx_{name}.npy")
            np.save(fn, v)
            v = np.load(fn, mmap_mode="r")
        out[name] = v

    put(f"{prefix}_p", p)
    rr, rmax, rsec, rn = _group_rank(r, p)
    other = np.where(p >= rmax, rsec, rmax)
    del rmax, rsec
    put(f"{prefix}_r_rank", rr); put(f"{prefix}_r_other", other); put(f"{prefix}_r_margin", p - other)
    put(f"{prefix}_r_n", rn)
    del rr, rn, other
    put(f"{prefix}_r_sum", np.bincount(r, weights=p)[r])
    ar, amax, asec, an = _group_rank(a, p)
    other = np.where(p >= amax, asec, amax)
    put(f"{prefix}_a_rank", ar); put(f"{prefix}_a_gap", p - amax); put(f"{prefix}_a_other", other)
    put(f"{prefix}_a_n", an)
    del ar, amax, asec, an, other
    put(f"{prefix}_a_sum", np.bincount(a, weights=p)[a])
    return out


def train_folds(X, y, f, rounds=600, params=None, n_folds=N_FOLDS, es=None):
    """Bin the sample once, then train each fold on subsets (no per-fold copies of the raw matrix)."""
    p = dict(PARAMS, **(params or {}))
    cols = list(X.columns)
    full = lgb.Dataset(X, y, params={"max_bin": p["max_bin"], "verbose": -1}, free_raw_data=True).construct()
    del X
    models = []
    for k in range(n_folds):
        if es is None:                         # original behaviour: early stop on the held-out fold
            tr, va = np.flatnonzero(f != k), np.flatnonzero(f == k)
        else:                                  # honest: early stop on an entity slice of the TRAINING folds
            tr, va = np.flatnonzero((f != k) & ~es), np.flatnonzero((f != k) & es)
        dtr, dva = full.subset(tr), full.subset(va)
        m = lgb.train(p, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        models.append(m)
        yield k, m
    return cols
