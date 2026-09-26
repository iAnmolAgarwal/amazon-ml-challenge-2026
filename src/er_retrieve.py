"""Extra candidate retrieval: character 3-gram TF-IDF nearest neighbours of each S1 in the pool of its country.

usage: python src/er_retrieve.py --split train|test --view name|address|text [--country C] [--shard i/n] [--queries PATH] [--k 40] [--tag ""]

For one view (name = name_core, address = addr_norm, text = name_core + " | " + addr_norm) the vectorizer is fitted on all pool records of the split:
character 3-grams (no lowercasing, the views are normalized already), terms kept if they occur in >= 2 pool records,
sublinear tf 1 + ln(tf), smoothed idf ln((n + 1) / (df + 1)) + 1, rows L2-normalized, float32. Each S1 (all S1 of the
split, or the s1_idx listed in --queries) is matched against the pool records of its own country only; the k best
cosine similarities > 0 are kept (ties broken by pool index). The pool is searched in blocks of 125k records (small
per-thread accumulators) and the per-block top-k lists are merged. Label-free.
Writes cache/cand/{split}_ret_{view}{tag}.parquet (s1_idx, pool_idx, similarity, rank); with --country, one file per
country cache/cand/{split}_ret_{view}{tag}_{country}.parquet (the vectorizer is still fitted on the whole pool).
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from common import load_config

VIEW = {"name": "name_core", "address": "addr_norm", "text": "text"}
BATCH = 20_000
BLOCK = 125_000


def weigh(x, idf):
    x = x.tocsr().astype(np.float32)
    np.log(x.data, out=x.data)
    x.data += 1
    x.data *= idf[x.indices]
    return normalize(x, norm="l2", copy=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--view", choices=list(VIEW), required=True)
    ap.add_argument("--country", help="retrieve only for S1 of this country")
    ap.add_argument("--shard", help="i/n: retrieve only for S1 with s1_idx % n == i")
    ap.add_argument("--queries", help="parquet with an s1_idx column: retrieve only for these S1")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cache = Path(load_config()["cache_dir"])
    th = int(os.environ.get("RK_THREADS", os.cpu_count() or 4))
    col = VIEW[a.view]
    t = time.time()
    load = lambda side: (pl.read_parquet(cache / "norm" / f"{a.split}_{side}.parquet", columns=["country_n", "name_core", "addr_norm"]).with_columns(text=pl.col("name_core").fill_null("") + " | " + pl.col("addr_norm").fill_null("")).select("country_n", "text")
                         if col == "text" else pl.read_parquet(cache / "norm" / f"{a.split}_{side}.parquet", columns=["country_n", col]))
    pool = load("pool").with_row_index("pool_idx").with_columns(pl.col(col).fill_null(""))
    s1 = load("s1").with_row_index("s1_idx").with_columns(pl.col(col).fill_null(""))
    if a.country:
        s1 = s1.filter(pl.col("country_n") == a.country)
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        s1 = s1.filter(pl.col("s1_idx") % n == i)
    if a.queries:
        s1 = s1.join(pl.read_parquet(a.queries, columns=["s1_idx"]).with_columns(pl.col("s1_idx").cast(pl.UInt32)), on="s1_idx", how="semi")
    cv = CountVectorizer(analyzer="char", ngram_range=(3, 3), lowercase=False, min_df=2, dtype=np.float32)
    P = cv.fit_transform(pool[col].to_list())
    df = np.bincount(P.indices, minlength=P.shape[1])
    idf = (np.log((P.shape[0] + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
    P = weigh(P, idf)
    Q = weigh(cv.transform(s1[col].to_list()), idf)
    print(f"{a.split} {a.view}: pool {P.shape[0]:,} x {P.shape[1]:,} terms, {Q.shape[0]:,} queries, {time.time() - t:.0f}s", flush=True)
    pc, qc = pool["country_n"].to_numpy(), s1["country_n"].to_numpy()
    qid = s1["s1_idx"].to_numpy()
    parts = []
    for ctry in sorted(set(qc)):
        pi, qi = np.flatnonzero(pc == ctry), np.flatnonzero(qc == ctry)
        if not len(pi) or not len(qi):
            continue
        blocks = [(pi[b : b + BLOCK], P[pi[b : b + BLOCK]].T.tocsr()) for b in range(0, len(pi), BLOCK)]
        for s in range(0, len(qi), BATCH):
            qq = qi[s : s + BATCH]
            Qb = Q[qq]
            rows, cols, vals = [], [], []
            for gidx, PT in blocks:
                m = sp_matmul_topn(Qb, PT, top_n=a.k, threshold=0.0, sort=False, n_threads=th, idx_dtype=np.int64).tocoo()
                rows.append(m.row.astype(np.int64))
                cols.append(gidx[m.col])
                vals.append(m.data.astype(np.float32))
            e = pl.DataFrame({"q": np.concatenate(rows), "pool_idx": np.concatenate(cols).astype(np.uint32), "similarity": np.concatenate(vals)}).filter(pl.col("similarity") > 0)
            e = e.with_columns(s1_idx=pl.Series(qid[qq][e["q"].to_numpy()]).cast(pl.UInt32)).drop("q")
            e = e.sort(["s1_idx", "similarity", "pool_idx"], descending=[False, True, False]).with_columns(rank=pl.int_range(1, pl.len() + 1).over("s1_idx"))
            parts.append(e.filter(pl.col("rank") <= a.k).select("s1_idx", "pool_idx", "similarity", pl.col("rank").cast(pl.UInt8)))
            if (s // BATCH) % 10 == 0:
                print(f"  {ctry}: {min(s + BATCH, len(qi)):,}/{len(qi):,} queries, {time.time() - t:.0f}s", flush=True)
        del blocks
    out = pl.concat(parts)
    path = cache / "cand" / f"{a.split}_ret_{a.view}{a.tag}{'_' + a.country if a.country else ''}{'_s' + a.shard.replace('/', 'of') if a.shard else ''}.parquet"
    out.write_parquet(path)
    print(f"wrote {path.name}: {len(out):,} pairs for {out['s1_idx'].n_unique():,} S1 in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
