"""Memory-bounded candidate generation, run inside each block of identical `country` string (any label).

Channel "conj": hashed IDF-weighted vector of rare unigrams + conjunction keys (address bigrams, house-number x
street-word, adjacent name-token pairs, cross-script skeleton keys, name x number). IDF is estimated on S1 of the
block (unsupervised). S2/S3 records are streamed in chunks; for each chunk:
  * record -> S1 top-k_r   (index = S1, features with S1-df > maxdf dropped)
  * S1 -> record top-k_a   (index = the chunk; results merged into a running per-S1 top-k_a)
Peak memory ~ S1 matrix + one chunk. The union of both directions is the candidate set the matcher scores.
"""
import multiprocessing as mp
import time

import memguard

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from block_feats import hashed_chunk, N_FEATURES

NCOLS = ["name_core", "name_skel", "addr_norm", "addr_nums"]


def iter_block(paths, country, columns, batch=400_000):
    """Yield DataFrames of rows with country == `country` from the given normalised parquet files, in order."""
    buf, n = [], 0
    for path in paths:
        pf = pq.ParquetFile(path)
        for b in pf.iter_batches(batch_size=batch, columns=columns):
            d = b.to_pandas()
            d = d[d.country == country]
            if len(d):
                buf.append(d)
                n += len(d)
            if n >= batch:
                yield pd.concat(buf, ignore_index=True)
                buf, n = [], 0
    if buf:
        yield pd.concat(buf, ignore_index=True)


def hashed(df, pool, sub=100_000):
    cols = [df[c].tolist() for c in NCOLS]
    jobs = [tuple(c[i:i + sub] for c in cols) for i in range(0, len(df), sub)]
    return sp.vstack(pool.map(hashed_chunk, jobs)).tocsr()


def _weight(M, idf):
    """In-place IDF weighting + L2 normalisation (no copy)."""
    M.data *= idf[M.indices]
    return normalize(M, copy=False)


def _drop_cols(M, keep):
    M = (M @ sp.diags(keep.astype(np.float32))).tocsr()
    M.eliminate_zeros()
    return M


def _merge_topk(best_s, best_r, C, offset, k, rows_per_block=200_000):
    """Merge a sorted top-k CSR (rows = S1) into running dense top-k arrays, in row blocks (bounded temporaries)."""
    n = best_s.shape[0]
    for lo in range(0, n, rows_per_block):
        hi = min(lo + rows_per_block, n)
        sub = C[lo:hi]
        cnt = np.diff(sub.indptr)
        if not cnt.any():
            continue
        pos = np.arange(sub.nnz) - np.repeat(sub.indptr[:-1], cnt)
        rows = np.repeat(np.arange(hi - lo), cnt)
        ns = np.full((hi - lo, k), -1.0, np.float32)
        nr = np.full((hi - lo, k), -1, np.int32)
        ns[rows, pos] = sub.data
        nr[rows, pos] = sub.indices + offset
        S = np.concatenate([best_s[lo:hi], ns], 1)
        Rr = np.concatenate([best_r[lo:hi], nr], 1)
        o = np.argsort(-S, axis=1, kind="stable")[:, :k]
        best_s[lo:hi] = np.take_along_axis(S, o, 1)
        best_r[lo:hi] = np.take_along_axis(Rr, o, 1)
    return best_s, best_r


def block(A, r_paths, country, cfg, log=print, workers=2):
    """A: S1 frame of the block (positional). Streams S2/S3 rows of the same block from r_paths.
    Returns (pairs DataFrame[a, r, via, b_conj], R_ids ndarray, R_offsets per chunk)."""
    t = time.time()
    k_r, k_a, maxdf = cfg["k_r"], cfg["k_a"], cfg["maxdf"]
    min_s = cfg.get("min_score", 0.05)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=1) as pool:  # recycle workers so they release memory
        Ab = hashed(A, pool)
        df = np.bincount(Ab.indices, minlength=N_FEATURES).astype(np.float32)
        idf = (np.log((1 + Ab.shape[0]) / (1 + df)) + 1).astype(np.float32)
        Am = _weight(Ab, idf)
        del Ab
        AmT_capped = _drop_cols(Am, df <= maxdf).T.tocsr()
        log(f"  S1 vectors {Am.shape[0]} nnz/row={Am.nnz / max(Am.shape[0], 1):.1f} {time.time() - t:.0f}s "
            f"rss={memguard.rss_gb():.2f}GB")
        best_s = np.full((Am.shape[0], k_a), -1.0, np.float32)
        best_r = np.full((Am.shape[0], k_a), -1, np.int32)
        r2a, ids, offset = [], [], 0
        for R in iter_block(r_paths, country, ["id", "country"] + NCOLS, batch=250_000):
            Rm = _weight(hashed(R, pool), idf)
            C = sp_matmul_topn(Rm, AmT_capped, top_n=k_r, sort=True, threshold=min_s,
                               n_threads=cfg.get("threads", 6))
            rk = (np.arange(C.nnz) - np.repeat(C.indptr[:-1], np.diff(C.indptr)) + 1).astype(np.int8)
            C = C.tocoo()
            r2a.append(pd.DataFrame({"a": C.col.astype(np.int32), "r": (C.row + offset).astype(np.int32),
                                     "b_conj": C.data.astype(np.float32), "rk": rk}))
            # chunk-local df cap, scaled so it matches maxdf relative to the S1 block size
            cdf = np.bincount(Rm.indices, minlength=N_FEATURES)
            cap = maxdf * Rm.shape[0] / max(Am.shape[0], 1)
            RmT = _drop_cols(Rm, cdf <= max(cap, 50)).T.tocsr()
            C2 = sp_matmul_topn(Am, RmT, top_n=k_a, sort=True, n_threads=cfg.get("threads", 6))
            best_s, best_r = _merge_topk(best_s, best_r, C2, offset, k_a)
            ids.append(R.id.values)
            offset += len(R)
            log(f"  chunk -> {offset} records {time.time() - t:.0f}s rss={memguard.rss_gb():.2f}GB")
            del R, Rm, RmT, C, C2
    P1 = pd.concat(r2a, ignore_index=True)
    P1["via"] = np.int8(1)
    m = (best_r >= 0) & (best_s > min_s)
    rk2 = np.broadcast_to(np.arange(1, k_a + 1, dtype=np.int8), best_r.shape)[m]
    P2 = pd.DataFrame({"a": np.repeat(np.arange(Am.shape[0], dtype=np.int32), k_a)[m.ravel()], "r": best_r[m],
                       "b_conj": best_s[m], "via": np.int8(2), "rk": rk2})
    del m, rk2
    a = np.concatenate([P1.a.values, P2.a.values])
    r = np.concatenate([P1.r.values, P2.r.values])
    s = np.concatenate([P1.b_conj.values, P2.b_conj.values])
    via = np.concatenate([P1.via.values, P2.via.values])
    rk_r = np.concatenate([P1.rk.values, np.full(len(P2), 127, np.int8)])
    rk_a = np.concatenate([np.full(len(P1), 127, np.int8), P2.rk.values])
    del P1, P2, best_s, best_r, r2a
    key = (a.astype(np.int64) << 32) | r.astype(np.int64)
    o = np.argsort(key, kind="stable")
    key, a, r, s, via, rk_r, rk_a = key[o], a[o], r[o], s[o], via[o], rk_r[o], rk_a[o]
    del o
    first = np.r_[True, key[1:] != key[:-1]]
    grp = np.cumsum(first) - 1
    via_u = np.zeros(first.sum(), np.int8)
    np.add.at(via_u, grp, via)  # 1 = r2a only, 2 = a2r only, 3 = both
    s_u = np.full(first.sum(), -1.0, np.float32)
    np.maximum.at(s_u, grp, s)
    rr_u = np.full(first.sum(), 127, np.int8)
    np.minimum.at(rr_u, grp, rk_r)
    ra_u = np.full(first.sum(), 127, np.int8)
    np.minimum.at(ra_u, grp, rk_a)
    P = pd.DataFrame({"a": a[first], "r": r[first], "b_conj": s_u, "via": via_u, "b_rk_r": rr_u, "b_rk_a": ra_u})
    del key, a, r, s, via, first, grp, rk_r, rk_a
    log(f"  union rss={memguard.rss_gb():.2f}GB {len(P)} pairs ({len(P) / max(Am.shape[0], 1):.1f}/S1) {time.time() - t:.0f}s")
    return P, np.concatenate(ids)
