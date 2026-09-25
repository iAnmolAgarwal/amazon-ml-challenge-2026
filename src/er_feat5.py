"""Side-car name-specificity features (feat5): how informative the name agreement of a candidate pair is.

usage: python src/er_feat5.py --split train|test [--tag ""]

Per country, token weights come from the S1 records of the same split only (label-free, identical on train and test):
w(t) = 1 + log((N + 1) / (df(t) + 1)), df = number of S1 core names containing t (unseen tokens get the maximum).
With A = S1 core-name token set and B = pool core-name token set:
- sp_shared: sum of w over A & B; sp_s1_only: over A - B; sp_pool_only: over B - A
- sp_s1_only_frac / sp_pool_only_frac: one-sided mass divided by that side's total mass (-1 for an empty side)
- sp_pool_only_max: largest w of a pool-only token (0 if none)
- cos_name / cos_addr: cosine of w-weighted core-name tokens / of w-weighted address tokens (address weights from S1
  addresses), -1 when either side is empty
- sk_n_s1 / sk_n_pool: log1p number of S1 in the country sharing the S1's / the pool record's consonant skeleton
Writes cache/feat/{split}_feat5{tag}.parquet keyed by (s1_idx, pool_idx).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from common import load_config
from er_norm import skeleton

COLS = ["sp_shared", "sp_s1_only", "sp_pool_only", "sp_s1_only_frac", "sp_pool_only_frac", "sp_pool_only_max", "cos_name", "cos_addr", "sk_n_s1", "sk_n_pool"]


def weighted(a_text, b_text):
    """Binary token matrices over a shared vocabulary and the S1-document-frequency weights w."""
    cv = CountVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, binary=True, dtype=np.float32)
    cv.fit(list(a_text) + list(b_text))
    a, b = cv.transform(a_text).tocsr(), cv.transform(b_text).tocsr()
    df = np.asarray(a.sum(axis=0)).ravel()
    w = (1.0 + np.log((a.shape[0] + 1.0) / (df + 1.0))).astype(np.float32)
    return a, b, w


def pair_feats(a, b, w, r, c, batch=2_000_000):
    """Shared / one-sided IDF masses, max pool-only weight and weighted cosine for pairs (r, c)."""
    W = sparse.diags(w)
    aw, bw = (a @ W).tocsr(), (b @ W).tocsr()
    ta, tb = np.asarray(aw.sum(axis=1)).ravel(), np.asarray(bw.sum(axis=1)).ravel()
    na, nb = np.sqrt(np.asarray(aw.multiply(aw).sum(axis=1)).ravel()), np.sqrt(np.asarray(bw.multiply(bw).sum(axis=1)).ravel())
    out = {k: np.empty(len(r), np.float32) for k in ("shared", "a_only", "b_only", "b_only_max", "cos")}
    for s in range(0, len(r), batch):
        rr, cc = r[s : s + batch], c[s : s + batch]
        A, B, BW, AW = a[rr], b[cc], bw[cc], aw[rr]
        shared = np.asarray(BW.multiply(A).sum(axis=1)).ravel()
        only = BW - BW.multiply(A)
        dot = np.asarray(AW.multiply(BW).sum(axis=1)).ravel()
        den = na[rr] * nb[cc]
        out["shared"][s : s + batch] = shared
        out["a_only"][s : s + batch] = ta[rr] - shared
        out["b_only"][s : s + batch] = tb[cc] - shared
        out["b_only_max"][s : s + batch] = only.max(axis=1).toarray().ravel()
        out["cos"][s : s + batch] = np.where(den > 0, dot / np.maximum(den, 1e-12), -1.0)
    return out, ta, tb


def build(split, tag):
    cfg = load_config()
    cache = Path(cfg["cache_dir"])
    t = time.time()
    cols = ["country_n", "name_core", "name_join", "addr_norm"]
    s1 = pl.read_parquet(cache / "norm" / f"{split}_s1.parquet", columns=cols).with_row_index("s1_idx")
    pool = pl.read_parquet(cache / "norm" / f"{split}_pool.parquet", columns=cols).with_row_index("pool_idx")
    s1 = s1.with_columns(skel=pl.Series([skeleton(x) for x in s1["name_join"].to_list()]))
    pool = pool.with_columns(skel=pl.Series([skeleton(x) for x in pool["name_join"].to_list()]))
    cand = pl.read_parquet(cache / "cand" / f"{split}_cand{tag}.parquet", columns=["s1_idx", "pool_idx", "country_n"])
    parts = []
    for ctry in sorted(cand["country_n"].unique().to_list()):
        tc = time.time()
        a = s1.filter(pl.col("country_n") == ctry)
        b = pool.filter(pl.col("country_n") == ctry)
        k = cand.filter(pl.col("country_n") == ctry).select("s1_idx", "pool_idx")
        amap = np.full(len(s1), -1, np.int64)
        amap[a["s1_idx"].to_numpy()] = np.arange(len(a))
        bmap = np.full(len(pool), -1, np.int64)
        bmap[b["pool_idx"].to_numpy()] = np.arange(len(b))
        r, c = amap[k["s1_idx"].to_numpy()], bmap[k["pool_idx"].to_numpy()]
        na_, nb_, w = weighted(a["name_core"].to_list(), b["name_core"].to_list())
        f, ta, tb = pair_feats(na_, nb_, w, r, c)
        aa, ab, wa = weighted(a["addr_norm"].to_list(), b["addr_norm"].to_list())
        fa, _, _ = pair_feats(aa, ab, wa, r, c)
        sk = a.group_by("skel").len().filter(pl.col("skel") != "")
        ska = a.select("skel").join(sk, on="skel", how="left", maintain_order="left")["len"].fill_null(0).to_numpy()
        skb = b.select("skel").join(sk, on="skel", how="left", maintain_order="left")["len"].fill_null(0).to_numpy()
        ta_r, tb_c = ta[r], tb[c]
        parts.append(k.with_columns(
            sp_shared=pl.Series(f["shared"]),
            sp_s1_only=pl.Series(f["a_only"]),
            sp_pool_only=pl.Series(f["b_only"]),
            sp_s1_only_frac=pl.Series(np.where(ta_r > 0, f["a_only"] / np.maximum(ta_r, 1e-9), -1.0).astype(np.float32)),
            sp_pool_only_frac=pl.Series(np.where(tb_c > 0, f["b_only"] / np.maximum(tb_c, 1e-9), -1.0).astype(np.float32)),
            sp_pool_only_max=pl.Series(f["b_only_max"]),
            cos_name=pl.Series(f["cos"]),
            cos_addr=pl.Series(fa["cos"]),
            sk_n_s1=pl.Series(np.log1p(ska[r]).astype(np.float32)),
            sk_n_pool=pl.Series(np.log1p(skb[c]).astype(np.float32)),
        ))
        print(f"  {ctry}: {len(k):,} pairs in {time.time() - tc:.0f}s", flush=True)
    out = pl.concat(parts).with_columns([pl.col(x).cast(pl.Float32) for x in COLS])
    path = cache / "feat" / f"{split}_feat5{tag}.parquet"
    out.write_parquet(path)
    print(f"wrote {path.name}: {len(out):,} rows in {time.time() - t:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    build(a.split, a.tag)


if __name__ == "__main__":
    main()
