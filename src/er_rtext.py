"""Extra candidate source: reverse character retrieval from pool records that no S1 claims.

usage: python src/er_rtext.py --split train|test --parent <attempt> [--k 5] [--queries PATH] [--tag ""] [--threads 12]

View = name_core + " | " + addr_norm. Per country, character 3-gram TF-IDF fitted on the split's pool records (terms in
>= 2 records and <= 2% of them, sublinear tf, smoothed idf, L2 rows). Queries = pool records whose best probability
under the parent attempt is <= 0.5 (records in no candidate pair count as 0); index = S1 of the same country (with
--queries, only those S1). The k best cosine matches per query are kept. Label-free.
Writes cache/cand/{split}_ret_rtext{tag}.parquet (s1_idx, pool_idx, similarity), read by er_newpairs.py union.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from common import load_config

KEYS = ["s1_idx", "pool_idx"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--queries")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    cache = Path(load_config()["cache_dir"])
    t = time.time()
    view = lambda df: df.select(pl.col("country_n"), (pl.col("name_core").fill_null("") + " | " + pl.col("addr_norm").fill_null("")).alias("v"))
    s1 = view(pl.read_parquet(cache / "norm" / f"{a.split}_s1.parquet", columns=["country_n", "name_core", "addr_norm"])).with_row_index("s1_idx")
    pool = view(pl.read_parquet(cache / "norm" / f"{a.split}_pool.parquet", columns=["country_n", "name_core", "addr_norm"])).with_row_index("pool_idx")
    name = a.parent if a.split == "train" else f"test_{a.parent}"
    best = pl.read_parquet(cache / "oof" / f"{name}.parquet", columns=["pool_idx", "prob"]).group_by("pool_idx").agg(pl.col("prob").max())
    pool = pool.join(best.with_columns(pl.col("pool_idx").cast(pl.UInt32)), on="pool_idx", how="left").with_columns(pl.col("prob").fill_null(0.0))
    if a.queries:
        s1 = s1.join(pl.read_parquet(a.queries, columns=["s1_idx"]).with_columns(pl.col("s1_idx").cast(pl.UInt32)).unique(), on="s1_idx", how="semi")
    out = []
    for c in pool["country_n"].unique().to_list():
        pc, sc = pool.filter(pl.col("country_n") == c), s1.filter(pl.col("country_n") == c)
        if sc.height == 0:
            continue
        vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), lowercase=False, min_df=2, max_df=0.02, sublinear_tf=True, smooth_idf=True, dtype=np.float32)
        vec.fit(pc["v"].to_list())
        qc = pc.filter(pl.col("prob") <= 0.5)
        S = vec.transform(sc["v"].to_list()).T.tocsr()
        sidx, qidx = sc["s1_idx"].to_numpy(), qc["pool_idx"].to_numpy()
        for s in range(0, qc.height, 200_000):
            Q = vec.transform(qc["v"][s:s + 200_000].to_list())
            m = sp_matmul_topn(Q, S, top_n=a.k, threshold=0.0, sort=False, n_threads=a.threads).tocoo()
            out.append(pl.DataFrame({"s1_idx": sidx[m.col].astype(np.uint32), "pool_idx": qidx[s + m.row].astype(np.uint32), "similarity": m.data.astype(np.float32)}))
        print(f"{c}: {qc.height:,} unclaimed pool queries against {sc.height:,} S1, {time.time() - t:.0f}s", flush=True)
    res = pl.concat(out).sort(KEYS)
    path = cache / "cand" / f"{a.split}_ret_rtext{a.tag}.parquet"
    res.write_parquet(path)
    print(f"wrote {path.name}: {res.height:,} pairs, {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
