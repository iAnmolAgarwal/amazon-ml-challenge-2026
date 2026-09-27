"""Card C3: name-specificity and asymmetric-difference features added to cached pair tables (no re-blocking).

Per country block, from the normalised records only (label-free, computed the same way on train and test):
  s1_name_chain   : number of S1 entities in the block with exactly this S1's name_core (chain size)
  r_name_s1count  : number of S1 entities in the block whose name_core equals the record's name_core
  r_name_recfreq  : number of S2/S3 records in the block with the record's name_core
  s1_skel_chain   : chain size on the phonetic skeleton
  nm_shared_idf   : IDF mass of name tokens present on both sides (IDF over S1 names of the block)
  nm_shared_maxidf: IDF of the rarest shared name token
  nm_only_r_idf   : IDF mass of record-only name tokens ("true name + extra word" distractors)
  nm_only_a_idf   : IDF mass of S1-only name tokens
  nm_only_r_n / nm_only_a_n : counts of one-sided tokens
  r_empty_x_chain : empty record address x log(1 + s1_name_chain)   (empty-address ownership signal)
Input : cache/pairs_<split>_<tag(DEFAULT)>/<block>.parquet ; Output: cache/pairs_<split>_<tag(DEFAULT+spec)>/.

  python src/feat_spec.py --split train
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import HashingVectorizer

sys.path.insert(0, os.path.dirname(__file__))
from normalize import load_norm, CACHE                         # noqa: E402
from pairs import out_dir, safe, DEFAULT                       # noqa: E402
from blocking import iter_block                                # noqa: E402

TAG = {"spec": 1}
HV = HashingVectorizer(n_features=2 ** 20, token_pattern=r"\S+", lowercase=False, alternate_sign=False,
                       norm=None, binary=True, dtype=np.float32)


def block(split, b, log=print):
    t0 = time.time()
    src = os.path.join(out_dir(split, dict(DEFAULT)), safe(b) + ".parquet")
    dst_dir = out_dir(split, dict(DEFAULT, **TAG))
    os.makedirs(dst_dir, exist_ok=True)
    json.dump(dict(DEFAULT, **TAG), open(os.path.join(dst_dir, "cfg.json"), "w"))
    dst = os.path.join(dst_dir, safe(b) + ".parquet")
    if os.path.exists(dst):
        return {"cached": True}
    A = load_norm(split, 1, ["id", "name_core", "name_skel"], country=b).reset_index(drop=True)
    r_paths = [os.path.join(CACHE, f"norm_{split}_s{k}.parquet") for k in (2, 3)]
    R = pd.concat(list(iter_block(r_paths, b, ["id", "country", "name_core", "addr_norm"], batch=500_000)),
                  ignore_index=True)
    a_name, r_name = A.name_core.fillna("").values, R.name_core.fillna("").values
    s1_name_count = pd.Series(a_name).value_counts()
    s1_chain = pd.Series(a_name).map(s1_name_count).values.astype(np.float32)
    s1_skel = pd.Series(A.name_skel.fillna("").values)
    s1_skel_chain = s1_skel.map(s1_skel.value_counts()).values.astype(np.float32)
    r_s1count = pd.Series(r_name).map(s1_name_count).fillna(0).values.astype(np.float32)
    r_recfreq = pd.Series(r_name).map(pd.Series(r_name).value_counts()).values.astype(np.float32)
    r_empty = (R.addr_norm.fillna("") == "").values
    MA, MR = HV.transform(a_name).tocsr(), HV.transform(r_name).tocsr()
    df = np.bincount(MA.indices, minlength=MA.shape[1]).astype(np.float32)
    idf = (np.log((1 + MA.shape[0]) / (1 + df)) + 1).astype(np.float32)
    WA, WR = MA.copy(), MR.copy()
    WA.data = idf[WA.indices]
    WR.data = idf[WR.indices]
    a_mass = np.asarray(WA.sum(1)).ravel()
    r_mass = np.asarray(WR.sum(1)).ravel()
    a_n, r_n = np.diff(MA.indptr), np.diff(MR.indptr)
    del R
    pf = pq.ParquetFile(src)
    writer = None
    for g in range(min(pf.num_row_groups, int(os.environ.get("SPEC_MAX_GROUPS", 10**9)))):
        d = pf.read_row_group(g).to_pandas()
        ai, ri = d.a.values.astype(np.int64), d.r.values.astype(np.int64)
        inter = WA[ai].multiply(MR[ri])                          # idf weights of shared tokens
        shared = np.asarray(inter.sum(1)).ravel()
        shared_max = np.asarray(inter.max(1).todense()).ravel()
        shared_n = np.diff(inter.tocsr().indptr)
        d["s1_name_chain"] = s1_chain[ai]
        d["s1_skel_chain"] = s1_skel_chain[ai]
        d["r_name_s1count"] = r_s1count[ri]
        d["r_name_recfreq"] = r_recfreq[ri]
        d["nm_shared_idf"] = shared.astype(np.float32)
        d["nm_shared_maxidf"] = shared_max.astype(np.float32)
        d["nm_only_r_idf"] = (r_mass[ri] - shared).astype(np.float32)
        d["nm_only_a_idf"] = (a_mass[ai] - shared).astype(np.float32)
        d["nm_only_r_n"] = (r_n[ri] - shared_n).astype(np.float32)
        d["nm_only_a_n"] = (a_n[ai] - shared_n).astype(np.float32)
        d["r_empty_x_chain"] = (r_empty[ri] * np.log1p(s1_chain[ai])).astype(np.float32)
        tbl = pa.Table.from_pandas(d, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(dst + ".tmp", tbl.schema)
        writer.write_table(tbl)
    writer.close()
    os.replace(dst + ".tmp", dst)
    st = {"pairs": int(pf.metadata.num_rows), "seconds": round(time.time() - t0, 1)}
    log(f"[{split}/{b}] {json.dumps(st)}")
    return st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    s1 = pq.read_table(os.path.join(CACHE, f"norm_{a.split}_s1.parquet"), columns=["country"]).column(0).unique().to_pylist()
    print(json.dumps({b: block(a.split, b) for b in sorted(s1) if not a.only or b in a.only}))
