"""Build the candidate-pair feature table for one split, block by block (block = identical `country` string; every
label present in the data is processed, nothing is filtered). Memory-bounded: S2/S3 records are streamed in chunks.

Output: cache/pairs_<split>_<cfgtag>/<block>.parquet with columns
  a_id, r_id, a, r, via, b_conj, b_rk_r, b_rk_a, <pair features>, [y on train]
"""
import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
import memguard  # noqa: E402
from io_utils import ROOT, DATA, read_tsv, gt_pairs  # noqa: E402
from normalize import load_norm, CACHE  # noqa: E402
from blocking import block, iter_block  # noqa: E402
from features import PairFeaturizer  # noqa: E402

COLS = ["id", "src", "country", "name_core", "legal", "alias", "name_skel", "is_domain",
        "addr_norm", "addr_nums", "name_nonlatin", "addr_nonlatin"]
DEFAULT = {"k_r": 3, "k_a": 15, "maxdf": 3000, "min_score": 0.1}


def cfg_tag(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def safe(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s) or "blank"


def out_dir(split, cfg):
    return os.path.join(ROOT, "cache", f"pairs_{split}_{cfg_tag(cfg)}")


def build(split, cfg, only=None, log=print):
    od = out_dir(split, cfg)
    os.makedirs(od, exist_ok=True)
    json.dump(cfg, open(os.path.join(od, "cfg.json"), "w"))
    countries = sorted(pq.read_table(os.path.join(CACHE, f"norm_{split}_s1.parquet"), columns=["country"])
                       .column(0).unique().to_pylist())
    if only:
        countries = [c for c in countries if c in only]
    r_paths = [os.path.join(CACHE, f"norm_{split}_s{k}.parquet") for k in (2, 3)]
    for c in countries:
        path = os.path.join(od, f"{safe(c)}.parquet")
        if os.path.exists(path):
            log(f"[{split}/{c}] cached")
            continue
        t = time.time()
        A = load_norm(split, 1, ["id", "country", "name_core", "name_skel", "addr_norm", "addr_nums"], country=c)
        A = A.reset_index(drop=True)
        log(f"[{split}/{c}] S1={len(A)}")
        P, rids = block(A, r_paths, c, cfg, log=log)
        del rids, A
        gc.collect()
        A = load_norm(split, 1, [x for x in COLS if x not in ("alias", "is_domain", "addr_nonlatin", "src")],
                      country=c).reset_index(drop=True)
        truth = None
        if split == "train":
            gt = read_tsv(os.path.join(DATA, "train", "train_ground_truth.tsv"))
            gt = gt[gt.source1_entity_id.isin(set(A.id))]
            truth = gt_pairs(gt).rename(columns={"s1": "a_id", "rid": "r_id"})
            truth["y"] = np.int8(1)
            del gt
        P = P.sort_values("r", kind="stable").reset_index(drop=True)
        n_pairs = len(P)
        cand_path = path + ".cand.parquet"
        P.to_parquet(cand_path, index=False, row_group_size=500_000)
        r_pos = P.r.values.copy()
        del P
        gc.collect()
        Pf = pq.ParquetFile(cand_path)
        rg_start = np.cumsum([0] + [Pf.metadata.row_group(i).num_rows for i in range(Pf.num_row_groups)])

        def pairs_slice(lo, hi):
            """Rows [lo, hi) of the r-sorted candidate table, read from disk."""
            g0, g1 = np.searchsorted(rg_start, lo, "right") - 1, np.searchsorted(rg_start, hi, "left")
            t = pd.concat([Pf.read_row_group(g).to_pandas() for g in range(g0, g1)], ignore_index=True)
            return t.iloc[lo - rg_start[g0]:hi - rg_start[g0]]

        feat = PairFeaturizer(A)
        log(f"  featurizer ready {time.time() - t:.0f}s  rss={memguard.rss_gb():.2f}GB")
        tmp = path + ".tmp"
        writer, offset = None, 0
        for R in iter_block(r_paths, c, COLS, batch=250_000):
            lo, hi = np.searchsorted(r_pos, [offset, offset + len(R)])
            for sl in range(lo, hi, 200_000):
                Pc = pairs_slice(sl, min(sl + 200_000, hi))
                ai = Pc.a.values
                ri = Pc.r.values - offset
                F = feat.transform(R, ai, ri)
                F.insert(0, "a_id", A.id.values[ai])
                F.insert(1, "r_id", R.id.values[ri])
                for j, col in enumerate(["a", "r", "via", "b_conj", "b_rk_r", "b_rk_a"]):
                    F.insert(2 + j, col, Pc[col].values)
                if truth is not None:
                    F = F.merge(truth, on=["a_id", "r_id"], how="left")
                    F["y"] = F.y.fillna(0).astype(np.int8)
                tbl = pa.Table.from_pandas(F, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(tmp, tbl.schema)
                writer.write_table(tbl)
                del F, tbl, Pc
                gc.collect()
            offset += len(R)
            log(f"  features -> {offset} records {time.time() - t:.0f}s rss={memguard.rss_gb():.2f}GB")
            del R
            gc.collect()
        writer.close()
        os.replace(tmp, path)
        os.remove(cand_path)
        log(f"  done {c}: {n_pairs} pairs {time.time() - t:.0f}s peak={memguard.peak_gb():.2f}GB")
        del A, feat, truth, r_pos
        gc.collect()
    return od


if __name__ == "__main__":
    memguard.start()
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--cfg", default=json.dumps(DEFAULT))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    print(build(a.split, json.loads(a.cfg), a.only))
