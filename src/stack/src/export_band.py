"""Export the uncertain band of candidate pairs (lo < p2 < hi) with raw texts, for a pair-model (cross-encoder).

train: out-of-fold stage-2 p2 of an experiment (cache/exp_<exp>/p2_<block>.npy) + label y + country + fold.
test : saved test p2 (cache/predict_work/p2_<block>.npy) + country; no labels.
Columns: a_id, r_id, country, split, fold (train), y (train), p2, a_name, a_addr, r_name, r_addr (raw strings).
Output: <out>/<split>_band.parquet. Row contents are never printed; only counts.

  python src/export_band.py --split train --exp final_s25_spec --out ../kaggle/band
  python src/export_band.py --split test  --out ../kaggle/band
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from normalize import CACHE                                # noqa: E402
from pairs import out_dir, safe, DEFAULT                   # noqa: E402
from model import fold_of                                  # noqa: E402

LO, HI = 0.02, 0.98


def texts(split, k, ids):
    """Raw name/address for the given ids of source k (Arrow filter; memory-light)."""
    t = pq.read_table(os.path.join(CACHE, f"norm_{split}_s{k}.parquet"), columns=["id", "name_raw", "addr_raw"])
    t = t.filter(pc.is_in(t.column("id"), value_set=pa.array(list(ids), pa.string())))
    return t.to_pandas().set_index("id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--exp", default="final_s25_spec")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    blocks = ["India", "US"] if a.split == "train" else ["France", "India", "US"]
    parts, stats = [], {}
    for b in blocks:
        path = os.path.join(out_dir(a.split, dict(DEFAULT)), safe(b) + ".parquet")
        cols = ["a_id", "r_id"] + (["y"] if a.split == "train" else [])
        t = pq.read_table(path, columns=cols).to_pandas()
        pdir = os.path.join(CACHE, f"exp_{a.exp}") if a.split == "train" else os.path.join(CACHE, "predict_work")
        p = np.load(os.path.join(pdir, f"p2_{safe(b)}.npy"))
        if len(p) != len(t):
            raise ValueError(f"{b}: p2 length {len(p)} != pairs {len(t)}")
        m = (p > LO) & (p < HI)
        d = t[m].copy()
        d["p2"] = p[m].astype(np.float32)
        d["country"] = b
        d["split"] = a.split
        if a.split == "train":
            d["fold"] = fold_of(d.a_id.values).astype(np.int8)
        A = texts(a.split, 1, set(d.a_id))
        R = pd.concat([texts(a.split, k, set(d.r_id)) for k in (2, 3)])
        d["a_name"], d["a_addr"] = A.name_raw.reindex(d.a_id).values, A.addr_raw.reindex(d.a_id).values
        d["r_name"], d["r_addr"] = R.name_raw.reindex(d.r_id).values, R.addr_raw.reindex(d.r_id).values
        miss = int(d.a_name.isna().sum() + d.r_name.isna().sum())
        if miss:
            raise ValueError(f"{b}: {miss} texts missing")
        stats[b] = {"pairs_total": int(len(t)), "band_pairs": int(len(d)), "band_share": round(float(m.mean()), 4),
                    **({"band_pos_rate": round(float(d.y.mean()), 4)} if a.split == "train" else {})}
        parts.append(d)
    res = pd.concat(parts, ignore_index=True)
    res.to_parquet(os.path.join(out, f"{a.split}_band.parquet"), index=False, compression="zstd")
    stats["file_mb"] = round(os.path.getsize(os.path.join(out, f"{a.split}_band.parquet")) / 1e6, 1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
