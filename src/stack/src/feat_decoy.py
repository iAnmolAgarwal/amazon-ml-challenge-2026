"""Append the label-free decoy features (decoy_feats.py) to EVERY candidate pair, for a decoy-aware stage 2.

Stage 2 decides ~94 % of France's emitted pairs (p >= 0.98 never reaches stage 3) but has no decoy evidence. This adds
the decoy feature columns (house relations incl. French fallback parser, legal/unit relations, word edits, marker
scores with country lift, direction and S1-name rarity) to the no-specificity candidate tables, streamed by row group:
cache/pairs_<split>_<tag(DEFAULT)>/<block>.parquet -> cache/pairs_<split>_<tag(DEFAULT + dk)>/<block>.parquet.
Marker tables per country and split exactly as decoy_feats.build_tables (no labels). Prints counts only.

  python src/feat_decoy.py --split train
  python src/feat_decoy.py --split test --only France
"""
import argparse
import json
import os
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from pairs import out_dir, safe, DEFAULT                 # noqa: E402
import decoy_feats as D                                   # noqa: E402

TAG = {"dk": 1}


def block(split, c, base=None):
    """Append decoy features to one block's table; base="spec" starts from the name-specificity tables instead."""
    b = dict(DEFAULT, **({"spec": 1} if base == "spec" else {}))
    src = os.path.join(out_dir(split, b), safe(c) + ".parquet")
    dst_dir = out_dir(split, dict(b, **TAG))
    os.makedirs(dst_dir, exist_ok=True)
    json.dump(dict(b, **TAG), open(os.path.join(dst_dir, "cfg.json"), "w"))
    dst = os.path.join(dst_dir, safe(c) + ".parquet")
    if os.path.exists(dst):
        print(split, c, "cached", flush=True)
        return
    t0 = time.time()
    S, P = D.records(split, c)
    tables = D.build_tables(split, c, S, P)
    pf = pq.ParquetFile(src)
    writer = None
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g)
        keys = t.select(["a_id", "r_id"]).to_pandas()
        f = D.features(keys, S, P, tables).drop(columns=["a_id", "r_id"])
        for col in f.columns:
            t = t.append_column(col, pa.array(f[col].values))
        if writer is None:
            writer = pq.ParquetWriter(dst + ".tmp", t.schema)
        writer.write_table(t)
    writer.close()
    os.replace(dst + ".tmp", dst)
    print(split, c, "pairs", pf.metadata.num_rows, f"{time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--base", choices=["spec"], help="start from the spec tables (spec + decoy stage 2)")
    a = ap.parse_args()
    for c in a.only or (["India", "US"] if a.split == "train" else ["France", "India", "US"]):
        block(a.split, c, a.base)


if __name__ == "__main__":
    main()
