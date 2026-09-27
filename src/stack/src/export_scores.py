"""Write the keyed probability exports: p_train_{India,US}, p_test_{France,India,US}, scores_{tag}_{h}_{heldout,test}.

p_train: cache/ic_q/<ic>_<c>.npz (stack_ce.py eval with SAVE_IC=<ic>), row-aligned with the train pair table; rows kept
if q >= 0.001 or cascade-kept (p2 > 0.02 and top 10 by p2 per S1).
p_test: <build>/final_p_<c>.npz (stack_ce.py test), codes mapped to ids through the test pair table.
scores: ../kaggle/out/scores_<tag>_half<h>_<split>.parquet copied as scores_<tag>_<h>_<split>.parquet.

  python src/export_scores.py --ic e1 --build output_v08 --out ../../cache/stack
"""
import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from io_utils import ROOT                                 # noqa: E402
from pairs import out_dir, safe, DEFAULT                  # noqa: E402
from cand_budget import keep_mask, rank_in_entity         # noqa: E402

KAG = os.environ.get("CE_DIR", os.path.join(ROOT, os.pardir, "kaggle"))


def train(ic, c, out):
    z = np.load(os.path.join(ROOT, "cache", "ic_q", f"{ic}_{c}.npz"))
    t = pq.read_table(os.path.join(out_dir("train", dict(DEFAULT)), safe(c) + ".parquet"), columns=["a_id", "r_id", "a"])
    a = t.column("a").to_numpy().astype(np.int64)
    q, p2 = z["q"].astype(np.float64), z["p2"].astype(np.float64)
    if not (len(q) == len(p2) == t.num_rows and np.array_equal(z["a"], a)):
        raise ValueError(f"{c}: ic_q not aligned with the train pair table")
    k = np.flatnonzero((q >= 0.001) | keep_mask(a, p2, rank_in_entity(a, p2), 0.02, 10))
    d = pd.DataFrame({"a_id": t.column("a_id").to_numpy()[k], "r_id": t.column("r_id").to_numpy()[k],
                      "q": q[k].astype(np.float32), "p2": p2[k].astype(np.float32)})
    d.to_parquet(os.path.join(out, f"p_train_{c}.parquet"), index=False, compression="zstd")
    print("train", c, len(d), flush=True)


def test(build, c, out):
    z = np.load(os.path.join(ROOT, build, f"final_p_{safe(c)}.npz"))
    t = pq.read_table(os.path.join(out_dir("test", dict(DEFAULT)), safe(c) + ".parquet"), columns=["a", "r", "a_id", "r_id"]).to_pandas()
    ids = t.drop_duplicates("a").set_index("a").a_id
    rid = t.drop_duplicates("r").set_index("r").r_id
    d = pd.DataFrame({"a_id": ids.reindex(z["a"]).values, "r_id": rid.reindex(z["r"]).values, "p": z["p"].astype(np.float32)})
    if d[["a_id", "r_id"]].isna().any().any():
        raise ValueError(f"{c}: final_p codes not found in the test pair table")
    d.to_parquet(os.path.join(out, f"p_test_{c}.parquet"), index=False, compression="zstd")
    print("test", c, len(d), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ic", default="e1")
    ap.add_argument("--build", default="output_v08")
    ap.add_argument("--out", default="../../cache/stack")
    ap.add_argument("--tags", default="e5i,e5b")
    a = ap.parse_args()
    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    for c in ("India", "US"):
        train(a.ic, c, out)
    for c in ("France", "India", "US"):
        test(a.build, c, out)
    for tag in a.tags.split(","):
        for h in (0, 1):
            for s in ("heldout", "test"):
                shutil.copyfile(os.path.join(KAG, "out", f"scores_{tag}_half{h}_{s}.parquet"), os.path.join(out, f"scores_{tag}_{h}_{s}.parquet"))


if __name__ == "__main__":
    main()
