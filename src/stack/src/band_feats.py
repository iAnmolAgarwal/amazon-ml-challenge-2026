"""All stage-1 pair features (+ name-specificity features when the spec tables exist) for the band pairs, so the
in-country stage-3 stacker sees every feature family (an earlier "band model with every feature" check:
+0.0009 CV there). Columns are prefixed pf_ to avoid clashing with the sidecar's recomputed features.
Streams the pair tables by row group and keeps only band rows. In-country countries only (France keeps its own
validated stacker). Writes <kaggle>/band/{split}_pairf.parquet keyed by (a_id, r_id). Aggregates only.

  python src/band_feats.py --split train
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from pairs import out_dir, safe, DEFAULT                  # noqa: E402
from model import feature_cols                            # noqa: E402
from stack_ce import KAG                                  # noqa: E402


def rows(path, keys):
    """Feature rows of the table at `path` for the (a_id, r_id) pairs in `keys` (DataFrame), in keys order."""
    t = pq.read_table(path, columns=["a_id", "r_id"])
    pos = pd.MultiIndex.from_arrays([t.column("a_id").to_numpy(), t.column("r_id").to_numpy()]).get_indexer(
        pd.MultiIndex.from_arrays([keys.a_id.values, keys.r_id.values]))
    del t
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} band pairs not in {path}")
    cols = feature_cols(path)
    pf = pq.ParquetFile(path)
    order = np.argsort(pos)
    sp = pos[order]
    parts, off = [], 0
    for g in range(pf.num_row_groups):
        n_g = pf.metadata.row_group(g).num_rows
        lo, hi = np.searchsorted(sp, [off, off + n_g])
        if hi > lo:
            parts.append(pf.read_row_group(g, columns=cols).take(pa.array(sp[lo:hi] - off)).to_pandas())
        off += n_g
    f = pd.concat(parts, ignore_index=True)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return f.iloc[inv].reset_index(drop=True).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    a = ap.parse_args()
    t0 = time.time()
    band = pd.read_parquet(os.path.join(KAG, "band", f"{a.split}_band.parquet"), columns=["a_id", "r_id", "country"])
    parts = []
    for c in ("India", "US"):
        b = band[band.country == c].reset_index(drop=True)
        spec = os.path.join(out_dir(a.split, dict(DEFAULT, spec=1)), safe(c) + ".parquet")
        base = os.path.join(out_dir(a.split, dict(DEFAULT)), safe(c) + ".parquet")
        f = rows(spec if os.path.exists(spec) else base, b)
        f.columns = ["pf_" + x for x in f.columns]
        parts.append(pd.concat([b[["a_id", "r_id"]], f], axis=1))
        print(a.split, c, len(b), "cols", f.shape[1], "spec" if os.path.exists(spec) else "base", f"{time.time() - t0:.0f}s", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(os.path.join(KAG, "band", f"{a.split}_pairf.parquet"), index=False)
    print("written", out.shape, flush=True)


if __name__ == "__main__":
    main()
