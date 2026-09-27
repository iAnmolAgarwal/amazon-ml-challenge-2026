"""Mean logit of several OOF files with the same keys -> cache/oof/<out>.parquet."""
import sys

import numpy as np
import polars as pl

out, names = sys.argv[1], sys.argv[2:]
K = ["s1_idx", "pool_idx"]
b = pl.read_parquet(f"cache/oof/{names[0]}.parquet").sort(K)
z = []
for n in names:
    p = pl.read_parquet(f"cache/oof/{n}.parquet").sort(K)
    assert p.select(K).equals(b.select(K))
    q = p["prob"].to_numpy().astype(np.float64).clip(1e-7, 1 - 1e-7)
    z.append(np.log(q / (1 - q)))
b.select(K).with_columns(prob=pl.Series((1 / (1 + np.exp(-np.mean(z, axis=0)))).astype(np.float32))).write_parquet(f"cache/oof/{out}.parquet")
print(out, len(names), b.height)
