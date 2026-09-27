import sys
from pathlib import Path

import numpy as np
import polars as pl

K = ["s1_idx", "pool_idx"]
d = Path("cache/r19s")
out, parts = sys.argv[1], sys.argv[2].split(",")
for split, pre, base in [("train", "oof_", "006_stage2"), ("test", "test_", "test_006_stage2")]:
    z = np.mean([np.load(d / f"{pre}{n}.npy") for n in parts], axis=0)
    np.save(d / f"{pre}{out}.npy", z.astype(np.float32))
    k = pl.read_parquet(d / f"{split}_keys.parquet").with_columns(p3=pl.Series((1 / (1 + np.exp(-z))).astype(np.float32)))
    b = pl.read_parquet(f"cache/oof/{base}.parquet", columns=K + ["prob"])
    b.join(k, on=K, how="left", maintain_order="left").with_columns(prob=pl.coalesce("p3", "prob").cast(pl.Float32)).select(K + ["prob"]).write_parquet(f"cache/oof/{'' if split == 'train' else 'test_'}r19_{out}.parquet")
