import sys
from pathlib import Path

import numpy as np
import polars as pl

K = ["s1_idx", "pool_idx"]
d = Path("cache/r19s")
names, tag, mtr, mte = sys.argv[1].split(","), sys.argv[2], sys.argv[3], sys.argv[4]
for split, main in [("train", mtr), ("test", "test_" + mte)]:
    k = pl.read_parquet(d / f"{split}_keys.parquet")
    for n in names:
        k = k.with_columns(pl.Series(f"l_{n}", np.load(d / (f"oof_{n}.npy" if split == "train" else f"test_{n}.npy"))))
    m = pl.read_parquet(f"cache/oof/{main}.parquet", columns=K + ["prob"]).filter(pl.col("prob") >= 0.001).join(k.select(K), on=K, how="anti")
    q = m["prob"].clip(1e-6, 1 - 1e-6)
    lm = (q / (1 - q)).log().cast(pl.Float32)
    k = pl.concat([k, m.select(K).with_columns([lm.alias(f"l_{n}") for n in names])])
    k.write_parquet(f"cache/feat/{split}_r19{tag}.parquet")
    print(split, k.height, "main pairs outside the band, imputed with the main logit:", m.height)
