import sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, "src")
import er_fix2 as F2
from common import load_config

cfg = load_config(); cache = Path(cfg["cache_dir"]); K = ["s1_idx", "pool_idx"]
split, parent, full, out = sys.argv[1:5]
pre = "" if split == "train" else "test_"
_, ps = F2.s1_side(cache, split)
flag = ps.select(pl.col("pool_idx").cast(pl.UInt32), emp=pl.col("emp"), pw=(pl.col("pw_share").fill_null(0) == 1) & ~pl.col("business_name").fill_null("").str.contains(r"[ऀ-෿]"))
a = pl.read_parquet(cache / "oof" / f"{pre}{parent}.parquet", columns=K + ["prob"]).rename({"prob": "p0"})
b = pl.read_parquet(cache / "oof" / f"{pre}{full}.parquet", columns=K + ["prob"]).rename({"prob": "p1"})
d = a.join(b, on=K, how="left", maintain_order="left").join(flag, on="pool_idx", how="left", maintain_order="left")
up = pl.col("emp") | (pl.col("pw") if "pw" in out else pl.lit(False))
d.select(K, prob=pl.when(up).then(pl.col("p1")).otherwise(pl.min_horizontal("p0", "p1")).cast(pl.Float32)).write_parquet(cache / "oof" / f"{pre}{out}.parquet")
print(out, d.filter(pl.col("pw") & (pl.col("p1") > pl.col("p0"))).height)
