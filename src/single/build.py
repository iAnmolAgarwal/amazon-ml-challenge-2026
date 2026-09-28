"""Single-model feature table: every stage-2 band pair (stage-2 probability >= 0.001) with all side-car feature
families (pair, frequency, competition, generator-operator, cross-encoder) and pool-side contrasts.

usage: python src/single/build.py <out_dir> <train stage-2 attempt> <test stage-2 attempt> train test
Writes <out_dir>/{split}_keys.parquet, {split}_X.npy (float32 memmap) and names.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import er_model as M
from common import load_config
from er_pool import contrasts, pool_side

K = ["s1_idx", "pool_idx"]
FE = ["feat", "feat2", "feat4", "feat5p", "feat6l", "gen", "ce"]
EX = ["a_ratio", "a_partial"]
PC = ["n_tset", "n_jw", "a_tset", "num_jacc", "house_eq", "street_eq", "c_jw", "ns_cos", "as_cos", "lg_eq", "dm_add"]
cfg = load_config()
cache = Path(cfg["cache_dir"])
out = Path(sys.argv[1])
out.mkdir(exist_ok=True)


def build(split):
    t = time.time()
    src = sys.argv[2] if split == "train" else "test_" + sys.argv[3]
    p = pool_side(pl.read_parquet(cache / "oof" / f"{src}.parquet", columns=K + ["prob"]))
    b = p.filter(pl.col("prob") >= 0.001).rename({"prob": "p006"})
    b = b.join(M.load_s1(cfg, split).select("s1_idx", "country_n"), on="s1_idx", how="left").with_columns(cty=(pl.col("country_n") == "india").cast(pl.Float32) + 2 * (pl.col("country_n") == "france").cast(pl.Float32)).drop("country_n")
    del p
    for f in FE:
        x = pl.scan_parquet(cache / "feat" / f"{split}_{f}.parquet").join(b.lazy().select(K), on=K, how="semi").collect()
        x = x.drop([c for c in EX if c in x.columns])
        x = x.rename({c: f"{c}_{f}" for c in x.columns if c not in K and c in b.columns})
        b = b.join(x, on=K, how="left")
        print(split, f, b.width, f"{time.time() - t:.0f}s", flush=True)
    b = b.join(contrasts(cache, split, FE[:5], PC, b.select(K)), on=K, how="left")
    b = b.sort(K)
    names = [c for c in b.columns if c not in K]
    b.select(K).write_parquet(out / f"{split}_keys.parquet")
    X = np.lib.format.open_memmap(out / f"{split}_X.npy", mode="w+", dtype=np.float32, shape=(b.height, len(names)))
    for j, c in enumerate(names):
        X[:, j] = b[c].cast(pl.Float32).to_numpy()
    X.flush()
    json.dump(names, open(out / "names.json", "w"))
    print(split, b.height, len(names), f"{time.time() - t:.0f}s", flush=True)


for s in sys.argv[4:]:
    build(s)
