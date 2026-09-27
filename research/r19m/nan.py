"""Side-car coverage on the meta union, train vs test, by pair source (main only / stack only / both) and country."""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
import er_meta as E
from common import load_attempt, load_config

cfg = load_config()
mod = load_attempt(sys.argv[1])
mod.FEATS = []
cache = Path(cfg["cache_dir"])
for split in ["train", "test"]:
    u, _ = E.build(cache, split, mod)
    u = u.filter(pl.col("country_n") != "france").select(E.KEYS + ["country_n"], src=pl.when(pl.col("pm").is_null()).then(pl.lit("stack")).when(pl.col("ps").is_null()).then(pl.lit("main")).otherwise(pl.lit("both")))
    for f in sys.argv[2:]:
        names = [f] if split == "train" else f.split("+")
        k = pl.concat([pl.read_parquet(cache / "feat" / f"{split}_{n}.parquet", columns=E.KEYS) for n in (names if split == "test" else [f.split("+")[0]])]).unique().with_columns(h=pl.lit(1))
        r = u.join(k, on=E.KEYS, how="left").group_by("country_n", "src").agg(n=pl.len(), cov=pl.col("h").is_not_null().mean()).sort("country_n", "src")
        s1 = u.select("s1_idx", "country_n").unique().group_by("country_n").len()
        print(split, f, r.join(s1.rename({"len": "ns1"}), on="country_n").with_columns(per_s1=pl.col("n") / pl.col("ns1")).drop("ns1"), flush=True)
