"""Avail blend (mean logit of main and stack where both exist, else the one present) on the meta union, train and test,
plus per-country mixes with a meta file: hybrid (avail for one country) and probability / logit averages."""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import er_meta as E
from common import load_attempt, load_config

cfg = load_config()
att, metas = sys.argv[1], sys.argv[2:]
mod = load_attempt(att)
mod.FEATS = []
cache = Path(cfg["cache_dir"])
for split, pre in [("train", ""), ("test", "test_")]:
    u, main = E.build(cache, split, mod)
    fr = u.filter(pl.col("country_n") == "france").select(E.KEYS)
    u = u.filter(pl.col("country_n") != "france").with_columns(av=pl.coalesce((pl.col("lm") + pl.col("ls")) / 2, "lm", "ls")).select(E.KEYS + ["country_n", "av"])
    av = u.with_columns(prob=(1 / (1 + (-pl.col("av")).exp())).cast(pl.Float32))
    if split == "test":
        av = pl.concat([av.select(E.KEYS + ["prob"]), main.join(fr, on=E.KEYS, how="semi").with_columns(pl.col("prob").cast(pl.Float32))])
    av.select(E.KEYS + ["prob"]).write_parquet(cache / "oof" / f"{pre}m19_av_{att}.parquet")
    for m in metas:
        z = pl.read_parquet(cache / "oof" / f"{pre}{m}.parquet").join(u, on=E.KEYS, how="left")
        lz = E.lg(pl.col("prob"))
        pa = 1 / (1 + (-pl.col("av")).exp())
        for tag, e in [("hi", pl.when(pl.col("country_n") == "india").then(pa).otherwise(pl.col("prob"))),
                       ("pa50", pl.when(pl.col("av").is_not_null()).then(0.5 * pl.col("prob") + 0.5 * pa).otherwise(pl.col("prob"))),
                       ("la50", pl.when(pl.col("av").is_not_null()).then(1 / (1 + (-(0.5 * lz + 0.5 * pl.col("av"))).exp())).otherwise(pl.col("prob"))),
                       ("la50i", pl.when(pl.col("av").is_not_null() & (pl.col("country_n") == "india")).then(1 / (1 + (-(0.5 * lz + 0.5 * pl.col("av"))).exp())).otherwise(pl.col("prob"))),
                       ("la25", pl.when(pl.col("av").is_not_null()).then(1 / (1 + (-(0.75 * lz + 0.25 * pl.col("av"))).exp())).otherwise(pl.col("prob")))]:
            z.select(E.KEYS, prob=e.cast(pl.Float32)).write_parquet(cache / "oof" / f"{pre}m19_{tag}_{m}.parquet")
    print(split, "done", flush=True)
