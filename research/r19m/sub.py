"""Paired difference on the S1 subgroup that has an in-band (0.1 < main p < 0.99) pair without the side-car column."""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
import er_meta as E
import er_model as M
from common import load_attempt, load_config

cfg = load_config()
a, b, col, att = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
mod = load_attempt(att)
cache = Path(cfg["cache_dir"])
s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1); fo = M.folds(cfg, s1, gold)
ents = s1.select("s1_idx", "country_n").join(fo.select("s1_idx", "fold"), on="s1_idx")
u, _ = E.build(cache, "train", mod)
flag = u.filter((pl.col("pm") > 0.1) & (pl.col("pm") < 0.99) & pl.col(col).is_null()).select("s1_idx").unique().with_columns(g=pl.lit(1))
r = {}
for n in (a, b):
    p = pl.read_parquet(cache / "oof" / f"{n}.parquet")
    r[n] = M.per_entity_f05(M.decide(p, cfg, floor=0.6).select(E.KEYS), gold, ents).select("s1_idx", "fold", "country_n", pl.col("f05").alias(n))
j = r[a].join(r[b].select("s1_idx", b), on="s1_idx").join(flag, on="s1_idx", how="left").with_columns(pl.col("g").fill_null(0), d=pl.col(b) - pl.col(a))
print(j.group_by("country_n", "g").agg(n=pl.len(), d=pl.col("d").mean(), dsum=pl.col("d").sum()).sort("country_n", "g"))
print(j.group_by("g", "fold").agg(pl.col("d").mean()).sort("g", "fold"))
