"""Paired B vs A restricted to the S1 that carry extra-key train pairs in the sampled main file (keys of S minus keys of R)."""
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import er_model as M
from common import load_config

cfg = load_config()
K = ["s1_idx", "pool_idx"]
a, b, smp, ref = sys.argv[1:5]
s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1); fo = M.folds(cfg, s1, gold)
ents = s1.select("s1_idx", "country_n").join(fo.select("s1_idx", "fold"), on="s1_idx")
sel = pl.read_parquet(f"cache/oof/{smp}.parquet", columns=K).join(pl.read_parquet(f"cache/oof/{ref}.parquet", columns=K), on=K, how="anti").select("s1_idx").unique()
ents = ents.join(sel, on="s1_idx", how="semi")
r = {n: M.per_entity_f05(M.decide(pl.read_parquet(f"cache/oof/{n}.parquet", columns=K + ["prob"]), cfg, floor=0.6).select(K), gold, ents) for n in (a, b)}
j = r[a].select("s1_idx", "fold", "country_n", x="f05").join(r[b].select("s1_idx", z="f05"), on="s1_idx").with_columns(d=pl.col("z") - pl.col("x"))
d = j.group_by("fold").agg(pl.col("d").mean()).sort("fold")["d"].to_numpy()
print(f"sample S1 {ents.height:,}: {a} {j['x'].mean():.5f} {b} {j['z'].mean():.5f} diff {d.mean():+.5f} SE {d.std(ddof=1) / np.sqrt(5):.5f} folds+ {(d > 0).sum()}/5", {k: round(v, 5) for k, v in j.group_by("country_n").agg(pl.col("d").mean()).iter_rows()})
