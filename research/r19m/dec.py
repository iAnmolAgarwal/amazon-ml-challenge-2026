"""Decision-layer sweep on a meta OOF: per-country floor and EF prefix vs plain threshold; pick on folds 0-2, confirm on 3-4."""
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import er_model as M
from common import load_config

cfg = load_config()
K = ["s1_idx", "pool_idx"]
name = sys.argv[1]
s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1); fo = M.folds(cfg, s1, gold)
ents = s1.select("s1_idx", "country_n").join(fo.select("s1_idx", "fold"), on="s1_idx")
p = pl.read_parquet(f"cache/oof/{name}.parquet", columns=K + ["prob"])


def score(pred):
    e = M.per_entity_f05(pred.select(K), gold, ents)
    return e.select("s1_idx", "fold", "country_n", "f05")


def plain(p, t):
    d = p.sort(["pool_idx", "prob", "s1_idx"], descending=[False, True, False]).unique("pool_idx", keep="first", maintain_order=True)
    return d.filter(pl.col("prob") > t)


base = score(M.decide(p, cfg, floor=0.6)).rename({"f05": "b"})
rows = []
dec0 = M.decide(p, cfg, floor=0.0)
for fl in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
    rows.append(("ef", fl, score(dec0.filter(pl.col("prob") > fl))))
for t in [0.4, 0.5, 0.6, 0.7]:
    rows.append(("plain", t, score(plain(p, t))))
for er in [0.8, 1.2, 1.5]:
    rows.append((f"ef_er{er}", 0.6, score(M.decide(p, cfg, floor=0.6, empty_ratio=er))))
for kind, v, e in rows:
    j = e.join(base.select("s1_idx", "b"), on="s1_idx").with_columns(d=pl.col("f05") - pl.col("b"))
    g = j.group_by("country_n", "fold").agg(pl.col("d").mean(), n=pl.len()).sort("country_n", "fold")
    out = []
    for c in ["india", "us"]:
        d = g.filter(pl.col("country_n") == c)["d"].to_numpy()
        out.append(f"{c}: f0-2 {d[:3].mean():+.6f} f3-4 {d[3:].mean():+.6f} ({(d > 0).sum()}/5)")
    print(kind, v, " | ".join(out), flush=True)
