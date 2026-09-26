"""Blend our final probabilities with a second, independent pipeline's pair probabilities (same team) on the union of
both candidate sets.

usage: python src/er_blend.py <attempt_id> [--grid]

The attempt sets OURS (our attempt: cache/oof/<OURS>.parquet train OOF; test_<OURS_TEST or OURS>.parquet test), EXT_DIR (folder with
p_train_{India,US}.parquet: a_id, r_id, q = out-of-fold score; p_test_{France,India,US}.parquet: a_id, r_id, p) and the
blend parameters W (weight of our logit), FILL_EXT (their probability for pairs they do not list), FILL_OURS (our
probability for pairs we do not list) and FRANCE ("blend" or "ours").
p = sigmoid(W * logit(ours) + (1 - W) * logit(theirs)), with the fill values for one-sided pairs.
--grid evaluates W x FILL_EXT x FILL_OURS on the train union with the harness decision (choose on folds 0-2, confirm on
folds 3-4) and prints the paired gain vs OURS alone; without --grid it writes cache/oof/<attempt>.parquet (train) and
cache/oof/test_<attempt>.parquet (test) with the attempt's parameters.
"""
import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import polars as pl

import er_model as M
from common import load_attempt, load_config

KEYS = ["s1_idx", "pool_idx"]


def ids(cache, split):
    s = pl.read_parquet(cache / "norm" / f"{split}_s1.parquet", columns=["entity_id", "country_n"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32)).rename({"entity_id": "a_id"})
    p = pl.read_parquet(cache / "norm" / f"{split}_pool.parquet", columns=["entity_id"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32)).rename({"entity_id": "r_id"})
    return s, p


def theirs(ext, split, countries, s, p, col):
    d = pl.concat([pl.read_parquet(ext / f"p_{split}_{c}.parquet", columns=["a_id", "r_id", col]) for c in countries]).rename({col: "pt"})
    d = d.join(s.select("a_id", "s1_idx"), on="a_id", how="inner").join(p, on="r_id", how="inner")
    return d.select(KEYS + [pl.col("pt").cast(pl.Float32)])


def lg(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def blend(u, w, fe, fo):
    po = u["po"].fill_null(fo).to_numpy().astype(np.float64)
    pt = u["pt"].fill_null(fe).to_numpy().astype(np.float64)
    z = w * lg(po) + (1 - w) * lg(pt)
    return u.select(KEYS).with_columns(prob=pl.Series((1 / (1 + np.exp(-z))).astype(np.float32)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attempt_id")
    ap.add_argument("--grid", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    mod = load_attempt(a.attempt_id)
    cache, ext = Path(cfg["cache_dir"]), Path(mod.EXT_DIR)
    t = time.time()
    s, p = ids(cache, "train")
    ours = pl.read_parquet(cache / "oof" / f"{mod.OURS}.parquet", columns=KEYS + ["prob"]).rename({"prob": "po"})
    th = theirs(ext, "train", ["India", "US"], s, p, "q")
    u = ours.join(th, on=KEYS, how="full", coalesce=True)
    print(f"train union {u.height:,}: ours {ours.height:,}, theirs {th.height:,}, both {u.filter(pl.col('po').is_not_null() & pl.col('pt').is_not_null()).height:,}, {time.time() - t:.0f}s", flush=True)
    if a.grid:
        s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1); fo = M.folds(cfg, s1, gold)
        ents = s1.select("s1_idx", "country_n").join(fo.select("s1_idx", "fold"), on="s1_idx")
        score = lambda pr: M.per_entity_f05(M.decide(pr, cfg, floor=0.6).select(KEYS), gold, ents).select("s1_idx", "fold", "country_n", "f05")
        base = score(ours.rename({"po": "prob"}))
        print(f"ours alone F0.5 {base['f05'].mean():.6f}", flush=True)
        for w, fe, fo_ in itertools.product(mod.GRID_W, mod.GRID_FILL_EXT, mod.GRID_FILL_OURS):
            e = score(blend(u, w, fe, fo_)).join(base.select("s1_idx", x="f05"), on="s1_idx").with_columns(d=pl.col("f05") - pl.col("x"))
            d = e.group_by("fold").agg(pl.col("d").mean()).sort("fold")["d"].to_numpy()
            c = dict(e.group_by("country_n").agg(pl.col("d").mean()).iter_rows())
            print(f"W {w} fill_ext {fe} fill_ours {fo_}: F0.5 {e['f05'].mean():.6f} diff {d.mean():+.6f} folds0-2 {d[:3].mean():+.6f} folds3-4 {d[3:].mean():+.6f} (+{(d > 0).sum()}/5) us {c.get('us', 0):+.6f} india {c.get('india', 0):+.6f}, {time.time() - t:.0f}s", flush=True)
        return
    blend(u, mod.W, mod.FILL_EXT, mod.FILL_OURS).write_parquet(cache / "oof" / f"{a.attempt_id}.parquet")
    ts, tp = ids(cache, "test")
    tours = pl.read_parquet(cache / "oof" / f"test_{getattr(mod, 'OURS_TEST', mod.OURS)}.parquet", columns=KEYS + ["prob"]).rename({"prob": "po"})
    countries = ["India", "US"] + (["France"] if mod.FRANCE == "blend" else [])
    tth = theirs(ext, "test", countries, ts, tp, "p")
    tu = tours.join(tth, on=KEYS, how="full", coalesce=True)
    out = blend(tu, mod.W, mod.FILL_EXT, mod.FILL_OURS)
    if mod.FRANCE != "blend":
        fr = ts.filter(pl.col("country_n") == "france").select("s1_idx")
        out = pl.concat([out.join(fr, on="s1_idx", how="anti"), tours.join(fr, on="s1_idx", how="semi").rename({"po": "prob"})])
    out.write_parquet(cache / "oof" / f"test_{a.attempt_id}.parquet")
    print(f"test union {tu.height:,} (theirs {tth.height:,}); wrote test_{a.attempt_id}: {out.height:,} pairs, {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
