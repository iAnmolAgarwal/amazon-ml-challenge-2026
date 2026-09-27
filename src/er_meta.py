"""Meta combination of the main pipeline's pair probabilities with the in-country stack model family (src/stack).

usage: python src/er_meta.py <attempt_id>

The attempt sets MAIN (train cache/oof/<MAIN>.parquet, out-of-fold), MAIN_TEST (cache/oof/test_<MAIN_TEST>.parquet),
MIN_P (main pairs below it are left out), STACK_DIR (p_train_{India,US}.parquet: a_id, r_id, q, p2 out-of-fold;
p_test_{France,India,US}.parquet: a_id, r_id, p; scores_{e5i,e5b}_{0,1}_{heldout,test}.parquet: a_id, r_id, ce),
CASCADE (p2 floor, top k per S1: the stack's kept-pair rule, applied to train so both splits hold the same pairs),
FEATS (side-car cache/feat/{split}_<name>.parquet columns), PARAMS and ROUNDS.
Rows = union of main pairs (prob >= MIN_P) and stack pairs. Inputs: logits of both probabilities, the min / max /
sum / difference, stack cross-encoder scores, FEATS, country, and pool-side competition of each probability (rank of
the pair among the S1 claiming the pool record, number of claimers, claimers above 0.5, best other claimer, margin).
Train: 5-fold cross-fit on the harness folds -> cache/oof/<attempt>.parquet. Test: mean of the fold models for US and
India; France keeps the main test probabilities -> cache/oof/test_<attempt>.parquet.
"""
import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

import er_model as M
from common import load_attempt, load_config

KEYS = ["s1_idx", "pool_idx"]


def lg(e):
    e = e.clip(1e-6, 1 - 1e-6)
    return (e / (1 - e)).log()


def ids(cache, split):
    s = pl.read_parquet(cache / "norm" / f"{split}_s1.parquet", columns=["entity_id", "country_n"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32)).rename({"entity_id": "a_id"})
    p = pl.read_parquet(cache / "norm" / f"{split}_pool.parquet", columns=["entity_id"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32)).rename({"entity_id": "r_id"})
    return s, p


def stack(sd, split, s, p, cascade):
    cs = ["India", "US"] + (["France"] if split == "test" else [])
    d = pl.concat([pl.read_parquet(sd / f"p_{split}_{c}.parquet") for c in cs])
    d = d.join(s.select("a_id", "s1_idx"), on="a_id").join(p, on="r_id").drop("a_id", "r_id")
    if split == "train":
        d = d.rename({"q": "ps"}).with_columns(r=pl.col("p2").rank("ordinal", descending=True).over("s1_idx"))
        d = d.filter((pl.col("p2") > cascade[0]) & (pl.col("r") <= cascade[1])).drop("r", "p2")
    else:
        d = d.rename({"p": "ps"})
    kind = "heldout" if split == "train" else "test"
    for m in ["e5i", "e5b"]:
        ce = pl.concat([pl.read_parquet(sd / f"scores_{m}_{h}_{kind}.parquet", columns=["a_id", "r_id", "ce"]) for h in (0, 1)])
        ce = ce.group_by("a_id", "r_id").agg(pl.col("ce").mean().cast(pl.Float32).alias(m))
        d = d.join(ce.join(s.select("a_id", "s1_idx"), on="a_id").join(p, on="r_id").select(KEYS + [m]), on=KEYS, how="left")
    return d.select(KEYS + ["ps", "e5i", "e5b"])


def pool_side(u, c):
    v = u.filter(pl.col(c).is_not_null()).select("pool_idx", "s1_idx", c).sort(["pool_idx", c, "s1_idx"], descending=[False, True, False])
    v = v.with_columns(rk=pl.int_range(1, pl.len() + 1).over("pool_idx"), n=pl.len().over("pool_idx"), t1=pl.col(c).first().over("pool_idx"), t2=pl.col(c).slice(1, 1).first().over("pool_idx"), nh=(pl.col(c) > 0.5).sum().over("pool_idx"))
    v = v.with_columns(oth=pl.when(pl.col("rk") == 1).then(pl.col("t2")).otherwise(pl.col("t1")).fill_null(0.0))
    v = v.select(KEYS + [pl.col("rk").cast(pl.Float32).alias(f"rk_{c}"), pl.col("n").cast(pl.Float32).alias(f"n_{c}"), pl.col("nh").cast(pl.Float32).alias(f"nh_{c}"), lg(pl.col("oth")).cast(pl.Float32).alias(f"oth_{c}"), (lg(pl.col(c)) - lg(pl.col("oth"))).cast(pl.Float32).alias(f"mg_{c}")])
    return u.join(v, on=KEYS, how="left")


def build(cache, split, mod):
    s, p = ids(cache, split)
    main = pl.read_parquet(cache / "oof" / f"{mod.MAIN if split == 'train' else 'test_' + mod.MAIN_TEST}.parquet", columns=KEYS + ["prob"])
    u = main.rename({"prob": "pm"}).filter(pl.col("pm") >= mod.MIN_P).join(stack(Path(mod.STACK_DIR), split, s, p, mod.CASCADE), on=KEYS, how="full", coalesce=True)
    for f in mod.FEATS:
        u = u.join(pl.read_parquet(cache / "feat" / f"{split}_{f}.parquet"), on=KEYS, how="left")
    u = u.join(s.select("s1_idx", "country_n"), on="s1_idx").sort(KEYS)
    u = pool_side(pool_side(u, "pm"), "ps")
    u = u.with_columns(lm=lg(pl.col("pm")).cast(pl.Float32), ls=lg(pl.col("ps")).cast(pl.Float32), cty=(pl.col("country_n") == "india").cast(pl.Float32) + 2 * (pl.col("country_n") == "france").cast(pl.Float32))
    u = u.with_columns(dms=pl.col("lm") - pl.col("ls"), sms=pl.col("lm") + pl.col("ls"), mn=pl.min_horizontal("lm", "ls"), mx=pl.max_horizontal("lm", "ls"))
    return u, main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attempt_id")
    a = ap.parse_args()
    cfg = load_config()
    mod = load_attempt(a.attempt_id)
    cache = Path(cfg["cache_dir"])
    t = time.time()
    s1 = M.load_s1(cfg, "train")
    gold = M.load_gold(cfg, s1)
    fo = M.folds(cfg, s1, gold)
    u, main_tr = build(cache, "train", mod)
    u = u.join(fo.select("s1_idx", "fold"), on="s1_idx", how="left", maintain_order="left").join(gold.with_columns(y=pl.lit(1, pl.Int8)), on=KEYS, how="left", maintain_order="left")
    names = [c for c in u.columns if c not in KEYS + ["pm", "ps", "country_n", "fold", "y"]]
    X = u.select(names).to_numpy().astype(np.float32)
    y, fold = u["y"].fill_null(0).to_numpy(), u["fold"].to_numpy()
    print(f"train: {len(y):,} pairs, {int(y.sum()):,} positives, features {names}, {time.time() - t:.0f}s", flush=True)
    oof, models = np.empty(len(y)), []
    for f in range(5):
        tr, va = fold != f, fold == f
        models.append(lgb.train(mod.PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=names, free_raw_data=True), mod.ROUNDS))
        oof[va] = models[-1].predict(X[va], raw_score=True)
        print(f"fold {f}: {time.time() - t:.0f}s", flush=True)
    res = u.select(KEYS).with_columns(prob=pl.Series((1 / (1 + np.exp(-oof))).astype(np.float32)))
    res.write_parquet(cache / "oof" / f"{a.attempt_id}.parquet")
    ents = s1.select("s1_idx", "country_n").join(fo.select("s1_idx", "fold"), on="s1_idx")
    for n, pr in [(mod.MAIN, main_tr), (a.attempt_id, res)]:
        e = M.per_entity_f05(M.decide(pr, cfg, floor=mod.FLOOR).select(KEYS), gold, ents)
        print(f"{n}: CV F0.5 {e['f05'].mean():.5f}", {k: round(v, 5) for k, v in e.group_by("country_n").agg(pl.col("f05").mean()).iter_rows()}, flush=True)
    del X
    T, main_te = build(cache, "test", mod)
    T = T.filter(pl.col("country_n") != "france")
    z = np.mean([m.predict(T.select(names).to_numpy().astype(np.float32), raw_score=True) for m in models], axis=0)
    fr = main_te.join(ids(cache, "test")[0].filter(pl.col("country_n") == "france").select("s1_idx"), on="s1_idx", how="semi")
    out = pl.concat([T.select(KEYS).with_columns(prob=pl.Series((1 / (1 + np.exp(-z))).astype(np.float32))), fr.with_columns(pl.col("prob").cast(pl.Float32))])
    out.write_parquet(cache / "oof" / f"test_{a.attempt_id}.parquet")
    print(f"test: {T.height:,} US/India pairs scored, {fr.height:,} France pairs kept from {mod.MAIN_TEST}; {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
