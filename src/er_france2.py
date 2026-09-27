"""France rule on test with a third input: the logit blend of er_france.py (main pA, stack family pS, W) mixed with an
extra France score z3 on the logit scale, then the decoy-word veto.

usage: python src/er_france2.py <attempt_id>

The attempt sets everything er_france.py reads (PARENT, FRANCE_BASE, STACK_DIR, W, VETO_WORDS, BROAD_WORDS) plus
THIRD (parquet with s1_idx, pool_idx and a logit column), THIRD_COL (that column) and C (weight of the third input).
Per France pair: z2 = W * logit(pA) + (1 - W) * logit(pS) when both exist, else the logit of the one that exists;
z = (1 - C) * z2 + C * z3 where z3 exists, else z2; p = sigmoid(z); pairs with a VETO word added get p = 0.
Writes cache/oof/test_<attempt>.parquet (s1_idx, pool_idx, prob), read by er_predict.py --from-probs.
"""
import argparse
import time
from pathlib import Path

import polars as pl

from common import load_attempt, load_config
from er_france import KEYS, logit, words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attempt_id")
    a = ap.parse_args()
    cfg = load_config()
    mod = load_attempt(a.attempt_id)
    cache = Path(cfg["cache_dir"])
    t = time.time()
    s1 = pl.read_parquet(cache / "norm" / "test_s1.parquet", columns=["entity_id", "country_n"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
    pool = pl.read_parquet(cache / "norm" / "test_pool.parquet", columns=["entity_id"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32))
    fr = s1.filter(pl.col("country_n") == "france").select("s1_idx", a_id="entity_id")
    parent = pl.read_parquet(cache / "oof" / f"test_{mod.PARENT}.parquet", columns=KEYS + ["prob"])
    base = pl.read_parquet(cache / "oof" / f"test_{mod.FRANCE_BASE}.parquet", columns=KEYS + ["prob"]).join(fr.select("s1_idx"), on="s1_idx")
    stack = pl.read_parquet(Path(mod.STACK_DIR) / "p_test_France.parquet").join(fr, on="a_id").join(pool.rename({"entity_id": "r_id"}), on="r_id").select(KEYS + [pl.col("p").alias("pS")])
    third = pl.read_parquet(mod.THIRD).select(pl.col("s1_idx").cast(pl.UInt32), pl.col("pool_idx").cast(pl.UInt32), pl.col(mod.THIRD_COL).cast(pl.Float64).alias("z3"))
    n1 = pl.read_parquet(cache / "raw" / "test_source1.parquet", columns=["business_name"]).with_row_index("s1_idx").select(pl.col("s1_idx").cast(pl.UInt32), ts=words("business_name"))
    n2 = pl.concat([pl.read_parquet(cache / "raw" / f"test_source{i}.parquet", columns=["business_name"]) for i in (2, 3)]).with_row_index("pool_idx").select(pl.col("pool_idx").cast(pl.UInt32), tp=words("business_name"))
    d = base.rename({"prob": "pA"}).join(stack, on=KEYS, how="full", coalesce=True).join(third, on=KEYS, how="left").join(n1, on="s1_idx", how="left").join(n2, on="pool_idx", how="left")
    adds = lambda ws: pl.any_horizontal([pl.col("tp").list.contains(w) & ~pl.col("ts").list.contains(w) for w in ws])
    d = d.with_columns(veto=adds(mod.VETO_WORDS), broad=adds(mod.BROAD_WORDS)).drop("ts", "tp")
    d = d.with_columns(pS=pl.when((pl.col("pS") == 0) & pl.col("broad") & ~pl.col("veto")).then(None).otherwise(pl.col("pS")))
    z2 = pl.when(pl.col("pA").is_null()).then(logit(pl.col("pS"))).when(pl.col("pS").is_null()).then(logit(pl.col("pA"))).otherwise(mod.W * logit(pl.col("pA")) + (1 - mod.W) * logit(pl.col("pS")))
    z = pl.when(pl.col("z3").is_null()).then(z2).otherwise((1 - mod.C) * z2 + mod.C * pl.col("z3"))
    d = d.with_columns(prob=pl.when(pl.col("veto")).then(0.0).otherwise(1 / (1 + (-z).exp())).fill_null(0.0).cast(pl.Float32))
    out = pl.concat([parent.join(fr.select("s1_idx"), on="s1_idx", how="anti").with_columns(pl.col("prob").cast(pl.Float32)), d.select(KEYS + ["prob"])]).sort(KEYS)
    out.write_parquet(cache / "oof" / f"test_{a.attempt_id}.parquet")
    print(f"France: {base.height:,} main pairs, {stack.height:,} stack pairs, {d.filter(pl.col('z3').is_not_null()).height:,} with the third input (C={mod.C}), {d.filter(pl.col('veto')).height:,} vetoed; wrote test_{a.attempt_id}: {out.height:,} pairs, {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
