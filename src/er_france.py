"""France rule on test: blend of the main France probabilities with the in-country stack model family's France
probabilities (src/stack), on the logit scale.

usage: python src/er_france.py <attempt_id>

The attempt sets PARENT (test attempt whose US/India rows are kept as they are), FRANCE_BASE (test attempt whose France
rows give the main probability pA), STACK_DIR (folder with p_test_France.parquet: a_id, r_id, p = pS), W (weight of the
main logit), VETO_WORDS, BROAD_WORDS, and optionally C / THIRD / THIRD_COL (default C = 0) and RENAME (default False).
Per France pair (union of both sets): the stack family's zeros on pairs whose pool name adds a BROAD word count as
missing; p = sigmoid(W * logit(pA) + (1 - W) * logit(pS)) when both exist, else the one that exists; pairs whose pool
name has a VETO word that the S1 name does not get p = 0.
C > 0: where THIRD (parquet with s1_idx, pool_idx and the logit column THIRD_COL) has a score z3, the blend logit z becomes
(1 - C) * z + C * z3.
RENAME: on renamed pairs (the pool name shares no core-name token with the S1 name, same house number, a shared street
word, and the S1 is the only one of its country at that house number and street-word set) p = max(p, pS) before the veto.
Writes cache/oof/test_<attempt>.parquet (s1_idx, pool_idx, prob), read by er_predict.py --from-probs.
"""
import argparse
import time
from pathlib import Path

import polars as pl

from common import load_attempt, load_config

KEYS = ["s1_idx", "pool_idx"]


def words(c):
    return pl.col(c).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars().str.split(" ")


def logit(e):
    e = e.clip(1e-6, 1 - 1e-6)
    return (e / (1 - e)).log()


def rename_pairs(cache, d):
    """Flags the renamed pairs of d (s1_idx, pool_idx) at the S1's own unique address."""
    toks = lambda c: words(c).list.eval(pl.element().filter(pl.element() != ""))
    s = pl.read_parquet(cache / "raw" / "test_source1.parquet", columns=["business_address"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32), t=toks("business_address"))
    s = s.join(pl.read_parquet(cache / "norm" / "test_s1.parquet", columns=["country_n", "name_core"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32)), on="s1_idx").filter(pl.col("country_n") == "france")
    p = pl.concat([pl.read_parquet(cache / "raw" / f"test_source{i}.parquet", columns=["business_address"]) for i in (2, 3)]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32), t=toks("business_address"))
    p = p.join(pl.read_parquet(cache / "norm" / "test_pool.parquet", columns=["name_core"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32)), on="pool_idx").join(d.select("pool_idx").unique(), on="pool_idx")
    stop = s.select(w="t").explode("w").group_by("w").len().sort("len", descending=True).head(150)["w"].to_list()
    view = lambda x: x.select(pl.exclude("t", "name_core", "business_address", "country_n"), h=pl.col("t").list.eval(pl.element().filter(pl.element().str.contains(r"^\d{1,5}$"))).list.first().str.replace(r"^0+", ""),
                              w=pl.col("t").list.eval(pl.element().filter((pl.element().str.len_chars() > 2) & ~pl.element().str.contains(r"\d"))).list.set_difference(pl.lit(stop)).list.unique().list.sort(),
                              n=pl.col("name_core").fill_null("").str.split(" ").list.unique())
    s, p = view(s), view(p)
    s = s.with_columns(nk=pl.when(pl.col("h").is_not_null() & (pl.col("w").list.len() > 0)).then(pl.len().over("h", pl.col("w").list.join(" "))))
    x = d.select(KEYS).join(s, on="s1_idx").join(p, on="pool_idx", suffix="_p")
    x = x.filter((pl.col("n").list.set_intersection("n_p").list.len() == 0) & (pl.col("h") == pl.col("h_p")) & (pl.col("w").list.set_intersection("w_p").list.len() > 0) & (pl.col("nk") == 1))
    return d.join(x.select(KEYS, rn=pl.lit(True)), on=KEYS, how="left").with_columns(pl.col("rn").fill_null(False))


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
    n1 = pl.read_parquet(cache / "raw" / "test_source1.parquet", columns=["business_name"]).with_row_index("s1_idx").select(pl.col("s1_idx").cast(pl.UInt32), ts=words("business_name"))
    n2 = pl.concat([pl.read_parquet(cache / "raw" / f"test_source{i}.parquet", columns=["business_name"]) for i in (2, 3)]).with_row_index("pool_idx").select(pl.col("pool_idx").cast(pl.UInt32), tp=words("business_name"))
    d = base.rename({"prob": "pA"}).join(stack, on=KEYS, how="full", coalesce=True).join(n1, on="s1_idx", how="left").join(n2, on="pool_idx", how="left")
    adds = lambda ws: pl.any_horizontal([pl.col("tp").list.contains(w) & ~pl.col("ts").list.contains(w) for w in ws])
    d = d.with_columns(veto=adds(mod.VETO_WORDS), broad=adds(mod.BROAD_WORDS)).drop("ts", "tp")
    d = d.with_columns(pS=pl.when((pl.col("pS") == 0) & pl.col("broad") & ~pl.col("veto")).then(None).otherwise(pl.col("pS")))
    both = pl.col("pA").is_not_null() & pl.col("pS").is_not_null()
    blend = 1 / (1 + (-(mod.W * logit(pl.col("pA")) + (1 - mod.W) * logit(pl.col("pS")))).exp())
    d = d.with_columns(prob=pl.when(both).then(blend).otherwise(pl.coalesce("pA", "pS")))
    if getattr(mod, "C", 0):
        z3 = pl.read_parquet(mod.THIRD).select(pl.col("s1_idx").cast(pl.UInt32), pl.col("pool_idx").cast(pl.UInt32), pl.col(mod.THIRD_COL).cast(pl.Float64).alias("z3"))
        d = d.join(z3, on=KEYS, how="left").with_columns(prob=pl.when(pl.col("z3").is_null() | pl.col("prob").is_null()).then(pl.col("prob")).otherwise(1 / (1 + (-((1 - mod.C) * logit(pl.col("prob")) + mod.C * pl.col("z3"))).exp()))).drop("z3")
    if getattr(mod, "RENAME", False):
        d = rename_pairs(cache, d).with_columns(prob=pl.when(pl.col("rn") & pl.col("pS").is_not_null()).then(pl.max_horizontal("prob", "pS")).otherwise(pl.col("prob"))).drop("rn")
    d = d.with_columns(prob=pl.when(pl.col("veto")).then(0.0).otherwise(pl.col("prob")).fill_null(0.0).cast(pl.Float32))
    out = pl.concat([parent.join(fr.select("s1_idx"), on="s1_idx", how="anti").with_columns(pl.col("prob").cast(pl.Float32)), d.select(KEYS + ["prob"])]).sort(KEYS)
    out.write_parquet(cache / "oof" / f"test_{a.attempt_id}.parquet")
    print(f"France: {base.height:,} main pairs, {stack.height:,} stack pairs, {d.filter(both).height:,} in both, {d.filter(pl.col('pA').is_null()).height:,} stack only, {d.filter(pl.col('veto')).height:,} vetoed; wrote test_{a.attempt_id}: {out.height:,} pairs, {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
