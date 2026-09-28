"""Label-free derivation of the France decoy-word lists used by er_france.py (VETO_WORDS, BROAD_WORDS).

usage: python src/er_frwords.py [--country france] [--min-shift 300]

Over the country's test candidate pairs (blocking candidates, no labels), pairs whose pool name equals the S1 name
plus exactly one inserted word are split by house relation on the same street: equal house number vs a typical decoy
shift (|pool - S1| in 1, 2, 3, 4, 5, 7, 9, 11, 13, 21). Legal forms are left out (legal-form changes have their own
features). For each inserted word with at least --min-shift shifted insertions: shift_share = n_shift / (n_shift + n_equal).
- VETO words: shift_share >= 0.97 (the word is inserted almost only into decoys; a pure decoy marker);
- BROAD words: 0.60 <= shift_share < 0.97 (a decoy marker that also appears as a filler in true matches).
Prints the table and both lists.
"""
import argparse
from pathlib import Path

import polars as pl

from common import load_config

SHIFTS = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
LEGAL = ["sarl", "sas", "sa", "sasu", "sci", "eurl", "snc", "scop", "scp", "selarl", "ei", "eirl", "sca", "gie", "llc", "inc", "corp", "co", "ltd", "limited", "pvt", "private", "llp", "public", "plc", "lp", "company", "corporation", "incorporated"]


def toks(c):
    return pl.col(c).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_lowercase().str.extract_all(r"[a-z]+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="france")
    ap.add_argument("--min-shift", type=int, default=300)
    a = ap.parse_args()
    cache = Path(load_config()["cache_dir"])
    num = lambda c: pl.col(c).str.extract(r"(\d+)").cast(pl.Int64, strict=False)
    s1 = pl.read_parquet(cache / "raw" / "test_source1.parquet", columns=["business_name"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32), t1=toks("business_name")).drop("business_name")
    s1 = s1.join(pl.read_parquet(cache / "norm" / "test_s1.parquet", columns=["country_n", "house", "street1"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32)), on="s1_idx").filter(pl.col("country_n") == a.country)
    s1 = s1.select("s1_idx", "t1", h1=num("house"), st1="street1")
    pool = pl.concat([pl.read_parquet(cache / "raw" / f"test_source{i}.parquet", columns=["business_name"]) for i in (2, 3)]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32), t2=toks("business_name")).drop("business_name")
    pool = pool.join(pl.read_parquet(cache / "norm" / "test_pool.parquet", columns=["house", "street1"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32)), on="pool_idx").select("pool_idx", "t2", h2=num("house"), st2="street1")
    c = pl.read_parquet(cache / "cand" / "test_cand.parquet", columns=["s1_idx", "pool_idx"]).with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col("pool_idx").cast(pl.UInt32))
    d = c.join(s1, on="s1_idx").join(pool, on="pool_idx")
    d = d.with_columns(add=pl.col("t2").list.set_difference("t1"), rem=pl.col("t1").list.set_difference("t2"))
    d = d.filter((pl.col("add").list.len() == 1) & (pl.col("rem").list.len() == 0) & (pl.col("st1") == pl.col("st2")) & (pl.col("st1").fill_null("") != ""))
    d = d.with_columns(w=pl.col("add").list.first(), dh=(pl.col("h2") - pl.col("h1")).abs()).filter(~pl.col("w").is_in(LEGAL))
    d = d.with_columns(rel=pl.when(pl.col("dh") == 0).then(pl.lit("equal")).when(pl.col("dh").is_in(SHIFTS)).then(pl.lit("shift")).otherwise(None)).filter(pl.col("rel").is_not_null())
    t = d.group_by("w").agg(n_shift=(pl.col("rel") == "shift").sum(), n_equal=(pl.col("rel") == "equal").sum())
    t = t.filter(pl.col("n_shift") >= a.min_shift).with_columns(shift_share=(pl.col("n_shift") / (pl.col("n_shift") + pl.col("n_equal"))).round(4)).sort("n_shift", descending=True)
    with pl.Config(tbl_rows=40):
        print(t)
    print("VETO_WORDS", sorted(t.filter(pl.col("shift_share") >= 0.97)["w"].to_list()))
    print("BROAD_WORDS", sorted(t.filter((pl.col("shift_share") >= 0.60) & (pl.col("shift_share") < 0.97))["w"].to_list()))


if __name__ == "__main__":
    main()
