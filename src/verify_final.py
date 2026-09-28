"""Quick check of the last step without rerunning the chain: rebuild both output TSVs from the final probability table
and compare them with output/.

usage: python src/verify_final.py [--probs final_probs.parquet] [--attempt 029f6_final] [--out verify]
The table is left out of the package because of the upload size limit (SHA256 in artifacts.sha256; the team can
provide it).

final_probs.parquet (written by src/er_predict.py as NN_<attempt>.probs.parquet) has every candidate pair (s1_idx,
pool_idx), its final probability (prob) and whether the decision considers it (decided). Ids come from the organisers'
test TSVs: s1_idx = row of test_source1, pool_idx = row of test_source2 followed by test_source3. The decision and the
writing are those of er_predict.py (decide_and_write). Prints the SHA256 of both rebuilt files and whether they equal
output/matching_results.tsv and output/candidate_pairs.tsv.
"""
import argparse
from pathlib import Path

import polars as pl

import er_model as M
from common import ROOT, load_attempt, load_config, sha256
from er import raw_dir
from er_predict import decide_and_write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default=str(ROOT / "final_probs.parquet"))
    ap.add_argument("--attempt", default="029f6_final")
    ap.add_argument("--out", default=str(ROOT / "verify"))
    a = ap.parse_args()
    cfg = load_config()
    mod = load_attempt(a.attempt)
    get = lambda name: getattr(mod, name, getattr(M, name))
    d = raw_dir(cfg) / "test"
    ids = lambda f: pl.read_csv(d / f"{f}.tsv", separator="\t", quote_char=None, infer_schema_length=0, columns=["entity_id"])
    ts1 = ids("test_source1").with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
    tpool = pl.concat([ids("test_source2"), ids("test_source3")]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32))
    cand = pl.read_parquet(a.probs).with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col("pool_idx").cast(pl.UInt32))
    pairs = cand.filter(pl.col("decided")).select("s1_idx", "pool_idx", "prob")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mp, cp = out / "matching_results.tsv", out / "candidate_pairs.tsv"
    decide_and_write(pairs, cand, ts1, tpool, get, cfg, mp, cp)
    for f in (mp, cp):
        ref = ROOT.parent.parent / "output" / f.name
        same = ref.exists() and sha256(ref) == sha256(f)
        print(f"{f.name}: sha256 {sha256(f)}; equal to {ref}: {same}")


if __name__ == "__main__":
    main()
