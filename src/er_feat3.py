"""Side-car coherence features (feat3): agreement of each candidate with the S1's other strong candidates.

usage: python src/er_feat3.py --split train|test [--tag ""] [--sample N]

Anchors (deterministic, label-free): per S1, up to 3 candidates with conservative-name token_set >= 90
AND address evidence (house number equal, or address token_set >= 80), ranked by (c_tset, word_sim).
For every candidate x of S1 s, x is compared with the anchors of s other than x itself:
- coh_n_anchor: number of such anchors; coh_self_anchor: x is itself an anchor
- coh_name_best / coh_name_2nd: best and second-best conservative-name token_set vs the anchors
- coh_name_ratio_best: best conservative-name ratio
- coh_addr_best: best normalized-address token_set; coh_addr_exact: any identical non-empty address
- coh_house_agree: any anchor with the same non-empty house number; coh_house_conflict: any anchor with a different one
- coh_postal_conflict: any anchor with a different non-empty postal code
- coh_support: anchors with (name >= 85) or (house agree and address >= 80)
- coh_support_xsrc: supporting anchors from the other source (S2 vs S3)
Candidates of S1 with no anchors get -1 for similarity features and 0 for counts.
Writes cache/feat/{split}_feat3{tag}.parquet keyed by (s1_idx, pool_idx).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from common import load_config
from er_feat2 import conservative

K_ANCHOR = 3


def pool_table(cache, split):
    """Per pool record: conservative name, normalized address, house, postal, source flag."""
    n = pl.read_parquet(cache / "norm" / f"{split}_pool.parquet", columns=["entity_id", "name_lat", "addr_norm", "house", "postal"]).with_row_index("pool_idx")
    return n.with_columns(
        pl.col("pool_idx").cast(pl.UInt32),
        cname=pl.Series([conservative(x) for x in n["name_lat"].to_list()]),
        s3=(pl.col("entity_id").str.slice(0, 3) == "S3-").cast(pl.Int8),
    ).drop("entity_id", "name_lat")


def anchors(cand):
    """Up to K_ANCHOR anchors per S1 from candidate features (label-free)."""
    a = cand.filter((pl.col("c_tset") >= 90) & ((pl.col("house_eq") == 1) | (pl.col("a_tset") >= 80)))
    a = a.sort(["s1_idx", "c_tset", "word_sim", "pool_idx"], descending=[False, True, True, False])
    return a.with_columns(r=pl.int_range(pl.len()).over("s1_idx")).filter(pl.col("r") < K_ANCHOR).select("s1_idx", pl.col("pool_idx").alias("anchor"))


def coherence(pairs, pool):
    """pairs: s1_idx, pool_idx, anchor (anchor != pool_idx). Returns per (s1_idx, pool_idx) aggregates."""
    x = pool.select(pl.col("pool_idx"), pl.col("cname").alias("cx"), pl.col("addr_norm").alias("ax"), pl.col("house").alias("hx"), pl.col("postal").alias("zx"), pl.col("s3").alias("sx"))
    y = pool.select(pl.col("pool_idx").alias("anchor"), pl.col("cname").alias("cy"), pl.col("addr_norm").alias("ay"), pl.col("house").alias("hy"), pl.col("postal").alias("zy"), pl.col("s3").alias("sy"))
    p = pairs.join(x, on="pool_idx", how="left").join(y, on="anchor", how="left")
    cx, cy, ax, ay = p["cx"].to_list(), p["cy"].to_list(), p["ax"].to_list(), p["ay"].to_list()
    name = cpdist(cx, cy, scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32)
    nratio = cpdist(cx, cy, scorer=fuzz.ratio, workers=-1).astype(np.float32)
    addr = cpdist(ax, ay, scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32)
    p = p.with_columns(name=pl.Series(name), nratio=pl.Series(nratio), addr=pl.Series(np.where((p["ax"] == "").to_numpy() | (p["ay"] == "").to_numpy(), -1.0, addr).astype(np.float32)))
    both_h = (pl.col("hx") != "") & (pl.col("hy") != "")
    both_z = (pl.col("zx") != "") & (pl.col("zy") != "")
    p = p.with_columns(
        h_agree=(both_h & (pl.col("hx") == pl.col("hy"))).cast(pl.Int8),
        h_conf=(both_h & (pl.col("hx") != pl.col("hy"))).cast(pl.Int8),
        z_conf=(both_z & (pl.col("zx") != pl.col("zy"))).cast(pl.Int8),
        a_exact=((pl.col("ax") != "") & (pl.col("ax") == pl.col("ay"))).cast(pl.Int8),
    ).with_columns(sup=((pl.col("name") >= 85) | ((pl.col("h_agree") == 1) & (pl.col("addr") >= 80))).cast(pl.Int8))
    p = p.with_columns(sup_x=((pl.col("sup") == 1) & (pl.col("sx") != pl.col("sy"))).cast(pl.Int8))
    return p.group_by(["s1_idx", "pool_idx"]).agg(
        coh_n_anchor=pl.len().cast(pl.Float32),
        coh_name_best=pl.col("name").max(),
        coh_name_2nd=pl.col("name").sort(descending=True).get(1, null_on_oob=True),
        coh_name_ratio_best=pl.col("nratio").max(),
        coh_addr_best=pl.col("addr").max(),
        coh_addr_exact=pl.col("a_exact").max().cast(pl.Float32),
        coh_house_agree=pl.col("h_agree").max().cast(pl.Float32),
        coh_house_conflict=pl.col("h_conf").max().cast(pl.Float32),
        coh_postal_conflict=pl.col("z_conf").max().cast(pl.Float32),
        coh_support=pl.col("sup").sum().cast(pl.Float32),
        coh_support_xsrc=pl.col("sup_x").sum().cast(pl.Float32),
    )


def build(split, tag, sample=None, s1_batch=150_000):
    cfg = load_config()
    cache = Path(cfg["cache_dir"])
    t = time.time()
    f1 = pl.scan_parquet(cache / "feat" / f"{split}_feat{tag}.parquet").select("s1_idx", "pool_idx", "word_sim", "house_eq", "a_tset")
    f2 = pl.scan_parquet(cache / "feat" / f"{split}_feat2{tag}.parquet").select("s1_idx", "pool_idx", "c_tset")
    cand = f1.join(f2, on=["s1_idx", "pool_idx"], how="left").collect()
    if sample:
        keep = cand["s1_idx"].unique().sample(sample, seed=42)
        cand = cand.filter(pl.col("s1_idx").is_in(keep.implode()))
    anc = anchors(cand)
    is_anchor = anc.rename({"anchor": "pool_idx"}).with_columns(coh_self_anchor=pl.lit(1.0, pl.Float32))
    pool = pool_table(cache, split)
    print(f"{len(cand):,} candidates, {len(anc):,} anchors ({anc['s1_idx'].n_unique():,} S1 with >=1) in {time.time() - t:.0f}s", flush=True)
    keys = cand.select("s1_idx", "pool_idx")
    s1s = keys["s1_idx"].unique().sort()
    out_dir = cache / "feat"
    parts = []
    for b, s in enumerate(range(0, len(s1s), s1_batch)):
        tb = time.time()
        ids = s1s.slice(s, s1_batch).implode()
        kb = keys.filter(pl.col("s1_idx").is_in(ids))
        pairs = kb.join(anc.filter(pl.col("s1_idx").is_in(ids)), on="s1_idx", how="inner").filter(pl.col("anchor") != pl.col("pool_idx"))
        agg = coherence(pairs, pool) if len(pairs) else None
        res = kb.join(agg, on=["s1_idx", "pool_idx"], how="left") if agg is not None else kb
        res = res.join(is_anchor, on=["s1_idx", "pool_idx"], how="left").with_columns(
            [pl.col(c).fill_null(-1.0) for c in ("coh_name_best", "coh_name_2nd", "coh_name_ratio_best", "coh_addr_best")]
            + [pl.col(c).fill_null(0.0) for c in ("coh_n_anchor", "coh_addr_exact", "coh_house_agree", "coh_house_conflict", "coh_postal_conflict", "coh_support", "coh_support_xsrc", "coh_self_anchor")]
        )
        part = out_dir / f"{split}_feat3{tag}{'_sample' if sample else ''}.part{b:04d}.parquet"
        res.write_parquet(part)
        parts.append(part)
        print(f"  batch {b + 1}/{(len(s1s) + s1_batch - 1) // s1_batch}: {len(pairs):,} comparisons in {time.time() - tb:.0f}s", flush=True)
    final = out_dir / (f"{split}_feat3{tag}_sample.parquet" if sample else f"{split}_feat3{tag}.parquet")
    pl.scan_parquet(parts).sink_parquet(final)
    for part in parts:
        part.unlink()
    print(f"wrote {final.name} ({pl.scan_parquet(final).select(pl.len()).collect().item():,} rows) in {time.time() - t:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--sample", type=int)
    a = ap.parse_args()
    build(a.split, a.tag, a.sample)


if __name__ == "__main__":
    main()
