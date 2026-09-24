"""Test inference: train on train pairs, score test pairs, decide, write the two submission TSVs.

usage: python src/er_predict.py <attempt_id> [--tag ""] [--train-frac 0.3]

Writes submissions/NN_<attempt_id>.tsv (matching_results format) and
submissions/NN_<attempt_id>.candidates.tsv (candidate_pairs format: every candidate the model
scored, union the matches). One row per test S1 in test_source1 order; ids S2-/S3- only.
Run src/er_validate.py on the Mac afterwards (it needs the raw test TSVs).
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import polars as pl

import er_model as M
from common import ROOT, load_attempt, load_config


def next_n(d):
    nums = [int(m.group(1)) for p in d.glob("*.tsv") if (m := re.match(r"(\d+)_", p.name))]
    return max(nums, default=0) + 1


def write_lists(path, ids, groups, col):
    """One row per S1 id; comma-joined, de-duplicated S2-/S3- ids; empty string when none."""
    g = dict(zip(groups["entity_id"].to_list(), groups["ids"].to_list()))
    with open(path, "w", newline="") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for i in ids:
            lst = [x for x in dict.fromkeys(g.get(i) or []) if x[:3] in ("S2-", "S3-")]
            f.write(f"{i}\t{','.join(lst)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attempt_id")
    ap.add_argument("--tag", default="")
    ap.add_argument("--train-frac", type=float, default=0.3)
    a = ap.parse_args()
    cfg = load_config()
    mod = load_attempt(a.attempt_id)
    get = lambda name: getattr(mod, name, getattr(M, name))
    cache = Path(cfg["cache_dir"])
    t0 = time.time()
    s1 = M.load_s1(cfg, "train")
    gold = M.load_gold(cfg, s1)
    extras = getattr(mod, "EXTRA_FEATS", [])

    def load_feat(split):
        f = pl.read_parquet(cache / "feat" / f"{split}_feat{a.tag}.parquet")
        for extra in extras:
            f = f.join(pl.read_parquet(cache / "feat" / f"{split}_{extra}{a.tag}.parquet"), on=["s1_idx", "pool_idx"], how="left")
        return f

    ftr = load_feat("train")
    cols = [c for c in M.feature_cols(ftr, getattr(mod, "FEATURES", None)) if c not in getattr(mod, "EXCLUDE", [])]
    rng = np.random.default_rng(cfg["seed"])
    ents = ftr["s1_idx"].unique().to_numpy()
    keep = rng.choice(ents, size=int(len(ents) * a.train_frac), replace=False)
    tr = ftr.filter(pl.col("s1_idx").is_in(keep)).join(gold.with_columns(y=pl.lit(1, pl.Int8)), on=["s1_idx", "pool_idx"], how="left").with_columns(pl.col("y").fill_null(0))
    del ftr
    model = get("fit")(tr.select(cols).to_numpy(), tr["y"].to_numpy(), cfg)
    print(f"trained on {len(tr):,} pairs ({int(tr['y'].sum()):,} pos) from {len(keep):,} entities in {time.time() - t0:.0f}s", flush=True)
    del tr
    fte = load_feat("test")
    pairs = fte.select("s1_idx", "pool_idx").with_columns(prob=pl.Series(get("predict")(model, fte.select(cols).to_numpy())))
    del fte
    pred = get("decide")(pairs, cfg)
    ts1 = pl.read_parquet(cache / "norm" / "test_s1.parquet", columns=["entity_id"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
    tpool = pl.read_parquet(cache / "norm" / "test_pool.parquet", columns=["entity_id"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32))
    name = lambda df: df.join(ts1, on="s1_idx").join(tpool.rename({"entity_id": "pid"}), on="pool_idx")
    m = name(pred).sort(["s1_idx", "prob"], descending=[False, True]).group_by("entity_id", maintain_order=True).agg(pl.col("pid").alias("ids"))
    c = name(pl.concat([pairs.select("s1_idx", "pool_idx"), pred.select("s1_idx", "pool_idx")]).unique()).group_by("entity_id").agg(pl.col("pid").alias("ids"))
    out = ROOT / "submissions"
    out.mkdir(exist_ok=True)
    n = next_n(out)
    mp, cp = out / f"{n:02d}_{a.attempt_id}.tsv", out / f"{n:02d}_{a.attempt_id}.candidates.tsv"
    ids = ts1["entity_id"].to_list()
    write_lists(mp, ids, m, "matched_entity_ids")
    write_lists(cp, ids, c, "candidate_entity_ids")
    k = pred.group_by("s1_idx").len()
    print(f"wrote {mp.name} and {cp.name}: {len(ids):,} S1, {len(k):,} with matches, {len(pred):,} matched pairs (mean {len(pred) / len(ids):.2f}/S1), {len(pairs):,} candidates, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
