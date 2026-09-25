"""Test inference: train on train pairs, score test pairs, decide, write the two submission TSVs.

usage: python src/er_predict.py <attempt_id> [--tag ""] [--train-frac F]   (default: the attempt's TRAIN_FRAC)

Writes submissions/NN_<attempt_id>.tsv (matching_results format),
submissions/NN_<attempt_id>.candidates.tsv (candidate_pairs format: exactly the pairs the model scored)
and submissions/NN_<attempt_id>.json (run manifest: fraction, sample hash, features, params, source hashes).
One row per test S1 in test_source1 order; ids S2-/S3- only, in a fixed order.
Run src/er_validate.py on the Mac afterwards (it needs the raw test TSVs).
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

import er_model as M
from common import ROOT, git_commit, load_attempt, load_config, sha256


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
    ap.add_argument("--train-frac", type=float)
    a = ap.parse_args()
    cfg = load_config()
    mod = load_attempt(a.attempt_id)
    get = lambda name: getattr(mod, name, getattr(M, name))
    frac = a.train_frac or get("TRAIN_FRAC")
    cache = Path(cfg["cache_dir"])
    t0 = time.time()
    s1 = M.load_s1(cfg, "train")
    gold = M.load_gold(cfg, s1)
    extras = getattr(mod, "EXTRA_FEATS", [])

    def load_feat(split):
        f = pl.read_parquet(cache / "feat" / f"{split}_feat{a.tag}.parquet")
        for extra in extras:
            f = f.join(pl.read_parquet(cache / "feat" / f"{split}_{extra}{a.tag}.parquet"), on=["s1_idx", "pool_idx"], how="left", maintain_order="left")
        return f

    ftr = load_feat("train")
    cols = [c for c in M.feature_cols(ftr, getattr(mod, "FEATURES", None)) if c not in getattr(mod, "EXCLUDE", [])]
    rng = np.random.default_rng(cfg["seed"])
    ents = np.sort(ftr["s1_idx"].unique().to_numpy())
    keep = rng.choice(ents, size=int(len(ents) * frac), replace=False)
    tr = ftr.filter(pl.col("s1_idx").is_in(keep)) if frac < 1 else ftr
    del ftr
    tr = tr.join(gold.with_columns(y=pl.lit(1, pl.Int8)), on=["s1_idx", "pool_idx"], how="left", maintain_order="left").with_columns(pl.col("y").fill_null(0))
    yv = tr["y"].to_numpy()
    X = M.to_matrix(tr.select(cols), cols)
    del tr
    model = get("fit")(X, yv, cfg) if hasattr(mod, "fit") else M.fit(X, yv, cfg, get("PARAMS"), get("ROUNDS"))
    print(f"trained on {len(yv):,} pairs ({int(yv.sum()):,} pos) from {len(keep):,} entities in {time.time() - t0:.0f}s", flush=True)
    del X
    fte = load_feat("test")
    keys = fte.select("s1_idx", "pool_idx")
    Xt = M.to_matrix(fte.select(cols), cols)
    del fte
    prob = M.predict_chunked(model, Xt, np.arange(len(keys))) if not hasattr(mod, "predict") else get("predict")(model, Xt)
    pairs = keys.with_columns(prob=pl.Series(prob))
    (cache / "oof").mkdir(exist_ok=True)
    pairs.write_parquet(cache / "oof" / f"test_{a.attempt_id}{a.tag}.parquet")
    del Xt
    pred = get("decide")(pairs, cfg, floor=get("FLOOR"))
    ts1 = pl.read_parquet(cache / "norm" / "test_s1.parquet", columns=["entity_id"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
    tpool = pl.read_parquet(cache / "norm" / "test_pool.parquet", columns=["entity_id"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32))
    name = lambda df: df.join(ts1, on="s1_idx", maintain_order="left").join(tpool.rename({"entity_id": "pid"}), on="pool_idx", maintain_order="left")
    m = name(pred.sort(["s1_idx", "prob", "pool_idx"], descending=[False, True, False])).group_by("entity_id", maintain_order=True).agg(pl.col("pid").alias("ids"))
    assert pred.join(pairs, on=["s1_idx", "pool_idx"], how="anti").is_empty(), "a match is not a scored candidate"
    c = name(pairs.select("s1_idx", "pool_idx").sort("s1_idx", "pool_idx")).group_by("entity_id", maintain_order=True).agg(pl.col("pid").alias("ids"))
    out = ROOT / "submissions" / "archive"
    out.mkdir(exist_ok=True)
    n = next_n(out)
    mp, cp = out / f"{n:02d}_{a.attempt_id}.tsv", out / f"{n:02d}_{a.attempt_id}.candidates.tsv"
    ids = ts1["entity_id"].to_list()
    write_lists(mp, ids, m, "matched_entity_ids")
    write_lists(cp, ids, c, "candidate_entity_ids")
    src = {p.name: sha256(p) for p in sorted((ROOT / "src").glob("*.py"))} | {"train.py": sha256(ROOT / "attempts" / a.attempt_id / "train.py")}
    man = dict(attempt=a.attempt_id, tag=a.tag, train_frac=frac, n_entities=len(keep), sample_sha256=hashlib.sha256(np.sort(keep).tobytes()).hexdigest(), n_train_pairs=len(yv), n_pos=int(yv.sum()),
               features=cols, params=get("PARAMS"), rounds=get("ROUNDS"), floor=get("FLOOR"), custom_fit=hasattr(mod, "fit"), git_commit=git_commit(), src_sha256=src,
               versions=dict(python=sys.version.split()[0], polars=pl.__version__, numpy=np.__version__, lightgbm=lgb.__version__),
               matching_sha256=sha256(mp), candidates_sha256=sha256(cp), n_matched_pairs=len(pred), n_candidate_pairs=len(pairs), runtime_s=round(time.time() - t0))
    mp.with_suffix(".json").write_text(json.dumps(man, indent=1))
    k = pred.group_by("s1_idx").len()
    print(f"wrote {mp.name} and {cp.name}: {len(ids):,} S1, {len(k):,} with matches, {len(pred):,} matched pairs (mean {len(pred) / len(ids):.2f}/S1), {len(pairs):,} candidates, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
