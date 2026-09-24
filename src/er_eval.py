"""Cross-validation for entity-resolution attempts on cached candidates and features.

usage: python src/er_eval.py <attempt_id> [--tag _s20k] [--train-frac 0.2] [--folds 0,1,2,3,4]

Every training pair gets an out-of-fold probability from a model that never saw its S1 entity
(trained on a TRAIN_FRAC sample of the other folds' entities). The decision rule then runs once on
the full out-of-fold graph, so rival S1 entities compete exactly as they will on test. Scores are
per-entity F0.5 averaged within each fold (macro F0.5, singletons included); entities without
candidates count with an empty prediction. Writes attempts/<id>/results.json and entity_scores.parquet.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

import er_model as M
from common import ROOT, git_commit, load_attempt, load_config

TEST_WEIGHTS = {"us": 0.38, "india": 0.47}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attempt_id")
    ap.add_argument("--tag", default="")
    ap.add_argument("--train-frac", type=float)
    ap.add_argument("--folds")
    a = ap.parse_args()
    cfg = load_config()
    mod = load_attempt(a.attempt_id)
    get = lambda name: getattr(mod, name, getattr(M, name))
    t_all = time.time()
    s1 = M.load_s1(cfg, "train")
    gold = M.load_gold(cfg, s1)
    fo = M.folds(cfg, s1, gold)
    feat = pl.read_parquet(Path(cfg["cache_dir"]) / "feat" / f"train_feat{a.tag}.parquet")
    side = Path(cfg["cache_dir"]) / "cand" / f"train_s1{a.tag}.parquet"
    ents = s1.join(pl.read_parquet(side), on="s1_idx", how="semi") if side.exists() else s1
    ents = ents.join(fo.select("s1_idx", "fold"), on="s1_idx")
    feat = feat.join(fo.select("s1_idx", "fold"), on="s1_idx").join(gold.with_columns(y=pl.lit(1, pl.Int8)), on=["s1_idx", "pool_idx"], how="left").with_columns(pl.col("y").fill_null(0))
    cols = M.feature_cols(feat, getattr(mod, "FEATURES", None))
    frac = a.train_frac or get("TRAIN_FRAC")
    fold_ids = [int(x) for x in a.folds.split(",")] if a.folds else sorted(fo["fold"].unique().to_list())
    print(f"{len(feat):,} pairs, {len(ents):,} entities, {len(cols)} features, train_frac {frac}", flush=True)
    prob = np.full(len(feat), np.nan, dtype=np.float32)
    fold_col = feat["fold"].to_numpy()
    rng = np.random.default_rng(cfg["seed"])
    for f in sorted(fo["fold"].unique().to_list()):
        t = time.time()
        tr_ents = ents.filter(pl.col("fold") != f)["s1_idx"].to_numpy()
        tr_ents = rng.choice(tr_ents, size=max(1, int(len(tr_ents) * frac)), replace=False)
        tr = feat.filter(pl.col("s1_idx").is_in(tr_ents) & (pl.col("fold") != f))
        model = get("fit")(tr.select(cols).to_numpy(), tr["y"].to_numpy(), cfg)
        m = fold_col == f
        prob[m] = get("predict")(model, feat.filter(pl.Series(m)).select(cols).to_numpy())
        print(f"fold {f}: trained on {len(tr):,} pairs ({int(tr['y'].sum()):,} pos) in {time.time() - t:.0f}s", flush=True)
        del tr, model
    pairs = feat.select("s1_idx", "pool_idx").with_columns(prob=pl.Series(prob))
    pred = get("decide")(pairs, cfg)
    ent = M.per_entity_f05(pred.select("s1_idx", "pool_idx"), gold.filter(pl.col("s1_idx").is_in(ents["s1_idx"].implode())), ents.select("s1_idx", "fold", "country_n"))
    cs = feat.filter(pl.col("y") == 1).group_by("s1_idx").len().rename({"len": "c"})
    ent = ent.join(cs, on="s1_idx", how="left").fill_null(0).with_columns(
        oracle=pl.when(pl.col("t") == 0).then(1.0).otherwise(1.25 * pl.col("c") / (0.25 * pl.col("t") + pl.col("c")).clip(1e-9))
    )
    scores = [float(ent.filter(pl.col("fold") == f)["f05"].mean()) for f in fold_ids]
    by_c = ent.group_by("country_n").agg(pl.col("f05").mean(), pl.col("oracle").mean(), pl.len()).sort("country_n")
    by_t = ent.group_by(pl.col("t").clip(0, 6).alias("true_matches")).agg(pl.col("f05").mean(), pl.col("oracle").mean(), (pl.col("k").cast(pl.Int64) - pl.col("t").cast(pl.Int64)).mean().alias("k_minus_t"), pl.len()).sort("true_matches")
    cw = {r["country_n"]: r["f05"] for r in by_c.to_dicts()}
    tw = sum(TEST_WEIGHTS[c] * cw[c] for c in TEST_WEIGHTS if c in cw) / sum(TEST_WEIGHTS[c] for c in TEST_WEIGHTS if c in cw)
    tp = ent["tp"].sum(); k = ent["k"].sum(); t_ = ent["t"].sum()
    res = {
        "attempt_id": a.attempt_id, "metric": "f05_er", "higher_is_better": True, "folds": fold_ids,
        "fold_scores": scores, "mean": float(np.mean(scores)), "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "test_weighted": tw, "oracle": float(ent["oracle"].mean()), "pair_precision": float(tp / max(k, 1)), "pair_recall": float(tp / max(t_, 1)),
        "by_country": by_c.to_dicts(), "by_true_matches": by_t.to_dicts(), "n_entities": len(ent), "n_pairs": len(feat),
        "tag": a.tag, "train_frac": frac, "features": cols, "runtime_s": time.time() - t_all, "git_commit": git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_rows": len(ent),
    }
    out = ROOT / "attempts" / a.attempt_id
    (out / ("results.json" if not a.tag else f"results{a.tag}.json")).write_text(json.dumps(res, indent=2, default=float))
    ent.select("s1_idx", "fold", "country_n", "t", "k", "tp", "f05", "oracle").write_parquet(out / f"entity_scores{a.tag}.parquet")
    print(by_c)
    print(by_t)
    print(f"macro F0.5 {res['mean']:.5f} ± {res['std']:.5f} | test-weighted {tw:.5f} | oracle {res['oracle']:.5f} | pair P {res['pair_precision']:.4f} R {res['pair_recall']:.4f} | {res['runtime_s']:.0f}s")


if __name__ == "__main__":
    main()
