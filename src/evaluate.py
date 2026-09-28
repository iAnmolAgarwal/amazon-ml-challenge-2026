import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from common import ROOT, git_commit, load_attempt, load_config
from harness_util import get_metric, load_folds, load_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attempt_id")
    ap.add_argument("--subsample", type=float, default=1.0)
    ap.add_argument("--folds", help="comma-separated fold ids, default all")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    cfg["pool"] = "train"
    mod = load_attempt(a.attempt_id)
    metric_name = getattr(mod, "METRIC", cfg["metric"])
    metric, higher = get_metric(metric_name)
    df = load_train(cfg).join(load_folds(cfg), on=cfg["id_col"], how="inner")
    if a.subsample < 1.0:
        df = df.sample(fraction=a.subsample, seed=cfg["seed"], shuffle=True)
    fold_ids = [int(f) for f in a.folds.split(",")] if a.folds else sorted(df["fold"].unique().to_list())
    scores, times, cand = [], [], []
    t_all = time.time()
    for f in fold_ids:
        tr, va = df.filter(pl.col("fold") != f).drop("fold"), df.filter(pl.col("fold") == f).drop("fold")
        t0 = time.time()
        out = mod.fit_predict(tr, va.drop(cfg["target_col"]), cfg)
        pred = out["matches"] if isinstance(out, dict) else out
        yt = va[cfg["target_col"]].to_list()
        s = metric(yt, list(pred))
        scores.append(s)
        times.append(time.time() - t0)
        extra = ""
        if isinstance(out, dict) and "candidates" in out:
            from harness_util import candidate_stats

            cs = candidate_stats(yt, out["candidates"])
            cand.append(cs)
            extra = f" | cand recall={cs['pair_recall']:.4f} oracle={cs['oracle_f05']:.4f} avg={cs['avg_candidates']:.1f}"
        print(f"fold {f}: {metric_name}={s:.5f} ({times[-1]:.1f}s){extra}", flush=True)
    res = {
        "attempt_id": a.attempt_id,
        "metric": metric_name,
        "higher_is_better": higher,
        "folds": fold_ids,
        "fold_scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "runtime_s": time.time() - t_all,
        "fold_runtime_s": times,
        "subsample": a.subsample,
        "candidates": {k: float(np.mean([c[k] for c in cand])) for k in cand[0]} if cand else None,
        "n_rows": len(df),
        "git_commit": git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = ROOT / "attempts" / a.attempt_id / ("results.json" if a.subsample >= 1.0 else f"results_sub{a.subsample}.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"{metric_name} mean={res['mean']:.5f} std={res['std']:.5f} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
