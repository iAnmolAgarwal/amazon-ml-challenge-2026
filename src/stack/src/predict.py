"""Inference on the test split: cached test candidate pairs -> averaged fold models (stage 1 -> context -> stage 2)
-> decision layer -> output/matching_results.tsv + output/candidate_pairs.tsv.

candidate_pairs.tsv is exactly the set of pairs the matcher scored (every blocked pair of every block).
Every S1 test entity gets a row (empty lists allowed). Country is only used as a block key.
"""
import argparse
import gc
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
import memguard  # noqa: E402
from io_utils import ROOT, DATA  # noqa: E402
from normalize import CACHE  # noqa: E402
from pairs import out_dir, safe  # noqa: E402
from model import load_keys, predict_stream, context, SCALE  # noqa: E402
from run_experiment import decide, log  # noqa: E402
from loco_rules import Fast  # noqa: E402


def write_lists(path, header, s1_ids, a_idx, r_ids):
    """One row per S1 entity; comma-joined S2/S3 ids (no quoting)."""
    d = pd.DataFrame({"a": a_idx, "r": r_ids}).drop_duplicates()
    lists = d.groupby("a").r.agg(",".join)
    col = pd.Series(lists.reindex(np.arange(len(s1_ids))).fillna("").values)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        for s, v in zip(s1_ids, col.values):
            fh.write(f"{s}\t{v}\n")


def main():
    memguard.start()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models/final")
    ap.add_argument("--rule", default='{"type": "expected_f"}')
    ap.add_argument("--out", default="output")
    ap.add_argument("--n_models", type=int, default=5, help="average the first N fold models (speed/accuracy)")
    ap.add_argument("--rule_unseen", default=None,
                    help='rule for test countries with NO training labels, e.g. {"T": 1.5, "min_p": 0.7, "gate": 0.9} '
                         '(chosen by leave-one-country-out, src/loco_rules.py); other countries use --rule')
    ap.add_argument("--only", default=None, help="comma-separated countries to score (default: all test countries)")
    ap.add_argument("--work", default="cache/predict_work", help="where test p2 arrays are saved (keep runs apart)")
    a = ap.parse_args()
    md = os.path.join(ROOT, a.models)
    meta = json.load(open(os.path.join(md, "meta.json")))
    rule = json.loads(a.rule)
    m1 = [lgb.Booster(model_file=os.path.join(md, f"s1_fold{i}.txt")) for i in range(a.n_models)]
    m2 = [lgb.Booster(model_file=os.path.join(md, f"s2_fold{i}.txt")) for i in range(a.n_models)]
    pdir = out_dir("test", meta["pairs_cfg"])
    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    mpath, cpath = os.path.join(out, "matching_results.tsv"), os.path.join(out, "candidate_pairs.tsv")
    for p, h in ((mpath, "matched_entity_ids"), (cpath, "candidate_entity_ids")):
        open(p, "w").write(f"source1_entity_id\t{h}\n")
    s1 = pq.read_table(os.path.join(CACHE, "norm_test_s1.parquet"), columns=["id", "country"]).to_pandas()
    rule_unseen = json.loads(a.rule_unseen) if a.rule_unseen else None
    train_path = os.path.join(CACHE, "norm_train_s1.parquet")
    trained = set(pq.read_table(train_path, columns=["country"]).column(0).unique().to_pylist())         if os.path.exists(train_path) else set(meta.get("train_countries", []))
    unseen = sorted(set(s1.country.unique()) - trained) if rule_unseen else []
    log("countries without training labels (unseen rule):", unseen, "rule:", rule_unseen)
    work = os.path.join(ROOT, a.work)
    os.makedirs(work, exist_ok=True)
    stats = {}
    countries = sorted(s1.country.unique()) if a.only is None else a.only.split(",")
    for c in countries:
        ids = s1.id.values[s1.country.values == c]
        path = os.path.join(pdir, safe(c) + ".parquet")
        if meta.get("rate_features"):                    # same count -> rate rescaling as in training
            SCALE[path] = 1e6 / len(ids)
        k = load_keys(path, ("a", "r"))
        r_ids = pq.read_table(path, columns=["r_id"]).column(0).to_numpy(zero_copy_only=False)
        p1 = np.mean([predict_stream(path, meta["feats1"], m) for m in m1], axis=0).astype(np.float32)
        ctx = context(k["a"], k["r"], p1, store=work)
        base = [f for f in meta["feats2"] if f not in ctx]
        p2 = np.mean([predict_stream(path, base, m, extra=ctx) for m in m2], axis=0).astype(np.float32)
        del ctx
        gc.collect()
        if c in unseen:
            sel = Fast(k["a"].astype(np.int64), k["r"].astype(np.int64), p2.astype(np.float64), len(ids)).select(**rule_unseen)
        else:
            sel = decide(k["a"], k["r"], p2, None, len(ids), None, rule)
        np.save(os.path.join(work, f"p2_{safe(c)}.npy"), p2)
        write_lists(cpath, "c", ids, k["a"], r_ids)
        write_lists(mpath, "m", ids, k["a"][sel], r_ids[sel])
        n_pred = np.bincount(k["a"][sel], minlength=len(ids))
        stats[c] = {"rule": rule_unseen if c in unseen else rule, "n_s1": int(len(ids)), "n_cand_pairs": int(len(p2)), "cands_per_s1": len(p2) / len(ids),
                    "pred_pairs": int(sel.sum()), "pred_singleton_rate": float((n_pred == 0).mean()),
                    "matches_per_s1": float(n_pred.mean()),
                    "p2_quantiles": np.quantile(p2, [0.5, 0.9, 0.99]).round(4).tolist(),
                    "p2_hist": np.histogram(p2, bins=10, range=(0, 1))[0].tolist()}
        log(c, json.dumps(stats[c]))
        del k, r_ids, p1, p2, sel
        gc.collect()
    json.dump(stats, open(os.path.join(out, "predict_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
