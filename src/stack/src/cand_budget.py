"""Candidate-budget analysis (analysis only): how small can the candidate set per Source 1 entity be?

The organisers rank smaller candidate sets per S1 higher (guidelines.md, new_additional_guidelines). This measures,
on train out-of-fold stage-2 probabilities, what a cascade filter would cost: keep a pair only if p2 > t and it is
within the top k of its S1 entity. For each (t, k) it reports mean / p95 candidates per S1, candidate recall
(true links kept / all true links) and macro F0.5 of the unchanged decision rule on the kept pairs. The test side
reports only candidate counts per S1 (no labels). Prints aggregates only.

  python src/cand_budget.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from io_utils import ROOT, DATA, read_tsv                 # noqa: E402
from normalize import CACHE, load_norm                   # noqa: E402
from pairs import out_dir, safe, DEFAULT                 # noqa: E402
from loco_rules import Fast, f05                          # noqa: E402

GRID_T = [0.0, 0.001, 0.005, 0.02]
GRID_K = [None, 10, 6]


def rank_in_entity(a, p):
    """Rank (0 = best) of each pair's p within its S1 entity."""
    o = np.lexsort((-p, a))
    a_o = a[o]
    start = np.r_[True, a_o[1:] != a_o[:-1]]
    idx = np.arange(len(o)) - np.maximum.accumulate(np.where(start, np.arange(len(o)), 0))
    rank = np.empty(len(o), np.int64)
    rank[o] = idx
    return rank


def keep_mask(a, p, rank, t, k):
    """Cascade filter: p > t and (optionally) within top k of the entity."""
    m = p > t if t > 0 else np.ones(len(p), bool)
    return m & (rank < k) if k else m


def train_side():
    """Out-of-fold trade-off per country: candidates per S1, candidate recall, macro F0.5."""
    gt = read_tsv(os.path.join(DATA, "train", "train_ground_truth.tsv"))
    cnt = pd.Series(gt.matched_entity_ids.map(lambda x: 0 if x == "" else x.count(",") + 1).values,
                    index=gt.source1_entity_id.values)
    res = {}
    for b in ("India", "US"):
        t = pq.read_table(os.path.join(out_dir("train", dict(DEFAULT)), safe(b) + ".parquet"),
                          columns=["a", "r", "y"]).to_pandas()
        p = np.load(os.path.join(ROOT, "cache", "exp_final_s25_spec", f"p2_{safe(b)}.npy")).astype(np.float64)
        ids = load_norm("train", 1, ["id"], country=b).id.values
        n_true = cnt.reindex(ids).fillna(0).values
        a, r, y = t.a.values.astype(np.int64), t.r.values.astype(np.int64), t.y.values.astype(bool)
        rank = rank_in_entity(a, p)
        rows = []
        for th in GRID_T:
            for k in GRID_K:
                m = keep_mask(a, p, rank, th, k)
                n_c = np.bincount(a[m], minlength=len(ids))
                sel = np.zeros(len(a), bool)
                sel[np.flatnonzero(m)[Fast(a[m], r[m], p[m], len(ids)).select()]] = True
                tp = np.bincount(a[sel & y], minlength=len(ids))
                fp = np.bincount(a[sel & ~y], minlength=len(ids))
                rows.append({"t": th, "k": k, "cand_per_s1": round(float(n_c.mean()), 2),
                             "cand_p95": int(np.percentile(n_c, 95)), "cand_max": int(n_c.max()),
                             "cand_recall": round(float(y[m].sum() / n_true.sum()), 5),
                             "f05": round(float(f05(tp, fp, n_true - tp).mean()), 5)})
                print(b, json.dumps(rows[-1]), flush=True)
        res[b] = {"n_s1": int(len(ids)), "true_links": int(n_true.sum()), "rows": rows}
    return res


def test_side():
    """Candidates per S1 on test for the same grid (counts only)."""
    s1 = pq.read_table(os.path.join(CACHE, "norm_test_s1.parquet"), columns=["id", "country"]).to_pandas()
    res = {}
    for c in sorted(s1.country.unique()):
        n = int((s1.country.values == c).sum())
        t = pq.read_table(os.path.join(out_dir("test", dict(DEFAULT)), safe(c) + ".parquet"), columns=["a"]).to_pandas()
        a = t.a.values.astype(np.int64)
        p = np.load(os.path.join(CACHE, "predict_work", f"p2_{safe(c)}.npy")).astype(np.float64)
        rank = rank_in_entity(a, p)
        res[c] = {"n_s1": n, "rows": []}
        for th in GRID_T:
            for k in GRID_K:
                n_c = np.bincount(a[keep_mask(a, p, rank, th, k)], minlength=n)
                res[c]["rows"].append({"t": th, "k": k, "cand_per_s1": round(float(n_c.mean()), 2),
                                       "cand_p95": int(np.percentile(n_c, 95))})
        print(c, json.dumps(res[c]["rows"][0]), flush=True)
    return res


if __name__ == "__main__":
    out = {"train": train_side(), "test": test_side()}
    os.makedirs(os.path.join(ROOT, "experiments"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "experiments", "cand_budget.json"), "w"), indent=1)
