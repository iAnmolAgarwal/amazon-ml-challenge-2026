"""Export the pairs a France-focused cross-country cross-encoder must score beyond the spec band (raw texts, for Kaggle).

1. extra_heldout.parquet: for each LOCO direction, the held-out country's pairs in the blend band (logit average of
   the spec and no-spec LOCO models, 0.02 < p < 0.98, cascade-kept: p > 0.02 and top 10 per S1) that are NOT in
   train_band.parquet (those already get held-out scores). With them, loco_transfer.py needs no median fill.
2. france_band.parquet: the whole France blend band on test (spec + no-spec test p2, 0.02 < p < 0.98), as in
   stack_ce.py --unseen_blend.
Columns as export_band.py (a_id, r_id, country, split, [fold, y], p2, a_name, a_addr, r_name, r_addr).
Prints counts only.

  python src/export_extra.py --out ce_work/band_extra
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from io_utils import ROOT                                 # noqa: E402
from normalize import CACHE                              # noqa: E402
from pairs import out_dir, safe, DEFAULT                 # noqa: E402
from model import fold_of                                 # noqa: E402
from cand_budget import rank_in_entity, keep_mask         # noqa: E402
from export_band import texts, LO, HI                     # noqa: E402

KAG = os.environ.get("CE_DIR", os.path.join(ROOT, "ce_work"))


def logit(p):
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def with_texts(d, split):
    """Attach raw S1 and record texts (raises if any are missing)."""
    A = texts(split, 1, set(d.a_id))
    R = pd.concat([texts(split, k, set(d.r_id)) for k in (2, 3)])
    d["a_name"], d["a_addr"] = A.name_raw.reindex(d.a_id).values, A.addr_raw.reindex(d.a_id).values
    d["r_name"], d["r_addr"] = R.name_raw.reindex(d.r_id).values, R.addr_raw.reindex(d.r_id).values
    if int(d.a_name.isna().sum() + d.r_name.isna().sum()):
        raise ValueError("texts missing")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = os.path.expanduser(ap.parse_args().out)
    os.makedirs(out, exist_ok=True)
    band = pd.read_parquet(os.path.join(KAG, "band", "train_band.parquet"), columns=["a_id", "r_id"])
    have = pd.MultiIndex.from_arrays([band.a_id.values, band.r_id.values])
    parts, stats = [], {}
    for held, d_ in (("US", "in2us"), ("India", "us2in")):
        t = pq.read_table(os.path.join(out_dir("train", dict(DEFAULT)), safe(held) + ".parquet"),
                          columns=["a_id", "r_id", "a", "y"]).to_pandas()
        lp = [logit(np.load(os.path.join(ROOT, "cache", f"exp_{e}_{d_}", f"p2_{safe(held)}.npy"))) for e in ("loco_s25spec", os.environ.get("EXTRA_NS", "loco_s25"))]   # EXTRA_NS=loco_s25dk: DK blend
        p = 1 / (1 + np.exp(-(lp[0] + lp[1]) / 2))
        a = t.a.values.astype(np.int64)
        m = keep_mask(a, p, rank_in_entity(a, p), LO, 10) & (p > LO) & (p < HI)
        m &= ~pd.MultiIndex.from_arrays([t.a_id.values, t.r_id.values]).isin(have)
        d = t.loc[m, ["a_id", "r_id", "y"]].copy()
        d["p2"], d["country"], d["split"] = p[m].astype(np.float32), held, "train"
        d["fold"] = fold_of(d.a_id.values).astype(np.int8)
        parts.append(with_texts(d, "train"))
        stats[f"heldout_{held}"] = {"blend_band_kept": int(((p > LO) & (p < HI)).sum()), "extra_pairs": int(len(d)),
                                    "extra_pos_rate": round(float(d.y.mean()), 4)}
    ex = pd.concat(parts, ignore_index=True)
    ex.to_parquet(os.path.join(out, "extra_heldout.parquet"), index=False, compression="zstd")
    t = pq.read_table(os.path.join(out_dir("test", dict(DEFAULT)), "France.parquet"), columns=["a_id", "r_id"]).to_pandas()
    ps = np.load(os.path.join(CACHE, "predict_work", "p2_France.npy"))
    pn = np.load(os.path.join(ROOT, "cache", os.environ.get("FR_NS", "predict_work_nospec"), "p2_France.npy"))   # FR_NS=predict_work_dk
    if len(ps) != len(t) or len(pn) != len(t):
        raise ValueError("France p2 not aligned")
    p = 1 / (1 + np.exp(-(logit(ps) + logit(pn)) / 2))
    m = (p > LO) & (p < HI)
    d = t[m].copy()
    d["p2"], d["country"], d["split"] = p[m].astype(np.float32), "France", "test"
    d = with_texts(d, "test")
    d.to_parquet(os.path.join(out, "france_band.parquet"), index=False, compression="zstd")
    stats["france_blend_band"] = int(len(d))
    stats["mb"] = {f: round(os.path.getsize(os.path.join(out, f)) / 1e6, 1) for f in ("extra_heldout.parquet", "france_band.parquet")}
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
