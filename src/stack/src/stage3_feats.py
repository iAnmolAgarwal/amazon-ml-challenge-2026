"""Stage-3 sidecar features for uncertain-band pairs: address evidence, supporting records and chain size.

The stage-3 stacker only saw scores (p2, cross-encoder) and their context. Branches of a chain share the name, so the
address must decide. For every band pair (S1 a, record r) this adds:
- cached pair features about addresses and numbers (cosines, token-set scores, number agreement, missingness);
- address character-trigram cosine (dropped from stage 1/2 for memory; here computed for band pairs only);
- supporting-record evidence: similarity of r's address/name to the entity's OTHER confident records (anchors, p2 >= 0.9;
  train p2 is out-of-fold, so anchors are fold-safe; r itself is excluded; no name-only transitive merging);
- chain size: number of S1 in the country block sharing the S1 name core;
- cross-field evidence: record NAME vs S1 ADDRESS (and S1 name vs record address) trigram cosine, the share of the
  record name's extra words (not in the S1 name) found in the S1 address, and its rank among the record's competing
  S1 (an address-less record of a chain often names its locality, which appears in the right branch's address).
Output: <kaggle>/band/<split>_s3feats.parquet keyed by (a_id, r_id). Prints counts only.

  python src/stage3_feats.py --split train
  python src/stage3_feats.py --split test
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import HashingVectorizer

sys.path.insert(0, os.path.dirname(__file__))
from io_utils import ROOT                                 # noqa: E402
from normalize import CACHE, load_norm                   # noqa: E402
from pairs import out_dir, safe, DEFAULT                 # noqa: E402

KAG = os.environ.get("CE_DIR", os.path.join(ROOT, "ce_work"))
PAIR_FEATS = ["cos_addr_w", "ad_tset", "ad_tsort", "ad_ptset", "ad_ratio", "nums_shared", "nums_a", "nums_r", "nums_jac",
              "num1_eq", "num1_missing", "addr_len_a", "addr_len_r", "r_addr_empty", "cos_name_c3", "nm_tset"]
ANCHOR_P = 0.9
HV = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=2 ** 20, norm="l2", alternate_sign=False)


def rowdot(A, B):
    """Row-wise dot product of two L2-normalised sparse matrices with equal row counts (= cosine)."""
    return np.asarray(A.multiply(B).sum(axis=1)).ravel().astype(np.float32)


def build(split, c, band, p=None):
    """Sidecar features for the band pairs of one country block (p: optional stage-2 probabilities for anchors)."""
    t = pq.read_table(os.path.join(out_dir(split, dict(DEFAULT)), safe(c) + ".parquet"),
                      columns=["a_id", "r_id", "a"] + PAIR_FEATS).to_pandas()
    if p is None:
        p = np.load(os.path.join(ROOT, "cache", "exp_final_s25_spec", f"p2_{safe(c)}.npy") if split == "train"
                    else os.path.join(CACHE, "predict_work", f"p2_{safe(c)}.npy")).astype(np.float64)
    b = band[band.country == c][["a_id", "r_id"]].reset_index(drop=True)
    pos = pd.MultiIndex.from_arrays([t.a_id.values, t.r_id.values]).get_indexer(
        pd.MultiIndex.from_arrays([b.a_id.values, b.r_id.values]))
    if (pos < 0).any():
        raise ValueError(f"{split}/{c}: band pairs missing from the pair table")
    out = b.copy()
    for f in PAIR_FEATS:
        out[f] = t[f].values[pos].astype(np.float32)
    a_band = t.a.values[pos].astype(np.int64)
    s1 = load_norm(split, 1, ["id", "addr_norm", "name_core", "name_norm"], country=c)
    _, inv, cc = np.unique(s1.name_core.values, return_inverse=True, return_counts=True)
    out["chain_n"] = cc[inv][a_band].astype(np.float32)
    anc = (p >= ANCHOR_P)
    anchors = pd.DataFrame({"a": t.a.values[anc].astype(np.int64), "r_id": t.r_id.values[anc]})
    need = pd.Index(pd.unique(np.concatenate([b.r_id.values, anchors.r_id.values])))
    tg = pd.concat([load_norm(split, k, ["id", "addr_norm", "name_norm"], country=c) for k in (2, 3)], ignore_index=True)
    tg = tg.set_index("id").reindex(need)
    T_addr = HV.transform(tg.addr_norm.fillna("").values)
    T_name = HV.transform(tg.name_norm.fillna("").values)
    S_addr = HV.transform(s1.addr_norm.fillna("").values)
    ri = need.get_indexer(b.r_id.values)
    out["addr_c3"] = rowdot(S_addr[a_band], T_addr[ri])
    S_name = HV.transform(s1.name_norm.fillna("").values)
    out["xf_rname_aaddr"] = rowdot(S_addr[a_band], T_name[ri])
    out["xf_aname_raddr"] = rowdot(S_name[a_band], T_addr[ri])
    a_name = s1.name_norm.fillna("").values[a_band]
    a_addr = s1.addr_norm.fillna("").values[a_band]
    r_name = tg.name_norm.fillna("").values[ri]
    share, n_extra = np.zeros(len(b), np.float32), np.zeros(len(b), np.float32)
    for i, (an, aa, rn) in enumerate(zip(a_name, a_addr, r_name)):
        extra = set(rn.split()) - set(an.split())
        extra = {w for w in extra if len(w) > 2}
        if extra:
            n_extra[i] = len(extra)
            share[i] = len(extra & set(aa.split())) / len(extra)
    out["xf_rextra_in_aaddr"], out["xf_rextra_n"] = share, n_extra
    ev = out.xf_rextra_in_aaddr + out.xf_rname_aaddr
    out["xf_rank_r"] = ev.groupby(out.r_id).rank(ascending=False, method="min").astype(np.float32)
    out["xf_gap_r"] = (ev - ev.groupby(out.r_id).transform("max")).astype(np.float32)
    # supporting records: band pair (a, r) x anchors (a, r'), r' != r
    bb = pd.DataFrame({"row": np.arange(len(b)), "a": a_band, "ri": ri})
    anchors["rj"] = need.get_indexer(anchors.r_id.values)
    j = bb.merge(anchors[["a", "rj"]], on="a")
    j = j[j.ri.values != j.rj.values]
    sa = np.zeros(len(j), np.float32)
    sn = np.zeros(len(j), np.float32)
    for s in range(0, len(j), 2_000_000):                    # chunked to bound memory
        x = j.iloc[s:s + 2_000_000]
        sa[s:s + len(x)] = rowdot(T_addr[x.ri.values], T_addr[x.rj.values])
        sn[s:s + len(x)] = rowdot(T_name[x.ri.values], T_name[x.rj.values])
    j = j.assign(sa=sa, sn=sn)
    g = j.groupby("row")
    agg = pd.DataFrame({"sup_addr_max": g.sa.max(), "sup_addr_mean": g.sa.mean(), "sup_name_max": g.sn.max(),
                        "sup_n": g.size()})
    out = out.join(agg, how="left")
    out["sup_n"] = out.sup_n.fillna(0).astype(np.float32)
    out["sup_addr_minus_s1"] = (out.sup_addr_max - out.addr_c3).astype(np.float32)
    print(split, c, "band pairs", len(out), "with anchors", round(float((out.sup_n > 0).mean()), 4),
          "mean anchors", round(float(out.sup_n.mean()), 2), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    a = ap.parse_args()
    band = pd.read_parquet(os.path.join(KAG, "band", f"{a.split}_band.parquet"), columns=["a_id", "r_id", "country"])
    parts = [build(a.split, c, band) for c in sorted(band.country.unique())]
    res = pd.concat(parts, ignore_index=True)
    res.to_parquet(os.path.join(KAG, "band", f"{a.split}_s3feats.parquet"), index=False)
    print("written", len(res), "rows,", len(res.columns) - 2, "features")


if __name__ == "__main__":
    main()
