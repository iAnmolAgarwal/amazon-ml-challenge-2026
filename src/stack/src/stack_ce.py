"""Stage 3: stack a cross-encoder (pair-model) score onto the stage-2 LightGBM probability for uncertain-band pairs.

Leakage safety
- p2 on train is out-of-fold (entity folds); the cross-encoder score of a train pair comes from the model trained on
  the OTHER country's band (US model -> India pairs, India model -> US pairs), so it never saw that pair's label.
- Stage 3 (LightGBM, few features) is trained with the same entity folds -> out-of-fold p3 for every band pair.
- Test: India pairs use the US-trained cross-encoder, US pairs the India-trained one (same as training); countries
  with no training labels (France) use the mean of both (both are "other-country" models for them).
Outside the band p = p2. Decision: one owner per record + expected-F0.5 prefix, min_p 0.5 (unchanged).

  python src/stack_ce.py eval  --tag e5s
  python src/stack_ce.py test  --tag e5s --out output_ce
  python src/stack_ce.py test  --tag e5s --out output_v03 --cand_t 0.02 --cand_k 10          --unseen_p2 cache/predict_work_nospec --rule_unseen '{"T": 2.0, "min_p": 0.55}'

Options:
- --cand_t/--cand_k: cascade candidate filter. Candidate generation = blocking + stage-1/2 LightGBM pruning (keep
  p2 > t and the top k per S1); the final matcher (stage-3 stack + decision) scores only the kept pairs, which are
  exactly the pairs exported to cache/stack.
- --unseen_p2/--rule_unseen: countries without training labels (France) use stage-2 probabilities of another model
  (no name-specificity features, which transfer badly across countries: LOCO) and a rule chosen by leave-one-country-out tests.
"""
import argparse
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from io_utils import ROOT, DATA, read_tsv                 # noqa: E402
from normalize import CACHE, load_norm                   # noqa: E402
from pairs import out_dir, safe, DEFAULT                 # noqa: E402
from model import load_keys                              # noqa: E402
from loco_rules import Fast, f05                          # noqa: E402
from cand_budget import rank_in_entity, keep_mask         # noqa: E402

KAG = os.environ.get("CE_DIR", os.path.join(ROOT, "ce_work"))   # band/ and out/
FEATS = ["lp2", "ce", "ce_rank_a", "ce_gap_a", "ce_n_a", "ce_rank_r", "ce_margin_r", "ce_n_r", "lp2_x_ce"]
# optional second cross-encoder (in-country, entity halves; --tag2): same features with the ce2 prefix
FEATS2 = FEATS + ["ce2", "ce2_rank_a", "ce2_gap_a", "ce2_margin_r", "lp2_x_ce2"]
# optional address / supporting-record / chain sidecar (src/stage3_feats.py; --extra)
EXTRA = ["cos_addr_w", "ad_tset", "ad_tsort", "ad_ptset", "ad_ratio", "nums_shared", "nums_a", "nums_r", "nums_jac",
         "num1_eq", "num1_missing", "addr_len_a", "addr_len_r", "r_addr_empty", "cos_name_c3", "nm_tset", "chain_n",
         "addr_c3", "sup_addr_max", "sup_addr_mean", "sup_name_max", "sup_n", "sup_addr_minus_s1"]


XF = ["xf_rname_aaddr", "xf_aname_raddr", "xf_rextra_in_aaddr", "xf_rextra_n", "xf_rank_r", "xf_gap_r"]


# stage-3 modifiers set from the CLI: decoy = add decoy_feats.py features; nosup = drop the supporting-record
# (coherence) features, which can learn "shared deviation means true" from clustered decoys on test
MODS = {"decoy": False, "nosup": False, "lex": False, "decoy2": False, "ops": False, "pairf": False, "cnt": False}
CNT = ["cnt_nconf", "cnt_need", "cnt_mr", "cnt_share"]   # count-prior competition features (feat_cnt.py, a research script not shipped; --cnt, off in the submission)
OPS = ["op_acro", "op_add_fill", "op_add_other", "op_drop_fill", "op_drop_other", "op_filler_only", "op_ed1",
       "op_unmatched_after_ed1", "op_num_trunc", "op_num_new"]    # generator-operator features (feat_ops.py; --ops)
LEX = [f"lx_{c}_{s}" for c in ("nm_add", "nm_drop", "ad_add", "ad_drop", "legal")
       for s in ("n", "max", "min", "sum", "mean", "unseen", "support")]
DECOY = ["h_both", "h_equal", "h_shift", "h_logdiff", "h_suffix", "lg_equal", "lg_differ", "lg_one_side",
         "nm_n_added", "nm_n_dropped", "dk_added", "dk_dropped", "dk_swap", "lg_nested", "u_equal", "u_conflict", "u_one_side"]
DECOY2 = ["h_up_shift", "h_le10", "h_reorder", "h_prefix", "dk_add_rarity", "dk_drop_rarity", "dk_direction",
          "num1_eq_fb", "num1_missing_fb"]                       # house agreement with the French fallback parser     # extra house relations (--decoy2, pending validation)


# monotone directions for the unseen-country stacker (MONO=1, off by default): only features whose sign is
# unambiguous by construction are constrained, so source-country quirks cannot invert them on France
MONO_UP = {"lp2", "lp2b", "ce", "ce2", "ce_margin_r", "ce_gap_a", "cos_addr_w", "ad_tset", "ad_tsort", "ad_ptset", "ad_ratio",
           "nums_shared", "nums_jac", "num1_eq", "cos_name_c3", "nm_tset", "addr_c3", "h_equal", "lg_equal", "u_equal",
           "num1_eq_fb"}
MONO_DOWN = {"ce_rank_a", "ce_rank_r", "h_shift", "h_up_shift", "lg_differ", "u_conflict", "dk_added", "dk_swap"}


def mono_params(feats):
    """LightGBM monotone-constraint params for a feature list (empty unless MONO=1)."""
    if os.environ.get("MONO") != "1":
        return {}
    return {"monotone_constraints": [1 if f in MONO_UP else -1 if f in MONO_DOWN else 0 for f in feats],
            "monotone_constraints_method": "advanced"}


def pf_cols():
    """All stage-1/spec pair-feature columns exported for the band (band_feats.py; --pairf)."""
    return [c for c in pq.ParquetFile(os.path.join(KAG, "band", "train_pairf.parquet")).schema_arrow.names if c.startswith("pf_")]


def feat_list(tag2=None, extra=False, xf=False, p2b=False, tag3=None):
    fl = _feat_list(tag2, extra, xf, p2b, tag3)
    if MODS["nosup"]:
        fl = [f for f in fl if not f.startswith("sup_")]
    return (fl + (DECOY if MODS["decoy"] else []) + (DECOY2 if MODS["decoy2"] else []) + (LEX if MODS["lex"] else [])
            + (OPS if MODS["ops"] else []) + (pf_cols() if MODS["pairf"] else []) + (CNT if MODS["cnt"] else []))


def _feat_list(tag2=None, extra=False, xf=False, p2b=False, tag3=None):
    """Stage-3 feature set: one CE (FEATS), plus the in-country CE (tag2), the sidecar (extra), cross-field (xf) and
    the logit of a second stage-2 model's probability (p2b, e.g. the 50%-sample model)."""
    return ((FEATS2 if tag2 else FEATS) + (EXTRA if extra else []) + (XF if xf else []) + (["lp2b"] if p2b else [])
            + [f.replace("ce2", f"ce{3 + j}") for j, _ in enumerate(tag3.split(",") if tag3 else []) for f in FEATS2[len(FEATS):]])


def attach_p2b(d, split, p2b_dir):
    """lp2b = logit of a second stage-2 model's p2 (row-aligned with the pair tables) for every band pair of d.
    Train p2 must be out-of-fold with the same entity folds (exp_final_s50_spec is)."""
    d = d.copy()
    d["lp2b"] = np.float32(np.nan)
    for c in d.country.unique():
        m = (d.country == c).values
        t = pq.read_table(os.path.join(out_dir(split, dict(DEFAULT)), safe(c) + ".parquet"), columns=["a_id", "r_id"])
        pos = pd.MultiIndex.from_arrays([t.column("a_id").to_numpy(), t.column("r_id").to_numpy()]).get_indexer(
            pd.MultiIndex.from_arrays([d.a_id.values[m], d.r_id.values[m]]))
        pb = np.load(os.path.join(ROOT, p2b_dir, f"p2_{safe(c)}.npy"))
        if (pos < 0).any() or len(pb) != t.num_rows:
            raise ValueError(f"{split}/{c}: p2b not aligned with the pair table")
        d.loc[m, "lp2b"] = logit(pb[pos].astype(np.float64)).astype(np.float32)
    return d
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8, seed=0)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def add_feats(d):
    """Cross-encoder features within the band: rank/gap within the S1 entity and within the S2/S3 record."""
    d = d.copy()
    d["lp2"] = logit(d.p2.values)
    for key, suf in (("a_id", "a"), ("r_id", "r")):
        g = d.groupby(key).ce
        d[f"ce_rank_{suf}"] = g.rank(ascending=False, method="first").astype(np.float32)
        d[f"ce_n_{suf}"] = g.transform("size").astype(np.float32)
        mx = g.transform("max")
        if suf == "a":
            d["ce_gap_a"] = (d.ce - mx).astype(np.float32)
        else:
            o = np.lexsort((-d.ce.values, pd.factorize(d.r_id)[0]))       # vectorised second-best per record
            codes = pd.factorize(d.r_id)[0][o]
            ce_o = d.ce.values[o]
            start = np.r_[True, codes[1:] != codes[:-1]]
            first_idx = np.flatnonzero(start)
            size = np.diff(np.r_[first_idx, len(o)])
            second_g = np.where(size > 1, ce_o[np.minimum(first_idx + 1, len(o) - 1)], -20.0)
            sec = np.empty(len(o))
            sec[o] = second_g[np.cumsum(start) - 1]
            d["ce_margin_r"] = np.where(d.ce >= mx, d.ce - sec, d.ce - mx).astype(np.float32)
    d["lp2_x_ce"] = (d.lp2 * d.ce).astype(np.float32)
    return d


def add_ce2(d, k="ce2"):
    """Features of an extra cross-encoder score column k (ce2, ce3), named like the first with the k prefix."""
    t = add_feats(d[["a_id", "r_id", "p2"]].assign(ce=d[k].values))
    for c in ("rank_a", "gap_a", "margin_r"):
        d[f"{k}_{c}"] = t[f"ce_{c}"].values
    d[f"lp2_x_{k}"] = t["lp2_x_ce"].values
    return d


def merge_halves(d, tag, k, split):
    """Attach an in-country (entity-halves) CE score as column k: heldout = other half's model; test = mean of both."""
    if split == "train":
        s = pd.concat([pd.read_parquet(os.path.join(KAG, "out", f"scores_{tag}_half{h}_heldout.parquet")) for h in (0, 1)])
        s = s[["a_id", "r_id", "ce"]].rename(columns={"ce": k})
    else:
        s = [pd.read_parquet(os.path.join(KAG, "out", f"scores_{tag}_half{h}_test.parquet"))[["a_id", "r_id", "ce"]] for h in (0, 1)]
        s = s[0].merge(s[1], on=["a_id", "r_id"], suffixes=("_0", "_1"))
        s = s.assign(**{k: (s.ce_0 + s.ce_1) / 2})[["a_id", "r_id", k]]
    d = d.merge(s, on=["a_id", "r_id"], how="left")
    if split == "train" and d[k].isna().any():
        raise ValueError(f"{int(d[k].isna().sum())} band pairs lack a {tag} score")
    return add_ce2(d, k)


def train_band(tag, tag2=None, extra=False, tag3=None):
    band = pd.read_parquet(os.path.join(KAG, "band", "train_band.parquet"),
                           columns=["a_id", "r_id", "country", "fold", "y", "p2"])
    sc = pd.concat([pd.read_parquet(os.path.join(KAG, "out", f"scores_{tag}_{c}_heldout.parquet")) for c in ("US", "India")])
    d = band.merge(sc[["a_id", "r_id", "ce"]], on=["a_id", "r_id"], how="left")
    if d.ce.isna().any():
        raise ValueError(f"{int(d.ce.isna().sum())} band pairs lack a cross-encoder score")
    d = add_feats(d)
    if tag2:                                              # in-country halves: each pair scored by the other half's model
        s2 = pd.concat([pd.read_parquet(os.path.join(KAG, "out", f"scores_{tag2}_half{h}_heldout.parquet")) for h in (0, 1)])
        d = d.merge(s2[["a_id", "r_id", "ce"]].rename(columns={"ce": "ce2"}), on=["a_id", "r_id"], how="left")
        if d.ce2.isna().any():
            raise ValueError(f"{int(d.ce2.isna().sum())} band pairs lack a second cross-encoder score")
        d = add_ce2(d)
    for j, t3 in enumerate(tag3.split(",") if tag3 else []):   # more in-country CEs (e5b, ...): columns ce3, ce4, ...
        d = merge_halves(d, t3, f"ce{3 + j}", "train")
    if extra:
        x = pd.read_parquet(os.path.join(KAG, "band", "train_s3feats.parquet"))
        d = d.merge(x, on=["a_id", "r_id"], how="left")
    if MODS["decoy"]:
        d = d.merge(pd.read_parquet(os.path.join(KAG, "band", "train_decoy.parquet")), on=["a_id", "r_id"], how="left")
    if MODS["lex"]:
        d = d.merge(pd.read_parquet(os.path.join(KAG, "band", "train_lex.parquet")), on=["a_id", "r_id"], how="left")
    if MODS["ops"]:
        d = d.merge(pd.read_parquet(os.path.join(KAG, "band", "train_ops.parquet")), on=["a_id", "r_id"], how="left")
    if MODS["pairf"]:
        d = d.merge(pd.read_parquet(os.path.join(KAG, "band", "train_pairf.parquet")), on=["a_id", "r_id"], how="left")
    if MODS["cnt"]:
        d = d.merge(pd.read_parquet(os.path.join(KAG, "band", "train_cnt.parquet")), on=["a_id", "r_id"], how="left")
    if len(d) != len(band):                               # duplicate keys in a score/feature file would duplicate rows
        raise ValueError(f"train band {len(band)} rows, {len(d)} after merges")
    return d


def entity_weights(d, kind):
    """Per-pair training weights so each S1 entity counts more equally (macro F0.5 weighs entities, not pairs)."""
    if isinstance(kind, np.ndarray):                     # explicit per-row weights (e.g. utility_weights)
        return kind.astype(np.float32)
    if kind in (None, "none"):
        return None
    n = d.groupby("a_id").a_id.transform("size").values.astype(np.float64)
    w = 1 / n if kind == "inv" else 1 / np.sqrt(n)
    return (w / w.mean()).astype(np.float32)


def utility_weights(d, p3):
    """Metric-aware training weights: for every band row, |change of its S1's F0.5| if the pair's
    current OOF decision (p2 outside the band, p3 inside; one owner + expected-F0.5 rule) were flipped; clipped to
    [0.02, 1], mean 1. Training rows only (labels of the training band)."""
    gt = read_tsv(os.path.join(DATA, "train", "train_ground_truth.tsv"))
    cnt = pd.Series(gt.matched_entity_ids.map(lambda x: 0 if x == "" else x.count(",") + 1).values,
                    index=gt.source1_entity_id.values)
    w = np.ones(len(d))

    def F(t, f, tot):
        den = t + f + 0.25 * tot
        return np.divide(1.25 * t, den, out=np.ones_like(den, dtype=float), where=den > 0)
    for c in d.country.unique():
        m = (d.country == c).values
        t = pq.read_table(os.path.join(out_dir("train", dict(DEFAULT)), safe(c) + ".parquet"),
                          columns=["a_id", "r_id", "a", "r", "y"]).to_pandas()
        p = np.load(os.path.join(ROOT, "cache", "exp_final_s25_spec", f"p2_{safe(c)}.npy")).astype(np.float64)
        ids = load_norm("train", 1, ["id"], country=c).id.values
        nt = cnt.reindex(ids).fillna(0).values
        pos = pd.MultiIndex.from_arrays([t.a_id.values, t.r_id.values]).get_indexer(
            pd.MultiIndex.from_arrays([d.a_id.values[m], d.r_id.values[m]]))
        if (pos < 0).any():
            raise ValueError(f"{c}: band rows not in the pair table")
        p[pos] = p3[m]
        a, y = t.a.values.astype(np.int64), t.y.values.astype(bool)
        sel = Fast(a, t.r.values.astype(np.int64), p, len(ids)).select()
        tp = np.bincount(a[sel & y], minlength=len(ids)).astype(float)
        fp = np.bincount(a[sel & ~y], minlength=len(ids)).astype(float)
        ab, yb, sb = a[pos], y[pos], sel[pos]
        step = np.where(sb, -1.0, 1.0)
        before = F(tp[ab], fp[ab], nt[ab])
        after = F(tp[ab] + step * yb, fp[ab] + step * (~yb), nt[ab])
        w[m] = np.clip(np.abs(after - before), 0.02, 1.0)
        print(c, "utility weights: mean", round(float(w[m].mean()), 4), "p90", round(float(np.percentile(w[m], 90)), 4), flush=True)
    return (w / w.mean()).astype(np.float32)


def oof_stage3(d, feats=FEATS, weight=None, params=None, rounds=400):
    """Out-of-fold p3 over the band (entity folds, both countries pooled)."""
    p3 = np.zeros(len(d), np.float32)
    w = entity_weights(d, weight)
    prm = {**PARAMS, **(params or {})}
    for k in range(5):
        tr, va = d.fold.values != k, d.fold.values == k
        m = lgb.train(prm, lgb.Dataset(d.loc[tr, feats], d.y.values[tr], weight=None if w is None else w[tr]), rounds)
        p3[va] = m.predict(d.loc[va, feats])
    return p3


def evaluate(tag, tag2=None, extra=False, weight=None, xf=False, p2b=None, params=None, rounds=400, tag3=None):
    """OOF macro F0.5 of stage 3 against the CURRENT combined pipeline (complementarity, not standalone gain):
    base = one-CE stack, plus the sidecar when --extra is used together with --tag2 or --weight; new = requested set."""
    d = train_band(tag, tag2, extra, tag3)
    if p2b:                                               # base = the same stack without lp2b
        d = attach_p2b(d, "train", p2b)
    d["p3"] = oof_stage3(d, feat_list(tag2, extra, xf, bool(p2b), tag3), weight, params, rounds)
    if os.environ.get("UTILITY") == "1":                  # base = the same stack, unweighted
        d["p3_base"] = d["p3"].values
        d["p3"] = oof_stage3(d, feat_list(tag2, extra, xf, bool(p2b), tag3), utility_weights(d, d["p3_base"].values), params, rounds)
    elif MODS["cnt"]:                                     # base = the same stack without the count-prior features
        MODS["cnt"] = False
        d["p3_base"] = oof_stage3(d, feat_list(tag2, extra, xf, bool(p2b), tag3), weight, params, rounds)
        MODS["cnt"] = True
    elif MODS["pairf"]:                                   # base = the same stack without the pair features
        MODS["pairf"] = False
        d["p3_base"] = oof_stage3(d, feat_list(tag2, extra, xf, bool(p2b), tag3), weight)
        MODS["pairf"] = True
    elif MODS["ops"]:                                     # base = the same stack without the operator features
        MODS["ops"] = False
        d["p3_base"] = oof_stage3(d, feat_list(tag2, extra, xf, bool(p2b), tag3), weight)
        MODS["ops"] = True
    elif MODS["decoy2"]:                                  # base = the same stack without the extra decoy features
        MODS["decoy2"] = False
        d["p3_base"] = oof_stage3(d, feat_list(tag2, extra, xf, bool(p2b), tag3), weight)
        MODS["decoy2"] = True
    elif MODS["lex"]:                                     # base = the same stack without the lexical features
        MODS["lex"] = False
        d["p3_base"] = oof_stage3(d, feat_list(tag2, extra, xf, bool(p2b), tag3), weight)
        MODS["lex"] = True
    elif MODS["decoy"] or MODS["nosup"]:                  # base = the same stack without the modifiers
        d["p3_base"] = oof_stage3(d, _feat_list(tag2, extra, xf, bool(p2b), tag3), weight)
    elif tag3:                                              # base = the same stack without the LAST extra CE
        d["p3_base"] = oof_stage3(d, feat_list(tag2, extra, xf, bool(p2b), ",".join(tag3.split(",")[:-1]) or None), weight)
    elif params or rounds != 400:                           # base = same features, default LightGBM settings
        d["p3_base"] = oof_stage3(d, feat_list(tag2, extra, xf, bool(p2b)), weight)
    elif p2b:
        d["p3_base"] = oof_stage3(d, feat_list(tag2, extra, xf))
    elif tag2 or extra or weight or xf:
        base_feats = (feat_list(tag2, extra) if (tag2 and xf) else feat_list(None, extra)) if (tag2 or weight or xf) else FEATS
        d["p3_base"] = oof_stage3(d, base_feats)
    gt = read_tsv(os.path.join(DATA, "train", "train_ground_truth.tsv"))
    cnt = pd.Series(gt.matched_entity_ids.map(lambda x: 0 if x == "" else x.count(",") + 1).values,
                    index=gt.source1_entity_id.values)
    from sklearn.metrics import roc_auc_score
    res = {"band_auc": {"p2": round(float(roc_auc_score(d.y, d.p2)), 5), "ce": round(float(roc_auc_score(d.y, d.ce)), 5),
                        "p3": round(float(roc_auc_score(d.y, d.p3)), 5)}}
    for b in ("India", "US"):
        path = os.path.join(out_dir("train", dict(DEFAULT)), safe(b) + ".parquet")
        t = pq.read_table(path, columns=["a_id", "r_id", "a", "r", "y"]).to_pandas()
        p2 = np.load(os.path.join(ROOT, "cache", "exp_final_s25_spec", f"p2_{safe(b)}.npy")).astype(np.float64)
        ids = load_norm("train", 1, ["id"], country=b).id.values
        n_true = cnt.reindex(ids).fillna(0).values
        db = d[d.country == b]
        key_t = pd.MultiIndex.from_arrays([t.a_id.values, t.r_id.values])
        pos = key_t.get_indexer(pd.MultiIndex.from_arrays([db.a_id.values, db.r_id.values]))
        p = p2.copy()
        p[pos] = db.p3.values
        pb = p2.copy()
        if "p3_base" in db:
            pb[pos] = db.p3_base.values
        a, r, y = t.a.values.astype(np.int64), t.r.values.astype(np.int64), t.y.values.astype(bool)
        from model import fold_of
        fold = fold_of(ids)
        out = {}
        for name, pp in (("base", pb), ("stack", p)):
            sel = Fast(a, r, pp, len(ids)).select()
            tp = np.bincount(a[sel & y], minlength=len(ids))
            fp = np.bincount(a[sel & ~y], minlength=len(ids))
            F = f05(tp, fp, n_true - tp)
            out[name] = {"f05": round(float(F.mean()), 5), "folds": [round(float(F[fold == k].mean()), 5) for k in range(5)],
                         "singleton": round(float(F[n_true == 0].mean()), 5),
                         "P": round(float(np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1).mean()), 5),
                         "R": round(float(np.where(n_true > 0, tp / np.maximum(n_true, 1), 1).mean()), 5)}
        if os.environ.get("SAVE_IC"):                   # cache in-country OOF final p for decision-level studies
            qd = os.path.join(ROOT, "cache", "ic_q")
            os.makedirs(qd, exist_ok=True)
            np.savez(os.path.join(qd, f"{os.environ['SAVE_IC']}_{b}.npz"), a=a, r=r, q=p, p2=p2, y=y, n_true=n_true, fold=fold)
        out["gain"] = round(out["stack"]["f05"] - out["base"]["f05"], 5)
        out["fold_gains"] = [round(x - y_, 5) for x, y_ in zip(out["stack"]["folds"], out["base"]["folds"])]
        res[b] = out
    res["all"] = {k: round((res["India"][k]["f05"] * 883188 + res["US"][k]["f05"] * 1323633) / 2206821, 5)
                  for k in ("base", "stack")}
    os.makedirs(os.path.join(ROOT, "experiments"), exist_ok=True)
    name = f"stack_{tag}{'_' + tag2 if tag2 else ''}{'_extra' if extra else ''}{'_w' + weight if weight else ''}{'_xf' if xf else ''}{'_p2b' if p2b else ''}{'_prm' if (params or rounds != 400) else ''}{'_' + tag3.replace(',', '_') if tag3 else ''}{'_decoy' if MODS['decoy'] else ''}{'_nosup' if MODS['nosup'] else ''}{'_lex' if MODS['lex'] else ''}{'_d2' if MODS['decoy2'] else ''}.json"
    json.dump(res, open(os.path.join(ROOT, "experiments", name), "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


def test_apply(tag, out_name, cand_t=None, cand_k=None, unseen_p2=None, rule_unseen=None, tag2=None, extra=False,
               p2b=None, unseen_blend=None, tag3=None, unseen_ce=None, unseen_ce2=None, unseen_extra=False,
               unseen_high=False, cr_add=None, unseen_highmodel=False, unseen_veto=None):
    """Final stage-3 model on all train band pairs -> test band; write both output TSVs in predict.py format.

    Optional cascade filter (cand_t, cand_k) and a separate model/rule for countries without training labels."""
    d = train_band(tag, tag2, extra, tag3)
    p2b_tr, p2b_te = p2b.split(",") if p2b else (None, None)   # "train OOF dir,test dir"
    if p2b:
        d = attach_p2b(d, "train", p2b_tr)
    fl = feat_list(tag2, extra, False, bool(p2b), tag3)
    m = lgb.train(PARAMS, lgb.Dataset(d[FEATS], d.y.values), 400)
    prm2 = {**PARAMS, **json.loads(os.environ.get("STACK_PARAMS", "{}"))}   # validated capacity override (eval --params)
    rounds2 = int(os.environ.get("STACK_ROUNDS", 400))
    wts = None
    if os.environ.get("UTILITY") == "1" and (tag2 or extra or p2b):   # metric-aware weights from the train-band OOF
        wts = utility_weights(d, oof_stage3(d, fl, None, json.loads(os.environ.get("STACK_PARAMS", "{}")), rounds2))
    m2 = lgb.train(prm2, lgb.Dataset(d[fl], d.y.values, weight=wts), rounds2) if (tag2 or extra or p2b) else None
    te = pd.read_parquet(os.path.join(KAG, "band", "test_band.parquet"), columns=["a_id", "r_id", "country", "p2"])
    s_us = pd.read_parquet(os.path.join(KAG, "out", f"scores_{tag}_US_test.parquet"))[["a_id", "r_id", "ce"]]
    s_in = pd.read_parquet(os.path.join(KAG, "out", f"scores_{tag}_India_test.parquet"))[["a_id", "r_id", "ce"]]
    n_band = len(te)
    te = te.merge(s_us.rename(columns={"ce": "ce_us"}), on=["a_id", "r_id"]).merge(
        s_in.rename(columns={"ce": "ce_in"}), on=["a_id", "r_id"])
    if len(te) != n_band:                                 # every test band pair must have both cross-country scores
        raise ValueError(f"test band {n_band} rows, {len(te)} after the CE merge")
    trained = {"India", "US"}
    du = None                                             # stacker data for countries without labels
    if unseen_ce or unseen_extra:
        du = train_band(unseen_ce or tag, None, bool(unseen_extra))
        if unseen_extra and "h_both" not in du:           # + label-free decoy features (decoy_feats.py)
            du = du.merge(pd.read_parquet(os.path.join(KAG, "band", "train_decoy.parquet")), on=["a_id", "r_id"], how="left")
    fu_feats = (FEATS + ([f for f in FEATS2 if f not in FEATS] if unseen_ce2 else [])
                + ([f for f in EXTRA if not (MODS["nosup"] and f.startswith("sup_"))] + DECOY
                   + (DECOY2 if MODS["decoy2"] else []) if unseen_extra else [])
                + (LEX if (unseen_extra and MODS["lex"]) else []))
    if unseen_ce2:                                        # second cross-country CE (e.g. e5s) as ce2 features for France
        s2 = pd.concat([pd.read_parquet(os.path.join(KAG, "out", f"scores_{unseen_ce2}_{c}_heldout.parquet"), columns=["a_id", "r_id", "ce"])
                        for c in ("US", "India")]).rename(columns={"ce": "ce2"})
        du = add_ce2(du.merge(s2, on=["a_id", "r_id"], how="left"))
    te["ce"] = np.where(te.country == "India", te.ce_us, np.where(te.country == "US", te.ce_in, (te.ce_us + te.ce_in) / 2))
    te = add_feats(te)
    te["p3"] = m.predict(te[FEATS])
    if tag2:
        s2 = [pd.read_parquet(os.path.join(KAG, "out", f"scores_{tag2}_half{h}_test.parquet"))[["a_id", "r_id", "ce"]]
              for h in (0, 1)]
        s2 = s2[0].merge(s2[1], on=["a_id", "r_id"], suffixes=("_0", "_1"))
        te = te.merge(s2.assign(ce2=(s2.ce_0 + s2.ce_1) / 2)[["a_id", "r_id", "ce2"]], on=["a_id", "r_id"], how="left")
        te = add_ce2(te)
    if extra:
        te = te.merge(pd.read_parquet(os.path.join(KAG, "band", "test_s3feats.parquet")), on=["a_id", "r_id"], how="left")
    for j, t3 in enumerate(tag3.split(",") if tag3 else []):
        te = merge_halves(te, t3, f"ce{3 + j}", "test")
    if p2b:
        te = attach_p2b(te, "test", p2b_te)
    if MODS["decoy"]:
        te = te.merge(pd.read_parquet(os.path.join(KAG, "band", "test_decoy.parquet")), on=["a_id", "r_id"], how="left")
    if MODS["lex"]:
        te = te.merge(pd.read_parquet(os.path.join(KAG, "band", "test_lex.parquet")), on=["a_id", "r_id"], how="left")
    if MODS["ops"]:
        te = te.merge(pd.read_parquet(os.path.join(KAG, "band", "test_ops.parquet")), on=["a_id", "r_id"], how="left")
    if MODS["pairf"]:
        te = te.merge(pd.read_parquet(os.path.join(KAG, "band", "test_pairf.parquet")), on=["a_id", "r_id"], how="left")
    if MODS["cnt"]:
        te = te.merge(pd.read_parquet(os.path.join(KAG, "band", "test_cnt.parquet")), on=["a_id", "r_id"], how="left")
    if len(te) != n_band:
        raise ValueError(f"test band {n_band} rows, {len(te)} after the feature merges (duplicate keys?)")
    if m2 is not None:                                    # in-country countries use the richer stack; others keep FEATS
        inc = te.country.isin(trained).values
        ce_cols = [c for c in fl if c.startswith("ce") and "_" not in c]
        if te.loc[inc, ce_cols].isna().any().any():       # in-country CE scores must cover every in-country band pair
            raise ValueError(f"in-country band pairs without a score in {ce_cols}")
        te.loc[inc, "p3"] = m2.predict(te.loc[inc, fl])
    out = os.path.join(ROOT, out_name)
    os.makedirs(out, exist_ok=True)
    mpath, cpath = os.path.join(out, "matching_results.tsv"), os.path.join(out, "candidate_pairs.tsv")
    for pth, h in ((mpath, "matched_entity_ids"), (cpath, "candidate_entity_ids")):
        open(pth, "w", encoding="utf-8", newline="\n").write(f"source1_entity_id\t{h}\n")
    s1 = pq.read_table(os.path.join(CACHE, "norm_test_s1.parquet"), columns=["id", "country"]).to_pandas()
    from predict import write_lists
    stats = {}
    for c in sorted(s1.country.unique()):
        ids = s1.id.values[s1.country.values == c]
        path = os.path.join(out_dir("test", dict(DEFAULT)), safe(c) + ".parquet")
        t = pq.read_table(path, columns=["a_id", "r_id", "a", "r"]).to_pandas()
        a, r = t.a.values.astype(np.int64), t.r.values.astype(np.int64)
        unseen = unseen_p2 is not None and c not in trained      # swap in another stage-2 model's probabilities
        rule = rule_unseen if (rule_unseen and c not in trained) else {}
        if unseen_blend is not None and c not in trained:
            # France: logit average of the specificity and no-spec stage-2 models, band and cascade filter on
            # the blend, FEATS stack with the cross-country CE; band pairs without a CE score (outside the spec band
            # that was scored) get the median score, exactly as in the LOCO test (loco_transfer.py blend, a research script not shipped)
            spec_dir, ns_dir = unseen_blend.split(",") if "," in unseen_blend else ("cache/predict_work", unseen_blend)
            ps = np.load(os.path.join(ROOT, spec_dir, f"p2_{safe(c)}.npy")).astype(np.float64)   # "spec dir,no-spec dir"
            pn = np.load(os.path.join(ROOT, ns_dir, f"p2_{safe(c)}.npy")).astype(np.float64)
            if len(ps) != len(t) or len(pn) != len(t):
                raise ValueError(f"{c}: blend p2 length mismatch")
            p = 1 / (1 + np.exp(-(logit(ps) + logit(pn)) / 2))
            p_filter = p.copy()
            # band = cascade-kept pairs only, exactly as in the LOCO test (research script loco_transfer.py, not shipped) (CE ranks/margins over the same population)
            keep_u = keep_mask(a, p, rank_in_entity(a, p), cand_t, cand_k) if cand_t is not None else np.ones(len(a), bool)
            bpos = np.flatnonzero(keep_u & (p > 0.02) & ((p < 0.98) | unseen_high))   # high: also re-score p >= 0.98
            tb = pd.DataFrame({"a_id": t.a_id.values[bpos], "r_id": t.r_id.values[bpos], "p2": p[bpos]})
            fu, fi, mu = s_us, s_in, m
            if unseen_ce:                                 # France-focused CE (ce_france.py): scores the whole blend band
                fu, fi = [pd.read_parquet(os.path.join(KAG, "out", f"scores_{unseen_ce}_{k}_france.parquet"))[["a_id", "r_id", "ce"]]
                          for k in ("US", "India")]
            if du is not None:
                mu = lgb.train({**PARAMS, **mono_params(fu_feats)}, lgb.Dataset(du[fu_feats], du.y.values), 400)
            tb = tb.merge(fu.rename(columns={"ce": "ce_us"}), on=["a_id", "r_id"], how="left").merge(
                fi.rename(columns={"ce": "ce_in"}), on=["a_id", "r_id"], how="left")
            tb["ce"] = (tb.ce_us + tb.ce_in) / 2
            has_ce = float(tb.ce.notna().mean())
            tb["ce"] = tb.ce.fillna(float(tb.ce.median()))
            tb = add_feats(tb)
            if unseen_ce2:
                c2 = [pd.read_parquet(os.path.join(KAG, "out", f"scores_{unseen_ce2}_{k}_test.parquet"))[["a_id", "r_id", "ce"]]
                      for k in ("US", "India")]
                c2 = c2[0].merge(c2[1], on=["a_id", "r_id"], suffixes=("_0", "_1"))
                tb = tb.merge(c2.assign(ce2=(c2.ce_0 + c2.ce_1) / 2)[["a_id", "r_id", "ce2"]], on=["a_id", "r_id"], how="left")
                tb["ce2"] = tb.ce2.fillna(float(tb.ce2.median()))
                tb = add_ce2(tb)
            if unseen_extra:                              # sidecar on the blend band (anchors from the blend p) + decoy
                from stage3_feats import build
                side = build("test", c, tb[["a_id", "r_id"]].assign(country=c), p=p.copy())
                tb = tb.merge(side, on=["a_id", "r_id"], how="left")
                dk = pd.read_parquet(os.path.join(KAG, "band", "test_decoy.parquet"))
                if unseen_high:
                    dk = pd.concat([dk, pd.read_parquet(os.path.join(KAG, "band", "test_decoy_hi.parquet"))]).drop_duplicates(["a_id", "r_id"])
                tb = tb.merge(dk, on=["a_id", "r_id"], how="left")
                if MODS["lex"]:
                    tb = tb.merge(pd.read_parquet(os.path.join(KAG, "band", "test_lex.parquet")), on=["a_id", "r_id"], how="left")
                miss = tb[DECOY].isna().all(axis=1).values
                if miss.any():                            # band keys outside the precomputed file (e.g. DK blend):
                    import decoy_feats as DF              # computed now, label-free, same tables, as in the LOCO test
                    Sd, Pd = DF.records("test", c)
                    fx = DF.features(tb.loc[miss, ["a_id", "r_id"]].reset_index(drop=True), Sd, Pd, DF.build_tables("test", c, Sd, Pd))
                    cols = [x for x in fx.columns if x in tb.columns and x not in ("a_id", "r_id")]
                    tb.loc[miss, cols] = fx[cols].values
                    print(c, "band pairs given decoy features now", int(miss.sum()), flush=True)
            p_blend = p.copy()
            p[bpos] = mu.predict(tb[fu_feats])
            if unseen_highmodel:                          # dedicated no-CE model for kept blend p >= 0.98 (high_model.py)
                import high_model as HM
                hm, hcal, hfeats = HM.train_final()
                hpos = np.flatnonzero(keep_u & (p_blend >= 0.98))
                p[hpos] = HM.apply_test(hm, hcal, hfeats, c, hpos, p_blend)
                print(c, "high-model pairs", len(hpos), flush=True)
            if unseen_veto is not None:                   # label-free marker veto (marker_veto.py; LOCO US->India +0.0044)
                import marker_veto as MV
                import decoy_feats as DF
                Sd, Pd = DF.records("test", c)
                mk = MV.markers(MV.lift_table("test", c, Sd, Pd), unseen_veto)
                kpos = np.flatnonzero(keep_u)
                v = MV.veto_mask(t.a_id.values[kpos], t.r_id.values[kpos], Sd, Pd, mk)
                p[kpos[v]] = 0.0
                print(c, "marker veto words", sorted(mk), "vetoed kept pairs", int(v.sum()), flush=True)
            if os.environ.get("UNSEEN_ADDR_SHIFT"):       # address-less pool pairs: logit + d (fr_addr_rule.py, a research script not shipped; off in the submission; LOCO +0.00018 / +0.00009 at 0.5)
                d_ = float(os.environ["UNSEEN_ADDR_SHIFT"])
                kpos = np.flatnonzero(keep_u)
                emp = pq.read_table(path, columns=["r_addr_empty"]).column("r_addr_empty").to_numpy()[kpos].astype(bool)
                sh = kpos[emp & (p[kpos] > 0)]
                pc_ = np.clip(p[sh], 1e-6, 1 - 1e-6)
                p[sh] = 1 / (1 + np.exp(-(np.log(pc_ / (1 - pc_)) + d_)))
                print(c, "address-less shift", d_, "pairs", len(sh), flush=True)
            print(c, "blend band pairs", len(tb), "with CE score", round(has_ce, 4), flush=True)
        elif unseen:                                      # no-spec stage-2 model, no stage 3 (no CE for its band)
            p = np.load(os.path.join(ROOT, unseen_p2, f"p2_{safe(c)}.npy")).astype(np.float64)
            if len(p) != len(t):
                raise ValueError(f"{c}: unseen p2 length {len(p)} != candidate table {len(t)}")
            p_filter, tb = p, te.iloc[:0]
        else:
            p = np.load(os.path.join(CACHE, "predict_work", f"p2_{safe(c)}.npy")).astype(np.float64)
            p_filter = p.copy()                           # the cascade filter uses stage-2 p2, before stage 3
            tb = te[te.country == c]
            pos = pd.MultiIndex.from_arrays([t.a_id.values, t.r_id.values]).get_indexer(
                pd.MultiIndex.from_arrays([tb.a_id.values, tb.r_id.values]))
            if (pos < 0).any():
                raise ValueError(f"{c}: band pairs not found in the candidate table")
            p[pos] = tb.p3.values
        keep = keep_mask(a, p_filter, rank_in_entity(a, p_filter), cand_t, cand_k) if cand_t is not None             else np.ones(len(a), bool)
        kept = np.flatnonzero(keep)
        A_, R_, P_, RID_ = a[kept], r[kept], p[kept], t.r_id.values[kept]
        n_cr = 0
        cr_file = os.path.join(ROOT, "cache", "cr_test", f"{safe(c)}.parquet")
        if cr_add is not None and os.path.exists(cr_file):  # character-retrieval pairs the blocker missed (cr_score.py, a research script not shipped; off in the submission)
            nk = pd.read_parquet(cr_file, columns=["a_id", "r_id"])
            pn_ = np.load(cr_file.replace(".parquet", "_p.npy")).astype(np.float64)
            mk = pn_ > cr_add
            codes = pd.factorize(np.concatenate([t.r_id.values[kept], nk.r_id.values[mk]]))[0]
            A_ = np.concatenate([A_, pd.Index(ids).get_indexer(nk.a_id.values[mk]).astype(np.int64)])
            R_ = codes.astype(np.int64)
            P_ = np.concatenate([P_, pn_[mk]])
            RID_ = np.concatenate([RID_, nk.r_id.values[mk]])
            n_cr = int(mk.sum())
        sel = Fast(A_, R_, P_, len(ids)).select(**rule)
        np.savez_compressed(os.path.join(out, f"final_p_{safe(c)}.npz"), a=A_, r=R_, p=P_,
                            n=len(ids))                  # kept pairs' final p, for decision-level studies (not submitted)
        np.save(os.path.join(out, f"all_p_{safe(c)}.npy"), p.astype(np.float32))   # every table pair (tm_assemble --sem table)
        write_lists(cpath, "c", ids, A_, RID_)
        write_lists(mpath, "m", ids, A_[sel], RID_[sel])
        n_pred = np.bincount(A_[sel], minlength=len(ids))
        n_cand = np.bincount(A_, minlength=len(ids))
        stats[c] = {"n_s1": int(len(ids)), "band_pairs": int(len(tb)), "pred_pairs": int(sel.sum()),
                    "model": ("spec/no-spec blend + FEATS stack" if (unseen_blend is not None and c not in trained)
                              else "no-spec stage 2" if unseen else "spec stage 2 + stage-3 stack"),
                    "rule": rule or {"T": 1.0, "min_p": 0.5}, "cand_per_s1": round(float(n_cand.mean()), 3), "retrieval_pairs_added": n_cr,
                    "cand_p95": int(np.percentile(n_cand, 95)), "cand_max": int(n_cand.max()),
                    "pred_singleton_rate": round(float((n_pred == 0).mean()), 4),
                    "matches_per_s1": round(float(n_pred.mean()), 4)}
        print(c, json.dumps(stats[c]), flush=True)
    json.dump(stats, open(os.path.join(out, "predict_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "test"])
    ap.add_argument("--tag", default="e5s")
    ap.add_argument("--out", default="output_ce")
    ap.add_argument("--cand_t", type=float, default=None, help="cascade filter: keep pairs with p2 > t")
    ap.add_argument("--cand_k", type=int, default=None, help="cascade filter: keep at most k pairs per S1")
    ap.add_argument("--unseen_p2", default=None, help="dir with p2_<country>.npy for countries without train labels")
    ap.add_argument("--rule_unseen", default=None, help='e.g. {"T": 2.0, "min_p": 0.55}')
    ap.add_argument("--tag2", default=None, help="second (in-country) cross-encoder tag, e.g. e5i")
    ap.add_argument("--extra", action="store_true", help="add the stage3_feats.py sidecar features")
    ap.add_argument("--weight", default=None, choices=["inv", "sqrt"], help="entity-balanced training weights (eval)")
    ap.add_argument("--xf", action="store_true", help="cross-field features (record name vs S1 address; needs --extra)")
    ap.add_argument("--p2b", default=None, help="second stage-2 p2 as a stage-3 feature: eval: train OOF dir; "
                    "test: 'train OOF dir,test dir' (e.g. cache/exp_final_s50_spec,cache/predict_work_s50)")
    ap.add_argument("--unseen_blend", default=None, help="countries without labels: logit average of the spec p2 and "
                    "the p2 in this dir (e.g. cache/predict_work_nospec) + FEATS stack")
    ap.add_argument("--tag3", default=None, help="further in-country cross-encoder tags, comma-separated, e.g. e5b or e5b,mml")
    ap.add_argument("--unseen_ce", default=None, help="CE tag for countries without labels (e.g. e5bc): "
                    "scores_<tag>_<US|India>_france.parquet + a FEATS stacker trained with that CE")
    ap.add_argument("--unseen_ce2", default=None, help="with --unseen_ce: second cross-country CE tag (e.g. e5s) as ce2 features")
    ap.add_argument("--unseen_veto", type=float, default=None, help="countries without labels: label-free marker veto, "
                    "lift threshold (marker_veto.py; e.g. 1.5)")
    ap.add_argument("--unseen_highmodel", action="store_true", help="countries without labels: dedicated HIGH model for p >= 0.98")
    ap.add_argument("--cr_add", type=float, default=None, help="add character-retrieval pairs with scorer p > this")
    ap.add_argument("--unseen_high", action="store_true", help="countries without labels: also re-score kept p >= 0.98 pairs")
    ap.add_argument("--unseen_extra", action="store_true", help="countries without labels: FEATS + sidecar + decoy stacker")
    ap.add_argument("--params", default=None, help='stage-3 LightGBM overrides, e.g. {"num_leaves": 63}')
    ap.add_argument("--rounds", type=int, default=400, help="stage-3 boosting rounds")
    ap.add_argument("--decoy", action="store_true", help="add decoy-aware features (decoy_feats.py)")
    ap.add_argument("--nosup", action="store_true", help="drop the supporting-record (coherence) features")
    ap.add_argument("--lex", action="store_true", help="add lexical-correction features (lexical.py, a research script not shipped; off in the submission)")
    ap.add_argument("--ops", action="store_true", help="add generator-operator features (feat_ops.py), in-country stack")
    ap.add_argument("--pairf", action="store_true", help="add all stage-1/spec pair features (band_feats.py), in-country stack")
    ap.add_argument("--cnt", action="store_true", help="add count-prior competition features (feat_cnt.py, a research script not shipped; off in the submission), in-country stack")
    ap.add_argument("--decoy2", action="store_true", help="add the extra house relations (signed shift, <=10, reorder, prefix)")
    a = ap.parse_args()
    MODS["decoy"], MODS["nosup"], MODS["lex"], MODS["decoy2"], MODS["ops"], MODS["pairf"], MODS["cnt"] = a.decoy, a.nosup, a.lex, a.decoy2, a.ops, a.pairf, a.cnt
    if a.cmd == "eval":
        evaluate(a.tag, a.tag2, a.extra, a.weight, a.xf, a.p2b, json.loads(a.params) if a.params else None, a.rounds, a.tag3)
    else:
        test_apply(a.tag, a.out, a.cand_t, a.cand_k, a.unseen_p2, json.loads(a.rule_unseen) if a.rule_unseen else None,
                   a.tag2, a.extra, a.p2b, a.unseen_blend, a.tag3, a.unseen_ce, a.unseen_ce2, a.unseen_extra, a.unseen_high, a.cr_add, a.unseen_highmodel, a.unseen_veto)
