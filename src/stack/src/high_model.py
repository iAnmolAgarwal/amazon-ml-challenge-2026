"""Dedicated high-confidence model for countries without labels (LOCO evaluation).

Decoys that keep the house number (marker word, legal swap, unit change) can receive blend p >= 0.98 and are then never
re-scored by the France band stack. This trains a small LightGBM WITHOUT cross-encoder or coherence (sup_*) features on
source-country cascade-kept pairs with blend p >= 0.90 (source-only out-of-fold LOCO probabilities), calibrates it
within the source (entity-fold OOF + Platt scaling), and replaces the probability of held-out-country kept pairs with
blend p >= 0.98. Everything else (band stack, one owner, rule T 2.0 / min_p 0.6) is unchanged.
Features: logit of the blend p, direct pair features of the candidate table (no blocking ranks), decoy features
(decoy_feats.py tables of the pair's own country and split).
Baseline = cached final probabilities of the current France configuration (the leave-one-country-out test with DECOY=1, SAVE_Q=dec_extra,
SAVE_VAR=extra). Aggregates only -> experiments/high_model.json.

  python src/high_model.py
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from io_utils import ROOT, DATA, read_tsv                 # noqa: E402
from pairs import out_dir, safe, DEFAULT                 # noqa: E402
from model import feature_cols, fold_of                   # noqa: E402
from loco_rules import Fast, f05                          # noqa: E402
from cand_budget import rank_in_entity, keep_mask         # noqa: E402
import decoy_feats as D                                   # noqa: E402

RULE = {"T": 2.0, "min_p": 0.6}
DIRS = {"US": ("India", "in2us"), "India": ("US", "us2in")}      # held-out: (source, LOCO direction)
PRM = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.9,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8, seed=0)
SKIP = {"via", "b_conj", "b_rk_r", "b_rk_a"}


W_SPEC = float(os.environ.get('BLEND_W', '0.5'))     # weight of the spec model in the France stage-2 blend


def lg(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def blend_p(country, d_):
    ps = np.load(os.path.join(ROOT, "cache", f"exp_loco_s25spec_{d_}", f"p2_{safe(country)}.npy")).astype(np.float64)
    pn = np.load(os.path.join(ROOT, "cache", f"exp_{os.environ.get('BLEND_NS', 'loco_s25')}_{d_}", f"p2_{safe(country)}.npy")).astype(np.float64)
    return 1 / (1 + np.exp(-(W_SPEC * lg(ps) + (1 - W_SPEC) * lg(pn))))


def frame(country, pos, p):
    """Pair features at table positions pos + decoy features + lp2."""
    path = os.path.join(out_dir("train", dict(DEFAULT)), safe(country) + ".parquet")
    cols = [c for c in feature_cols(path) if c not in SKIP]
    pf = pq.ParquetFile(path)                             # stream row groups, keep only requested positions
    order = np.argsort(pos)
    sp = np.asarray(pos)[order]
    parts, off = [], 0
    for g in range(pf.num_row_groups):
        n_g = pf.metadata.row_group(g).num_rows
        lo, hi = np.searchsorted(sp, [off, off + n_g])
        if hi > lo:
            parts.append(pf.read_row_group(g, columns=["a_id", "r_id", "y"] + cols).take(pa.array(sp[lo:hi] - off)).to_pandas())
        off += n_g
    f = pd.concat(parts, ignore_index=True)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    f = f.iloc[inv].reset_index(drop=True)                # back to the requested order
    S, P = D.records("train", country)
    dk = D.features(f[["a_id", "r_id"]], S, P, D.build_tables("train", country, S, P))
    f = pd.concat([f, dk.drop(columns=["a_id", "r_id"])], axis=1)
    f["lp2"] = lg(p[pos])
    return f, cols + [c for c in dk.columns if c not in ("a_id", "r_id")] + ["lp2"]


def blend_final(country):
    """Production blend inputs on train: pooled-model OOF p2 of the spec and no-spec stage 2."""
    from normalize import CACHE  # noqa: F401
    ps = np.load(os.path.join(ROOT, "cache", "exp_final_s25_spec", f"p2_{safe(country)}.npy")).astype(np.float64)
    pn = np.load(os.path.join(ROOT, "cache", "exp_final_s25", f"p2_{safe(country)}.npy")).astype(np.float64)
    return 1 / (1 + np.exp(-(lg(ps) + lg(pn)) / 2))


def train_final():
    """HIGH model for countries without labels: India+US OOF blend pairs with p >= 0.90 (cascade-kept), entity-fold OOF
    Platt calibration. Returns (model, calibrator, feature list)."""
    parts = []
    for c in ("India", "US"):
        p = blend_final(c)
        t = pq.read_table(os.path.join(out_dir("train", dict(DEFAULT)), safe(c) + ".parquet"), columns=["a"]).to_pandas()
        a = t.a.values.astype(np.int64)
        pos = np.flatnonzero(keep_mask(a, p, rank_in_entity(a, p), 0.02, 10) & (p >= 0.90))
        X, feats = frame(c, pos, p)
        parts.append(X)
    X = pd.concat(parts, ignore_index=True)
    fold = fold_of(X.a_id.values)
    oof = np.zeros(len(X))
    for k in range(5):
        m = fold != k
        oof[~m] = lgb.train(PRM, lgb.Dataset(X.loc[m, feats], X.y.values[m]), 200).predict(X.loc[~m, feats], raw_score=True)
    from sklearn.linear_model import LogisticRegression
    cal = LogisticRegression(C=1e6).fit(oof.reshape(-1, 1), X.y.values)
    return lgb.train(PRM, lgb.Dataset(X[feats], X.y.values), 200), cal, feats


def apply_test(model, cal, feats, country, pos, p):
    """Calibrated HIGH probabilities for test pairs at table positions pos (p = the country's blend p2)."""
    path = os.path.join(out_dir("test", dict(DEFAULT)), safe(country) + ".parquet")
    cols = [c for c in feats if c not in D.DECOY and c not in ("lp2", "num1_eq_fb", "num1_missing_fb")]
    pf = pq.ParquetFile(path)
    order = np.argsort(pos); sp = np.asarray(pos)[order]
    parts, off = [], 0
    for g in range(pf.num_row_groups):
        n_g = pf.metadata.row_group(g).num_rows
        lo, hi = np.searchsorted(sp, [off, off + n_g])
        if hi > lo:
            parts.append(pf.read_row_group(g, columns=["a_id", "r_id"] + cols).take(pa.array(sp[lo:hi] - off)).to_pandas())
        off += n_g
    f = pd.concat(parts, ignore_index=True)
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    f = f.iloc[inv].reset_index(drop=True)
    S, P = D.records("test", country)
    dk = D.features(f[["a_id", "r_id"]], S, P, D.build_tables("test", country, S, P))
    f = pd.concat([f, dk.drop(columns=["a_id", "r_id"])], axis=1)
    f["lp2"] = lg(p[pos])
    return cal.predict_proba(model.predict(f[feats], raw_score=True).reshape(-1, 1))[:, 1]


def main():
    res = {}
    for held, (src, d_) in DIRS.items():
        # source: its own OOF LOCO blend p, cascade-kept, p >= 0.90
        ps_ = blend_p(src, d_)
        t = pq.read_table(os.path.join(out_dir("train", dict(DEFAULT)), safe(src) + ".parquet"), columns=["a"]).to_pandas()
        a = t.a.values.astype(np.int64)
        pos_s = np.flatnonzero(keep_mask(a, ps_, rank_in_entity(a, ps_), 0.02, 10) & (ps_ >= 0.90))
        Xs, feats = frame(src, pos_s, ps_)
        fold = fold_of(Xs.a_id.values)
        oof = np.zeros(len(Xs))
        for k in range(5):
            m = fold != k
            mdl = lgb.train(PRM, lgb.Dataset(Xs.loc[m, feats], Xs.y.values[m]), 200)
            oof[~m] = mdl.predict(Xs.loc[~m, feats], raw_score=True)
        from sklearn.linear_model import LogisticRegression
        cal = LogisticRegression(C=1e6).fit(oof.reshape(-1, 1), Xs.y.values)     # Platt scaling on source OOF only
        full = lgb.train(PRM, lgb.Dataset(Xs[feats], Xs.y.values), 200)
        # held-out: cached final probabilities of the current France configuration
        base = os.environ.get("HM_BASE", "dec_extra")     # cached baseline of the configuration under test
        z = dict(np.load(os.path.join(ROOT, "cache", "loco_q", f"{base}_{held}.npz")))
        hi = z["pblend"] >= 0.98
        ph_all = blend_p(held, d_)
        Xh, _ = frame(held, z["pos"][hi], ph_all)
        qh = cal.predict_proba(full.predict(Xh[feats], raw_score=True).reshape(-1, 1))[:, 1]
        n = len(z["n_true"])
        out = {"source_pairs": int(len(Xs)), "source_neg": int((Xs.y == 0).sum()), "held_high_pairs": int(hi.sum()),
               "held_high_neg": int((~z["y"][hi]).sum())}
        q_new = z["q"].copy()
        q_new[hi] = qh
        np.savez(os.path.join(ROOT, "cache", "loco_q", f"{base}_high_{held}.npz"), **{**z, "q": q_new})   # for combo studies
        sels = {}
        for name, q in (("base", z["q"]), ("high_model", q_new)):
            sel = Fast(z["a"], z["r"], q, n).select(**RULE)
            sels[name] = sel
            y = z["y"].astype(bool)
            tp = np.bincount(z["a"][sel & y], minlength=n)
            fp = np.bincount(z["a"][sel & ~y], minlength=n)
            F = f05(tp, fp, z["n_true"] - tp)
            out[name] = {"f05": round(float(F.mean()), 5), "singleton": round(float(F[z["n_true"] == 0].mean()), 5),
                         "per_s1": round(float(sel.sum() / n), 3),
                         "high_removed_tp": int(((~sel) & y & hi).sum()), "high_removed_fp": int(((~sel) & ~y & hi).sum())}
        out["gain"] = round(out["high_model"]["f05"] - out["base"]["f05"], 5)
        yb = z["y"].astype(bool)
        drop, add = sels["base"] & ~sels["high_model"], ~sels["base"] & sels["high_model"]
        out["transitions"] = {"dropped_tp": int((drop & yb).sum()), "dropped_fp": int((drop & ~yb).sum()),
                              "added_tp": int((add & yb).sum()), "added_fp": int((add & ~yb).sum())}
        res[f"{src}->{held}"] = out
        print(src, "->", held, json.dumps(out), flush=True)
    json.dump(res, open(os.path.join(ROOT, "experiments", "high_model.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
