"""Final pool-context correction of a parent's pair probabilities: pool-side competition recomputed from the parent's
own probabilities, name-twin structure among the pool record's claimers, pool-record flags and raw-name pair features,
boosted from logit(parent p), for US/India pairs with lo <= p <= hi.

usage: python src/er_fix.py <attempt_id> [--no-cv] [--no-test]

The attempt sets PARENT, BAND (lo, hi), PARAMS, ROUNDS, USE (feature groups: "pr", "twin", "flags", "raw") and NONEMPTY.
With FRANCE = True (and TRAIN_PARENT for the train out-of-fold file) the model is fitted on US + India without the country
flag and applied to the France pairs of test_<PARENT> only; US / India rows keep the parent probability.
NONEMPTY:
"full" keeps the corrected p everywhere, "min" keeps it for empty-address pool records only and takes min(p, corrected p)
for the rest (decoys never have an empty address, so only those records may gain probability).
Pool-side only: nothing compares a pair with the S1's other candidates. France and pairs outside the band keep p.
Train: 5-fold out-of-fold on the harness folds -> cache/oof/<attempt>.parquet; test: one model on all train band pairs
-> cache/oof/test_<attempt>.parquet.
"""
import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

import er_model as M
from common import load_attempt, load_config
from er_pool import KEYS, PR, merge, pool_side

TWIN = ["tw_n", "tw_best", "nt_sum", "tw_share"]
FLAGS = ["a_none", "n_nat", "a_nat", "p_brk", "p_upper", "p_lower", "p_acc", "s_unit", "s_ntok", "p_ntok"]
RAW = ["raw_eq", "low_eq", "punct_eq", "punct_s", "punct_p", "join_eq", "core_jac", "last_eq", "p_in_s", "s_in_p", "len_r"]


def records(cache, split):
    """Raw and normalized name / address fields of S1 and pool records, keyed by s1_idx / pool_idx."""
    s = pl.read_parquet(cache / "norm" / f"{split}_s1.parquet", columns=["entity_id", "name_join", "name_core"]).with_row_index("s1_idx")
    p = pl.read_parquet(cache / "norm" / f"{split}_pool.parquet", columns=["entity_id", "name_join", "name_core"]).with_row_index("pool_idx")
    r1 = pl.read_parquet(cache / "raw" / f"{split}_source1.parquet", columns=["entity_id", "business_name", "business_address"])
    r2 = pl.concat([pl.read_parquet(cache / "raw" / f"{split}_source{i}.parquet", columns=["entity_id", "business_name", "business_address"]) for i in (2, 3)])
    s = s.join(r1, on="entity_id", how="left").drop("entity_id").with_columns(pl.col("s1_idx").cast(pl.UInt32))
    p = p.join(r2, on="entity_id", how="left").drop("entity_id").with_columns(pl.col("pool_idx").cast(pl.UInt32))
    return s.rename({c: "s_" + c for c in s.columns if c != "s1_idx"}), p.rename({c: "p_" + c for c in p.columns if c != "pool_idx"})


def twins(pairs, s):
    """Name-twin structure among the S1 claiming each pool record (p > 0.01)."""
    c = pairs.filter(pl.col("prob") > 0.01).join(s.select("s1_idx", "s_name_join"), on="s1_idx", how="left")
    g, q = ["pool_idx", "s_name_join"], pl.col("prob")
    c = c.with_columns(tw_n=pl.len().over(g), _ps=q.sum().over("pool_idx"), _pt=q.sum().over(g), _t1=q.max().over(g))
    c = c.with_columns(_c1=(q == pl.col("_t1")).sum().over(g), _t2=pl.when(q < pl.col("_t1")).then(q).otherwise(None).max().over(g))
    c = c.with_columns(tw_best=pl.when((q == pl.col("_t1")) & (pl.col("_c1") == 1)).then(pl.col("_t2")).otherwise(pl.col("_t1")).fill_null(0.0), nt_sum=pl.col("_ps") - pl.col("_pt"), tw_share=pl.col("_pt") / pl.col("_ps"))
    return c.select(KEYS + [pl.col(x).cast(pl.Float32) for x in TWIN])


def punct(e):
    return e.fill_null("").str.extract_all(r"[^\w\s]").list.unique().list.sort().list.join("")


def build(cache, split, pairs, mod, country, france=False):
    lo, hi = mod.BAND
    s, p = records(cache, split)
    b = pool_side(pairs).filter((pl.col("prob") >= lo) & (pl.col("prob") <= hi))
    b = b.join(country, on="s1_idx", how="left").filter((pl.col("country_n") == "france") if france else (pl.col("country_n") != "france")).with_columns(india=(pl.col("country_n") == "india").cast(pl.Float32))
    b = b.join(twins(pairs, s), on=KEYS, how="left").with_columns([pl.col(x).fill_null(0.0) for x in TWIN])
    b = b.join(s, on="s1_idx", how="left").join(p, on="pool_idx", how="left")
    sn, pn, pa, sa = pl.col("s_business_name").fill_null(""), pl.col("p_business_name").fill_null(""), pl.col("p_business_address").fill_null(""), pl.col("s_business_address").fill_null("")
    sl, plw = sn.str.to_lowercase().str.replace_all(r"[^a-z0-9]", ""), pn.str.to_lowercase().str.replace_all(r"[^a-z0-9]", "")
    st, pt = pl.col("s_name_core").fill_null("").str.split(" ").list.unique(), pl.col("p_name_core").fill_null("").str.split(" ").list.unique()
    f = lambda e: e.cast(pl.Float32)
    b = b.with_columns(
        a_none=f(pa.str.strip_chars().str.to_lowercase().is_in(["none", ""])),
        n_nat=f(pn.str.contains(r"[ऀ-෿]")),
        a_nat=f(pa.str.contains(r"[ऀ-෿]")),
        p_brk=f(pn.str.contains(r"[\[\(]")),
        p_upper=f((pn == pn.str.to_uppercase()) & (pn != pn.str.to_lowercase())),
        p_lower=f((pn == pn.str.to_lowercase()) & (pn != pn.str.to_uppercase())),
        p_acc=f(pn.str.contains(r"[À-ÿ]")),
        s_unit=f(sa.str.contains(r"(?i)\bunit\b")),
        s_ntok=f(sn.str.split(" ").list.len()),
        p_ntok=f(pn.str.split(" ").list.len()),
        raw_eq=f(sn == pn),
        low_eq=f(sn.str.to_lowercase() == pn.str.to_lowercase()),
        punct_eq=f(punct(sn) == punct(pn)),
        punct_s=f(sn.str.count_matches(r"[^\w\s]")),
        punct_p=f(pn.str.count_matches(r"[^\w\s]")),
        join_eq=f(pl.col("s_name_join") == pl.col("p_name_join")),
        core_jac=f(st.list.set_intersection(pt).list.len() / st.list.set_union(pt).list.len().clip(1)),
        last_eq=f(sn.str.to_lowercase().str.replace_all(r"[^\w\s]", "").str.split(" ").list.last() == pn.str.to_lowercase().str.replace_all(r"[^\w\s]", "").str.split(" ").list.last()),
        p_in_s=f((plw.str.len_chars() > 0) & sl.str.contains(plw, literal=True)),
        s_in_p=f((sl.str.len_chars() > 0) & plw.str.contains(sl, literal=True)),
        len_r=f(plw.str.len_chars() / sl.str.len_chars().clip(1)),
    )
    groups = {"pr": PR, "twin": TWIN, "flags": FLAGS, "raw": RAW}
    cols = ([] if getattr(mod, "FRANCE", False) else ["india"]) + [c for g in mod.USE for c in groups[g]]
    pr = np.clip(b["prob"].to_numpy().astype(np.float64), 1e-6, 1 - 1e-6)
    lp = np.log(pr / (1 - pr))
    X = np.column_stack([lp] + [b[c].cast(pl.Float32).fill_null(np.nan).to_numpy() for c in cols]).astype(np.float32)
    return b.select(KEYS + ["prob", "a_none"]), lp, X, ["lp"] + cols


def fit_predict(params, rounds, X, y, lp, names, tr, va):
    m = lgb.train(params, lgb.Dataset(X[tr], y[tr], init_score=lp[tr], feature_name=names, free_raw_data=True), rounds)
    return m, 1 / (1 + np.exp(-(lp[va] + m.predict(X[va], raw_score=True))))


def restrict(mod, keys, p):
    if getattr(mod, "NONEMPTY", "full") == "full":
        return p
    return np.where(keys["a_none"].to_numpy() == 1, p, np.minimum(p, keys["prob"].to_numpy().astype(np.float64)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attempt_id")
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--no-test", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    mod = load_attempt(a.attempt_id)
    cache = Path(cfg["cache_dir"])
    t = time.time()
    s1 = M.load_s1(cfg, "train")
    gold = M.load_gold(cfg, s1)
    fo = M.folds(cfg, s1, gold)
    oof = pl.read_parquet(cache / "oof" / f"{getattr(mod, 'TRAIN_PARENT', mod.PARENT)}.parquet", columns=KEYS + ["prob"])
    keys, lp, X, names = build(cache, "train", oof, mod, s1.select("s1_idx", "country_n"))
    lab = keys.join(fo.select("s1_idx", "fold"), on="s1_idx", how="left", maintain_order="left").join(gold.with_columns(y=pl.lit(1, pl.Int8)), on=KEYS, how="left", maintain_order="left")
    y, fold = lab["y"].fill_null(0).to_numpy(), lab["fold"].to_numpy()
    print(f"train: {len(keys):,} band pairs, {int(y.sum()):,} positives, {len(names)} features, {time.time() - t:.0f}s", flush=True)
    if not a.no_cv:
        out = np.empty(len(y))
        for f in sorted(np.unique(fold)):
            tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            _, out[va] = fit_predict(mod.PARAMS, mod.ROUNDS, X, y, lp, names, tr, va)
            print(f"fold {f}: {len(tr):,} train pairs, {time.time() - t:.0f}s", flush=True)
        merge(oof, keys, restrict(mod, keys, out)).write_parquet(cache / "oof" / f"{a.attempt_id}.parquet")
    if a.no_test:
        return
    m = lgb.train(mod.PARAMS, lgb.Dataset(X, y, init_score=lp, feature_name=names, free_raw_data=True), mod.ROUNDS)
    del X
    tst = pl.read_parquet(cache / "oof" / f"test_{mod.PARENT}.parquet", columns=KEYS + ["prob"])
    tkeys, tlp, TX, tnames = build(cache, "test", tst, mod, M.load_s1(cfg, "test").select("s1_idx", "country_n"), france=getattr(mod, "FRANCE", False))
    assert tnames == names
    tp = 1 / (1 + np.exp(-(tlp + m.predict(TX, raw_score=True))))
    merge(tst, tkeys, restrict(mod, tkeys, tp)).write_parquet(cache / "oof" / f"test_{a.attempt_id}.parquet")
    imp = sorted(zip(names, m.feature_importance("gain")), key=lambda x: -x[1])[:12]
    print(f"test: {len(tkeys):,} of {len(tst):,} pairs re-scored; top gain: " + ", ".join(f"{n} {g:.0f}" for n, g in imp) + f"; {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
