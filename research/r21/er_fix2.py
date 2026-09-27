"""Pool-context correction (er_fix) with extra label-free inputs for the two remaining non-twin miss types:
empty-address pool records whose name is exact or edited, and made-up pool names at a shared street address.

usage: python src/er_fix2.py <attempt_id> [--no-cv] [--no-test]

The attempt sets what er_fix.py reads (PARENT, BAND, PARAMS, ROUNDS, USE, NONEMPTY) and may add to USE:
- "emp": bracket-free name equality, alphanumeric name equality, and per-country name counts: S1 sharing the pool
  record's name, S1 sharing the S1's name, empty-address pool records sharing the S1's name, and their ratio;
- "addr": made-up pool name share (pool name words seen in no S1 name of either split), pool unit flag, and building
  twins among the pool record's claimers (claimers with the S1's house and street, best other such claimer's p);
- "cnt": out-of-fold S1 match-count prior from S1-only record signals (count model trained on the harness folds).
NONEMPTY "pw" also lets pool records with a made-up Latin name (pw_share = 1) gain probability.
Nothing compares a pair with the S1's other candidates. France and pairs outside the band keep p.
"""
import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

import er_fix as F
import er_model as M
from common import load_attempt, load_config
from er_pool import KEYS, PR, merge, pool_side

EMP = ["nb_eq", "an_eq", "n_s1_p", "n_s1_s", "n_pe_s", "pe_ratio"]
ADDR = ["pw_share", "p_unit", "bt_n", "bt_best"]
CNT = ["t_hat"]
CNT_IN = ["c_india", "c_ntok", "c_len", "c_unit", "c_nhouse", "c_nnum", "c_web", "c_alt", "c_twin", "c_pe", "c_nat"]


def core(e):
    return e.fill_null("").str.to_lowercase().str.replace_all(r"[^a-z0-9]", "")


def vocab(cache):
    v = pl.concat([pl.read_parquet(cache / "norm" / f"{s}_s1.parquet", columns=["name_core"]) for s in ("train", "test")])
    return v.select(pl.col("name_core").fill_null("").str.split(" ").explode().alias("w")).filter(pl.col("w") != "").unique()


def s1_side(cache, split):
    """Per-S1 and per-pool label-free counts within the split and country."""
    s = pl.read_parquet(cache / "norm" / f"{split}_s1.parquet", columns=["entity_id", "country_n", "name_join", "name_core", "name_alt", "is_web", "house", "street1", "nums"]).with_row_index("s1_idx")
    p = pl.read_parquet(cache / "norm" / f"{split}_pool.parquet", columns=["entity_id", "country_n", "name_join", "name_core"]).with_row_index("pool_idx")
    r = ["entity_id", "business_name", "business_address"]
    r1 = pl.read_parquet(cache / "raw" / f"{split}_source1.parquet", columns=r)
    r2 = pl.concat([pl.read_parquet(cache / "raw" / f"{split}_source{i}.parquet", columns=r) for i in (2, 3)])
    s = s.join(r1, on="entity_id", how="left", maintain_order="left").drop("entity_id")
    p = p.join(r2, on="entity_id", how="left", maintain_order="left").drop("entity_id")
    p = p.with_columns(emp=pl.col("business_address").fill_null("").str.strip_chars().str.to_lowercase().is_in(["none", ""]))
    ns = s.group_by("country_n", "name_join").len("n_s1")
    pe = p.filter(pl.col("emp")).group_by("country_n", "name_join").len("n_pe")
    w = vocab(cache)
    pw = p.select("pool_idx", w=pl.col("name_core").fill_null("").str.split(" ")).explode("w").filter(pl.col("w") != "")
    pw = pw.join(w.with_columns(k=pl.lit(1)), on="w", how="left").group_by("pool_idx").agg(pw_share=pl.col("k").is_null().mean())
    s = s.join(ns, on=["country_n", "name_join"], how="left").join(pe, on=["country_n", "name_join"], how="left").with_columns(pl.col("n_pe").fill_null(0))
    p = p.join(ns.rename({"n_s1": "n_s1_p"}), on=["country_n", "name_join"], how="left").join(pw, on="pool_idx", how="left")
    a, n = pl.col("business_address").fill_null(""), pl.col("business_name").fill_null("")
    s = s.with_columns(
        c_india=pl.col("country_n") == "india", c_ntok=n.str.split(" ").list.len(), c_len=n.str.len_chars(),
        c_unit=a.str.contains(r"(?i)\b(unit|suite|ste|apt|floor|fl|#)\b"), c_nhouse=pl.col("house").fill_null("") == "",
        c_nnum=pl.col("nums").fill_null("").str.split(" ").list.len(), c_web=pl.col("is_web"), c_alt=pl.col("name_alt").fill_null("") != "",
        c_twin=pl.col("n_s1"), c_pe=pl.col("n_pe"), c_nat=n.str.contains(r"[ऀ-෿]"),
    )
    s = s.with_columns([pl.col(c).cast(pl.Float32) for c in CNT_IN])
    return s, p


def count_prior(cfg, cache):
    """Out-of-fold predicted number of true matches per train S1 (harness folds), and a full-train model for test."""
    s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1); fo = M.folds(cfg, s1, gold)
    s, _ = s1_side(cache, "train")
    t = s.select("s1_idx", *CNT_IN).with_columns(pl.col("s1_idx").cast(pl.UInt32)).join(gold.group_by("s1_idx").len("t"), on="s1_idx", how="left").join(fo.select("s1_idx", "fold"), on="s1_idx", how="left").sort("s1_idx")
    X, y, fold = t.select(CNT_IN).to_numpy().astype(np.float32), t["t"].fill_null(0).to_numpy().astype(np.float64), t["fold"].to_numpy()
    prm = dict(objective="poisson", learning_rate=0.1, num_leaves=31, min_data_in_leaf=500, num_threads=10, verbose=-1, seed=42, deterministic=True, force_col_wise=True)
    out = np.empty(len(y))
    for f in range(5):
        tr, va = fold != f, fold == f
        out[va] = lgb.train(prm, lgb.Dataset(X[tr], y[tr]), 200).predict(X[va])
    full = lgb.train(prm, lgb.Dataset(X, y), 200)
    st, _ = s1_side(cache, "test")
    tt = full.predict(st.select(CNT_IN).to_numpy().astype(np.float32))
    print(f"count prior: oof corr {np.corrcoef(out, y)[0, 1]:.3f}, mean train {out.mean():.3f} test {tt.mean():.3f}", flush=True)
    return {"train": pl.DataFrame({"s1_idx": t["s1_idx"], "t_hat": out.astype(np.float32)}), "test": pl.DataFrame({"s1_idx": st["s1_idx"].cast(pl.UInt32), "t_hat": tt.astype(np.float32)})}


def building(pairs, s):
    """Claimers (p > 0.01) of each pool record that share this S1's house and street."""
    c = pairs.filter(pl.col("prob") > 0.01).join(s.select(pl.col("s1_idx").cast(pl.UInt32), bk=pl.when(pl.col("house").fill_null("") != "").then(pl.col("house") + "|" + pl.col("street1").fill_null(""))), on="s1_idx", how="left").filter(pl.col("bk").is_not_null())
    g, q = ["pool_idx", "bk"], pl.col("prob")
    c = c.with_columns(bt_n=pl.len().over(g), _t1=q.max().over(g))
    c = c.with_columns(_c1=(q == pl.col("_t1")).sum().over(g), _t2=pl.when(q < pl.col("_t1")).then(q).otherwise(None).max().over(g))
    c = c.with_columns(bt_best=pl.when((q == pl.col("_t1")) & (pl.col("_c1") == 1)).then(pl.col("_t2")).otherwise(pl.col("_t1")).fill_null(0.0))
    return c.select(KEYS + [pl.col("bt_n").cast(pl.Float32), pl.col("bt_best").cast(pl.Float32)])


def build(cache, split, pairs, mod, country, prior):
    lo, hi = mod.BAND
    s, p = F.records(cache, split)
    ss, ps = s1_side(cache, split)
    b = pool_side(pairs).filter((pl.col("prob") >= lo) & (pl.col("prob") <= hi))
    b = b.join(country, on="s1_idx", how="left").filter(pl.col("country_n") != "france").with_columns(india=(pl.col("country_n") == "india").cast(pl.Float32))
    b = b.join(F.twins(pairs, s), on=KEYS, how="left").with_columns([pl.col(x).fill_null(0.0) for x in F.TWIN])
    b = b.join(building(pairs, ss), on=KEYS, how="left").with_columns([pl.col(x).fill_null(0.0) for x in ("bt_n", "bt_best")])
    b = b.join(s, on="s1_idx", how="left").join(p, on="pool_idx", how="left")
    b = b.join(ss.select(pl.col("s1_idx").cast(pl.UInt32), n_s1_s="n_s1", n_pe_s="n_pe"), on="s1_idx", how="left")
    b = b.join(ps.select(pl.col("pool_idx").cast(pl.UInt32), "n_s1_p", "pw_share"), on="pool_idx", how="left")
    if prior is not None:
        b = b.join(prior, on="s1_idx", how="left")
    sn, pn, pa, sa = pl.col("s_business_name").fill_null(""), pl.col("p_business_name").fill_null(""), pl.col("p_business_address").fill_null(""), pl.col("s_business_address").fill_null("")
    sl, plw = core(sn), core(pn)
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
        punct_eq=f(F.punct(sn) == F.punct(pn)),
        punct_s=f(sn.str.count_matches(r"[^\w\s]")),
        punct_p=f(pn.str.count_matches(r"[^\w\s]")),
        join_eq=f(pl.col("s_name_join") == pl.col("p_name_join")),
        core_jac=f(st.list.set_intersection(pt).list.len() / st.list.set_union(pt).list.len().clip(1)),
        last_eq=f(sn.str.to_lowercase().str.replace_all(r"[^\w\s]", "").str.split(" ").list.last() == pn.str.to_lowercase().str.replace_all(r"[^\w\s]", "").str.split(" ").list.last()),
        p_in_s=f((plw.str.len_chars() > 0) & sl.str.contains(plw, literal=True)),
        s_in_p=f((sl.str.len_chars() > 0) & plw.str.contains(sl, literal=True)),
        len_r=f(plw.str.len_chars() / sl.str.len_chars().clip(1)),
        nb_eq=f(sl == core(pn.str.replace_all(r"\[[^\]]*\]|\([^\)]*\)", ""))),
        an_eq=f(sl == plw),
        pe_ratio=f(pl.col("n_pe_s") / pl.col("n_s1_s").clip(1)),
        p_unit=f(pa.str.contains(r"(?i)\b(unit|suite|ste|apt|floor|fl|#)\b")),
    )
    groups = {"pr": PR, "twin": F.TWIN, "flags": F.FLAGS, "raw": F.RAW, "emp": EMP, "addr": ADDR, "cnt": CNT}
    cols = ["india"] + [c for g in mod.USE for c in groups[g]]
    pr = np.clip(b["prob"].to_numpy().astype(np.float64), 1e-6, 1 - 1e-6)
    lp = np.log(pr / (1 - pr))
    X = np.column_stack([lp] + [b[c].cast(pl.Float32).fill_null(np.nan).to_numpy() for c in cols]).astype(np.float32)
    return b.select(KEYS + ["prob", "a_none", "n_nat", pl.col("pw_share").fill_null(0.0)]), lp, X, ["lp"] + cols


def restrict(mod, keys, p):
    rule = getattr(mod, "NONEMPTY", "full")
    if rule == "full":
        return p
    up = keys["a_none"].to_numpy() == 1
    if rule == "pw":
        up |= (keys["pw_share"].to_numpy() == 1) & (keys["n_nat"].to_numpy() == 0)
    return np.where(up, p, np.minimum(p, keys["prob"].to_numpy().astype(np.float64)))


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
    prior = count_prior(cfg, cache) if "cnt" in mod.USE else {"train": None, "test": None}
    s1 = M.load_s1(cfg, "train")
    gold = M.load_gold(cfg, s1)
    fo = M.folds(cfg, s1, gold)
    oof = pl.read_parquet(cache / "oof" / f"{mod.PARENT}.parquet", columns=KEYS + ["prob"])
    keys, lp, X, names = build(cache, "train", oof, mod, s1.select("s1_idx", "country_n"), prior["train"])
    lab = keys.join(fo.select("s1_idx", "fold"), on="s1_idx", how="left", maintain_order="left").join(gold.with_columns(y=pl.lit(1, pl.Int8)), on=KEYS, how="left", maintain_order="left")
    y, fold = lab["y"].fill_null(0).to_numpy(), lab["fold"].to_numpy()
    print(f"train: {len(keys):,} band pairs, {int(y.sum()):,} positives, {len(names)} features, {time.time() - t:.0f}s", flush=True)
    if not a.no_cv:
        out = np.empty(len(y))
        for f in sorted(np.unique(fold)):
            tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            _, out[va] = F.fit_predict(mod.PARAMS, mod.ROUNDS, X, y, lp, names, tr, va)
            print(f"fold {f}: {len(tr):,} train pairs, {time.time() - t:.0f}s", flush=True)
        merge(oof, keys, restrict(mod, keys, out)).write_parquet(cache / "oof" / f"{a.attempt_id}.parquet")
    if a.no_test:
        return
    m = lgb.train(mod.PARAMS, lgb.Dataset(X, y, init_score=lp, feature_name=names, free_raw_data=True), mod.ROUNDS)
    del X
    tst = pl.read_parquet(cache / "oof" / f"test_{mod.PARENT}.parquet", columns=KEYS + ["prob"])
    tkeys, tlp, TX, tnames = build(cache, "test", tst, mod, M.load_s1(cfg, "test").select("s1_idx", "country_n"), prior["test"])
    assert tnames == names
    tp = 1 / (1 + np.exp(-(tlp + m.predict(TX, raw_score=True))))
    merge(tst, tkeys, restrict(mod, tkeys, tp)).write_parquet(cache / "oof" / f"test_{a.attempt_id}.parquet")
    imp = sorted(zip(names, m.feature_importance("gain")), key=lambda x: -x[1])[:14]
    print(f"test: {len(tkeys):,} of {len(tst):,} pairs re-scored; top gain: " + ", ".join(f"{n} {g:.0f}" for n, g in imp) + f"; {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
