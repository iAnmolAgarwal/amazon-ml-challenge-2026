"""Side-car pair features (feat5): name / address specificity, house-number relations, legal forms, char n-grams.

usage: python src/er_feat5.py --split train|test [--country us|india|france] [--tag ""]
       python src/er_feat5.py --split train|test --merge [--tag ""]

Every feature compares one pool record with one S1 record only (no other candidates are looked at), and all
statistics are label-free, per split and per country (token weights from that split's S1 of the country).
- name specificity (ns_*): w(t) = 1 + ln((N+1)/(df(t)+1)) over name_core tokens, df over the country's S1; shared,
  S1-only and pool-only weight mass, their fractions of each side, the largest one-sided weight on each side,
  weighted cosine, one-sided token counts
- address specificity (as_*): the same over addr_norm tokens
- house relations (h_*, hd*): one house number is a suffix of the other (dropped leading digits), log |difference|,
  difference <= 10 (and on the same street), edit distance, same digits reordered, each side's house number among
  the other side's numbers, signed difference pool - S1 and whether it is a typical shift (1,2,3,4,5,7,9,11,13,21)
- units (u_*): unit / suite / floor number equal, conflicting, present on one side only
- legal forms (lg_*): legal-word sets from name_lat (website-glued suffixes included) equal, nested, disjoint, one-sided
- cos_nc3: cosine of char-3-gram vectors of name_join, IDF from the country's S1
Per country the pairs are processed in 5M chunks written to cache/feat/f5parts/; --merge concatenates them into
cache/feat/{split}_feat5{tag}.parquet (per-country chunks in (s1_idx, pool_idx) order).
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from common import load_config
from er_norm import LEGAL, MS, NAME_ABBR, TOKEN, WEB, fold, merge_split_legal

CANON = {"pvt": "private", "ltd": "limited", "inc": "incorporated", "corp": "corporation", "co": "company"}
GLUE = ("privatelimited", "pvtltd", "limited", "private", "llc", "inc", "corporation", "corp", "company", "ltd", "pvt", "llp")
UNIT = {"suite", "apartment", "unit", "room", "rm", "floor", "box"}
SHIFTS = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
NS = ["ns_sh", "ns_a1", "ns_b1", "ns_a1_frac", "ns_b1_frac", "ns_a1_max", "ns_b1_max", "ns_cos", "ns_n_a1", "ns_n_b1"]
AS = ["as_sh", "as_a1", "as_b1", "as_a1_max", "as_b1_max", "as_cos"]
HS = ["h_suffix", "h_diffl", "h_near", "h_near_st", "h_lev", "h_swap", "h_in_p", "h_in_s", "u_eq", "u_conf", "u_one"]
LG = ["lg_eq", "lg_sub", "lg_disj", "lg_one"]
HD = ["hd", "hd_shift"]
COLS = NS + AS + HS + LG + ["cos_nc3"] + HD
CHUNK = 5_000_000


def legal(raw):
    s = MS.sub(" ", fold(raw or ""))
    out = set()
    w = WEB.search(s)
    if w:
        d = w.group(1).replace("-", "").split(".")[-1]
        for g in GLUE:
            if d.endswith(g) and len(d) - len(g) >= 3:
                out |= {CANON.get(x, x) for x in (("private", "limited") if g in ("privatelimited", "pvtltd") else (g,))}
                break
    toks = merge_split_legal([NAME_ABBR.get(t, t) for t in TOKEN.findall(s.replace("&", " and "))])
    out |= {CANON.get(t, t) for i, t in enumerate(toks) if t in LEGAL and (t != "co" or i > 0)}
    return " ".join(sorted(out))


def unit(a):
    t = (a or "").split()
    for i in range(len(t) - 1):
        if t[i] in UNIT and t[i + 1][:1].isdigit():
            return t[i + 1].lstrip("0") or "0"
    return ""


def rs(m):
    return np.asarray(m.sum(axis=1)).ravel()


def prep(s1_txt, pool_txt, **kw):
    """Binary term matrices over a shared vocabulary; weights from the S1 document frequencies."""
    cv = CountVectorizer(binary=True, dtype=np.float32, lowercase=False, **kw)
    m = cv.fit_transform(s1_txt + pool_txt).tocsr()
    n = len(s1_txt)
    a, b = m[:n], m[n:]
    df = np.bincount(a.indices, minlength=m.shape[1])
    w = (1.0 + np.log((n + 1.0) / (df + 1.0))).astype(np.float32)
    W = sparse.diags(w)
    aw, bw = (a @ W).tocsr(), (b @ W).tocsr()
    return dict(a=a, b=b, aw=aw, bw=bw, ta=rs(aw), tb=rs(bw), qa=rs(aw.multiply(aw)), qb=rs(bw.multiply(bw)), ca=np.diff(a.indptr), cb=np.diff(b.indptr))


def pairfe(m, r, c, full, bs=1_000_000):
    keys = ["sh", "a1", "b1", "a1_max", "b1_max", "cos", "n_a1", "n_b1"] if full else ["cos"]
    out = {k: np.empty(len(r), np.float32) for k in keys}
    for s in range(0, len(r), bs):
        rr, cc = r[s : s + bs], c[s : s + bs]
        A, B, AW, BW = m["a"][rr], m["b"][cc], m["aw"][rr], m["bw"][cc]
        dot = rs(AW.multiply(BW))
        den = np.sqrt(m["qa"][rr] * m["qb"][cc])
        out["cos"][s : s + bs] = np.where(den > 0, dot / np.maximum(den, 1e-12), -1.0)
        if full:
            sh = rs(BW.multiply(A))
            nsh = rs(A.multiply(B))
            out["sh"][s : s + bs] = sh
            out["a1"][s : s + bs] = m["ta"][rr] - sh
            out["b1"][s : s + bs] = m["tb"][cc] - sh
            out["a1_max"][s : s + bs] = (AW - AW.multiply(B)).max(axis=1).toarray().ravel()
            out["b1_max"][s : s + bs] = (BW - BW.multiply(A)).max(axis=1).toarray().ravel()
            out["n_a1"][s : s + bs] = m["ca"][rr] - nsh
            out["n_b1"][s : s + bs] = m["cb"][cc] - nsh
    return out


def sparse_feats(mats, amap, bmap, k):
    mn, ma, mc = mats
    r, c = amap[k["s1_idx"].to_numpy()], bmap[k["pool_idx"].to_numpy()]
    assert (r >= 0).all() and (c >= 0).all()
    f = pairfe(mn, r, c, True)
    ta, tb = f["sh"] + f["a1"], f["sh"] + f["b1"]
    res = dict(ns_sh=f["sh"], ns_a1=f["a1"], ns_b1=f["b1"], ns_a1_frac=np.where(ta > 0, f["a1"] / np.maximum(ta, 1e-9), -1.0), ns_b1_frac=np.where(tb > 0, f["b1"] / np.maximum(tb, 1e-9), -1.0),
               ns_a1_max=f["a1_max"], ns_b1_max=f["b1_max"], ns_cos=f["cos"], ns_n_a1=f["n_a1"], ns_n_b1=f["n_b1"])
    f = pairfe(ma, r, c, True)
    res.update(as_sh=f["sh"], as_a1=f["a1"], as_b1=f["b1"], as_a1_max=f["a1_max"], as_b1_max=f["b1_max"], as_cos=f["cos"])
    res["cos_nc3"] = pairfe(mc, r, c, False)["cos"]
    return k.select("s1_idx", "pool_idx").with_columns(**{x: pl.Series(np.asarray(v, np.float32)) for x, v in res.items()})


def rec_attrs(df):
    return df.with_columns(
        lg=pl.Series([legal(x) for x in df["name_lat"].to_list()], dtype=pl.Utf8),
        un=pl.Series([unit(x) for x in df["addr_norm"].to_list()], dtype=pl.Utf8),
        hs=pl.Series(["".join(sorted(x)) for x in df["house"].to_list()], dtype=pl.Utf8),
    ).select("idx", "house", "street1", "nums", "lg", "un", "hs")


def pair_attrs(k, s1a, pa, workers):
    x = k.select("s1_idx", "pool_idx").join(s1a.rename({"idx": "s1_idx"}), on="s1_idx", how="left", maintain_order="left").join(
        pa.rename({"idx": "pool_idx"}).rename({c: c + "_p" for c in pa.columns if c != "idx"}), on="pool_idx", how="left", maintain_order="left")
    h1, h2 = pl.col("house"), pl.col("house_p")
    both = (h1 != "") & (h2 != "")
    neq = both & (h1 != h2)
    n1, n2 = h1.cast(pl.Int64, strict=False), h2.cast(pl.Int64, strict=False)
    diff = (n1 - n2).abs()
    sd = n2 - n1
    lev = cpdist(x["house"].to_list(), x["house_p"].to_list(), scorer=Levenshtein.distance, workers=workers)
    s1l, pll = pl.col("lg"), pl.col("lg_p")
    lgb = (s1l != "") & (pll != "")
    pad = lambda e: pl.concat_str([pl.lit(" "), e, pl.lit(" ")])
    sub = lgb & (s1l != pll) & (pad(s1l).str.contains(pad(pll), literal=True) | pad(pll).str.contains(pad(s1l), literal=True))
    u1, u2 = pl.col("un"), pl.col("un_p")
    out = x.with_columns(h_lev_raw=pl.Series(lev.astype(np.float32))).select(
        "s1_idx", "pool_idx",
        h_suffix=(neq & (h1.str.ends_with(h2) | h2.str.ends_with(h1))),
        h_diffl=pl.when(both & diff.is_not_null()).then((diff.cast(pl.Float64) + 1).log()).otherwise(-1.0),
        h_near=neq & (diff <= 10).fill_null(False),
        h_near_st=neq & (diff <= 10).fill_null(False) & (pl.col("street1") != "") & (pl.col("street1") == pl.col("street1_p")),
        h_lev=pl.when(both).then(pl.col("h_lev_raw")).otherwise(-1.0),
        h_swap=neq & (pl.col("hs") == pl.col("hs_p")),
        h_in_p=pl.when(h1 != "").then(pad(pl.col("nums_p")).str.contains(pad(h1), literal=True)).otherwise(None),
        h_in_s=pl.when(h2 != "").then(pad(pl.col("nums")).str.contains(pad(h2), literal=True)).otherwise(None),
        u_eq=(u1 != "") & (u1 == u2),
        u_conf=(u1 != "") & (u2 != "") & (u1 != u2),
        u_one=(u1 != "") ^ (u2 != ""),
        lg_eq=lgb & (s1l == pll),
        lg_sub=sub,
        lg_disj=lgb & (s1l != pll) & ~sub,
        lg_one=(s1l != "") ^ (pll != ""),
        hd=pl.when(both).then(sd.clip(-100000, 100000).cast(pl.Float32)).otherwise(None),
        hd_shift=pl.when(both & sd.is_not_null()).then(sd.is_in(SHIFTS)).otherwise(None),
    )
    return out.with_columns(*[pl.col(c).cast(pl.Float32).fill_null(-1.0) for c in HS + LG + ["hd_shift"]], pl.col("hd").cast(pl.Float32))


def build_country(cache, split, tag, ctry, workers):
    t = time.time()
    cols = ["country_n", "name_lat", "name_core", "name_join", "addr_norm", "house", "street1", "nums"]
    k_all = pl.scan_parquet(cache / "cand" / f"{split}_cand{tag}.parquet").filter(pl.col("country_n") == ctry).select("s1_idx", "pool_idx").collect().sort("s1_idx", "pool_idx")
    s1 = pl.scan_parquet(cache / "norm" / f"{split}_s1.parquet").select(cols).with_row_index("s1_idx").filter(pl.col("country_n") == ctry).collect().with_columns(pl.col("s1_idx").cast(pl.UInt32))
    need = k_all.select("pool_idx").unique()
    pool = pl.scan_parquet(cache / "norm" / f"{split}_pool.parquet").select(cols).with_row_index("pool_idx").filter(pl.col("country_n") == ctry).collect().with_columns(pl.col("pool_idx").cast(pl.UInt32)).join(need, on="pool_idx", how="semi")
    print(f"{split} {ctry}: {len(k_all):,} pairs, {len(s1):,} S1, {len(pool):,} pool, {time.time() - t:.0f}s", flush=True)
    amap = np.full(int(s1["s1_idx"].max()) + 1, -1, np.int64)
    amap[s1["s1_idx"].to_numpy()] = np.arange(len(s1))
    bmap = np.full(int(pool["pool_idx"].max()) + 1, -1, np.int64)
    bmap[pool["pool_idx"].to_numpy()] = np.arange(len(pool))
    mats = (
        prep(s1["name_core"].to_list(), pool["name_core"].to_list(), tokenizer=str.split, token_pattern=None),
        prep(s1["addr_norm"].to_list(), pool["addr_norm"].to_list(), tokenizer=str.split, token_pattern=None),
        prep(s1["name_join"].to_list(), pool["name_join"].to_list(), analyzer="char_wb", ngram_range=(3, 3)),
    )
    s1a = rec_attrs(s1.join(k_all.select("s1_idx").unique(), on="s1_idx", how="semi").rename({"s1_idx": "idx"}))
    pa = rec_attrs(pool.rename({"pool_idx": "idx"}))
    del s1, pool
    print(f"  matrices and record attributes {time.time() - t:.0f}s", flush=True)
    out = cache / "feat" / "f5parts"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(f"{split}{tag}_{ctry}_*.parquet"):
        old.unlink()
    for i, s in enumerate(range(0, len(k_all), CHUNK)):
        k = k_all.slice(s, CHUNK)
        part = sparse_feats(mats, amap, bmap, k).join(pair_attrs(k, s1a, pa, workers), on=["s1_idx", "pool_idx"], how="left", maintain_order="left")
        part.select(["s1_idx", "pool_idx"] + COLS).write_parquet(out / f"{split}{tag}_{ctry}_{i:03d}.parquet")
        print(f"  chunk {i}: {len(k):,} pairs, {time.time() - t:.0f}s", flush=True)


def merge(cache, split, tag):
    t = time.time()
    parts = sorted((cache / "feat" / "f5parts").glob(f"{split}{tag}_*.parquet"))
    n_cand = pl.scan_parquet(cache / "cand" / f"{split}_cand{tag}.parquet").select(pl.len()).collect().item()
    path = cache / "feat" / f"{split}_feat5{tag}.parquet"
    pl.scan_parquet(parts).sink_parquet(path)
    n = pl.scan_parquet(path).select(pl.len()).collect().item()
    assert n == n_cand, f"{n} feature rows vs {n_cand} candidates"
    print(f"wrote {path.name}: {n:,} rows from {len(parts)} parts in {time.time() - t:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--country")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cache = Path(load_config()["cache_dir"])
    workers = int(os.environ.get("RK_THREADS", os.cpu_count() or 4))
    if a.merge:
        merge(cache, a.split, a.tag)
        return
    countries = [a.country] if a.country else pl.scan_parquet(cache / "cand" / f"{a.split}_cand{a.tag}.parquet").select(pl.col("country_n").unique()).collect()["country_n"].sort().to_list()
    for c in countries:
        build_country(cache, a.split, a.tag, c, workers)
    if not a.country:
        merge(cache, a.split, a.tag)


if __name__ == "__main__":
    main()
