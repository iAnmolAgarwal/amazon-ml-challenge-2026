"""Lexical correction: learned likelihoods of the words and character edits that separate a pool record from its S1.

usage: python src/er_lexical.py <attempt_id>

The attempt sets SRC (attempt whose probabilities are corrected: train cache/oof/<SRC>.parquet, test
cache/oof/test_<SRC>.parquet), GATE_SRC and GATE (only pairs whose GATE_SRC probability is >= GATE are corrected),
LEX_PARAMS and LEX_ROUNDS.
Six token channels per pair, from the raw records: name words added / dropped (accents folded, native script
transliterated, punctuation removed), address words added / dropped (non-numeric address tokens), the legal-word
transition, and character edit operations between the two names with one character of context. For every token the
training pairs give owner-deduplicated counts among true matches and non-matches; a token seen >= 5 times scores
log((pos + .5)/(n_pos_owners + 1)) - log((neg + .5)/(n_neg_owners + 1)), shrunk by n/(n + 10). Each channel is summarised by 7
numbers (token count, max, min, sum, mean of scores, share of unseen tokens, max support): 42 features.
Train: each pair's features use counts from the other folds only (cross-fitted); the LightGBM correction (boosted from
logit(p_SRC)) is fitted on all gated train pairs. Test: counts from all train pairs.
Writes cache/oof/test_<attempt_id>.parquet (SRC test probabilities with the gated pairs corrected).
"""
import argparse
import collections
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein

import er_model as M
from common import load_attempt, load_config
from er_feat2 import PUNCT
from er_norm import addr_views, fold, latinize

KEYS = ["s1_idx", "pool_idx"]
CHANNELS = ["nadd", "ndel", "aadd", "adel", "legal", "nedit"]
LEGAL = set("private pvt ltd limited llc inc incorporated corp corporation company co llp pllc pc pvtltd".split())
CHUNK = 200_000


def name_tokens(s, india):
    return " ".join(PUNCT.sub(" ", fold(latinize(s or "", "name", india))).split())


def tokens(sn, sa, pn, pa, country):
    india = country == "india"
    na, nb = name_tokens(sn, india), name_tokens(pn, india)
    A, B = set(na.split()), set(nb.split())
    AA = {x for x in addr_views(latinize(sa or "", "addr", india))[0] if not x.isdigit()}
    BB = {x for x in addr_views(latinize(pa or "", "addr", india))[0] if not x.isdigit()}
    legal = " ".join(sorted(A & LEGAL)) + "->" + " ".join(sorted(B & LEGAL))
    ops = set()
    if len(na) <= 120 and len(nb) <= 120:
        for op, i, j in Levenshtein.editops(na, nb):
            ops.add(op + ":" + na[max(0, i - 1) : i + 2] + ">" + nb[max(0, j - 1) : j + 2])
    return {"nadd": B - A, "ndel": A - B, "aadd": BB - AA, "adel": AA - BB, "legal": {legal}, "nedit": ops}


def tok_chunk(rows):
    return [tokens(*r) for r in rows]


def count_chunk(args):
    """Owner-deduplicated token counts per fold and label for one chunk (rows sorted by owner)."""
    rows, owners, labels, folds = args
    T = tok_chunk(rows)
    c = {(f, y, ch): collections.Counter() for f in range(5) for y in (0, 1) for ch in CHANNELS}
    own = collections.defaultdict(set)
    seen = set()
    for t, o, y, f in zip(T, owners, labels, folds):
        own[(f, y)].add(o)
        for ch in CHANNELS:
            for tok in t[ch]:
                k = (o, y, ch, tok)
                if k not in seen:
                    seen.add(k)
                    c[(f, y, ch)][tok] += 1
    return c, {k: len(v) for k, v in own.items()}


def encode(T, allc, excl, den):
    excl = excl or []
    """42 features per pair from counts over all training folds minus the counts of the excluded fold (excl may be None)."""
    out = np.empty((len(T), 42), np.float32)
    for r, t in enumerate(T):
        row = []
        for ch in CHANNELS:
            vals, sup, unseen = [], [], 0
            a0, a1 = allc[ch]
            ex = [e[ch] for e in excl]
            for tok in t[ch]:
                neg = a0.get(tok, 0) - sum(e[0].get(tok, 0) for e in ex)
                pos = a1.get(tok, 0) - sum(e[1].get(tok, 0) for e in ex)
                n = neg + pos
                if n < 5:
                    unseen += 1
                    continue
                ratio = math.log((pos + 0.5) / (den[1] + 1)) - math.log((neg + 0.5) / (den[0] + 1))
                vals.append(ratio * n / (n + 10))
                sup.append(n)
            k = len(t[ch])
            row += [k, max(vals, default=0), min(vals, default=0), sum(vals), sum(vals) / len(vals) if vals else 0, unseen / (k or 1), max(sup, default=0)]
        out[r] = row
    return out


_STATS = None


def enc_chunk(args):
    """Features of rows; each row excludes the counts of its own fold and of fold `outer` (fold_ids None: all counts)."""
    rows, fold_ids, outer = args
    T = tok_chunk(rows)
    allc, perfold, den = _STATS
    if fold_ids is None:
        return encode(T, allc, [], dens_of(den, ()))
    out = np.empty((len(T), 42), np.float32)
    for f in range(5):
        i = [k for k, x in enumerate(fold_ids) if x == f]
        if i:
            ex = sorted({f} | ({outer} if outer is not None else set()))
            out[i] = encode([T[k] for k in i], allc, [perfold[g] for g in ex], dens_of(den, ex))
    return out


def dens_of(den, excl):
    return [sum(v for (g, y), v in den.items() if y == yy and g not in excl) for yy in (0, 1)]


def raw_pairs(cache, split, keys):
    """(s1 name, s1 address, pool name, pool address, country) for keys, via entity ids of the normalized tables."""
    s1 = pl.read_parquet(cache / "norm" / f"{split}_s1.parquet", columns=["entity_id", "country_n"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
    pool = pl.read_parquet(cache / "norm" / f"{split}_pool.parquet", columns=["entity_id"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32))
    r1 = pl.read_parquet(cache / "raw" / f"{split}_source1.parquet", columns=["entity_id", "business_name", "business_address"])
    rp = pl.concat([pl.read_parquet(cache / "raw" / f"{split}_source{i}.parquet", columns=["entity_id", "business_name", "business_address"]) for i in (2, 3)])
    a = s1.join(keys.select("s1_idx").unique(), on="s1_idx", how="semi").join(r1, on="entity_id", how="left").select("s1_idx", "country_n", sn="business_name", sa="business_address")
    b = pool.join(keys.select("pool_idx").unique(), on="pool_idx", how="semi").join(rp, on="entity_id", how="left").select("pool_idx", pn="business_name", pa="business_address")
    x = keys.join(a, on="s1_idx", how="left", maintain_order="left").join(b, on="pool_idx", how="left", maintain_order="left")
    return list(zip(x["sn"].to_list(), x["sa"].to_list(), x["pn"].to_list(), x["pa"].to_list(), x["country_n"].to_list()))


def gated(cache, split, mod, limit=None):
    g = pl.read_parquet(cache / "oof" / f"{'' if split == 'train' else 'test_'}{mod.GATE_SRC}.parquet", columns=KEYS + ["prob"]).filter(pl.col("prob") >= mod.GATE).select(KEYS)
    if limit:
        g = g.filter(pl.col("s1_idx") % limit == 0)
    p = pl.read_parquet(cache / "oof" / f"{'' if split == 'train' else 'test_'}{mod.SRC}.parquet", columns=KEYS + ["prob"])
    return g.join(p, on=KEYS, how="left").sort(KEYS), p


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def main():
    global _STATS
    ap = argparse.ArgumentParser()
    ap.add_argument("attempt_id")
    ap.add_argument("--cv", action="store_true", help="write cross-fitted out-of-fold train probabilities cache/oof/<attempt>.parquet (outer fold excluded from the token counts too) instead of the test file")
    ap.add_argument("--limit", type=int, help="smoke test: only S1 with s1_idx % limit == 0")
    a = ap.parse_args()
    cfg = load_config()
    mod = load_attempt(a.attempt_id)
    cache = Path(cfg["cache_dir"])
    procs = int(os.environ.get("RK_THREADS", os.cpu_count() or 4))
    t = time.time()
    s1 = M.load_s1(cfg, "train")
    gold = M.load_gold(cfg, s1)
    fo = M.folds(cfg, s1, gold)
    tr, _ = gated(cache, "train", mod, a.limit)
    tr = tr.join(fo.select(pl.col("s1_idx").cast(pl.UInt32), "fold"), on="s1_idx", how="left", maintain_order="left").join(gold.with_columns(y=pl.lit(1, pl.Int8)), on=KEYS, how="left", maintain_order="left").with_columns(pl.col("y").fill_null(0))
    rows = raw_pairs(cache, "train", tr.select(KEYS))
    owners, labels, folds = tr["s1_idx"].to_numpy(), tr["y"].to_numpy(), tr["fold"].to_numpy()
    print(f"train: {len(tr):,} gated pairs, {int(labels.sum()):,} positives, {time.time() - t:.0f}s", flush=True)
    cuts = list(range(0, len(rows), CHUNK))
    jobs = [(rows[s : s + CHUNK], owners[s : s + CHUNK], labels[s : s + CHUNK], folds[s : s + CHUNK]) for s in cuts]
    total = {(f, y, ch): collections.Counter() for f in range(5) for y in (0, 1) for ch in CHANNELS}
    den = collections.Counter()
    with Pool(procs) as pool:
        for c, d in pool.imap(count_chunk, jobs):
            for k, v in c.items():
                total[k].update(v)
            den.update(d)
    del jobs
    print(f"counts ready {time.time() - t:.0f}s", flush=True)

    allc = {ch: [collections.Counter(), collections.Counter()] for ch in CHANNELS}
    perfold = {f: {ch: [total[(f, 0, ch)], total[(f, 1, ch)]] for ch in CHANNELS} for f in range(5)}
    for f in range(5):
        for ch in CHANNELS:
            for y in (0, 1):
                allc[ch][y].update(total[(f, y, ch)])
    _STATS = (allc, perfold, den)
    del total
    L = logit(tr["prob"].to_numpy())
    names = ["logit_p"] + [f"{ch}_{s}" for ch in CHANNELS for s in ("n", "max", "min", "sum", "mean", "unseen", "support")]
    if a.cv:
        oof = np.empty(len(L))
        for f in range(5):
            with Pool(procs) as pool:
                Zf = np.concatenate(list(pool.imap(enc_chunk, [(rows[s : s + CHUNK], folds[s : s + CHUNK].tolist(), f) for s in cuts])))
            tri, vai = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            Xf = np.column_stack([L.astype(np.float32), Zf])
            mf = lgb.train(mod.LEX_PARAMS, lgb.Dataset(Xf[tri], labels[tri], init_score=L[tri], feature_name=names, free_raw_data=True), mod.LEX_ROUNDS)
            oof[vai] = 1 / (1 + np.exp(-(L[vai] + mf.predict(Xf[vai], raw_score=True))))
            print(f"  cv fold {f} {time.time() - t:.0f}s", flush=True)
        src = pl.read_parquet(cache / "oof" / f"{mod.SRC}.parquet", columns=KEYS + ["prob"])
        upd = tr.select(KEYS).with_columns(p4=pl.Series(oof.astype(np.float32)))
        src.join(upd, on=KEYS, how="left", maintain_order="left").with_columns(prob=pl.coalesce("p4", "prob").cast(pl.Float32)).select(KEYS + ["prob"]).write_parquet(cache / "oof" / f"{a.attempt_id}.parquet")
        print(f"wrote {a.attempt_id}.parquet (cross-fitted OOF) {time.time() - t:.0f}s", flush=True)
        return
    with Pool(procs) as pool:
        Z = np.concatenate(list(pool.imap(enc_chunk, [(rows[s : s + CHUNK], folds[s : s + CHUNK].tolist(), None) for s in cuts])))
    del rows
    X = np.column_stack([L.astype(np.float32), Z])
    m = lgb.train(mod.LEX_PARAMS, lgb.Dataset(X, labels, init_score=L, feature_name=names, free_raw_data=True), mod.LEX_ROUNDS)
    del X, Z
    print(f"model fitted {time.time() - t:.0f}s; top gain: " + ", ".join(f"{n} {g:.0f}" for n, g in sorted(zip(names, m.feature_importance('gain')), key=lambda x: -x[1])[:8]), flush=True)
    te, full = gated(cache, "test", mod, a.limit)
    rows = raw_pairs(cache, "test", te.select(KEYS))
    with Pool(procs) as pool:
        Z = np.concatenate(list(pool.imap(enc_chunk, [(rows[s : s + CHUNK], None, None) for s in range(0, len(rows), CHUNK)])))
    L = logit(te["prob"].to_numpy())
    p = 1 / (1 + np.exp(-(L + m.predict(np.column_stack([L.astype(np.float32), Z]), raw_score=True))))
    upd = te.select(KEYS).with_columns(p4=pl.Series(p.astype(np.float32)))
    out = full.join(upd, on=KEYS, how="left", maintain_order="left").with_columns(prob=pl.coalesce("p4", "prob").cast(pl.Float32)).select(KEYS + ["prob"])
    out.write_parquet(cache / "oof" / f"test_{a.attempt_id}{'_smoke' if a.limit else ''}.parquet")
    print(f"test: corrected {len(te):,} of {len(full):,} pairs; wrote test_{a.attempt_id}.parquet in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
