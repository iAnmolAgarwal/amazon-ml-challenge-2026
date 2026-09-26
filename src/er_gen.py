"""Generator-operator pair features (gen): how the pool record's name and address were derived from the S1 record.

usage: python src/er_gen.py --split train|test --parent <attempt> [--lo 0.003] [--hi 0.997] [--workers 8]

For the pairs of the parent attempt with lo <= p <= hi, from the raw records and the normalized views, label-free and
pair-local (one pool record against its S1; nothing about other candidates):
- numbers: multiset differences, typical decoy shifts, +-1/2, truncated or dropped digits, the operation that turns the
  S1 house number into the pool one, share of S1 numbers kept, first/last number equality;
- name edits: best edit between one-sided tokens (distance, position, digit-for-letter, doubled letter, transposition,
  confusable characters), edit-operation counts over the core names, prefix drop/add, reorder, join/split, acronyms,
  legal-form ids, fillers, honorifics, alias markers, web forms, id tags, native script;
- raw cues: exact / case-insensitive equality, upper case, empty or placeholder address, '#', PO box, fractions;
- address words only on one side, street equality and overlap.
Writes cache/feat/{split}_gen.parquet keyed by (s1_idx, pool_idx).
"""
import argparse
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz

from common import load_config

KEYS = ["s1_idx", "pool_idx"]
D = {3, 4, 5, 7, 9, 11, 13, 21}
LEGAL = {"inc": 1, "llc": 2, "ltd": 3, "limited": 3, "corp": 4, "corporation": 4, "co": 5, "company": 5, "lp": 6, "llp": 7, "pvt": 8, "private": 8, "pc": 9, "pllc": 10, "plc": 11, "public": 12, "sarl": 13, "sas": 14, "sa": 15, "eurl": 16, "sci": 17, "snc": 18}
FILLER = {"the", "center", "centre", "services", "service", "group", "and", "of", "shop", "store", "enterprises", "enterprise", "solutions", "company"}
HONOR = {"dr", "mr", "mrs", "ms", "shri", "sri", "smt", "st", "saint"}
ALIAS = re.compile(r"\b(aka|fka|dba|formerly|nee|née|d/b/a|a/k/a)\b", re.I)
CONF = {frozenset(p) for p in ["il", "o0", "l1", "s5", "b8", "z2", "g9", "i1", "e3", "a4", "t7"]}
VOW = set("aeiou")
NUM = re.compile(r"\d+")
TOK = re.compile(r"[^\W_]+", re.U)
NAMES = (["nA", "nB", "n_eq", "shift_d", "shift_12", "trunc_last", "drop_first", "num_delta", "keep_share", "first_eq", "last_eq", "h_op", "h_delta", "s_op", "s_delta", "h_lead0", "h_letter"]
         + ["ed_best", "ed_end", "ed_start", "ed_mid", "ed_digit", "ed_double", "ed_transp", "ed_conf", "one_swap", "tok_typo", "tok_unknown"]
         + ["core_lev", "core_ratio", "op_end", "op_vowel", "op_double", "op_digit", "op_ins", "op_del", "op_conf", "drop_pos", "prefix_drop", "prefix_add", "reorder", "joinsplit", "acronym"]
         + ["lg_s", "lg_p", "lg_ns", "lg_np", "fill_add", "fill_drop", "honor", "alias_s", "alias_p", "web_p", "idtag", "first_tok", "last_tok", "ntok_s", "ntok_p", "len_s", "len_p", "native_s", "native_p"]
         + ["raw_eq", "low_eq", "upper_p", "upper_s", "empty_p", "empty_s", "hash", "pobox", "frac", "aw_s", "aw_p", "street_eq", "addr_jac", "addr_ratio", "comp_jac"])


def op_code(a, b):
    if not a or not b:
        return -1, 0.0
    if a == b:
        return 0, 0.0
    ia, ib = int(a), int(b)
    d = ib - ia
    if len(a) > 1 and b == a[:-1] or len(b) > 1 and a == b[:-1]:
        return 1, float(d)
    if len(a) > 1 and b == a[1:] or len(b) > 1 and a == b[1:]:
        return 2, float(d)
    if d in D:
        return 3, float(d)
    if -d in D:
        return 4, float(d)
    if abs(d) <= 2:
        return 5, float(d)
    if a.lstrip("0") == b.lstrip("0"):
        return 11, 0.0
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diff) == 1:
            return 6, float(d)
        if len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]:
            return 7, float(d)
    if abs(len(a) - len(b)) == 1 and Levenshtein.distance(a, b) == 1:
        return 8, float(d)
    if b.startswith(a) or a.startswith(b):
        return 9, float(d)
    if b.endswith(a) or a.endswith(b):
        return 10, float(d)
    return 12, float(d)


def best_edit(xs, ys):
    best = None
    for x in xs:
        for y in ys:
            e = Levenshtein.distance(x, y)
            if best is None or e < best[0]:
                best = (e, x, y)
    return best


def edit_kind(x, y):
    ops = Levenshtein.editops(x, y)
    end = start = mid = digit = double = conf = 0
    for op in ops:
        pos = op.src_pos
        if pos >= len(x) - 1:
            end = 1
        elif pos == 0:
            start = 1
        else:
            mid = 1
        if op.tag == "replace":
            c1, c2 = x[op.src_pos], y[op.dest_pos]
            digit |= int(c1.isdigit() != c2.isdigit())
            conf |= int(frozenset((c1, c2)) in CONF)
        if op.tag in ("insert", "delete"):
            s, i = (y, op.dest_pos) if op.tag == "insert" else (x, op.src_pos)
            double |= int((i > 0 and s[i] == s[i - 1]) or (i + 1 < len(s) and s[i] == s[i + 1]))
    transp = int(len(x) == len(y) and len(ops) == 2 and sorted(x) == sorted(y))
    return end, start, mid, digit, double, transp, conf


def feats(r):
    sn, sa, pn, pa, sc, pc, sh, ph, sst, pst = r
    sn, sa, pn, pa, sc, pc = sn or "", sa or "", pn or "", pa or "", sc or "", pc or ""
    na, nb = NUM.findall(sa), NUM.findall(pa)
    A, B = list(na), list(nb)
    for x in na:
        if x in B:
            B.remove(x)
            A.remove(x)
    neq = len(na) - len(A)
    sd = s12 = tl = df = 0
    dmin = -1.0
    for a in A:
        for b in B:
            d = int(b) - int(a)
            sd |= int(d in D)
            s12 |= int(abs(d) <= 2)
            tl |= int((len(a) > 1 and b == a[:-1]) or (len(b) > 1 and a == b[:-1]))
            df |= int((len(a) > 1 and b == a[1:]) or (len(b) > 1 and a == b[1:]))
            dmin = abs(d) if dmin < 0 else min(dmin, abs(d))
    keep = neq / len(na) if na else -1.0
    fe = float(bool(na and nb and na[0] == nb[0])) if na and nb else -1.0
    le = float(bool(na and nb and na[-1] == nb[-1])) if na and nb else -1.0
    hop, hd = op_code(sh or "", ph or "")
    so, sdl = (op_code(A[0], min(B, key=lambda b: abs(int(b) - int(A[0])))) if A and B else (-1, 0.0))
    lead0 = int(bool(re.search(r"(^|\s)0\d", pa)))
    hlet = int(bool(re.search(r"\b\d+[a-zA-Z]\b", pa)))
    ts, tp = TOK.findall(sn.lower()), TOK.findall(pn.lower())
    so_t, po_t = [t for t in ts if t not in tp], [t for t in tp if t not in ts]
    be = best_edit(so_t, po_t)
    if be:
        ek = edit_kind(be[1], be[2])
        edb = be[0]
    else:
        ek, edb = (0,) * 7, -1
    one_swap = int(len(so_t) == 1 and len(po_t) == 1)
    typo = sum(1 for t in so_t if any(Levenshtein.distance(t, u) <= 2 for u in po_t))
    unk = len(po_t) - sum(1 for u in po_t if any(Levenshtein.distance(t, u) <= 2 for t in so_t))
    ops = Levenshtein.editops(sc, pc) if sc and pc else []
    oe = ov = odb = odg = oi = od = ocf = 0
    for op in ops:
        oe += int(op.src_pos >= len(sc) - 1)
        oi += int(op.tag == "insert")
        od += int(op.tag == "delete")
        if op.tag == "replace":
            c1, c2 = sc[op.src_pos], pc[op.dest_pos]
            ov += int(c1 in VOW and c2 in VOW)
            odg += int(c1.isdigit() or c2.isdigit())
            ocf += int(frozenset((c1, c2)) in CONF)
        else:
            s, i = (pc, op.dest_pos) if op.tag == "insert" else (sc, op.src_pos)
            odb += int((i > 0 and i < len(s) and s[i] == s[i - 1]) or (i + 1 < len(s) and s[i] == s[i + 1]))
    dpos = 0
    if len(so_t) == 1 and not po_t:
        i = ts.index(so_t[0])
        dpos = 1 if i == 0 else (3 if i == len(ts) - 1 else 2)
    pdrop = int(len(ts) > 1 and tp == ts[1:])
    padd = int(len(tp) > 1 and ts == tp[1:])
    reord = int(sorted(ts) == sorted(tp) and ts != tp)
    js = int("".join(ts) == "".join(tp) and ts != tp)
    acr = int(bool(ts and tp) and (("".join(t[0] for t in ts) == "".join(tp)) or ("".join(t[0] for t in tp) == "".join(ts))))
    ls = [LEGAL[t] for t in ts if t in LEGAL]
    lp = [LEGAL[t] for t in tp if t in LEGAL]
    fa = sum(1 for t in po_t if t in FILLER)
    fd = sum(1 for t in so_t if t in FILLER)
    hon = int(any(t in HONOR for t in po_t + so_t))
    nat_s = int(any(ord(c) > 0x900 for c in sn))
    nat_p = int(any(ord(c) > 0x900 for c in pn))
    aws = set(TOK.findall(sa.lower())) - set(NUM.findall(sa))
    awp = set(TOK.findall(pa.lower())) - set(NUM.findall(pa))
    so_a, po_a = len(aws - awp), len(awp - aws)
    ajac = len(aws & awp) / len(aws | awp) if aws | awp else -1.0
    ar = fuzz.ratio(NUM.sub("", sa.lower()), NUM.sub("", pa.lower())) / 100 if sa and pa else -1.0
    cs, cp = [c.strip().lower() for c in sa.split(",")[1:]], [c.strip().lower() for c in pa.split(",")[1:]]
    cjac = len(set(cs) & set(cp)) / len(set(cs) | set(cp)) if set(cs) | set(cp) else -1.0
    ste = float(bool(sst) and sst == pst) if sst or pst else -1.0
    ep = int(pa.strip() == "" or pa.strip().lower() in ("none", "<null>", "null", "nan"))
    es = int(sa.strip() == "" or sa.strip().lower() in ("none", "<null>", "null", "nan"))
    return [len(A), len(B), neq, sd, s12, tl, df, np.log1p(dmin) if dmin >= 0 else -1.0, keep, fe, le, hop, hd, so, sdl, lead0, hlet,
            edb, *ek, one_swap, typo, unk,
            Levenshtein.distance(sc, pc) if sc and pc else -1, fuzz.ratio(sc, pc) / 100 if sc and pc else -1.0, oe, ov, odb, odg, oi, od, ocf, dpos, pdrop, padd, reord, js, acr,
            ls[-1] if ls else 0, lp[-1] if lp else 0, len(ls), len(lp), fa, fd, hon, int(bool(ALIAS.search(sn))), int(bool(ALIAS.search(pn))), int(bool(re.search(r"www\.|\.com|\.in\b|\.net|\.org", pn.lower()))), int("id:" in pn.lower() or "id:" in sn.lower()), int(bool(ts and tp) and ts[0] == tp[0]), int(bool(ts and tp) and ts[-1] == tp[-1]), len(ts), len(tp), len(sn), len(pn), nat_s, nat_p,
            int(sn == pn), int(sn.lower() == pn.lower()), int(pn.isupper()), int(sn.isupper()), ep, es, int("#" in sa or "#" in pa), int(bool(re.search(r"\b(pmb|p\.?o\.? box)\b", (sa + " " + pa).lower()))), int(("/" in sa) != ("/" in pa)), so_a, po_a, ste, ajac, ar, cjac]


def records(cache, split, which):
    ix = "s1_idx" if which == "s1" else "pool_idx"
    nm = pl.read_parquet(cache / "norm" / f"{split}_{which}.parquet", columns=["entity_id", "name_core", "house", "street1"])
    raw = pl.read_parquet(cache / "raw" / f"{split}_source1.parquet") if which == "s1" else pl.concat([pl.read_parquet(cache / "raw" / f"{split}_source2.parquet"), pl.read_parquet(cache / "raw" / f"{split}_source3.parquet")])
    assert (raw["entity_id"] == nm["entity_id"]).all()
    nm = nm.with_columns(raw["business_name"].alias("name"), raw["business_address"].alias("addr"), raw["entity_id"].str.starts_with("S3-").cast(pl.Float32).alias("is3"))
    return nm.with_row_index(ix).with_columns(pl.col(ix).cast(pl.UInt32)).drop("entity_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--lo", type=float, default=0.003)
    ap.add_argument("--hi", type=float, default=0.997)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cache = Path(load_config()["cache_dir"])
    t = time.time()
    name = a.parent if a.split == "train" else f"test_{a.parent}"
    k = pl.read_parquet(cache / "oof" / f"{name}.parquet", columns=KEYS + ["prob"]).filter(pl.col("prob").is_between(a.lo, a.hi)).select(KEYS)
    s, p = records(cache, a.split, "s1"), records(cache, a.split, "pool")
    z = k.join(s, on="s1_idx", how="left").join(p, on="pool_idx", how="left", suffix="_p")
    rows = list(zip(z["name"], z["addr"], z["name_p"], z["addr_p"], z["name_core"], z["name_core_p"], z["house"], z["house_p"], z["street1"], z["street1_p"]))
    print(f"{a.split}: {len(rows):,} band pairs, {time.time() - t:.0f}s", flush=True)
    with Pool(a.workers) as pool:
        out = pool.map(feats, rows, chunksize=20000)
    arr = np.asarray(out, dtype=np.float32)
    df = pl.DataFrame({n: arr[:, j] for j, n in enumerate(NAMES)}).with_columns(z.select(KEYS).to_series(0), z.select(KEYS).to_series(1)).select(KEYS + NAMES)
    df = df.with_columns(is3=z["is3_p"])
    df.write_parquet(cache / "feat" / f"{a.split}_gen.parquet")
    print(f"wrote {a.split}_gen.parquet: {df.height:,} rows, {len(NAMES)} features, {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
