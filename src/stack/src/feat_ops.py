"""Generator-operator features for band pairs.

Pair-local and label-free (one pool record against its own S1):
  op_acro       : initials of one core name (>= 2 tokens) equal the other's space-free core name
  op_add_fill / op_add_other / op_drop_fill / op_drop_other : core-name words added to / dropped from the pool name,
                  split into filler words (label-free lift < 0.5 in marker_veto lift tables: center, services, the,
                  honorifics, et fils/cie ...) and other words
  op_filler_only: names differ, and only by filler words
  op_ed1        : unmatched name tokens pairing up at edit distance 1 (single typo), op_unmatched_after_ed1
  op_num_trunc  : an address number of one side is a proper suffix/prefix of one on the other (1995 vs 995)
  op_num_new    : address numbers present on one side only (count)
Writes <kaggle>/band/{train,test}_ops.parquet keyed by (a_id, r_id). Aggregates only.

  python src/feat_ops.py --split train
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from normalize import load_norm                           # noqa: E402
from stack_ce import KAG                                  # noqa: E402

# filler words = measured label-free lift < 0.5 + the generator's documented fillers
FILLERS = {"center", "centre", "services", "service", "the", "inc", "smt", "sri", "shri", "mr", "mrs", "dr", "partners",
           "p", "fils", "associes", "cie", "s", "co", "and", "et"}


def ed1(u, v):
    """True when strings u, v are at Levenshtein distance exactly 1."""
    if u == v or abs(len(u) - len(v)) > 1:
        return False
    if len(u) == len(v):                                  # one adjacent transposition (Damerau) counts as one typo
        d = [i for i in range(len(u)) if u[i] != v[i]]
        if len(d) == 2 and d[1] == d[0] + 1 and u[d[0]] == v[d[1]] and u[d[1]] == v[d[0]]:
            return True
    if len(u) > len(v):
        u, v = v, u
    i = 0
    while i < len(u) and u[i] == v[i]:
        i += 1
    return u[i:] == v[i + 1:] if len(u) < len(v) else u[i + 1:] == v[i + 1:]


def pair_ops(ca, cr, na, nr):
    """Operator features of one (S1, pool) pair from core names and address-number strings."""
    ta, tr = ca.split(), cr.split()
    sa, sr = set(ta), set(tr)
    add, drop = sr - sa, sa - sr
    af, df_ = sum(w in FILLERS for w in add), sum(w in FILLERS for w in drop)
    ao, do = len(add) - af, len(drop) - df_
    acro = 0
    if len(ta) >= 2 and "".join(w[0] for w in ta) == cr.replace(" ", ""):
        acro = 1
    elif len(tr) >= 2 and "".join(w[0] for w in tr) == ca.replace(" ", ""):
        acro = 1
    ua, ur = [w for w in drop if w not in FILLERS], [w for w in add if w not in FILLERS]
    e1, used = 0, set()
    for u in ua:
        for j, v in enumerate(ur):
            if j not in used and ed1(u, v):
                e1 += 1
                used.add(j)
                break
    xa, xr = set(na.split()), set(nr.split())
    trunc = int(any(x != y and (x.endswith(y) or y.endswith(x) or x.startswith(y) or y.startswith(x))
                    for x in xa - xr for y in xr - xa if len(x) >= 2 and len(y) >= 2))
    return (acro, af, ao, df_, do, int(bool(add or drop) and ao == 0 and do == 0), e1, len(ua) + len(ur) - 2 * e1,
            trunc, len(xa ^ xr))


COLS = ["op_acro", "op_add_fill", "op_add_other", "op_drop_fill", "op_drop_other", "op_filler_only", "op_ed1",
        "op_unmatched_after_ed1", "op_num_trunc", "op_num_new"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    a = ap.parse_args()
    t0 = time.time()
    band = pd.read_parquet(os.path.join(KAG, "band", f"{a.split}_band.parquet"), columns=["a_id", "r_id", "country"])
    parts = []
    for c in band.country.unique():
        b = band[band.country == c]
        S = load_norm(a.split, 1, ["id", "name_core", "addr_nums"], country=c).set_index("id")
        P = pd.concat([load_norm(a.split, k, ["id", "name_core", "addr_nums"], country=c) for k in (2, 3)]).set_index("id")
        ca, na = S.name_core.reindex(b.a_id.values).fillna("").values, S.addr_nums.reindex(b.a_id.values).fillna("").values
        cr, nr = P.name_core.reindex(b.r_id.values).fillna("").values, P.addr_nums.reindex(b.r_id.values).fillna("").values
        f = np.array([pair_ops(*x) for x in zip(ca, cr, na, nr)], dtype=np.float32)
        parts.append(pd.DataFrame(f, columns=COLS).assign(a_id=b.a_id.values, r_id=b.r_id.values))
        print(a.split, c, len(b), f"{time.time() - t0:.0f}s", {k: round(float(v), 4) for k, v in zip(COLS, f.mean(axis=0))}, flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(os.path.join(KAG, "band", f"{a.split}_ops.parquet"), index=False)
    print("written", len(out), flush=True)


if __name__ == "__main__":
    main()
