"""Label-free decoy-marker veto for countries without labels (validated leave-one-country-out).

The generator's decoys add one country-specific word to a real business name (and decoys often also shift the house
number). Per country, from ITS OWN candidate pairs and without labels, count for every word w that a pool name adds
to its S1 name (S1 words a subset of pool words, exactly one extra word): n(w), and the ratio of pairs whose pool
house is the S1 house shifted upward by a decoy shift {1,2,3,4,5,7,9,11,13,21} to pairs with equal houses. The lift
= that ratio / the ratio over all single-word additions. Markers = words with n >= N_MIN and lift >= LIFT.
Veto: final probability 0 for pairs whose pool name adds a marker word, then the usual one-owner + expected-F0.5 rule.

LOCO: markers of the HELD country from its own (label-free) pairs, applied to the cached held-out probabilities of a
France configuration (cache/loco_q/<tag>_<held>.npz). Aggregates only -> experiments/marker_veto_<tag>.json.

  python src/marker_veto.py dkd2e5bc_blend
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from io_utils import ROOT                                 # noqa: E402
from pairs import out_dir, safe, DEFAULT                 # noqa: E402
from loco_rules import Fast, f05                          # noqa: E402
import decoy_feats as D                                   # noqa: E402

SHIFTS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
N_MIN, LIFTS = 500, (1.2, 1.5, 2.0)


def added_word(ca, cr):
    """The single word a pool core name adds to its S1 core name, else ''."""
    if not ca or not cr:
        return ""
    ta, tr = set(ca.split()), set(cr.split())
    ad = tr - ta
    return next(iter(ad)) if len(ad) == 1 and ta <= tr else ""


def lift_table(split, c, S=None, P=None):
    """word -> (n, lift) over all candidate pairs of one country block (label-free)."""
    if S is None:
        S, P = D.records(split, c)
    pf = pq.ParquetFile(os.path.join(out_dir(split, dict(DEFAULT)), safe(c) + ".parquet"))
    cnt = defaultdict(lambda: [0, 0, 0])
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=["a_id", "r_id"]).to_pandas()
        ca, cr = S.core.reindex(t.a_id.values).fillna("").values, P.core.reindex(t.r_id.values).fillna("").values
        ha, hr = S.house.reindex(t.a_id.values).fillna("").values, P.house.reindex(t.r_id.values).fillna("").values
        for x, y, h1, h2 in zip(ca, cr, ha, hr):
            w = added_word(x, y)
            if not w:
                continue
            s = cnt[w]
            s[0] += 1
            if h1 and h2:
                if h1 == h2:
                    s[2] += 1
                elif h1.isdigit() and h2.isdigit() and int(h2) - int(h1) in SHIFTS:
                    s[1] += 1
    base = sum(v[1] for v in cnt.values()) / max(sum(v[2] for v in cnt.values()), 1)
    return {w: (n, (sh / max(eq, 1)) / max(base, 1e-9)) for w, (n, sh, eq) in cnt.items()}


def markers(table, lift, n_min=N_MIN):
    return {w for w, (n, l) in table.items() if n >= n_min and l >= lift}


def veto_mask(aid, rid, S, P, mk, broad=True):
    """True where the pool name adds a marker word to the S1 name. broad (adopted: US->India +0.00488 vs strict
    +0.00441): marker in the pool words and not in the S1 words, whatever else differs; strict: the only added word."""
    ca, cr = S.core.reindex(aid).fillna("").values, P.core.reindex(rid).fillna("").values
    if broad:
        return np.array([bool((set(y.split()) - set(x.split())) & mk) for x, y in zip(ca, cr)])
    return np.array([added_word(x, y) in mk for x, y in zip(ca, cr)])


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "dkd2e5bc_blend"
    res = {}
    for h in ("India", "US"):
        S, P = D.records("train", h)
        tab = lift_table("train", h, S, P)
        z = dict(np.load(os.path.join(ROOT, "cache", "loco_q", f"{tag}_{h}.npz")))
        n, y, a = len(z["n_true"]), z["y"].astype(bool), z["a"]
        t = pq.read_table(os.path.join(out_dir("train", dict(DEFAULT)), safe(h) + ".parquet"), columns=["a_id", "r_id"]).to_pandas()
        aid, rid = t.a_id.values[z["pos"]], t.r_id.values[z["pos"]]
        del t

        def F(q):
            sel = Fast(a, z["r"], q, n).select(T=2.0, min_p=0.6)
            tp = np.bincount(a[sel & y], minlength=n)
            fp = np.bincount(a[sel & ~y], minlength=n)
            return float(f05(tp, fp, z["n_true"] - tp).mean()), sel
        base, sel0 = F(z["q"])
        out = {"base": round(base, 5)}
        for L in LIFTS:
            mk = markers(tab, L)
            v = veto_mask(aid, rid, S, P, mk, broad=False)
            q2 = z["q"].copy()
            q2[v] = 0.0
            f2, _ = F(q2)
            out[f"lift{L}"] = {"markers": sorted(mk), "vetoed_emitted": int((sel0 & v).sum()), "vetoed_emitted_true": int((sel0 & v & y).sum()),
                               "f05": round(f2, 5), "gain": round(f2 - base, 5)}
        res[h] = out
        print(h, "held", json.dumps(out), flush=True)
    json.dump(res, open(os.path.join(ROOT, "experiments", f"marker_veto_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
