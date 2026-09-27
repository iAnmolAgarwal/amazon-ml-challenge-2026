"""Decoy-aware pair features (label-free) for stage 3 and the unseen-country stack.

The data generator makes decoys (pool records owned by no S1) as near-copies of real businesses: house number shifted by
one of SHIFTS, one per-country marker word inserted in the name, the legal form swapped, or a unit changed. Test is
about twice as decoy-dense as train, France the densest. Per pair (S1 a, pool record r), comparing r with a only:
- house: both present, equal, |difference| is a decoy shift, log |difference|, one number is a suffix of the other;
- legal form: equal, both present and different, one side only;
- name edits: number of core-name words added to / dropped from r;
- decoy-marker scores (label-free): per country and split, over pairs where a is among r's 3 best blocking matches,
  count for each added (dropped) word how often the edit co-occurs with a decoy-shifted house (n_d) versus an equal house
  (n_0); score = (n_d + 1) / (n_d + n_0 + 10). Country lift: logit(score) - log(sum n_d / sum n_0) of that country and
  kind, so an unseen, decoy-denser country's own table is on the train countries' scale. Features: max lifted score
  over added words, over dropped words, and of the one-for-one swap (dropped word -> added word).
House numbers use the French fallback parser when the address has no leading number (house_of); num1_eq_fb /
num1_missing_fb are the house-agreement flags with that parser (used by the unseen-country stack).
No labels are used anywhere. Outputs (keyed by a_id, r_id): <kaggle>/band/{train,test}_decoy.parquet for the train band
+ LOCO extra pairs (train) and the test band + France blend band (test). Prints counts only.

  python src/decoy_feats.py --split train
  python src/decoy_feats.py --split test
"""
import argparse
import math
import re
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from io_utils import ROOT                                 # noqa: E402
from normalize import load_norm                          # noqa: E402
from pairs import out_dir, safe, DEFAULT                 # noqa: E402

KAG = os.environ.get("CE_DIR", os.path.join(ROOT, os.pardir, "kaggle"))
SHIFTS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
DECOY = ["h_both", "h_equal", "h_shift", "h_logdiff", "h_suffix", "lg_equal", "lg_differ", "lg_one_side",
         "nm_n_added", "nm_n_dropped", "dk_added", "dk_dropped", "dk_swap", "lg_nested", "u_equal", "u_conflict", "u_one_side",
         "h_up_shift", "h_le10", "h_reorder", "h_prefix", "dk_add_rarity", "dk_drop_rarity", "dk_direction"]


STREET = (r"(rue|r|av|avenue|bd|boulevard|blvd|bvd|ch|chemin|crs|cr|cours|cour|quai|q|pl|place|all|allee|allée|imp|impasse|"
          r"rte|route|parvis|esplanade|lotissement|chaussee|chaussée|mail|venelle|ruelle|sente|digue|corniche|grande rue|"
          r"rond-point|square|sq|faubourg|fbg|passage|cite|cité|voie|hameau|zi|za|zac)")
FR_HOUSE = re.compile(r"\b(\d{1,4})\s*(?:bis|ter|quater|[a-d])?\s*,?\s*" + STREET + r"\b", re.I)
POBOX = re.compile(r"\b(bp|cs|tsa|cedex)\s*\d+", re.I)


UNIT = re.compile(r"\b(?:suite|ste|apt|apartment|unit|flat|floor|fl|room|rm|bldg|building|office|shop|"
                  r"etage|étage|bat|bât|batiment|bâtiment|porte|bureau|lot|appt|app)\b" + "\s*[.#:]?\s*(\w{1,5})|#\s*(\d{1,5})", re.I)


def unit_of(raw):
    """Unit / floor / suite identifiers of an address (sorted, joined); '' if none."""
    return " ".join(sorted({(m.group(1) or m.group(2)).lower() for m in UNIT.finditer(str(raw or ""))}))


def first_num(s):
    toks = str(s).split() if s is not None else []
    return toks[0] if toks and toks[0].isdigit() else ""


def house_of(nums, raw):
    """House number: first address number; if none, a 1-4 digit number (optionally bis/ter/quater or a letter) that
    precedes a street type (French house fallback; postal-box references removed first)."""
    h = first_num(nums)
    if h:
        return h
    m = FR_HOUSE.search(POBOX.sub(" ", str(raw or "")))
    return m.group(1).lstrip("0") or "0" if m else ""


def records(split, c):
    """S1 and pool record attributes of one country block, indexed by id."""
    cols = ["id", "name_core", "addr_nums", "legal", "addr_raw"]
    s1 = load_norm(split, 1, cols, country=c)
    pool = pd.concat([load_norm(split, k, cols, country=c) for k in (2, 3)], ignore_index=True)
    out = []
    for d in (s1, pool):
        out.append(pd.DataFrame({"core": d.name_core.fillna("").values,
                                 "house": [house_of(x, y) for x, y in zip(d.addr_nums.values, d.addr_raw.values)],
                                 "legal": d.legal.fillna("").astype(str).values,
                                 "unit": [unit_of(x) for x in d.addr_raw.values]}, index=d.id.values))
    return out


def edits(ca, cr):
    """Added / dropped core-name words of r relative to a (words of length >= 2)."""
    ta = {w for w in ca.split() if len(w) > 1}
    tr = {w for w in cr.split() if len(w) > 1}
    return sorted(tr - ta), sorted(ta - tr)


def house_rel(ha, hr):
    """0 = missing, 1 = equal, 2 = decoy shift, 3 = other difference."""
    if not ha or not hr:
        return 0
    if ha == hr:
        return 1
    return 2 if abs(int(ha) - int(hr)) in SHIFTS else 3


def build_tables(split, c, S, P):
    """Label-free decoy-marker counts over pairs where the S1 is among the record's 3 best blocking matches."""
    t = pq.read_table(os.path.join(out_dir(split, dict(DEFAULT)), safe(c) + ".parquet"), columns=["a_id", "r_id", "b_rk_r"])
    m = t.column("b_rk_r").to_numpy() <= 3
    a_ids = t.column("a_id").to_numpy(zero_copy_only=False)[m]
    r_ids = t.column("r_id").to_numpy(zero_copy_only=False)[m]
    ca, ha = S.core.reindex(a_ids).values, S.house.reindex(a_ids).values
    cr, hr = P.core.reindex(r_ids).values, P.house.reindex(r_ids).values
    cnt = {k: (Counter(), Counter()) for k in ("added", "dropped", "swap")}
    up, dn = Counter(), Counter()                         # added word with pool house = S1 house + shift / - shift
    NU = ND = n_eq = 0                                    # eligible up / down / equal-house pairs (exposures)
    for x, y, u, v in zip(ca, cr, ha, hr):
        if isinstance(u, str) and isinstance(v, str) and u and v:
            dd = int(v) - int(u)
            n_eq += dd == 0
            NU += dd in SHIFTS
            ND += -dd in SHIFTS
            if dd in SHIFTS or -dd in SHIFTS:
                tgt = up if dd in SHIFTS else dn
                for w in edits(x if isinstance(x, str) else "", y if isinstance(y, str) else "")[0]:
                    tgt[w] += 1
        rel = house_rel(u, v)
        if rel not in (1, 2):
            continue
        add, drop = edits(x, y)
        j = 0 if rel == 2 else 1                          # 0: n_d (decoy shift), 1: n_0 (equal house)
        for w in add:
            cnt["added"][j][w] += 1
        for w in drop:
            cnt["dropped"][j][w] += 1
        if len(add) == 1 and len(drop) == 1:
            cnt["swap"][j][drop[0] + ">" + add[0]] += 1
    tables = {}
    # word rarity in real S1 names: generator marker words are added to many pool names at an unchanged house but
    # almost never belong to a real S1 name; true-noise fillers (services, cie, fils) do occur in S1 names
    s1c = Counter(w for x in S.core.values for w in {v for v in str(x).split() if len(v) > 1})
    N1 = len(S)                                           # exposure-normalised: share of equal-house pairs adding w
    tables["rarity_add"] = ({w: math.log((cnt["added"][1][w] + 0.5) / (n_eq + 1)) - math.log((s1c[w] + 0.5) / (N1 + 1))
                             for w in cnt["added"][1]}, 0.0)   # vs share of S1 names containing w
    # directional evidence: decoys shift the house UPwards; a marker word rides with up-shifts much more than with
    # down-shifts (France markers 10-14x vs fillers 2-3x, baseline 3x); smoothed log-odds contrast, shrunk by support
    tables["direction"] = ({w: (up[w] + dn[w]) / (up[w] + dn[w] + 50) * (math.log((up[w] + 0.5) / (NU - up[w] + 0.5))
                                                                     - math.log((dn[w] + 0.5) / (ND - dn[w] + 0.5)))
                            for w in set(up) | set(dn)}, 0.0)
    tables["rarity_drop"] = ({w: math.log((cnt["dropped"][1][w] + 0.5) / (n_eq + 1)) - math.log((s1c[w] + 0.5) / (N1 + 1))
                              for w in cnt["dropped"][1]}, 0.0)
    for k, (nd, n0) in list(cnt.items()):
        lift = math.log((sum(nd.values()) + 1) / (sum(n0.values()) + 1))
        tables[k] = ({w: math.log((nd[w] + 1) / (n0[w] + 9)) - lift for w in set(nd) | set(n0)},   # logit(score) - lift
                     math.log(1 / 9) - lift)               # unseen word: prior score 1/10
    print(split, c, "table pairs", int(m.sum()), {k: len(v[0]) for k, v in tables.items()}, flush=True)
    return tables


def features(keys, S, P, tables):
    """Decoy features for (a_id, r_id) pairs."""
    ca, ha, la = S.core.reindex(keys.a_id).values, S.house.reindex(keys.a_id).values, S.legal.reindex(keys.a_id).values
    cr, hr, lr = P.core.reindex(keys.r_id).values, P.house.reindex(keys.r_id).values, P.legal.reindex(keys.r_id).values
    ua, ur = S.unit.reindex(keys.a_id).values, P.unit.reindex(keys.r_id).values
    n = len(keys)
    F = {f: np.zeros(n, np.float32) for f in DECOY}
    F["dk_swap"][:] = np.nan
    F2 = {"num1_eq_fb": np.zeros(n, np.float32), "num1_missing_fb": np.zeros(n, np.float32)}
    ta, tdr, ts = tables["added"], tables["dropped"], tables["swap"]
    for i in range(n):
        u, v = ha[i] if isinstance(ha[i], str) else "", hr[i] if isinstance(hr[i], str) else ""
        rel = house_rel(u, v)
        F["h_both"][i] = rel > 0
        F2["num1_eq_fb"][i] = rel == 1
        F2["num1_missing_fb"][i] = rel == 0
        F["h_equal"][i] = rel == 1
        F["h_shift"][i] = rel == 2
        if rel > 1:
            diff = int(v) - int(u)                         # pool minus S1: decoys shift the house upwards
            F["h_logdiff"][i] = math.log1p(abs(diff))
            F["h_suffix"][i] = u.endswith(v) or v.endswith(u)
            F["h_up_shift"][i] = diff in SHIFTS
            F["h_le10"][i] = abs(diff) <= 10
            F["h_reorder"][i] = sorted(u) == sorted(v)
            F["h_prefix"][i] = u.startswith(v) or v.startswith(u)
        x, y = la[i] if isinstance(la[i], str) else "", lr[i] if isinstance(lr[i], str) else ""
        F["lg_equal"][i] = bool(x) and x == y
        F["lg_differ"][i] = bool(x) and bool(y) and x != y
        F["lg_one_side"][i] = bool(x) != bool(y)
        sx, sy = set(x.split()), set(y.split())
        F["lg_nested"][i] = bool(sx) and bool(sy) and sx != sy and (sx <= sy or sy <= sx)
        p_, q_ = ua[i] if isinstance(ua[i], str) else "", ur[i] if isinstance(ur[i], str) else ""
        F["u_equal"][i] = bool(p_) and p_ == q_
        F["u_conflict"][i] = bool(p_) and bool(q_) and p_ != q_
        F["u_one_side"][i] = bool(p_) != bool(q_)
        add, drop = edits(ca[i] if isinstance(ca[i], str) else "", cr[i] if isinstance(cr[i], str) else "")
        F["nm_n_added"][i], F["nm_n_dropped"][i] = len(add), len(drop)
        F["dk_added"][i] = max((ta[0].get(w, ta[1]) for w in add), default=-5.0)
        F["dk_dropped"][i] = max((tdr[0].get(w, tdr[1]) for w in drop), default=-5.0)
        if len(add) == 1 and len(drop) == 1:
            F["dk_swap"][i] = ts[0].get(drop[0] + ">" + add[0], ts[1])
        F["dk_add_rarity"][i] = max((tables["rarity_add"][0].get(w, 0.0) for w in add), default=-5.0)
        F["dk_drop_rarity"][i] = max((tables["rarity_drop"][0].get(w, 0.0) for w in drop), default=-5.0)
        F["dk_direction"][i] = max((tables["direction"][0].get(w, 0.0) for w in add), default=-5.0)
    return pd.DataFrame({"a_id": keys.a_id.values, "r_id": keys.r_id.values, **F, **F2})


def hi_keys(split):
    """Cascade-kept pairs with blend p >= 0.98 (spec + no-spec logit average) of the unseen-country configuration."""
    from cand_budget import rank_in_entity, keep_mask
    from normalize import CACHE
    lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
    parts = []
    runs = ((("US", "in2us"), ("India", "us2in")) if split == "train" else (("France", None),))
    for c, d_ in runs:
        t = pq.read_table(os.path.join(out_dir(split, dict(DEFAULT)), safe(c) + ".parquet"), columns=["a_id", "r_id", "a"]).to_pandas()
        if split == "train":
            ps = np.load(os.path.join(ROOT, "cache", f"exp_loco_s25spec_{d_}", f"p2_{safe(c)}.npy")).astype(np.float64)
            pn = np.load(os.path.join(ROOT, "cache", f"exp_loco_s25_{d_}", f"p2_{safe(c)}.npy")).astype(np.float64)
        else:
            ps = np.load(os.path.join(CACHE, "predict_work", f"p2_{safe(c)}.npy")).astype(np.float64)
            pn = np.load(os.path.join(ROOT, "cache", "predict_work_nospec", f"p2_{safe(c)}.npy")).astype(np.float64)
        p = 1 / (1 + np.exp(-(lg(ps) + lg(pn)) / 2))
        a = t.a.values.astype(np.int64)
        m = keep_mask(a, p, rank_in_entity(a, p), 0.02, 10) & (p >= 0.98)
        parts.append(t.loc[m, ["a_id", "r_id"]].assign(country=c))
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--hi", action="store_true", help="keys = blend-p >= 0.98 cascade-kept pairs of the unseen-country "
                    "configuration (LOCO held-out countries on train, France on test) -> <split>_decoy_hi.parquet")
    a = ap.parse_args()
    if a.hi:
        keys = hi_keys(a.split)
    elif a.split == "train":
        keys = pd.concat([pd.read_parquet(os.path.join(KAG, "band", "train_band.parquet"), columns=["a_id", "r_id", "country"]),
                          pd.read_parquet(os.path.join(KAG, "band_extra", "extra_heldout.parquet"), columns=["a_id", "r_id", "country"])])
    else:
        keys = pd.concat([pd.read_parquet(os.path.join(KAG, "band", "test_band.parquet"), columns=["a_id", "r_id", "country"]),
                          pd.read_parquet(os.path.join(KAG, "band_extra", "france_band.parquet"), columns=["a_id", "r_id", "country"])])
    keys = keys.drop_duplicates(["a_id", "r_id"]).reset_index(drop=True)
    print("keys", len(keys), flush=True)
    parts = []
    for c in sorted(keys.country.unique()):
        S, P = records(a.split, c)
        tables = build_tables(a.split, c, S, P)
        k = keys[keys.country == c]
        f = features(k, S, P, tables)
        print(a.split, c, "pairs", len(f), "h_shift", round(float(f.h_shift.mean()), 4), "lg_differ",
              round(float(f.lg_differ.mean()), 4), flush=True)
        parts.append(f)
    res = pd.concat(parts, ignore_index=True)
    res.to_parquet(os.path.join(KAG, "band", f"{a.split}_decoy{'_hi' if a.hi else ''}.parquet"), index=False)
    print("written", len(res))


if __name__ == "__main__":
    main()
