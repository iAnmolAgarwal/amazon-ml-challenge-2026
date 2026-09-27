"""Approach 1: choose a decision rule for countries WITHOUT training labels, using leave-one-country-out proxies.

Inputs: saved held-out stage-2 probabilities of loco_s25spec_in2us (US held out) and loco_s25spec_us2in (India held
out), plus the pair keys (a, r, y) of the candidate tables. Rule family, applied after the one-owner step:
  T      : temperature on the logit (T > 1 = less confident), before the expected-F0.5 prefix selection
  min_p  : minimum probability floor of the prefix selection (0.5 = submitted rule)
  gate   : predict an empty list when the entity's best probability < gate (singleton guard); 0 = off
Selection uses only training labels: grid on each direction, cross-checked on the other direction (choose on A,
score on B), and the final rule maximises the mean of the two held-out macro F0.5 while improving both.

  python src/loco_rules.py
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from io_utils import ROOT, DATA, read_tsv                   # noqa: E402
from normalize import load_norm                            # noqa: E402
from pairs import out_dir, safe, DEFAULT                   # noqa: E402
from model import load_keys                                # noqa: E402
from decision import expected_f_select                     # noqa: E402

GRID = {"T": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0], "min_p": [0.5, 0.55, 0.6, 0.65, 0.7, 0.8], "gate": [0.0, 0.8, 0.9]}
HELD = {"loco_s25spec_in2us": "US", "loco_s25spec_us2in": "India"}


def f05(tp, fp, fn):
    den = 1.25 * tp + fp + 0.25 * fn
    return np.where(den == 0, 1.0, 1.25 * tp / np.where(den == 0, 1, den))


class Fast:
    """Precomputes the one-owner mask and the (entity, probability desc) order once; rules are then vectorised.

    Equivalent to decision.expected_f_select after the one-owner step (checked against it for the base rule).
    """

    def __init__(self, a, r, p, n):
        best = pd.Series(p).groupby(r).transform("max").values
        self.pe = np.where(p >= best, p, 0.0)
        self.a, self.n = a, n
        self.o = np.lexsort((-self.pe, a))
        ao = a[self.o]
        self.start = np.r_[True, ao[1:] != ao[:-1]]
        self.gid = np.cumsum(self.start) - 1
        self.ao = ao

    def select(self, T=1.0, min_p=0.5, gate=0.0):
        pe = self.pe[self.o]
        if T != 1.0:
            z = np.log(np.clip(pe, 1e-9, 1 - 1e-9) / np.clip(1 - pe, 1e-9, 1))
            pe = np.where(pe > 0, 1 / (1 + np.exp(-z / T)), 0.0)
        g = self.gid
        cs = np.cumsum(pe)
        first = np.flatnonzero(self.start)
        cum = cs - (cs[first] - pe[first])[g]
        k = np.arange(len(pe)) - first[g] + 1
        EN = np.bincount(g, weights=pe)[g]
        ef = 1.25 * cum / (k + 0.25 * EN)
        ef0 = np.exp(np.bincount(g, weights=np.log(np.clip(1 - pe, 1e-12, 1)))[g])
        bestf = np.maximum.reduceat(ef, first)[g]
        kk = np.where(ef >= bestf, k, 10 ** 9)
        bestk = np.minimum.reduceat(kk, first)[g]
        sel_o = (bestf > ef0) & (k <= bestk) & (pe > min_p)
        if gate > 0:
            top = np.maximum.reduceat(pe, first)[g]
            sel_o &= top >= gate
        sel = np.zeros(len(pe), bool)
        sel[self.o] = sel_o
        return sel


def load_block(exp, b, cnt):
    path = os.path.join(out_dir("train", dict(DEFAULT)), safe(b) + ".parquet")
    k = load_keys(path, ("a", "r", "y"))
    ids = load_norm("train", 1, ["id"], country=b).id.values
    p = np.load(os.path.join(ROOT, "cache", f"exp_{exp}", f"p2_{safe(b)}.npy")).astype(np.float64)
    return k["a"].astype(np.int64), k["r"].astype(np.int64), k["y"].astype(bool), p, cnt.reindex(ids).fillna(0).values


def score(fast, a, y, n_true, rule):
    n = len(n_true)
    sel = fast.select(**rule)
    tp = np.bincount(a[sel & y], minlength=n)
    fp = np.bincount(a[sel & ~y], minlength=n)
    F = f05(tp, fp, n_true - tp)
    return {"f05": round(float(F.mean()), 5), "singleton": round(float(F[n_true == 0].mean()), 5),
            "P": round(float(np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1).mean()), 5),
            "R": round(float(np.where(n_true > 0, tp / np.maximum(n_true, 1), 1).mean()), 5)}


def main():
    gt = read_tsv(os.path.join(DATA, "train", "train_ground_truth.tsv"))
    cnt = pd.Series(gt.matched_entity_ids.map(lambda x: 0 if x == "" else x.count(",") + 1).values,
                    index=gt.source1_entity_id.values)
    data = {}
    for exp, b in HELD.items():
        a, r, y, p, n_true = load_block(exp, b, cnt)
        fast = Fast(a, r, p, len(n_true))
        ref = expected_f_select(pd.DataFrame({"a": a, "p": fast.pe}), "p", min_p=0.5)
        assert (ref == fast.select()).all(), "vectorised rule differs from decision.expected_f_select"
        data[exp] = (fast, a, y, n_true)
    rules = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    table = []
    for rule in rules:
        row = {"rule": rule}
        for exp, b in HELD.items():
            row[b] = score(*data[exp], rule)
        row["mean"] = round((row["US"]["f05"] + row["India"]["f05"]) / 2, 5)
        table.append(row)
        print(json.dumps(row), flush=True)
    base = next(t for t in table if t["rule"] == {"T": 1.0, "min_p": 0.5, "gate": 0.0})
    best_us = max(table, key=lambda t: t["US"]["f05"])
    best_in = max(table, key=lambda t: t["India"]["f05"])
    both = [t for t in table if t["US"]["f05"] > base["US"]["f05"] and t["India"]["f05"] > base["India"]["f05"]]
    chosen = max(both, key=lambda t: t["mean"]) if both else base
    out = {"baseline_rule": base, "chosen_rule": chosen,
           "cross_check": {"chosen_on_US_scored_on_India": {"rule": best_us["rule"], "India": best_us["India"],
                                                            "India_gain": round(best_us["India"]["f05"] - base["India"]["f05"], 5)},
                           "chosen_on_India_scored_on_US": {"rule": best_in["rule"], "US": best_in["US"],
                                                            "US_gain": round(best_in["US"]["f05"] - base["US"]["f05"], 5)}},
           "n_rules": len(table)}
    os.makedirs(os.path.join(ROOT, "experiments"), exist_ok=True)
    json.dump({"summary": out, "table": table}, open(os.path.join(ROOT, "experiments", "loco_rules.json"), "w"), indent=1)
    print("SUMMARY", json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
