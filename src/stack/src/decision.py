"""Decision layer: pair probabilities -> per-S1 match lists.

F0.5 per entity = 1.25*TP / (k + 0.25*N)   (k = #predicted, N = #true). With calibrated pair probabilities we
choose, per S1 entity, the prefix size k (incl. k=0, worth P(N=0)=prod(1-p)) maximising the plug-in expected F0.5.
Before that, each S2/S3 record is assigned to at most one S1 (verified structure): probabilities of the same record
across S1 entities are normalised so they sum to <= 1, and non-best S1s can be dropped entirely.
"""
import numpy as np
import pandas as pd


def exclusive(df, p="p", mode="best"):
    """mode='best': keep only the argmax S1 per record. mode='norm': divide by max(1, sum_a p)."""
    d = df.copy()
    if mode == "best":
        idx = d.groupby("r")[p].idxmax()
        keep = np.zeros(len(d), bool)
        keep[d.index.get_indexer(idx)] = True
        d.loc[~keep, p] = 0.0
    elif mode == "norm":
        tot = d.groupby("r")[p].transform("sum")
        d[p] = d[p] / np.maximum(tot, 1.0)
    return d


def expected_f_select(df, p="p", extra_missing=0.0, min_p=0.0):
    """Return boolean mask of selected pairs using the expected-F0.5 prefix rule per S1 entity 'a'."""
    d = df[["a", p]].copy()
    d["_i"] = np.arange(len(d))
    d = d.sort_values(["a", p], ascending=[True, False], kind="mergesort")
    pv = d[p].values.astype(np.float64)
    a = d.a.values
    g = pd.Series(pv).groupby(a)
    cum = g.cumsum().values
    k = g.cumcount().values + 1
    EN = g.transform("sum").values + extra_missing
    ef = 1.25 * cum / (k + 0.25 * EN)
    logq = np.log(np.clip(1 - pv, 1e-12, 1))
    ef0 = np.exp(pd.Series(logq).groupby(a).transform("sum").values)
    # best prefix per entity
    best_prefix = pd.Series(ef).groupby(a).transform("max").values
    best_k = pd.Series(np.where(ef >= best_prefix, k, 10**9)).groupby(a).transform("min").values
    choose_nonempty = best_prefix > ef0
    sel = choose_nonempty & (k <= best_k) & (pv > min_p)
    out = np.zeros(len(d), bool)
    out[d._i.values] = sel
    return out


def threshold_select(df, p="p", t=0.5):
    return (df[p].values > t)
