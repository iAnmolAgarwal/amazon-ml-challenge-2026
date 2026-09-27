"""Macro F0.5 over all S1 entities (singletons included), plus diagnostics.

Inputs are pair frames: truth (s1, rid) and pred (s1, rid); `entities` lists every S1 id in the eval universe.
"""
import numpy as np
import pandas as pd


def per_entity(truth, pred, entities):
    ent = pd.Index(pd.unique(pd.Series(entities)), name="s1")
    t = truth[["s1", "rid"]].drop_duplicates()
    p = pred[["s1", "rid"]].drop_duplicates()
    nt = t.groupby("s1").size().reindex(ent, fill_value=0)
    npred = p.groupby("s1").size().reindex(ent, fill_value=0)
    tp = t.merge(p, on=["s1", "rid"]).groupby("s1").size().reindex(ent, fill_value=0)
    nt, npred, tp = nt.values.astype(float), npred.values.astype(float), tp.values.astype(float)
    P = np.divide(tp, npred, out=np.zeros_like(tp), where=npred > 0)
    R = np.divide(tp, nt, out=np.zeros_like(tp), where=nt > 0)
    den = 0.25 * P + R
    F = np.divide(1.25 * P * R, den, out=np.zeros_like(tp), where=den > 0)
    single = nt == 0
    F[single] = (npred[single] == 0).astype(float)
    # conventions for reporting P/R on edge cases: empty prediction -> P=1; singleton -> R=1
    P = np.where(npred == 0, 1.0, P)
    R = np.where(single, 1.0, R)
    return pd.DataFrame({"s1": ent, "n_true": nt, "n_pred": npred, "tp": tp, "P": P, "R": R, "F": F})


def summarize(pe, s1_country=None, truth=None, pred=None):
    out = {
        "macro_f05": float(pe.F.mean()),
        "mean_P": float(pe.P.mean()),
        "mean_R": float(pe.R.mean()),
        "singleton_acc": float(pe.loc[pe.n_true == 0, "F"].mean()) if (pe.n_true == 0).any() else float("nan"),
        "n_entities": int(len(pe)),
    }
    if s1_country is not None:
        c = pe.s1.map(s1_country)
        out["by_country"] = {k: float(v) for k, v in pe.F.groupby(c.values).mean().items()}
    if truth is not None and pred is not None:
        for src in ("S2", "S3"):
            t = truth[truth.rid.str.startswith(src)]
            p = pred[pred.rid.str.startswith(src)]
            out[f"f05_{src}_only"] = float(per_entity(t, p, pe.s1).F.mean())
    return out


def blocking_report(truth, cand, entities, n_r, s1_country=None):
    """Recall ceiling (pair-level and entity-level share of true pairs covered) and reduction ratio."""
    ent = pd.Index(pd.unique(pd.Series(entities)))
    t = truth[truth.s1.isin(ent)]
    hit = t.merge(cand[["s1", "rid"]].drop_duplicates(), on=["s1", "rid"], how="left", indicator=True)
    hit = hit["_merge"] == "both"
    n_c = len(cand)
    out = {"pair_recall": float(hit.mean()), "n_candidates": int(n_c),
           "cands_per_s1": n_c / max(len(ent), 1),
           "reduction_ratio": 1.0 - n_c / (len(ent) * n_r)}
    if s1_country is not None:
        cc = t.s1.map(s1_country).values
        out["pair_recall_by_country"] = {k: float(v) for k, v in hit.groupby(cc).mean().items()}
    return out


def _test():
    truth = pd.DataFrame({"s1": ["S1-00001", "S1-00001"], "rid": ["S2-00047", "S3-00812"]})
    pred = pd.DataFrame({"s1": ["S1-00001"] * 3, "rid": ["S2-00047", "S2-00193", "S3-00812"]})
    pe = per_entity(truth, pred, ["S1-00001"])
    assert abs(pe.F.iloc[0] - 0.7142857) < 1e-6, pe
    # singleton: empty -> 1, any prediction -> 0; matched entity with empty pred -> 0
    pe = per_entity(truth, pred.iloc[:0], ["S1-00001", "S1-9"])
    assert list(pe.F) == [0.0, 1.0]
    pe = per_entity(truth.iloc[:0], pd.DataFrame({"s1": ["S1-9"], "rid": ["S2-1"]}), ["S1-9"])
    assert list(pe.F) == [0.0]
    print("metric tests pass")


if __name__ == "__main__":
    _test()


def entity_scores(n_ent, a_sel, y_sel, n_true):
    """Array version. a_sel/y_sel: entity index and label of each SELECTED pair; n_true: true matches per entity
    (from ground truth, so matches missed by blocking still count). Returns per-entity F0.5, P, R."""
    n_pred = np.bincount(a_sel, minlength=n_ent).astype(float)
    tp = np.bincount(a_sel, weights=y_sel, minlength=n_ent).astype(float)
    den = n_pred + 0.25 * n_true
    F = np.divide(1.25 * tp, den, out=np.zeros(n_ent), where=den > 0)
    F[(n_true == 0) & (n_pred == 0)] = 1.0
    P = np.divide(tp, n_pred, out=np.ones(n_ent), where=n_pred > 0)
    R = np.divide(tp, n_true, out=np.ones(n_ent), where=n_true > 0)
    return F, P, R
