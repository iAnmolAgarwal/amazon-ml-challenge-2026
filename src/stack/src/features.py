"""Pairwise features for (S1 record, S2/S3 record) candidate pairs — memory-bounded.

S1-side vectors for one country block are built once (hashed TF-IDF, IDF estimated on S1 of the block, unsupervised);
S2/S3 records are processed chunk by chunk, so peak memory ~ S1 matrices + one chunk + its pairs.
Country is never a feature.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import process, fuzz, distance
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

NF = 2 ** 20
FAMILIES = {
    # name: (column, hashing kwargs, idf?)
    "name_c3": ("name_ns", dict(analyzer="char", ngram_range=(3, 3)), True),
    "name_w": ("name_core", dict(token_pattern=r"\S+"), True),
    "skel_w": ("name_skel", dict(token_pattern=r"\S+"), True),
    "skel_c2": ("skel_ns", dict(analyzer="char", ngram_range=(2, 3)), True),
    "addr_w": ("addr_norm", dict(token_pattern=r"\S+"), True),
    "nums": ("addr_nums", dict(token_pattern=r"\S+"), False),
    "legal": ("legal", dict(token_pattern=r"\S+"), False),
}
COS = ["name_c3", "name_w", "skel_w", "skel_c2", "addr_w"]
# addr_c3 (char-trigram address cosine) was dropped: ~470 MB for the US S1 block on an 8 GB machine; covered by
# addr_w + rapidfuzz address scores. Old caches may still contain the column; configs list it in drop_features.


def _prep(df):
    df = df.copy()
    df["name_ns"] = df.name_core.str.replace(" ", "", regex=False)
    df["skel_ns"] = df.name_skel.str.replace(" ", "", regex=False)
    return df


def _hv(kw):
    return HashingVectorizer(n_features=NF, alternate_sign=False, norm=None, binary=True, lowercase=False,
                             dtype=np.float32, **kw)


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=4, dtype=np.float32)


class PairFeaturizer:
    def __init__(self, A):
        self.A = _prep(A)
        self.idf, self.Am, self.cnt = {}, {}, {}
        n = len(self.A)
        for fam, (col, kw, use_idf) in FAMILIES.items():
            M = _hv(kw).transform(self.A[col].values).tocsr()
            if use_idf:
                df = np.bincount(M.indices, minlength=NF).astype(np.float32)
                idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
                self.idf[fam] = idf
                M.data *= idf[M.indices]
                M = normalize(M, copy=False)
            else:
                self.cnt[fam] = np.asarray(M.sum(1)).ravel().astype(np.float32)
            self.Am[fam] = M

    def _rmat(self, R, fam):
        col, kw, use_idf = FAMILIES[fam]
        M = _hv(kw).transform(R[col].values).tocsr()
        if use_idf:
            M.data *= self.idf[fam][M.indices]
            M = normalize(M, copy=False)
        return M

    def transform(self, R, ai, ri):
        """R: one chunk of S2/S3 records (positional); ai: S1 positions, ri: positions within the chunk."""
        A = self.A
        R = _prep(R)
        f = {}
        for fam in FAMILIES:
            Rm = self._rmat(R, fam)
            dot = np.asarray(self.Am[fam][ai].multiply(Rm[ri]).sum(1)).ravel().astype(np.float32)
            if fam in COS:
                f["cos_" + fam] = dot
            else:
                na = self.cnt[fam][ai]
                nr = np.asarray(Rm.sum(1)).ravel().astype(np.float32)[ri]
                f[fam + "_shared"] = dot
                f[fam + "_a"] = na
                f[fam + "_r"] = nr
                f[fam + "_jac"] = dot / np.maximum(na + nr - dot, 1)
            del Rm
        ac, rc = A.name_core.values[ai], R.name_core.values[ri]
        f["nm_ratio"] = _cp(ac, rc, fuzz.ratio)
        f["nm_tset"] = _cp(ac, rc, fuzz.token_set_ratio)
        f["nm_tsort"] = _cp(ac, rc, fuzz.token_sort_ratio)
        f["nm_partial"] = _cp(ac, rc, fuzz.partial_ratio)
        f["nm_jw"] = _cp(ac, rc, distance.JaroWinkler.normalized_similarity)
        an, rn = A.name_ns.values[ai], R.name_ns.values[ri]
        f["nm_ns_ratio"] = _cp(an, rn, fuzz.ratio)
        f["nm_ns_partial"] = _cp(an, rn, fuzz.partial_ratio)
        ask, rsk = A.name_skel.values[ai], R.name_skel.values[ri]
        f["sk_ratio"] = _cp(ask, rsk, fuzz.ratio)
        f["sk_tset"] = _cp(ask, rsk, fuzz.token_set_ratio)
        ral = R.alias.values[ri]
        has_alias = ral != ""
        f["alias_tset"] = np.where(has_alias, _cp(ac, ral, fuzz.token_set_ratio), -1).astype(np.float32)
        aa, ra = A.addr_norm.values[ai], R.addr_norm.values[ri]
        f["ad_tset"] = _cp(aa, ra, fuzz.token_set_ratio)
        f["ad_tsort"] = _cp(aa, ra, fuzz.token_sort_ratio)
        f["ad_ptset"] = _cp(aa, ra, fuzz.partial_token_set_ratio)
        f["ad_ratio"] = _cp(aa, ra, fuzz.ratio)
        fa = A.addr_nums.str.split(" ", n=1).str[0].values[ai]
        fr = R.addr_nums.str.split(" ", n=1).str[0].values[ri]
        f["num1_eq"] = ((fa == fr) & (fa != "")).astype(np.float32)
        f["num1_missing"] = ((fa == "") | (fr == "")).astype(np.float32)
        la = A.name_core.str.len().values[ai].astype(np.float32)
        lr = R.name_core.str.len().values[ri].astype(np.float32)
        f["nm_len_a"], f["nm_len_r"] = la, lr
        f["nm_len_ratio"] = np.minimum(la, lr) / np.maximum(np.maximum(la, lr), 1)
        f["ntok_a"] = (A.name_core.str.count(" ").values[ai] + 1).astype(np.float32)
        f["ntok_r"] = (R.name_core.str.count(" ").values[ri] + 1).astype(np.float32)
        f["addr_len_a"] = A.addr_norm.str.len().values[ai].astype(np.float32)
        f["addr_len_r"] = R.addr_norm.str.len().values[ri].astype(np.float32)
        f["r_addr_empty"] = (ra == "").astype(np.float32)
        f["r_is_domain"] = R.is_domain.values[ri].astype(np.float32)
        f["r_name_nonlatin"] = R.name_nonlatin.values[ri].astype(np.float32)
        f["r_addr_nonlatin"] = R.addr_nonlatin.values[ri].astype(np.float32)
        f["a_name_nonlatin"] = A.name_nonlatin.values[ai].astype(np.float32)
        f["r_has_alias"] = has_alias.astype(np.float32)
        f["r_src3"] = (R.src.values[ri] == 3).astype(np.float32)
        f["name_exact"] = (ac == rc).astype(np.float32)
        return pd.DataFrame(f)


def context_features(df, score_col, prefix):
    """Rank/competition features of a pair score within the S2/S3 record ('r') and within the S1 entity ('a')."""
    s = df[score_col]
    g_r = s.groupby(df.r)
    g_a = s.groupby(df.a)
    out = {}
    rmax = g_r.transform("max")
    out[prefix + "_r_rank"] = g_r.rank(ascending=False, method="min").astype(np.float32)
    out[prefix + "_a_rank"] = g_a.rank(ascending=False, method="min").astype(np.float32)
    # best score among the *other* S1 entities competing for this record
    srt = df[["r"]].assign(_s=s.values).sort_values(["r", "_s"], ascending=[True, False])
    second = srt.groupby("r")._s.nth(1)
    second = pd.Series(second.values, index=srt.loc[second.index, "r"].values)
    sec = df.r.map(second).fillna(-1).values
    other = np.where(s.values >= rmax.values, sec, rmax.values)
    out[prefix + "_r_other"] = other.astype(np.float32)
    out[prefix + "_r_margin"] = (s.values - other).astype(np.float32)
    out[prefix + "_a_gap"] = (s - g_a.transform("max")).astype(np.float32)
    out[prefix + "_r_n"] = g_r.transform("size").astype(np.float32)
    out[prefix + "_a_n"] = g_a.transform("size").astype(np.float32)
    return pd.DataFrame(out, index=df.index)
