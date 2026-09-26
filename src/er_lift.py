"""Country-lifted decoy-marker scores (feat6l) and French house-number repair (feat5p).

usage: python src/er_lift.py        (after er_feat5.py and er_feat6.py for both splits)

1. feat6l: dm_add, dm_drop, dm_swap of feat6 mapped to logit(score) - log(sum nd / sum n0) of the country's marker table
   (train tables for US/India, the test-built table for France), so every country's scores sit on the same scale.
   Writes cache/feat/{train,test}_feat6l.parquet.
2. feat5p: test_feat5 with the house columns of France pairs recomputed after a French fallback house parser (padded or
   parenthesised numbers, extra street types), used only where the standard parser found no house number. Train has no
   France, so train_feat5p links to train_feat5. Writes cache/feat/test_feat5p.parquet.
"""
import os
import re
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist

import er_norm as n
from common import load_config

K = ["s1_idx", "pool_idx"]
DM = ["dm_add", "dm_drop", "dm_swap"]
SHIFTS = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
HS = ["h_suffix", "h_diffl", "h_near", "h_near_st", "h_lev", "h_swap", "h_in_p", "h_in_s"]
EXTRA = "(?:" + n.FR_STREET[3:-1] + "|ch|crs|cr|cour|q|blvd|bvd|parvis|parv|esplanade|espl|lotissement|chaussee|mail|venelle|ruelle|sente|digue|corniche|grande rue|petite avenue|rpt|rp|rond[- ]point)"
HR = re.compile(r"\b0*(\d{1,4})\s*(?:bis|ter|quater|[a-z])?[\s,().-]*" + EXTRA + r"\b")


def bases(path):
    t = pl.read_parquet(path).group_by("kind").agg(pl.col("nd").sum(), pl.col("n0").sum())
    return {"dm_" + k: float(np.log(a / b)) for k, a, b in t.select("kind", "nd", "n0").iter_rows()}


def lift(C, t0):
    tb = C / "feat" / "f6tables"
    B = {"us": bases(tb / "train_us.parquet"), "india": bases(tb / "train_india.parquet"), "france": bases(tb / "test_france.parquet")}
    print("base logits", B, flush=True)
    for split in ("train", "test"):
        ctry = pl.read_parquet(C / "norm" / f"{split}_s1.parquet", columns=["country_n"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
        f6 = pl.read_parquet(C / "feat" / f"{split}_feat6.parquet").join(ctry, on="s1_idx", how="left", maintain_order="left")
        out = f6.with_columns([(pl.col(c).clip(1e-6, 1 - 1e-6).log() - (1 - pl.col(c).clip(1e-6, 1 - 1e-6)).log() - pl.col("country_n").replace_strict({k: v[c] for k, v in B.items()}, default=None, return_dtype=pl.Float64)).cast(pl.Float32).alias(c) for c in DM]).drop("country_n")
        out.write_parquet(C / "feat" / f"{split}_feat6l.parquet")
        print(f"{split}_feat6l: {len(out):,} rows, {time.time() - t0:.0f}s", flush=True)
        del f6, out


def fixed(h, a):
    if h:
        return h
    q = HR.search(n.SUBUNIT.sub(" ", n.FR_POSTREF.sub(" ", n.fold(a or ""))))
    return (q.group(1).lstrip("0") or "0") if q else ""


def attrs(C, which):
    ix = "s1_idx" if which == "s1" else "pool_idx"
    nm = pl.read_parquet(C / "norm" / f"test_{which}.parquet", columns=["entity_id", "country_n", "house", "street1", "nums"])
    raw = pl.read_parquet(C / "raw" / "test_source1.parquet") if which == "s1" else pl.concat([pl.read_parquet(C / "raw" / "test_source2.parquet"), pl.read_parquet(C / "raw" / "test_source3.parquet")])
    assert (raw["entity_id"] == nm["entity_id"]).all()
    nm = nm.with_columns(raw["business_address"].alias("addr")).with_row_index(ix).with_columns(pl.col(ix).cast(pl.UInt32)).filter(pl.col("country_n") == "france")
    hf = [fixed(h or "", a) for h, a in zip(nm["house"].to_list(), nm["addr"].to_list())]
    return nm.select(ix, pl.col("house").fill_null(""), pl.col("street1").fill_null(""), pl.col("nums").fill_null("")).with_columns(hf=pl.Series(hf))


def house_feats(k, sa, pa, hcol):
    a = sa.select("s1_idx", pl.col(hcol).alias("house"), "street1", "nums")
    b = pa.select("pool_idx", pl.col(hcol).alias("house_p"), pl.col("street1").alias("street1_p"), pl.col("nums").alias("nums_p"))
    z = k.select(K).join(a, on="s1_idx", how="left", maintain_order="left").join(b, on="pool_idx", how="left", maintain_order="left")
    h1, h2 = pl.col("house"), pl.col("house_p")
    both = (h1 != "") & (h2 != "")
    neq = both & (h1 != h2)
    n1, n2 = h1.cast(pl.Int64, strict=False), h2.cast(pl.Int64, strict=False)
    diff, sd = (n1 - n2).abs(), n2 - n1
    lev = cpdist(z["house"].to_list(), z["house_p"].to_list(), scorer=Levenshtein.distance, workers=8)
    pad = lambda e: pl.concat_str([pl.lit(" "), e, pl.lit(" ")])
    z = z.with_columns(h_lev_raw=pl.Series(lev.astype(np.float32)), hs=pl.Series(["".join(sorted(s)) for s in z["house"].to_list()]), hs_p=pl.Series(["".join(sorted(s)) for s in z["house_p"].to_list()]))
    out = z.select(*K,
        h_suffix=(neq & (h1.str.ends_with(h2) | h2.str.ends_with(h1))),
        h_diffl=pl.when(both & diff.is_not_null()).then((diff.cast(pl.Float64) + 1).log()).otherwise(-1.0),
        h_near=neq & (diff <= 10).fill_null(False),
        h_near_st=neq & (diff <= 10).fill_null(False) & (pl.col("street1") != "") & (pl.col("street1") == pl.col("street1_p")),
        h_lev=pl.when(both).then(pl.col("h_lev_raw")).otherwise(-1.0),
        h_swap=neq & (pl.col("hs") == pl.col("hs_p")),
        h_in_p=pl.when(h1 != "").then(pad(pl.col("nums_p")).str.contains(pad(h1), literal=True)).otherwise(None),
        h_in_s=pl.when(h2 != "").then(pad(pl.col("nums")).str.contains(pad(h2), literal=True)).otherwise(None),
        hd=pl.when(both).then(sd.clip(-100000, 100000)).otherwise(None).cast(pl.Float32),
        hd_shift=pl.when(both & sd.is_not_null()).then(sd.is_in(SHIFTS)).otherwise(None))
    return out.with_columns(*[pl.col(c).cast(pl.Float32).fill_null(-1.0) for c in HS + ["hd_shift"]])


def parser(C, t0):
    src, dst = C / "feat" / "train_feat5.parquet", C / "feat" / "train_feat5p.parquet"
    if not dst.exists():
        os.symlink(src, dst)
    sa, pa = attrs(C, "s1"), attrs(C, "pool")
    print(f"france: S1 houses recovered {(sa['house'] != sa['hf']).sum():,}/{len(sa):,}; pool {(pa['house'] != pa['hf']).sum():,}/{len(pa):,}; {time.time() - t0:.0f}s", flush=True)
    f5 = pl.read_parquet(C / "feat" / "test_feat5.parquet")
    frk = f5.select(K).join(sa.select("s1_idx"), on="s1_idx", how="semi")
    chk = house_feats(frk.head(300000), sa, pa, "house").join(f5.select(K + HS + ["hd", "hd_shift"]), on=K, suffix="_o")
    par = {c: float(np.nanmax(np.abs(chk[c].to_numpy() - chk[c + "_o"].to_numpy()))) for c in HS + ["hd_shift"]}
    print("parity of the unchanged parser vs test_feat5 (300k France pairs):", par, flush=True)
    assert max(par.values()) == 0.0
    fx = house_feats(frk, sa, pa, "hf")
    cols = HS + ["hd", "hd_shift"]
    chg = fx.join(f5.select(K + ["h_diffl"]), on=K, suffix="_o").filter(pl.col("h_diffl") != pl.col("h_diffl_o")).height
    f5p = f5.join(fx, on=K, how="left", suffix="_fx").with_columns([pl.when(pl.col("h_diffl_fx").is_not_null()).then(pl.col(c + "_fx")).otherwise(pl.col(c)).alias(c) for c in cols]).drop([c + "_fx" for c in cols])
    assert f5p.height == f5.height
    f5p.write_parquet(C / "feat" / "test_feat5p.parquet")
    print(f"test_feat5p: {f5p.height:,} rows; France pairs {frk.height:,}, with changed house features {chg:,}; {time.time() - t0:.0f}s")


def main():
    C = Path(load_config()["cache_dir"])
    t0 = time.time()
    lift(C, t0)
    parser(C, t0)


if __name__ == "__main__":
    main()
