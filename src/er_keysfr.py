"""Extra candidate source for countries whose exact-key set lacks the name-token keys: name token | house (th) and
name token | any number (tn).

usage: python src/er_keysfr.py --split train|test [--keys france:th,tn] [--queries PATH] [--tag ""] [--cap 50]

--keys lists, per country, which of th / tn / an to build (default France only, th and tn). Token = name_core token of
at least 2 characters; address word = alphabetic addr_norm token of at least 4 characters; house, nums and addr_norm
from cache/norm (nulls read as ""). A key value is kept only if at most --cap S1 and at
most --cap pool records of the country carry it (counted over the whole split, as in er_keys.py); pairs = S1 x pool
records sharing a kept value. With --queries (parquet with s1_idx) only those S1 are paired. Label-free. Writes
cache/cand/{split}_ret_keysfr{tag}.parquet (s1_idx, pool_idx, similarity = 0), read by er_newpairs.py union as an
extra view.
"""
import argparse
import time
from pathlib import Path

import polars as pl

from common import load_config

KEYS = ["s1_idx", "pool_idx"]


def key_values(df, kinds):
    idx = df.columns[0]
    tok = pl.col("name_core").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)).list.unique()
    nl = pl.col("nums").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    tk = df.select(idx, "house", tok=tok, nl=nl).explode("tok").drop_nulls("tok")
    parts = []
    if "an" in kinds:
        aw = pl.col("addr_norm").str.split(" ").list.eval(pl.element().filter(pl.element().str.contains(r"^[a-z]{4,}$"))).list.unique()
        an = df.select(idx, aw=aw, nl=nl).explode("aw").drop_nulls("aw").explode("nl").drop_nulls("nl")
        parts.append(an.select(idx, ("an:" + pl.col("aw") + "|" + pl.col("nl")).alias("k")))
    if "th" in kinds:
        parts.append(tk.filter(pl.col("house") != "").select(idx, ("th:" + pl.col("tok") + "|" + pl.col("house")).alias("k")))
    if "tn" in kinds:
        parts.append(tk.explode("nl").drop_nulls("nl").select(idx, ("tn:" + pl.col("tok") + "|" + pl.col("nl")).alias("k")))
    return pl.concat(parts).unique()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--keys", default="france:th,tn")
    ap.add_argument("--queries")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cap", type=int, default=50)
    a = ap.parse_args()
    cache = Path(load_config()["cache_dir"])
    t = time.time()
    spec = {c: k.split(",") for c, k in (x.split(":") for x in a.keys.split(";"))}
    cols = ["country_n", "name_core", "house", "nums", "addr_norm"]
    s1 = pl.read_parquet(cache / "norm" / f"{a.split}_s1.parquet", columns=cols).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col(cols[1:]).fill_null(""))
    pool = pl.read_parquet(cache / "norm" / f"{a.split}_pool.parquet", columns=cols).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32), pl.col(cols[1:]).fill_null(""))
    q = pl.read_parquet(a.queries, columns=["s1_idx"]).with_columns(pl.col("s1_idx").cast(pl.UInt32)).unique() if a.queries else None
    out = []
    for c, kinds in spec.items():
        ks = key_values(s1.filter(pl.col("country_n") == c).drop("country_n"), kinds)
        kp = key_values(pool.filter(pl.col("country_n") == c).drop("country_n"), kinds)
        ns, np_ = ks.group_by("k").len(), kp.group_by("k").len()
        keep = ns.filter(pl.col("len") <= a.cap).select("k").join(np_.filter(pl.col("len") <= a.cap).select("k"), on="k")
        ks, kp = ks.join(keep, on="k", how="semi"), kp.join(keep, on="k", how="semi")
        if q is not None:
            ks = ks.join(q, on="s1_idx", how="semi")
        pairs = ks.join(kp, on="k").select(KEYS).unique()
        print(f"{c} {kinds}: {keep.height:,} key values kept, {pairs.height:,} pairs, {time.time() - t:.0f}s", flush=True)
        out.append(pairs)
    res = pl.concat(out).with_columns(similarity=pl.lit(0.0, pl.Float32)).sort(KEYS)
    path = cache / "cand" / f"{a.split}_ret_keysfr{a.tag}.parquet"
    res.write_parquet(path)
    print(f"wrote {path.name}: {res.height:,} pairs, {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
