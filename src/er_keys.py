"""Extra candidate source: exact keys aimed at the name operators the other routes miss (K9).

usage: python src/er_keys.py --split train|test [--queries PATH] [--tag ""] [--cap 50]

Per country, from cache/norm/{split}_{s1,pool}.parquet (nulls read as ""):
  p6n  name_join[:6] | n  for every number n in nums (name_join >= 6 chars)
  njn  name_join | n      for every number n in nums (name_join >= 3 chars)
  p8   name_join[:8]      (>= 8 chars)
  sk   name_skel          (>= 3 chars)
  fl   name_fl
  skh  name_skel | house
  tt   any two distinct name_core tokens
  th   name token (>= 2 chars) | house   (India only)
  tn   name token | any number           (US only)
A key value is kept only if at most --cap S1 and at most --cap pool records of the country carry it (counted over the
whole split); pairs = S1 x pool records sharing a kept value, unioned over keys. With --queries (parquet with
s1_idx) only those S1 are paired. Label-free. Writes cache/cand/{split}_ret_keys{tag}.parquet (s1_idx, pool_idx,
similarity = 0), read by er_newpairs.py union as an extra view.
"""
import argparse
import time
from pathlib import Path

import polars as pl

from common import load_config

KEYS = ["s1_idx", "pool_idx"]


def key_values(df, country):
    j = pl.col("name_join")
    nl = pl.col("nums").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    tok = pl.col("name_core").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique().list.sort()
    parts = []
    idx = df.columns[0]
    base = df.with_columns(nl=nl, tok=tok)
    p6 = base.filter(j.str.len_chars() >= 6).select(idx, j.str.slice(0, 6).alias("p"), "nl").explode("nl").drop_nulls("nl")
    parts.append(p6.select(idx, ("p6n:" + pl.col("p") + "|" + pl.col("nl")).alias("k")))
    nj = base.filter(j.str.len_chars() >= 3).select(idx, "name_join", "nl").explode("nl").drop_nulls("nl")
    parts.append(nj.select(idx, ("njn:" + pl.col("name_join") + "|" + pl.col("nl")).alias("k")))
    parts.append(base.filter(j.str.len_chars() >= 8).select(idx, ("p8:" + j.str.slice(0, 8)).alias("k")))
    parts.append(base.filter(pl.col("name_skel").str.len_chars() >= 3).select(idx, ("sk:" + pl.col("name_skel")).alias("k")))
    parts.append(base.filter(pl.col("name_fl") != "").select(idx, ("fl:" + pl.col("name_fl")).alias("k")))
    parts.append(base.filter((pl.col("name_skel").str.len_chars() >= 3) & (pl.col("house") != "")).select(idx, ("skh:" + pl.col("name_skel") + "|" + pl.col("house")).alias("k")))
    tt = base.filter(pl.col("tok").list.len().is_between(2, 12)).select(idx, "tok").with_columns(t2=pl.col("tok")).explode("tok").explode("t2").filter(pl.col("tok") < pl.col("t2"))
    parts.append(tt.select(idx, ("tt:" + pl.col("tok") + "|" + pl.col("t2")).alias("k")))
    tk = base.select(idx, "tok", "house", "nl").explode("tok").drop_nulls("tok").filter(pl.col("tok").str.len_chars() >= 2)
    if country == "india":
        parts.append(tk.filter(pl.col("house") != "").select(idx, ("th:" + pl.col("tok") + "|" + pl.col("house")).alias("k")))
    if country == "us":
        tn = tk.select(idx, "tok", "nl").explode("nl").drop_nulls("nl")
        parts.append(tn.select(idx, ("tn:" + pl.col("tok") + "|" + pl.col("nl")).alias("k")))
    return pl.concat(parts).unique()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--queries")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cap", type=int, default=50)
    a = ap.parse_args()
    cache = Path(load_config()["cache_dir"])
    t = time.time()
    cols = ["country_n", "name_core", "name_join", "name_skel", "name_fl", "house", "nums"]
    s1 = pl.read_parquet(cache / "norm" / f"{a.split}_s1.parquet", columns=cols).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col(cols[1:]).fill_null(""))
    pool = pl.read_parquet(cache / "norm" / f"{a.split}_pool.parquet", columns=cols).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32), pl.col(cols[1:]).fill_null(""))
    q = pl.read_parquet(a.queries, columns=["s1_idx"]).with_columns(pl.col("s1_idx").cast(pl.UInt32)).unique() if a.queries else None
    out = []
    for c in s1["country_n"].unique().to_list():
        ks = key_values(s1.filter(pl.col("country_n") == c).drop("country_n"), c)
        kp = key_values(pool.filter(pl.col("country_n") == c).drop("country_n"), c)
        ns, np_ = ks.group_by("k").len(), kp.group_by("k").len()
        keep = ns.filter(pl.col("len") <= a.cap).select("k").join(np_.filter(pl.col("len") <= a.cap).select("k"), on="k")
        ks, kp = ks.join(keep, on="k", how="semi"), kp.join(keep, on="k", how="semi")
        if q is not None:
            ks = ks.join(q, on="s1_idx", how="semi")
        pairs = ks.join(kp, on="k").select(KEYS).unique()
        print(f"{c}: {keep.height:,} key values kept, {pairs.height:,} pairs, {time.time() - t:.0f}s", flush=True)
        out.append(pairs)
    res = pl.concat(out).with_columns(similarity=pl.lit(0.0, pl.Float32)).sort(KEYS)
    path = cache / "cand" / f"{a.split}_ret_keys{a.tag}.parquet"
    res.write_parquet(path)
    print(f"wrote {path.name}: {res.height:,} pairs, {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
