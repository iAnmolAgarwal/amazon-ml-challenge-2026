"""Side-car name-operator features (feat6): label-free decoy-marker scores for the words a pool name adds, drops or swaps.

usage: python src/er_feat6.py --split train|test [--country us|india|france] [--tag ""] [--pairs PATH]
       python src/er_feat6.py --split train|test --merge [--tag ""]

Names are compared as conservative token sets of name_lat (accents stripped, lowercase, split on non-alphanumerics,
legal words kept): A = S1-only tokens, B = pool-only tokens.
Marker scores come from the candidate pairs themselves, without labels: among pairs with |A|, |B| <= 2 and both house
numbers parsed, pairs whose pool house number is the S1's shifted by a typical generator shift d in
{1,2,3,4,5,7,9,11,13,21} (nd) are contrasted with pairs whose house numbers are equal (n0):
score = (nd + 1) / (nd + n0 + 10), per country and token (added tokens, dropped tokens, and one-for-one swaps).
Train uses tables from train pairs; test uses the train tables for countries seen in train and tables from its own
pairs for new countries. Features: nA, nB, dm_add (max score over added tokens), dm_drop (over dropped tokens),
dm_swap (score of the swap when |A| = |B| = 1). Writes cache/feat/{split}_feat6{tag}.parquet (per-country chunks in
cache/feat/f6parts, concatenated by --merge); tables in cache/feat/f6tables/.
"""
import argparse
import time
from pathlib import Path

import polars as pl

from common import load_config

SHIFTS = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
COLS = ["nA", "nB", "dm_add", "dm_drop", "dm_swap"]
CHUNK = 5_000_000


def tokens(c):
    return (pl.col(c).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_lowercase()
            .str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars().str.split(" ").list.unique().list.sort()
            .list.eval(pl.element().filter(pl.element() != "")))


def records(cache, split, side, ctry, need):
    idx = "s1_idx" if side == "s1" else "pool_idx"
    df = pl.scan_parquet(cache / "norm" / f"{split}_{side}.parquet").select("country_n", "house", "name_lat").with_row_index(idx)
    df = df.filter(pl.col("country_n") == ctry).with_columns(pl.col(idx).cast(pl.UInt32)).join(need.lazy(), on=idx, how="semi")
    return df.select(idx, "house", t=tokens("name_lat")).collect()


def pair_ops(k, s1, pool):
    x = k.join(s1, on="s1_idx", how="left", maintain_order="left").join(pool.rename({"house": "house_p", "t": "tp"}), on="pool_idx", how="left", maintain_order="left")
    hd = pl.col("house_p").cast(pl.Int64, strict=False) - pl.col("house").cast(pl.Int64, strict=False)
    x = x.with_columns(hd=hd, A=pl.col("t").list.set_difference("tp"), B=pl.col("tp").list.set_difference("t")).drop("t", "tp", "house", "house_p")
    return x.with_columns(nA=pl.col("A").list.len().cast(pl.Float32), nB=pl.col("B").list.len().cast(pl.Float32))


def counts(x):
    lab = x.filter((pl.col("nA") <= 2) & (pl.col("nB") <= 2) & pl.col("hd").is_not_null()).with_columns(dz=pl.col("hd").is_in(SHIFTS), z0=pl.col("hd") == 0).filter(pl.col("dz") | pl.col("z0"))
    out = []
    for kind, side in (("add", "B"), ("drop", "A")):
        out.append(lab.select("dz", "z0", pl.col(side).alias("tok")).explode("tok").drop_nulls("tok").group_by("tok").agg(pl.col("dz").sum().alias("nd"), pl.col("z0").sum().alias("n0")).with_columns(kind=pl.lit(kind)))
    sw = lab.filter((pl.col("nA") == 1) & (pl.col("nB") == 1)).select("dz", "z0", tok=pl.col("A").list.first() + ">" + pl.col("B").list.first())
    out.append(sw.group_by("tok").agg(pl.col("dz").sum().alias("nd"), pl.col("z0").sum().alias("n0")).with_columns(kind=pl.lit("swap")))
    return pl.concat(out).with_columns(pl.col("nd").cast(pl.Int64), pl.col("n0").cast(pl.Int64))


def features(x, table):
    sc = table.with_columns(sc=((pl.col("nd") + 1) / (pl.col("nd") + pl.col("n0") + 10)).cast(pl.Float32)).select("kind", "tok", "sc")
    x = x.with_row_index("r")
    res = x.select("r", "s1_idx", "pool_idx", "nA", "nB")
    for kind, side, name in (("add", "B", "dm_add"), ("drop", "A", "dm_drop")):
        e = x.select("r", pl.col(side).alias("tok")).explode("tok").drop_nulls("tok").join(sc.filter(pl.col("kind") == kind).drop("kind"), on="tok", how="left")
        res = res.join(e.group_by("r").agg(pl.col("sc").max().alias(name)), on="r", how="left", maintain_order="left")
    key = x.select("r", tok=pl.when((pl.col("nA") == 1) & (pl.col("nB") == 1)).then(pl.col("A").list.first() + ">" + pl.col("B").list.first()))
    key = key.join(sc.filter(pl.col("kind") == "swap").drop("kind"), on="tok", how="left").select("r", pl.col("sc").alias("dm_swap"))
    res = res.join(key, on="r", how="left", maintain_order="left")
    return res.select(["s1_idx", "pool_idx"] + COLS).with_columns([pl.col(c).cast(pl.Float32) for c in COLS])


def build_country(cache, split, tag, ctry, pairs_path=None):
    t = time.time()
    src = Path(pairs_path) if pairs_path else cache / "cand" / f"{split}_cand{tag}.parquet"
    k_all = pl.scan_parquet(src).filter(pl.col("country_n") == ctry).select("s1_idx", "pool_idx").collect().sort("s1_idx", "pool_idx")
    s1 = records(cache, split, "s1", ctry, k_all.select("s1_idx").unique())
    pool = records(cache, split, "pool", ctry, k_all.select("pool_idx").unique())
    print(f"{split} {ctry}: {len(k_all):,} pairs, {len(s1):,} S1, {len(pool):,} pool, {time.time() - t:.0f}s", flush=True)
    tdir = cache / "feat" / "f6tables"
    tdir.mkdir(parents=True, exist_ok=True)
    tpath = tdir / f"train{tag}_{ctry}.parquet"
    own = split == "train" or not tpath.exists()
    if own:
        parts = [counts(pair_ops(k_all.slice(s, CHUNK), s1, pool)) for s in range(0, len(k_all), CHUNK)]
        table = pl.concat(parts).group_by("kind", "tok").agg(pl.col("nd").sum(), pl.col("n0").sum())
        table.write_parquet(tdir / f"{split}{tag}_{ctry}.parquet")
        top = table.filter((pl.col("kind") == "add") & (pl.col("nd") + pl.col("n0") >= 200)).with_columns(sc=(pl.col("nd") + 1) / (pl.col("nd") + pl.col("n0") + 10)).sort("sc", descending=True).head(15)
        print(f"  own {ctry} table: {len(table):,} rows; top added markers: " + ", ".join(f"{a} {b:.2f}" for a, b in top.select("tok", "sc").iter_rows()), flush=True)
    else:
        table = pl.read_parquet(tpath)
        print(f"  using the train {ctry} table ({len(table):,} rows)", flush=True)
    out = cache / "feat" / "f6parts"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(f"{split}{tag}_{ctry}_*.parquet"):
        old.unlink()
    for i, s in enumerate(range(0, len(k_all), CHUNK)):
        features(pair_ops(k_all.slice(s, CHUNK), s1, pool), table).write_parquet(out / f"{split}{tag}_{ctry}_{i:03d}.parquet")
        print(f"  chunk {i}: {min(CHUNK, len(k_all) - s):,} pairs, {time.time() - t:.0f}s", flush=True)


def merge(cache, split, tag):
    t = time.time()
    parts = sorted((cache / "feat" / "f6parts").glob(f"{split}{tag}_*.parquet"))
    n_cand = pl.scan_parquet(cache / "cand" / f"{split}_cand{tag}.parquet").select(pl.len()).collect().item()
    path = cache / "feat" / f"{split}_feat6{tag}.parquet"
    pl.scan_parquet(parts).sink_parquet(path)
    n = pl.scan_parquet(path).select(pl.len()).collect().item()
    assert n == n_cand, f"{n} feature rows vs {n_cand} candidates"
    print(f"wrote {path.name}: {n:,} rows from {len(parts)} parts in {time.time() - t:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--country")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--pairs", help="build only these candidate pairs (s1_idx, pool_idx, country_n), e.g. a research sample")
    a = ap.parse_args()
    cache = Path(load_config()["cache_dir"])
    if a.merge:
        merge(cache, a.split, a.tag)
        return
    countries = [a.country] if a.country else pl.scan_parquet(cache / "cand" / f"{a.split}_cand{a.tag}.parquet").select(pl.col("country_n").unique()).collect()["country_n"].sort().to_list()
    if a.split == "test":
        countries = sorted(countries, key=lambda c: 0 if (cache / "feat" / "f6tables" / f"train{a.tag}_{c}.parquet").exists() else 1)
    for c in countries:
        build_country(cache, a.split, a.tag, c, a.pairs)
    if not a.country and not a.pairs:
        merge(cache, a.split, a.tag)


if __name__ == "__main__":
    main()
