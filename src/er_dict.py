"""Learn native-script -> Latin token maps from training gold pairs (names and addresses).

usage: python src/er_dict.py [--min-owners 5]

Tokens are runs of Indic-block characters (U+0900-U+0DFF, dandas excluded; zero-width joiners removed) or runs of Latin letters/digits,
on lowercased text; er_norm.latinize substitutes exactly these runs.
- names: for gold pairs whose token counts match, tokens are aligned by position (the generator transliterates
  word by word); tokens never seen in an aligned pair fall back to co-occurrence (Dice).
- addresses: co-occurrence (Dice) only; the native part is the state name.
Support is counted in distinct S1 owners; a token is kept with >= min_owners owners and a target share >= 0.5.
Latin spelling variants that compete for the same native tokens (share >= 0.2 each, on >= 3 native tokens)
form classes mapped to their most frequent member ("canon"; applied to India names on both sides).
Writes cache/dict/indic.json {"name": {...}, "addr": {...}, "canon": {...}}.
"""
import argparse
import json
import time
from pathlib import Path

import polars as pl

from common import load_config

IND = r"[\x{0900}-\x{0963}\x{0966}-\x{0DFF}]"
TOK = r"[\x{0900}-\x{0963}\x{0966}-\x{0DFF}]+|[0-9a-z\x{00DF}-\x{024F}]+"


def toks(c):
    return pl.col(c).fill_null("").str.to_lowercase().str.replace_all(r"[\x{200C}\x{200D}]", "").str.extract_all(TOK)


def pairs(cache, field):
    """(s1_idx, S1 tokens, pool tokens) for gold pairs whose pool field has native-script text."""
    pool = pl.concat([pl.scan_parquet(cache / "raw" / f"train_source{i}.parquet").select("entity_id", field) for i in (2, 3)])
    s1 = pl.scan_parquet(cache / "raw" / "train_source1.parquet").select("entity_id", field).with_row_index("s1_idx")
    g = pl.scan_parquet(cache / "raw" / "train_ground_truth.parquet").select(
        pl.col("source1_entity_id").alias("s1_id"), pl.col("matched_entity_ids").str.split(",").alias("pool_id")).explode("pool_id")
    p = pool.filter(pl.col(field).str.contains(IND)).select(pl.col("entity_id").alias("pool_id"), toks(field).alias("pt"))
    return g.join(p, on="pool_id").join(s1.select(pl.col("entity_id").alias("s1_id"), "s1_idx", toks(field).alias("lt")), on="s1_id").select("s1_idx", "lt", "pt").collect()


def best_target(e, min_owners):
    """e: s1_idx, pt, lt evidence rows -> best Latin target per native token with owner support and share."""
    e = e.unique(["s1_idx", "pt", "lt"])
    sc = e.group_by("pt", "lt").agg(n=pl.len()).join(e.unique(["s1_idx", "pt"]).group_by("pt").agg(nt=pl.len()), on="pt").with_columns(share=pl.col("n") / pl.col("nt"))
    return sc, sc.sort(["pt", "n", "lt"], descending=[False, True, False]).unique("pt", keep="first", maintain_order=True).filter((pl.col("nt") >= min_owners) & (pl.col("share") >= 0.5))


def dice(df, min_owners):
    """Co-occurrence of native tokens with Latin tokens of the same S1, ranked by Dice over owners."""
    pt = df.select("s1_idx", "pt").explode("pt").filter(pl.col("pt").str.contains(IND)).unique()
    lt = df.select("s1_idx", "lt").explode("lt").drop_nulls().filter(~pl.col("lt").str.contains(IND)).unique()
    co = pt.join(lt, on="s1_idx").group_by("pt", "lt").agg(n=pl.len())
    co = co.join(pt.group_by("pt").agg(nt=pl.len()), on="pt").join(lt.join(pt.select("s1_idx").unique(), on="s1_idx").group_by("lt").agg(nl=pl.len()), on="lt")
    co = co.with_columns(dice=2 * pl.col("n") / (pl.col("nt") + pl.col("nl")), share=pl.col("n") / pl.col("nt"))
    return co.sort(["pt", "dice", "lt"], descending=[False, True, False]).unique("pt", keep="first", maintain_order=True).filter((pl.col("nt") >= min_owners) & (pl.col("share") >= 0.5))


def canon_classes(sc, lt_freq):
    """Latin variants competing for >= 3 native tokens -> most frequent member."""
    comp = sc.filter(pl.col("share") >= 0.2).group_by("pt").agg(pl.col("lt").sort()).filter(pl.col("lt").list.len() == 2)
    cls = comp.group_by("lt").len().filter(pl.col("len") >= 3)["lt"].to_list()
    out = {}
    for members in cls:
        head = max(members, key=lambda t: (lt_freq.get(t, 0), t))
        out.update({m: head for m in members if m != head})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-owners", type=int, default=5)
    a = ap.parse_args()
    cfg = load_config()
    cache = Path(cfg["cache_dir"])
    t = time.time()
    df = pairs(cache, "business_name")
    same = df.filter(pl.col("lt").list.len() == pl.col("pt").list.len()).explode("lt", "pt").filter(pl.col("pt").str.contains(IND))
    sc, pos = best_target(same.select("s1_idx", "pt", "lt"), a.min_owners)
    fb = dice(df, a.min_owners).join(pos.select("pt"), on="pt", how="anti")
    lt_freq = dict(df.select(pl.col("lt").explode()).group_by("lt").len().iter_rows())
    canon = canon_classes(sc, lt_freq)
    name = {p: canon.get(l, l) for p, l in zip(pos["pt"].to_list() + fb["pt"].to_list(), pos["lt"].to_list() + fb["lt"].to_list())}
    print(f"name: {len(df):,} gold pairs with native script; {len(pos):,} tokens aligned by position + {len(fb):,} by co-occurrence -> {len(set(name.values())):,} Latin tokens; canon {canon}", flush=True)
    da = pairs(cache, "business_address")
    ad = dice(da, a.min_owners)
    addr = dict(zip(ad["pt"].to_list(), ad["lt"].to_list()))
    print(f"addr: {len(da):,} gold pairs with native script; {len(addr):,} tokens: {addr}", flush=True)
    out = cache / "dict"
    out.mkdir(exist_ok=True)
    (out / "indic.json").write_text(json.dumps({"name": dict(sorted(name.items())), "addr": dict(sorted(addr.items())), "canon": dict(sorted(canon.items()))}, ensure_ascii=False, indent=0))
    print(f"wrote {out / 'indic.json'} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
