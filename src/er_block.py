"""Candidate generation (blocking) for business entity resolution.

Per country: (1) word-view retrieval: TF-IDF over hashed name+address word 1-2 grams, exact top-K
cosine neighbours via sparse_dot_topn on 125k-row pool chunks merged with zip_sp_matmul_topn;
(2) exact keys joined in polars: house+street, space-free name, sorted space-free name,
name+house, initials+house (acronyms), and alternate ("formerly") name. Key values whose block is
larger than --cap on either side are skipped (never truncated by row order). The union is written
to cache/cand/{split}_cand.parquet with the word cosine for every pair and one flag per key.
With --eval (train only) prints pair recall and oracle macro F0.5 by country.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sparse_dot_topn import sp_matmul_topn, zip_sp_matmul_topn

from common import load_config

KEYS = ["k_hs", "k_nj", "k_ns", "k_njh", "k_ih", "k_alt"]


def vectorize(a_text, b_text):
    """Fit sublinear TF-IDF on hashed word 1-2 grams of pool+query text; return L2-normalized float32 CSR matrices."""
    hv = HashingVectorizer(n_features=2**23, ngram_range=(1, 2), token_pattern=r"(?u)\S+", lowercase=False, alternate_sign=False, norm=None, dtype=np.float32)
    ha, hb = hv.transform(a_text), hv.transform(b_text)
    tf = TfidfTransformer(sublinear_tf=True).fit(hb)
    return tf.transform(ha).astype(np.float32).tocsr(), tf.transform(hb).astype(np.float32).tocsr()


def topk(qa, pb, k, chunk=125_000, qchunk=200_000, threads=8):
    """Exact top-k cosine neighbours of each query row among pool rows, chunked on both sides."""
    pts = [pb[i : i + chunk].T.tocsr() for i in range(0, pb.shape[0], chunk)]
    rows, cols, vals = [], [], []
    for s in range(0, qa.shape[0], qchunk):
        q = qa[s : s + qchunk]
        c = zip_sp_matmul_topn(k, [sp_matmul_topn(q, p, top_n=k, n_threads=threads, sort=True) for p in pts]).tocoo()
        rows.append(c.row + s)
        cols.append(c.col)
        vals.append(c.data)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def key_pairs(a, b, cap):
    """Exact-key candidate pairs; key values with more than `cap` rows on either side are skipped."""
    specs = {
        "k_hs": (pl.concat_str(["house", "street1"], separator="|"), pl.concat_str(["house", "street1"], separator="|"), ["house", "street1"]),
        "k_nj": (pl.col("name_join"), pl.col("name_join"), ["name_join"]),
        "k_ns": (pl.col("name_sorted"), pl.col("name_sorted"), ["name_sorted"]),
        "k_njh": (pl.concat_str(["name_join", "house"], separator="|"), pl.concat_str(["name_join", "house"], separator="|"), ["name_join", "house"]),
        "k_ih": (pl.concat_str(["initials", "house"], separator="|"), pl.concat_str(["name_join", "house"], separator="|"), ["initials", "house"]),
        "k_alt": (pl.col("name_join"), pl.col("name_alt").str.replace_all(" ", ""), ["name_join"]),
    }
    out = []
    for k, (ea, eb, need) in specs.items():
        ka = a.filter(pl.all_horizontal([pl.col(c) != "" for c in need])).select(pl.col("s1_idx"), ea.alias("key"))
        bb = b.filter(pl.col("name_alt") != "") if k == "k_alt" else b
        kb = bb.select(pl.col("pool_idx"), eb.alias("key")).filter(pl.col("key").str.len_chars() >= 3)
        big_a = ka.group_by("key").len().filter(pl.col("len") > cap).select("key")
        big_b = kb.group_by("key").len().filter(pl.col("len") > cap).select("key")
        ka = ka.join(big_a, on="key", how="anti").join(big_b, on="key", how="anti")
        p = ka.join(kb, on="key", how="inner").select("s1_idx", "pool_idx").unique().with_columns(pl.lit(1, pl.UInt8).alias(k))
        out.append(p)
        print(f"    key {k}: {len(p):,} pairs (skipped {len(big_a) + len(big_b):,} oversized values)", flush=True)
    res = out[0]
    for p in out[1:]:
        res = res.join(p, on=["s1_idx", "pool_idx"], how="full", coalesce=True)
    return res.with_columns([pl.col(k).fill_null(0) for k in KEYS])


def row_cos(qa, pb, r, c, batch=2_000_000):
    """Cosine of explicit (query row, pool row) pairs, computed in batches."""
    out = np.empty(len(r), dtype=np.float32)
    for s in range(0, len(r), batch):
        out[s : s + batch] = np.asarray(qa[r[s : s + batch]].multiply(pb[c[s : s + batch]]).sum(axis=1)).ravel()
    return out


def block_country(a, b, k, cap):
    """Candidate pairs for one country."""
    t = time.time()
    qa, pb = vectorize(a["text"].to_list(), b["text"].to_list())
    r, c, v = topk(qa, pb, k)
    word = pl.DataFrame({"s1_idx": a["s1_idx"].to_numpy()[r], "pool_idx": b["pool_idx"].to_numpy()[c], "word_sim": v.astype(np.float32)})
    print(f"  word top-{k}: {len(word):,} pairs in {time.time() - t:.0f}s", flush=True)
    keys = key_pairs(a, b, cap)
    cand = word.join(keys, on=["s1_idx", "pool_idx"], how="full", coalesce=True).with_columns([pl.col(x).fill_null(0) for x in KEYS])
    miss = cand.filter(pl.col("word_sim").is_null())
    if len(miss):
        amap = np.full(a["s1_idx"].max() + 1, -1, dtype=np.int64)
        amap[a["s1_idx"].to_numpy()] = np.arange(len(a))
        bmap = np.full(b["pool_idx"].max() + 1, -1, dtype=np.int64)
        bmap[b["pool_idx"].to_numpy()] = np.arange(len(b))
        cos = row_cos(qa, pb, amap[miss["s1_idx"].to_numpy()], bmap[miss["pool_idx"].to_numpy()])
        cand = pl.concat([cand.filter(pl.col("word_sim").is_not_null()), miss.with_columns(pl.Series("word_sim", cos))])
    print(f"  union: {len(cand):,} pairs ({len(cand) / max(len(a), 1):.1f} per S1) in {time.time() - t:.0f}s", flush=True)
    return cand


def oracle(cand, s1, pool, gt):
    """Pair recall and oracle macro F0.5 of a candidate set against gold, by country."""
    g = gt.select(pl.col("source1_entity_id").alias("entity_id"), pl.col("matched_entity_ids").fill_null("").str.split(",").alias("m")).explode("m").filter(pl.col("m") != "")
    g = g.join(s1.select("entity_id", "s1_idx", "country_n"), on="entity_id").join(pool.select(pl.col("entity_id").alias("m"), "pool_idx"), on="m")
    hit = g.join(cand.select("s1_idx", "pool_idx"), on=["s1_idx", "pool_idx"], how="semi").group_by("s1_idx").len().rename({"len": "c"})
    tot = g.group_by("s1_idx").len().rename({"len": "t"})
    e = s1.select("s1_idx", "country_n").join(tot, on="s1_idx", how="left").join(hit, on="s1_idx", how="left").fill_null(0)
    e = e.with_columns(f=pl.when(pl.col("t") == 0).then(1.0).otherwise(1.25 * pl.col("c") / (0.25 * pl.col("t") + pl.col("c")).clip(1e-9)))
    n = cand.group_by("s1_idx").len()
    rep = e.group_by("country_n").agg(pl.col("c").sum() / pl.col("t").sum(), pl.col("f").mean(), (pl.col("c") == 0).and_(pl.col("t") > 0).mean().alias("all_missed")).sort("country_n")
    print(rep.rename({"c": "pair_recall", "f": "oracle_f05"}))
    print(f"overall pair recall {e['c'].sum() / e['t'].sum():.4f}, oracle F0.5 {e['f'].mean():.4f}, candidates/S1 mean {n['len'].mean():.1f} p99 {n['len'].quantile(0.99):.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--sample", type=int, help="random S1 sample per country (for quick evaluation)")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = load_config()
    nd = Path(cfg["cache_dir"]) / "norm"
    s1 = pl.read_parquet(nd / f"{a.split}_s1.parquet").with_row_index("s1_idx")
    pool = pl.read_parquet(nd / f"{a.split}_pool.parquet").with_row_index("pool_idx")
    parts, sampled_ids = [], []
    for c in sorted(s1["country_n"].unique().to_list()):
        a1 = s1.filter(pl.col("country_n") == c)
        if a.sample:
            a1 = a1.sample(min(a.sample, len(a1)), seed=42)
        b1 = pool.filter(pl.col("country_n") == c)
        sampled_ids.append(a1["s1_idx"])
        print(f"{c}: {len(a1):,} S1 x {len(b1):,} pool", flush=True)
        if len(b1):
            parts.append(block_country(a1, b1, a.k, a.cap).with_columns(pl.lit(c).alias("country_n")))
    cand = pl.concat(parts).with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col("pool_idx").cast(pl.UInt32))
    sampled = pl.concat(sampled_ids).cast(pl.UInt32).alias("s1_idx").to_frame()
    out = Path(cfg["cache_dir"]) / "cand"
    out.mkdir(parents=True, exist_ok=True)
    name = f"{a.split}_cand{a.tag}.parquet"
    cand.write_parquet(out / name)
    sampled.write_parquet(out / f"{a.split}_s1{a.tag}.parquet")
    print(f"wrote {name}: {len(cand):,} pairs")
    if a.eval and a.split == "train":
        gt = pl.read_parquet(Path(cfg["cache_dir"]) / "raw" / "train_ground_truth.parquet")
        s1e = s1.join(sampled.with_columns(pl.col("s1_idx").cast(s1["s1_idx"].dtype)), on="s1_idx", how="semi")
        oracle(cand, s1e, pool, gt)


if __name__ == "__main__":
    main()
