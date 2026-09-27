"""Cross-encoder data steps around the two Kaggle notebooks (CPU, each under a minute).

usage: python src/ce/ce_export.py records            -> cache/ce/records/{train,test}_{s1,pool}.parquet (n, a, c; raw row order)
       python src/ce/ce_export.py pairs              -> cache/ce/pairs/ce_pairs.parquet + ce_cfg.json (needs cache/cand/train_cand.parquet)
       python src/ce/ce_export.py band [src lo hi]   -> cache/ce/band/{train,test}_band.parquet (default 018_lexical 0.003 0.997)
       python src/ce/ce_export.py collect <dir> [col] -> cache/feat/{train,test}_ce.parquet from ce_{A,B}_{train,test}.parquet in <dir>
       python src/ce/ce_export.py extra test <band_src> <attempt>        -> cache/ce/extra/test_x.parquet, test_xn.parquet
       python src/ce/ce_export.py extra train <band_src> <attempt> ...   -> cache/ce/extra/train_x<i>.parquet, train_x<i>n.parquet
       python src/ce/ce_export.py collectx <col>=<dir1>,<dir2>,... ...   -> cache/feat/train_ce_x.parquet
       python src/ce/ce_export.py ce2                                    -> cache/feat/train_ce2.parquet, test_ce2_par.parquet

Pool index = source2 rows then source3 rows; s1_idx = source1 row order (same as cache/norm).
Training pairs: 470k random S1 per half (half 0 = folds 0,1; half 1 = folds 2,3,4); per S1 every gold candidate pair,
the top 8 non-gold candidates by word_sim and 2 random other non-gold candidates.
Collect: train = model A on half 1 + model B on half 0; test = mean of the A and B logits; column name default ce.
Extra: pairs of later attempts outside the band (prob > 0.001, not in cache/feat/{split}_ce.parquet); train attempts are
taken in order, each minus the pairs of the earlier ones, with the S1 half; the *n files keep the pairs absent from
<band_src> (the subset also scored by a second model). collectx joins, per extra train file i, each column's half-model
scores from its i-th directory (ce_{A,B}_train.parquet); test uses `er_extend.py parity ce_x_par <col>=<dir> ...`.
ce2: train_ce2 = train_ce + train_ce_x, test_ce2_par = test_ce_par + test_ce_x_par; the first file wins on duplicate keys.
Kaggle (private): kaggle datasets create/version -p cache/ce/{records,pairs,band}; kaggle kernels push with
src/ce/ce_train.py then src/ce/ce_score.py (enable_gpu, --accelerator NvidiaTeslaT4); kaggle kernels output.
"""
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import er_model as M
from common import load_config

K = ["s1_idx", "pool_idx"]


def halves(cfg):
    s1 = M.load_s1(cfg, "train")
    gold = M.load_gold(cfg, s1)
    fo = M.folds(cfg, s1, gold).select(pl.col("s1_idx").cast(pl.UInt32), (pl.col("fold") >= 2).cast(pl.Int8).alias("half"))
    return gold, fo


def records(cache, out):
    for sp in ["train", "test"]:
        for side, fs in [("s1", ["source1"]), ("pool", ["source2", "source3"])]:
            d = pl.concat([pl.read_parquet(cache / "raw" / f"{sp}_{f}.parquet", columns=["business_name", "business_address", "country"]) for f in fs])
            d = d.select(pl.col("business_name").fill_null("").alias("n"), pl.col("business_address").fill_null("").alias("a"), pl.col("country").cast(pl.Categorical).alias("c"))
            d.write_parquet(out / f"{sp}_{side}.parquet", compression="zstd", compression_level=9)
            print(sp, side, d.shape, flush=True)


def pairs(cfg, cache, out, k_top=8, k_rand=2, n_s1=470000):
    gold, f = halves(cfg)
    s = pl.concat([f.filter(pl.col("half") == h).sample(n_s1, seed=h) for h in (0, 1)])
    g = gold.with_columns(y=pl.lit(1, pl.Int8))
    c = pl.scan_parquet(cache / "cand" / "train_cand.parquet").select(pl.col("s1_idx").cast(pl.UInt32), pl.col("pool_idx").cast(pl.UInt32), "word_sim").join(s.lazy(), on="s1_idx").collect()
    c = c.join(g, on=K, how="left").with_columns(pl.col("y").fill_null(0), r=pl.Series(np.random.default_rng(0).random(len(c))))
    n = c.filter(pl.col("y") == 0).with_columns(rk=pl.col("word_sim").rank("ordinal", descending=True).over("s1_idx"), rr=pl.col("r").rank("ordinal").over("s1_idx"))
    n = n.filter(pl.col("rk") <= k_top).vstack(n.filter((pl.col("rk") > k_top) & (pl.col("rr") <= k_rand)))
    d = pl.concat([c.filter(pl.col("y") == 1), n.select(c.columns)]).select("s1_idx", "pool_idx", "y", "half")
    print(d.group_by("half").agg(pl.len(), pl.col("y").mean()))
    d.write_parquet(out / "ce_pairs.parquet")
    (out / "ce_cfg.json").write_text('{"budget_min": 170, "maxn": 6200000}\n')


def band(cfg, cache, out, src="018_lexical", lo=0.003, hi=0.997):
    _, h = halves(cfg)
    for sp, fn in [("train", f"{src}.parquet"), ("test", f"test_{src}.parquet")]:
        d = pl.read_parquet(cache / "oof" / fn, columns=K + ["prob"]).filter(pl.col("prob").is_between(lo, hi)).select(pl.col("s1_idx").cast(pl.UInt32), pl.col("pool_idx").cast(pl.UInt32))
        if sp == "train":
            d = d.join(h, on="s1_idx", how="left")
        d.write_parquet(out / f"{sp}_band.parquet")
        print(sp, "band", d.height, flush=True)


def collect(cache, src, col="ce"):
    rd = lambda fn, nm: pl.read_parquet(Path(src) / fn).select(pl.col("s1_idx").cast(pl.UInt32), pl.col("pool_idx").cast(pl.UInt32), pl.col("ce").cast(pl.Float32).alias(nm))
    for sp in ["train", "test"]:
        b = pl.read_parquet(cache / "ce" / "band" / f"{sp}_band.parquet").select(K)
        if sp == "train":
            x = pl.concat([rd("ce_A_train.parquet", col), rd("ce_B_train.parquet", col)])
        else:
            x = rd("ce_A_test.parquet", "a").join(rd("ce_B_test.parquet", "b"), on=K).select(*K, ((pl.col("a") + pl.col("b")) / 2).cast(pl.Float32).alias(col))
        assert x.height == b.height == x.unique(K).height and x[col].null_count() == 0, sp
        fp = cache / "feat" / f"{sp}_ce.parquet"
        base = pl.read_parquet(fp).drop(col, strict=False) if fp.exists() else b
        out = base.join(x, on=K, how="left")
        assert out.height == b.height and out[col].null_count() == 0, sp
        out.write_parquet(fp)
        print("wrote", fp, out.shape, out.columns, flush=True)


def extra(cfg, cache, out, split, src, attempts):
    band = pl.read_parquet(cache / "feat" / f"{split}_ce.parquet", columns=K)
    old = pl.read_parquet(cache / "oof" / f"{'test_' if split == 'test' else ''}{src}.parquet", columns=K)
    ext = lambda a: pl.read_parquet(cache / "oof" / f"{a}.parquet", columns=K + ["prob"]).filter(pl.col("prob") > 0.001).join(band, on=K, how="anti").select(pl.col("s1_idx").cast(pl.UInt32), pl.col("pool_idx").cast(pl.UInt32))
    if split == "test":
        x = ext(f"test_{attempts[0]}")
        x.write_parquet(out / "test_x.parquet"); x.join(old, on=K, how="anti").write_parquet(out / "test_xn.parquet")
        print("test extra", x.height, flush=True)
        return
    _, h = halves(cfg)
    prev = None
    for i, a in enumerate(attempts, 1):
        x = ext(a)
        if prev is not None:
            x = x.join(prev, on=K, how="anti")
        x = x.unique(K, maintain_order=True).join(h, on="s1_idx", how="left")
        x.write_parquet(out / f"train_x{i}.parquet"); x.join(old, on=K, how="anti").write_parquet(out / f"train_x{i}n.parquet")
        prev = x.select(K) if prev is None else pl.concat([prev, x.select(K)])
        print(a, "train extra", x.height, flush=True)


def collectx(cache, specs):
    rd = lambda d, c: pl.concat([pl.read_parquet(Path(d) / f"ce_{m}_train.parquet").select(pl.col("s1_idx").cast(pl.UInt32), pl.col("pool_idx").cast(pl.UInt32), pl.col("ce").cast(pl.Float32).alias(c)) for m in "AB"])
    cols = [(c, d.split(",")) for c, d in (a.split("=", 1) for a in specs)]
    parts = []
    for i in range(len(cols[0][1])):
        x = pl.read_parquet(cache / "ce" / "extra" / f"train_x{i + 1}.parquet").select(K)
        for c, ds in cols:
            x = x.join(rd(ds[i], c), on=K, how="left")
        parts.append(x)
    x = pl.concat(parts)
    band = pl.read_parquet(cache / "feat" / "train_ce.parquet", columns=K)
    assert x[cols[0][0]].null_count() == 0 and x.select(K).is_unique().all() and x.join(band, on=K).height == 0
    x.write_parquet(cache / "feat" / "train_ce_x.parquet")
    print("train_ce_x", x.shape, {c: x.height - x[c].null_count() for c, _ in cols}, flush=True)


def ce2(cache):
    for out, a, b in [("train_ce2", "train_ce", "train_ce_x"), ("test_ce2_par", "test_ce_par", "test_ce_x_par")]:
        x = pl.read_parquet(cache / "feat" / f"{a}.parquet")
        y = pl.read_parquet(cache / "feat" / f"{b}.parquet").select(x.columns).cast(x.schema)
        z = pl.concat([x, y.join(x.select(K), on=K, how="anti")])
        assert z.select(K).is_unique().all()
        z.write_parquet(cache / "feat" / f"{out}.parquet")
        print(out, x.height, y.height, z.height, flush=True)


def main():
    cfg = load_config()
    cache = Path(cfg["cache_dir"])
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "collect":
        return collect(cache, *args)
    if cmd == "collectx":
        return collectx(cache, args)
    if cmd == "ce2":
        return ce2(cache)
    out = cache / "ce" / cmd
    os.makedirs(out, exist_ok=True)
    if cmd == "records":
        records(cache, out)
    elif cmd == "pairs":
        pairs(cfg, cache, out)
    elif cmd == "extra":
        extra(cfg, cache, out, args[0], args[1], args[2:])
    elif cmd == "band":
        band(cfg, cache, out, *(args[:1] + [float(x) for x in args[1:3]]))


if __name__ == "__main__":
    main()
