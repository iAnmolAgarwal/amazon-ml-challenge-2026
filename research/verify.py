import json
import sys
import numpy as np
import polars as pl
sys.path.insert(0, "src")
import er_model as M
from common import load_config

cfg = load_config()
K = ["s1_idx", "pool_idx"]


def paired(a, b):
    s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1); fo = M.folds(cfg, s1, gold)
    ents = s1.select("s1_idx", "country_n").join(fo.select("s1_idx", "fold"), on="s1_idx")
    r = {}
    for n in (a, b):
        p = pl.read_parquet(f"cache/oof/{n}.parquet", columns=K + ["prob"])
        r[n] = M.per_entity_f05(M.decide(p, cfg, floor=0.6).select(K), gold, ents)
        print(n, f"F0.5 {r[n]['f05'].mean():.5f}", {k: round(v, 5) for k, v in r[n].group_by("country_n").agg(pl.col("f05").mean()).iter_rows()}, flush=True)
    j = r[a].select("s1_idx", "fold", "country_n", x="f05").join(r[b].select("s1_idx", z="f05"), on="s1_idx").with_columns(d=pl.col("z") - pl.col("x"))
    d = j.group_by("fold").agg(pl.col("d").mean()).sort("fold")["d"].to_numpy()
    c = {k: round(v, 5) for k, v in j.group_by("country_n").agg(pl.col("d").mean()).iter_rows()}
    ok = d.mean() > 2 * d.std(ddof=1) / np.sqrt(len(d)) and (d > 0).sum() >= 4
    print(f"PAIRED {b} vs {a}: {d.mean():+.5f} SE {d.std(ddof=1) / np.sqrt(len(d)):.5f} folds+ {(d > 0).sum()}/5 by country {c} -> {'PASS' if ok else 'FAIL'}")


def guard(*names):
    s1 = pl.read_parquet("cache/norm/test_s1.parquet", columns=["country_n", "house"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
    pool = pl.read_parquet("cache/norm/test_pool.parquet", columns=["house"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32)).rename({"house": "house_p"})
    prev = None
    for n in names:
        p = pl.read_parquet(f"cache/oof/test_{n}.parquet", columns=K + ["prob"])
        d = M.decide(p, cfg, floor=0.6).select(K).join(s1, on="s1_idx").join(pool, on="pool_idx")
        out = {}
        for c in ("us", "india", "france"):
            e = d.filter(pl.col("country_n") == c); nn = s1.filter(pl.col("country_n") == c).height
            conf = e.filter((pl.col("house").fill_null("") != "") & (pl.col("house_p").fill_null("") != "") & (pl.col("house") != pl.col("house_p"))).height
            out[c] = (round(e.height / nn, 4), round(1 - e.select("s1_idx").n_unique() / nn, 4), round(conf / max(e.height, 1), 4))
        extra = ""
        if prev is not None:
            extra = f"; vs previous: added {d.select(K).join(prev, on=K, how='anti').height:,}, dropped {prev.join(d.select(K), on=K, how='anti').height:,}"
        print(f"GUARD {n} (pairs/S1, empty, house-conflict): {out}; candidates {p.height / s1.height:.2f}/S1{extra}", flush=True)
        prev = d.select(K)


def expect(attempt, want, tol):
    r = json.load(open(f"attempts/{attempt}/results.json"))
    ok = abs(r["mean"] - want) <= tol
    print(f"EXPECT {attempt}: CV {r['mean']:.5f} vs original {want:.5f} (tol {tol}) -> {'PASS' if ok else 'CHECK'}")


if __name__ == "__main__":
    {"paired": paired, "guard": guard, "expect": lambda a, w, t: expect(a, float(w), float(t))}[sys.argv[1]](*sys.argv[2:])
