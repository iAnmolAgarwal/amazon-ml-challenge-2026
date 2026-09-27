import sys, json
import polars as pl
sys.path.insert(0, "src")
import er_model as M
from common import load_config
cfg = load_config(); K = ["s1_idx", "pool_idx"]
SH = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
src = open("research/r23/lo.py").read()
exec(src[src.index("def tf("):src.index("def score(")])
s1 = pl.read_parquet("cache/norm/test_s1.parquet", columns=["country_n", "house"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
hp = pl.read_parquet("cache/norm/test_pool.parquet", columns=["house"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32)).rename({"house": "house_p"})
fr = s1.filter(pl.col("country_n") == "france")
num = lambda c: pl.col(c).str.extract(r"^(\d+)").cast(pl.Int64, strict=False)
q = pl.read_parquet("cache/oof/test_030q_fix.parquet").join(fr.select("s1_idx", "house"), on="s1_idx").join(hp, on="pool_idx", how="left")
q = q.with_columns(dec=(num("house_p") - num("house")).abs().is_in(SH).fill_null(False))
n = fr.height
def prof(e):
    e = e.select(K).join(q.select(*K, "house", "house_p", "dec"), on=K)
    hc = ((pl.col("house").fill_null("") != "") & (pl.col("house_p").fill_null("") != "") & (pl.col("house") != pl.col("house_p")))
    r = e.select(pl.len(), hc.mean(), pl.col("dec").mean()).row(0)
    return r[0], r[1] or 0.0, r[2] or 0.0
f7 = pl.read_parquet("cache/oof/test_029f7_final.parquet").join(fr.select("s1_idx"), on="s1_idx")
ref = M.decide(f7, cfg, floor=0.6).select(K)
b0 = M.decide(q.select(*K, "prob"), cfg, floor=0.6).select(K)
print("030q", prof(b0), "029f7", prof(ref), "refS1", ref.height / n, "empty", 1 - ref["s1_idx"].n_unique() / n, flush=True)
arms = json.loads(sys.argv[1])
for nm, kw in arms.items():
    p = tf(q, q["prob"].cast(pl.Float64), **kw)
    e = M.decide(p, cfg, floor=0.6).select(K)
    dr, ad = ref.join(e, on=K, how="anti"), e.join(ref, on=K, how="anti")
    a, b, c = prof(e); x, y, z = prof(dr); u, v, w = prof(ad)
    print(f"{nm}: S1 {a / n:.4f} empty {1 - e['s1_idx'].n_unique() / n:.4f} conf {b:.4f} dsh {c:.4f} | dropped {x} conf {y:.4f} dsh {z:.4f} | added {u} conf {v:.4f} dsh {w:.4f}", flush=True)
    if len(sys.argv) > 2 and nm in sys.argv[2].split(","):
        p.write_parquet(f"research/r23/fr_{nm}.parquet")
