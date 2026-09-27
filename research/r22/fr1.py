import sys
import polars as pl
sys.path.insert(0, "src"); sys.path.insert(0, "research/r22")
import er_model as M
from common import load_config
from rcls import views, flag
cfg = load_config(); K = ["s1_idx", "pool_idx"]
s1 = pl.read_parquet("cache/norm/test_s1.parquet", columns=["entity_id", "country_n", "house"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
pool = pl.read_parquet("cache/norm/test_pool.parquet", columns=["entity_id", "house"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32))
fr = s1.filter(pl.col("country_n") == "france")
base = pl.read_parquet("cache/oof/test_029n_final.parquet")
bf = base.join(fr.select("s1_idx"), on="s1_idx")
st = pl.read_parquet("cache/stack/p_test_France.parquet").join(fr.select("s1_idx", a_id="entity_id"), on="a_id").join(pool.select("pool_idx", r_id="entity_id"), on="r_id").select(K + [pl.col("p").alias("pS")])
d = flag(bf.join(st, on=K, how="left"), *views("test"))
new = d.with_columns(prob=pl.when(pl.col("R") & (pl.col("prob") > 0) & pl.col("pS").is_not_null()).then(pl.max_horizontal("prob", "pS")).otherwise(pl.col("prob")).cast(pl.Float32))
print("R pairs", d["R"].sum(), "changed", new.filter(pl.col("prob") != d["prob"]).height)
out = pl.concat([base.join(fr.select("s1_idx"), on="s1_idx", how="anti"), new.select(K + ["prob"])]).sort(K)
out.write_parquet("research/r22/test_fr_R.parquet")
hs = s1.select("s1_idx", hs="house"); hp = pool.select("pool_idx", hp="house")
def guard(p, tag):
    e = M.decide(p, cfg, floor=0.6).select(K).join(s1.select("s1_idx", "country_n"), on="s1_idx").join(hs, on="s1_idx").join(hp, on="pool_idx")
    g = e.group_by("country_n").agg(k=pl.len(), ne=pl.col("s1_idx").n_unique(), hc=((pl.col("hs").fill_null("") != "") & (pl.col("hp").fill_null("") != "") & (pl.col("hs") != pl.col("hp"))).mean())
    g = g.join(s1.group_by("country_n").len(), on="country_n").with_columns(perS1=pl.col("k") / pl.col("len"), empty=1 - pl.col("ne") / pl.col("len")).sort("country_n")
    print(tag, g.select("country_n", "k", "perS1", "empty", "hc"))
    return e
e0, e1 = guard(base, "029n"), guard(out, "R")
add = e1.join(e0, on=K, how="anti"); drop = e0.join(e1, on=K, how="anti")
sh = lambda x: x.with_columns(dh=(pl.col("hp").cast(pl.Int64, strict=False) - pl.col("hs").cast(pl.Int64, strict=False)).abs()).select(pl.len(), shift=pl.col("dh").is_in([1, 2, 3, 4, 5, 7, 9, 11, 13, 21]).mean(), hc=((pl.col("hs").fill_null("") != "") & (pl.col("hp").fill_null("") != "") & (pl.col("hs") != pl.col("hp"))).mean())
print("added", add.group_by("country_n").len(), sh(add)); print("dropped", drop.group_by("country_n").len(), sh(drop))
add.write_parquet("research/r22/added_R.parquet")
