import sys
import polars as pl
sys.path.insert(0, "src")
import er_model as M
from common import load_config
cfg = load_config(); K = ["s1_idx", "pool_idx"]
s1 = pl.read_parquet("cache/norm/test_s1.parquet", columns=["country_n", "house"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
a = pl.read_parquet("cache/oof/test_029n_final.parquet"); b = pl.read_parquet("cache/oof/test_029f2_final.parquet"); r = pl.read_parquet("research/r22/test_fr_R.parquet")
fr = s1.filter(pl.col("country_n") == "france").select("s1_idx")
ua, ub = a.join(fr, on="s1_idx", how="anti"), b.join(fr, on="s1_idx", how="anti")
print("US/India identical:", ua.equals(ub), ua.height)
fa, fb, fq = a.join(fr, on="s1_idx"), b.join(fr, on="s1_idx"), r.join(fr, on="s1_idx")
j = fa.join(fb, on=K, how="full", coalesce=True, suffix="_b")
print("France rows", fa.height, fb.height, "changed", j.filter(pl.col("prob") != pl.col("prob_b")).height, "raised", j.filter(pl.col("prob_b") > pl.col("prob")).height, "lowered", j.filter(pl.col("prob_b") < pl.col("prob")).height)
jr = fq.join(fb, on=K, suffix="_b"); print("vs research file: differing pairs", jr.filter((pl.col("prob") - pl.col("prob_b")).abs() > 1e-6).height)
pool = pl.read_parquet("cache/norm/test_pool.parquet", columns=["house"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32))
for tag, p in (("029n", a), ("029f2", b)):
    e = M.decide(p, cfg, floor=0.6).select(K).join(s1, on="s1_idx").join(pool, on="pool_idx", suffix="_p")
    g = e.group_by("country_n").agg(k=pl.len(), ne=pl.col("s1_idx").n_unique(), hc=((pl.col("house").fill_null("") != "") & (pl.col("house_p").fill_null("") != "") & (pl.col("house") != pl.col("house_p"))).mean()).join(s1.group_by("country_n").len(), on="country_n")
    print(tag, [(c, k, round(k / n, 4), round(1 - ne / n, 5), round(hc, 5)) for c, k, ne, hc, n in g.sort("country_n").iter_rows()])
