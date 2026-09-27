import sys
sys.path.insert(0, "research/r23")
from ladder import *
t = pl.read_parquet("cache/oof/test_029f7_final.parquet").join(bb, on=K, how="left").filter(pl.col("pb").is_null() | (pl.col("pb") >= 0.001)).drop("pb")
d = M.decide(t, cfg, floor=0.6).select(K + ["prob"]).join(s1, on="s1_idx").join(pool, on="pool_idx")
hs, hq = pl.col("house").str.extract(r"(\d+)").cast(pl.Int64, strict=False), pl.col("hp").str.extract(r"(\d+)").cast(pl.Int64, strict=False)
both = (pl.col("house").fill_null("") != "") & (pl.col("hp").fill_null("") != "")
d = d.with_columns(both=both, conf=both & (pl.col("house") != pl.col("hp")), shift=(both & (pl.col("house") != pl.col("hp")) & (hq - hs).abs().is_in(STEPS)).fill_null(False),
    pb=pl.when(pl.col("prob") < 0.9).then(pl.lit("a<.9")).when(pl.col("prob") < 0.99).then(pl.lit("b<.99")).when(pl.col("prob") < 0.999).then(pl.lit("c<.999")).otherwise(pl.lit("d>=.999")))
print(d.group_by("country_n", "pb").agg(n=pl.len(), both=pl.col("both").mean(), conf=pl.col("conf").mean(), shift=pl.col("shift").mean()).sort("country_n", "pb"))
print(d.filter((pl.col("country_n") == "france") & pl.col("conf")).group_by("pb").agg(n=pl.len(), shift=pl.col("shift").mean()).sort("pb"))
