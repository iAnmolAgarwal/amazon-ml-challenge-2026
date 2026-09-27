import sys
sys.path.insert(0, "research/r23")
from ladder import *
bb2 = bb
q = load("030q_fix")
for T, fl in [(1.0, 0.5), (1.0, 0.4), (0.8, 0.6), (0.67, 0.6), (0.5, 0.6), (0.67, 0.4)]:
    report(f"030q T{T} floor{fl}", temp(q, T), fl)
t = pl.read_parquet("cache/oof/test_029f7_final.parquet").join(bb, on=K, how="left").filter(pl.col("pb").is_null() | (pl.col("pb") >= 0.001)).drop("pb")
d = M.decide(t, cfg, floor=0.6).select(K).join(s1.select("s1_idx", "country_n"), on="s1_idx")
for c in ["us", "india", "france"]:
    nn = s1.filter(pl.col("country_n") == c).height; e = d.filter(pl.col("country_n") == c)
    print(c, "test emitted/S1", round(e.height / nn, 4), "empty", round(1 - e["s1_idx"].n_unique() / nn, 4), "conflict", prof(e.select(K)), flush=True)
