import sys, json
src = open("research/r23/fr.py").read()
exec(src[:src.index("arms = ")])
L = {"a": {"T": 2, "fl": 0.7, "er": 3, "dq": 0.8}, "b": {"T": 2.5, "fl": 0.7, "er": 3, "dq": 0.8}, "c": {"T": 3, "fl": 0.7, "er": 3, "dq": 0.8}}
full7 = pl.read_parquet("cache/oof/test_029f7_final.parquet")
nf = full7.join(fr.select("s1_idx"), on="s1_idx", how="anti")
d7 = M.decide(full7, cfg, floor=0.6).select(K)
d7n = d7.join(fr.select("s1_idx"), on="s1_idx", how="anti")
for nm, kw in L.items():
    p = tf(q, q["prob"].cast(pl.Float64), **kw)
    exp = M.decide(p, cfg, floor=0.6).select(K)
    out = pl.concat([nf, p.with_columns(pl.col("prob").cast(pl.Float32))]).sort(K)
    fn = f"cache/oof/test_029f9{nm}_final.parquet"
    out.write_parquet(fn)
    back = pl.read_parquet(fn)
    e = M.decide(back, cfg, floor=0.6).select(K)
    ef = e.join(fr.select("s1_idx"), on="s1_idx")
    en = e.join(fr.select("s1_idx"), on="s1_idx", how="anti")
    print(nm, fn, "rows", back.height, "vs 029f7", full7.height, "| France set == research", ef.sort(K).equals(exp.sort(K)), ef.height, exp.height, "| non-France emitted == 029f7", en.sort(K).equals(d7n.sort(K)), en.height, "| non-France rows identical", back.join(fr.select("s1_idx"), on="s1_idx", how="anti").sort(K).equals(nf.sort(K)), flush=True)
