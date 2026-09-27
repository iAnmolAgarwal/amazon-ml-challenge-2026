import sys, json
src = open("research/r23/lo.py").read()
exec(src[:src.index("ARMS = ")])
arms = json.loads(sys.argv[1])
hc = ((pl.col("house").fill_null("") != "") & (pl.col("house_p").fill_null("") != "") & (pl.col("house") != pl.col("house_p")))
for cty in ["india", "us"]:
    x0 = d.filter(pl.col("country_n") == cty)
    ne = ent.filter(pl.col("country_n") == cty).height
    ref = M.decide(tf(x0, x0["pA_lo"].cast(pl.Float64), T=1.5), cfg, floor=0.6).select(K).join(x0, on=K)
    r0 = ref.select(pl.len(), hc.mean(), pl.col("dec").mean(), pl.col("y").mean()).row(0)
    print(cty, "ref T1.5", f"S1 {r0[0] / ne:.4f} conf {r0[1]:.4f} dsh {r0[2]:.4f} prec {r0[3]:.4f}", flush=True)
    for nm, kw in arms.items():
        e = M.decide(tf(x0, x0["pA_lo"].cast(pl.Float64), **kw), cfg, floor=0.6).select(K)
        f, k, em = score(e, cty)
        dr = ref.join(e, on=K, how="anti")
        r = dr.select(pl.len(), hc.mean(), pl.col("dec").mean(), pl.col("y").mean()).row(0)
        print(cty, nm, f"F {f.mean():.5f} S1 {k:.4f} empty {em:.4f} | dropped/S1 {r[0] / ne:.4f} conf {r[1] or 0:.4f} ({(r[1] or 0) / r0[1]:.1f}x) dsh {r[2] or 0:.4f} ({(r[2] or 0) / r0[2]:.1f}x) prec {r[3] or 0:.3f}", flush=True)
