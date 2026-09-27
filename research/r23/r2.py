import sys, types, json
sys.path.insert(0, "src")
import numpy as np
import polars as pl
import lightgbm as lgb
import er_model as M
import er_fix as F
from common import load_attempt, load_config
from pathlib import Path
cfg = load_config(); cache = Path(cfg["cache_dir"]); K = ["s1_idx", "pool_idx"]
tag, extra = sys.argv[1], sys.argv[2] == "1"
over = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
b = load_attempt("030q_fix")
mod = types.SimpleNamespace(**{k: getattr(b, k) for k in dir(b) if not k.startswith("_")})
for k, v in over.items(): setattr(mod, k, tuple(v) if k == "BAND" else v)
mod.PARAMS = dict(mod.PARAMS, num_threads=6)
STEPS = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
def dec(split, keys):
    s = pl.read_parquet(f"cache/norm/{split}_s1.parquet", columns=["house", "name_core", "postal", "street1"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
    p = pl.read_parquet(f"cache/norm/{split}_pool.parquet", columns=["house", "name_core", "postal", "street1"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32))
    x = keys.select(K).join(s, on="s1_idx", how="left").join(p, on="pool_idx", how="left", suffix="_p")
    hs, hq = pl.col("house").str.extract(r"(\d+)").cast(pl.Float64, strict=False), pl.col("house_p").str.extract(r"(\d+)").cast(pl.Float64, strict=False)
    ne = lambda c: pl.col(c).fill_null("") != ""
    st, pt = pl.col("name_core").fill_null("").str.split(" "), pl.col("name_core_p").fill_null("").str.split(" ")
    x = x.with_columns(h_conf=(ne("house") & ne("house_p") & (pl.col("house") != pl.col("house_p"))).cast(pl.Float32), h_d=(hq - hs).cast(pl.Float32))
    x = x.with_columns(h_step=pl.col("h_d").abs().is_in([float(v) for v in STEPS]).fill_null(False).cast(pl.Float32) * pl.col("h_conf"), h_ad=pl.col("h_d").abs(),
        st_eq=(pl.col("street1") == pl.col("street1_p")).cast(pl.Float32), pc_eq=(pl.col("postal") == pl.col("postal_p")).cast(pl.Float32),
        w_add=pt.list.set_difference(st).list.len().cast(pl.Float32), w_drop=st.list.set_difference(pt).list.len().cast(pl.Float32))
    cols = ["h_conf", "h_d", "h_step", "h_ad", "st_eq", "pc_eq", "w_add", "w_drop"]
    return x.select(cols).fill_null(np.nan).to_numpy().astype(np.float32), cols
s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1); fo = M.folds(cfg, s1, gold)
oof = pl.read_parquet(cache / "oof" / "030p_fix.parquet", columns=K + ["prob"])
keys, lp, X, names = F.build(cache, "train", oof, mod, s1.select("s1_idx", "country_n"))
if extra:
    E, en = dec("train", keys); X = np.hstack([X, E]); names = names + en
lab = keys.join(s1.select("s1_idx", "country_n"), on="s1_idx", how="left", maintain_order="left").join(gold.with_columns(y=pl.lit(1, pl.Int8)), on=K, how="left", maintain_order="left")
y, cty = lab["y"].fill_null(0).to_numpy(), lab["country_n"].to_numpy()
print(tag, "band pairs", len(y), "pos", int(y.sum()), names, flush=True)
ents = s1.select("s1_idx", "country_n").join(fo.select("s1_idx", "fold"), on="s1_idx")
base = M.per_entity_f05(M.decide(oof, cfg, floor=0.6).select(K), gold, ents).select("s1_idx", "fold", "country_n", x="f05")
for src, dst in [("us", "india"), ("india", "us")]:
    tr, va = np.flatnonzero(cty == src), np.flatnonzero(cty == dst)
    _, p = F.fit_predict(mod.PARAMS, mod.ROUNDS, X, y, lp, names, tr, va)
    out = keys["prob"].to_numpy().astype(np.float64).copy(); out[va] = p
    z = M.per_entity_f05(M.decide(F.merge(oof, keys, F.restrict(mod, keys, out)), cfg, floor=0.6).select(K), gold, ents).select("s1_idx", z="f05")
    j = base.filter(pl.col("country_n") == dst).join(z, on="s1_idx").with_columns(d=pl.col("z") - pl.col("x"))
    d = j.group_by("fold").agg(pl.col("d").mean()).sort("fold")["d"].to_numpy(); se = d.std(ddof=1) / np.sqrt(5)
    print(f"{tag} round2 trained {src} -> {dst}: {d.mean():+.6f} SE {se:.6f} folds+ {(d > 0).sum()}/5 {'PASS' if d.mean() > 2 * se and (d > 0).sum() >= 4 else 'FAIL'}", flush=True)
m = lgb.train(mod.PARAMS, lgb.Dataset(X, y, init_score=lp, feature_name=names), mod.ROUNDS)
del X
tst = pl.read_parquet(cache / "oof" / "test_030q_fix.parquet", columns=K + ["prob"])
tk, tlp, TX, tn = F.build(cache, "test", tst, mod, M.load_s1(cfg, "test").select("s1_idx", "country_n"), france=True)
if extra:
    E, _ = dec("test", tk); TX = np.hstack([TX, E])
tp = 1 / (1 + np.exp(-(tlp + m.predict(TX, raw_score=True))))
F.merge(tst, tk, F.restrict(mod, tk, tp)).write_parquet(cache / "oof" / f"test_r23_{tag}.parquet")
imp = sorted(zip(names, m.feature_importance("gain")), key=lambda x: -x[1])[:14]
print(tag, "test France band", len(tk), "top gain:", ", ".join(f"{n} {g:.0f}" for n, g in imp), flush=True)
