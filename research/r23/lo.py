import sys
import numpy as np
import polars as pl
sys.path.insert(0, "src")
import er_model as M
from common import load_config
cfg = load_config(); K = ["s1_idx", "pool_idx"]
SH = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
d = pl.read_parquet("research/r22/loco_preds2.parquet", columns=K + ["y", "country_n", "fold", "pA_ic", "pA_lo"])
s1 = M.load_s1(cfg, "train")
hs = pl.read_parquet("cache/norm/train_s1.parquet", columns=["house"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
hp = pl.read_parquet("cache/norm/train_pool.parquet", columns=["house"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32)).rename({"house": "house_p"})
num = lambda c: pl.col(c).str.extract(r"^(\d+)").cast(pl.Int64, strict=False)
d = d.join(hs, on="s1_idx", how="left").join(hp, on="pool_idx", how="left").with_columns(dec=(num("house_p") - num("house")).abs().is_in(SH).fill_null(False))
fo = d.select("s1_idx", "fold").unique()
smp = s1.filter(pl.col("s1_idx").hash(17) % 10 == 0)
gold = M.load_gold(cfg, s1)
t = gold.group_by("s1_idx").len().rename({"len": "t"})
ent = smp.join(t, on="s1_idx", how="left").with_columns(pl.col("t").fill_null(0))
ff = pl.read_parquet("data/folds.parquet").select(pl.col("s1_idx").cast(pl.UInt32), pl.col("fold").cast(pl.Int8))
chk = fo.join(ff, on="s1_idx"); print("fold agree", (chk["fold"] == chk["fold_right"]).mean(), flush=True)
ent = ent.join(ff, on="s1_idx")
G = d.filter(pl.col("y") == 1).select(K)

def tf(x, p, T=1.0, fl=None, er=1.0, cap=None, capb=1.0, dq=None):
    x = x.with_columns(prob=p)
    if T != 1.0:
        z = (x["prob"].clip(1e-9, 1 - 1e-7) / (1 - x["prob"].clip(1e-9, 1 - 1e-7))).log()
        x = x.with_columns(prob=pl.when(pl.col("prob") > 0).then(1 / (1 + (-z / T).exp())).otherwise(0.0))
    if dq is not None:
        x = x.with_columns(prob=pl.when(pl.col("dec") & (pl.col("prob") < dq)).then(0.0).otherwise(pl.col("prob")))
    if fl is not None:
        x = x.with_columns(prob=pl.when(pl.col("prob") <= fl).then(0.0).otherwise(pl.col("prob")))
    if cap is not None:
        o = x.sort(["pool_idx", "prob", "s1_idx"], descending=[False, True, False]).unique("pool_idx", keep="first", maintain_order=True).select(*K, "prob")
        o = o.with_columns(rk=pl.col("prob").rank("ordinal", descending=True).over("s1_idx"), top=pl.col("prob").max().over("s1_idx"))
        z = o.filter((pl.col("rk") > cap) & (pl.col("top") < capb)).select(*K, zz=pl.lit(True))
        x = x.join(z, on=K, how="left").with_columns(prob=pl.when(pl.col("zz").fill_null(False)).then(0.0).otherwise(pl.col("prob"))).drop("zz")
    if er != 1.0:
        keep = M.decide(x.select(*K, "prob"), cfg, floor=0.0, empty_ratio=er).select("s1_idx").unique()
        x = x.with_columns(prob=pl.when(pl.col("s1_idx").is_in(keep["s1_idx"].implode())).then(pl.col("prob")).otherwise(0.0))
    return x.select(*K, "prob")

def score(e, cty):
    en = ent.filter(pl.col("country_n") == cty)
    f = M.per_entity_f05(e.select(K), G, en.select("s1_idx")).join(en.select("s1_idx", "fold"), on="s1_idx")
    g = f.group_by("fold").agg(pl.col("f05").mean()).sort("fold")
    return g["f05"].to_numpy(), e.height / en.height, 1 - e["s1_idx"].n_unique() / en.height

ARMS = {"base": {}}
for T in [1.5, 2, 2.5, 3]:
    ARMS[f"T{T}"] = {"T": T}
for T in [1.0, 1.5, 2, 2.5]:
    for fl in [0.7, 0.8]:
        ARMS[f"T{T}_fl{fl}"] = {"T": T, "fl": fl}
for T in [1.5, 2]:
    for er in [1.5, 2, 3]:
        ARMS[f"T{T}_er{er}"] = {"T": T, "er": er}
    for c, b in [(1, 0.9), (2, 0.9), (1, 0.8), (3, 1.01)]:
        ARMS[f"T{T}_cap{c}b{b}"] = {"T": T, "cap": c, "capb": b}
    for q in [0.8, 0.9, 0.95, 0.99]:
        ARMS[f"T{T}_dq{q}"] = {"T": T, "dq": q}
ARMS["T2_fl0.7_dq0.95"] = {"T": 2, "fl": 0.7, "dq": 0.95}
ARMS["T2_fl0.7_er2"] = {"T": 2, "fl": 0.7, "er": 2}
ARMS["T2_dq0.95_er2"] = {"T": 2, "dq": 0.95, "er": 2}
ARMS["T2.5_dq0.95_er2"] = {"T": 2.5, "dq": 0.95, "er": 2}
ARMS["T3_dq0.95"] = {"T": 3, "dq": 0.95}
ARMS["T3_er2"] = {"T": 3, "er": 2}
if len(sys.argv) > 1 and sys.argv[1].startswith("{"):
    import json; ARMS = {"base": {}, "T1.5": {"T": 1.5}, **json.loads(sys.argv[1])}
elif len(sys.argv) > 1:
    ARMS = {k: v for k, v in ARMS.items() if k == "base" or any(s in k for s in sys.argv[1:])}
rows = []
for cty in ["india", "us"]:
    x0 = d.filter(pl.col("country_n") == cty)
    for st in ["lo", "ic"]:
        fb = None
        for nm, kw in ARMS.items():
            e = M.decide(tf(x0, x0[f"pA_{st}"].cast(pl.Float64), **kw), cfg, floor=0.6)
            f, k, em = score(e, cty)
            if fb is None: fb = f; fb15 = None
            if nm == "T1.5": fb15 = f
            dd = f - fb
            rows.append(dict(cty=cty, st=st, arm=nm, F=f.mean(), d=dd.mean(), se=dd.std(ddof=1) / np.sqrt(5), pos=int((dd > 0).sum()), d15=(f - fb15).mean() if fb15 is not None else np.nan, k=k, empty=em))
            print(rows[-1], flush=True)
R = pl.DataFrame(rows)
R.write_parquet(f"research/r23/lo{len(sys.argv)}.parquet")
with pl.Config(tbl_rows=200, tbl_width_chars=250):
    print(R.with_columns(pl.col(c).round(5) for c in ["F", "d", "se", "d15"]).with_columns(pl.col("k").round(3), pl.col("empty").round(4)).pivot(on=["cty", "st"], index="arm", values="d"))
