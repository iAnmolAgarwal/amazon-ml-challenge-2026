import sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, "src")
import er_meta as E
import er_model as M
from common import load_config
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(250)
cfg = load_config(); cache = Path(cfg["cache_dir"]); K = ["s1_idx", "pool_idx"]


def views(split):
    v = []
    for side in ("s1", "pool"):
        x = pl.read_parquet(cache / "norm" / f"{split}_{side}.parquet", columns=["name_core", "house", "street1", "addr_norm", "country_n"]).with_row_index(f"{side}_idx").with_columns(pl.col(f"{side}_idx").cast(pl.UInt32))
        v.append(x)
    s, p = v
    s = s.with_columns(nadr=pl.len().over("country_n", "house", "street1")).with_columns(nadr=pl.when(pl.col("house").is_null() | (pl.col("house") == "")).then(None).otherwise(pl.col("nadr")))
    return s, p


def classify(d, s, p):
    d = d.join(s.select("s1_idx", "country_n", ns="name_core", hs="house", ss="street1", nadr="nadr"), on="s1_idx").join(p.select("pool_idx", np_="name_core", hp="house", sp="street1", ap="addr_norm"), on="pool_idx")
    ts, tp = pl.col("ns").fill_null("").str.split(" ").list.unique(), pl.col("np_").fill_null("").str.split(" ").list.unique()
    d = d.with_columns(sh=ts.list.set_intersection(tp).list.len(), a1=ts.list.set_difference(tp).list.len(), b1=tp.list.set_difference(ts).list.len())
    nm = (pl.when(pl.col("ns") == pl.col("np_")).then(pl.lit("same")).when(pl.col("sh") == 0).then(pl.lit("pseudo"))
          .when((pl.col("a1") == 0)).then(pl.lit("add")).when(pl.col("b1") == 0).then(pl.lit("drop"))
          .when((pl.col("a1") == 1) & (pl.col("b1") == 1)).then(pl.lit("swap1")).otherwise(pl.lit("other")))
    hs_ok, hp_ok = pl.col("hs").fill_null("") != "", pl.col("hp").fill_null("") != ""
    ad = (pl.when(pl.col("ap").fill_null("").str.strip_chars().is_in(["", "none"])).then(pl.lit("empty"))
          .when(hs_ok & hp_ok & (pl.col("hs") == pl.col("hp")) & (pl.col("ss") == pl.col("sp"))).then(pl.lit("eqHS"))
          .when(hs_ok & hp_ok & (pl.col("hs") == pl.col("hp"))).then(pl.lit("eqH"))
          .when(hs_ok & hp_ok).then(pl.lit("diffH")).otherwise(pl.lit("noH")))
    return d.with_columns(nm=nm, ad=ad, uniq=pl.when(pl.col("nadr") == 1).then(pl.lit("1")).when(pl.col("nadr").is_null()).then(pl.lit("-")).otherwise(pl.lit("2+"))).select(K + ["country_n", "nm", "ad", "uniq", "pA", "pS", "p"] + (["y"] if "y" in d.columns else []))


split = sys.argv[1]
s, p = views(split)
if split == "train":
    s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1)
    si, pi = E.ids(cache, "train")
    sp = E.stack(Path("cache/stack"), "train", si, pi, (0.02, 10), "none").rename({"ps": "pS"})
    a = pl.read_parquet(cache / "oof" / "024_keys.parquet", columns=K + ["prob"]).filter(pl.col("prob") >= 0.001).rename({"prob": "pA"})
    f = pl.read_parquet(cache / "oof" / "030l_fix.parquet", columns=K + ["prob"]).rename({"prob": "p"})
    d = a.join(sp, on=K, how="full", coalesce=True).join(f, on=K, how="full", coalesce=True)
    d = d.join(gold.with_columns(y=pl.lit(1, pl.Int8)), on=K, how="left").with_columns(pl.col("y").fill_null(0))
    em = M.decide(f.rename({"p": "prob"}), cfg, floor=0.6).select(K)
else:
    si, pi = E.ids(cache, "test")
    fr = si.filter(pl.col("country_n") == "france")
    a = pl.read_parquet(cache / "oof" / "test_024_keys.parquet").join(fr.select("s1_idx"), on="s1_idx").filter(pl.col("prob") >= 0.001).rename({"prob": "pA"})
    sp = pl.read_parquet("cache/stack/p_test_France.parquet").join(fr, on="a_id").join(pi, on="r_id").select(K + [pl.col("p").alias("pS")])
    f = pl.read_parquet(cache / "oof" / "test_029l_final.parquet").join(fr.select("s1_idx"), on="s1_idx").rename({"prob": "p"})
    d = a.join(sp, on=K, how="full", coalesce=True).join(f, on=K, how="full", coalesce=True)
    em = pl.read_parquet("research/r22/emit_029l.parquet").select(K)
d = classify(d, s, p).join(em.with_columns(em=pl.lit(1, pl.Int8)), on=K, how="left").with_columns(pl.col("em").fill_null(0))
d.write_parquet(f"research/r22/cls_{split}.parquet")
print(d.height)
