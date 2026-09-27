import sys
sys.path.insert(0, "src")
import polars as pl
import er_model as M
from common import load_config
cfg = load_config(); K = ["s1_idx", "pool_idx"]
LF = ["llc", "inc", "corp", "ltd", "lp", "llp", "pllc", "pc", "corporation", "incorporated", "pvt", "private", "limited", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scop", "selarl", "gmbh", "plc", "co", "company"]
def names(split):
    s = pl.read_parquet(f"cache/norm/{split}_s1.parquet", columns=["entity_id", "country_n"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
    s = s.join(pl.read_parquet(f"cache/raw/{split}_source1.parquet", columns=["entity_id", "business_name"]), on="entity_id", how="left").drop("entity_id")
    p = pl.read_parquet(f"cache/norm/{split}_pool.parquet", columns=["entity_id"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32))
    p = p.join(pl.concat([pl.read_parquet(f"cache/raw/{split}_source{i}.parquet", columns=["entity_id", "business_name"]) for i in (2, 3)]), on="entity_id", how="left").drop("entity_id").rename({"business_name": "pn"})
    return s, p
tok = lambda c: pl.col(c).fill_null("").str.to_lowercase().str.replace_all(r"\.", "").str.replace_all(r"[^\w\s]", " ").str.split(" ").list.eval(pl.element().filter(pl.element().is_in(LF))).list.unique().list.sort()
def tagit(d, s, p):
    d = d.join(s, on="s1_idx").join(p, on="pool_idx").with_columns(ls=tok("business_name"), lq=tok("pn"))
    return d.with_columns(swap=(pl.col("ls").list.len() > 0) & (pl.col("lq").list.len() > 0) & (pl.col("ls").list.set_intersection(pl.col("lq")).list.len() == 0))
s, p = names("train")
s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1)
d = tagit(M.decide(pl.read_parquet("cache/oof/030p_fix.parquet"), cfg, floor=0.6).select(K + ["prob"]), s, p).join(gold.with_columns(y=pl.lit(1)), on=K, how="left").with_columns(pl.col("y").fill_null(0))
print(d.group_by("country_n", "swap").agg(n=pl.len(), prec=pl.col("y").mean()).sort("country_n", "swap"))
fp = d.filter(pl.col("y") == 0)
print("train FP swap share", fp.group_by("country_n").agg(pl.col("swap").mean()))
s, p = names("test")
bb = pl.read_parquet("cache/oof/test_006_stage2.parquet", columns=K + ["prob"]).rename({"prob": "pb"})
t = pl.read_parquet("cache/oof/test_029f8b_final.parquet").join(bb, on=K, how="left").filter(pl.col("pb").is_null() | (pl.col("pb") >= 0.001)).drop("pb")
d = tagit(M.decide(t, cfg, floor=0.6).select(K + ["prob"]), s, p).with_columns(pb=pl.when(pl.col("prob") < 0.9).then(pl.lit("a")).when(pl.col("prob") < 0.99).then(pl.lit("b")).when(pl.col("prob") < 0.999).then(pl.lit("c")).otherwise(pl.lit("d")))
print(d.group_by("country_n", "pb").agg(n=pl.len(), swap=pl.col("swap").mean()).sort("country_n", "pb"))
pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_rows(30)
print(d.filter((pl.col("country_n") == "france") & pl.col("swap")).sample(20, seed=2).select("prob", "business_name", "pn"))
