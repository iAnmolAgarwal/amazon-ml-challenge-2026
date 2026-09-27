import sys
import polars as pl
split = sys.argv[1]
K = ["s1_idx", "pool_idx"]
LEG = {"sarl", "sas", "sa", "eurl", "sasu", "sci", "snc", "ei", "llc", "inc", "incorporated", "ltd", "limited", "pvt", "private", "public", "corp", "corporation", "co", "company", "lp", "llp", "plc", "pllc", "pc"}
def toks(c):
    return pl.col(c).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_lowercase().str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars().str.split(" ")
def side(f, idx, rows=None):
    x = pl.concat([pl.read_parquet(f"cache/raw/{split}_{s}.parquet", columns=["business_name", "business_address"]) for s in f]).with_row_index(idx).with_columns(pl.col(idx).cast(pl.UInt32))
    return x.with_columns(n=toks("business_name"), a=toks("business_address")).drop("business_name", "business_address")
s = side(["source1"], "s1_idx"); p = side(["source2", "source3"], "pool_idx")
d = pl.read_parquet(f"research/r22/cls_{split}.parquet").select(K + ["country_n", "pA", "pS", "p", "em"] + (["y"] if split == "train" else []))
d = d.join(s, on="s1_idx").join(p, on="pool_idx", suffix="_p")
d = d.with_columns(ns=pl.col("n").list.unique(), np=pl.col("n_p").list.unique())
d = d.with_columns(a1=pl.col("ns").list.set_difference("np"), b1=pl.col("np").list.set_difference("ns"))
hs = lambda c: pl.col(c).list.eval(pl.element().filter(pl.element().str.contains(r"^\d{1,5}$"))).list.first().str.replace(r"^0+", "").cast(pl.Int64, strict=False)
sw = lambda c: pl.col(c).list.eval(pl.element().filter((pl.element().str.len_chars() > 3) & ~pl.element().str.contains(r"\d")))
fq = s.join(pl.read_parquet(f"cache/norm/{split}_s1.parquet", columns=["country_n"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32)), on="s1_idx").select("country_n", w="a").explode("w").group_by("country_n", "w").len().sort("len", descending=True).group_by("country_n").head(150)
stop = fq["w"].unique().to_list()
d = d.with_columns(h1=hs("a"), h2=hs("a_p"), w1=sw("a").list.set_difference(pl.lit(stop)), w2=sw("a_p").list.set_difference(pl.lit(stop)))
d = d.with_columns(so=pl.col("w1").list.set_intersection("w2").list.len())
d = d.with_columns(dh=(pl.col("h2") - pl.col("h1")).abs())
d = d.with_columns(hr=pl.when(pl.col("h1").is_null() | pl.col("h2").is_null()).then(pl.lit("na")).when(pl.col("dh") == 0).then(pl.lit("eq")).when(pl.col("dh").is_in([1, 2, 3, 4, 5, 7, 9, 11, 13, 21])).then(pl.lit("shift")).when(pl.col("dh").is_in([6, 8, 10, 12, 14, 15, 16])).then(pl.lit("ctrl")).otherwise(pl.lit("far")))
one = (pl.col("a1").list.len() == 1) & (pl.col("b1").list.len() == 1)
x, z = pl.col("a1").list.first(), pl.col("b1").list.first()
d = d.with_columns(kind=pl.when((pl.col("a1").list.len() == 0) & (pl.col("b1").list.len() == 0)).then(pl.lit("same"))
    .when(one & x.is_in(list(LEG)) & z.is_in(list(LEG))).then(pl.lit("legalswap"))
    .when(one & ~x.is_in(list(LEG)) & ~z.is_in(list(LEG)) & (x.str.len_chars() > 2) & (z.str.len_chars() > 2) & ~x.str.contains(r"\d") & ~z.str.contains(r"\d")).then(pl.lit("wordswap"))
    .otherwise(pl.lit("other")))
d = d.filter((pl.col("so") > 0) & (pl.col("kind") != "other"))
d.select(K + ["country_n", "kind", "hr", "dh", "pA", "pS", "p", "em"] + (["y"] if split == "train" else []) + [pl.col("a1").list.first().alias("wS"), pl.col("b1").list.first().alias("wP")]).write_parquet(f"research/r22/sw2_{split}.parquet")
agg = [pl.len().alias("n"), pl.col("em").mean().alias("em")] + ([pl.col("y").mean().alias("y")] if split == "train" else [pl.col("pA").mean().alias("mA"), pl.col("pS").mean().alias("mS")])
t = d.group_by("country_n", "kind", "hr").agg(agg).sort("country_n", "kind", "hr").with_columns(pl.col(pl.Float64, pl.Float32).round(3))
for r in t.iter_rows(): print(" ".join(str(v) for v in r))
