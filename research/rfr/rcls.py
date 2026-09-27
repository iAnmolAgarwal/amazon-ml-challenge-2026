import polars as pl
K = ["s1_idx", "pool_idx"]


def toks(c):
    return pl.col(c).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_lowercase().str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars().str.split(" ")


def views(split):
    s = pl.read_parquet(f"cache/raw/{split}_source1.parquet", columns=["business_address"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32), t=toks("business_address")).drop("business_address")
    s = s.join(pl.read_parquet(f"cache/norm/{split}_s1.parquet", columns=["country_n", "name_core"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32)), on="s1_idx")
    p = pl.concat([pl.read_parquet(f"cache/raw/{split}_source{i}.parquet", columns=["business_address"]) for i in (2, 3)]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32), t=toks("business_address")).drop("business_address")
    p = p.join(pl.read_parquet(f"cache/norm/{split}_pool.parquet", columns=["name_core"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32)), on="pool_idx")
    stop = s.select("country_n", w="t").explode("w").group_by("country_n", "w").len().sort("len", descending=True).group_by("country_n").head(150)["w"].unique().to_list()
    def parse(x):
        return x.with_columns(h=pl.col("t").list.eval(pl.element().filter(pl.element().str.contains(r"^\d{1,5}$"))).list.first().str.replace(r"^0+", ""),
                              w=pl.col("t").list.eval(pl.element().filter((pl.element().str.len_chars() > 2) & ~pl.element().str.contains(r"\d"))).list.set_difference(pl.lit(stop)).list.unique().list.sort(),
                              n=pl.col("name_core").fill_null("").str.split(" ").list.unique()).drop("t", "name_core")
    s, p = parse(s), parse(p)
    s = s.with_columns(nk=pl.when(pl.col("h").is_not_null() & (pl.col("w").list.len() > 0)).then(pl.len().over("country_n", "h", pl.col("w").list.join(" "))).otherwise(None))
    return s, p


def flag(d, s, p):
    """Class R: pool name shares no core token with the S1 name, same house number, a shared street word, S1 alone at its address."""
    x = d.select(K).join(s, on="s1_idx").join(p, on="pool_idx", suffix="_p")
    x = x.with_columns(R=(pl.col("n").list.set_intersection("n_p").list.len() == 0) & (pl.col("h") == pl.col("h_p")) & (pl.col("w").list.set_intersection("w_p").list.len() > 0) & (pl.col("nk") == 1))
    return d.join(x.select(K + ["R"]), on=K, how="left").with_columns(pl.col("R").fill_null(False))
