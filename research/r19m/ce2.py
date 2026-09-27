"""cache/feat/train_ce2 = train_ce + train_ce_x, test_ce2_par = test_ce_par + test_ce_x_par (first file wins on duplicate keys)."""
import polars as pl

K = ["s1_idx", "pool_idx"]
for out, a, b in [("train_ce2", "train_ce", "train_ce_x"), ("test_ce2_par", "test_ce_par", "test_ce_x_par")]:
    x = pl.read_parquet(f"cache/feat/{a}.parquet")
    y = pl.read_parquet(f"cache/feat/{b}.parquet").select(x.columns).cast(x.schema)
    z = pl.concat([x, y.join(x.select(K), on=K, how="anti")])
    assert z.select(K).is_unique().all()
    z.write_parquet(f"cache/feat/{out}.parquet")
    print(out, x.height, y.height, z.height, flush=True)
