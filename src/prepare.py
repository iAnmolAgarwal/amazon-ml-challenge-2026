from pathlib import Path

import polars as pl

from common import load_config
from er import raw_dir
from harness_util import load_split, read_tsv


def main():
    cfg = load_config()
    s1, pool = load_split(cfg, "train")
    gt = read_tsv(raw_dir(cfg) / "train" / "train_ground_truth.tsv").rename({"source1_entity_id": "entity_id"})
    tr = s1.join(gt, on="entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
    n = pl.col("matched_entity_ids").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len()
    tr = tr.with_columns(n.alias("n_match")).with_columns(
        (pl.col("country") + "_" + pl.col("n_match").clip(0, 3).cast(pl.Utf8)).alias("strat")
    )
    te, te_pool = load_split(cfg, "test")
    out = Path(cfg["data_dir"])
    tr.write_parquet(out / cfg["train_file"])
    te.write_parquet(out / cfg["test_file"])
    print(f"train S1 {tr.shape}, pool {pool.shape}; test S1 {te.shape}, pool {te_pool.shape}")
    print(tr.group_by("strat").len().sort("strat"))


if __name__ == "__main__":
    main()
