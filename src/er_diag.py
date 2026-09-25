"""Diagnostics for an evaluated attempt (uses cache/oof/<attempt>.parquet written by er_eval.py).

usage: python src/er_diag.py <attempt_id> [--test]

Train (labels): stage losses (gold edges lost to retrieval / ownership / k selection), macro F0.5 by
country and by true-match count, calibration table, singleton check (actual vs mean prod(1-p)),
sum of probabilities vs true count, ownership conflicts (records owned but not emitted, runner-up gold).
--test: label-free test statistics per country (no rows printed): candidate counts, empty-address and
house/postal/street parse rates, non-Latin rates, name-frequency spread.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

import er_model as M
from common import ROOT, load_config


def train_diag(cfg, attempt):
    cache = Path(cfg["cache_dir"])
    oof = pl.read_parquet(cache / "oof" / f"{attempt}.parquet")
    s1 = M.load_s1(cfg, "train")
    gold = M.load_gold(cfg, s1).with_columns(y=pl.lit(1, pl.Int8))
    p = oof.join(gold, on=["s1_idx", "pool_idx"], how="left").with_columns(pl.col("y").fill_null(0)).join(s1.select("s1_idx", "country_n"), on="s1_idx")
    out = {}
    bins = [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1.0001]
    cal = p.with_columns(b=pl.col("prob").cut(bins[1:-1])).group_by("b").agg(pl.len(), pl.col("prob").mean().alias("mean_p"), pl.col("y").mean().alias("actual")).sort("b")
    out["calibration"] = cal.to_dicts()
    owner = p.sort(["pool_idx", "prob", "s1_idx"], descending=[False, True, False]).unique("pool_idx", keep="first", maintain_order=True)
    pred = M.decide(oof, cfg).with_columns(emitted=pl.lit(1, pl.Int8))
    t = gold.group_by("s1_idx").len().rename({"len": "t"})
    retrieved = p.filter(pl.col("y") == 1).group_by("s1_idx").len().rename({"len": "c"})
    owned_gold = owner.filter(pl.col("y") == 1).group_by("s1_idx").len().rename({"len": "o"})
    emitted_gold = pred.join(gold, on=["s1_idx", "pool_idx"], how="semi").group_by("s1_idx").len().rename({"len": "e"})
    n_gold = gold.height
    out["stage_loss_gold_edges"] = {
        "gold_edges": n_gold,
        "lost_to_retrieval": n_gold - int(retrieved["c"].sum()),
        "lost_to_ownership": int(retrieved["c"].sum() - owned_gold["o"].sum()),
        "lost_to_selection": int(owned_gold["o"].sum() - emitted_gold["e"].sum()),
        "false_positive_edges": int(pred.height - emitted_gold["e"].sum()),
    }
    ent = s1.select("s1_idx", "country_n").join(t, on="s1_idx", how="left").fill_null(0)
    agg = p.group_by("s1_idx").agg(pl.col("prob").sum().alias("sum_p"), (1 - pl.col("prob")).log().sum().exp().alias("p0"))
    ent = ent.join(agg, on="s1_idx", how="left").with_columns(pl.col("sum_p").fill_null(0), pl.col("p0").fill_null(1))
    out["singleton_check"] = ent.group_by("country_n").agg((pl.col("t") == 0).mean().alias("actual_singleton_rate"), pl.col("p0").mean().alias("mean_prod_1_minus_p"), pl.col("t").mean().alias("mean_t"), pl.col("sum_p").mean().alias("mean_sum_p")).sort("country_n").to_dicts()
    notemit = owner.join(pred, on=["s1_idx", "pool_idx"], how="anti")
    out["owned_not_emitted_per_s1"] = notemit.height / s1.height
    runner = p.sort(["pool_idx", "prob"], descending=[False, True]).with_columns(r=pl.int_range(pl.len()).over("pool_idx")).filter(pl.col("r") == 1)
    rg = notemit.select("pool_idx").join(runner, on="pool_idx").filter(pl.col("y") == 1)
    out["notemitted_runner_up_is_gold"] = {"count": rg.height, "median_runner_p": float(rg["prob"].median() or 0)}
    ef = M.per_entity_f05(pred.select("s1_idx", "pool_idx"), gold.select("s1_idx", "pool_idx"), s1.select("s1_idx", "country_n"))
    out["f05_by_country"] = ef.group_by("country_n").agg(pl.col("f05").mean(), pl.len()).sort("country_n").to_dicts()
    out["f05_by_true_matches"] = ef.group_by(pl.col("t").clip(0, 6)).agg(pl.col("f05").mean(), (pl.col("k").cast(pl.Int64) - pl.col("t").cast(pl.Int64)).mean().alias("k_minus_t"), pl.len()).sort("t").to_dicts()
    return out


def test_diag(cfg):
    cache = Path(cfg["cache_dir"])
    n = pl.read_parquet(cache / "norm" / "test_s1.parquet").with_row_index("s1_idx")
    pool = pl.read_parquet(cache / "norm" / "test_pool.parquet", columns=["country_n", "addr_norm", "house", "postal", "street1", "name_core"])
    cand = pl.read_parquet(cache / "cand" / "test_cand.parquet", columns=["s1_idx", "country_n"])
    rates = lambda df: df.group_by("country_n").agg(
        pl.len(), (pl.col("addr_norm") == "").mean().alias("empty_addr"), (pl.col("house") != "").mean().alias("house_parsed"),
        (pl.col("postal") != "").mean().alias("postal_parsed"), (pl.col("street1") != "").mean().alias("street_parsed"),
        pl.col("name_core").str.contains(r"[ऀ-෿]").mean().alias("nonlatin_name")).sort("country_n")
    per = cand.group_by("s1_idx", "country_n").len()
    counts = n.select("s1_idx", "country_n").join(per.select("s1_idx", "len"), on="s1_idx", how="left").fill_null(0)
    return {
        "s1": rates(n).to_dicts(), "pool": rates(pool).to_dicts(),
        "candidates_per_s1": counts.group_by("country_n").agg(pl.col("len").mean().alias("mean"), pl.col("len").quantile(0.01).alias("p01"), pl.col("len").quantile(0.99).alias("p99"), (pl.col("len") == 0).mean().alias("zero")).sort("country_n").to_dicts(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attempt_id")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    res = test_diag(cfg) if a.test else train_diag(cfg, a.attempt_id)
    path = ROOT / "attempts" / a.attempt_id / ("diag_test.json" if a.test else "diag.json")
    path.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
