import argparse
import re
from pathlib import Path

import polars as pl

from common import ROOT, load_attempt, load_config, load_test, load_train, read_table


def next_sub_n(subdir):
    nums = [int(m.group(1)) for p in subdir.glob("*.csv") if (m := re.match(r"(\d+)_", p.name))]
    return max(nums, default=0) + 1


def validate(sub, sample, id_col):
    errs = []
    if list(sub.columns) != list(sample.columns):
        errs.append(f"columns {sub.columns} != sample {sample.columns}")
    if len(sub) != len(sample):
        errs.append(f"rows {len(sub)} != sample {len(sample)}")
    if id_col in sub.columns and id_col in sample.columns:
        if sub[id_col].n_unique() != len(sub):
            errs.append("duplicate ids")
        if set(sub[id_col].to_list()) != set(sample[id_col].to_list()):
            errs.append("id set differs from sample")
    for c in sub.columns:
        if sub[c].null_count():
            errs.append(f"nulls in {c}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attempt_id")
    ap.add_argument("--config")
    ap.add_argument("--no-validate", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    mod = load_attempt(a.attempt_id)
    train, test = load_train(cfg), load_test(cfg)
    pred = mod.fit_predict(train, test, cfg)
    sample_path = Path(cfg["data_dir"]) / cfg["sample_submission"]
    sample = read_table(sample_path) if sample_path.exists() else None
    pred_col = [c for c in sample.columns if c != cfg["id_col"]][0] if sample is not None else cfg["target_col"]
    sub = pl.DataFrame({cfg["id_col"]: test[cfg["id_col"]], pred_col: list(pred)})
    if sample is not None:
        sub = sample.select(cfg["id_col"]).join(sub, on=cfg["id_col"], how="left")
        if not a.no_validate:
            errs = validate(sub, sample, cfg["id_col"])
            if errs:
                raise SystemExit("submission invalid: " + "; ".join(errs))
    subdir = ROOT / "submissions"
    subdir.mkdir(exist_ok=True)
    out = subdir / f"{next_sub_n(subdir):02d}_{a.attempt_id}.csv"
    sub.write_csv(out)
    print(f"wrote {out.relative_to(ROOT)} ({len(sub)} rows, validated={sample is not None and not a.no_validate})")


if __name__ == "__main__":
    main()
