"""Helpers for the extra-candidate train sample.

usage:
  python src/er_extend.py sample <n> <r> <path>                 # S1 with s1_idx % n == r (label-free) -> parquet (s1_idx)
  python src/er_extend.py union <out> <base> <extra> <old>       # cache/oof/<out> = <base> plus the pairs of <extra> not in <old>

union gives a main attempt whose train out-of-fold file also holds the extra candidates (keys / reverse text) of the
sampled S1, scored out-of-fold by the retrieval-route classifier (er_newpairs.py score <extra> --split train).
"""
import sys
from pathlib import Path

import polars as pl

from common import load_config

KEYS = ["s1_idx", "pool_idx"]


def main():
    cache = Path(load_config()["cache_dir"])
    cmd = sys.argv[1]
    if cmd == "sample":
        n, r, path = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        m = pl.read_parquet(cache / "norm" / "train_s1.parquet", columns=["entity_id"]).height
        pl.DataFrame({"s1_idx": pl.int_range(0, m, eager=True)}).filter(pl.col("s1_idx") % n == r).with_columns(pl.col("s1_idx").cast(pl.UInt32)).write_parquet(path)
        print(f"wrote {path}")
    elif cmd == "union":
        out, base, extra, old = sys.argv[2:6]
        b = pl.read_parquet(cache / "oof" / f"{base}.parquet")
        e = pl.read_parquet(cache / "oof" / f"{extra}.parquet", columns=KEYS + ["prob"])
        new = e.join(pl.read_parquet(cache / "oof" / f"{old}.parquet", columns=KEYS), on=KEYS, how="anti").select(b.columns).cast(b.schema)
        pl.concat([b, new]).sort(KEYS).write_parquet(cache / "oof" / f"{out}.parquet")
        print(f"wrote {out}: {b.height:,} + {new.height:,} extra pairs")


if __name__ == "__main__":
    main()
