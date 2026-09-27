"""Helpers for the extra-candidate train sample.

usage:
  python src/er_extend.py sample <n> <r> <path>                 # S1 with s1_idx % n == r (label-free) -> parquet (s1_idx)
  python src/er_extend.py union <out> <base> <extra> <old>       # cache/oof/<out> = <base> plus the pairs of <extra> not in <old>
  python src/er_extend.py parity <out> <col>=<dir> ...          # cache/feat/test_<out>: per test S1 one half model's score

union gives a main attempt whose train out-of-fold file also holds the extra candidates (keys / reverse text) of the
sampled S1, scored out-of-fold by the retrieval-route classifier (er_newpairs.py score <extra> --split train).
parity takes, for each column, the directory holding ce_A_test.parquet and ce_B_test.parquet (the two half models'
test scores) and keeps for each test S1 the score of half model crc32(S1 id) % 2 (0 -> A, 1 -> B), so every test pair
has one model's score, as every train pair has.
"""
import sys
import zlib
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
    elif cmd == "parity":
        out, specs = sys.argv[2], [a.split("=", 1) for a in sys.argv[3:]]
        ids = pl.read_parquet(cache / "norm" / "test_s1.parquet", columns=["entity_id"]).with_row_index("s1_idx")
        h = ids.select(pl.col("s1_idx").cast(pl.UInt32), h=pl.Series([zlib.crc32(x.encode()) % 2 for x in ids["entity_id"]], dtype=pl.Int32))
        res = None
        for col, d in specs:
            a = pl.read_parquet(Path(d) / "ce_A_test.parquet").rename({"ce": "a"})
            b = pl.read_parquet(Path(d) / "ce_B_test.parquet").rename({"ce": "b"})
            x = a.join(b, on=KEYS).join(h, on="s1_idx").select(KEYS + [pl.when(pl.col("h") == 0).then(pl.col("a")).otherwise(pl.col("b")).alias(col)])
            res = x if res is None else res.join(x, on=KEYS)
        res.sort(KEYS).write_parquet(cache / "feat" / f"test_{out}.parquet")
        print(f"wrote test_{out}: {res.height:,} pairs, columns {res.columns[2:]}")


if __name__ == "__main__":
    main()
