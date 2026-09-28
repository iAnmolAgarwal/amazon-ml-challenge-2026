import argparse
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from common import load_config
from harness_util import load_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["kfold", "stratified", "group"])
    ap.add_argument("--n-folds", type=int)
    ap.add_argument("--group-col")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    cv = cfg["cv"]
    kind = a.type or cv["type"]
    n = a.n_folds or cv["n_folds"]
    group_col = a.group_col or cv.get("group_col")
    df = load_train(cfg)
    ids = df[cfg["id_col"]].to_numpy()
    sc = cv.get("strat_col") or cfg["target_col"]
    y = df[sc].to_numpy() if sc in df.columns else None
    folds = np.zeros(len(df), dtype=np.int32)
    if kind == "stratified":
        it = StratifiedKFold(n, shuffle=True, random_state=a.seed).split(ids, y)
    elif kind == "group":
        g = df[group_col].to_numpy()
        rng = np.random.RandomState(a.seed)
        ug = np.unique(g)
        perm = {v: i for i, v in enumerate(rng.permutation(ug))}
        it = GroupKFold(n).split(ids, groups=np.array([perm[v] for v in g]))
    else:
        it = KFold(n, shuffle=True, random_state=a.seed).split(ids)
    for f, (_, va) in enumerate(it):
        folds[va] = f
    out = Path(cfg["data_dir"]) / "folds.csv"
    pl.DataFrame({cfg["id_col"]: ids, "fold": folds}).write_csv(out)
    print(f"wrote {out}: {kind}, {n} folds, seed {a.seed}, sizes {np.bincount(folds).tolist()}")


if __name__ == "__main__":
    main()
