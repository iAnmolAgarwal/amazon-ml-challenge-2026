import sys, types
sys.path.insert(0, "src")
import numpy as np
import polars as pl
import lightgbm as lgb
import er_model as M
import er_fix as F
from common import load_attempt, load_config
from pathlib import Path
cfg = load_config(); cache = Path(cfg["cache_dir"]); K = ["s1_idx", "pool_idx"]
b = load_attempt("030q_fix")
mod = types.SimpleNamespace(**{k: getattr(b, k) for k in dir(b) if not k.startswith("_")})
mod.NONEMPTY = "full"; mod.PARAMS = dict(mod.PARAMS, num_threads=6)
s1 = M.load_s1(cfg, "train"); gold = M.load_gold(cfg, s1)
oof = pl.read_parquet(cache / "oof" / "027p_meta.parquet", columns=K + ["prob"])
keys, lp, X, names = F.build(cache, "train", oof, mod, s1.select("s1_idx", "country_n"))
y = keys.join(gold.with_columns(y=pl.lit(1, pl.Int8)), on=K, how="left", maintain_order="left")["y"].fill_null(0).to_numpy()
m = lgb.train(mod.PARAMS, lgb.Dataset(X, y, init_score=lp, feature_name=names), mod.ROUNDS)
tst = pl.read_parquet(cache / "oof" / "test_029f5_final.parquet", columns=K + ["prob"])
tk, tlp, TX, tn = F.build(cache, "test", tst, mod, M.load_s1(cfg, "test").select("s1_idx", "country_n"), france=True)
tp = 1 / (1 + np.exp(-(tlp + m.predict(TX, raw_score=True))))
F.merge(tst, tk, tp).write_parquet(cache / "oof" / "test_r23_full.parquet")
print("done", len(tk), flush=True)
