"""Experiment engine: cached pairs -> two-stage LightGBM (grouped 5-fold OOF) -> decision layer -> metrics.

  python src/run_experiment.py --config configs/<name>.yaml

Config keys: name, pairs_cfg (blocking config dict), train_blocks (list of country strings present in train),
eval_blocks (blocks to score; blocks not in train_blocks are scored with the averaged fold models = holdout, e.g. LOCO),
sample_rate (entity sampling rate for model training), rounds, params (LightGBM overrides), drop_features,
rate_features (true: chain/frequency counts as rates per million S1 of the block), decision.
Writes experiments/<ts>_<name>/{config.yaml, metrics.json, errors.tsv} and appends to experiments/results.tsv (created on first run).
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(__file__))
import memguard  # noqa: E402
from io_utils import ROOT, DATA, read_tsv  # noqa: E402
from normalize import load_norm  # noqa: E402
from pairs import out_dir, safe, DEFAULT  # noqa: E402
from model import SCALE  # noqa: E402
from model import (N_FOLDS, fold_of, feature_cols, load_keys, sample_rows, gather, train_folds,  # noqa
                   predict_stream, context)
from metric import entity_scores  # noqa: E402
from decision import expected_f_select  # noqa: E402

EXP = os.path.join(ROOT, "experiments")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, f"[rss={memguard.rss_gb():.2f}GB]", flush=True)


def block_info(block):
    """S1 ids of the block (same order as pairs build), their fold and true-match counts."""
    ids = load_norm("train", 1, ["id"], country=block).id.values
    fold = fold_of(ids)
    gt = read_tsv(os.path.join(DATA, "train", "train_ground_truth.tsv"))
    cnt = gt.matched_entity_ids.map(lambda x: 0 if x == "" else x.count(",") + 1)
    n_true = pd.Series(cnt.values, index=gt.source1_entity_id.values).reindex(ids).fillna(0).values.astype(float)
    return ids, fold, n_true


def decide(a, r, p, y, n_ent, n_true, rule):
    """Exclusive assignment (record -> best S1) then selection rule. Returns selected mask."""
    d = pd.DataFrame({"a": a, "r": r, "p": p})
    best = d.groupby("r").p.transform("max").values
    pe = np.where(p >= best, p, 0.0).astype(np.float32)
    if rule["type"] == "threshold":
        return pe > rule["t"]
    d["p"] = pe
    return expected_f_select(d, "p", min_p=rule.get("min_p", 0.0))


def score(a, sel, y, n_ent, n_true, fold=None):
    F, P, R = entity_scores(n_ent, a[sel], y[sel].astype(float), n_true)
    out = {"macro_f05": float(F.mean()), "mean_P": float(P.mean()), "mean_R": float(R.mean()),
           "singleton_acc": float(F[n_true == 0].mean()) if (n_true == 0).any() else None}
    if fold is not None:
        out["fold_f05"] = [float(F[fold == k].mean()) for k in range(N_FOLDS)]
    return out, F


def run(cfg):
    t0 = time.time()
    pcfg = dict(DEFAULT, **cfg.get("pairs_cfg", {}))
    pdir = out_dir("train", pcfg)
    train_blocks, eval_blocks = cfg["train_blocks"], cfg.get("eval_blocks", cfg["train_blocks"])
    work = os.path.join(ROOT, "cache", "exp_" + cfg["name"])
    os.makedirs(work, exist_ok=True)
    path = {b: os.path.join(pdir, safe(b) + ".parquet") for b in set(train_blocks) | set(eval_blocks)}
    feats = [c for c in feature_cols(path[train_blocks[0]]) if c not in set(cfg.get("drop_features", []))]
    info = {b: block_info(b) for b in path}
    if cfg.get("rate_features"):                         # counts -> rates per million S1 of the block (model.SCALE)
        for b in path:
            SCALE[path[b]] = 1e6 / len(info[b][0])
    rate, rounds, params = cfg.get("sample_rate", 0.25), cfg.get("rounds", 600), cfg.get("params", {})

    # ---------------- stage 1 ----------------
    Xs, ys, fs, ess, rows = [], [], [], [], {}
    for b in train_blocks:
        k = load_keys(path[b])
        m = sample_rows(k["a"], info[b][1], rate, seed=1)
        rows[b] = m
        X = gather(path[b], feats, m)
        Xs.append(X); ys.append(k["y"][m]); fs.append(info[b][1][k["a"][m]])
        ess.append((fold_of(info[b][0], n=10, salt=b"early_stop") == 0)[k["a"][m]])
        del k
    X, y, f = pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(fs)
    es = np.concatenate(ess) if cfg.get("honest_early_stop", True) else None
    del Xs, ys, fs
    log(f"stage1 sample rows={len(y)} pos={y.mean():.3f}")
    m1 = []
    gen = train_folds(X, y, f, rounds, params, es=es)
    del X  # the generator owns the only reference and frees it once binned
    for k, m in gen:
        m1.append(m)
        log(f"  s1 fold {k} iters={m.best_iteration}")
    gc.collect()
    for b in path:
        k = load_keys(path[b])
        if b in train_blocks:
            p1 = predict_stream(path[b], feats, m1, row_fold=info[b][1][k["a"]])
        else:
            p1 = np.mean([predict_stream(path[b], feats, m) for m in m1], axis=0).astype(np.float32)
        np.save(os.path.join(work, f"p1_{safe(b)}.npy"), p1)
        log(f"  p1 {b} done")
        del k, p1
        gc.collect()

    # ---------------- stage 2 ----------------
    Xs, ys = [], []
    for b in train_blocks:
        k = load_keys(path[b])
        p1 = np.load(os.path.join(work, f"p1_{safe(b)}.npy"))
        ctx = context(k["a"], k["r"], p1, store=work)
        m = rows[b]
        X = gather(path[b], feats, m, extra=ctx)
        Xs.append(X); ys.append(k["y"][m])
        del k, p1, ctx
        gc.collect()
    X, y = pd.concat(Xs, ignore_index=True), np.concatenate(ys)
    del Xs, ys
    feats2 = list(X.columns)
    m2 = []
    gen = train_folds(X, y, f, rounds, params, es=es)
    del X
    for k, m in gen:
        m2.append(m)
        log(f"  s2 fold {k} iters={m.best_iteration}")
    imp = pd.Series(np.mean([mm.feature_importance("gain") for mm in m2], axis=0), index=feats2)
    if cfg.get("save_models"):
        md = os.path.join(ROOT, "models", cfg["name"])
        os.makedirs(md, exist_ok=True)
        for i, (a1, a2) in enumerate(zip(m1, m2)):
            a1.save_model(os.path.join(md, f"s1_fold{i}.txt"))
            a2.save_model(os.path.join(md, f"s2_fold{i}.txt"))
        json.dump({"feats1": feats, "feats2": feats2, "pairs_cfg": pcfg, "rate_features": bool(cfg.get("rate_features"))},
                  open(os.path.join(md, "meta.json"), "w"))
    del y, f, gen
    gc.collect()

    # ---------------- predict + decide + score ----------------
    rules = cfg.get("decision", [{"type": "threshold", "t": 0.5}, {"type": "expected_f"}])
    res = {"name": cfg["name"], "blocks": {}}
    errs = []
    for b in eval_blocks:
        k = load_keys(path[b])
        ids, fold, n_true = info[b]
        p1 = np.load(os.path.join(work, f"p1_{safe(b)}.npy"))
        ctx = context(k["a"], k["r"], p1, store=work)
        base = [c for c in feats2 if c not in ctx]
        if b in train_blocks:
            p2 = predict_stream(path[b], base, m2, row_fold=fold[k["a"]], extra=ctx)
        else:
            p2 = np.mean([predict_stream(path[b], base, m, extra=ctx) for m in m2], axis=0).astype(np.float32)
        np.save(os.path.join(work, f"p2_{safe(b)}.npy"), p2)
        del ctx
        gc.collect()
        n_ent = len(ids)
        in_cand = np.bincount(k["a"], weights=k["y"], minlength=n_ent)
        br = {"n_entities": n_ent, "n_pairs": int(len(p2)),
              "recall_ceiling": float(in_cand.sum() / max(n_true.sum(), 1)),
              "holdout": b not in train_blocks}
        for rule in rules:
            sel = decide(k["a"], k["r"], p2, k["y"], n_ent, n_true, rule)
            key = rule["type"] + (f"_{rule['t']}" if "t" in rule else "") + (f"_min{rule['min_p']}" if "min_p" in rule else "")
            br[key], F = score(k["a"], sel, k["y"], n_ent, n_true, fold if b in train_blocks else None)
            if key == cfg.get("error_rule", "expected_f"):
                fp = sel & (k["y"] == 0)
                fn = (~sel) & (k["y"] == 1)
                idx = np.flatnonzero(fp | fn)
                idx = idx[np.random.RandomState(0).permutation(len(idx))[:3000]]
                errs.append(pd.DataFrame({"block": b, "kind": np.where(fp[idx], "FP", "FN"),
                                          "a": k["a"][idx], "r": k["r"][idx], "p2": p2[idx]}))
        res["blocks"][b] = br
        log(f"  eval {b}: " + json.dumps({kk: (v["macro_f05"] if isinstance(v, dict) else v) for kk, v in br.items()}))
        del k, p2, p1
        gc.collect()
    res["runtime_s"] = round(time.time() - t0)
    res["peak_gb"] = round(memguard.peak_gb(), 2)
    res["top_features"] = imp.sort_values(ascending=False).head(25).round(0).to_dict()
    return res, pd.concat(errs, ignore_index=True) if errs else None


def main():
    memguard.start()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    res, errs = run(cfg)
    ts = time.strftime("%Y%m%d_%H%M%S")
    d = os.path.join(EXP, f"{ts}_{cfg['name']}")
    os.makedirs(d, exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(d, "config.yaml"), "w"))
    json.dump(res, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    if errs is not None:
        errs.to_csv(os.path.join(d, "errors.tsv"), sep="\t", index=False)
    try:
        gh = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        gh = "nogit"
    rows = []
    for b, br in res["blocks"].items():
        for key, v in br.items():
            if isinstance(v, dict):
                rows.append({"time": ts, "name": cfg["name"], "git": gh, "block": b, "holdout": br["holdout"],
                             "rule": key, "macro_f05": round(v["macro_f05"], 5), "P": round(v["mean_P"], 4),
                             "R": round(v["mean_R"], 4), "singleton_acc": v["singleton_acc"],
                             "fold_std": round(float(np.std(v["fold_f05"])), 5) if "fold_f05" in v else None,
                             "recall_ceiling": round(br["recall_ceiling"], 4), "runtime_s": res["runtime_s"]})
    rt = os.path.join(EXP, "results.tsv")
    pd.DataFrame(rows).to_csv(rt, sep="\t", index=False, mode="a", header=not os.path.exists(rt))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
