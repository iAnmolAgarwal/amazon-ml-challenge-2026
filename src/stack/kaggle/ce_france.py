"""Kaggle kernel: cross-country cross-encoder for countries without training labels (France), with LOCO scoring.

One model per training country (India, US; one GPU each). Each model scores
- the other country's band (train_band.parquet) AND its extra LOCO blend-band pairs (extra_heldout.parquet), so the
  leave-one-country-out test of the France configuration needs no median fill;
- the France blend band (france_band.parquet).
CE_AUG=1 (robustness package, provided rows only, no external text): on 30 % of training pairs a French-style rewrite
of either side (street words rue/av/bd/route, legal forms SARL/SAS/EURL, accents on e, "No."/"#" number prefixes
dropped), and the two records' order swapped on 50 % of pairs; France pairs are then scored in both orders and averaged.
Outputs: scores_<TAG>_<train>_heldout.parquet, scores_<TAG>_<train>_france.parquet, metrics_<TAG>_<train>.json.
Only aggregate metrics are printed. Model: intfloat/multilingual-e5-base (MIT, 278M), fine-tuned on provided pairs only.
"""
import glob
import json
import math
import os
import random
import re
import subprocess
import sys
import time

MODEL = os.environ.get("CE_MODEL", "intfloat/multilingual-e5-base")
REVISION = os.environ.get("CE_REVISION", "d128750597153bb5987e10b1c3493a34e5a4502a")
TAG = os.environ.get("CE_TAG", "e5bc")
AUG = os.environ.get("CE_AUG", "0") == "1"
MAX_LEN = int(os.environ.get("CE_MAXLEN", "128"))
BATCH = int(os.environ.get("CE_BATCH", "64"))
LR = float(os.environ.get("CE_LR", "3e-5"))
TRAIN_MIN = float(os.environ.get("CE_TRAIN_MIN", "75"))      # per-model training time budget (minutes)
OUT = os.environ.get("CE_OUT", "/kaggle/working")
INP = os.environ.get("CE_INPUT", "/kaggle/input")

RULES = [(re.compile(r"\b(street|st)\b", re.I), ["rue"]), (re.compile(r"\b(road|rd)\b", re.I), ["route", "rte"]),
         (re.compile(r"\b(avenue|ave)\b", re.I), ["av", "avenue"]), (re.compile(r"\b(boulevard|blvd)\b", re.I), ["bd", "boulevard"]),
         (re.compile(r"\b(pvt\.? ?ltd\.?|private limited|llc|inc\.?|ltd\.?|limited|corp\.?|co\.)", re.I), ["SARL", "SAS", "EURL", "SASU"]),
         (re.compile(r"(?:\bno\.?|#)\s*(?=\d)", re.I), [""])]


def frenchify(s, rng):
    """French-style rewrite of one 'name ; address' string (provided text only)."""
    for pat, subs in RULES:
        if rng.random() < 0.7:
            s = pat.sub(lambda m: rng.choice(subs), s)
    if rng.random() < 0.3:
        w = s.split(" ")
        idx = [i for i, t in enumerate(w) if "e" in t]
        if idx:
            i = rng.choice(idx)
            w[i] = w[i].replace("e", rng.choice(["é", "è"]), 1)
            s = " ".join(w)
    return s


def find(name):
    return glob.glob(f"{INP}/**/{name}", recursive=True)[0]


def text(df, side):
    n, a = df[f"{side}_name"].fillna("").astype(str), df[f"{side}_addr"].fillna("").astype(str)
    return (n + " ; " + a).tolist()


def worker(train_country, gpu):
    """Train on one country's band; score the other country's band + extras and the France blend band."""
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import roc_auc_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(0)
    rng = random.Random(0)
    dev = f"cuda:{gpu}"
    other = "India" if train_country == "US" else "US"
    band = pd.read_parquet(find("train_band.parquet"))
    extra = pd.read_parquet(find("extra_heldout.parquet"))
    fr = pd.read_parquet(find("france_band.parquet"))
    tr = band[band.country == train_country].sample(frac=1.0, random_state=0).reset_index(drop=True)
    ho = pd.concat([band[band.country == other], extra[extra.country == other]], ignore_index=True)
    n_band_ho = int((band.country == other).sum())
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION or None)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, revision=REVISION or None).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = math.ceil(len(tr) / BATCH)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.cuda.amp.GradScaler()
    lossf = torch.nn.BCEWithLogitsLoss()
    A, R, Y = text(tr, "a"), text(tr, "r"), tr.y.values.astype("float32")
    if AUG:
        for i in range(len(A)):
            if rng.random() < 0.3:
                if rng.random() < 0.5:
                    A[i] = frenchify(A[i], rng)
                else:
                    R[i] = frenchify(R[i], rng)
            if rng.random() < 0.5:
                A[i], R[i] = R[i], A[i]
    model.train()
    t0, seen = time.time(), 0
    for s in range(steps):
        i = slice(s * BATCH, (s + 1) * BATCH)
        enc = tok(A[i], R[i], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            logit = model(**enc).logits.squeeze(-1)
            loss = lossf(logit.float(), torch.tensor(Y[i], device=dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sch.step()
        seen += len(Y[i])
        if s % 500 == 0:
            print(f"[{TAG}/{train_country}] step {s}/{steps} loss {loss.item():.4f} {seen / (time.time() - t0):.0f} pairs/s", flush=True)
        if (time.time() - t0) / 60 > TRAIN_MIN:
            print(f"[{TAG}/{train_country}] time budget reached at step {s}/{steps}", flush=True)
            break
    train_min = (time.time() - t0) / 60
    model.eval()

    @torch.no_grad()
    def score(a, r, bs=512):
        out = []
        for k in range(0, len(a), bs):
            enc = tok(a[k:k + bs], r[k:k + bs], truncation=True, max_length=MAX_LEN, padding=True,
                      return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                out.append(model(**enc).logits.squeeze(-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, "float32")

    t1 = time.time()
    s_ho = score(text(ho, "a"), text(ho, "r"))
    fa, frr = text(fr, "a"), text(fr, "r")
    s_fr = score(fa, frr)
    if AUG:                                            # trained with both orders: average both on France
        s_fr = (s_fr + score(frr, fa)) / 2
    infer_min = (time.time() - t1) / 60
    met = {"model": MODEL, "revision": getattr(model.config, "_commit_hash", None) or REVISION, "max_len": MAX_LEN,
           "tag": TAG, "aug": AUG, "train_country": train_country, "heldout_country": other,
           "train_pairs_seen": int(seen), "train_steps_share": round(seen / len(tr), 3),
           "train_min": round(train_min, 1), "infer_min": round(infer_min, 1),
           "heldout_band_auc_ce": round(float(roc_auc_score(ho.y[:n_band_ho], s_ho[:n_band_ho])), 5),
           "heldout_band_auc_lgbm": round(float(roc_auc_score(ho.y[:n_band_ho], ho.p2[:n_band_ho])), 5),
           "heldout_extra_auc_ce": round(float(roc_auc_score(ho.y[n_band_ho:], s_ho[n_band_ho:])), 5),
           "france_pairs": int(len(fr)), "france_mean_logit": round(float(s_fr.mean()), 4)}
    print(json.dumps(met), flush=True)
    pd.DataFrame({"a_id": ho.a_id, "r_id": ho.r_id, "ce": s_ho.astype("float32")}).to_parquet(
        f"{OUT}/scores_{TAG}_{train_country}_heldout.parquet", index=False)
    pd.DataFrame({"a_id": fr.a_id, "r_id": fr.r_id, "ce": s_fr.astype("float32")}).to_parquet(
        f"{OUT}/scores_{TAG}_{train_country}_france.parquet", index=False)
    json.dump(met, open(f"{OUT}/metrics_{TAG}_{train_country}.json", "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1:                              # child process: python ce_france.py <country> <gpu>
        worker(sys.argv[1], int(sys.argv[2]))
        sys.exit(0)
    import torch
    n = torch.cuda.device_count()
    print(json.dumps({"gpus": n, "model": MODEL, "tag": TAG, "aug": AUG}), flush=True)
    jobs = [("US", 0), ("India", 1 if n > 1 else 0)]
    if n > 1:
        procs = [subprocess.Popen([sys.executable, __file__, c, str(g)]) for c, g in jobs]
        codes = [p.wait() for p in procs]
    else:
        codes = [subprocess.call([sys.executable, __file__, c, str(g)]) for c, g in jobs]
    print(json.dumps({"exit_codes": codes}))
    sys.exit(max(codes))
