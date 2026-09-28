"""Kaggle kernel: fine-tune a multilingual cross-encoder on the uncertain band of ER candidate pairs.

Leakage-safe by country: the model trained on the US band scores the India band (and vice versa), so every train
score is out-of-sample and doubles as a held-out-country (unseen-country) measurement. Both models score the test
band (France, India, US). Outputs (keyed by a_id, r_id): /kaggle/working/scores_<model_tag>.parquet + metrics json.
Only aggregate metrics are printed. Licence: intfloat/multilingual-e5-small and intfloat/multilingual-e5-base (MIT).
"""
import glob
import json
import math
import os
import subprocess
import sys
import time

MODEL = os.environ.get("CE_MODEL", "intfloat/multilingual-e5-small")
REVISION = os.environ.get("CE_REVISION", "614241f622f53c4eeff9890bdc4f31cfecc418b3")   # MIT, 117.7M params
# Offline use: pre-download this revision (or point CE_MODEL to a local copy) and set HF_HUB_OFFLINE=1.
TAG = os.environ.get("CE_TAG", "e5s")
# "country": train on one country, score the other (scores double as unseen-country tests).
# "incountry": two entity halves pooled over both countries (band fold parity); each half's model scores the other
#              half (out-of-fold) and the whole test band.
MODE = os.environ.get("CE_MODE", "country")
MAX_LEN = int(os.environ.get("CE_MAXLEN", "96"))
BATCH = int(os.environ.get("CE_BATCH", "128"))
LR = float(os.environ.get("CE_LR", "5e-5"))
TRAIN_MIN = float(os.environ.get("CE_TRAIN_MIN", "110"))      # per-model training time budget (minutes)
OUT = os.environ.get("CE_OUT", "/kaggle/working")
INP = os.environ.get("CE_INPUT", "/kaggle/input")


def band_paths():
    tr = glob.glob(f"{INP}/**/train_band.parquet", recursive=True)
    te = glob.glob(f"{INP}/**/test_band.parquet", recursive=True)
    return tr[0], te[0]


def text(df, side):
    n, a = df[f"{side}_name"].fillna("").astype(str), df[f"{side}_addr"].fillna("").astype(str)
    return (n + " ; " + a).tolist()


def worker(train_country, gpu):
    """Train on one country's band; score the other country's band and the whole test band."""
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import roc_auc_score, log_loss
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(0)
    dev = f"cuda:{gpu}"
    tr_path, te_path = band_paths()
    band = pd.read_parquet(tr_path)
    if MODE == "incountry":                        # train_country is the held-out half here ("0" or "1")
        half = int(train_country)
        other = f"half{half}"
        train_country = f"half{half}"
        tr = band[band.fold % 2 != half].sample(frac=1.0, random_state=0).reset_index(drop=True)
        ho = band[band.fold % 2 == half].reset_index(drop=True)
    else:
        other = "India" if train_country == "US" else "US"
        tr = band[band.country == train_country].sample(frac=1.0, random_state=0).reset_index(drop=True)
        ho = band[band.country == other].reset_index(drop=True)
    te = pd.read_parquet(te_path)
    rev = None if (os.path.isdir(MODEL) or not REVISION) else REVISION
    tok = AutoTokenizer.from_pretrained(MODEL, revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, revision=rev).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = math.ceil(len(tr) / BATCH)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.cuda.amp.GradScaler()
    lossf = torch.nn.BCEWithLogitsLoss()
    A, R, Y = text(tr, "a"), text(tr, "r"), tr.y.values.astype("float32")
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
    def score(df, bs=512):
        a, r, out = text(df, "a"), text(df, "r"), []
        for k in range(0, len(df), bs):
            enc = tok(a[k:k + bs], r[k:k + bs], truncation=True, max_length=MAX_LEN, padding=True,
                      return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                out.append(model(**enc).logits.squeeze(-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, "float32")

    t1 = time.time()
    s_ho = score(ho)
    s_te = score(te)
    infer_min = (time.time() - t1) / 60
    p_ho = 1 / (1 + np.exp(-s_ho))
    met = {"model": MODEL, "revision": getattr(model.config, "_commit_hash", None) or rev, "max_len": MAX_LEN,
           "tag": TAG, "train_country": train_country, "heldout_country": other,
           "train_pairs_seen": int(seen), "train_min": round(train_min, 1), "infer_min": round(infer_min, 1),
           "heldout_auc_ce": round(float(roc_auc_score(ho.y, s_ho)), 5),
           "heldout_auc_lgbm": round(float(roc_auc_score(ho.y, ho.p2)), 5),
           "heldout_logloss_ce": round(float(log_loss(ho.y, np.clip(p_ho, 1e-6, 1 - 1e-6))), 5),
           "heldout_logloss_lgbm": round(float(log_loss(ho.y, np.clip(ho.p2, 1e-6, 1 - 1e-6))), 5)}
    print(json.dumps(met), flush=True)
    pd.DataFrame({"a_id": ho.a_id, "r_id": ho.r_id, "split": "train", "ce": s_ho.astype("float32")}).to_parquet(
        f"{OUT}/scores_{TAG}_{train_country}_heldout.parquet", index=False)
    pd.DataFrame({"a_id": te.a_id, "r_id": te.r_id, "split": "test", "country": te.country,
                  "ce": s_te.astype("float32")}).to_parquet(f"{OUT}/scores_{TAG}_{train_country}_test.parquet", index=False)
    json.dump(met, open(f"{OUT}/metrics_{TAG}_{train_country}.json", "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1:                          # child process: python ce.py <country> <gpu>
        worker(sys.argv[1], int(sys.argv[2]))
        sys.exit(0)
    import torch
    n = torch.cuda.device_count()
    print(json.dumps({"gpus": n, "names": [torch.cuda.get_device_name(i) for i in range(n)], "model": MODEL}), flush=True)
    jobs = [("0", 0), ("1", 1 if n > 1 else 0)] if MODE == "incountry" else [("US", 0), ("India", 1 if n > 1 else 0)]
    if n > 1:                                      # both countries in parallel, one GPU each
        procs = [subprocess.Popen([sys.executable, __file__, c, str(g)]) for c, g in jobs]
        codes = [p.wait() for p in procs]
    else:
        codes = [subprocess.call([sys.executable, __file__, c, str(g)]) for c, g in jobs]
    print(json.dumps({"exit_codes": codes}))
    sys.exit(max(codes))
