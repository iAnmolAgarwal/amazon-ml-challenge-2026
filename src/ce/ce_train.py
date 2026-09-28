"""Cross-encoder training (Kaggle notebook, GPU T4 x2, internet on for the model download).

Model intfloat/multilingual-e5-small (MIT) as AutoModelForSequenceClassification(num_labels=1), BCE on logits.
Input pair: (S1 "name ; address", pool "name ; address"), raw strings, max_len 96, batch 128, AdamW lr 5e-5,
weight decay 0.01, 3% linear warm-up then linear decay, grad clip 1.0, fp16 autocast, 1 epoch, 170 min time box.
Two models on S1 halves of the harness folds: A trains on half 0 (folds 0,1), B on half 1 (folds 2,3,4); one per GPU,
seeds 0 (A) and 1 (B). Inputs (Kaggle datasets, built by src/ce/ce_export.py): train_s1/train_pool.parquet (n, a, c in
raw row order), ce_pairs.parquet (s1_idx, pool_idx, y, half), ce_cfg.json {"budget_min": 170, "maxn": 6200000}.
Outputs /kaggle/working/mA, mB (model + tokenizer), val_{A,B}.json, log_{A,B}.txt.
ce_cfg.json may also set "model", "lr" and "revision" (defaults e5-small, 5e-5 and the pinned revision in REV); the
e5-base score (column ce_base) used {"model": "intfloat/multilingual-e5-base", "lr": 3e-5, "budget_min": 140,
"maxn": 6200000} (about a third of an epoch in the time box).
Run 26 Sep: ~608 pairs/s per GPU, 6.10M pairs per model in ~170 min; validation AUC 0.9996 on the other half.
"""
import os, sys, subprocess, time, json

W = r'''
import os, sys, time, glob, json, math
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score, log_loss

def f(name):
    return sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))[0]

half, budget, maxn, MN = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
LR = float(sys.argv[5]) if len(sys.argv) > 5 else 5e-5
REV = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] else None
hid = 0 if half == "A" else 1
out = f"/kaggle/working/m{half}"
log = open(f"/kaggle/working/log_{half}.txt", "a")
def P(*a):
    s = time.strftime("%H:%M:%S") + " " + " ".join(map(str, a)); print(half, s, flush=True); log.write(s + "\n"); log.flush()

torch.manual_seed(hid); rng = np.random.default_rng(hid)
s1 = pd.read_parquet(f("train_s1.parquet")); pool = pd.read_parquet(f("train_pool.parquet"))
S1N, S1A, S1C = [np.asarray(s1[c].astype(str) if c == "c" else s1[c], dtype=object) for c in "nac"]
PN, PA = [np.asarray(pool[c], dtype=object) for c in "na"]
del s1, pool
pr = pd.read_parquet(f("ce_pairs.parquet"))
tr = pr[pr.half == hid].sample(frac=1.0, random_state=hid).head(maxn)
va = pr[pr.half != hid].sample(n=min(200000, int((pr.half != hid).sum())), random_state=7)
del pr
P("train pairs", len(tr), "pos", float(tr.y.mean()), "val", len(va))

tok = AutoTokenizer.from_pretrained(MN, revision=REV)
BS, ML = 128, 96

class DS(torch.utils.data.Dataset):
    def __init__(self, df, bs):
        self.a = df.s1_idx.values.astype(np.int64); self.b = df.pool_idx.values.astype(np.int64); self.y = df.y.values.astype(np.float32); self.bs = bs
    def __len__(self):
        return math.ceil(len(self.a) / self.bs)
    def __getitem__(self, i):
        sl = slice(i * self.bs, (i + 1) * self.bs); a = self.a[sl]; b = self.b[sl]
        x = tok([f"{n} ; {d}" for n, d in zip(S1N[a], S1A[a])], [f"{n} ; {d}" for n, d in zip(PN[b], PA[b])], truncation=True, max_length=ML, padding=True, return_tensors="pt")
        return x["input_ids"], x["attention_mask"], x.get("token_type_ids"), torch.from_numpy(self.y[sl])

def loader(df, bs):
    return torch.utils.data.DataLoader(DS(df, bs), batch_size=None, shuffle=False, num_workers=2, prefetch_factor=8, persistent_workers=False)

model = AutoModelForSequenceClassification.from_pretrained(MN, num_labels=1, revision=REV).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, fused=True)
nb = len(DS(tr, BS)); total = [nb]; warm = max(1, int(0.03 * nb))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total[0] - s) / max(1, total[0] - warm))))
scaler = torch.amp.GradScaler("cuda"); lf = torch.nn.BCEWithLogitsLoss()
model.train(); t0 = time.time(); run = 0.0
for step, (ii, am, tt, y) in enumerate(loader(tr, BS)):
    if step >= total[0]:
        break
    with torch.autocast("cuda", dtype=torch.float16):
        o = model(input_ids=ii.cuda(non_blocking=True), attention_mask=am.cuda(non_blocking=True), token_type_ids=None if tt is None else tt.cuda(non_blocking=True)).logits.squeeze(-1)
    loss = lf(o.float(), y.cuda())
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
    run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
    if step == 300:
        rate = 300 / (time.time() - t0); total[0] = min(nb, int(rate * budget * 60)); P("rate steps/s", round(rate, 2), "pairs/s", round(rate * BS), "total steps", total[0], "of", nb)
    if step % 2000 == 0:
        P("step", step, "loss", round(run, 4), "elapsed min", round((time.time() - t0) / 60, 1))
P("trained steps", step, "pairs", step * BS, "min", round((time.time() - t0) / 60, 1))
model.eval(); model.save_pretrained(out); tok.save_pretrained(out)
ps = []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    for ii, am, tt, y in loader(va, 512):
        ps.append(model(input_ids=ii.cuda(), attention_mask=am.cuda(), token_type_ids=None if tt is None else tt.cuda()).logits.squeeze(-1).float().cpu().numpy())
va["ce"] = np.concatenate(ps); va["c"] = S1C[va.s1_idx.values]
res = {}
for c, g in [("all", va)] + list(va.groupby("c")):
    p = 1 / (1 + np.exp(-g.ce.values))
    res[c] = dict(n=len(g), pos=float(g.y.mean()), auc=float(roc_auc_score(g.y, g.ce)), ll=float(log_loss(g.y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])))
P("val (other half, ce training-pair distribution)", json.dumps(res))
json.dump(res, open(f"/kaggle/working/val_{half}.json", "w"))
'''

cfg = json.load(open([os.path.join(r, x) for r, _, fs in os.walk("/kaggle/input") for x in fs if x == "ce_cfg.json"][0]))
REV = {"intfloat/multilingual-e5-small": "614241f622f53c4eeff9890bdc4f31cfecc418b3", "intfloat/multilingual-e5-base": "d128750597153bb5987e10b1c3493a34e5a4502a"}
MN, LR = cfg.get("model", "intfloat/multilingual-e5-small"), str(cfg.get("lr", 5e-5))
RV = cfg.get("revision", REV.get(MN, ""))
open("/kaggle/working/w.py", "w").write(W)
import torch
ng = torch.cuda.device_count()
print("gpus", ng, [torch.cuda.get_device_name(i) for i in range(ng)], "cfg", cfg, flush=True)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
AutoTokenizer.from_pretrained(MN, revision=RV or None); AutoModelForSequenceClassification.from_pretrained(MN, num_labels=1, revision=RV or None)
if ng >= 2:
    ps = [subprocess.Popen([sys.executable, "/kaggle/working/w.py", h, str(cfg["budget_min"]), str(cfg["maxn"]), MN, LR, RV], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}) for i, h in enumerate("AB")]
    rc = [p.wait() for p in ps]
else:
    rc = [subprocess.call([sys.executable, "/kaggle/working/w.py", h, str(cfg["budget_min"] / 2), str(cfg["maxn"]), MN, LR, RV]) for h in "AB"]
print("return codes", rc, flush=True)
