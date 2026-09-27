"""Cross-encoder scoring (Kaggle notebook, GPU T4 x2, internet off).

Inputs: the training notebook's output (mA, mB), train/test records, train_band.parquet (s1_idx, pool_idx, half) and
test_band.parquet (s1_idx, pool_idx) from src/ce/ce_export.py band. Model A scores train band pairs of half 1 and
model B those of half 0 (out-of-fold by S1); both score all test band pairs. fp16, batch 1024, max_len 96,
pairs sorted by text length for batching and written back in band order.
Outputs /kaggle/working/ce_{A,B}_{train,test}.parquet (s1_idx, pool_idx, ce = logit).
Run 27 Sep: ~4,200 pairs/s per GPU, ~22 min for 3.0M train + 2 x 2.8M test pairs.
KIN / KOUT / KBS / KNW override the input dir, output dir, batch size and loader workers (for a local CPU check).
"""
import os, sys, subprocess, json

W = r'''
import os, sys, time, glob, gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

IN, OUT = os.environ.get("KIN", "/kaggle/input"), os.environ.get("KOUT", "/kaggle/working")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
def f(name):
    return sorted(glob.glob(f"{IN}/**/{name}", recursive=True))[0]

half = sys.argv[1]; hid = 0 if half == "A" else 1
md = os.path.dirname(f(f"m{half}/config.json"))
def P(*a):
    print(half, time.strftime("%H:%M:%S"), *a, flush=True)
tok = AutoTokenizer.from_pretrained(md)
model = AutoModelForSequenceClassification.from_pretrained(md).to(DEV).eval()
if DEV == "cuda":
    model = model.half()
BS, ML = int(os.environ.get("KBS", 1024)), 96

class DS(torch.utils.data.Dataset):
    def __init__(self, a, b, bs):
        self.a, self.b, self.bs = a, b, bs
    def __len__(self):
        return (len(self.a) + self.bs - 1) // self.bs
    def __getitem__(self, i):
        sl = slice(i * self.bs, (i + 1) * self.bs); a = self.a[sl]; b = self.b[sl]
        x = tok([f"{n} ; {d}" for n, d in zip(SN[a], SA[a])], [f"{n} ; {d}" for n, d in zip(PN[b], PA[b])], truncation=True, max_length=ML, padding=True, return_tensors="pt")
        return x["input_ids"], x["attention_mask"], x.get("token_type_ids")

for sp in ["train", "test"]:
    s1 = pd.read_parquet(f(f"{sp}_s1.parquet")); pool = pd.read_parquet(f(f"{sp}_pool.parquet"))
    SN, SA = [np.asarray(s1[c], dtype=object) for c in "na"]; PN, PA = [np.asarray(pool[c], dtype=object) for c in "na"]
    ls1 = s1.n.str.len().values + s1.a.str.len().values; lp = pool.n.str.len().values + pool.a.str.len().values
    del s1, pool
    pr = pd.read_parquet(f(f"{sp}_band.parquet"))
    assert len(pr) == {"train": 3008987, "test": 2815521}[sp], (sp, len(pr))
    if sp == "train":
        pr = pr[pr.half != hid]
    a = pr.s1_idx.values.astype(np.int64); b = pr.pool_idx.values.astype(np.int64)
    o = np.argsort(ls1[a] + lp[b], kind="stable")
    P(sp, "pairs", len(a))
    dl = torch.utils.data.DataLoader(DS(a[o], b[o], BS), batch_size=None, num_workers=int(os.environ.get("KNW", 3)), prefetch_factor=6 if int(os.environ.get("KNW", 3)) else None)
    out = []; t0 = time.time()
    with torch.no_grad():
        for i, (ii, am, tt) in enumerate(dl):
            out.append(model(input_ids=ii.to(DEV, non_blocking=True), attention_mask=am.to(DEV, non_blocking=True), token_type_ids=None if tt is None else tt.to(DEV, non_blocking=True)).logits.squeeze(-1).float().cpu().numpy())
            if i % 500 == 0:
                P(sp, i * BS, "pairs", round(time.time() - t0), "s")
    ce = np.empty(len(a), dtype=np.float32); ce[o] = np.concatenate(out)
    pd.DataFrame({"s1_idx": pr.s1_idx.values, "pool_idx": pr.pool_idx.values, "ce": ce}).to_parquet(f"{OUT}/ce_{half}_{sp}.parquet", index=False)
    P(sp, "done", len(ce), round(time.time() - t0), "s")
    del SN, SA, PN, PA, pr, a, b, o, out, dl; gc.collect()
'''

KOUT = os.environ.get("KOUT", "/kaggle/working")
open(f"{KOUT}/ws.py", "w").write(W)
import torch
ng = torch.cuda.device_count()
print("gpus", ng, flush=True)
if ng >= 2:
    ps = [subprocess.Popen([sys.executable, f"{KOUT}/ws.py", h], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}) for i, h in enumerate("AB")]
    rc = [p.wait() for p in ps]
else:
    rc = [subprocess.call([sys.executable, f"{KOUT}/ws.py", h]) for h in "AB"]
print("return codes", rc, flush=True)
