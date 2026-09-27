"""Mine a word-level non-Latin -> Latin dictionary from TRAINING ground-truth pairs only (no external data).

For true pairs whose S2/S3 name is written in a non-Latin script and whose S1 name is Latin, with the same number of
tokens, tokens are aligned by position (transliterated names keep word order). A mapping is kept when it has support
>= MIN_SUPPORT and is the majority translation (>= MIN_PREC) for that source token. Script-agnostic: any script works.
Output: cache/translit.json {non_latin_token: latin_token}.
"""
import collections
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import memguard  # noqa: E402
from io_utils import ROOT, DATA, read_tsv, gt_pairs  # noqa: E402
from normalize import load_norm, NONLATIN, NAME_ABBR, TOKEN  # noqa: E402
from unidecode import unidecode  # noqa: E402

PATH = os.path.join(ROOT, "cache", "translit.json")
MIN_SUPPORT, MIN_PREC = 3, 0.6
SPLIT = re.compile(r"[\s\-_/,()\[\]]+")


def src_tokens(s):
    return [t.strip(".") for t in SPLIT.split(s) if t.strip(".")]


def latin_tokens(s):
    return [NAME_ABBR.get(t, t) for t in TOKEN.findall(unidecode(s).lower())]


def mine():
    R = pd.concat([load_norm("train", k, ["id", "name_raw", "name_nonlatin"]) for k in (2, 3)])
    R = R[R.name_nonlatin].drop(columns="name_nonlatin")
    gt = read_tsv(os.path.join(DATA, "train", "train_ground_truth.tsv"))
    p = gt_pairs(gt)
    del gt
    p = p[p.rid.isin(set(R.id))]
    S = load_norm("train", 1, ["id", "name_raw"])
    S = S[S.id.isin(set(p.s1))]
    d = p.merge(R, left_on="rid", right_on="id").merge(S, left_on="s1", right_on="id", suffixes=("_r", "_a"))
    del R, S, p
    cnt = collections.defaultdict(collections.Counter)
    for rn, an in zip(d.name_raw_r.values, d.name_raw_a.values):
        if NONLATIN.search(an):
            continue
        rt, at = src_tokens(rn), latin_tokens(an)
        if len(rt) != len(at):
            continue
        for x, y in zip(rt, at):
            if NONLATIN.search(x):
                cnt[x][y] += 1
    out = {}
    for x, c in cnt.items():
        y, n = c.most_common(1)[0]
        tot = sum(c.values())
        if n >= MIN_SUPPORT and n / tot >= MIN_PREC:
            out[x] = y
    json.dump(out, open(PATH, "w"), ensure_ascii=False)
    print(f"aligned pairs={len(d)} source tokens={len(cnt)} kept={len(out)} peak={memguard.peak_gb():.2f}GB")
    for k in list(out)[:15]:
        print(" ", k, "->", out[k])
    return out




ADDR_PATH = os.path.join(ROOT, "cache", "translit_addr.json")


def mine_addr():
    """Non-Latin trailing address component (usually the state) -> S1's trailing component, majority vote."""
    R = pd.concat([load_norm("train", k, ["id", "addr_raw", "addr_nonlatin"]) for k in (2, 3)])
    R = R[R.addr_nonlatin].drop(columns="addr_nonlatin")
    gt = read_tsv(os.path.join(DATA, "train", "train_ground_truth.tsv"))
    p = gt_pairs(gt)
    del gt
    p = p[p.rid.isin(set(R.id))]
    S = load_norm("train", 1, ["id", "addr_raw"])
    S = S[S.id.isin(set(p.s1))]
    d = p.merge(R, left_on="rid", right_on="id").merge(S, left_on="s1", right_on="id", suffixes=("_r", "_a"))
    cnt = collections.defaultdict(collections.Counter)
    for ra, aa in zip(d.addr_raw_r.values, d.addr_raw_a.values):
        rc = ra.split(",")[-1].strip()
        ac = aa.split(",")[-1].strip()
        if NONLATIN.search(rc) and not NONLATIN.search(ac):
            cnt[rc][ac.lower()] += 1
    out = {x: c.most_common(1)[0][0] for x, c in cnt.items()
           if c.most_common(1)[0][1] >= MIN_SUPPORT and c.most_common(1)[0][1] / sum(c.values()) >= MIN_PREC}
    json.dump(out, open(ADDR_PATH, "w"), ensure_ascii=False)
    print(f"addr components={len(cnt)} kept={len(out)}", list(out.items())[:10])


def coverage(split="test"):
    names = json.load(open(PATH))
    tot = hit = 0
    for k in (2, 3):
        R = load_norm(split, k, ["name_raw", "name_nonlatin"])
        for s in R.name_raw.values[R.name_nonlatin.values]:
            for t in src_tokens(s):
                if NONLATIN.search(t):
                    tot += 1
                    hit += t in names
    print(f"{split}: non-Latin name tokens={tot} covered={hit / max(tot, 1):.4f}")


if __name__ == "__main__":
    memguard.start(limit_gb=2.5)
    mine() if "names" in sys.argv else None
    mine_addr() if "addr" in sys.argv else None
    coverage() if "cov" in sys.argv else None
