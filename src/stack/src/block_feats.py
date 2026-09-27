"""Blocking feature strings (rare unigrams + conjunction keys) hashed into a sparse binary matrix.

Kept deliberately compact (~40 keys/record) so matrices stay small on an 8 GB machine.
"""
import os

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

N_FEATURES = 2 ** 22
ADDR_GENERIC = {"st", "rd", "dr", "av", "bd", "ln", "ct", "no", "h", "fl", "apt", "ste", "unit", "near", "opposite",
                "po", "box", "blk", "plt", "sec", "r", "de", "du", "des", "la", "le", "the", "of", "and"}
_HV = HashingVectorizer(token_pattern=r"\S+", lowercase=False, n_features=N_FEATURES, alternate_sign=False,
                        norm=None, binary=True, dtype=np.float32)


def addr_feats(addr):
    t = addr.split()[:12]
    if not t:
        return []
    f = ["a:" + x for x in t]
    f += ["ab:" + x + "_" + y for x, y in zip(t, t[1:])]
    nums = [x for x in t if x.isdigit()][:2]
    words = [x for x in t if not x.isdigit() and len(x) >= 3 and x not in ADDR_GENERIC][:5]
    f += ["an:" + n + "_" + w for n in nums for w in words]
    return f


def name_feats(core, skel, nums):
    t = core.split()[:6]
    k = skel.split()[:6]
    f = ["n:" + x for x in t] + ["k:" + x for x in k]
    f += ["np:" + "_".join(sorted(p)) for p in zip(t, t[1:])]
    f += ["kp:" + "_".join(sorted(p)) for p in zip(k, k[1:])]
    if len(t) > 2:
        f.append("np:" + "_".join(sorted((t[0], t[-1]))))
    if t:
        f.append("nw:" + "".join(t))
        f.append("kw:" + "".join(k))
    if nums:
        f += ["x:" + x + "_" + nums[0] for x in k[:3]]
    return f


def record_feats(core, skel, addr, nums):
    return " ".join(name_feats(core, skel, nums.split()) + addr_feats(addr))


def route_feats(core, skel, addr, nums):
    """Single-channel feature string for diagnostic retrieval routes (BLOCK_ROUTE=addr|name); default = both."""
    route = os.environ.get("BLOCK_ROUTE", "")
    if route == "addr":
        return " ".join(addr_feats(addr))
    if route == "name":
        return " ".join(name_feats(core, skel, nums.split()))
    return record_feats(core, skel, addr, nums)


def hashed_chunk(args):
    """Worker: 4 aligned lists -> binary CSR (n, N_FEATURES). Env is read in the (spawned) worker, so routes propagate."""
    core, skel, addr, nums = args
    return _HV.transform([route_feats(*x) for x in zip(core, skel, addr, nums)]).tocsr()
