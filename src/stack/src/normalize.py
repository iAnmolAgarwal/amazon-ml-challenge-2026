"""Generic, country-agnostic text normalisation for business names and addresses.

All dictionaries are language-level (English/French/Indian-English business & street vocabulary), applied to every
record identically; there is no per-country branch.
"""
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from unidecode import unidecode

sys.path.insert(0, os.path.dirname(__file__))
from io_utils import ROOT, DATA, read_tsv  # noqa: E402

CACHE = os.path.join(ROOT, "cache")

# ---- name vocabulary ---------------------------------------------------------------------------------------
NAME_ABBR = {
    "pvt": "private", "pte": "private", "priv": "private", "ltd": "limited", "ltda": "limited", "lmtd": "limited",
    "corp": "corporation", "co": "company", "cos": "companies", "inc": "incorporated", "incorp": "incorporated",
    "intl": "international", "int": "international", "mfg": "manufacturing", "mfrs": "manufacturers",
    "svc": "services", "svcs": "services", "serv": "services", "bros": "brothers", "assoc": "associates",
    "assocs": "associates", "mgmt": "management", "natl": "national", "tech": "technologies",
    "techs": "technologies", "technology": "technologies", "grp": "group", "ent": "enterprises",
    "entp": "enterprises", "enterprise": "enterprises", "ind": "industries", "inds": "industries",
    "industry": "industries", "sys": "systems", "system": "systems", "dev": "development", "ctr": "center",
    "centre": "center", "hosp": "hospital", "inst": "institute", "univ": "university", "dept": "department",
    "solution": "solutions", "service": "services", "product": "products", "trader": "traders",
    "export": "exports", "import": "imports", "infra": "infrastructure", "ets": "etablissements",
    "st": "saint", "ste": "sainte", "sr": "senior", "jr": "junior", "mt": "mount", "ft": "fort",
}
LEGAL = {
    "private", "limited", "incorporated", "corporation", "company", "llc", "llp", "lp", "pllc", "pc", "plc", "pty",
    "gmbh", "ag", "bv", "nv", "sarl", "sas", "sasu", "eurl", "sa", "sci", "ei", "snc", "scop", "selarl", "scp",
    "sca", "scs", "eirl", "opc", "lllp", "pa", "ltee",
}
STOP = {"the", "of", "and", "a", "an", "de", "du", "des", "la", "le", "les", "et", "d", "l", "by", "for", "at", "en"}
ALIAS = re.compile(r"\b(?:formerly(?: known as)?|f/k/a|fka|dba|d/b/a|aka|a/k/a|trading as|t/a)\b[:\s]*")
DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|in|net|org|co|biz|info|fr|us|io|co\.in|org\.in)$")

# ---- address vocabulary (canonical short forms) -----------------------------------------------------------
ADDR_ABBR = {
    "street": "st", "saint": "st", "str": "st", "strt": "st", "road": "rd", "drive": "dr", "drv": "dr",
    "avenue": "av", "ave": "av", "aven": "av", "boulevard": "bd", "blvd": "bd", "boul": "bd", "lane": "ln",
    "court": "ct", "crt": "ct", "circle": "cir", "circ": "cir", "place": "pl", "plaza": "plz", "parkway": "pkwy",
    "highway": "hwy", "trail": "trl", "terrace": "ter", "square": "sq", "suite": "ste", "sainte": "ste",
    "apartment": "apt", "apartments": "apt", "appartments": "apt", "appt": "apt", "floor": "fl", "flr": "fl",
    "building": "bldg", "bldng": "bldg", "number": "no", "nr": "near", "opp": "opposite", "opposit": "opposite",
    "north": "n", "south": "s", "east": "e", "west": "w", "mount": "mt", "fort": "ft", "point": "pt",
    "township": "twp", "county": "cty", "center": "ctr", "centre": "ctr", "junction": "jct", "way": "wy",
    "expressway": "expy", "freeway": "fwy", "turnpike": "tpke", "extension": "ext", "extn": "ext",
    "sector": "sec", "sect": "sec", "colony": "col", "nagar": "ngr", "marg": "mg", "cross": "crs",
    "main": "mn", "layout": "lyt", "phase": "ph", "block": "blk", "plot": "plt", "road.": "rd",
    "rue": "r", "allee": "all", "chemin": "ch", "route": "rte", "impasse": "imp", "faubourg": "fbg",
    "quai": "qu", "cours": "crs", "rn": "rte", "residence": "res", "batiment": "bat", "cedex": "",
    "district": "dist", "dt": "dist", "distt": "dist", "taluk": "tq", "tal": "tq", "tq": "tq",
    "po": "po", "p.o": "po", "box": "box", "house": "h", "hno": "h", "door": "dr_", "null": "", "na": "",
}
NUM = re.compile(r"\d+")
TOKEN = re.compile(r"[a-z0-9]+")
NONLATIN = re.compile(r"[^\x00-ɏḀ-ỿ]")
LONGNUM = re.compile(r"\b\d{7,}\b")


_TR = None


def _translit():
    """Word-level non-Latin -> Latin dictionaries mined from TRAIN pairs by src/translit.py (optional)."""
    global _TR
    if _TR is None:
        import json
        names = os.path.join(CACHE, "translit.json")
        addr = os.path.join(CACHE, "translit_addr.json")
        _TR = (json.load(open(names)) if os.path.exists(names) else {},
               json.load(open(addr)) if os.path.exists(addr) else {})
    return _TR


_SPLIT_KEEP = re.compile(r"([\s\-_/,()\[\]]+)")


def map_nonlatin_name(s):
    if not NONLATIN.search(s):
        return s
    d = _translit()[0]
    return "".join(d.get(t.strip("."), t) for t in _SPLIT_KEEP.split(s))


def map_nonlatin_addr(s):
    if not NONLATIN.search(s):
        return s
    d = _translit()[1]
    return ",".join(d.get(c.strip(), c) for c in s.split(","))


def fold(s):
    return unidecode(s).lower() if s else ""


def skeleton_word(w):
    """Loose cross-script phonetic key: first letter + consonant skeleton (h/w/y/vowels dropped, repeats merged)."""
    if not w:
        return ""
    if w.isdigit():
        return w
    w = (w.replace("ph", "f").replace("ck", "k").replace("q", "k").replace("x", "ks").replace("c", "k")
          .replace("z", "s").replace("w", "v").replace("j", "g"))
    first = w[0]
    body = [ch for ch in w[1:] if ch not in "aeiouhy"]
    out = [first]
    for ch in body:
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


LEGAL_SKEL = None


def _canon_legal(t):
    """Map transliterated long legal words (e.g. 'limittedd', 'praiveett') to their canonical form by skeleton."""
    global LEGAL_SKEL
    if LEGAL_SKEL is None:
        LEGAL_SKEL = {skeleton_word(w): w for w in LEGAL if len(w) >= 7}
    if len(t) >= 5 and t not in LEGAL:
        return LEGAL_SKEL.get(skeleton_word(t), t)
    return t


def norm_name(raw):
    s = fold(map_nonlatin_name(raw))
    s = s.replace("&", " and ").replace("+", " and ")
    alias = ""
    m = ALIAS.search(s)
    if m:
        alias = s[m.end():]
        s = s[:m.start()] + " " + alias
    s = s.strip()
    dm = DOMAIN.match(s.lstrip("@").strip())
    is_dom = bool(dm) or s.startswith("@")
    if dm:
        s = dm.group(1)
    s = LONGNUM.sub(" ", s)
    toks = [_canon_legal(NAME_ABBR.get(t, t)) for t in TOKEN.findall(s)]
    legal = sorted({t for t in toks if t in LEGAL})
    core = [t for t in toks if t not in LEGAL and t not in STOP]
    if not core:
        core = [t for t in toks if t not in STOP] or toks
    atoks = [NAME_ABBR.get(t, t) for t in TOKEN.findall(LONGNUM.sub(" ", alias))]
    acore = [t for t in atoks if t not in LEGAL and t not in STOP]
    return (" ".join(toks), " ".join(core), " ".join(legal), " ".join(acore),
            " ".join(skeleton_word(t) for t in core), is_dom)


def norm_addr(raw):
    s = fold(map_nonlatin_addr(raw))
    s = s.replace("&", " and ")
    toks = []
    for t in TOKEN.findall(s):
        if t.isdigit():
            t = t.lstrip("0") or "0"
        else:
            m = re.match(r"^(\d+)(st|nd|rd|th)$", t)
            if m:
                t = m.group(1).lstrip("0") or "0"
            else:
                t = ADDR_ABBR.get(t, t)
        if t:
            toks.append(t)
    nums = [t for t in toks if t.isdigit()]
    return " ".join(toks), " ".join(nums)


def _process(args):
    names, addrs = args
    nn = [norm_name(x) for x in names]
    aa = [norm_addr(x) for x in addrs]
    return nn, aa


def normalize_frame(df, src, n_jobs=8):
    names = df.business_name.tolist()
    addrs = df.business_address.tolist()
    n = len(df)
    step = 50000
    chunks = [(names[i:i + step], addrs[i:i + step]) for i in range(0, n, step)]
    with Pool(n_jobs) as pool:
        res = pool.map(_process, chunks, chunksize=1)
    nn = [x for r in res for x in r[0]]
    aa = [x for r in res for x in r[1]]
    out = pd.DataFrame({
        "id": df.entity_id.values,
        "src": np.int8(src),
        "country": df.country.values,
        "name_raw": df.business_name.values,
        "addr_raw": df.business_address.values,
    })
    cols = ["name_norm", "name_core", "legal", "alias", "name_skel", "is_domain"]
    for j, c in enumerate(cols):
        out[c] = [x[j] for x in nn]
    out["addr_norm"] = [x[0] for x in aa]
    out["addr_nums"] = [x[1] for x in aa]
    out["name_nonlatin"] = df.business_name.str.contains(NONLATIN.pattern, regex=True).values
    out["addr_nonlatin"] = df.business_address.str.contains(NONLATIN.pattern, regex=True).values
    return out


def build(split, n_jobs=4, batch=1_000_000):
    """Stream each source file in batches; write parquet incrementally (bounded memory)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(CACHE, exist_ok=True)
    for k in (1, 2, 3):
        path = os.path.join(CACHE, f"norm_{split}_s{k}.parquet")
        if os.path.exists(path):
            continue
        src = os.path.join(DATA, split, f"{split}_source{k}.tsv")
        writer, n = None, 0
        for df in pd.read_csv(src, sep="\t", dtype=str, keep_default_na=False, quoting=3, chunksize=batch):
            out = normalize_frame(df, k, n_jobs=n_jobs)
            tbl = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path + ".tmp", tbl.schema)
            writer.write_table(tbl)
            n += len(out)
            del out, tbl, df
        writer.close()
        os.replace(path + ".tmp", path)
        print(split, k, n, flush=True)


def load_norm(split, k, columns=None, country=None):
    """Load normalised records; with `country`, only rows of that block are read (pyarrow row filter)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    filters = [("country", "==", country)] if country is not None else None
    tbl = pq.read_table(os.path.join(CACHE, f"norm_{split}_s{k}.parquet"), columns=columns, filters=filters)
    df = tbl.to_pandas()
    del tbl
    pa.default_memory_pool().release_unused()
    return df


if __name__ == "__main__":
    import memguard
    memguard.start()  # limits from MEM_LIMIT_GB / MEM_MIN_* env (defaults: 3 GB)
    for split in sys.argv[1:] or ["train", "test"]:
        build(split)
