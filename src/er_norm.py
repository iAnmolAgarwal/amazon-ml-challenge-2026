"""Normalization v2 for business entity resolution.

Each record is turned into several side-by-side views (never one destructive string):
core name, website-free name, space-free keys (original and sorted token order), initials,
alternate name ("formerly"/DBA), normalized address, house number, postal code and a
combined text used by the word-view retrieval. Indic scripts are preserved; only Latin
diacritics are folded. Run as a script to build cache/norm/{split}_{s1,pool}.parquet.
"""
import argparse
import os
import re
import unicodedata
from multiprocessing import Pool
from pathlib import Path

import polars as pl

from common import load_config

TOKEN = re.compile(r"[0-9a-zÀ-ɏऀ-෿]+")
WEB = re.compile(r"(?:https?://)?(?:www\.)?([0-9a-z\-]+(?:\.[0-9a-z\-]+)*)\.(?:com|in|net|org|co|biz|info|us|fr|co\.in)\b")
ALT = re.compile(r"\b(?:formerly|previously|dba|d/b/a|aka|a\.k\.a\.|trading as|t/a|now known as)\b\s*:?\s*", re.I)
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
PREFIX = {"the", "m", "mr", "mrs", "ms", "miss", "shri", "sri", "shree", "smt", "dr", "prof", "m/s", "ms."}
LEGAL_JOIN = sorted(["privatelimited", "limited", "private", "pvtltd", "pvt", "ltd", "llc", "incorporated", "inc", "corporation", "corp", "company", "llp", "plc", "pllc", "sarl", "sasu", "sas", "eurl", "gmbh", "co"], key=len, reverse=True)
LEGAL = {
    "pvt", "private", "ltd", "limited", "llc", "l", "c", "inc", "incorporated", "corp", "corporation", "co", "company",
    "llp", "lp", "plc", "pllc", "pc", "pa", "sarl", "sas", "sa", "eurl", "sasu", "sci", "snc", "gmbh", "ag", "opc", "pte", "pty",
}
NAME_ABBR = {"&": "and", "intl": "international", "mfg": "manufacturing", "svcs": "services", "svc": "service", "bros": "brothers", "assoc": "associates", "mgmt": "management", "tech": "technologies"}
ADDR_ABBR = {
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane",
    "hwy": "highway", "pkwy": "parkway", "ct": "court", "pl": "place", "sq": "square", "ste": "suite", "fl": "floor", "apt": "apartment",
    "bldg": "building", "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest", "se": "southeast",
    "sw": "southwest", "mt": "mount", "ft": "fort", "nr": "near", "opp": "opposite", "ngr": "nagar", "clny": "colony", "sec": "sector",
    "sect": "sector", "rte": "route", "chem": "chemin", "imp": "impasse", "fbg": "faubourg", "cir": "circle", "trl": "trail", "ter": "terrace",
}
HOUSE = re.compile(r"(?:#|\b(?:h\.?\s?no|hno|house no|door no|d\.?\s?no|plot no|flat no|shop no|survey no|sy no|kh no|khasra no|no|number)\.?\s*[:\-]?\s*)(\d+[a-z]?)", re.I)
NUM = re.compile(r"\d+")
EMPTY = {"", "none", "null", "nan", "na", "n/a", "-"}


def fold(s):
    """Lowercase with NFKC, strip combining marks only when they follow a Latin letter (Indic vowel signs are kept)."""
    s = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", s or "")).lower()
    out, prev_latin = [], False
    for ch in s:
        if unicodedata.combining(ch) and prev_latin:
            continue
        out.append(ch)
        prev_latin = "a" <= ch <= "z" or "À" <= ch <= "ɏ"
    return unicodedata.normalize("NFC", "".join(out))


def deleet(tok):
    """Undo digit-for-letter typos inside tokens that also contain letters (michae1ine -> michaeline)."""
    return tok.translate(LEET) if any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok) else tok


def name_views(raw):
    """Return (core tokens, alt-name core tokens, is_web) for a business name."""
    s = fold(raw)
    alt = ""
    m = ALT.search(s)
    if m:
        s, alt = s[: m.start()], s[m.end():]
        if not TOKEN.search(s):
            s, alt = alt, ""
    is_web = 0
    w = WEB.search(s)
    if w:
        s = s[: w.start()] + " " + w.group(1).replace(".", " ").replace("-", " ") + " " + s[w.end():]
        is_web = 1

    def core(x):
        toks = [deleet(NAME_ABBR.get(t, t)) for t in TOKEN.findall(x.replace("&", " and "))]
        while toks and toks[0] in PREFIX:
            toks = toks[1:]
        kept = [t for i, t in enumerate(toks) if not ((t in LEGAL and (t != "co" or i > 0)) or t == "and")]
        return kept or toks

    return core(s), core(alt) if alt else [], is_web


def join_key(toks):
    """Space-free name with legal words stripped from both ends of the joined string (handles 'Li mited', 'Llc X')."""
    j = "".join(toks)
    changed = True
    while changed and j:
        changed = False
        for w in LEGAL_JOIN:
            if j.endswith(w) and len(j) > len(w) + 2:
                j, changed = j[: -len(w)], True
                break
            if j.startswith(w) and len(j) > len(w) + 2 and w in ("llc", "inc", "pvt", "ltd", "the"):
                j, changed = j[len(w):], True
                break
    return j


def addr_views(raw):
    """Return (address tokens, house number, postal code, all numbers) for an address."""
    s = fold(raw)
    if s.strip() in EMPTY:
        return [], "", "", []
    h = HOUSE.search(s)
    nums = [n.lstrip("0") or "0" for n in NUM.findall(s)]
    house = (h.group(1).lstrip("0") or "0") if h else (nums[0] if nums else "")
    postal = next((n for n in NUM.findall(s) if len(n) in (5, 6)), "")
    toks = [ADDR_ABBR.get(t, t) for t in TOKEN.findall(s)]
    toks = [t.lstrip("0") or "0" if t.isdigit() else t for t in toks]
    return toks, re.sub(r"[a-z]$", "", house), postal, nums


def normalize_rows(rows):
    """Normalize a list of (entity_id, name, address, country) tuples into view dicts."""
    out = []
    for eid, name, addr, country in rows:
        core, alt, is_web = name_views(name)
        atoks, house, postal, nums = addr_views(addr)
        stop = {"no", "number", "h", "hno", "house", "door", "plot", "flat", "shop", "unit", "survey", "kh", "khasra", "suite", "apartment", "floor", "d", "sy"}
        after = atoks[atoks.index(house) + 1 :] if house and house in atoks else atoks
        street = next((t for t in after if not t.isdigit() and t not in stop), "") or next((t for t in atoks if not t.isdigit() and t not in stop), "")
        out.append({
            "entity_id": eid,
            "country_n": fold(country).strip(),
            "name_core": " ".join(core),
            "name_alt": " ".join(alt),
            "name_join": join_key(core),
            "name_sorted": "".join(sorted(core)),
            "initials": "".join(t[0] for t in core) if len(core) >= 2 else "",
            "is_web": is_web,
            "addr_norm": " ".join(atoks),
            "house": house,
            "street1": street,
            "postal": postal,
            "nums": " ".join(sorted(set(nums))),
            "text": " ".join(core) + " " + " ".join(atoks),
        })
    return out


def normalize_frame(df, workers=None, chunk=50_000):
    """Normalize a polars frame with entity_id, business_name, business_address, country using all cores."""
    rows = list(zip(*(df[c].fill_null("").to_list() for c in ("entity_id", "business_name", "business_address", "country"))))
    parts = [rows[i : i + chunk] for i in range(0, len(rows), chunk)]
    with Pool(workers or os.cpu_count()) as p:
        res = p.map(normalize_rows, parts)
    return pl.DataFrame([r for part in res for r in part]).with_columns(pl.col("is_web").cast(pl.Int8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    cfg = load_config()
    raw = Path(cfg["cache_dir"]) / "raw"
    out = Path(cfg["cache_dir"]) / "norm"
    out.mkdir(parents=True, exist_ok=True)
    s1 = pl.read_parquet(raw / f"{a.split}_source1.parquet")
    pool = pl.concat([pl.read_parquet(raw / f"{a.split}_source{i}.parquet") for i in (2, 3)])
    if a.limit:
        s1, pool = s1.head(a.limit), pool.head(a.limit * 5)
    for tag, df in (("s1", s1), ("pool", pool)):
        n = normalize_frame(df)
        n.write_parquet(out / f"{a.split}_{tag}.parquet")
        print(f"{a.split}_{tag}: {n.shape}", flush=True)


if __name__ == "__main__":
    main()
