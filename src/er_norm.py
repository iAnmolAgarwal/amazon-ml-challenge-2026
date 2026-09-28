"""Normalization v2 for business entity resolution.

Each record is turned into several side-by-side views (never one destructive string):
core name, website-free name, space-free keys (original and sorted token order), initials,
alternate name ("formerly"/DBA), normalized address, house number, postal code and a
combined text used by the word-view retrieval. Native-script tokens learned by er_dict.py are
replaced by their Latin form first (name_lat keeps the latinized raw name); other Indic text is
preserved and only Latin diacritics are folded. Run as a script to build cache/norm/{split}_{s1,pool}.parquet.
"""
import argparse
import json
import os
import re
import unicodedata
from multiprocessing import Pool
from pathlib import Path

import polars as pl

from common import load_config

TOKEN = re.compile(r"[0-9a-zÀ-ɏऀ-෿]+")
WEB = re.compile(r"(?:https?://)?(?:www\.)?([0-9a-z\-]+(?:\.[0-9a-z\-]+)*)\.(?:com|in|net|org|co|biz|info|us|fr|co\.in)\b")
ALT = re.compile(r"(?:\b(?:formerly|previously|now)\s+known\s+as\b|\b(?:formerly|previously|dba|d/b/a|trading as|t/a)\b|\ba\.?k\.?a\.?(?!\w))\s*:?\s*", re.I)
MS = re.compile(r"^\W*m\s*[/.]\s*s\b\.?\s*", re.I)
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
PREFIX = {"the", "m", "mr", "mrs", "ms", "miss", "shri", "sri", "shree", "smt", "dr", "prof", "mme", "mlle"}
LEGAL = {
    "pvt", "private", "ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation", "co", "company",
    "llp", "lp", "plc", "pllc", "pc", "pa", "sarl", "sas", "sa", "eurl", "sasu", "sci", "snc", "gmbh", "ag", "opc", "pte", "pty",
    "selarl", "selas", "scp", "eirl",
}
NAME_ABBR = {"&": "and", "intl": "international", "mfg": "manufacturing", "svcs": "services", "svc": "service", "bros": "brothers", "assoc": "associates", "mgmt": "management", "tech": "technologies"}
ADDR_ABBR = {
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane",
    "hwy": "highway", "pkwy": "parkway", "ct": "court", "pl": "place", "sq": "square", "ste": "suite", "fl": "floor", "apt": "apartment",
    "bldg": "building", "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest", "se": "southeast",
    "sw": "southwest", "mt": "mount", "ft": "fort", "nr": "near", "opp": "opposite", "ngr": "nagar", "clny": "colony", "sec": "sector",
    "sect": "sector", "rte": "route", "chem": "chemin", "imp": "impasse", "fbg": "faubourg", "cir": "circle", "trl": "trail", "ter": "terrace",
}
HOUSE = re.compile(r"(?:#|\b(?:h\.?\s?no|hno|house no|door no|d\.?\s?no|plot no|flat no|shop no|survey no|sy no|kh no|khasra no|no|number)\.?\s*[:\-]?\s*)(\d+[a-z]?)\b", re.I)
FR_ABBR = {"st": "saint", "ste": "sainte", "r": "rue", "all": "allee", "ter": "ter", "bd": "boulevard", "boul": "boulevard", "av": "avenue", "ave": "avenue", "rte": "route", "chem": "chemin", "imp": "impasse", "pl": "place", "sq": "square", "fg": "faubourg", "fbg": "faubourg", "qu": "quai"}
FR_SKIP = {"rue", "avenue", "boulevard", "place", "chemin", "impasse", "allee", "route", "quai", "square", "faubourg", "cours", "passage", "de", "du", "des", "la", "le", "les", "l", "d", "bis", "ter", "quater", "saint", "sainte"}
FR_STREET = r"(?:rue|r|avenue|av|ave|bd|boulevard|boul|place|pl|chemin|chem|impasse|imp|allee|all|route|rte|quai|square|sq|faubourg|fbg|fg|cours|passage|voie|sentier|villa|hameau|lieu[- ]dit|residence|cite)"
FR_POSTREF = re.compile(r"\b(?:bp|cs|tsa|cedex)\s*\d+\b", re.I)
SUBUNIT = re.compile(r"\b(?:suite|ste|apt|apartment|unit|room|rm|fl|floor|po box|p\.?o\.? box|box)\.?\s*#?\s*\d+[a-z]?\b", re.I)
NUM = re.compile(r"\d+")
EMPTY = {"", "none", "null", "nan", "na", "n/a", "-"}
IND_RUN = re.compile(r"[\u0900-\u0963\u0966-\u0DFF\u200c\u200d]+")
LAT_RUN = re.compile(r"[0-9a-z\u00df-\u024f]+", re.I)
_MAPS = None


def indic_maps():
    """Token maps learned by er_dict.py from training gold pairs (run python src/er_dict.py first)."""
    global _MAPS
    if _MAPS is None:
        p = Path(load_config()["cache_dir"]) / "dict" / "indic.json"
        if not p.exists():
            raise SystemExit(f"{p} not found: run python src/er_dict.py first")
        _MAPS = json.loads(p.read_text())
    return _MAPS


def latinize(s, field, india):
    """Replace learned native-script tokens by their Latin form; on India names also map Latin spelling variants to one canon."""
    m = indic_maps()
    s = s or ""
    if m[field] and IND_RUN.search(s):
        s = IND_RUN.sub(lambda x: m[field].get(x.group(0).replace("\u200c", "").replace("\u200d", ""), x.group(0)), s)
    if india and field == "name" and m["canon"]:
        s = LAT_RUN.sub(lambda x: m["canon"].get(x.group(0).lower(), x.group(0)), s)
    return s


def fold(s):
    """Lowercase with NFKC, strip combining marks only when they follow a Latin letter (Indic vowel signs are kept)."""
    s = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", s or "")).lower().replace("\u0153", "oe").replace("\u00e6", "ae")
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


def merge_split_legal(toks):
    """Re-join legal words broken by stray spaces ('li mited', 'pri vate', 'l l c') when the concatenation is a legal word."""
    out, i = [], 0
    while i < len(toks):
        for n in (4, 3, 2):
            if i + n <= len(toks) and "".join(toks[i : i + n]) in LEGAL:
                out.append("".join(toks[i : i + n]))
                i += n
                break
        else:
            out.append(toks[i])
            i += 1
    return out


def strip_web_legal(tok):
    """Remove a trailing legal word glued inside a website token ('rangareddymillsprivate' -> 'rangareddymills')."""
    changed = True
    while changed:
        changed = False
        for w in ("privatelimited", "limited", "private", "pvtltd", "pvt", "ltd", "llc", "inc", "corporation", "corp", "company"):
            if tok.endswith(w) and len(tok) - len(w) >= 3:
                tok, changed = tok[: -len(w)], True
                break
    return tok


def name_views(raw, france=False):
    """Return (core tokens, alt-name core tokens, is_web) for a business name ('st' -> 'saint' for France)."""
    s = MS.sub(" ", fold(raw))
    alt = ""
    m = ALT.search(s)
    if m:
        s, alt = s[: m.start()], s[m.end():]
        if not TOKEN.search(s):
            s, alt = alt, ""
    is_web = 0
    web_tok = None
    w = WEB.search(s)
    if w:
        parts = w.group(1).replace("-", " ").split(".")
        web_tok = strip_web_legal(parts[-1].replace(" ", ""))
        s = s[: w.start()] + " " + " ".join(parts[:-1] + [web_tok]) + " " + s[w.end():]
        is_web = 1

    def core(x):
        toks = merge_split_legal([deleet(NAME_ABBR.get(t, t)) for t in TOKEN.findall(x.replace("&", " and "))])
        if france:
            toks = ["saint" if t == "st" else t for t in toks]
        while toks and toks[0] in PREFIX:
            toks = toks[1:]
        kept = [t for i, t in enumerate(toks) if not ((t in LEGAL and (t != "co" or i > 0)) or t == "and")]
        return kept or toks

    return core(s), core(alt) if alt else [], is_web


def skeleton(s):
    """Consonant skeleton of a Latin string: first letter kept, later vowels dropped, repeats collapsed (absorbs vowel typos and transliteration variants)."""
    s = "".join(ch for ch in s if "a" <= ch <= "z")
    return re.sub(r"(.)\1+", r"\1", s[:1] + "".join(ch for ch in s[1:] if ch not in "aeiouy")) if s else ""


def join_key(toks):
    """Space-free name from whole tokens (legal words already removed as tokens); undoes word splits and website gluing."""
    return "".join(toks)


def addr_views(raw, france=False):
    """Return (address tokens, house number, postal code, all numbers, raw house token) for an address.

    France (gated): postal-service references (BP/CS/TSA/CEDEX n) are not house numbers, house numbers
    accept bis/ter/quater suffixes, and 5-digit postcodes are skipped in the house fallback.
    """
    s = fold(raw)
    if france:
        s = FR_POSTREF.sub(" ", s)
    if s.strip() in EMPTY:
        return [], "", "", [], ""
    h = HOUSE.search(s)
    all_nums = NUM.findall(s)
    nums = [n.lstrip("0") or "0" for n in all_nums]
    if h:
        house_raw = h.group(1)
    else:
        rest = SUBUNIT.sub(" ", s)
        if france:
            first = re.search(r"\b(\d{1,4})\s*(?:bis|ter|quater|[a-z])?\s*,?\s*" + FR_STREET + r"\b", rest)
        else:
            first = re.search(r"\b(\d+[a-z]?)\b", rest)
        house_raw = first.group(1) if first else ""
    house = re.sub(r"[a-z]$", "", house_raw).lstrip("0") or ("0" if house_raw else "")
    cands = [n for n in all_nums if len(n) in (5, 6) and (n.lstrip("0") or "0") != house]
    postal = cands[-1] if cands else ""
    abbr = {**ADDR_ABBR, **FR_ABBR} if france else ADDR_ABBR
    toks = [abbr.get(t, t) for t in TOKEN.findall(s)]
    toks = [t.lstrip("0") or "0" if t.isdigit() else t for t in toks]
    return toks, house, postal, nums, house_raw.lstrip("0")


def normalize_rows(rows):
    """Normalize a list of (entity_id, name, address, country) tuples into view dicts."""
    out = []
    stop = {"no", "number", "h", "hno", "house", "door", "plot", "flat", "shop", "unit", "survey", "kh", "khasra", "suite", "apartment", "floor", "d", "sy"}
    for eid, name, addr, country in rows:
        france = fold(country).strip() == "france"
        india = fold(country).strip() == "india"
        name, addr = latinize(name, "name", india), latinize(addr, "addr", india)
        core, alt, is_web = name_views(name, france)
        atoks, house, postal, nums, house_tok = addr_views(addr, france)
        skip = stop | FR_SKIP if france else stop
        anchor = house_tok if house_tok in atoks else house
        after = atoks[atoks.index(anchor) + 1 :] if anchor and anchor in atoks else atoks
        street = next((t for t in after if not t[0].isdigit() and t not in skip), "") or next((t for t in atoks if not t[0].isdigit() and t not in skip), "")
        out.append({
            "entity_id": eid,
            "country_n": fold(country).strip(),
            "name_lat": name,
            "name_core": " ".join(core),
            "name_alt": " ".join(alt),
            "name_join": join_key(core),
            "name_sorted": "".join(sorted(core)),
            "initials": "".join(t[0] for t in core) if len(core) >= 2 else "",
            "name_skel": skeleton(join_key(core)),
            "name_fl": f"{core[0]}|{core[-1]}" if len(core) >= 2 else "",
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
    del rows
    with Pool(workers or os.cpu_count()) as p:
        res = p.map(_norm_part, parts)
    return pl.concat(res).with_columns(pl.col("is_web").cast(pl.Int8))


def _norm_part(part):
    """Worker: normalize one chunk and return it as a compact polars frame (keeps parent memory low)."""
    return pl.DataFrame(normalize_rows(part))


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
