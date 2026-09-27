"""Validate a submission with the official validator and build the code zip required by the portal.

usage: python src/er_package.py submissions/NN_<attempt_id>.tsv [--validate-only] [--test-dir D] [--skip-validate] [--final --team NAME]

1. Runs the organisers' utils/validate_submission.py --check-ids on the matching TSV and its .candidates.tsv
   against the raw test TSVs. Stops on FAIL or on any WARNING (skipped id check, missing candidates, match
   outside candidates); on success writes NN_<attempt_id>.validation.txt (both sha256 + validator output).
2. Builds submissions/NN_<attempt_id>_code.zip:
     code/business_entity_resolution/src/          pipeline source (src/ + the attempt's train.py)
     code/business_entity_resolution/README.md     exact end-to-end reproduction steps
     code/business_entity_resolution/requirements.txt  pinned versions
   With --final --team T it builds submissions/T_submission.zip instead, adding output/matching_results.tsv,
   output/candidate_pairs.tsv and the filled docs/Documentation_template.md. It requires a clean .validation.txt
   for these exact files, the run manifest NN_<attempt_id>.json with matching source hashes, and no template placeholders.
Never includes data, caches, credentials or git metadata.
"""
import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import er_model as M
from common import ROOT, load_attempt, load_config, read_json, sha256
from er import raw_dir, validator_path

REQUIREMENTS = """polars==1.44.2
polars-runtime-32==1.44.2
numpy==2.5.3
scipy==1.18.1
scikit-learn==1.9.1
lightgbm==4.7.0
rapidfuzz==3.14.6
sparse-dot-topn==1.2.0
pyarrow==25.0.1
pyyaml==6.0.3
psutil==7.2.2
"""

README = """# Business entity resolution pipeline ({attempt})

Python 3.12, x86-64 Linux (run on 12-vCPU / 96 GB RAM machines). No external data and no online APIs.
The cross-encoder steps need a GPU (run on Kaggle, 2x T4); their scripts are in src/ce/ and src/stack/kaggle/.
Pretrained weights: intfloat/multilingual-e5-small and intfloat/multilingual-e5-base (MIT), fine-tuned on the provided
training pairs only.

## Setup
```
python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```
Place the organisers' `dataset/` folder (train/ and test/ TSVs) under `data/` (any depth).

## Reproduce end to end
```
python src/prepare_raw.py                         # TSV -> parquet (cache/raw)
{pre}{evalline}
{predictline}
```
`matching_results.tsv` = submissions/NN_{attempt}.tsv; `candidate_pairs.tsv` = submissions/NN_{attempt}.candidates.tsv.
The attempt's settings are in `attempts/{attempt}/train.py`; defaults in `src/er_model.py`.
"""

PREP = '''"""Convert the organisers' TSVs to zstd parquet in cache/raw (all columns as strings)."""
from pathlib import Path

import polars as pl

from common import load_config
from er import raw_dir

cfg = load_config()
out = Path(cfg["cache_dir"]) / "raw"
out.mkdir(parents=True, exist_ok=True)
for split in ("train", "test"):
    for f in [f"{split}_source{i}" for i in (1, 2, 3)] + (["train_ground_truth"] if split == "train" else []):
        pl.scan_csv(raw_dir(cfg) / split / f"{f}.tsv", separator="\\t", quote_char=None, infer_schema_length=0).sink_parquet(out / f"{f}.parquet", compression="zstd")
        print(f)
'''

DOC = ROOT / "docs" / "Documentation_template.md"
PLACEHOLDERS = ("[Your Team Name]", "[List all team members]", "[Date]", "[e.g.,", "[total]", "[]", "[brief description]", "[your best validation score]", "[Brief description", "[Blocking + Classifier", "*Provide a brief", "*Describe how", "*Key insights", "*Outline your", "*Summarize your", "*Include any", "*Your complete")
SRC_DIRS = ["stack", "ce"]
BANNED = re.compile(r"harsimran|akshit|nitya|anmol|apprentice|partner pipeline|teammate|team ?mate|handoff|second pipeline|their pipeline|claude|codex|anthropic|co-authored", re.I)
SRC_FILES = ["common.py", "er.py", "er_dict.py", "er_norm.py", "er_block.py", "er_feat.py", "er_feat2.py", "er_feat3.py", "er_feat4.py", "er_feat5.py", "er_feat6.py", "er_lift.py", "er_stage3.py", "er_retrieve.py", "er_newpairs.py", "er_lexical.py", "er_keys.py", "er_rtext.py", "er_pool.py", "er_gen.py", "er_comb.py", "er_ce.py", "er_meta.py", "er_france.py", "er_model.py", "er_eval.py", "er_predict.py"]


def validate(mt, ct, cfg, test_dir):
    v = validator_path(cfg)
    need = [mt, ct] + [test_dir / f"test_source{i}.tsv" for i in (1, 2, 3)]
    if v is None or not all(p.exists() for p in need):
        raise SystemExit(f"validator or inputs missing: {[str(p) for p in need if not p.exists()]}")
    r = subprocess.run([sys.executable, str(v), "--matching", str(mt), "--candidate", str(ct), "--test-dir", str(test_dir), "--check-ids"], capture_output=True, text=True)
    print(r.stdout[-4000:], r.stderr[-2000:])
    if r.returncode != 0 or "WARNING" in r.stdout or "PASS" not in r.stdout:
        raise SystemExit("VALIDATION FAILED or WARNED - do not upload")
    mt.with_suffix(".validation.txt").write_text(f"{sha256(mt)}  {mt.name}\n{sha256(ct)}  {ct.name}\n{r.stdout}")


def final_checks(mt, ct, man, attempt):
    rec = mt.with_suffix(".validation.txt")
    txt = rec.read_text() if rec.exists() else ""
    if not ct.exists() or txt.split("\n")[:2] != [f"{sha256(mt)}  {mt.name}", f"{sha256(ct)}  {ct.name}"] or "WARNING" in txt or "PASS" not in txt:
        raise SystemExit(f"--final needs {ct.name} and a clean {rec.name} for these exact files")
    if not DOC.exists() or any(p in DOC.read_text() for p in PLACEHOLDERS):
        raise SystemExit("fill docs/Documentation_template.md first")
    now = {f: sha256(ROOT / "src" / f) for f in SRC_FILES if f != "er_package.py"} | {"train.py": sha256(ROOT / "attempts" / attempt / "train.py")}
    stale = [f for f in now if man.get("src_sha256", {}).get(f) != now[f]]
    if not man or stale or man.get("matching_sha256") != sha256(mt) or man.get("candidates_sha256") != sha256(ct):
        raise SystemExit(f"outputs or source differ from the run manifest {mt.with_suffix('.json').name}: {stale or 'missing/mismatched manifest'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matching_tsv")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--test-dir")
    ap.add_argument("--team")
    a = ap.parse_args()
    cfg = load_config()
    mt = Path(a.matching_tsv).resolve()
    m = re.match(r"(\d+)_(.+)\.tsv$", mt.name)
    if not m or mt.name.endswith(".candidates.tsv"):
        raise SystemExit("expected submissions/NN_<attempt_id>.tsv")
    attempt = m.group(2)
    ct = mt.with_name(mt.stem + ".candidates.tsv")
    if not a.skip_validate:
        validate(mt, ct, cfg, Path(a.test_dir) if a.test_dir else raw_dir(cfg) / "test")
    if a.validate_only:
        return
    man = read_json(mt.with_suffix(".json"), {})
    if a.final:
        if not a.team:
            raise SystemExit("--final needs --team")
        final_checks(mt, ct, man, attempt)
    z = mt.with_name(f"{a.team}_submission.zip" if a.final else mt.stem + "_code.zip")
    tmp = z.with_suffix(".zip.tmp")
    base = "code/business_entity_resolution/"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in SRC_FILES:
            zf.write(ROOT / "src" / f, base + "src/" + f)
        for d in SRC_DIRS:
            for f in sorted((ROOT / "src" / d).rglob("*")):
                if f.is_file() and "__pycache__" not in f.parts and f.suffix in (".py", ".md", ".yaml", ".txt", "") and not f.name.startswith("."):
                    zf.write(f, base + "src/" + str(f.relative_to(ROOT / "src")))
        zf.writestr(base + "src/prepare_raw.py", PREP)
        zf.write(ROOT / "config.yaml", base + "config.yaml")
        mod = load_attempt(attempt)
        for dep in [attempt] + list(getattr(mod, "DEPENDS", [])):
            for f in (ROOT / "attempts" / dep).glob("*"):
                if f.suffix in (".py", ".json") and not f.name.startswith("diag"):
                    zf.write(f, base + f"attempts/{dep}/{f.name}")
        pre = "".join(f"{c}\n" for c in getattr(mod, "PRE_STEPS", []))
        frac = man.get("train_frac", getattr(mod, "TRAIN_FRAC", M.TRAIN_FRAC))
        evalline = getattr(mod, "EVAL", f"python src/er_eval.py {attempt}                   # optional: 5-fold CV, macro F0.5")
        predictline = getattr(mod, "PREDICT", f"python src/er_predict.py {attempt} --train-frac {frac}  # train on train, write submissions/NN_{attempt}.tsv (+ .candidates.tsv)")
        zf.writestr(base + "README.md", README.format(attempt=attempt, pre=pre, evalline=evalline, predictline=predictline))
        zf.writestr(base + "requirements.txt", REQUIREMENTS)
        if a.final:
            zf.write(mt, "output/matching_results.tsv")
            zf.write(ct, "output/candidate_pairs.tsv")
            zf.write(DOC, "Documentation_template.md")
    names = zipfile.ZipFile(tmp).namelist()
    with zipfile.ZipFile(tmp) as zf:
        hits = [n for n in names if n != "Documentation_template.md" and not n.startswith("output/") and BANNED.search(zf.read(n).decode("utf-8", "ignore"))]
    if hits:
        tmp.unlink()
        raise SystemExit(f"names or banned wording in: {hits}")
    if any(re.search(r"(\.pem|credentials|\.env|/cache/|/data/)", n) for n in names):
        tmp.unlink()
        raise SystemExit("forbidden file in zip")
    tmp.replace(z)
    print(f"built {z.name}: {len(names)} files, {z.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
