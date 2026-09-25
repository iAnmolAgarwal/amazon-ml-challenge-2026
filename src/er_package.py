"""Validate a submission with the official validator and build the code zip required by the portal.

usage: python src/er_package.py submissions/NN_<attempt_id>.tsv [--final]

1. Runs the organisers' utils/validate_submission.py on the matching TSV and its .candidates.tsv
   against the raw test TSVs. Stops on FAIL.
2. Builds submissions/NN_<attempt_id>_code.zip:
     code/business_entity_resolution/src/          pipeline source (src/ + the attempt's train.py)
     code/business_entity_resolution/README.md     exact end-to-end reproduction steps
     code/business_entity_resolution/requirements.txt  pinned versions
   With --final it also adds output/matching_results.tsv, output/candidate_pairs.tsv and
   Documentation_template.md (docs/Documentation_template.md) at the zip root.
Never includes data, caches, credentials or git metadata.
"""
import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path

from common import ROOT, load_config
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

Python 3.12, x86-64 Linux (tested on AWS r7a.2xlarge: 8 cores, 64 GB RAM). No external data, APIs or pretrained models.

## Setup
```
python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```
Place the organisers' `dataset/` folder (train/ and test/ TSVs) under `data/` (any depth).

## Reproduce end to end
```
python src/prepare_raw.py                         # TSV -> parquet (cache/raw)
python src/er_norm.py --split train               # normalization views
python src/er_norm.py --split test
python src/er_block.py --split train              # candidate generation (blocking)
python src/er_block.py --split test
python src/er_feat.py --split train               # pair features
python src/er_feat.py --split test
python src/er_eval.py {attempt}                   # optional: 5-fold CV, macro F0.5
python src/er_predict.py {attempt}                # train on train, write submissions/NN_{attempt}.tsv (+ .candidates.tsv)
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

SRC_FILES = ["common.py", "er.py", "er_norm.py", "er_block.py", "er_feat.py", "er_feat2.py", "er_model.py", "er_eval.py", "er_predict.py", "er_package.py"]


def validate(mt, ct, cfg):
    v = validator_path(cfg)
    if v is None:
        raise SystemExit("official validator not found under data/")
    r = subprocess.run([sys.executable, str(v), "--matching", str(mt), "--candidate", str(ct), "--test-dir", str(raw_dir(cfg) / "test")], capture_output=True, text=True)
    print(r.stdout[-4000:], r.stderr[-2000:])
    if r.returncode != 0:
        raise SystemExit("VALIDATION FAILED - do not upload")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matching_tsv")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--skip-validate", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    mt = Path(a.matching_tsv).resolve()
    m = re.match(r"(\d+)_(.+)\.tsv$", mt.name)
    if not m or mt.name.endswith(".candidates.tsv"):
        raise SystemExit("expected submissions/NN_<attempt_id>.tsv")
    attempt = m.group(2)
    ct = mt.with_name(mt.stem + ".candidates.tsv")
    if not a.skip_validate:
        validate(mt, ct, cfg)
    z = mt.with_name(mt.stem + "_code.zip")
    base = "code/business_entity_resolution/"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in SRC_FILES:
            zf.write(ROOT / "src" / f, base + "src/" + f)
        zf.writestr(base + "src/prepare_raw.py", PREP)
        zf.write(ROOT / "config.yaml", base + "config.yaml")
        for f in (ROOT / "attempts" / attempt).glob("*"):
            if f.suffix in (".py", ".json") and not f.name.startswith("diag"):
                zf.write(f, base + f"attempts/{attempt}/{f.name}")
        zf.writestr(base + "README.md", README.format(attempt=attempt))
        zf.writestr(base + "requirements.txt", REQUIREMENTS)
        if a.final:
            zf.write(mt, "output/matching_results.tsv")
            zf.write(ct, "output/candidate_pairs.tsv")
            doc = ROOT / "docs" / "Documentation_template.md"
            if doc.exists():
                zf.write(doc, "Documentation_template.md")
    names = zipfile.ZipFile(z).namelist()
    assert not any(re.search(r"(\.pem|credentials|\.env|/cache/|/data/)", n) for n in names), "forbidden file in zip"
    print(f"built {z.name}: {len(names)} files, {z.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
