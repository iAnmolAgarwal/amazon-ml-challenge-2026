"""Validate a submission with the official validator and build the code zip required by the portal.

usage: python src/er_package.py submissions/NN_<attempt_id>.tsv [--validate-only] [--test-dir D] [--skip-validate] [--final --team NAME]

1. Runs the organisers' utils/validate_submission.py --check-ids on the matching TSV and its .candidates.tsv
   against the raw test TSVs. Stops on FAIL or on any WARNING (skipped id check, missing candidates, match
   outside candidates); on success writes NN_<attempt_id>.validation.txt (both sha256 + validator output).
2. Builds submissions/NN_<attempt_id>_code.zip:
     code/business_entity_resolution/src/          pipeline source (SRC_FILES, SRC_DIRS, prepare_raw.py)
     code/business_entity_resolution/attempts/     the attempt's train.py and those of its DEPENDS
     code/business_entity_resolution/README.md     exact end-to-end reproduction steps
     code/business_entity_resolution/requirements.txt, requirements-gpu.txt, config.yaml
   With --final --team T it builds submissions/T_submission.zip instead, adding output/matching_results.tsv,
   output/candidate_pairs.tsv, the filled docs/Documentation_template.md, the run manifest (run_manifest.json),
   ARTIFACTS (the second model family's outputs, as artifacts/stack/) and SHA256SUMS. It requires a clean
   .validation.txt for these exact files, the run manifest NN_<attempt_id>.json with matching source hashes, and no
   template placeholders.
Never includes data, caches other than ARTIFACTS, credentials or git metadata.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import er_model as M
from common import ROOT, load_attempt, load_config, sha256
from harness_util import read_json
from er import raw_dir, validator_path

REQUIREMENTS = """# main pipeline (CPU), Python 3.12.14
polars==1.44.2
polars-runtime-32==1.44.2
numpy==2.5.3
scipy==1.18.1
scikit-learn==1.9.1
lightgbm==4.7.0
rapidfuzz==3.14.6
sparse-dot-topn==1.2.0
pyarrow==25.0.1
pyyaml==6.0.3
"""

REQUIREMENTS_GPU = """# cross-encoder steps of both families (src/ce/, src/stack/kaggle/): Kaggle GPU notebook, 2x T4
torch==2.14.0
transformers==5.17.0
pandas==3.0.6
pyarrow==25.0.1
scikit-learn==1.9.1
numpy==2.4.6
"""

CONFIG = """data_dir: data
cache_dir: cache
seed: 42
"""

README = """# Business entity resolution pipeline ({attempt})

Run every command from this folder (`code/business_entity_resolution/`).

## Package contents
- `src/`: the pipeline (Python modules run in the order below); `src/ce/`: cross-encoder GPU scripts; `src/single/`: the
  single flat LightGBM used as a meta-model input; `src/stack/`: the second model family (own environment and README).
- `attempts/<id>/train.py`: the settings of each stage named in the command list (loaded by `src/common.py`).
- `artifacts/stack/`: the second model family's outputs used by the submission (13 parquet files, about 455 MB), so the
  second family does not have to be rerun (`src/stack/README.md` regenerates them). The main family's own GPU steps
  still run on Kaggle (table below).
- `run_pipeline.sh`: the command list below as a script, in phases split at the GPU steps.
- `artifacts/final/final_probs.parquet`: the final probability of every candidate pair, for the quick check below.
- `config.yaml` (data and cache folders, seed), `requirements.txt` (CPU), `requirements-gpu.txt` (GPU steps).
- `run_manifest.json`: the record of the final step (git commit, SHA256 of every source file, of the attempt settings
  and of both outputs, library versions, runtime). `SHA256SUMS`: every file of the package (check with `sha256sum -c SHA256SUMS`).

## Overview (data -> blocking -> matching -> output)
1. **Data**: the organisers' TSVs are converted to parquet; names and addresses are normalized (native-script words mapped
   to Latin with a dictionary learned from training pairs only).
2. **Blocking**: per country, word TF-IDF neighbours plus exact keys, character 3-gram retrieval, extra name keys and
   reverse character retrieval. Word TF-IDF and exact keys give 78.9M test pairs (46 per S1); the other routes'
   pairs are kept if a route classifier gives p > 0.001.
3. **Matching**: stage-1 and stage-2 LightGBM pair models filter the raw pairs (stage-2 probability >= 0.001); on that
   band, LightGBM models (band model, retrieval-route classifier, lexical and generator-operator corrections), a
   fine-tuned cross-encoder score, a second model family (src/stack/), a cross-fitted LightGBM meta-model combining both
   families, an empty-address correction, and for France (no training labels) a blend of both families.
4. **Output**: expected-F0.5 decision (one owner per pool record, best prefix per S1, floor 0.6) writes
   matching_results.tsv.

## Candidate file (exact rule)
`candidate_pairs.tsv` lists every pair the final models run inference over:
- US and India: the meta-model's input, i.e. our chain's pairs (stage-2 probability >= 0.001, or added after stage 2 by
  retrieval, extra name keys or reverse retrieval) whose chain probability is >= 0.001, plus every candidate pair of
  the second family (its cascade: stage-2 p > 0.02 and top 10 per S1);
- France: our chain's France pairs with stage-2 probability >= 0.001 or added after stage 2, plus the second family's
  France candidates (the France blend combines the two).
The decision step uses the same set except 58,364 second-family pairs that our stage 2 scored below 0.001: they are
scored by the final models but never matched. Totals: 11,837,012 pairs, 6.83 per S1 (US 5.63, India 6.06,
France 12.31; France is larger because its chain pairs are not cut at chain probability 0.001). Every matched pair is
in the file.

## Models and licences
- LightGBM 4.7.0 (MIT) for every tree model.
- intfloat/multilingual-e5-small (MIT, ~118M parameters, revision 614241f622f53c4eeff9890bdc4f31cfecc418b3) and
  intfloat/multilingual-e5-base (MIT, ~278M parameters, revision d128750597153bb5987e10b1c3493a34e5a4502a), fine-tuned
  as pair cross-encoders on the provided training pairs only (the revisions current on 26 Sep 2026).
- No external data, no online APIs, no other pretrained models. Libraries are permissively licensed except Unidecode
  (GPL-2.0), a transliteration library used by the second family.

## Environments, hardware and compute
- Main pipeline: Python 3.12.14 on x86-64 Linux (Ubuntu 22.04), `pip install -r requirements.txt`. We used
  12-vCPU / 96 GB machines. CPU time is about 30-55 hours on one such machine (we spread the steps over four); the
  longest steps are blocking, stage-1/2 CV, character retrieval, the generator-operator features and the single-model
  build (1-2.5 h each). Disk: about 120 GB for `cache/` (pair features 52 GB, out-of-fold probabilities 30 GB,
  single-model matrices 32 GB, candidates 3.9 GB, normalized records 2.7 GB).
- GPU steps: Kaggle notebooks with 2x T4 and the default image (`requirements-gpu.txt`), about 15-20 GPU-hours in
  total (table below). The notebooks find their inputs under `/kaggle/input/**`; upload the listed folders as datasets.
- Second model family (src/stack/): Python 3.11, `src/stack/requirements.txt`, run from `src/stack/` as described in
  src/stack/README.md; its outputs are shipped in `artifacts/stack/`.
- Data: place the organisers' `dataset/` folder (train/ and test/ TSVs) anywhere under `data/`; the second family
  expects it at `src/stack/dataset/{{train,test}}/` (e.g. `ln -s ../../data/<...>/dataset src/stack/dataset`).

| # | GPU run | Script and settings | Time |
|---|---|---|---|
| 1 | e5-small, train two halves | `src/ce/ce_train.py`, ce_cfg.json `{{"budget_min": 170, "maxn": 6200000}}` | ~3 h |
| 2 | e5-small, score the band | `src/ce/ce_score.py` | ~22 min |
| 3 | e5-base, train two halves | `src/ce/ce_train.py`, ce_cfg.json `{{"model": "intfloat/multilingual-e5-base", "lr": 3e-5, "budget_min": 140, "maxn": 6200000}}` | ~2.5 h |
| 4 | e5-base, score the band | `src/ce/ce_score.py` | not recorded (< 1 h) |
| 5-8 | both models, score the added pairs | `src/ce/ce_score.py` with the four score_cfg.json files of the command list | not recorded (< 1 h each) |
| 9 | second family e5s | `src/stack/kaggle/run.py e5s` | ~47 min |
| 10 | second family e5i | `src/stack/kaggle/run.py e5i` | 70 min-2 h |
| 11 | second family e5b | `src/stack/kaggle/run.py e5b` | ~4.7 h |
| 12 | second family e5bc | `src/stack/kaggle/run.py e5bc` | not recorded (75-min training budget per model) |

## Setup
```
python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```
Keep `cache_dir: cache` in `config.yaml`: several steps use the relative path `cache/` (to use another disk, make
`cache/` a symlink).

## Reproduce end to end
`bash run_pipeline.sh 1`, then the GPU step it prints, `bash run_pipeline.sh 2`, and so on; the last phase copies both
outputs to `../../output/` and runs the organisers' validator. The same commands, in order:
```
python src/prepare_raw.py                         # TSV -> parquet (cache/raw)
{pre}{evalline}
{predictline}
```
Then copy the outputs to the package root (`NN` is the number the last command prints):
`cp submissions/archive/NN_{attempt}.tsv ../../output/matching_results.tsv && cp submissions/archive/NN_{attempt}.candidates.tsv ../../output/candidate_pairs.tsv`
and check them with the organisers' validator (`utils/validate_submission.py` in the challenge download) with `--check-ids`.
The attempt's settings are in `attempts/{attempt}/train.py`; defaults in `src/er_model.py`.

## Quick check of the last step (about a minute)
`artifacts/final/final_probs.parquet` holds every candidate pair with its final probability and whether the decision
considers it. With the organisers' dataset under `data/`:
```
python src/verify_final.py        # rebuilds both TSVs into verify/ and compares their SHA256 with output/
```
It runs the same decision and writing code as the last command of the list below (`er_predict.py`).

## Verification
Expected 5-fold CV macro F0.5 (train S1 of US + India, harness folds, the same decision rule) after each stage:

| Stage | Command | CV |
|---|---|---|
| stage 1 | `er_eval.py 005n_indic_nocoh` | 0.97008 |
| stage 2 | `er_eval.py 006_stage2` | 0.97423 |
| band model | `er_stage3.py 016_stage2b` | 0.98589 |
| lexical correction | `er_lexical.py 018_lexical --cv` | 0.98604 |
| generator-operator + pool-side correction | `er_comb.py 023_comb` | 0.98742 |
| cross-encoder stack | `er_ce.py 028_ce` | 0.98779 |
| meta-model over both families | `er_meta.py 027p_meta` | 0.99177 |
| empty-address correction (final) | `er_fix.py 030p_fix` | 0.99184 (US 0.99164, India 0.99213) |

The meta-model is cross-fitted on the main family's stratified folds, while the second family's out-of-fold inputs use
its own folds (crc32 of the S1 id): every input is out-of-fold, but not on the same folds, so the meta-model CV may be
slightly optimistic.
The oracle macro F0.5 of the final train candidate set (13.4M pairs, 6.06 per S1) is 0.99798
(`attempts/{attempt}/results.json`). France has no labels, so it has no CV.

What has been rerun:
- For this package, the final step (`{predictline_short}`) was rerun from the saved probabilities; it writes
  matching_results.tsv byte-identical to the leaderboard file (SHA256 in `SHA256SUMS` and `run_manifest.json`).
- An independent rebuild from the raw data on a second machine reproduced the blocking exactly (98,021,508 train pairs,
  same dictionary, same per-S1 oracle), stage 1 (CV 0.97005 vs 0.97008) and stage 2 (0.97425 vs 0.97423); the small
  differences are floating-point variation between CPU types.
- From this unpacked package (28 Sep), the last six steps (`ce_export.py ce2`, `er_meta.py 027p_meta`,
  `er_fix.py 030p_fix`, `er_france.py 029f5_final`, `er_fix.py 030q_fix --no-cv`, `er_predict.py 029f6_final
  --from-probs 030q_fix`) were rerun from the cached inputs of the earlier steps into an empty output folder: every
  intermediate probability file and both output TSVs came out byte-identical (meta-model CV 0.99177 reproduced).
- Not rerun after the fact: the GPU steps. They stop on a wall-clock budget in fp16, so a rerun reproduces the
  cross-encoder scores closely but not bit-identically. The second family's outputs are shipped; regenerating them
  has the limits listed in src/stack/README.md ("Known reproduction gaps").

## Notes
- Every folder in `attempts/` is used by the command list. Module options that no shipped attempt sets are research
  options from the challenge and are off by default.
- `028e_ce` and `028f_ce` are output names written by `src/er_extend.py`, not attempt folders.
- The France word lists (VETO_WORDS, BROAD_WORDS) are printed by `python src/er_frwords.py` from France's own test
  candidate pairs, without labels.
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

def runner(attempt, pre, predict):
    """run_pipeline.sh: the README command list as bash phases, split at the manual GPU (Kaggle) steps."""
    phases, gpu, cur, notes = [], [], ["python src/prepare_raw.py"], []
    for c in pre:
        if c.lstrip().startswith("#"):
            notes.append(c.lstrip()[1:].strip())
        else:
            if notes:
                phases.append((cur, notes))
                cur, notes = [], []
            cur.append(c)
    last = f"ls submissions/archive | grep -E '^[0-9]+_{attempt}\\.tsv$' | sort -n | tail -1"
    cur += [predict.split("#")[0].strip(),
            f'N=$({last})',
            "mkdir -p ../../output && cp \"submissions/archive/$N\" ../../output/matching_results.tsv && cp \"submissions/archive/${N%.tsv}.candidates.tsv\" ../../output/candidate_pairs.tsv",
            "V=$(find -L data -name validate_submission.py | head -1); T=$(dirname \"$(find -L data -name test_source1.tsv | head -1)\")",
            "if [ -n \"$V\" ]; then python \"$V\" --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir \"$T\" --check-ids; fi"]
    phases.append((cur, []))
    out = ["#!/usr/bin/env bash",
           "# The command list of README.md as a script, in phases split at the manual GPU steps (Kaggle notebooks).",
           f"# usage: bash run_pipeline.sh <phase 1-{len(phases)}>; after each phase except the last, do the GPU step it prints.",
           "# Run from code/business_entity_resolution/ with the organisers' dataset under data/; python = the main environment.",
           "set -euo pipefail", 'cd "$(dirname "$0")"', ""]
    for i, (cmds, gnotes) in enumerate(phases, 1):
        out.append(f"phase_{i}() {{")
        out += [f"  {c}" for c in cmds]
        if gnotes:
            out.append("  cat <<'GPU'")
            out.append(f"Phase {i} done. Next, the GPU step (Kaggle, see README), then: bash run_pipeline.sh {i + 1}")
            out += [f"- {g}" for g in gnotes]
            out.append("GPU")
        else:
            out.append('  echo "done: ../../output/matching_results.tsv and ../../output/candidate_pairs.tsv"')
        out += ["}", ""]
    out.append('case "${1:-}" in')
    out += [f"  {i}) phase_{i} ;;" for i in range(1, len(phases) + 1)]
    out += [f'  *) echo "usage: bash run_pipeline.sh <phase 1-{len(phases)}>"; exit 1 ;;', "esac", ""]
    return "\n".join(out)


DOC = ROOT / "docs" / "Documentation_template.md"
PLACEHOLDERS = ("{CV_", "[final]", "[update", "[Your Team Name]", "[List all team members]", "[Date]", "[e.g.,", "[total]", "[]", "[brief description]", "[your best validation score]", "[Brief description", "[Blocking + Classifier", "*Provide a brief", "*Describe how", "*Key insights", "*Outline your", "*Summarize your", "*Include any", "*Your complete")
SRC_DIRS = ["stack", "ce", "single"]
ARTIFACTS = ("stack", "p_*.parquet", "scores_*.parquet")
BANNED = re.compile(r"harsimran|akshit|nitya|anmol|apprentice|partner pipeline|teammate|team ?mate|handoff|second pipeline|their pipeline|claude|codex|anthropic|co-authored", re.I)
SRC_FILES = ["common.py", "er.py", "er_dict.py", "er_norm.py", "er_block.py", "er_feat.py", "er_feat2.py", "er_feat4.py", "er_feat5.py", "er_feat6.py", "er_lift.py", "er_stage3.py", "er_retrieve.py", "er_newpairs.py", "er_lexical.py", "er_keys.py", "er_rtext.py", "er_pool.py", "er_gen.py", "er_comb.py", "er_ce.py", "er_meta.py", "er_frwords.py", "er_france.py", "er_extend.py", "er_fix.py", "er_model.py", "er_eval.py", "er_predict.py", "verify_final.py"]


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
    mod = load_attempt(attempt)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in SRC_FILES:
            zf.write(ROOT / "src" / f, base + "src/" + f)
        for d in SRC_DIRS:
            for f in sorted((ROOT / "src" / d).rglob("*")):
                if f.is_file() and "__pycache__" not in f.parts and f.suffix in (".py", ".md", ".yaml", ".txt", "") and not f.name.startswith("."):
                    zf.write(f, base + "src/" + str(f.relative_to(ROOT / "src")))
        zf.writestr(base + "src/prepare_raw.py", PREP)
        zf.writestr(base + "config.yaml", CONFIG)
        for dep in [attempt] + list(getattr(mod, "DEPENDS", [])):
            if not (ROOT / "attempts" / dep / "train.py").exists():
                raise SystemExit(f"DEPENDS names a missing attempt folder: {dep}")
            for f in sorted((ROOT / "attempts" / dep).glob("*")):
                if f.suffix in (".py", ".json") and not f.name.startswith("diag"):
                    zf.write(f, base + f"attempts/{dep}/{f.name}")
        pre = "".join(f"{c}\n" for c in getattr(mod, "PRE_STEPS", []))
        frac = man.get("train_frac", getattr(mod, "TRAIN_FRAC", M.TRAIN_FRAC))
        evalline = getattr(mod, "EVAL", f"python src/er_eval.py {attempt}                   # optional: 5-fold CV, macro F0.5")
        predictline = getattr(mod, "PREDICT", f"python src/er_predict.py {attempt} --train-frac {frac}  # train on train, write submissions/NN_{attempt}.tsv (+ .candidates.tsv)")
        zf.writestr(base + "README.md", README.format(attempt=attempt, pre=pre, evalline=evalline, predictline=predictline, predictline_short=predictline.split("#")[0].strip()))
        zf.writestr(base + "run_pipeline.sh", runner(attempt, getattr(mod, "PRE_STEPS", []), predictline))
        zf.writestr(base + "requirements.txt", REQUIREMENTS)
        zf.writestr(base + "requirements-gpu.txt", REQUIREMENTS_GPU)
        if a.final:
            zf.write(mt, "output/matching_results.tsv")
            zf.write(ct, "output/candidate_pairs.tsv")
            zf.write(DOC, "Documentation_template.md")
            man_z = dict(man, src_sha256={k: v for k, v in man["src_sha256"].items() if k in SRC_FILES or k == "train.py"})
            zf.writestr(base + "run_manifest.json", json.dumps(man_z, indent=1))
            probs = mt.with_suffix(".probs.parquet")
            zf.write(probs, base + "artifacts/final/final_probs.parquet", compress_type=zipfile.ZIP_STORED)
            src = Path(load_config()["cache_dir"]) / ARTIFACTS[0]
            arts = sorted(f for g in ARTIFACTS[1:] for f in src.glob(g))
            if len(arts) != 13:
                raise SystemExit(f"expected 13 second-family files in {src}, found {len(arts)}")
            for f in arts:
                zf.write(f, base + f"artifacts/{ARTIFACTS[0]}/{f.name}", compress_type=zipfile.ZIP_STORED)
    if a.final:
        with zipfile.ZipFile(tmp, "a", zipfile.ZIP_DEFLATED) as zf:
            rel = lambda n: n[len(base):] if n.startswith(base) else "../../" + n
            zf.writestr(base + "SHA256SUMS", "".join(f"{hashlib.sha256(zf.read(n)).hexdigest()}  {rel(n)}\n" for n in sorted(zf.namelist())))
    names = zipfile.ZipFile(tmp).namelist()
    with zipfile.ZipFile(tmp) as zf:
        hits = [n for n in names if n != "Documentation_template.md" and not n.startswith("output/") and not n.endswith(".parquet") and BANNED.search(zf.read(n).decode("utf-8", "ignore"))]
    if hits:
        tmp.unlink()
        raise SystemExit(f"names or banned wording in: {hits}")
    if any(re.search(r"(\.pem|credentials|\.env|/cache/|/data/)", n) for n in names) or any(n.endswith(".parquet") and not re.search(r"/artifacts/(stack|final)/", n) for n in names):
        tmp.unlink()
        raise SystemExit("forbidden file in zip")
    tmp.replace(z)
    print(f"built {z.name}: {len(names)} files, {z.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
