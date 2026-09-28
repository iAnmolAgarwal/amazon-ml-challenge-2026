# Evaluator A: reproducibility and engineering quality

Package: `submissions/final_package/Apprentice_submission.zip` (team "Apprentice", 659 MB, 122 entries). I judged only what is inside the zip. Paths below are relative to `code/business_entity_resolution/` unless they start with `output/` or `Documentation_template.md`.

Bar: top 50 of about 1,000 teams (top 5%), judged strictly.

## Scores

| Criterion | Score | One-line reason |
|---|---|---|
| Reproducibility | **7 / 10** | The last step rebuilds both outputs byte for byte in 16 s, and all hashes match. A full rebuild takes about 81 hand-typed commands, about 12 manual Kaggle kernels and 45-75 compute hours. The GPU steps are not deterministic, and the second family's shipped outputs come from commands the team itself says were not logged or run end to end. |
| README clarity | **7.5 / 10** | Accurate and complete: every script, flag and attempt resolves; there is an expected CV per stage, an exact candidate-file rule, hardware, runtime, disk and licences. It is also a dense wall of commands with opaque stage IDs, and it overclaims once (see H3). |
| Dependency pinning | **8 / 10** | Every direct import is pinned in three requirement files, and the Python versions are stated. There is no lock for transitive dependencies and no pinned Kaggle image. |
| Code quality | **6 / 10** | Consistent conventions, a clear docstring in every module, seeds, assertions at the joins. Against that: 235 lines over 150 characters (up to 408), no tests, dead code, leftover research branches that name scripts not shipped, repeated helpers, and `cache/` paths that bypass the config. |
| Structure | **7 / 10** | Matches the organisers' layout, with a clear split into `src/`, `ce/`, `single/`, `stack/`, `attempts/` and `artifacts/`. The two families use different conventions (polars vs pandas, attempt modules vs YAML, different fold schemes), there is a nested `src/stack/src/`, and stage IDs do not follow run order. |
| **Overall** | **7 / 10** | |

**Verdict: would advance to top tier on this lens, by a thin margin.** Few of 1,000 packages will offer a one-minute byte-identical check, a full SHA256 manifest, per-stage expected CV and an honest list of gaps. The item that could drop it to *borderline* with a strict panel is H2: the provenance of the second family's shipped outputs, which the final model depends on. H1 and H3 are the other two points a judge who tries to rerun will hit first.

## What I checked (evidence)

- **Integrity.** All 121 entries of `SHA256SUMS` match (108 extracted files with `shasum -c`, plus the 13 `artifacts/stack/*.parquet` hashed from the zip stream). `run_manifest.json` `matching_sha256` / `candidates_sha256` equal the shipped `output/` files, and every `src_sha256` entry matches the shipped source.
- **Quick check run.** With the organisers' dataset linked under `data/` and the pinned versions (polars 1.44.2, lightgbm 4.7.0, numpy 2.5.3, Python 3.12.13), I ran `python src/verify_final.py`. Both rebuilt TSVs are **byte-identical** to `output/`: 16 s, 2.0 GB peak memory.
- **Organisers' validator.** Run on `output/` with `--candidate` and without `--check-ids`: **PASS**. 1,732,544 rows each; 99,885 empty matches and 23,666 empty candidate rows.
- **Static checks.**
  - All 67 `.py` files under `src/` and the 16 attempt files compile (`py_compile`), and so do the worker scripts embedded as strings in `src/ce/ce_train.py` and `src/ce/ce_score.py`.
  - I checked every `python src/...` line of the README: all 81 invocations name existing scripts, every flag exists in its script's argparse, and every attempt ID exists. `028e_ce` and `028f_ce` are output names, as README:119 and README:186 say.
  - I did the same for all 27 script invocations in the `src/stack/README.md` run order.
- **Data-dependency order.** I traced every read and write through the README list (README:81-139): raw, dict, norm, block, feat, stage 1, stage 2, lift, band, route, lexical, gen, comb, keys, CE, meta, fix, France, fix, predict. Every input is produced by an earlier line. No step reads a file that is written later.
- **Import scan against requirements.**
  - Main `src/` needs polars, numpy, scipy, sklearn, lightgbm, rapidfuzz, sparse_dot_topn and yaml. All are in `requirements.txt`.
  - The GPU scripts need torch, transformers, pandas, numpy and sklearn. All are in `requirements-gpu.txt`.
  - `src/stack/` also needs pyarrow, unidecode and psutil. All are in `src/stack/requirements.txt`.
  - Nothing is missing.
- **Runtime and hardware** are stated in README:49-58:
  - CPU: 30-55 h on a 12-vCPU / 96 GB machine, and about 120 GB of `cache/`.
  - GPU: 15-20 h on Kaggle 2x T4 (the table at README:62-72).
  - The second family's CPU time and disk are not totalled (`src/stack/README.md:93-96` gives only per-step figures).
  - My estimate for a full rebuild on one machine: 2-3 days of CPU plus about 17 h of sequential Kaggle kernels (within the 30 h weekly quota, each kernel under 12 h). This is doable but heavy, and nobody on a judging panel will do it.

## Reproducing, step by step: where an engineer gets stuck

1. **Setup (README:74-77).** Clean. It uses `python3.12 -m venv`, and all pins install on x86-64 Linux.
2. **Data (README:59-60).** `raw_dir` (`src/er.py:13-18`) finds the TSVs anywhere under `data/`. One snag: the quick check (`src/verify_final.py`) needs only test files, yet it fails unless `train_source1.tsv` is present, because `raw_dir` searches for it.
3. **CPU chain, part 1 (README:81-114).** About 45 commands typed by hand. There is no runner, even though the full list already exists as data in `attempts/029f6_final/train.py:16-73` (`PRE_STEPS`, which no shipped code reads). Some lines are shell commands that assume `cache_dir: cache` (README:100, 107, 109, 123). Every `er_predict.py` call also writes full submission TSVs to `submissions/archive/`. The stage-1 call (README:88) writes one over about 79M test pairs; this is not mentioned.
4. **GPU, part 1 (README:115-116).** It is documented which folders to upload, which script, which JSON config, and where to download. But:
   - The Kaggle kernels are set up by hand.
   - `ce_cfg.json` must be edited by hand for e5-base.
   - Training stops on a wall-clock budget (`src/ce/ce_train.py:77-78`: `total[0] = min(nb, int(rate * budget * 60))`), and the LR schedule depends on that total, so a rerun trains a different number of steps.
   - The step counts actually used are not recorded in the package.
5. **CPU part 2, GPU part 2, CPU part 3 (README:117-139).** Consistent. Line 130 copies the shipped second-family outputs into `cache/stack/`. That **overwrites** what `src/stack/README.md:72` writes to the same folder (`--out ../../cache/stack`) if someone did regenerate them.
6. **Final copy (README:141-143).** It needs the `NN` printed by the last command. This is fine, but a runner would remove it.
7. **Second family (`src/stack/README.md`).** A separate Python 3.11 environment. The README must be read with its "Known reproduction gaps" section (lines 98-114) in view:
   - the export command was not logged and has not been run end to end;
   - the command lines of steps 7-8 were rebuilt from memory;
   - the Kaggle kernels were set up in the UI;
   - the transliteration dictionaries were not kept;
   - the France pre-veto scores are not stored.

   All of this is candid, but it means the 455 MB of shipped `artifacts/stack/` outputs cannot be shown to come from this code.

## Prioritised fix list

**Guard for every fix.** After any edit, regenerate `SHA256SUMS`. If any top-level `src/*.py` or `attempts/029f6_final/train.py` changes, rerun `python src/er_predict.py 029f6_final --from-probs 030q_fix` (38 s) so that `run_manifest.json` hashes the shipped code (`src/er_predict.py:144` hashes exactly those files). Then rerun `src/verify_final.py` and the validator. Otherwise the manifest contradicts the package. Changes that touch only the README or `src/stack/` need only a new `SHA256SUMS`.

### HIGH

**H1. Add a runner script for the CPU chain.**
- **Where:** new `run_pipeline.sh` (or `src/run_steps.py`) at the package root. Build it from `attempts/029f6_final/train.py:16-73`, which already equals README:82-138.
- **What to change:**
  - Use `set -euo pipefail`.
  - Split it into phases that stop at each `# GPU` checkpoint (phase A up to `ce_export.py band`, phase B up to `ce_export.py extra`, phase C to the end), with a `--from N` resume option.
  - End it with the copy to `output/` (globbing the newest `submissions/archive/*_029f6_final.tsv` instead of `NN`) and a run of the validator.
  - Make README:79-140 point to it.
- **Why a judge cares:** the organisers ask that the folder "regenerate both output files ... using only the contents of this folder". Eighty-one hand-typed commands is the first thing a judge who tries to rerun will object to. `PRE_STEPS` and `DEPENDS` sitting unused in the attempt file suggest a runner existed and was not shipped.

**H2. Close or contain the second family's provenance gap.**
- **Where:** `src/stack/README.md:98-114` and `artifacts/stack/`.
- **What to change:**
  - If the team's stack caches still exist (`cache/ic_q/e1_*.npz`, `output_v08/final_p_*.npz`, `$CE_DIR/out/`), run step 9 (`export_scores.py`) now and compare the result with the shipped parquets, byte for byte or row for row.
  - Replace "It has not been run end to end" with the outcome.
  - Re-mine the transliteration dictionaries (step 1, minutes) and ship them in `artifacts/stack/` so later reruns are pinned.
  - If the caches are gone, say so in one sentence in the main README next to README:130 rather than only inside the stack README.
- **Why a judge cares:** the meta-model (`src/er_meta.py`, +0.0027 CV per `Documentation_template.md:122`) and the France blend (`src/er_france.py`) both read these files. "Not logged / not run end to end / not kept" is exactly the wording that makes a reviewer doubt that the outputs came from this code.

**H3. Ship the main family's cross-encoder scores and correct the "runs from this package alone" claim.**
- **Where:** README:9-10 and README:179-181; `src/ce/ce_train.py:65-78`; `src/stack/kaggle/ce.py:27`.
- **What to change:**
  - Add `artifacts/ce/` with the collected score files `cache/feat/{train_ce,test_ce,train_ce_x,test_ce_par,test_ce_x_par}.parquet` (my estimate: about 100-200 MB; `train_ce2` and `test_ce2_par` are derived from these by `ce_export.py ce2`).
  - Add a "CPU-only replay" path to the README: skip the three `# GPU` blocks and copy `artifacts/ce/*` into `cache/feat/`.
  - Reword README:9-10. The main pipeline currently still needs about 7-10 GPU-hours of retraining of its own (README table rows 1-8).
  - Record the trained step counts from the Kaggle logs, and add a `"steps"` option to `ce_cfg.json` (and a `CE_STEPS` variable in `kaggle/ce.py`) that overrides the wall-clock budget.
- **Why a judge cares:** without these files, nothing from `er_ce.py 028_ce` onward (the CE stack, meta-model, fixes, France, final) can be replayed deterministically. The current wording claims more than the package delivers.

### MEDIUM

**M1. Make `config.yaml` `cache_dir` actually configurable, or say that it isn't.**
- **Where:**
  - relative `cache/...` paths in the attempts: `attempts/027p_meta/train.py:5`, `attempts/029f5_final/train.py:5,9`, `attempts/029f6_final/train.py:3,6,11`;
  - README shell lines 100, 107, 109, 123, 130 and 131 (`cache/r19s`);
  - `src/single/build.py:26` takes its output folder from the CLI (relative to the working directory), while `src/single/fit.py:24` and `src/single/sidecar.py:18` use `cfg["cache_dir"]/r19s`.
- **What to change:** the cheapest fix touches only the README. Add one line: "keep `cache_dir: cache`; to use another disk, symlink `cache/`". The alternative is to resolve relative `STACK_DIR` / `THIRD` / `CAND_ALSO` against `cfg["cache_dir"]` in code.
- **Why a judge cares:** anyone moving the 120 GB cache to a larger disk by editing the config gets a silent split: some steps read one folder and some the other.

**M2. Stop the stack-artifact overwrite.**
- **Where:** README:130.
- **What to change:** use `cp -n`, or add "skip if you regenerated `cache/stack/` with `src/stack/README.md`".
- **Why a judge cares:** as written, a faithful regeneration is silently replaced by the shipped files, so it proves nothing.

**M3. Remove the traces of unshipped research code and the dead code.**
- **Where:**
  - `run_manifest.json:62` ("research modules of the source tree are not shipped");
  - `src/stack/src/stack_ce.py:56, 452, 460, 518, 546, 606, 609`, `src/stack/src/high_model.py:10`, `src/stack/src/marker_veto.py:11` and `src/stack/src/export_extra.py:5`, which name `loco_transfer.py`, `feat_cnt.py`, `cr_score.py`, `lexical.py` and `fr_addr_rule.py`, none of which is shipped;
  - `src/stack/src/high_model.py` as a whole: it is reached only through `--unseen_highmodel` (`stack_ce.py:503`), which the documented command does not pass;
  - dead code:
    - in `src/er.py`, everything except `raw_dir` is unused (`validator_path`, `load_split`, `load_pool`, `f05*`, `candidate_stats`, `ABBR`, `LEGAL`, `norm_text`, `core_name`, `add_norm`), and it is a second, stale normalization next to `src/er_norm.py`;
    - `src/common.py:48` (`read_json`);
    - `src/stack/src/decision.py` (`exclusive`, `threshold_select`), `model.py` (`train_lgb`), `metric.py` (`summarize`, `blocking_report`), `features.py` (`context_features`);
    - the leftover `k = new` in `src/er_newpairs.py:125`.
- **What to change:** delete or reword the comments. Drop the `src_note`. Delete the dead functions. For `src/er.py` and `src/common.py` edits, follow the guard above.
- **Why a judge cares:** comments saying "a research script not shipped" invite the question "what else is not shipped?". Dead code makes the reader work to find the real pipeline.

**M4. Disclose, or align, the fold schemes between the two families.**
- **Where:**
  - `src/er_model.py:112-129`: the main family uses StratifiedKFold with seed 42, and its CE halves are folds {0,1} vs {2,3,4};
  - `src/stack/README.md:11` and `:91`: the stack family uses `crc32(id) % 5` folds, with CE halves by fold parity;
  - `src/er_meta.py:118-122`: the meta-model cross-fits on the main family's folds over stack out-of-fold inputs computed on the stack's folds.
- **What to change:** add two sentences to the README Verification section. If the caches exist, rerun `er_meta.py` CV once with the stack's folds to show the 0.99177 does not move beyond noise.
- **Why a judge cares:** misaligned folds in stacking are a classic second-order leak. The headline CV (0.99184) sits on top of it.

**M5. Make `run_manifest.json` complete and unambiguous.**
- **Where:** `src/er_predict.py:144-148`.
- **What to change:**
  - Hash `src/**/*.py` by relative path, not top-level names only. `src/ce`, `src/single` and `src/stack` are missing today, and so is `src/prepare_raw.py`, which shows it was added after the run.
  - Rename the key `"train.py"` to `attempts/029f6_final/train.py`.
  - Omit the training fields under `--from-probs`. Today they show `n_entities: 0` and `sample_sha256` = SHA256 of empty input (`e3b0c442...`), which looks like a broken run.
  - Then follow the guard.
- **Why a judge cares:** the manifest is the provenance record, and gaps in it weaken the strongest part of the package.

**M6. Lock the environments.**
- **Where:** `requirements.txt`, `requirements-gpu.txt:1`, `src/stack/requirements.txt`.
- **What to change:**
  - Add `requirements.lock.txt` (`pip freeze`) for the CPU and stack environments, which pins joblib, threadpoolctl and the like.
  - For the GPU steps, record the Kaggle image tag or date (Kaggle's "pin to original environment"), the Python version, and pins for tokenizers, safetensors and huggingface_hub.
- **Why a judge cares:** direct pins alone drift. A judge rerunning in six months resolves different transitive versions.

**M7. Add a stage map and naming key.**
- **Where:** the README, next to README:155-166.
- **What to change:** add a table: attempt ID -> role -> reads -> writes -> approximate time.
- **Why a judge cares:**
  - IDs do not follow run order: `027p_meta` runs after `028k_ce`, and `030p_fix` before `029f5_final`.
  - IDs and names carry research lineage: `005n_indic_nocoh`, `f5`/`f6`, `r19`, "K9" (`src/er_keys.py`), and `er_feat2/4/5/6` with no `er_feat3`.
  - The side effects of the intermediate `er_predict.py` calls are not documented.
  - Readers lose time reconstructing all of this.

### LOW

- **L1.** `src/stack/src/normalize.py:76-79` falls back silently to empty transliteration dictionaries. Fail loudly instead, as `src/er_norm.py:59-60` does.
- **L2.** The France word lists are hard-coded in `attempts/029f5_final/train.py:11-12` and `029f6_final/train.py:12-13`, while `src/er_frwords.py:50-51` only prints them. Add an assert, or have `er_france.py` compute them.
- **L3.** `src/er_gen.py:229-236` returns about 100 features by position, and `:266` names them from `NAMES` with no length check. Add `assert arr.shape[1] == len(NAMES)`.
- **L4.** Small text fixes:
  - README:152 says "the list below"; it is above.
  - The `src/er_lexical.py` docstring usage omits `--cv`.
  - `Documentation_template.md:124` says "about 20 steps" against 81 invocations plus about 25 stack commands.
  - `src/er_eval.py:23` `TEST_WEIGHTS` (the test country mix) has no comment.
- **L5.** `src/er_model.py:114` writes `folds.parquet` into `data_dir`. Move it to `cache/`, because datasets are often mounted read-only.
- **L6.** `src/ce/ce_train.py:25, 31-32, 93, 96, 100` hard-code `/kaggle/...`, while `src/ce/ce_score.py:21, 71` allow `KIN`/`KOUT` overrides. Make them consistent so the scripts can run on a local GPU.
- **L7.** State the second family's total CPU time and disk. Document `RK_THREADS` and the hard-coded `num_threads` 8/10/12 in the attempts. Mention the memguard default of 3 GB (`src/stack/src/memguard.py:16`), which aborts runs unless `MEM_LIMIT_GB` is set; the stack README does set it.
- **L8.** Style: 235 lines exceed 150 characters (the maximum is 408, at `src/er_gen.py:233`), and there are no unit tests. **Do not reformat now**: it would invalidate the manifest hashes for no score gain. Note it as future work.

## Notes outside this lens (for the other evaluators)

- `Documentation_template.md:35` and `:155` say one France change was "kept because the leaderboard rose". This is disclosed, but a methodology or fair-play reviewer will notice it.
- The France word lists and the marker table are derived label-free from test candidates (`src/er_frwords.py`, `Documentation_template.md:33`). This is disclosed, and it is transductive use of test features.
- Unidecode (GPL-2.0) is a library, not a model, and it is disclosed (README:46-47). The model licences (e5-small and e5-base, MIT, 118M and 278M parameters, pinned revisions) comply.
