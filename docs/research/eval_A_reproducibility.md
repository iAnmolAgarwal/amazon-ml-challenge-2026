# Evaluator A: reproducibility and engineering quality

Package: `submissions/final_package/Apprentice_submission.zip` (team Apprentice), unzipped to a scratch folder. Only the zip contents were judged.
Bar: top 50 of about 1000 teams (top 5%). Scores compare the package with the best ~5% of packages, not with an average one.
Paths below are relative to the zip root. `BER/` = `code/business_entity_resolution/`.

## Scores

| Criterion | Score /10 | One-line reason |
|---|---|---|
| Reproducibility | **3** | The documented chain breaks at the France step (missing attempt file). The second model family has steps that were reconstructed and never run. It needs 3 environments and ~12 manual Kaggle runs, and GPU results are only approximate. |
| README clarity | **5** | The command list is complete and correctly ordered, but it is a 60-line wall. GPU steps are squeezed into comment lines, and there is no total runtime, disk figure, working directory or checkpoint values. |
| Dependency pinning | **6** | Every pin is exact and Python versions are stated. But there are 3 separate environments, no lock file, a CUDA torch in the CPU environment, and xgboost is imported but not pinned. The cross-encoder revision is claimed but not enforced in the main code. |
| Code quality | **4** | Almost every module has a good docstring and a CLI. Against that: very dense lines, dead code, positional feature coupling, hidden env-var settings, worker code stored inside strings, and two different ways of resolving paths. |
| Structure | **6** | The organiser layout is followed exactly. But a second pipeline is nested inside `src/stack/`, with its own README, requirements and a Makefile that contradicts it. |

**Verdict: WOULD NOT clear the top-50 bar as shipped.** It becomes borderline once HIGH fixes 1 to 4 below are done. Most of the gap is fixable in a few hours; the unrecorded steps of the second family are not.

In its favour: both outputs pass the organisers' validator with `--check-ids`. They are consistent with each other: every match is a candidate, rows are in `test_source1` order, and there are no duplicate ids. The reported CV (0.99184; US 0.99164; India 0.99213) matches `BER/attempts/029f6_final/results.json` exactly. The "Not reproducible from this code alone" section (`BER/src/stack/README.md:95-111`) is unusually honest.

## Evidence

### 1. Can a competent engineer regenerate the outputs?
**Not as shipped.** I traced every command in `BER/README.md:51-108` against the scripts, their argparse options and the `attempts/` files:
- All scripts and options named in the README exist.
- The data dependencies are in the right order. Every cache file a step reads is written by an earlier step. I checked the full `er_newpairs` / `er_extend` / `ce_export` / `er_meta` chain.
- **Hard break:** `BER/README.md:105` runs `python src/er_france.py 029f5_final`. `er_france.py:61` calls `load_attempt()`, which opens `attempts/029f5_final/train.py` (`common.py:64-70`), and that file is not in the zip. The next step, `attempts/030q_fix/train.py:1` (`PARENT = "029f5_final"`), depends on it. The France keys (`W`, `RENAME`, `C`, `THIRD`, `VETO_WORDS`, `BROAD_WORDS`) sit in `attempts/029f6_final/train.py`. A reviewer can only guess that 029f5 had the same values. This also shows the README was never executed from a clean unzip.
- **Second family (`src/stack/`):**
  - Export step 9 "has not been run end to end" (`BER/src/stack/README.md:96-99`).
  - The command lines for steps 7-8 were rebuilt from logs (`:100-102`).
  - The transliteration dictionaries are re-mined and were never compared with the originals (`:107-109`).
  - The Kaggle kernel wrappers are not in the code (`:103-105`).
  - A harsh reviewer can therefore say that part of the final input was not produced by this code as shipped.
- **Hand-typed France constants:** `attempts/029f6_final/train.py:11-12` hard-codes `VETO_WORDS = ["participations", "holding", "distribution", "international"]` and `BROAD_WORDS = ["developpement", "france", "groupe"]`. No shipped command derives them. `src/stack/src/marker_veto.py` has a similar label-free rule but does not output these lists. France has no labels, so a reviewer will ask whether these words came from reading the test set.
- **GPU steps:** training stops on a wall-clock budget in fp16 (`BER/README.md:38-40`), so the scores cannot be bit-identical. No tolerance and no intermediate checkpoints are given, so a rerun cannot be checked against anything except the final TSVs.
- **Model weights:** the main cross-encoder scripts do not pin a model revision. `src/ce/ce_train.py:45,61,101` call `from_pretrained(MN)` with no `revision=`, although `README.md:36-37` claims pinned revisions. Only `src/stack/kaggle/ce.py:17` and `ce_france.py:24` pin one.
- **Config trap:** `README.md:86` gives `ce_cfg.json {"model": ..., "lr": 3e-5, "budget_min": 140}`. `ce_train.py:103` needs `cfg["maxn"]`, which is written only by `ce_export.py:65`. Anyone who uses the README JSON as written gets a KeyError.
- **Brittle assert:** `src/ce/ce_score.py:55` asserts the band sizes are exactly 3,008,987 (train) and 2,815,521 (test). Upstream models without `deterministic=True` (`attempts/016_stage2b/train.py:9`, and the default params in `src/er_model.py:14`) can move a few pairs across the 0.003/0.997 edge. The run would then fail on Kaggle after the upload.

**Runtime and hardware (my estimate; the package never gives a total):**
- Main pipeline: ~20-35 CPU-hours on the stated 12 vCPU / 96 GB machine. `README.md:24-25` says blocking, stage-1/2 CV and address retrieval each take 1-2.5 h, and there are about 25 further CPU steps.
- Second family: ~10-20 CPU-hours (`src/stack/README.md:92`: blocking 25-60 min per country block, about 35 min per stage-2 model, 8 experiment configs).
- GPU: ~16-18 hours on 2x T4 across about 12 manual Kaggle sessions. Main pipeline: e5-small 170+22 min, e5-base ~140 min + scoring, 4 extra scoring passes. Second family: e5s 47 min, e5i 1-2 h, e5b 4.7 h, e5bc 2x75 min plus scoring.
- That is about 3-5 working days, including manual uploads and downloads.
- Disk use and minimum RAM are stated nowhere. The main steps assume 96 GB; the stack watchdog is set to 48 GB.
- The working directory is never stated. It must be `BER/` for `sys.path.insert(0, "src")` (`src/single/fit.py:15`, `build.py:15`) and for the relative `cache/` paths.

### 2. requirements.txt
- **Pinned:** yes. Every line uses `==`, and Python 3.12 (main) and 3.11 (stack) are stated. `src/stack/kaggle/requirements.txt` pins the GPU step.
- **Import coverage:**
  - `xgboost` is imported in `src/single/fit.py:74,113` but not pinned (it is only on the unused `xgb` branch).
  - `torch`, `transformers` and `pandas` in `requirements.txt:13-15` are only needed on Kaggle. A CPU install still pulls multi-GB CUDA wheels.
  - `psutil` is pinned in the main file but only used by the stack.
  - `jellyfish` and `tabulate` (`src/stack/requirements.txt:1,12`) are never imported.
  - Transitive packages that affect results (`tokenizers`, `huggingface_hub`, `safetensors`) are unpinned, and there is no lock file.
- **Environments:** the three environments pin different numpy (2.5.3 / 2.4.6 / 2.4.6) and scipy (1.18.1 / 1.17.1) versions.
- **Licences:** Unidecode (GPL-2.0) is disclosed (`README.md:41`). `src/stack/kaggle/ce.py:6` mentions `microsoft/mdeberta-v3-base`, which is never listed as used or unused.

### 3. Code quality and structure
- **Positives:**
  - Module docstrings give usage, inputs and outputs.
  - The CLIs are consistent, and the `s1_idx` / `pool_idx` key conventions are the same everywhere.
  - Fold and sample seeds are fixed.
  - There are alignment asserts (`er_predict.py:150`, `ce_export.py:86,90,127`), and a run manifest with source hashes (`er_predict.py:159-164`), although the manifest is not shipped.
- **Density:** 558 source lines are longer than 120 characters and 65 are longer than 200 (the longest is 408, at `src/er_gen.py:233`). `er_gen.py:229-235` returns about 100 unnamed values that line up by position with `NAMES` at `er_gen.py:40`, so a reordering would silently mislabel features.
- **Dead code:**
  - `src/er_feat3.py`, which `README.md:31` admits is unused.
  - `common.py`: 14 template helpers that are never called (`smape`, `mae`, `rmse`, `f1_*`, `map_at_k`, `recall_at_k`, `get_metric`, `paired_gain`, `get_device`, `load_train/test/folds`, `read_json`).
  - `er.py`: everything except `raw_dir` (lines 37-105).
  - The `xgb` / `logit` branches of `single/fit.py`.
  - `attempts/019_band/`, which is not in the command list but is listed in `029f6_final/train.py:13`.
  - The unused `config.yaml` keys. `device: mps` (`config.yaml:3`) contradicts the "x86-64 Linux" claim.
- **Paths:**
  - `common.py:16-31` finds `data/` and `cache/` with `git rev-parse --git-common-dir`.
  - `attempts/027p_meta/train.py:5`, `029f6_final/train.py:5,9`, `src/single/sidecar.py:14,20,24`, `fit.py:22`, and the README's `cp` / `ln -s` lines use `cache/` relative to the current directory.
  - Inside any git checkout the two roots differ. `README.md:35` warns about this but does not fix it.
- **Hidden settings:** environment variables change behaviour: `R19D`, `R19B`, `R19T` (`single/fit.py:22,100,120`), `MONO`, `UNSEEN_ADDR_SHIFT`, `FR_NS`, `EXTRA_NS`, `BLOCK_ROUTE`, `HM_BASE`. Worker code is kept in strings (`ce_train.py:16-92`, `ce_score.py:16-72`), so linters never see it. It does compile.
- **Silent defaults:**
  - `src/stack/src/memguard.py:17` defaults to a 3 GB RSS limit, so jobs abort unless `MEM_LIMIT_GB` is exported.
  - Stack normalisation silently uses empty dictionaries if the translit files are missing (`src/stack/README.md:32`).
- **Warnings:** `src/stack/src/decoy_feats.py:53` has a non-raw regex string, which gives a SyntaxWarning on Python 3.14.
- **Unexplained magic numbers:** `er_eval.py:23` `TEST_WEIGHTS = {"us": 0.38, "india": 0.47}`, `ce_export.py:54` `n_s1=470000`, `ce_export.py:65` `maxn 6200000`, and the thresholds 0.003/0.997, 0.001 and 0.02/10 repeated across attempts.
- **Stale references:**
  - `attempts/027p_meta/train.py:15,19` (`029f2_final`, `027l_meta`).
  - `024f_keys/train.py:23` (PREDICT points to `024_keys`).
  - `er_predict.py:13` (`src/er_package.py`) and `src/stack/src/stack_ce.py:20` (`src/unseen_rule.py`); neither file is shipped.
  - The `single/fit.py:1-2` docstring says "full-train test model", but lines 110-111 average the 5 fold models.
- **Contradictory Makefile:** `src/stack/Makefile:1` says "Reproduce the final submission end-to-end". Its line 11 says the dictionaries are "shipped in cache/", which they are not. Its `stack` target (`:54-55`) lacks `--ops --pairf`.
- **Reruns overwrite evidence:** `er_eval.py:111-113` overwrites the shipped `attempts/<id>/results.json` on every run.

### 4. What would make a reviewer doubt the outputs came from this code
1. The final chain cannot run (`029f5_final` is missing), so it was never executed from this folder.
2. Parts of the second family's pipeline were reconstructed after the fact and never re-run.
3. The France word lists are typed in by hand, with no derivation.
4. Nothing links the TSVs to the code: no run manifest, no output hashes, and `git_commit: "nogit"` in every results.json. The shipped SHA-256 values are `matching_results.tsv` 2b885f1e…71f9e and `candidate_pairs.tsv` 5f06475a…eadb1, but the package never states them.
5. The model revision is claimed but not enforced in the main cross-encoder code.

## Prioritised fixes (each fits in a few hours)

| # | Sev | Where | Change | Why a judge cares |
|---|---|---|---|---|
| 1 | HIGH | `BER/attempts/029f5_final/train.py` (missing); `README.md:105`; `attempts/030q_fix/train.py:1` | Ship the exact France settings used for the submitted file, taken from the original run. If they equal `029f6_final/train.py`, reduce 029f6 to the keys `er_predict` needs (`FLOOR`, `CAND_BAND`) and say so. | This is the only hard break in the executable chain. A reviewer who hits it assumes the README was never run. |
| 2 | HIGH | `BER/attempts/029f6_final/train.py:11-12` | Add the command that produces `VETO_WORDS` / `BROAD_WORDS` (for example a `marker_veto.py` mode that prints the France markers at lift 1.5 and n_min), plus one README line on how the BROAD words were chosen. | Hand-typed word lists for the unlabelled test country read as test-set peeking. It is the easiest item for a harsh reviewer to cut on. |
| 3 | HIGH | `BER/README.md` (top); package root | Ship `submissions/archive/NN_029f6_final.json` (already written by `er_predict.py:159-164`) and a `SHA256SUMS` of both TSVs. Add a "Reproduction status and checkpoints" section: what was rerun from a clean unzip, what was reconstructed, and the expected values per stage (CV from the results.json files and Documentation table B; band sizes 3,008,987 / 2,815,521; 5,847,885 matched and 11,778,648 candidate pairs), with a tolerance for the GPU steps. | Links the outputs to the code and lets a reviewer check an approximate rerun. |
| 4 | HIGH | `BER/src/stack/src/export_scores.py`; `src/stack/README.md:96-102` | If the original caches still exist, run `export_scores.py`, compare against the delivered `cache/stack` files, and replace "has not been run end to end" with the result. Then do one clean-unzip dry run of the whole main command list, at least as far as loading each attempt and checking that every input exists. | Turns "reconstructed" into "verified". Without it, part of the final input has no provenance. |
| 5 | MEDIUM | `BER/src/ce/ce_train.py:45,61,101` | Pass `revision=` (e5-small `614241f…`, e5-base `d128750…`) from `ce_cfg.json`, as `src/stack/kaggle/ce.py:65-67` already does. | README.md:36-37 claims a pin that the code does not enforce. |
| 6 | MEDIUM | `BER/README.md:85-86,97` | Replace the three comment lines with a numbered table of the ~8 main-pipeline Kaggle runs: script, input datasets (including attaching the `mA`/`mB` training output to the scoring kernel), the full `ce_cfg.json` / `score_cfg.json` (note `maxn` is required, `ce_train.py:103`), time, and output folder. | This is where an outside engineer is most likely to fail, and the e5-base JSON as written raises a KeyError. |
| 7 | MEDIUM | `BER/README.md:22-27,43-47` | State the working directory (`BER/`), total CPU-hours, GPU-hours and Kaggle sessions, peak RAM, and disk (measure `du -sh cache`). | Required by the brief; a reviewer plans the rerun from these numbers. |
| 8 | MEDIUM | `BER/src/common.py:16-31`; `attempts/027p_meta/train.py:5`; `029f6_final/train.py:5,9`; `src/single/sidecar.py:14-24`; `fit.py:15,22`; `build.py:15` | Resolve `data`/`cache` from `ROOT` (drop `shared_root`), and resolve `STACK_DIR` / `THIRD` / `cache/r19s` against `cfg["cache_dir"]`. | Unzipping inside any git repo silently splits the cache into two places. |
| 9 | MEDIUM | `BER/src/ce/ce_score.py:55` | Check against the uploaded band's own row count, or warn instead of asserting the historical constants. | Otherwise any small upstream change fails after a slow Kaggle upload. |
| 10 | MEDIUM | `BER/src/stack/Makefile:1,11,54-55` | Delete the Makefile, or change its header to "historical, superseded by README command list". | It contradicts the README inside the same package. |
| 11 | MEDIUM | `BER/src/stack/src/memguard.py:17`; stack `normalize.py` translit loader | Default `MEM_LIMIT_GB` to a share of physical RAM, and raise an error when the translit files are missing. | Removes silent failure modes on the reviewer's machine. |
| 12 | LOW | `BER/requirements.txt:12-15`; `src/stack/requirements.txt:1,12`; `src/stack/kaggle/requirements.txt` | Move torch/transformers/pandas to a GPU file; drop the unused `psutil`, `jellyfish` and `tabulate`; pin `tokenizers`, `huggingface_hub` and `safetensors`; add `pip freeze` lock files for both CPU environments. | Cleaner install and a stronger pinning score. |
| 13 | LOW | `BER/src/er_feat3.py`; `common.py:34-172`; `er.py:37-105`; `single/fit.py:73-97,112-117`; `attempts/019_band/`; `config.yaml:3-18` | Remove the dead code, research branches and unused config keys (including `device: mps`). | Less for a reviewer to read; removes the Mac-versus-Linux contradiction. |
| 14 | LOW | `attempts/027p_meta/train.py:15,19`; `024f_keys/train.py:23`; `er_predict.py:13`; `src/stack/src/stack_ce.py:20`; `single/fit.py:1-2`; `src/stack/kaggle/ce.py:6`; `Documentation_template.md:10,44` | Fix the stale ids and paths and the wrong docstring. List e5-base next to e5-small for the main cross-encoder, and state that mdeberta is unused. | Makes the disclosures consistent for the licence audit. |
| 15 | LOW | `BER/src/stack/src/decoy_feats.py:53`; whole `src/` | Add the `r` prefix to the regex string. Run `ruff format --line-length 120`, which does not change behaviour. Name the `er_gen` output positions (a dict or a list of pairs). Give each magic number (`er_eval.py:23`, `ce_export.py:54,65`) one line of explanation. | Readability is the main complaint about the code; the positional coupling is a latent bug. |
| 16 | LOW | `BER/src/er_eval.py:111-113` | Write rerun results to `cache/` or `results_rerun.json`. | A rerun should not overwrite the shipped evidence. |
