# Evaluator A: reproducibility and engineering quality

Package: `submissions/final_package/Apprentice_submission.zip` (569 MB zipped, 122 entries, built 28 Sep 17:07).
I judged only what is inside the zip. I extracted it to a scratch folder, leaving out the 13 `artifacts/stack/*.parquet`. I checked those files' hashes by streaming them straight from the zip.

## Scores

| Criterion | Score | One-line reason |
|---|---|---|
| Reproducibility | **6 / 10** | Every script and attempt the command list names exists, and the steps are in dependency order. But the final model needs 455 MB of shipped second-family predictions (including test probabilities) that have never been regenerated end to end. The GPU steps are time-boxed and run by hand, and the full run needs roughly 60-100 machine-hours. |
| README clarity | **7 / 10** | Thorough, honest (it has a "known gaps" section), gives CV figures per stage and hashes. It is also dense: about 55 hand-run commands, a second copy of the list that disagrees with it, and some ambiguous GPU hand-off steps. |
| Dependency pinning | **7 / 10** | Direct dependencies use exact `==` pins, the Python version is stated for each environment, all 20 pins exist on PyPI and every import is covered. There is no lock file for transitive dependencies, there are four requirement files for three environments with conflicting numpy/scipy pins, the "default Kaggle image" is ambiguous, and a few pins are unused. |
| Code quality | **6 / 10** | Good module docstrings, consistent command-line interfaces and many integrity asserts. On the other side: cryptic research identifiers, a large dead research surface (30 flags, 9 environment switches, references to 6 scripts that are not shipped), code embedded in strings, paths relative to the working directory that bypass `config.yaml`, silent fallbacks and no tests. |
| Structure | **7 / 10** | Matches the organisers' layout exactly. But it nests a second project (`src/stack/src/`) with its own environment, keeps settings-as-code in `attempts/*/train.py` (files named "train" that do not train), and has two sources of truth for the command list. |

**Overall verdict: borderline.** Through this lens the package will not disqualify the team: outputs are hash-verified, the organisers' validator passes, and the documentation is candid. It also will not earn credit, because a judge cannot practically reproduce the outputs. Two things also invite provenance questions: the run manifest describes a different `src/` from the one shipped, and the decisive late gains rest on shipped test predictions. The three HIGH fixes below are cheap and would move this to "advance", because they remove the doubts a strict auditor raises first.

---

## What I verified (evidence)

- **Outputs match the declared hashes.** `output/matching_results.tsv` has SHA256 `2b885f1e…`, and `output/candidate_pairs.tsv` has `a8cda21e…`. Both match `SHA256SUMS` and `run_manifest.json:84-85`. All 13 `artifacts/stack/*.parquet` match `SHA256SUMS` (checked by streaming from the zip).
- **The organisers' validator passes.** The run included `--check-ids`: 1,732,544 rows, 9,969,589 valid S2/S3 IDs, "PASS — no blocking issues". Every matched pair is in the candidate file (no subset warning). The copy at `src/stack/utils/validate_submission.py` is byte-identical to the organisers' file.
- **The numbers agree across files.** The candidate total of 11,837,012 appears in the manifest (`run_manifest.json:88`), the README and Documentation §3. The 58,364 scored-but-not-decided pairs equal 11,837,012 − 11,778,648 (`n_decided_pairs`). CV 0.99184 appears in `attempts/029f6_final/results.json`, the README verification table and Documentation §5.
- **All 85 `.py` files compile.** `py_compile` raises no errors. There is one `SyntaxWarning`: an invalid escape `"\s"` at `src/stack/src/decoy_feats.py:53`.
- **Imports are clean.** An AST scan found that every non-stdlib import resolves to a shipped module or a pinned package, and no shipped module imports a missing one.
- **Every attempt folder is used.** All 16 folders in `attempts/` are referenced by the README command list, as the README claims (`README.md:174`).
- **Nondeterminism.** I found no Python `hash()`, unseeded RNG or set-order dependence. LightGBM uses fixed seeds. `deterministic=True` is set only in some attempts (017, 018, 027p, 030p/q), not in stages 1-2, 016, 023 or 028. The README honestly reports CPU-type float drift (stage 1 CV 0.97005 vs 0.97008).

---

## 1. Can a competent engineer reproduce both outputs?

**Walk-through of `README.md:78-141` (main environment):**

- **Prep and dictionary.** `prepare_raw.py` → `er_dict.py` → `er_norm.py` → `er_block.py`, then `er_feat` / `er_feat2`. Each step reads only what the previous one wrote (`cache/raw` → `cache/dict` → `cache/norm` → `cache/cand` → `cache/feat`). Data may sit anywhere under `data/`; `er.raw_dir` finds it with an rglob (`src/er.py:13-18`).
- **Stages 1-2 and the band models.** The stage 1 and 2 steps (`er_eval` / `er_predict` on 005n, 005f and 006; `er_feat4 --src …`), `er_lift`, `er_stage3 016_stage2b` and the retrieval route (017) are all in the right order. The `cp …016_stage2b … 017_route` step (`README.md:99`) is an explained shim.
- **Second-family inputs to the meta-model.** The train-side extra candidates (s10, then all), the `er_extend union` steps, CE export/collect/parity/collectx/ce2 and `single/build → fit → sidecar` are ordered correctly for data dependencies. I traced every input file of `er_meta.py`, `er_fix.py`, `er_france.py` and `er_predict.py` back to an earlier step.
- **Symlinks.** `ln -s 017_route cache/models/024_keys` (`README.md:106`, `:122`) fails on a rerun ("File exists") and stops any `&&` chain.
- **Final copy.** `README.md:140` runs `mkdir -p output && cp submissions/archive/NN_…` from `code/business_entity_resolution/`. That creates `code/business_entity_resolution/output/`, not the package-root `output/`. `NN` is really `03` on a clean run, because `er_predict` also writes archives for 005f_floor and 006_stage2 (`src/er_predict.py:127-133`).

**GPU steps (main family, `README.md:62-71`, `:114-115`, `:126`):**

- The steps are documented as manual Kaggle-UI operations. No kernel metadata is shipped (`src/ce/ce_export.py:21-22` only sketches `kaggle kernels push`).
- **`src/ce/ce_train.py` only runs on Kaggle.** It hard-codes `/kaggle/input` and `/kaggle/working` at lines 25, 31-32, 93, 96, 100, 107 and 110. `ce_score.py` does allow `KIN`/`KOUT` overrides.
- **The model config file is ambiguous.** `ce_train.py:96` takes the *first* `ce_cfg.json` that `os.walk` finds. The pairs dataset already contains the e5-small `ce_cfg.json` (written by `ce_export.py:65`), so "the same two scripts with ce_cfg.json {e5-base…}" is unclear: attaching a second config can silently retrain e5-small.
- **Training is time-boxed, not step-fixed.** The step count comes from the throughput measured at step 300 (`ce_train.py:77-78`). On one GPU the budget is halved (`ce_train.py:110`). The trained step counts, `log_*.txt` and `val_*.json` are not shipped, so a rerun cannot replay the original length.
- **Subprocess failures are not caught.** Return codes are printed but not checked (`ce_train.py:108-111`, `ce_score.py:83-86`), so a kernel can "succeed" with one half missing.
- **Hard-coded band sizes.** `ce_score.py:55` asserts band sizes of 3,008,987 and 2,815,521. That is a good guard, but it fails on any upstream float drift.

**Second environment (`src/stack/`, Python 3.11):**

- `src/stack/README.md:26-78` gives a complete command list.
- **"Known reproduction gaps" (`src/stack/README.md:99-115`) admits the following:**
  - The export step "has not been run end to end".
  - The command lines for steps 7-8 were "rebuilt from the recorded configuration".
  - The transliteration dictionaries of the actual run "were not kept".
  - The France pre-veto scores are not stored.
- **Why this matters.** The shipped `artifacts/stack/p_test_{France,India,US}.parquet` are test-set probabilities that feed the meta-model (the +0.0027 CV step), the France blend and the candidate file (`attempts/029f6_final/train.py:3`). The organisers ask that the pipeline "regenerate both output files … using only the contents of this folder". Here that holds only if the precomputed second-family test predictions count as "contents". A strict judge will note that the code path which would regenerate them has never been exercised.

**Runtime and hardware:**

| Part | Stated in the package | My estimate |
|---|---|---|
| Main CPU | 30-55 h on one 12 vCPU / 96 GB machine, 120 GB `cache/` (`README.md:48-53`) | — |
| GPU (both families) | 15-20 T4×2 hours; four steps "not recorded" | — |
| Second-family CPU | Not totalled. Per-step figures: blocking 25-60 min per country block, about 35 min per stage-2 model; 8 stage-2 configs | 15-25 h |
| Second-family disk | Not stated | — |
| **Total** | — | **~60-100 machine-hours, 96 GB RAM, ≥150 GB disk, Kaggle T4×2 and a person running about 12 kernels by hand** |

- **No short verification path is shipped.** The deterministic tail (`er_predict --from-probs`, 23 s according to `run_manifest.json:89`) needs `cache/oof/test_030q_fix.parquet`, which is not shipped. So a judge cannot confirm even the last step without the full chain.

## 2. requirements.txt

- **Pinned: yes.** `requirements.txt` uses exact pins and states Python 3.12.14 (line 1). `src/stack/requirements.txt` is used with Python 3.11 (`README.md:56`). All 20 distinct pins resolve on PyPI, and their `requires_python` ranges are consistent with the stated Python versions.
- **Complete for imports: yes.** The main pipeline needs polars, numpy, scipy, scikit-learn, lightgbm, rapidfuzz, sparse-dot-topn and pyyaml. The GPU steps need torch, transformers, pandas and scikit-learn. The stack needs pandas, pyarrow, unidecode, psutil and pyyaml.
- **Gaps:**
  - **No lock file.** Transitive dependencies are not locked (threadpoolctl, joblib, tokenizers, safetensors, huggingface-hub, OpenMP runtime). LightGBM/OpenMP and tokenizer versions can change results.
  - **Unused pins:** `psutil` in the main file (only `src/stack/src/memguard.py` imports it), `jellyfish` and `tabulate` in `src/stack/requirements.txt` (never imported), and `xgboost` (ablation only; labelled at `requirements.txt:13`).
  - **The GPU environment is ambiguous.** `README.md:54` says "the default image (`requirements-gpu.txt`)". It does not say whether torch 2.14.0 and transformers 5.17.0 were pip-installed or were the image's versions, nor which image date or CUDA version was used.
  - **Two GPU pin files disagree.** `requirements-gpu.txt` and `src/stack/kaggle/requirements.txt` cover the same Kaggle runs; only the second pins numpy (2.4.6). The main CPU environment pins numpy 2.5.3 / scipy 1.18.1, while the stack pins numpy 2.4.6 / scipy 1.17.1.

## 3. Code quality and structure

**Strengths:**
- Every module opens with a usage docstring that names its inputs and outputs.
- Attempts are pure settings files loaded by `common.load_attempt`.
- Integrity asserts guard key steps: uniqueness and null checks in `ce_export.py:86-90` and `:127`, overlap checks in `er_newpairs.py:128`, and "a match is not a scored candidate" in `er_predict.py:125`.
- The candidate rule is implemented, not just described (`er_predict.py:106-121`).
- The code is memory-aware (`er_model.to_matrix`).

**Weaknesses:**
- **Research naming leaks into the product.**
  - Identifiers such as `r19`, `r19a2`, `p006` (`src/single/*`), `028e_ce`/`028f_ce` (names that "are not attempt folders", `README.md:176`) and `029f5`/`029f6`/`030p`/`030q`.
  - Feature modules are numbered `er_feat`, `er_feat2`, `er_feat4`, `er_feat5`, `er_feat6`; `er_feat3` appears only in the manifest.
  - `attempts/*/train.py` holds settings but trains nothing.
- **Large dead research surface.**
  - `src/stack/src/stack_ce.py` is 617 lines with 30 command-line flags. Its behaviour also changes through the environment switches `MONO`, `UTILITY` and `UNSEEN_ADDR_SHIFT`, and `high_model.py` / `export_extra.py` / `block_feats.py` / `feat_spec.py` read `BLEND_NS`, `BLEND_W`, `HM_BASE`, `FR_NS`, `EXTRA_NS`, `BLOCK_ROUTE` and `SPEC_MAX_GROUPS`.
  - The code refers to six scripts that are not shipped: `unseen_rule.py` (`stack_ce.py:20`), `feat_cnt.py` (`:56`), `fr_addr_rule.py` (`:518`), `cr_score.py` (`:546`), `lexical.py` (`:606`) and `loco_transfer.py` (`high_model.py:10`, `export_extra.py:5`, `stack_ce.py:452`, `:460`).
  - `stack_ce.py:64` labels `--decoy2` "pending validation", yet the final stack build uses `--decoy2` (`src/stack/README.md:66`, `:71`).
- **Code in strings.** `ce_train.py:17-94` and `ce_score.py:16-72` embed the worker as a string, write it to disk and exec it, so it cannot be linted, imported or tested.
- **Paths that bypass `config.yaml`.**
  - `attempts/027p_meta/train.py:5`, `029f5_final/train.py:5,9` and `029f6_final/train.py:3,6,9` use `"cache/..."` relative to the working directory.
  - `src/single/sidecar.py:14,20,24` hard-codes `cache/r19s`, `cache/oof` and `cache/feat` and never calls `load_config`.
  - `src/single/fit.py:15,22` and `build.py:15` use `sys.path.insert(0, "src")`; `fit.py:22,100,120` call `__import__("os")` and use the `R19D`/`R19B`/`R19T` environment overrides.
  - `common.shared_root()` (`src/common.py:14-21`) is git-worktree logic left over from the team's harness.
- **Silent fallback.** `er_norm.indic_maps` (`src/er_norm.py:54-60`) quietly uses empty maps if `er_dict.py` was skipped, which degrades normalization without an error. The stack has the same behaviour (acknowledged at `src/stack/README.md:36`).
- **Magic numbers without inline rationale.**
  - Band edges and cutoffs: 0.001, 0.003/0.997 and 0.01/0.99, the cascade (0.02, 10), and France `W = 0.5`, `C = 0.1`.
  - `er_frwords.py`: the shift share cutoffs 0.97 / 0.60, `--min-shift 300` and the `SHIFTS` list (line 20).
  - `ce_export.py:54`: `n_s1 = 470000`, `k_top = 8`, `k_rand = 2`.
  - `er_eval.py:23`: `TEST_WEIGHTS = {"us": 0.38, "india": 0.47}`.
  - Many are justified in Documentation §4, but the code does not point there.
- **Derived word lists are pasted by hand.** `er_frwords.py:50-51` prints the France word lists, which are then copied into two attempt files (`029f5_final/train.py:11-12`, `029f6_final/train.py:12-13`). The label-free derivation is therefore not on the execution path.
- **Hard-coded, inconsistent thread counts.** 8, 10 and 12 threads appear across attempts; `er_model.py:14,53,62` fixes 8; `single/fit.py:62,68` uses 12. That is good for determinism on one machine type, but it is undocumented.
- **No tests.** There is not one test for `decide()`, `f05` or `write_lists`.
- **Structure.** `src/stack/` is a full second project (own `src/`, README, requirements, a copy of the validator) nested inside `src/`. The `attempts/` settings are Python code outside `src/`, whereas the organisers say "src/ (all source code)". There are two sources of truth for the command list (`README.md:78-138` and `attempts/029f6_final/train.py:16-73`), and the attempt copies have already drifted:
  - `attempts/024_keys/train.py:20` says `er_newpairs.py fit 024_keys`, where the README symlinks `017_route`.
  - `attempts/018_lexical/train.py:28` and `023_comb/train.py:18` point to `submissions/NN_…`, where the code writes `submissions/archive/`.

## 4. Things that make a reviewer doubt the outputs came from this code

1. **The run manifest describes a different `src/`.**
   - `run_manifest.json:29-77` hashes 46 files. 17 of them are not in the package: `embed.py`, `er_blend.py`, `er_diag.py`, `er_feat3.py`, `er_france2.py`, `er_keysfr.py`, `er_package.py`, `evaluate.py`, `harness_util.py`, `leaderboard.py`, `new_attempt.py`, `predict.py`, `prepare.py`, `split.py`, `sub_status.py`, `submit_log.py` and `train.py`.
   - The 29 shipped files all match their hashes, and no shipped module imports a missing one. Still, Documentation Appendix B.1 credits a "France extra keys" step, and the manifest shows an `er_keysfr.py` that is not shipped. A reviewer cannot tell from the package whether the final France candidates depended on it.
   - `git_commit` is `"nogit"` (`run_manifest.json:28`), although `README.md:12` says the manifest records the git commit.
   - `train_frac: 0.2`, `n_entities: 0`, `features: []` and the default `params` / `rounds` (lines 4-24) describe no real training, because the final step was a `--from-probs` run. They read like a misconfigured run.
   - The manifest hashes only top-level `src/*.py` (`er_predict.py:134`), so `src/ce`, `src/single`, `src/stack` and `attempts/*` except the final attempt are not covered by any hash.
2. **Shipped test predictions sit on the decisive path.** Three steps rest on second-family test probabilities whose generating commands were reconstructed after the fact: the meta-model (0.98779 → 0.99177 CV), the France blend and part of the candidate file.
3. **Intermediate evidence is thin.** Only 6 of 16 attempts ship a `results.json`. The README verification CVs for 018, 023, 028, 027p and 030p have no log or results file behind them, and no step logs are shipped.
4. **The fold schemes differ.** The second family uses `crc32(S1 id) % 5` folds (`src/stack/README.md:11`). The meta-model is cross-fitted on the main family's stratified folds (`src/er_model.py:112-129`). Each input is out-of-fold, but not on the same folds, so the meta-model CV can be slightly optimistic. It deserves one sentence, because Documentation §2.2 says "every stacked input is out-of-fold".

---

## Prioritised fixes (a few hours; none of these changes the output files)

| # | Severity | Where | Change | Why a judge cares |
|---|---|---|---|---|
| 1 | **HIGH** | `run_manifest.json`, `SHA256SUMS`, `src/er_predict.py:134-138`, `README.md:12` | Regenerate the manifest by rerunning the final `er_predict --from-probs` step from the *unpacked package*. The hash list then equals the shipped files. Alternatively, add a `note` field and a README paragraph naming the 17 unshipped modules as research or harness files that no shipped module imports and no step runs. **State explicitly which script produced "France extra keys"** (`er_keys.py` or the unshipped `er_keysfr.py`). In `--from-probs` mode, write `null` for `train_frac` / `params` / `rounds` / `features`. Extend `SHA256SUMS` to every shipped code file. Stop saying "git commit" if it is `nogit`. | Provenance is the first thing an auditor checks. Right now the manifest says the outputs came from a `src/` with 17 extra files. |
| 2 | **HIGH** | New `artifacts/final/` + README "Quick verification" section + new `run_all.sh` | (a) Ship the tail inputs: `cache/oof/test_030q_fix.parquet` and `test_006_stage2.parquet` (about 100-200 MB; optionally also `test_030p_fix`, `test_024_keys` and `feat/test_ce` so `er_france` and `er_fix 030q` can be rerun). Add a 10-minute path: `prepare_raw → er_dict → er_norm --split test → copy → er_predict --from-probs → sha256sum -c`, which should reproduce both TSVs byte for byte. (b) Transcribe the README list into `run_all.sh` with `set -euo pipefail`, phases `cpu1 \| gpu-handoff \| cpu2`, idempotent `ln -sfn` / `mkdir -p`, and a final copy to `../../output/`, the validator and `sha256sum -c`. | Today a judge cannot verify even the last step without 30-55 CPU-hours. One command and a 10-minute check turn "trust us" into "check us". |
| 3 | **HIGH** | `src/stack/README.md:99-115`, `src/stack/src/export_scores.py` | Close gap 1: run `export_scores.py` against the original `cache/ic_q` and `output_v08` if they still exist, and record that it reproduces the shipped `p_train_*` and `p_test_*` byte for byte. Otherwise, run the stack chain on a small S1 sample (e.g. 2% per country) end to end, ship the log, and say so in the README. | The meta-model and the France result rest on shipped test predictions whose generator "has not been run end to end". This is the single biggest reproducibility objection. |
| 4 | MEDIUM | `src/ce/ce_train.py:25,31-32,93,96,100,107-111`; `src/ce/ce_score.py:83-86`; `README.md:114-115` | Add `KIN`/`KOUT` overrides as in `ce_score`. `sys.exit(1)` if any worker return code is non-zero. Log the trained step count to `val_*.json` and ship `val_*.json` / `log_*.txt` for all runs. Accept an optional `"steps"` field in `ce_cfg.json` so a rerun can replay the original length. Say that `ce_cfg.json` must be *replaced* in the pairs dataset for e5-base (only the first one found is used), and that the scoring kernel needs the training kernel's output attached. Ship `kernel-metadata.json` templates. | The GPU half of the pipeline currently runs only through a hand-driven Kaggle UI with a wall-clock stop. |
| 5 | MEDIUM | `requirements*.txt`, `src/stack/kaggle/requirements.txt`, `README.md:49,54` | Add `pip freeze` lock files per environment (`requirements.lock`, `requirements-gpu.lock`, `src/stack/requirements.lock`). Say whether the GPU pins were installed or came from the Kaggle image (give the image date and CUDA version). Merge the two GPU pin files. Drop the unused `psutil` (main), `jellyfish` and `tabulate` (stack), or mark them as unused. | Pins cover only direct dependencies. LightGBM/OpenMP and tokenizer versions change results. |
| 6 | MEDIUM | `attempts/027p_meta/train.py:5`, `029f5_final/train.py:5,9`, `029f6_final/train.py:3,6,9`; `src/single/sidecar.py:14,20,24`; `src/single/fit.py:15,22,100,120`; `src/single/build.py:15` | Resolve every `cache/...` path against `cfg["cache_dir"]`. Use `Path(__file__)`-based `sys.path`. Replace `__import__("os")` with `import os`. | Moving `cache_dir` in `config.yaml` currently breaks five steps without any error. |
| 7 | MEDIUM | `attempts/*/train.py` `PRE_STEPS` / `PREDICT` strings | Delete these copies (or generate the README from them). Fix `024_keys/train.py:20` (`fit` vs symlink) and the `submissions/NN_` paths in `018_lexical/train.py:28` and `023_comb/train.py:18`. | Two drifting command lists make the reader wonder which one was run. |
| 8 | MEDIUM | README verification table (`README.md:144-159`), `attempts/*/` | Ship the stdout log of every step, plus the expected row counts and SHA256 of key intermediates (`cache/cand/test_cand.parquet` 78.9M rows, `oof/test_024_keys`, `oof/test_030q_fix`). Add a `results.json` for 018, 023, 028, 027p and 030p. | Lets a reproducer find where a rerun diverges instead of discovering it at the end. |
| 9 | MEDIUM | `src/er_norm.py:54-60` (and `src/stack/src/normalize.py`) | Raise if `indic.json` (or the stack's translit dictionaries) is missing, instead of silently using empty maps. | A skipped step now degrades the output without any error. |
| 10 | LOW | `README.md:140` | Change `mkdir -p output && cp …` to target `../../output/`, and state that the clean-run file is `03_029f6_final.tsv`. | The documented copy puts the outputs in the wrong folder. |
| 11 | LOW | `src/stack/src/stack_ce.py:20,56,64,452,460,518,546,606`; `high_model.py:10`; `export_extra.py:5` | Remove the "pending validation" note on `--decoy2`, which the final build uses. List the six unshipped scripts in the stack README as "research scripts not included; not on the documented path", or strip those branches. | Dangling references and "pending validation" on a production flag erode trust. |
| 12 | LOW | `src/er_frwords.py:50-51` → `src/er_france.py` | Have `er_france` compute the lists (or assert equality with `VETO_WORDS` / `BROAD_WORDS`) so the label-free derivation is actually executed. | It shows the France word lists are derived, not hand-picked. |
| 13 | LOW | README (new "Constants" table) | Map each threshold (0.001, 0.003/0.997, 0.01/0.99, (0.02, 10), floor 0.6, W 0.5, C 0.1, the er_frwords cutoffs and SHIFTS, `TEST_WEIGHTS`, CE sample sizes, and the band sizes asserted at `ce_score.py:55`) to the Documentation section that justifies it. | Removes the "magic number" objection at low cost. |
| 14 | LOW | `src/stack/src/decoy_feats.py:53` | Make `"\s*[.#:]?\s*(\w{1,5})|#\s*(\d{1,5})"` a raw string. | It is a `SyntaxWarning` today and will be an error in a future Python. |
| 15 | LOW | new `tests/test_decide.py` | Add 3-4 second-scale tests: `decide()` (the one-owner rule, the empty-answer rule, floor), `f05`, and the `write_lists` format. | Minimal evidence that the core decision rule behaves as documented. |
| 16 | LOW | README or Documentation §2.2 | Add one sentence on the fold mismatch (second family on crc32 folds, meta-model on stratified folds) and its expected effect on meta CV. | Pre-empts a methodology question raised by an engineering reader. |
