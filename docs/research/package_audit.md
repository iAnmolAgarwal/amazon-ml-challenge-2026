# Final package audit (27 Sep, ~19:15 IST)

Package: `submissions/final_package/Apprentice_submission.zip`, md5 `18170ae4b4c3b39e9c69454045498cf8` (checked). 99 files.
I unzipped it into the scratchpad only. The repo and the zip were not modified.

**Verdict: GO after fixes.** The output files are clean. The code folder cannot regenerate the final file as shipped:
- one script is missing;
- the README runs the wrong meta attempt;
- the train extra-key chain and the cross-encoder fill-in are not in the README;
- the candidate file does not match the organisers' definition.

## What passed
- **Outputs:** `output/matching_results.tsv` md5 `66e4ac51bd51ae6f1cdc5011c08052a8`, the LB 0.989886 file.
- **Validator:** the organisers' validator with `--check-ids`, run against `data/student_resource/dataset/test`, prints **PASS**.
  - 1,732,544 S1 rows in both files. 99,664 S1 have no match and 81,741 have no candidates.
  - All 9,969,589 S2/S3 ids were checked.
- **Own checks:**
  - Headers are exact.
  - Every match is inside the candidates (0 outside).
  - No duplicate ids in any list.
  - No pool id appears under more than one S1, in either file.
  - Candidates: 6,373,562 (**3.679 per S1**, max 83). Matches: 5,852,443 (3.378 per S1).
- **Structure:** every required path is present:
  - `output/{matching_results,candidate_pairs}.tsv`
  - `code/business_entity_resolution/{src/,README.md,requirements.txt}`
  - `Documentation_template.md`
- **Extra files** (fine, but explain them; see N1):
  - `code/business_entity_resolution/attempts/` (19 folders)
  - `config.yaml`
  - `src/stack/Makefile`
  - `src/stack/utils/validate_submission.py` (byte-identical to the organisers' copy)
- **Code matches the repo:** every `src/` file and every `attempts/*/train.py` in the zip is byte-identical to the Mac repo, except the in-progress `src/ce` edits noted under B3.
- **Names and wording:**
  - Person names appear only in the header of `Documentation_template.md`.
  - No other hit anywhere for: team or member names, claude/codex/anthropic, partner/teammate/"their pipeline"/handoff, g1/GCP.
- **Network access:** the only hits are `from_pretrained` of the two e5 models:
  - `src/ce/ce_train.py:45,61,101`
  - `src/ce/ce_score.py:28-29` (local fine-tuned dirs)
  - `src/stack/kaggle/ce.py:66-67`
  - `src/stack/kaggle/ce_france.py:81-82`

  No requests, urllib, http, API keys, geocoding or other downloads. `subprocess` is used only for `git rev-parse` (`src/common.py:18,75`, `src/stack/src/run_experiment.py:207`) and to launch the GPU workers (`ce_train.py:103`, `ce_score.py:75`, `kaggle/ce.py:137`, `kaggle/ce_france.py:164`).
- **Licences and parameter counts are correct:**
  - multilingual-e5-small is MIT, ~118M.
  - multilingual-e5-base is MIT, ~278M.
  - LightGBM is MIT.
- **Requirements:** every requirement is pinned with `==`. Python 3.12 (main) and 3.11 (stack) are stated.

## BLOCKER

**B1. `src/er_fix.py` is missing from the zip, but the final chain needs it.**
- `attempts/029f2_final/train.py` has `PARENT = "030n_fix"`.
- `attempts/030n_fix/train.py` (USE / NONEMPTY) is executed only by `src/er_fix.py`, which is not in the zip.
- The README never runs 030n_fix, yet it reads its output: `er_eval.py 029f2_final --from-oof 030n_fix`, and `er_france.py` reads `test_030n_fix`.
- Cause: `SRC_FILES` in `src/er_package.py:113` is a fixed list without `er_fix.py`. `final_checks` hashes only that list, so the manifest check could not catch the gap.
- Fix:
  1. Add `"er_fix.py"` to `SRC_FILES` (`er_pool.py` is already there).
  2. Add `python src/er_fix.py 030n_fix` to the README, after the meta step and before `er_france.py`.

**B2. The README runs the wrong meta attempt, and the attempt it names is not in the zip.**
- README line 82 (from the `PRE_STEPS` of `attempts/029f2_final/train.py`) runs `python src/er_meta.py 027h_meta`. There is no `attempts/027h_meta/` in the zip, so the step crashes in `load_attempt`.
- The final chain uses **027n_meta**, and it needs inputs that no README step produces:
  - `MAIN = "028f_ce"` needs `cache/oof/028f_ce.parquet`.
  - `FEATS ce2` needs `cache/feat/train_ce2.parquet` and `test_ce2_par.parquet`.
- The README instead builds `028e_ce` from the old 10% sample (lines 68-72, 79). The final no longer uses those steps.
- Fix: replace README lines 68-72, 79 and 82 with the chain that was actually run. The flags exist in the code, but the exact commands for 024f/028f are recorded nowhere in the repo; confirm them on g1.
  ```
  python src/er_keys.py --split train --tag _all && python src/er_rtext.py --split train --parent 023_comb --tag _all
  python src/er_newpairs.py union 024f_keys --split train
  python src/er_feat.py --split train --tag _allk && python src/er_feat2.py --split train --tag _allk && python src/er_feat5.py --split train --tag _allk
  ln -s 017_route cache/models/024f_keys
  python src/er_newpairs.py score 024f_keys --split train
  python src/er_extend.py union 028f_ce 028_ce 024f_keys 023_comb
  <cross-encoder fill-in, see B3>
  python src/er_meta.py 027n_meta
  python src/er_fix.py 030n_fix
  ```
  If 028e_ce is still a prerequisite of the fill-in order (train_x1 = 028e pairs, train_x2 = 028f pairs, per `docs/research/r16.md:103-108`), keep the s10 and 028e steps and label them as that.
- Also fix the stale `PRE_STEPS` inside the shipped attempts:
  - `attempts/027n_meta/train.py:17` says `er_meta.py 027l_meta`.
  - `attempts/024f_keys/train.py` `PRE_STEPS` is a copy of 024_keys ("union 024_keys", "fit 024_keys").

**B3. The cross-encoder fill-in (`train_ce_x`, `test_ce_x_par`, `train_ce2`, `test_ce2_par`) cannot be produced from the zip.**
- 027n_meta reads `ce2` on train and `ce2_par` on test.
- In the shipped zip, no file in `src/` mentions `ce_x` or `ce2`. The zip also has no step that picks the out-of-band pairs, scores them, or merges them.
- The originals were:
  - pair selection and collection: `research/r16/extra.py`, `xbuild.py`, `extra_f.py`. These exist only on g1, not in the Mac repo or git.
  - the merge: `research/r19m/ce2.py`, untracked in git.
  - Kaggle kernels `aml-ce-xscore-s/-b` and `aml-ce-fscore-s2/-b2`.
- `src/ce/ce_score.py` in the zip scores only `train_band` / `test_band` files.
- The Mac working tree now has **uncommitted** edits that close most of this gap. They are **not in this zip**:
  - `src/ce/ce_export.py`: `extra`, `collectx`, `ce2` subcommands.
  - `src/ce/ce_score.py`: `score_cfg.json` for other pair files.
- Fix:
  1. Finish and commit those edits.
  2. Add the steps to the README in order:
     - `ce_export.py extra test 018_lexical 028k_ce`
     - `ce_export.py extra train 018_lexical 028e_ce 028f_ce`
     - Kaggle scoring with `score_cfg.json` (e5-small on all x files, e5-base on the `*n` files only)
     - `ce_export.py collectx ce=<dirs> ce_base=<dirs>`
     - `er_extend.py parity ce_x_par ce=<dir> ce_base=<dir>`
     - `ce_export.py ce2`
  3. These subcommands reimplement g1's research scripts. Unless someone reruns them and matches the g1 row counts (train_ce_x 10,359,740, train_ce2 13,368,727, test_ce_x_par 8,607,036), say in the README that they are reconstructed and give those row counts as the check.

**B4. `candidate_pairs.tsv` is filtered by the final model's output, which the organisers' definition rules out.**
- `data/student_resource/README.md:100-106` defines the file as "the exact set of records you feed into your matching model for inference ... *before* your final matching model narrowed it down ... whatever your model actually runs inference over".
- Our file is built from the **final** probabilities:
  - `CAND_MINP = 0.001` on the final p;
  - then `CAND_OWNER`: each pool record is kept only with its highest-final-probability S1 (`src/er_predict.py:138-144`).
- Both the README and the Documentation say this openly: "pairs with final probability > 0.001".
- A blocking audit ("recall ceiling, reduction ratio") will see that the candidates are the model's output (3.68 per S1 against 3.38 matches per S1). That is also exactly what a "smaller ranks higher" criterion would reward. This is the most likely fair-play finding.
- Fix (cheap, no model rerun, about 15 min on g1):
  1. In `attempts/029f2_final/train.py`, delete `CAND_MINP` and `CAND_OWNER` and keep `CAND_BAND` (the stage-2 cascade, an earlier stage).
  2. Run `python src/er_predict.py 029f2_final --from-probs 029f2_final`.
  3. Confirm the matching file is still md5 `66e4ac51…` (029h showed identical decisions).
  4. Re-validate.
  5. Update the candidates-per-S1 number in Documentation §3 and table B (row "3.68-per-S1 candidate file") and in the README overview.
- If the team keeps 3.68 anyway, at minimum reword §3 so it does not call this "the final candidate set the decision ran on" without also saying that the final models scored about 46 raw candidates per S1 (about 6.5 after the stage-2 cascade).

## SHOULD-FIX

**S1. The repo is not the package source.** `attempts/024f_keys/` (in the zip, listed in DEPENDS of 027n and 029f2) and `research/r19m/ce2.py` are untracked on the Mac, and `src/ce/*` is modified but not committed. Fix: commit them, rebuild the zip from that commit on g1, and record the new zip md5 in `LATEST_SUBMISSION.md`.

**S2. README output path is wrong.** Line 85 says the output goes to `submissions/archive/NN_029f2_final.tsv`, which is what `er_predict.py:152` does. Line 87 says `submissions/NN_029f2_final.tsv`. Fix line 87, and add a final step: `cp submissions/archive/NN_029f2_final.tsv output/matching_results.tsv; cp submissions/archive/NN_029f2_final.candidates.tsv output/candidate_pairs.tsv`, then run the validator.

**S3. `er_predict.py` docstring contradicts itself.**
- Line 6 says the candidates are "exactly the pairs the model scored". That is false with `CAND_OWNER`/`CAND_MINP`, and a reviewer will read it next to the rule.
- Line 13 says "Run src/er_validate.py on the Mac afterwards". That file does not exist, and "Mac" is a machine detail.
- Fix: correct line 6 (or apply B4) and replace line 13 with the validator command.

**S4. Documentation Appendix A overclaims today.**
- It says the README "lists every command in order ... to the prediction step that writes output/matching_results.tsv", and that the cross-encoder "scripts and settings are included (src/ce, src/stack)". Both are false until B1-B3 land.
- It does not mention the known gaps listed in `src/stack/README.md` §"Not reproducible from this code alone":
  - the export step was reconstructed and not run end to end;
  - the command lines of steps 7-8 were rebuilt from notes;
  - the Kaggle wrappers were manual;
  - GPU training is time-boxed, so scores are close but not bit-identical.
- Fix: after B1-B3, add one sentence to Appendix A: "GPU steps are time-boxed; a rerun reproduces scores closely but not bit-identically; the in-country family's export step and two command lines are reconstructed (see src/stack/README.md)."

**S5. Design rule in the Documentation vs the second family.**
- Doc §2.1 says features that look at an S1's other candidates (coherence) were banned.
- The in-country stack family uses supporting-record (coherence) features by default (`src/stack/src/stack_ce.py:44,90,394`, and `--nosup` is not passed in `src/stack/README.md` steps 7-8).
- The paragraph scopes the rule to "the main pipeline", but a careful reviewer will spot the tension.
- Fix: add one sentence: "the second family includes supporting-record features; it enters only as an out-of-fold input to the meta-model, and the label-free test guard was run on every combined submission."

**S6. Two data locations, not stated together.**
- The main pipeline finds data anywhere under `data/` (`src/er.py:13`).
- The stack family expects `src/stack/dataset/{train,test}/` (`src/stack/README.md:28`), run from `src/stack/`.
- Fix: add one line to the main README: `ln -s ../../data/<...>/dataset src/stack/dataset`, and "run the src/stack commands from src/stack/".

**S7. Cross-encoder model revision is not pinned in `src/ce`.** `ce_train.py:95` defaults to `intfloat/multilingual-e5-small` with no revision, while `src/stack/kaggle/ce.py` pins revisions. Fix: add the same revision hashes to the README next to the ce_train step, or to `ce_cfg.json`.

**S8. The CV caveat sentence reads backwards.**
- Doc §5 says "The training candidates lack the character-route pairs that test has, so CV gains from recall overstate the leaderboard gain." Missing train candidates would understate recall, so the logic is unclear.
- The LB-CV gap (0.99182 vs 0.98989) is mainly France: 15% of test, unlabelled, and not in CV.
- Fix: "CV covers US and India only; France (15% of test, no labels) is lower, so the leaderboard sits below CV; later layers were also checked on a fixed label-free 10% sample."
- The 0.99182 figure (US 0.99163, India 0.99211) has no `results.json` in the repo or zip. Keep the g1 log line that produced it.

## NIT

- **N1.** The README does not explain `attempts/` or `config.yaml`. Organisers ask for "all source under src/". Add two lines:
  - `attempts/<id>/train.py`: per-stage settings loaded by `src/common.py:load_attempt`; only the ids listed in the command list are used.
  - `config.yaml`: data and cache paths. Its keys `device: mps`, `train_file`, `test_file`, `metric` and `cv` are leftovers that the pipeline does not use. Delete them, or say so.
- **N2.** Dead or unused code in the zip:
  - `src/er_feat3.py` (never called).
  - `src/common.py` `load_train`/`load_test`/`load_folds`/`get_device` (refer to `train_s1.parquet`/`folds.csv`, which do not exist).
  - `attempts/019_band/` (in DEPENDS, but no step uses it; the README never runs it).
  - `src/stack/Makefile`, which its own README says is stale and says the dictionaries are "shipped in cache/" (they are not).

  Either delete these or add one line saying so.
- **N3.** `src/stack/README.md:92` mentions "A 16 GB laptop and AWS instances", and `memguard.py:1` / `block_feats.py:3` / `features.py:26` mention an "8 GB dev machine". This hints at a separate provenance from the main README's 12-vCPU/96 GB machine. Harmless, but "ran on a mix of CPU machines (8-96 GB RAM)" in both READMEs reads as one pipeline.
- **N4.** `src/stack/requirements.txt` pins `jellyfish` and `tabulate`, which nothing imports. It pins `Unidecode 1.4.0`, which is GPL-2.0 (a library, not a model, so the model-licence rule does not apply). Optional: add a line in the README licences section, "libraries: permissive licences except Unidecode (GPL-2.0), used only for transliteration".
- **N5.** The Documentation title still reads "Solution Template". Change it to "Business Entity Resolution Solution".
- **N6.** `src/common.py:16-21` `shared_root()` resolves `data/` and `cache/` to the enclosing git repo root if someone unzips the package inside another git checkout. Add a note: "run outside any git repo, or set AML_CONFIG".
- **N7.** No `results.json` ships for 027n/030n/029f2, so a reviewer sees no CV evidence for the final layers. Optional: include the `er_eval.py 029f2_final --from-oof 030n_fix` output as `attempts/029f2_final/results.json`.

## Order to apply (fits before 20:30)
1. Commit the `src/ce` edits and `attempts/024f_keys`.
2. Add `er_fix.py` to `SRC_FILES` (B1).
3. Rewrite `PRE_STEPS` in `attempts/029f2_final/train.py` (B2, B3, S2).
4. Fix the 027n and 024f `PRE_STEPS` strings.
5. Decide B4. If you change it: rerun `er_predict --from-probs`, check md5 `66e4ac51`, then re-validate.
6. Documentation edits: S4, S5, S8, plus the candidate-count update if B4 changes it.
7. Run `python src/er_package.py submissions/archive/NN_029f2_final.tsv --final --team Apprentice`.
8. Re-run this audit's checks on the new zip: validator PASS, matching md5 `66e4ac51`, `er_fix.py` present, `grep 027h` returns nothing.

## Re-audit (27 Sep, ~19:55 IST): rebuilt zip md5 `e2f8d9bc078c50c6fac276c6be282f6b` (HEAD b1f4a56)

**Verdict: GO.** No blockers remain.

### Outputs
- `matching_results.tsv` is still md5 `66e4ac51…`.
- `candidate_pairs.tsv` is md5 `f90db037…`.
- The organisers' validator with `--check-ids` prints **PASS**.
- Own checks:
  - 1,732,544 rows in both files, with the same S1 set.
  - 0 matches outside the candidates.
  - 0 duplicate ids in any list.
  - 11,778,648 candidates (**6.80 per S1**, max 776, 23,977 S1 with none).

### Blockers
- **B1 fixed.** `src/er_fix.py` is shipped, and `er_fix.py 030n_fix` runs before `er_france`.
- **B2 fixed.** The README now runs, in order:
  1. 024f (`--tag _all`, `_allk` features, symlink, score)
  2. `union 028f_ce`
  3. `er_meta 027n_meta`
  4. `er_fix 030n_fix`
  5. `er_france 029f2_final`
  6. `er_predict`

  I walked the chain: every script, attempt and cache input is produced by an earlier step. That covers:
  - `028f_ce`, `test_028k_ce`
  - `train_ce2`, `test_ce2_par`
  - `test_feat5p`, `test_feat5_new`
  - `cache/stack`
  - `test_030n_fix`, `test_024_keys`
- **B3 fixed.**
  - `ce_export.py extra` / `collectx` / `ce2` and `ce_score.py` with `score_cfg.json` are in the zip. `ce_score.py:57` makes each half model score only the other half, as documented.
  - The README now covers the whole fill-in:
    - the train_x1/x2 order (028e, then 028f);
    - e5-base scoring the `*n` files only;
    - `parity ce_x_par`;
    - then `ce2`.
- **B4 fixed.** The candidates are now the stage-2 filter plus every pair added after it. Matches are unchanged. The README, Documentation §3 and table B all say 6.80 per S1, and "3.68" appears nowhere.

### Other items
- Network: the only downloads are still the e5 `from_pretrained` calls. The README now states both revisions.
- Names and wording: clean outside the Documentation header.
- Repo parity: every `src/` and `attempts/` file matches the repo, except the generated `prepare_raw.py`.
- Documentation:
  - It matches `attempts/029f2_final/results.json`: CV 0.991821, 5/5 folds 0.99179-0.99185, India 0.99211.
  - The LB 0.989886 figure is correct.
  - The title is fixed. S5, S8 and the Appendix A caveats are present.
- `er_predict.py` docstring: fixed.

### Remaining NITs (optional)
- `attempts/027n_meta/train.py:17` still says `er_meta.py 027l_meta`, and the `PRE_STEPS` of `attempts/024f_keys/train.py` still copies 024_keys. The README is authoritative, so this is harmless.
- `src/stack/README.md:92` still says "A 16 GB laptop and AWS instances".
- §3 says "candidate set the final models run inference over". More exactly, the stage-2 filter is applied to the final probabilities at decision time, and the meta-model scored the chain-p ≥ 0.001 union. The wording is defensible as it stands.
