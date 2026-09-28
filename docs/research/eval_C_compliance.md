# Evaluation C: compliance and blocking audit of `Apprentice_submission.zip`

Date: 28 Sep 2026. Reviewer role: compliance and blocking-audit evaluator, calibrated strictly (only the top 50 of about 1,000 teams go through; any compliance doubt is treated as possibly decisive).
Object: `submissions/final_package/Apprentice_submission.zip` (105 entries, 116,986,163 bytes, md5 `fec2e45900dc65b72b56de049d18eab2`). It was unzipped into a scratch folder and neither the zip nor the repo was modified. Judgements rest only on the zip's contents, with one exception: item H3 compares the zip's output with the leaderboard upload record, and it is marked as such.

Checks run:
- the organisers' `validate_submission.py` in two passes: matching plus candidates, then matching with `--check-ids`;
- a polars pass for ID existence on both files, matches as a subset of candidates, cross-country pairs, the one-owner rule and counts per S1 and per country;
- a full read of the pipeline code that decides the candidate set (`er_predict.py`, `er_meta.py`, `er_france.py`, `er_stage3.py`, `er_comb.py`, `er_fix.py`, `er_ce.py`);
- a scan of all 105 files for network use, keys, URLs, lookup tables, foreign names and paths, and licence text.

---

## Verdict

| Rule | Result | Evidence |
|---|---|---|
| Format (organisers' validator) | **PASS** | Both passes print `PASS`. 1,732,544 rows in each file, header exact, TAB separated, ASCII, LF line endings |
| Every S1 present, no duplicate rows, no duplicate IDs in a list | **PASS** | 1,732,544 of 1,732,544 S1 present. 0 duplicate rows. 11,778,648 candidate pairs, all unique. 5,847,885 matched pairs, all unique |
| Matched and candidate IDs are S2-/S3- IDs that exist in test | **PASS** | 0 unknown IDs in either file (checked against 9,969,589 test S2/S3 IDs). 0 S1 IDs in the lists. 0 cross-country pairs |
| Matches are a subset of candidates | **PASS** | 0 matched pairs outside `candidate_pairs.tsv`, and the validator gives no subset warning |
| Candidate file matches the organisers' definition | **PASS with caveats** | Mostly consistent: the band models, the meta-model and the decision all run on the filtered set. But the file is carved out of the final probability table after scoring, and some pairs the final models scored are left out of it (section 2.3) |
| Models: MIT or Apache 2.0 licence, at most 8B parameters | **PASS** (disclosure incomplete) | multilingual-e5-small (MIT, 117.7M) and multilingual-e5-base (MIT, 278M) with pinned revisions, plus LightGBM (MIT). The Documentation names only e5-small and states no licences |
| No external data, APIs or lookups | **PASS** | The only network access is the Hugging Face weight download. No API keys, URLs, geocoding or registries. The hard-coded tables are generic abbreviation and legal-form lists (section 5) |
| Fair play and plagiarism signals | **PASS with warnings** | Only the four member names appear. But `src/stack/` differs strongly from the main code in style, environment and paths, and its README uses wording like "kept as found" (section 6) |
| Package completeness and reproducibility (organisers: "regenerate both output files ... using only what is in this folder") | **FAIL** | The README command `python src/er_france.py 029f5_final` needs `attempts/029f5_final/`, which is not in the zip. `src/stack/README.md` also lists 7 "not reproducible from this code alone" gaps |
| (Outside the zip) The zip holds the leaderboard file | **UNVERIFIED** | The zip's `matching_results.tsv` is byte-identical to upload #20 (029f6, LB 0.990203). The last upload was #21 (029fmax), which differs by 35,306 pairs |

**Blocking quality: 7 / 10**, strict, against what the best 5% of teams would submit.

**Overall risk verdict: serious risks.** Nothing reaches disqualification level: the outputs are valid, the licences are fine and there is no external data. In a detailed top-50 review, however, three things could lower the team's standing: a hard reproduction failure, candidate-definition details that a strict reader can call inconsistent, and provenance signals in `src/stack/`. Whether the zip holds the leaderboard file could be decisive on its own if the portal ranks the last upload.

---

## 1. Format and ID checks (measured)

| Quantity | Value |
|---|---|
| Rows (both files) | 1,732,544, equal to test S1 |
| `matching_results.tsv` | 5,847,885 pairs, 3.375 per S1. 99,885 empty rows (5.8%). At most 11 matches per S1 |
| `candidate_pairs.tsv` | 11,778,648 pairs, 6.798 per S1. 23,977 empty rows (1.4%). Median 6, p90 12, p99 29, max 776 |
| Unknown IDs, self matches, duplicates, cross-country pairs | 0 each |
| Pool records matched to more than one S1 | 0 (one owner per record holds) |
| Peak memory of the validator | 3.6 GB with candidates, 2.4 GB with `--check-ids` (matching only). Both were queued through `scripts/queue.sh` |

The package also ships `src/stack/utils/validate_submission.py`, which is byte-identical to the organisers' validator. The stack README says it is the organisers' file, so this is fine.

## 2. Candidate set

### 2.1 Size and reduction

| Country | S1 | Candidates | Per S1 | Median | p90 | p99 | Max | Matches per S1 | Candidates minus matches, per S1 |
|---|---|---|---|---|---|---|---|---|---|
| US | 663,106 | 3,729,267 | 5.62 | 5 | 10 | 19 | 162 | 3.39 | 2.23 |
| India | 809,986 | 4,872,958 | 6.02 | 5 | 10 | 17 | 776 | 3.38 | 2.64 |
| France | 259,452 | 3,176,423 | **12.24** | 9 | 23 | 51 | 205 | 3.32 | **8.92** |
| All | 1,732,544 | 11,778,648 | **6.80** | 6 | 12 | 29 | 776 | 3.38 | 3.42 |

- Reduction ratio against all S1 × pool pairs (1.727e13): **0.99999932**. Against all within-country pairs (6.72e12): **0.99999825**.
- The candidate file is 2.01 times the emitted matches, so 49.6% of candidates are matched.
- France holds 27% of all candidates but only 15% of S1. 2,602 France S1 have more than 50 candidates, against 58 in India and 23 in the US. 30 France S1 have more than 100.
- The largest lists belong to generic names with a city-only address: "Shree Foods Limited, Bangalore" has 776 candidates, and "Saint-Nazaire Amis SAS" and "Bordeaux Culturelle" have about 200 each.
- Train analogue, from `attempts/029f6_final/results.json` in the zip: the final out-of-fold set has 13,373,981 pairs for 2,206,821 S1 (6.06 per S1), with an **oracle macro F0.5 of 0.99798** and a final pair recall of 0.9756. The Documentation does not quote this number (see M5).

### 2.2 How the set is produced (as documented)

1. Raw blocking produces about 80.5M test pairs, 46.5 per S1. The sources are:
   - per-country word TF-IDF, top 40 neighbours;
   - exact keys (house plus street, space-free name, sorted name, name plus house, initials plus house, alternate name), where oversized blocks are skipped rather than truncated;
   - character 3-gram retrieval.
2. Stage-1 and stage-2 LightGBM models score all of those pairs.
3. Pairs with stage-2 p ≥ 0.001 form the "band". Later sources add pairs: the route classifier (p > 0.001), extra name keys, reverse retrieval, and the second model family's cascade (p2 > 0.02, top 10 per S1).
4. The band models, the cross-encoder stack, the meta-model and the corrections run on this set.

### 2.3 Is the file what the final models run inference over? (code trace)

The file is written by `er_predict.py` with `CAND_BAND = ("006_stage2", 0.001)`: it keeps final-probability rows where the stage-2 p is ≥ 0.001 or missing (lines 132-137). What the final models actually scored:

- **US and India.** `er_meta.py:81` takes the full join of the chain pairs with chain p ≥ `MIN_P` (0.001) and **all** second-family test pairs, and scores the union (lines 131-133). The effects:
  - Second-family pairs that coincide with main pairs rejected by stage 2 are scored by the meta-model and then removed by `CAND_BAND`. The final ML model therefore ran on more pairs than the file lists.
  - The pool-side competition features (`pool_side`, lines 70-75, and again in `er_fix.py:62`) are computed over sets that include those dropped pairs.
  - The README's rule ("stage-2 p ≥ 0.001 plus later additions") is not what the US/India file holds. Pairs whose chain probability fell below 0.001 before the meta-model are also absent.
- **France.** `er_meta.py:134` passes **all** France chain pairs, with no `MIN_P`. `er_france.py:68-73` blends the whole France union, about 46 per S1 before filtering. `CAND_BAND` then drops the pairs whose stage-2 p is below 0.001. The blend is a rule, not an ML model, so France is defensible. But the France set gets no chain-probability filter, which explains a good part of France's 12.24 per S1.
- **Post-hoc look.** The filter is applied after final scoring, to the final probability table, in the same function that writes the file. The same docstring (line 10) and code (lines 86-94 and 138-144) also contain `CAND_MINP`, `CAND_TOPK` and `CAND_OWNER`, "research options ... [that] cut the written set further using final probabilities". The final attempt does not use them. A strict reader will still see tooling for shrinking the candidate file after the model has run, which is exactly what the definition forbids.
- **Matches.** None of this changes the matches: decisions are taken on the filtered set, and 0 matches lie outside it. The claim in Documentation §3 that "deciding on this filtered set gives the same matches as deciding on the full candidate union" is plausible but cannot be checked from the zip, because `decide()` uses E[T] and p0 over all rows of an S1.
- **Wording risk.** The organisers write "not the raw output of an early blocking pass you later filter further ... the *last* one: whatever your model actually runs inference over". The 46.5 → 6.8 reduction is done by the stage-1/2 LightGBM, a supervised matcher with 127 leaves trained on labels. The definition allows filtering stages, so the team's reading is defensible, and the README and Documentation disclose the 46.5 per S1 figure honestly. A strict judge may still count the effective blocking as 46.5 per S1.

## 3. Blocking quality (score 7 / 10)

| Criterion | Assessment | Score effect |
|---|---|---|
| Reduction ratio | 0.9999993. At the ceiling, like every serious team | neutral |
| Candidates per S1 | 6.80 mean, 2.0 times the matches. Honest top-5% cascades probably land at about 5-15 per S1, so this is likely in the top quartile but not the smallest | + |
| Recall ceiling | Train analogue 6.06 per S1 with oracle 0.9980. Raw union holds 98.1% of true pairs. Test France is unknown | + |
| Method | Multi-route blocking (word TF-IDF, exact keys with an oversize skip, character 3-grams, reverse retrieval, extra keys) plus a learned cascade. Sophisticated and well documented | + |
| Where the reduction comes from | 46.5 → 6.8 by a supervised LightGBM, not by blocking itself | − |
| Country balance | France 12.24 per S1 (twice US and India), with a long tail | − |
| Definitional hygiene | Filter applied after scoring, dormant trimming options, scored-but-dropped second-family pairs, README rule imprecise | − |
| Reported metrics | No recall or oracle figure for the submitted-definition set, and 0.9969 (Documentation) vs 0.9980 (`results.json`) are not reconciled | − |

**Can the set shrink without changing the matches, within the definition?**
- **Post-hoc cuts (not allowed).** Top-k, min-p or owner-only cuts on final probabilities would make candidates nearly equal to matches. They violate "whatever your model actually runs inference over" and invite a gaming finding. Do not do this.
- **Top-k cap at the stage-2 filter (not realistic now).** The file's size distribution gives these upper bounds:
  - cap 30 → 6.69 per S1 (−1.5%)
  - cap 20 → 6.54 (−3.8%)
  - cap 10 → 5.92 (−13%, 231,799 S1 affected)
  
  A cap requires re-running everything from `016_stage2b` onwards and would change the matches. The gain below cap 10 is negligible.
- **France consistency filter (the only candidate, if matches stay identical).** Apply the same chain p ≥ 0.001 filter that US and India get, keeping any pair the second family also proposes, before `er_france.py`. The rerun is CPU-only: `er_france 029f5_final` → `er_fix 030q_fix --no-cv` → `er_predict`. Ship it only if `matching_results.tsv` comes out **byte-identical**. The E[T] and p0 terms in `decide()` and the pool-side features in 030q can flip marginal decisions, and a changed matching file would no longer be the leaderboard file. The size of the gain has not been measured.

## 4. Licences and parameter counts

| Component | Licence | Parameters | Stated where | Correct? |
|---|---|---|---|---|
| intfloat/multilingual-e5-small (rev `614241f6…`) | MIT | 117.7M | README, stack README, `ce.py` | yes |
| intfloat/multilingual-e5-base (rev `d1287505…`) | MIT | 278M | README, stack README, `ce_france.py` | yes, but **missing from the Documentation** |
| Fine-tuned checkpoints | MIT (derived) | 12 in total: main e5-small ×2 and e5-base ×2; stack e5s ×2, e5i ×2, e5b ×2 and e5bc ×2. Largest 278M, sum about 2.4B | — | all ≤ 8B |
| LightGBM 4.7.0 | MIT | tree models | README | yes |
| microsoft/mdeberta-v3-base | MIT | — | named in the `src/stack/kaggle/ce.py:6` docstring only | not used, but it contradicts "no other pretrained models" |
| Libraries | polars/polars-runtime MIT, numpy BSD-3, scipy BSD-3, scikit-learn BSD-3, rapidfuzz MIT, sparse-dot-topn Apache-2.0, pyarrow Apache-2.0, pyyaml MIT, psutil BSD-3, torch BSD-3/Apache, transformers Apache-2.0, pandas BSD-3, jellyfish MIT, tabulate MIT, **Unidecode GPL-2.0+** | — | README:41 discloses Unidecode | yes |

Minor points in this area:
- `src/ce/ce_train.py:45,61,101` load weights without the pinned revision that the README says was used.
- `src/single/fit.py:73-75,112-115` contains an `xgboost` branch, but xgboost is not pinned in any requirements file. The branch is not used.

## 5. External data scan

- **Network use.** The only network use is `from_pretrained` inside the Kaggle cross-encoder scripts, downloading the declared weights. There is no `requests`, `urllib`, socket or HTTP client, no URLs, API keys or tokens, and no geocoding, registry or ER-service names.
- **Hard-coded tables.**
  - Abbreviations and street words in English, Indian and French forms (`er.py`, `er_norm.py`).
  - Legal forms (`er_gen.py`, `er_feat5.py`), fillers and honorifics.
  - The optional `CE_AUG` French rewrite in `ce_france.py`, which is off by default.
  
  All of these are general language knowledge, not entity lookups. The organisers' own tips encourage "country specific address patterns".
- **Transliteration maps.** Both families mine them from training pairs (`er_dict.py`, `stack/src/translit.py`). Nothing reads test labels, since no such file exists.
- **Test-derived literals.** `VETO_WORDS` and `BROAD_WORDS` (`attempts/029f6_final/train.py:11-12`) were most likely produced label-free by the marker rule in `src/stack/src/marker_veto.py` (n ≥ 500, lift ≥ 1.5) and then copied in as literals. Nothing in the zip says so (see M3).

## 6. Fair-play and provenance signals

- **Names.** Only the four members in the Documentation header. There are no other person, team or tool names, no licence headers from other projects and no absolute paths. "Harness folds" appears in docstrings and is benign.
- **`src/stack/` differs markedly from `src/`:**
  - libraries and environment: pandas/pyarrow vs polars; Python 3.11 vs 3.12; different numpy and scipy pins;
  - code conventions: `sys.path` inserts with `# noqa: E402`, inline comments and a Makefile, vs comment-free modules with attempt configs;
  - leftover pointers to a different project layout: output files in an `experiments/` folder that does not exist in the zip, relative paths that leave the package (`../kaggle` in `stack_ce.py:40`, `Makefile:40-51`, `export_band.py:8-9`, `export_extra.py:11`; `../../cache/stack` in `export_scores.py:57`), build names like `output_v03`/`output_v08`, and a macOS-only memory watchdog for an "8 GB dev machine" while the Documentation reports 96 GB Linux machines.
- **Wording that suggests inherited code:**
  - `src/stack/README.md:20` "the original make targets, kept unchanged";
  - `:90` "Hardware used originally";
  - `:97` "The original export was an unrecorded one-off";
  - `:111` "It is kept as found";
  - `cand_budget.py:3` cites "guidelines.md, new_additional_guidelines", files that are not in the package.
  
  One by one these are harmless. Together they match the "very different coding styles between folders" signal, and the Documentation gives no account of how the second family was built.
- **France tuning.** Several France-only rules (no labels) appear in Appendix B next to a leaderboard delta. §4 does give leave-one-country-out (LOCO) evidence for each rule. The table rows alone read as leaderboard-guided tuning.
- **Country as an open set.** The pipeline switches on the literal `"france"` in `er_meta.py:92,132,134`, `er_comb.py:28`, `er_fix.py:63` and `er_france.py:66`, and the stack reads fixed per-country files. The organisers ask for country to be treated as an open set. An unseen fourth country would not crash, but it would silently take the US path.

## 7. Red flags, ranked by likely impact

1. **Reproduction fails from the zip (HIGH).** The step at README:105 needs the missing `attempts/029f5_final/`, and `030q_fix` has `PARENT = "029f5_final"`. The second family, which feeds both the meta-model and the France blend, is self-declared not reproducible: the export step has never been run end to end, the commands of steps 7-8 are reconstructed, the Kaggle wrappers are missing and the transliteration dictionaries are not shipped.
2. **Leaderboard file identity (HIGH, outside the zip).** The zip holds #20 (029f6), but the last upload was #21 (029fmax). If the portal ranks the last upload, the package does not contain "the same file you upload to the leaderboard". The #21 matches are all inside the zip's candidate set (0 outside, 35,303 removed and 3 added relative to the zip).
3. **Candidate-definition hygiene (MEDIUM-HIGH).** The file is carved out of the final probabilities, dormant post-hoc trimming options remain in the code, second-family pairs are scored but dropped, the README rule is imprecise, and the effective blocking is 46.5 per S1 (section 2.3).
4. **Provenance signals in `src/stack/` (MEDIUM).** See section 6.
5. **Hard-coded, test-derived France word lists and literal-country branching (MEDIUM).** France-only steps are shown next to leaderboard deltas.
6. **Model disclosure incomplete in the Documentation (MEDIUM-LOW).** e5-base and the licence and parameter statement are missing, and the mdeberta mention and unpinned revision are inconsistent with the README.
7. **Unreconciled blocking numbers (LOW-MEDIUM).** The Documentation gives oracle 0.9969 for the larger union, while the zip's `results.json` gives 0.9980 for the smaller final set. No recall is given for the submitted set.
8. **GPL-2.0 Unidecode (LOW).** The rule covers models, and README discloses it.
9. **Housekeeping (LOW).** `er_predict.py:13` names `src/er_package.py`, which is absent. The unused xgboost branch is unpinned. `config.yaml` has unused keys (`device: mps`, and others). Path resolution depends on git (`common.py:shared_root`).

## 8. Prioritised fix list

Every fix below except M-optional leaves both output files byte-identical, unless it says otherwise.

### HIGH

- **H1. Missing attempt folder.** In `code/business_entity_resolution/attempts/`, add `029f5_final/train.py`. The repo has `attempts/029f5_final/`. Then check that every id in the README command list resolves: README:105 `029f5_final`, and the `030q_fix` `PARENT`. In README (after line 44), add one line saying that `028e_ce` and `028f_ce` are output names written by `er_extend.py`, not attempt folders.
- **H2. Close the second family's reproducibility gaps (`src/stack/README.md:95-111`).**
  1. Ship the mined dictionaries (`translit.json`, `translit_addr.json`) under `src/stack/cache/`. They are small and mined from train only.
  2. Run `src/stack/src/export_scores.py --ic e1 --build output_v08` once. Compare the SHA-256 of its outputs with the delivered `cache/stack/*.parquet`. Replace "has not been run end to end" with the verified result.
  3. Add the Kaggle kernel cells (the environment settings in the stack README table) as `src/stack/kaggle/run_*.py`.
  4. If a gap cannot be closed, ship the `cache/stack/` parquet files as declared intermediate artefacts. That way the main pipeline regenerates both outputs from the zip.
  5. Reword or delete "kept as found" (`:111`), "the original make targets, kept unchanged" (`:20`), "unrecorded one-off" (`:97`) and "Hardware used originally" (`:90`).
- **H3. Confirm which upload the portal ranks (outside the zip).**
  - If the portal ranks the best upload, or lets the team select one: select #20. The zip is then consistent and nothing changes.
  - If it ranks the last upload (#21): replace `output/matching_results.tsv` with `submissions/final_029fmax/22_029fmax_final.tsv` (its matches are a subset of the current candidates; the validator still passes). Then update Documentation §4, §5 and Appendix B and README to 029fmax, and re-zip.
- **H4. Make the candidate-set story exact and remove the post-hoc trimming code.**
  1. In `src/er_predict.py`, delete the `CAND_MINP`/`CAND_TOPK`/`CAND_OWNER` docstring lines (10-11), `cascade()` (86-94) and the two blocks at lines 138-144. The final attempt does not use them, so the outputs stay the same.
  2. Rewrite the docstring at line 8 to say the band filter repeats the upstream filter (`016_stage2b` scores exactly stage-2 p ≥ 0.001), applied here only because the chain carries every pair.
  3. In `README.md:11-14` and `Documentation_template.md:32`, state the rule the code applies:
     - US and India: the meta-model's input rows (chain p ≥ 0.001 plus second-family pairs), minus second-family pairs that stage 2 had rejected;
     - France: pairs of the France blend with stage-2 p ≥ 0.001 or proposed after stage 2.
  4. Count the second-family pairs that the meta-model scored but the file drops, in the repo:
     `polars: test_027p_meta ⋈ test_006_stage2 on (s1_idx, pool_idx), filter prob_006 < 0.001, count by country`.
     Queue it through `scripts/queue.sh`; it reads about 80M rows. If the count is small (under 1% of the file), add those pairs to `candidate_pairs.tsv`. The file then equals the meta-model's input, and the matches stay unchanged because `decide()` still runs on the filtered set. Otherwise, state the number in the README.

### MEDIUM

- **M1. Model disclosure.**
  - `Documentation_template.md` §4 (after line 44): add "Pretrained models: intfloat/multilingual-e5-small (MIT, 118M) and intfloat/multilingual-e5-base (MIT, 278M), pinned revisions, fine-tuned on the provided training pairs only (12 checkpoints across both families, largest 278M); LightGBM 4.7.0 (MIT); no other pretrained model; libraries permissive except Unidecode (GPL-2.0) in the second family's text normalisation."
  - `src/stack/kaggle/ce.py:6`: delete "/ microsoft/mdeberta-v3-base (MIT)".
  - `src/ce/ce_train.py:45,61,101`: pass `revision=` for the pinned commit, read from `ce_cfg.json` or from a default per model.
- **M2. Provenance.**
  - Documentation Appendix A, and the first paragraph of `src/stack/README.md`: say that the second model family was built by the team during the challenge in its own environment (pandas, Python 3.11), and why. Keep member names out, per team policy.
  - `src/stack/src/cand_budget.py:3`: replace "(guidelines.md, new_additional_guidelines)" with "(organisers' README)".
  - Make the in-package `ce_work` the default in `stack_ce.py:40`, `export_band.py:8-9`, `export_extra.py:11` and the Makefile targets at lines 40-51, instead of `../kaggle`.
  - Rename `output_v08` to a neutral `build` in the README command list and in `export_scores.py:56` only if that stays consistent with the delivered cache names. Otherwise leave it.
- **M3. France word lists.**
  - In `attempts/029f6_final/train.py:11-12`, and in one line of Documentation §4: state that `VETO_WORDS` and `BROAD_WORDS` are the output of the label-free marker rule on France's own candidate pairs (n ≥ 500, lift ≥ 1.5, `src/stack/src/marker_veto.py`), and why `BROAD_WORDS` exists.
  - Better, have `er_france.py` compute them with that rule, but only if it reproduces the same words (then the outputs stay identical).
- **M4. Country as an open set.**
  - Replace the literal `"france"` tests (`er_meta.py:92,132,134`, `er_comb.py:28`, `er_fix.py:63`, `er_france.py:66`) with "country not in the training countries". The outputs are identical on this test set.
  - Add one Documentation line: any unseen country follows the France path.
- **M5. Blocking numbers in Documentation §3 (lines 32-34).**
  - Add: test 11,778,648 pairs, 6.80 per S1 (US 5.62, India 6.02, France 12.24, with the reason France is higher); train analogue 13,373,981 pairs, 6.06 per S1, oracle macro F0.5 0.9980 (`attempts/029f6_final/results.json`), and the pair recall of that set.
  - Reconcile or correct the 0.9969 figure.
- **M6. Appendix B.** On France-only rows, add "rule fixed by LOCO before upload; LB shown for reference". Do not remove the numbers.

### LOW

- **L1. Stale references and unused code.**
  - `src/er_predict.py:13`: drop "(src/er_package.py runs it)".
  - `src/single/fit.py`: remove the xgb branch, or pin `xgboost` in `requirements.txt`.
  - `config.yaml`: remove the unused keys (`device`, `dtype`, `num_workers`, `train_file`, `test_file`, `metric`, `cv`) or keep the README note.
- **L2. Unidecode.** Keep it: replacing it would change the second family's outputs. M1 covers the disclosure.
- **L3. `common.py:shared_root`.** Fall back to the package root when the folder is not a git checkout, and drop the README warning about running outside other checkouts.
- **L4 (optional, changes outputs).** Try the France consistency filter from section 3. Ship it only if `matching_results.tsv` is byte-identical to the uploaded file and the validator passes. Then update the candidate counts in README and Documentation.
