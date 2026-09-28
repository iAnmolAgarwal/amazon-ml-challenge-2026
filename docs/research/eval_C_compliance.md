# Evaluation C: compliance and blocking audit of `Apprentice_submission.zip`

Date: 28 Sep 2026. Role: compliance and blocking-audit evaluator, judged against the top-50-of-about-1,000 bar.
Object: `submissions/final_package/Apprentice_submission.zip`, built 28 Sep 22:59. It is 658,830,817 bytes with 122 entries and unpacks to 814,793,606 bytes. This file replaces the earlier version, which audited the 17:07 build.

Method:
- The zip was unzipped into a scratch folder, except the 13 `artifacts/stack/*.parquet` files. Those were hashed and read in place through Python `zipfile`.
- Neither the zip nor the repo was modified. Only the zip's contents were judged. The organisers' validator and the test TSVs served as the reference.
- Heavy checks were queued one at a time through `scripts/queue.sh`. Peak RSS was 2.98 GB.

Checks run:
- The organisers' `validate_submission.py`, three times: default with both files; `--check-ids` with the matching file; `--check-ids` with both files.
- A polars pass over both outputs: rows, duplicates, prefixes, ID existence, cross-country pairs, one owner per pool record, the subset rule, and counts per S1 and per country.
- `artifacts/final/final_probs.parquet` mapped to IDs and compared with `candidate_pairs.tsv` pair by pair. The second family's `p_test_*.parquet` pairs checked against the candidate file.
- `SHA256SUMS`: all 121 entries, including the 13 stack parquets, streamed from the zip. `run_manifest.json`: every source hash and both output hashes.
- A code read of the path that fixes the candidate set: `er_block`, `er_stage3`, `er_newpairs`, `er_meta`, `er_france`, `er_fix`, `er_pool`, `er_predict`, and attempts `024_keys`, `027p_meta`, `029f5_final`, `029f6_final`, `030p_fix` and `030q_fix`.
- A scan of every file for:
  - network access, URLs, keys, downloads and subprocess calls;
  - hard-coded tables;
  - imported modules against the pinned requirements;
  - names, foreign paths and licence headers;
  - parquet writer metadata.

---

## Verdict

| Rule | Result | Evidence |
|---|---|---|
| Format (organisers' validator, `--check-ids`, both files) | **PASS** | `PASS - no blocking issues found`. 1,732,544 required S1; 9,969,589 valid S2/S3 IDs; no warnings |
| Every S1 present; no duplicate rows; no duplicate IDs in a list; only S2-/S3- IDs that exist in test | **PASS** | Both files: 1,732,544 rows in `test_source1` order. 0 missing, 0 extra, 0 duplicates, 0 bad prefixes, 0 unknown IDs, 0 cross-country pairs |
| Matches are a subset of candidates | **PASS** | 0 matched pairs outside the candidate file. The validator gives no subset warning, and `er_predict.py` asserts the same rule |
| Candidate file matches the organisers' definition ("the last stage, what the model runs inference over") | **PASS** (France ordering note) | US and India: the file is exactly the meta-model's input rows. France: the result is the same, but the code runs the France blend over every France row and applies the stage-2 cut afterwards (section 2) |
| Models: MIT or Apache 2.0, at most 8B parameters | **PASS** | intfloat/multilingual-e5-small (MIT, 118M) and multilingual-e5-base (MIT, 278M) at pinned revisions; LightGBM (MIT). The largest model has 278M parameters |
| Library licences | **PASS** (disclosed exception) | All permissive except Unidecode (GPL-2.0-or-later, used by the second family). It is a library, not a model, and it is disclosed |
| No external data lookup | **PASS** | The only network use is `from_pretrained` for the two declared e5 models. No API clients, URLs, keys, geocoders or registries. The hard-coded tables are general vocabulary, or derived from the provided data and documented |
| Fair play and plagiarism signals | **PASS with warnings** | Only the four members named in the header appear. No foreign names, tools, paths or licence headers. But `src/stack/` has a clearly different style, is a complete standalone ER system, and its README reads as reconstructed (section 6) |
| Country as an open set (organisers' instruction) | **PASS with note** | France is fully handled, but routing and one meta-model input use literal country names. Documentation Limitation (5) discloses this |
| Package structure and integrity | **PASS** | The organisers' layout is complete. All 121 `SHA256SUMS` entries match. All 31 `run_manifest.json` source hashes and both output hashes match |

**Blocking-quality score: 7 / 10.**

**Overall risk verdict: MINOR RISKS.** Nothing found would plausibly disqualify the team. Four items could lower its standing in the detailed review:
1. literal country names versus the open-set instruction;
2. the France candidate path (12.31 per S1, blend before the cut);
3. provenance signals around the second model family;
4. the partial reproducibility of that family and of the GPU steps.

---

## 1. Format, subset and integrity checks

| Check | `matching_results.tsv` | `candidate_pairs.tsv` |
|---|---|---|
| Rows / unique S1 | 1,732,544 / 1,732,544 | 1,732,544 / 1,732,544 |
| Test S1 missing / S1 not in test | 0 / 0 | 0 / 0 |
| Empty rows | 99,885 | 23,666 |
| Pairs | 5,847,885 (3.38 per S1) | 11,837,012 (6.83 per S1) |
| Duplicate pairs or IDs within a list | 0 | 0 |
| Non-S2/S3 IDs; IDs absent from test S2/S3 | 0; 0 | 0; 0 |
| Cross-country pairs | 0 | 0 |
| Pool IDs matched to more than one S1 | 0 | n/a |
| Matched pairs not in candidates | 0 | |
| SHA-256 vs `SHA256SUMS` and `run_manifest.json` | equal (`2b885f1e…071f9e`) | equal (`a8cda21e…49debe`) |

- **Validator runs**:
  - Default run with both files: PASS, with only the "ID check is off" warning.
  - `--check-ids` with the matching file: PASS.
  - `--check-ids` with both files: PASS, no warnings, peak 2.46 GB.
- **`final_probs.parquet`**:
  - It has 11,837,012 rows and equals the candidate file pair for pair (0 differences in either direction).
  - `decided` is true for 11,778,648 rows. Every matched pair is decided, and the lowest matched probability is 0.60004 against a floor of 0.6.
  - The 58,364 undecided rows (US 6,433, India 34,464, France 17,467) are exactly the README's "scored, never matched" pairs.

## 2. Is the candidate file the organisers' "last stage"?

Pipeline, as read from the code:
1. **Blocking** (`er_block.py`): word TF-IDF top 40 plus exact keys give 78.9M test pairs (46 per S1).
2. **Stage-1 and stage-2 LightGBM** score all of them (`er_predict.py 005f_floor`, `006_stage2`).
3. **Corrections**: the band model re-scores pairs with stage-2 p >= 0.001 (`er_stage3.py`). Route pairs from character retrieval, extra keys and reverse retrieval are added at route-classifier p > 0.001 (`er_newpairs.py`). The lexical, generator-operator and cross-encoder corrections re-score band pairs only. Every other pair keeps its stage-2 p, so the chain files carry every blocked pair.
4. **US and India** (`er_meta.py`): the rows are our chain pairs with p >= `MIN_P` 0.001 (`attempts/027p_meta`), full-joined with the second family's cascade (p2 > 0.02, top 10 per S1). The meta LightGBM scores these rows. `er_fix 030p` then re-scores the 0.01-0.99 band of the same rows.
5. **France** (`er_france.py`): our chain and the second family are blended. `er_fix 030q` re-scores the 0.01-0.99 band.
6. **Output** (`er_predict.py`): `CAND_BAND = ("006_stage2", 0.001)` defines the decided set. `CAND_ALSO` adds the second-family pairs that the final models scored but the decision skips.

Findings:
- **US and India (85% of S1): the file is exactly the meta-model's input.**
  - Every chain row of the meta input either is in the stage-2 band or was route-added, and both kinds are decided.
  - All 7,092,310 second-family test pairs (`p_test_*`) are in the file; none is missing.
  - The row counts reproduce the README's totals exactly.
- **The learned filters are allowed.** Stages 1 and 2 are classifier-based filters. The organisers' text explicitly allows several filtering stages and asks for the last one, so reporting the meta input rather than the 78.9M raw pairs is correct. The Documentation shows both numbers (46 and 6.83 per S1), which is transparent.
- **France (15% of S1, 27% of candidate pairs): correct in effect, but the order is reversed.**
  - `er_france.py:68` reads `base` as every France row of `test_024_keys`, which includes every stage-2-scored France pair. The blend at lines 72-79 therefore runs over all France blocking pairs, and the stage-2 cut is applied afterwards in `er_predict.py` (`CAND_BAND`).
  - The rows left out of the file all have p < 0.001 and no second-family score, and they sit outside `er_fix`'s band, so they cannot be matched. The outputs are therefore the same as if the cut came first.
  - A reviewer who traces the code will still see a filter applied after the final France step, which is the reverse of the definition. The README's rule (lines 29-40) describes the result, not the code order. Fix H1 removes this.
- **Over-inclusion is harmless.** The 58,364 scored-but-undecided pairs make the file larger than the decided set, which errs on the side of disclosure.
- **Feature context is outside the definition.** Upstream features (stage-2 competition, pool-side features in `er_comb` and `er_fix`) are computed over pairs that are not in the file. The definition covers the inference set, not feature context, so this is not a violation.

## 3. Blocking quality as a judge would see it

| | US | India | France | All |
|---|---|---|---|---|
| S1 / pool records | 663,106 / 3,817,031 | 809,986 / 4,717,565 | 259,452 / 1,434,993 | 1,732,544 / 9,969,589 |
| Candidate pairs | 3,735,700 | 4,907,422 | 3,193,890 | 11,837,012 |
| Per S1: mean / median / p90 / p99 / max | 5.63 / 5 / 10 / 19 / 162 | 6.06 / 5 / 10 / 17 / 776 | 12.31 / 9 / 23 / 51 / 205 | 6.83 / 6 / 12 / 29 / 776 |
| S1 with >= 20 candidates | 5,199 | 4,731 | 35,677 | 45,607 |
| Matched pairs per S1 | 3.39 | 3.38 | 3.32 | 3.38 |
| Share of candidates matched | 60% | 56% | 27% | 49% |
| From the second family's cascade (per S1) | 4.09 | 3.97 | 4.49 | |
| Only from our chain (per S1) | 1.54 | 2.09 | 7.82 | |
| Final p >= 0.01 (per S1) | 3.95 | 3.93 | 5.25 | 4.14 |

- **Reduction ratio**:
  - Against all within-country pairs (6.72 x 10^12): 1 - 11.84M / 6.72T = **0.99999824**.
  - Against the full cross product (1.73 x 10^13): 0.99999931.
  - The learned filters remove 85% of the raw 78.9M blocking pairs.
- **Recall ceiling** (train only; not checkable on test):
  - `results.json` reports an oracle macro F0.5 of 0.99798 for the final train candidate set, at 6.06 per S1.
  - The Documentation reports 51,473 true pairs never retrieved, about 0.7% of all true pairs.
- **How the set is produced**:
  - Retrieval and keys: per-country word TF-IDF plus exact keys, a character 3-gram route, capped name keys and reverse retrieval. Pairs from the added routes are cut by a route classifier.
  - Filters and union: two LightGBM filters, then a chain cut at 0.001 (US and India only), then a union with a second family that uses its own blocking and its own cascade (top 10 per S1).
  - The method is documented step by step with numbers in both the README and the Documentation.
- **Competitiveness at the top-50 bar**:
  - Strong: 6.8 candidates per S1 at a 0.998 ceiling, with a 0.9999982 reduction ratio, is a strong result.
  - Weak point: France is twice the size of the other countries, with a long tail (35,677 France S1 have 20 or more candidates).
  - Weak point: one India S1 has 776 candidates. 3 are matched and the median candidate is at p of about 3e-6. It is S1-246634726, "Shree Foods Limited", with the address "Bangalore, Karnataka".
  - Weak point: half the file is never matched, because it is the union of two families' candidate lists.
  - A team with a similar ceiling that reports 3.5-5 candidates per S1 would rank above this one on the blocking axis.
- **Can it shrink without changing the matches?** The matching file must stay byte-identical to the leaderboard upload, so only changes that cannot alter a decision qualify.
  - **France pre-blend cut: realistic and within the definition.** It is a real filter stage placed before the France blend, mirroring `MIN_P` for US and India: drop France pairs where both families' probabilities are below t (or missing). The estimates below come from `final_probs` plus `p_test_France`.
    - **t = 0.001**: removes about 496k pairs. France goes from 12.31 to about 10.40 per S1, and the total from 6.83 to about 6.55. These pairs end below 0.001 and outside `er_fix`'s band, so a decision change is practically impossible; only `er_fix`'s pool-side sums move, by less than 0.001.
    - **t = 0.01**: removes about 1.79M pairs. France goes to about 5.41 per S1, and the total to about 5.80 (-15%). A blend of two logits below logit(0.01) stays below 0.01 before the cross-encoder nudge (C = 0.1). However, that nudge and `er_fix`'s empty-address uplift can lift a pair into the correction band, so byte-identity must be verified before shipping.
  - **US and India: not possible without new matches.** The meta-model's pool-side inputs depend on the row set, and the model was trained with the current rule. Any pre-meta cut therefore needs retraining and changes matches.
  - **Not recommended**: relabelling the meta-model as a "filter" and listing only the pairs that `er_fix` re-scores. `er_fix` does not score matched pairs above 0.99, so that file would no longer be what the final model runs over.
- **Score: 7 / 10.**
  - Credit: the ceiling and reduction ratio; a file that honestly reflects the last stage and is exact to the pair; per-country accounting; the subset rule.
  - Deductions: France at 12.31 per S1 plus its tail; inflation from the union of two families; the France filter applied after the blend.

## 4. Licences and parameter counts

| Model | Licence | Parameters | Where stated | Correct? |
|---|---|---|---|---|
| intfloat/multilingual-e5-small, rev `614241f6…` | MIT | ~118M (117.7M) | README lines 41-47, Documentation 4.2, `kaggle/ce.py:17`, stack README | yes |
| intfloat/multilingual-e5-base, rev `d1287505…` | MIT | ~278M | same places | yes |
| LightGBM 4.7.0 (every tree model) | MIT | n/a | README, Documentation | yes |

All 12 fine-tuned checkpoints derive from these two e5 models, and the largest is far below 8B.

Libraries: every module imported by the code is pinned. Their licences:
- **MIT**: polars, lightgbm, rapidfuzz, pyyaml.
- **BSD-3**: numpy, scipy, scikit-learn, pandas, psutil, torch.
- **Apache-2.0**: sparse-dot-topn, pyarrow, transformers.
- **GPL-2.0-or-later**: unidecode. It is written "GPL-2.0" in `README.md:47` and Documentation line 87.

## 5. External data scan

- **Network**: `from_pretrained` only, at pinned revisions:
  - `src/ce/ce_train.py:47,63,105`;
  - `src/stack/kaggle/ce.py:66-67`;
  - `src/stack/kaggle/ce_france.py:81-82`;
  - `src/ce/ce_score.py:31-32` loads the local fine-tuned folders.

  There is no `requests`, `urllib`, `socket`, `wget` or `curl`, and no keys, tokens, secrets, cloud SDKs or geocoders. The only `https?://` is a regex that strips websites from names (`er_norm.py:23`).
- **Subprocess**: only `git rev-parse` and the launch of Kaggle two-GPU workers.
- **Hard-coded tables**:
  - General vocabulary: abbreviations, legal forms (including the French sarl, sas, eurl, sasu, selarl, eirl), French street types, honorifics, fillers and confusable characters (`er.py:70-84`, `er_norm.py:27-48`, `er_gen.py:32-36`, `er_feat5.py:36-38`, `er_frwords.py:21`, stack `normalize.py:21-45`, `block_feats.py:11`, `feat_ops.py:29`).
  - Derived from train and documented: the decoy shift set.
  - Derived label-free from France's test candidates and reproducible: the France VETO and BROAD words (`er_frwords.py`).
  - None of these is looked-up data. The Documentation does not say that the French vocabularies are hand-written, which fix L2 addresses.
- **Transliteration**: dictionaries are learned from train gold pairs only (`er_dict.py`; stack `translit.py`).
- **Dead augmentation path**: `src/stack/kaggle/ce_france.py:7,26,34-52` contains an off-by-default "frenchify" augmentation (`CE_AUG=0` for the shipped run). It rewrites provided text with hand rules. It is not external data, but a reviewer scanning for "augmentation" will stop on it (fix L3).
- **Label-free use of test data is wider than stated**. The Documentation (line 33) says "three places". The code also uses:
  - the IDF fitted on each split's pool (`er_block.py`, `vectorize`);
  - name-frequency counts (`er_feat2`);
  - competition and pool-side features over test pairs;
  - the France base rates in `er_lift`;
  - `TEST_WEIGHTS` (`er_eval.py:23`).

  None of this is prohibited, but the statement undercounts (fix M3).

## 6. Fair-play and plagiarism signals

- **Names and paths**: only the four team members in the Documentation header. There are no other people, teams, tools or assistants, no foreign project names or paths (no `/Users`, `/home`), and no licence headers. Parquet metadata carries only library versions.
- **Style split**:
  - Main pipeline: polars, Python 3.12, compact docstrings, `s1_idx`/`pool_idx`.
  - `src/stack/`: pandas, Python 3.11, inline `#` comments, `a_id`/`r_id`, `# noqa` imports, and a macOS memory watchdog that reads `kern.memorystatus_level` (`memguard.py:29-34`).
  - The same generator findings are implemented twice: `SHIFTS` in `er_feat5`, `er_feat6`, `er_lift` and `er_frwords`, and again in stack `decoy_feats` and `marker_veto`.
  - Documentation 4.2 item 6 and the stack README present the split as deliberate.
- **Second complete system**: `src/stack/src/predict.py:1-4,59` and `stack_ce.py:436` write their own `matching_results.tsv` and `candidate_pairs.tsv`. `predict.py` even defines candidates differently ("every blocked pair of every block").
- **Reconstructed-sounding wording**:
  - `src/stack/README.md:98-115` says "not logged", "not logged verbatim… rebuilt from the recorded configuration", "inferred from the change history", and "those files were not kept".
  - Documentation Limitation (1) says "two command lines of the second family were rebuilt from its run records".
  - Together these read as if the person who packaged the code reconstructed someone else's run.
- **Assessment**: there is no evidence of copying. This combination is the most likely trigger of a provenance question in the detailed review (fix M1).

## 7. Red flags, ranked by how likely they are to disqualify the team or lower its ranking

1. **Literal country names versus the open-set instruction.** Effect: lower ranking; disqualification is unlikely.
   - The organisers wrote: "Treat country as an open set… do not hard-code, filter, or one-hot your pipeline to only {US, India}".
   - The main pipeline routes France by name: `er_meta.py:132-134`, `er_comb.py:28`, `er_fix.py:63`, `er_pool.py:128`, `er_france.py:42,66`, `er_lift.py:40,63`.
   - It feeds a country code, cty = india + 2 x france, to the meta-model (`er_meta.py:92`, `single/build.py:35`).
   - The stack loops over fixed country lists: `export_band.py:45`, `feat_decoy.py:64`, `export_scores.py:62-64`, `decoy_feats.py:214`, `high_model.py:34`.
   - A fourth country would be scored as US by the meta-model and would get no second-family scores.
   - This does not break the letter of the instruction, since France is fully handled and Limitation (5) discloses it. A reviewer checking the instruction will find it quickly all the same.
2. **Provenance signals around `src/stack/`** (section 6).
3. **France candidate path**: 12.31 per S1 with a long tail, and the blend-before-cut code order (sections 2 and 3).
4. **Reproducibility**:
   - The second family's outputs are shipped as artefacts rather than rerun. Two of its command lines were rebuilt, and its transliteration dictionaries were not kept.
   - The GPU steps are not bit-reproducible.
   - Research scripts are referenced but not shipped (`stack_ce.py:56,452,460,518,546,606,609`; `high_model.py:10`; `export_extra.py:5`; `marker_veto.py:11`).
   - The LOCO grids and the test guard behind the France rules are not in the package (Limitation 6).
   - A full rerun costs 30-55 CPU hours, 15-20 GPU hours and 120 GB of disk. The organisers ask that "anyone should be able to regenerate both output files… using only what is in this folder".
5. **A France change was kept on leaderboard evidence** (B.1 last row; section 2.2). It is honestly disclosed and not prohibited, but it is a public-leaderboard-selected change with a small private-leaderboard risk.
6. **The test-data usage statement undercounts** (section 5).
7. **GPL-2.0-or-later Unidecode**: a library, and disclosed.
8. **Hygiene**:
   - `requirements-gpu.txt` pins pandas 3.0.6 and pyarrow 25.0.1, but the shipped cross-encoder score parquets were written by pandas 2.3.3 and pyarrow 24.0.0 (from their metadata).
   - `run_manifest.json` has placeholder fields: `n_entities 0`, a `sample_sha256` of empty input (`e3b0c442…`), `features []` and `params null`.
   - `results.json` records `git_commit: "nogit"`, while the manifest records `be563bd`.
   - The "frenchify" dead path.

## 8. Prioritised fix list

Every fix must leave `output/matching_results.tsv` byte-identical (`2b885f1e…071f9e`). After any file change, regenerate `SHA256SUMS` (and, if `er_predict.py` reruns, `run_manifest.json` and `final_probs.parquet`), then rebuild the zip.

### HIGH

**H1. France: cut before the blend (fixes the order and shrinks the file).**
- Where: `src/er_france.py` (after line 75); `attempts/029f5_final/train.py` (new key); `README.md:29-40`; Documentation section 3 table, line 51 and the 4.1 pipeline.
- Change:
  1. Add `FR_MIN_P = 0.001` to `attempts/029f5_final/train.py`.
  2. In `er_france.py`, after the line that nulls broad-word zeros (line 75), insert `t = getattr(mod, "FR_MIN_P", 0)` and `d = d.filter((pl.col("pA") >= t) | (pl.col("pS") >= t))` when `t > 0`.
  3. Re-run `python src/er_france.py 029f5_final`, then `python src/er_fix.py 030q_fix --no-cv`, then `python src/er_predict.py 029f6_final --from-probs 030q_fix`.
- Ship only if the matching SHA-256 is unchanged. France goes from 12.31 to about 10.4 per S1 and the total from 6.83 to about 6.55. Update the numbers in the README and the Documentation.
- Optionally try `FR_MIN_P = 0.01` first (France about 5.4, total about 5.8), under the same identity gate.
- If the output is not identical: revert, and add one README sentence explaining that the France blend is computed over all France rows, that the cut sits in `er_predict.py` (`CAND_BAND`), and why the excluded rows (p < 0.001, no second-family score) cannot be matched.

**H2. Open-set country handling without literal names.**
- Where: `src/common.py` (new helper); `er_meta.py:92,132,134`; `er_comb.py:28`; `er_fix.py:63`; `er_pool.py:71,128`; `er_france.py:42,66`; `er_lift.py:40,63`; `single/build.py:35`; Documentation Limitation (5).
- Change:
  1. Add `countries(cfg)` to `common.py`. It returns the train countries ordered by train S1 count (`us`, `india`) and the unlabelled set (test countries minus train countries).
  2. Replace `== "france"` with `.is_in(unlabelled)` and `!= "france"` with `.is_in(labelled)`.
  3. Compute `cty` as the index in the labelled order, and as `len(labelled)` for unlabelled countries. This reproduces 0, 1 and 2 exactly.
  4. Before shipping, assert on `cache/norm/{train,test}_s1.parquet` that the new expressions equal the old ones column for column, and re-run the three cheap final steps for the SHA check.
  5. Reword Limitation (5): the unlabelled path is chosen by "no training labels", not by name.
  6. Optionally, derive the stack's country lists from the data in the same way.
- If the equivalence check cannot be run, leave the code unchanged and expand Limitation (5) instead. An untested edit to upstream steps is worse than a disclosed limitation.

### MEDIUM

**M1. Provenance of `src/stack/`.**
- Where: `src/stack/README.md:1-6` and `98-115`; Documentation 4.2 item 6 and Limitation (1); `src/stack/src/predict.py:1-4,59`; `src/stack/src/stack_ce.py:436`.
- Change:
  1. Add a dated, neutral provenance line: when the team started it, why it uses pandas and Python 3.11, and that it was developed first on a small machine, which explains `memguard.py`.
  2. Rewrite the reproduction gaps as first-person facts ("we did not save the export command; `export_scores.py` writes the documented format"). Drop "inferred from the change history" and similar wording.
  3. Rename the stack's standalone outputs to `stack_matching.tsv` and `stack_candidates.tsv`, and label them intermediate diagnostics that are not submitted.
  4. Fix the candidate definition in `predict.py`'s docstring.

**M2. Remove references to unshipped research code.**
- Where: `src/stack/src/stack_ce.py:56,452,460,518,546,606,609`; `high_model.py:10`; `export_extra.py:5`; `marker_veto.py:11`.
- Change: delete the off-by-default branches (`MONO`, `UNSEEN_ADDR_SHIFT`, `--cnt`, `--lex`, `cr_add`) and their mentions of `loco_transfer.py`, `feat_cnt.py`, `fr_addr_rule.py`, `cr_score.py` and `lexical.py`. This does not change behaviour, and the shipped artefacts do not depend on these branches.
- If the LOCO and test-guard scripts behind the France rules exist, ship them (they close Limitation 6).

**M3. State the full label-free use of test data.**
- Where: Documentation line 33 (section 2.2).
- Change: replace "in three places" with a complete list:
  - per-split unsupervised statistics: IDF fitted on each split's pool (`er_block.py`), name frequencies (`er_feat2.py`), competition and pool-side features over test candidates;
  - the France decoy-marker table and country lift (`er_feat6.py`, `er_lift.py`);
  - the France word lists (`er_frwords.py`);
  - the test guard's counts.

### LOW

- **L1. Unidecode licence wording.**
  - Where: `README.md:46-47`; Documentation line 87.
  - Change: write "GPL-2.0-or-later, used unmodified as a pip dependency and not redistributed; the licence rule applies to models".
- **L2. Disclose the hand-written vocabularies.**
  - Where: Documentation section 3 or "Features used".
  - Change: add one sentence naming the abbreviation, legal-form (including French), street-type and honorific lists, with their files (section 5), and stating that they are general knowledge with no external tables.
- **L3. Remove the dead "frenchify" augmentation.**
  - Where: `src/stack/kaggle/ce_france.py:7,26,34-52,89,136`.
  - Change: delete it. `CE_AUG=0` was used for every shipped run.
- **L4. Record the versions that actually wrote the cross-encoder scores.**
  - Where: `requirements-gpu.txt`.
  - Change: record pandas 2.3.3 and pyarrow 24.0.0 (from the shipped score parquets), or add a comment that the Kaggle default image was used.
- **L5. Clean the manifest fields.**
  - Where: `src/er_predict.py:145`.
  - Change: for `--from-probs` runs, omit the training fields (`n_entities`, `sample_sha256`, `n_train_pairs`, `features`, `params`). Make `results.json` `git_commit` consistent with the manifest. Do this only if H1 reruns `er_predict.py` anyway.
- **L6. Show the candidate-size distribution.**
  - Where: README, section "Candidate file".
  - Change: add the per-S1 distribution (median 6, p90 12, p99 29, max 776) and one line on the tail, so the reviewer sees it is known.
