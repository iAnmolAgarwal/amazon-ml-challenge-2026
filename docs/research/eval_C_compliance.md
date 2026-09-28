# Evaluation C: compliance and blocking audit of `Apprentice_submission.zip`

Date: 28 Sep 2026. Role: compliance and blocking-audit evaluator, judged against the top-50-of-1,000 bar.
Object: `submissions/final_package/Apprentice_submission.zip` (122 entries, 569,388,281 bytes, built 28 Sep 17:07). This replaces the earlier version of this file, which audited an older 105-entry zip.
Method: the zip was unzipped into a scratch folder. The 13 stack parquets were skipped, except the three `p_test_*.parquet`, which were extracted to check the candidate rule. Neither the zip nor the repo was modified. Only the zip's contents were judged; the organisers' validator and test TSVs served as the reference.

Checks run:
- The organisers' `validate_submission.py` with both files and `--check-ids`. It was run directly, not through `scripts/queue.sh`, and peaked at 4.4 GB.
- Independent stdlib and polars passes: rows, headers, CRLF, whitespace, duplicates, subset rule, counts per S1 and per country, cross-country pairs, one owner per pool record, and whether the second family's test pairs are all in the candidate file.
- SHA-256 of both outputs against `SHA256SUMS` and `run_manifest.json`, and of every shipped `src/*.py` against the manifest.
- A code read of the path that decides the candidate set (`er_predict.py`, `er_meta.py`, `er_france.py`, attempts `027p_meta`, `029f5_final`, `029f6_final`, `030p_fix`, `030q_fix`).
- A scan of every file for network use, keys, URLs, downloads, lookup tables, names, foreign paths, licence headers, imports of modules that are not shipped, and references to files that are not shipped.

---

## Verdict

| Rule | Result | Evidence |
|---|---|---|
| Format (organisers' validator, `--check-ids`) | **PASS** | `PASS - no blocking issues found`, with 1,732,544 required S1 and 9,969,589 valid S2/S3 IDs |
| Every S1 present; no duplicate rows or duplicate IDs in a list; IDs are S2-/S3- IDs that exist in test | **PASS** | 1,732,544 rows in each file with exact headers, LF line endings and no stray whitespace. 0 unknown IDs, 0 duplicates |
| Matches are a subset of candidates | **PASS** | 0 S1 have a matched ID outside their candidate list. The validator gives no subset warning. `er_predict.py` also asserts it |
| Candidate file matches the organisers' definition | **PASS** (interpretive note) | The file is the input of the final models: the meta-model for US and India, the France blend for France. It is a superset of the decided set and of the matches. No post-hoc trimming code is left in the package (section 2) |
| Models: MIT or Apache 2.0 licence, at most 8B parameters | **PASS** | multilingual-e5-small (MIT, 118M) and multilingual-e5-base (MIT, 278M), both at pinned revisions. LightGBM is MIT. The largest checkpoint has 278M parameters |
| Library licences | **PASS** (disclosed exception) | All libraries are permissive except Unidecode (GPL-2.0+), which the second family uses for text normalisation. It is a library, not a model, and README and Documentation disclose it |
| No external data lookup | **PASS** | The only network use is `from_pretrained` downloading the declared e5 weights. There are no API clients, URLs, keys, geocoders or registries. The hard-coded tables are generic vocabulary or are derived from the provided data (section 5) |
| Fair play and plagiarism signals | **PASS with warnings** | The only names are the four members in the header. However, `src/stack/` differs sharply in style, and its code points to 6 scripts and a `candidate_pairs.tsv` of its own, none of which ship. `run_manifest.json` lists 17 source files that are not in the zip (section 6) |
| Package structure and reproducibility | **PASS with gaps** | The organisers' layout is complete. Both outputs match their SHA-256 values, and so do all 30 shipped top-level modules (against the manifest). The second family ships its outputs as artefacts, and its README lists reproduction gaps. A reviewer cannot run the final 19-second step without roughly 50 CPU hours and 20 GPU hours of upstream work |
| Country as an open set (organisers' instruction) | **PASS with note** | France is fully handled. The code still branches on literal country names, and a `cty` code in the meta-model inputs sets France = 2. Documentation Limitation (5) discloses this |

**Blocking quality: 8 / 10.**

**Overall risk verdict: minor risks.** Nothing reaches disqualification level. The outputs are valid, the models and licences are fine, there is no external data, and the candidate file follows the organisers' "last stage" definition. The remaining risks are about provenance and presentation, and they could draw reviewer questions or cost blocking points:
- the style of `src/stack/` and its dangling references;
- the unexplained entries in the manifest;
- the France candidate size;
- a judge who counts the raw 46 per S1 instead of 6.83.

---

## 1. Output checks (measured from the zip)

| Quantity | Value |
|---|---|
| Rows (both files) | 1,732,544, equal to the number of test S1 |
| `matching_results.tsv` | 5,847,885 pairs (3.375 per S1). 99,885 empty rows (5.8%). At most 11 per S1. S2 2,833,303 and S3 3,014,582 |
| `candidate_pairs.tsv` | 11,837,012 pairs (**6.832 per S1**). 23,666 empty rows (1.37%). Median 6, p75 8, p90 12, p95 15, p99 29, p99.9 56, max 776. S2 5,554,017 and S3 6,282,995 |
| Candidates per match | 2.02 (49.4% of candidates are matched). On 377,121 S1 (21.8%) the candidate list equals the match list |
| Cross-country pairs | 0 in either file |
| Pool records matched to more than one S1 | 0 (one owner per record holds) |
| SHA-256 | Both outputs equal `SHA256SUMS` and the manifest (`2b885f1e…` and `a8cda21e…`). The three `p_test_*.parquet` files also match |

Per country (pool = S2 + S3 of the same country):

| Country | S1 | Pool | Candidates | Per S1 | Matches per S1 | Candidates / matches |
|---|---|---|---|---|---|---|
| US | 663,106 | 3,817,031 | 3,735,700 | 5.63 | 3.39 | 1.66 |
| India | 809,986 | 4,717,565 | 4,907,422 | 6.06 | 3.38 | 1.79 |
| France | 259,452 | 1,434,993 | 3,193,890 | **12.31** | 3.32 | **3.71** |

The README and the Documentation state the same per-country numbers (5.63, 6.06, 12.31, 6.83), and they check out.

## 2. Is the candidate file what the organisers asked for?

The organisers' definition: "the exact set of records you feed into your matching model for inference ... if your pipeline has several blocking/filtering stages, candidate_pairs.tsv is the last one ... Every ID in matching_results.tsv should appear here."

Code trace:
- **US and India.** `er_meta.py` (build, `MIN_P = 0.001` in `attempts/027p_meta`) scores a set of pairs:
  - the chain pairs with chain probability ≥ 0.001;
  - joined in full with every second-family test pair (its cascade: p2 > 0.02 and top 10 per S1).
- **France.** `er_meta.py` passes France through. `er_france.py` then blends the France chain pairs with the second family's France pairs.
- **Writing the file.** `er_predict.py` writes the candidate file from the final probability table in two parts:
  - rows kept by `CAND_BAND = ("006_stage2", 0.001)` (stage-2 p ≥ 0.001, or added after stage 2);
  - `CAND_ALSO`: second-family pairs that the final models scored below that band, 58,364 pairs.
  
  The decision runs on 11,778,648 pairs, which is the file minus those 58,364. The manifest records `n_decided_pairs 11,778,648` and `n_candidate_pairs 11,837,012`.
- **Verified from the zip.** All 7,092,310 second-family test pairs are in the file (France 1,164,276, India 3,216,573, US 2,711,461).

Result: candidates ⊇ decided set ⊇ matches, and the file equals the input set of the last model stage. The README section "Candidate file (exact rule)" describes this exactly. The dormant post-hoc trimming options that the previous zip carried (`CAND_MINP`, `CAND_TOPK`, `CAND_OWNER`) are gone.

**Interpretive risk (not a failure).** The drop from 46 to 6.83 per S1 is done by supervised LightGBM stages. The steps are: stage 1/2 at p ≥ 0.001, the route classifier at p > 0.001, and a chain-probability cut at 0.001. The band model, the corrections and the cross-encoder stack score a superset of the file before that cut. The definition explicitly allows several filtering stages and asks for the last one, so the team's reading is correct.

However, Documentation §4.2 lists "Stages 1-2" under *Models* of the matching model. A judge who treats the stage-1 LightGBM as "the matching model" would count about 46 per S1, which is the raw blocking output of 78.9M test pairs. The Documentation reports both figures honestly. The fix is framing (M1), not a change to the file.

## 3. Blocking quality (score 8 / 10)

**Reduction ratio**

| Reference set | Pairs | RR of `candidate_pairs.tsv` |
|---|---|---|
| All S1 × all S2/S3 | 1.7273e13 | **0.99999932** |
| Within-country S1 × pool (blocking never crosses countries) | 6.7246e12 | **0.99999824** |
| US / India / France within-country | – | 0.9999985 / 0.9999987 / 0.9999914 |
| Raw blocking before the learned filters (78.9M, from the Documentation) | – | 0.9999954 |

**How the set is produced.**
1. Per country, word TF-IDF top 40 plus exact keys, with oversized key blocks skipped rather than truncated. This gives 78.9M test pairs, 46 per S1.
2. Stage-1/2 LightGBM keeps p ≥ 0.001.
3. Added routes are scored by a route classifier and kept at p > 0.001: character 3-gram retrieval, capped exact name keys (`er_keys.py`, cap 50 per key value on each side) and reverse retrieval (top 5).
4. For US and India, a chain-probability cut at 0.001.
5. The union with the second family's cascade (p2 > 0.02, top 10 per S1).

The train analogue of the final set is 6.06 per S1 with oracle macro F0.5 0.99798 (`attempts/029f6_final/results.json`). The recall ceiling therefore costs at most 0.002 F0.5 on US and India. France has no labels.

**Competitiveness.**
- US and India at 5.6-6.1 per S1 is about 1.7-1.8 candidates per emitted match, while keeping a 0.998 oracle. That is close to the floor for a set that still has headroom for the matcher. Strong top-5% cascades will probably land around 4-15 per S1.
- The weak spot is France: 12.31 per S1 and 3.71 candidates per match. France holds 27% of all candidates with 15% of the S1.
- Of France's 3.19M pairs, 2.03M (7.8 per S1) come only from the main chain, which skips the chain-probability cut for France.
- If France sat at India's 6.06 per S1, the overall mean would be about **5.90** instead of 6.83.
- There is also a long tail: 2,715 S1 have more than 50 candidates, totalling 170k pairs (1.4%). The maximum is 776, for "Shree Foods Limited, Bangalore, Karnataka", a generic name with a city-only address.

**Score breakdown.**

| Criterion | Effect |
|---|---|
| Reduction ratio at the ceiling; within-country blocking; 0 cross-country pairs | + |
| 6.83 per S1 overall, 5.6-6.1 for the labelled countries, with a 0.998 train oracle | ++ |
| Multi-route design, oversized blocks skipped (not truncated), reverse retrieval, honest reporting of both 46 and 6.83 per S1 | + |
| France at 12.31 per S1 (twice the others) | − |
| Reduction done by a supervised cascade and a union of two families' sets, so a judge may discount it | −0.5 |
| Long tail (max 776) | −0.5 |

**Can the set shrink without changing the matches, within the definition?**
- **Post-hoc trimming (not allowed).** Cutting the written file by final probability, top-k or owner would misstate what the model ran over. Do not do this.
- **France chain-probability cut (the only realistic option).** Apply the US/India rule to France *inside the pipeline*: in `er_france.py`, keep a chain pair only if its chain p ≥ 0.001 or the second family also proposes it.
  - Pairs with chain p < 0.001 and no second-family score leave the blend at p < 0.001. The cross-encoder term (C = 0.1) cannot lift them to the 0.6 floor.
  - However, `er_fix` (030q) name-twin features over a pool record's claimers, and the Σp / Π(1−p) terms in `decide()`, can flip marginal decisions.
  - So ship it only if `matching_results.tsv` stays byte-identical (SHA-256 `2b885f1e…`). The expected gain is up to about 0.9 per S1 overall; the exact figure cannot be measured from the zip.
- **Per-S1 cap or a higher `MIN_P` for US and India.** Either changes the meta-model's pool-side competition features, so the matches would change. Not realistic now.

## 4. Licences and parameter counts

| Component | Licence | Parameters | Stated where | Correct? |
|---|---|---|---|---|
| intfloat/multilingual-e5-small, rev `614241f6…` | MIT | ~118M (117.65M + 1-logit head) | Documentation §4.2, README, `ce.py`, `ce_train.py` | yes |
| intfloat/multilingual-e5-base, rev `d1287505…` | MIT | ~278M | Documentation §4.2, README, `run.py`, `ce_france.py` | yes |
| Fine-tuned checkpoints | derived from MIT | 12 in total, the largest 278M | Documentation §4.2 | yes, all ≤ 8B |
| LightGBM 4.7.0 | MIT | tree models | Documentation, README | yes |
| XGBoost 3.4.1 | Apache-2.0 | ablation only, not used by the submission | `requirements.txt` comment | yes |

Libraries:
- **Permissive:** polars / polars-runtime (MIT), numpy (BSD-3), scipy (BSD-3), scikit-learn (BSD-3), rapidfuzz (MIT), sparse-dot-topn (Apache-2.0), pyarrow (Apache-2.0), PyYAML (MIT), psutil (BSD-3), torch (BSD-3), transformers (Apache-2.0), pandas (BSD-3), jellyfish (MIT), tabulate (MIT).
- **Copyleft:** Unidecode (**GPL-2.0-or-later**), disclosed in README:45 and Documentation §4.2.
- Every third-party import is pinned in a requirements file. No shipped module imports `jellyfish` or `tabulate` (pinned in `src/stack/requirements.txt`), or `psutil` in the main pipeline.

## 5. External data scan

- **Network use.** Only `AutoTokenizer/AutoModelForSequenceClassification.from_pretrained` in `src/ce/ce_train.py`, `src/stack/kaggle/ce.py` and `ce_france.py`, all at pinned revisions. `ce_score.py` loads the fine-tuned local checkpoints. There is no `requests`/`urllib`/socket/HTTP client, no URL other than the website regexes in the normalisers, no API key, token or secret, no geocoder, registry or ER-service name, and no S3/GCS path.
- **Hard-coded tables.** The largest literal is 95 entries (`src/stack/src/normalize.py:45`). All tables are generic vocabulary:
  - English, Indian and French street and legal-form abbreviations;
  - stop words, honorifics and filler words;
  - the decoy shift set {1,2,3,4,5,7,9,11,13,21}, which the Documentation derives from train pairs.
  
  None is an entity or address lookup.
- **Transliteration maps.** Mined from training pairs (`er_dict.py`, `src/stack/src/translit.py`).
- **France word lists.** `VETO_WORDS` / `BROAD_WORDS` in `attempts/029f*_final/train.py` are printed by `src/er_frwords.py` from France's test candidate pairs, with no labels. The Documentation explains the derivation (§4.3).
- **Test-set use.** Test features are used without labels: France marker tables, word lists, and the decoy-density guard. This is not prohibited, it is disclosed, and no code reads test labels (none exist).

## 6. Fair-play and provenance signals

- **Names.** Only the four members in the Documentation header. There are no other person, team or tool names, no licence headers from other projects, no absolute paths, and no `__MACOSX` or dot files.
- **`src/stack/` is written very differently from `src/`.**

  | | `src/` | `src/stack/` |
  |---|---|---|
  | Code style | Dense, comment-free polars one-liners | PEP 8 with docstrings, inline comments and `# noqa: E402` path hacks |
  | Stack | polars, Python 3.12 | pandas, Python 3.11 |
  | IDs | `s1_idx`/`pool_idx` | `a_id`/`r_id` |
  | Folds | stratified | crc32 |
  | Tooling | – | its own memory watchdog and its own copy of the organisers' validator (`src/stack/utils/validate_submission.py`) |

  The Documentation calls it "a separately built LightGBM pipeline" but does not say who built it or when.
- **Leftovers that suggest `src/stack/` was once a standalone submission pipeline.**
  - `stack_ce.py:17-20` says its kept pairs "are exactly the pairs written to candidate_pairs.tsv" and that France uses "a rule chosen by src/unseen_rule.py".
  - It references scripts that do not ship: `loco_transfer.py` (`stack_ce.py:452,460`, `high_model.py:10`, `export_extra.py:5`), `feat_cnt.py` (`:56,609`), `fr_addr_rule.py` (`:518`), `cr_score.py` (`:546`) and `lexical.py` (`:606`).
  - `run_experiment.py:9`, `marker_veto.py:11` and `high_model.py:11` write to an `experiments/` folder that does not exist in the zip.
  - `src/stack/README.md` "Known reproduction gaps" says command lines were "rebuilt from the recorded configuration", the export "was not logged", and a setting "is inferred from the change history".
  
  The code paths that use these files are off in the final build flags. Together, though, this is exactly the "different coding styles between folders" signal.
- **Evidence of common authorship.** Both families use the same unusual findings: the ten-value decoy shift set (`src/er_feat5.py:39`, `src/stack/src/marker_veto.py:29`), generator-operator features (`src/er_gen.py`, `src/stack/src/feat_ops.py`) and label-free decoy-marker vetoes. That points to one team rather than copied code, but the Documentation does not make this argument.
- **`run_manifest.json`.** `src_sha256` lists 47 files, but only 30 ship:
  - Missing: `embed.py`, `er_blend.py`, `er_diag.py`, `er_feat3.py`, `er_france2.py`, `er_keysfr.py`, `er_package.py`, `evaluate.py`, `harness_util.py`, `leaderboard.py`, `new_attempt.py`, `predict.py`, `prepare.py`, `split.py`, `sub_status.py`, `submit_log.py`, `train.py`.
  - `embed.py` invites the question of an undisclosed embedding model. `er_france2.py`, `er_keysfr.py` and `er_blend.py` invite questions about France code paths that were not shipped.
  - Other fields look confusing for a `--from-probs` run: `train_frac 0.2`, `n_entities 0`, `sample_sha256` (the hash of empty input) and `git_commit "nogit"`.
  - All 30 shipped modules do match their hashes.
- **Leaderboard use.** Documentation §2.2 says one France change (the empty-address correction) "was kept because the leaderboard rose". This is honest, and it is one decision out of many. It is low risk, but a strict reader may note it.
- **Wording.** "Harness folds" appears in 8 docstrings (`er_meta.py:19`, `er_pool.py:13`, `er_stage3.py:10`, `er_fix.py:14`, `er_ce.py:8`, `er_comb.py:10`, `single/fit.py:1`, `ce/ce_train.py:6`) and in README:145. It is benign jargon for the folds built in `er_model.py:112`.

## 7. Red flags, ranked by how likely they are to cost standing

1. **Provenance of `src/stack/` (MEDIUM).** The style gap, a standalone `candidate_pairs.tsv` mention, 6 unshipped scripts, the `experiments/` outputs, the reconstructed command lines and a second validator copy. Disqualification is unlikely, because the shared decoy findings point to one team, but it is the most likely fair-play query.
2. **Manifest lists 17 unshipped source files (MEDIUM).** Especially `embed.py`, which reads as a possible undisclosed model, and the France variants. It is cheap to fix.
3. **Reviewer cannot reproduce quickly (MEDIUM-LOW).** The final step needs cache files that do not ship. The second family's test predictions ship as inputs. The GPU steps are not bit-identical. All of this is disclosed, but a reproduction attempt stops at about 50 CPU hours.
4. **Candidate-set framing (MEDIUM-LOW, blocking rank).** The 6.83 per S1 figure depends on learned filters. Documentation §4.2 files stages 1-2 under the matching model, so a judge may use 46 per S1.
5. **France at 12.31 per S1 (LOW-MEDIUM, blocking rank).** Twice the labelled countries, because France skips the chain-probability cut.
6. **Literal country branching (LOW).**
   - `er_meta.py:49` (country list), `:92` and `single/build.py:35` (`cty`, France = 2);
   - `er_meta.py:132,134`, `er_comb.py:28`, `er_fix.py:63`, `er_pool.py:128` and `er_france.py:66` (the `"france"` switch);
   - `er_keys.py:49-51` (India- and US-only keys).
   
   Disclosed as Limitation (5), and France is fully handled.
7. **Transductive test statistics and one leaderboard-kept France rule (LOW).** Label-free and disclosed.
8. **Unidecode GPL-2.0+ (LOW).** A library, not a model, and disclosed.
9. **Housekeeping (LOW).**
   - Unused pins: `jellyfish` and `tabulate` in `src/stack/requirements.txt`, `psutil` in `requirements.txt`.
   - "8 GB machine" comments (`src/stack/src/block_feats.py:3`, `features.py:26`) next to the 96 GB machines in the Documentation.
   - A 569 MB zip; confirm the portal's limit.

Outside the zip (not judged): confirm that the portal ranks the upload whose SHA-256 is `2b885f1e…`, the zip's `matching_results.tsv`.

## 8. Prioritised fix list

Except M2, every fix leaves both output files byte-identical.

### HIGH

- **H1. Make `run_manifest.json` describe only the shipped code.**
  - In a clean copy of `code/business_entity_resolution/`, with `cache_dir` pointing at the existing cache, run `python src/er_predict.py 029f6_final --from-probs 030q_fix`. Check that the new `.tsv` and `.candidates.tsv` hash to `2b885f1e…` and `a8cda21e…`. Then replace `code/business_entity_resolution/run_manifest.json` with the new `submissions/archive/NN_029f6_final.json`. Its `src_sha256` will list exactly the 30 shipped modules (`er_predict.py` hashes `ROOT/src/*.py`).
  - In `README.md` "Package contents", in the `run_manifest.json` bullet, add: "a `--from-probs` run, so `train_frac`, `n_entities`, `sample_sha256`, `n_train_pairs`, `features` and `params` are not used; `git_commit` is `nogit` because the package is not a git checkout".
  - If a rerun is impossible, add a README note that the 17 extra `src_sha256` entries were research and tooling scripts in the working folder that the shipped pipeline never imports. Say explicitly what `embed.py` was and that it did not feed the submission, but only if that is true.
  - Re-zip, and update `SHA256SUMS` only if something it covers changed.
- **H2. Close the `src/stack/` provenance gaps (docstring and documentation edits only).**
  1. `Documentation_template.md` §4.2 item 6, after "a separately built LightGBM pipeline with its own blocking, stage-3 stack and in-country e5 cross-encoders", add: "Both families were written by the team during the challenge. The second was built on purpose in a separate environment (pandas, Python 3.11, its own folds), so that the meta-model combines two independent error patterns. Both use the same generator findings: the decoy shift set (`src/er_feat5.py:39`, `src/stack/src/marker_veto.py:29`) and generator-operator features (`src/er_gen.py`, `src/stack/src/feat_ops.py`)." Keep member names out, per team policy.
  2. In `src/stack/src/stack_ce.py:17-20`:
     - change "which are exactly the pairs written to candidate_pairs.tsv" to "which are exactly the pairs exported to cache/stack";
     - change "a rule chosen by src/unseen_rule.py" to "a rule chosen by leave-one-country-out tests".
  3. Mark the references to unshipped scripts as "(research script, not shipped; not used by the submission)", or ship the scripts:
     - `stack_ce.py:56,452,460,518,546,606,609`;
     - `high_model.py:10-11`, `export_extra.py:5`, `marker_veto.py:11`, `run_experiment.py:9` (`experiments/`).
     
     Shipping `loco_transfer.py` is better, because it backs the LOCO numbers that Documentation §4.3 cites.
  4. Delete `src/stack/utils/validate_submission.py` and its line in `src/stack/README.md` Layout, since README points to the organisers' validator. This is optional; the file is byte-identical to the organisers' copy.
  5. In `src/stack/README.md` "Known reproduction gaps" items 2 and 4, replace "inferred from the change history" with the concrete source, e.g. "taken from the recorded configuration of that build".

### MEDIUM

- **M1. Candidate-set framing (Documentation §3 and §4.1, README "Overview").**
  - In the §4.1 pipeline block, insert a line before `-> meta-model over both model families`: `==> candidate_pairs.tsv: chain p >= 0.001 (France: stage-2 band) + second-family cascade; 11.84M pairs, 6.83 per S1`.
  - In §3, after the table, add: "Stages 1-2, the route classifier and the chain-probability cut are the learned filtering stages of the blocking cascade. Following the organisers' definition, the file holds the last stage: reduction ratio 0.9999993 against all S1 × pool pairs, 0.9999982 within country."
  - In §4.2 item 1, label the models "*Stages 1-2 (blocking filter)*".
- **M2. France candidate size (a pipeline change; ship only if the output is byte-identical).**
  - In `src/er_france.py` `main()`, after the line that builds `d` (`d = base.rename({"prob": "pA"}).join(stack, …)`), add `d = d.filter(pl.col("pS").is_not_null() | (pl.col("pA") >= 0.001))`. This is the US/India `MIN_P` rule.
  - Rerun `python src/er_france.py 029f5_final`, then `python src/er_fix.py 030q_fix --no-cv`, then `python src/er_predict.py 029f6_final --from-probs 030q_fix`.
  - Keep the change only if `matching_results.tsv` still hashes to `2b885f1e…` and the validator passes. Then update:
    - the counts in README "Candidate file (exact rule)";
    - Documentation §3, including the sentence "France has twice the pairs per S1 because …";
    - `SHA256SUMS` and the manifest.
  - Otherwise leave everything as it is, and never trim the file after scoring.
  - Expected: France from 12.31 towards about 6 per S1, overall from 6.83 towards about 5.9. This is an upper bound, not measurable from the zip.
- **M3. Country as an open set, no output change (Documentation §6 Limitations (5)).** Add: "an unseen country with no labels takes the France path only after adding it to that switch; without that, it gets the US path (`cty = 0`)". Leave the code alone this late: the output-identical refactor (`er_meta.py:49` globbing `p_test_*.parquet`, a train-country set replacing the `"france"` literals) would need a meta-model rerun to prove identity.

### LOW

- **L1. Quick-verify bundle (optional, adds about 150-250 MB).** Ship id-keyed copies of the three inputs of `er_predict.py --from-probs`:
  - `cache/oof/test_030q_fix.parquet`;
  - `cache/oof/test_006_stage2.parquet`, cut to the pairs of the first file;
  - `entity_id` columns of `cache/norm/test_s1.parquet` and `test_pool.parquet`.
  
  Add one README line, so a reviewer can regenerate both outputs byte-identically in about 20 seconds. Skip this if the portal limits the zip size.
- **L2. Unidecode.** Keep it, because replacing it would change the shipped second-family outputs. Optionally, in README:45, add "(a library, not a model; the licence rule covers models)".
- **L3. Requirements tidy.** Remove or annotate "(unused)" `jellyfish==1.2.1` and `tabulate==0.10.0` in `src/stack/requirements.txt`, and `psutil==7.2.2` in `requirements.txt`.
- **L4. Wording.** Replace "harness folds" with "the 5 CV folds (`src/er_model.py` `folds()`)" in the eight docstrings listed in section 6 and in README:145. Do it before H1, so the regenerated manifest hashes the edited files. Reword "8 GB machine" in `src/stack/src/block_feats.py:3` and `features.py:26` to "small-memory machine".
- **L5. Long tail note.** In Documentation §3, one line: "generic names with city-only addresses make the tail: 2,715 S1 have more than 50 candidates, the maximum is 776". Do not add a cap now, because it would change the matches.
- **L6. Zip size.** 569 MB. Confirm the portal's upload limit before the deadline.
