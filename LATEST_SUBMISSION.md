# Latest submission and current best approach (handoff)

Only the team leader uploads. This file is updated after every scored upload, so check it before starting new work. The same content is shared as the team handoff document.

As of 26 Sep 2026, 01:40 IST. Deadline: 27 Sep 23:59 IST. Uploads left: 10 of 15 overall (5 per IST day).
Numbers are labelled **measured** (computed on data), **derived** (arithmetic on measured numbers) or **guessed**.

## 1. Where we are

| Submission | What changed | 5-fold CV macro F0.5 (all 2.2M train S1) | Public LB |
|---|---|---|---|
| 1 `001_baseline` | blocking + LightGBM | 0.94292 | 0.929 |
| 2 `002_namefreq` | name-frequency + conservative-name features | 0.96050 | 0.949 |
| 4 `006_stage2` | native-script dictionary, 0.6 floor, stage-2 competition features | 0.97423 | 0.963 |
| 5 `008_stage3` | + stage-3 correction with 34 label-free pair features | 0.98254 (India 0.98029, US 0.98403) | 0.974 |
| **6 `009_stage3m`** | **+ 5 label-free decoy-marker name features (`er_feat6.py`)** | **0.98498** (India 0.98361, US 0.98589) | **0.981** |

- The CV→LB gap shrank from −0.011 (006) to −0.0085 (008) to −0.004 (009). The new features reject the decoys that test has more of (measured).
- **Implied France** (LB minus US/India at CV weights 0.38/0.47): 0.902 (006) → 0.929 (008) → **0.960** (009) (derived). France is 15% of test and has no labels. The decoy-marker table France builds from its own test pairs rejects French decoys (word swaps like Club↔École, inserted Groupe/Holding/Développement/Participations, SARL→SCI swaps, shifted house numbers).
- 008 details (measured; 009 adds +0.00244 CV, SE 0.00002, 5/5):
  - pair precision 0.9966, recall 0.9548; blocking oracle 0.9940;
  - F0.5 by true-match count: 0 → 0.985, 1 → 0.944 (weakest), 2 → 0.980, 3+ → 0.985–0.987.

### Latest submission record

| | |
|---|---|
| Submission | #6, uploaded 26 Sep 2026, ~01:30 IST |
| Attempt | `attempts/009_stage3m` (parent `008_stage3`); code commit `f192703` |
| CV | 0.98498; paired vs 008: +0.00244, SE 0.00002, 5/5 folds (vs 006: +0.01075) |
| Public LB | **0.981** |
| matching_results.tsv | sha256 `7d974a6d68d4cd06f3391d0b2387a5fd383da56f39e32b40d7ef42603c417783` (md5 `4474c1898f1bc1c375e7ea7c21f6cdd2`); 1,732,544 rows, 5,739,657 matches, 5.9% empty |
| Validator | organisers' `validate_submission.py --check-ids`: PASS (matching + candidates) |

| # | Uploaded (IST) | Attempt | CV | Public LB |
|---|---|---|---|---|
| 1 | 25 Sep 05:54 | 001_baseline | 0.94292 | 0.929 |
| 2 | 25 Sep 07:08 | 002_namefreq | 0.96050 | 0.949 |
| 3 | 25 Sep ~11:00 | 004_coherence_full | - | not scored (portal failure) |
| 4 | 25 Sep 15:22 | 006_stage2 | 0.97423 | 0.963 |
| 5 | 25 Sep 23:00 | 008_stage3 | 0.98254 | 0.974 |
| **6** | **26 Sep 01:30** | **009_stage3m** | **0.98498** | **0.981** |

## 2. What the data generator does (the key insight)

Four independent analyses converged on this. Reports are in `docs/research/r1.md`, `r7.md`, `r8.md` and `r9.md` in the repo.

**Decoys (unmatched pool records) are near-copies of real entities.**
- The house number is shifted by d in {+1, +2, +3, +4, +5, +7, +9, +11, +13, +21}.
- A name word is inserted from a fixed per-country list:
  - US: group, holdings, associates, ventures, north/east/south/west, downtown, midtown, lakeside, riverside, eastgate…
  - India: industries, ventures, public, group, exports, overseas, infratech, holdings…, and native-script shop words;
  - France: participations, holding, distribution, associés, international, développement, groupe, "et fils".
- Or the legal suffix is swapped: llc↔inc, private→public, limited→llp.
- Or a floor/unit/sub-number is changed while the house number is kept.

**True-match noise is different.**
- Digits are truncated (1995 vs 995, 15050 vs 1505).
- Tokens get typos inside them.
- Filler words are added (center, services, the, inc).
- Alias markers appear (aka, fka, dba, née, "ID: nnn").
- Addresses drop numbers, add state names or codes, or become "None".

**Test is decoy-denser.** Test US has **1.94x** train's pairs at every shift d (India ~1.7x, France the same spikes), while background deltas are unchanged (measured, label-free).

**TRAP: never use sibling or support features.** On test, decoy entities come in clusters that share the same perturbed house number (9.6x train in the low-probability band); on train they are singletons.
- So any feature that looks at an S1's *other* candidates learns "shared deviation = true" on train and emits decoys on test. This covers coherence, house agreement, source balance and S1-context stackers.
- Every feature must compare one pool record with its S1 only.
- This failure has been measured three times: the coherence feature, a context stacker, and cloned-decoy simulations.

## 3. The 008 pipeline (all code in `src/`; settings in `attempts/<id>/train.py`)

1. **Normalization** (`er_dict.py`, `er_norm.py`):
   - a native-script→Latin token dictionary learned from train gold (1,347 tokens → 147 words);
   - multi-view names and addresses;
   - French street and particle rules for rows labelled France.
2. **Blocking per country** (`er_block.py`): word TF-IDF top-40 plus 6 exact keys. About 45 candidates per S1 (98M train pairs, 79M test).
3. **Stage 1** (`er_feat.py`, `er_feat2.py`): LightGBM (127 leaves, lr 0.05, 600 rounds) on ~48 pair features.
4. **Stage 2** (`er_feat4.py`): stage-1 features plus competition features from out-of-fold stage-1 probabilities (rank, margin, best other, sums).
5. **Stage 3 (new in 008)** (`er_feat5.py` + `er_stage3.py`):
   - A LightGBM correction boosted from **init_score = logit(p_stage2)**, with inputs logit(p) + 34 features. Params: 31 leaves, lr 0.05, 300 rounds, min_data_in_leaf 200, feature and bagging fraction 0.8, lambda_l2 1, seed 42.
   - Only pairs with p_stage2 ≥ 0.001 are re-scored: 12.9M of 98M train pairs, holding 7.49M of 7.49M positives. Other pairs keep p_stage2.
   - The 34 features, all label-free, computed per split and per country from that split's own S1, and each comparing one pool record with one S1:
     - **name specificity (10):** IDF-weighted shared / S1-only / pool-only name-token mass, their fractions, max one-sided weight, weighted cosine, one-sided counts;
     - **address specificity (6):** the same over address tokens;
     - **house relations (11):** suffix (dropped leading digits), log |diff|, diff ≤ 10 (and on the same street), edit distance, same digits reordered, each side's house among the other side's numbers, unit/suite equal / conflict / one-sided;
     - **legal form (4):** equal, nested, disjoint, one-sided;
     - **char-3-gram name cosine (1)**;
     - **signed house shift (2):** pool − S1, and whether it is in the decoy shift set.
   - **009 adds 5 name-operator features** (`er_feat6.py`): words added / dropped by the pool name, and label-free decoy-marker scores of the added, dropped and swapped words. Score = (n_d + 1)/(n_d + n_0 + 10), where n_d counts candidate pairs in which the word is added and the pool house number is the S1's shifted by d ∈ {1,2,3,4,5,7,9,11,13,21}, and n_0 counts pairs with equal house numbers. Tables come from train pairs; France builds its own table from its test pairs.
6. **Decision** (`er_model.decide`):
   - each pool record goes to its highest-probability S1;
   - per S1, keep the prefix k that maximises 5·Σp₁..ₖ/(Σp + 4k), unless P(no match) = Π(1−p) is larger;
   - then drop pairs with p ≤ 0.6.
   - This rule measured as already Bayes-optimal for calibrated probabilities; no decision tweak helps.

**Reproduce 008:** the README inside `05_008_stage3_code.zip` lists every command in order. The last four steps:
`er_feat5.py --split train|test` → `er_stage3.py 008_stage3` → `er_eval.py 008_stage3 --from-oof 008_stage3` (CV) → `er_predict.py 008_stage3 --from-probs 008_stage3` (writes matching + candidates TSVs).

**Measured time and memory on one 8-core / 64 GB machine:**
- feat5: about 10 min per split with 4 country jobs in parallel, ≤ 14 GB each;
- stage 3 (5-fold CV + full fit + test): about 10–15 min uncontended, peak ~20 GB;
- er_eval / er_predict: about 10 min each.
- Stage-2 CV alone peaks at 61.7 of 64 GB, so do not add feature columns to stage 2 without a join-free loader.

## 4. Validation rules we follow (do not skip)

- **Paired rule:** same folds (`data/folds.parquet`, 5 folds stratified by country × match count); a change counts only if its mean per-fold gain is > 2 SE and positive in ≥ 4 of 5 folds.
- **Label-free test guard** before any upload:
  - Compare the house-number-conflict share of emitted test pairs, the empty-answer rate and pairs per S1 by country against the previous submission.
  - 008: US 8.4% / India 17.0% / France 1.0% (006: 7.8 / 16.8 / 1.8).
  - A large rise, or added pairs per S1 on test far above the train rate, means decoys are being emitted.
- **Fast iteration** uses a shared 10% sample (220,678 S1, 9.8M pairs) with an exact copy of the decision rule (0 mismatches vs full CV): `~/aml/rk/rk.py` on the build machines. Iterate on its 2% subset, confirm on the 10% sample, then run full data.
- Research numbers and scripts for every lane are in `docs/research/r1.md`–`r9.md`.

## 5. Next steps, ranked by measured evidence

| # | Step | Evidence | Expected gain |
|---|---|---|---|
| 1 | **(done in 009) Decoy name-marker features + signed house delta.** Label-free marker score per country: (n_d + 1)/(n_d + n_0 + 10), counting pairs where a token is added and the house shift is in D vs where the houses are equal. France gets its own table from test pairs. Features: hd, hd_D, nA, nB, dm_add, dm_drop, dm_swap. | Measured on the 10% sample, same stage-3 protocol: +0.00292 on top of the 008 feature set (SE 0.00009, 5/5). Total +0.01014 vs 006. Gain grows to +0.0125 under doubled decoys. | CV about 0.985; LB about +0.003 (derived) |
| 2 | **France / unseen-country robustness.** Leave-one-country-out loses 0.05–0.07 (train US → India 0.896 vs 0.965 in-country). | Measured, quick subset. Recipes not yet measured: generator features, country-size-normalised name counts (US test has half the S1, so raw counts shift), self-training on the target country. | France +0.01 to +0.03 → LB +0.002 to +0.005 (guessed) |
| 3 | **Candidate-set reduction.** New organiser rule: smaller candidate_pairs.tsv per S1 ranks higher beyond the LB. Treat stage 1 as a learned filter (keep pairs with stage-1 p ≥ ~0.001, about 6 per S1 instead of 45) and run the final model only on those. | 14% of pairs keep 99.991% of positives (measured, sample); the F0.5 cost table is not measured yet. | Ranking criterion, not LB |
| 4 | **Cheap retrieval:** reverse top-1 (each pool record's best S1), +0.14 pairs per S1; key "first 6 letters of name + house". | Oracle, measured: catches 24–28% of never-retrieved gold pairs. | LB +0.0015 (derived); conflicts with #3, so keep it small |
| 5 | **Label-derived "lexical corruption" residual + char-3-gram retrieval (~56 extra pairs per S1).** Tried as a cohort experiment in the other team line. | Claimed 0.985372 vs 0.974773 (006) on the 17,241-S1 cohort s1_idx%128==6. Our 008 scores 0.982474 on the same cohort. An independent audit is running (`docs/research/audit_0985.md`). | Overlaps with #1; retrieval part +0.0013 on the cohort |

Full-data integration (putting the stage-3 features into stages 1 and 2 instead of a correction) is untested at full scale. It needs the join-free loader first (draft exists).

## 6. Tried and rejected (do not retry)

- Coherence, house-agreement, source-balance and context stackers: large CV gains, but they emit decoys on test (Section 2).
- Decision-rule changes: exact expected F0.5, calibration, ratio rules, joint assignment. All within ±0.0001.
- Mapping the 171 residual native-script shop words: they appear only on decoys (0 of 64,020 train pool records match), so mapping would remove a perfect decoy signal.
- More training data or capacity: 5x more training entities adds only +0.0007.
- Hard house-shift veto rule: helps US, hurts India, where the parsed house is often a template number.

## 7. Infrastructure

- **Code:** private repo `amazon-ml-2026` (this file is `LATEST_SUBMISSION.md` at its root). A self-contained copy of the 008 code is also in the code zip `05_008_stage3_code.zip`. The dataset is not in git: download the official `student_resource` zip from the portal and unzip it to `data/student_resource/dataset/{train,test}/`.
- **Build machines (AWS; access through the repo owner):**
  - A: us-east-1 r7a.2xlarge, 64 GB. Holds all caches under `~/aml/cache` and the research kit `~/aml/rk`.
  - C: us-west-2 r7a.2xlarge, 64 GB.
  - D: us-east-2 r7a.xlarge, 32 GB.
  - All auto-shut down 27 Sep 18:30 UTC. Spot capacity in us-east-1 was reclaimed repeatedly; use on-demand.
- Run heavy jobs through `~/aml/rk/run.sh <GB> <threads> <cmd>`, a memory-capped queue. One heavy job per machine. Under CPU contention, use 1 LightGBM thread per job.

## 8. Final package checklist (organiser rules)

- The zip contains:
  - `output/matching_results.tsv` and `output/candidate_pairs.tsv`, where candidate_pairs.tsv is exactly the set the final model scored and every match is inside it;
  - `code/business_entity_resolution/` with src, README and requirements;
  - the filled-in `Documentation_template.md`.
- Build it with `python src/er_package.py submissions/archive/NN_<attempt>.tsv --final --team "<registered team name>"`. The team name must match the portal registration.
- Validate with the organisers' `utils/validate_submission.py --check-ids`. On an 8 GB laptop it can run out of memory on the 1 GB candidates file; run it on a 64 GB machine.
- Country is an open set: every test S1, including France, must appear. No external data, lookups or APIs.
