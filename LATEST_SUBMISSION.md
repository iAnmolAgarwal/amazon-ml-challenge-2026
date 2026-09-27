# Latest submission and current best approach (handoff)

Only the team leader uploads. This file is updated after every scored upload, so check it before starting new work. The same content is shared as the team handoff document.

As of 27 Sep 2026, ~10:05 IST. Deadline: 27 Sep 23:59 IST. Uploads left today: 4 (after upload 12).
Numbers are labelled **measured** (computed on data), **derived** (arithmetic on measured numbers) or **guessed**.

## 1. Where we are

| Submission | What changed | 5-fold CV macro F0.5 (all 2.2M train S1) | Public LB |
|---|---|---|---|
| 1 `001_baseline` | blocking + LightGBM | 0.94292 | 0.929 |
| 2 `002_namefreq` | name-frequency + conservative-name features | 0.96050 | 0.949 |
| 4 `006_stage2` | native-script dictionary, 0.6 floor, stage-2 competition features | 0.97423 | 0.963 |
| 5 `008_stage3` | + stage-3 correction with 34 label-free pair features | 0.98254 (India 0.98029, US 0.98403) | 0.974 |
| 6 `009_stage3m` | + 5 label-free decoy-marker name features (`er_feat6.py`) | 0.98498 (India 0.98361, US 0.98589) | 0.981 |
| 10 `018_lexical` | band stage 2 with every feature set (016) + country-lifted markers and French house parser (`er_lift.py`) + char-3-gram retrieval route (`er_retrieve.py`, `er_newpairs.py`) + lexical correction (`er_lexical.py`) | 0.98618 (India 0.98502, US 0.98695)* | 0.98393 |
| 11 `020_gen` | + generator-operator residual for US/India: number operation codes, name edit operations, confusable characters, legal-form ids, reorder/join, fillers, honorifics, alias and web markers, address-word differences | 0.98700 (India 0.98601, US 0.98766)* | 0.985 |
| **12 `029b_final`** | **rebuilt chain (GCP) + final correction (023: generator-operator + pool-side features) + e5-small cross-encoder band stack (028) + cross-fitted meta-model over the union with the in-country stack model family (027d, `src/er_meta.py`, `src/stack/`) + France logit blend with the stack family (`src/er_france.py`)** | **0.99046** (India 0.99009, US 0.99071) | **0.989** |

*Full-graph CV with retrieval-route pairs on the 10% sample owners only; the route adds +0.00126 on those owners, so full-route CV is about 0.9873 (derived).

- The CV→LB gap shrank from −0.011 (006) to −0.0085 (008) to −0.004 (009) to −0.0023 (018). The new features reject the decoys that test has more of (measured).
- **Implied France** (LB minus US/India at CV weights 0.38/0.47): 0.902 (006) → 0.929 (008) → 0.960 (009) → **0.967-0.973** (018, from the exact LB 0.98393; the range covers the route correction) (derived). France is 15% of test and has no labels. The decoy-marker table France builds from its own test pairs rejects French decoys (word swaps like Club↔École, inserted Groupe/Holding/Développement/Participations, SARL→SCI swaps, shifted house numbers).
- 008 details (measured; 009 adds +0.00244 CV, SE 0.00002, 5/5):
  - pair precision 0.9966, recall 0.9548; blocking oracle 0.9940;
  - F0.5 by true-match count: 0 → 0.985, 1 → 0.944 (weakest), 2 → 0.980, 3+ → 0.985–0.987.

### Latest submission record

| | |
|---|---|
| Submission | #12 (file 06_029b_final), uploaded 27 Sep 2026, ~10:00 IST; public LB **0.989**. Chain: 016 band → 017 route → 018 lexical → 023_comb (CV 0.98742, +0.00138 vs 018, 5/5) → 028_ce cross-encoder stack (0.98779, +0.00037, 5/5) → 027d_meta meta-model (0.99046, +0.00267 vs 028, SE 0.00002, 5/5; re-verified independently; the train candidates lack the route pairs test has, so the expected LB gain was ~+0.001-0.0018) → 029b_final France blend (r17 rule, guard clean). Validator PASS; byte-identical rerun from the zip code; guard vs #11 PASS. matching sha256 `22510911e746bb01dbac0f6b2c53e1cb9af0064eb9b68db7ab703cbc722d78a4`, md5 `eb8d8813b2ab0eb0b7907aafc2e4b4a1`. Final package `submissions/final_029b/Apprentice_submission.zip` (md5 `089d3b12ea45c73337778842ac696fdb`). |
| Submission #11 | (file 12_020_gen), uploaded 26 Sep 2026, ~11:15 IST; public LB **0.985**. Attempt `attempts/020_gen` = 018 + a residual LightGBM (127 leaves, 400 rounds, lr 0.05, min data 300, L2 10, boosted from logit p018) on US/India pairs with 0.003 <= p018 <= 0.997, using pair-local generator-operator features; full CV 0.98700, paired vs 018 +0.00082 (SE 0.00001, 5/5; US +0.00071, India +0.00099), re-verified independently; test guard clean (France unchanged); validator PASS; matching sha256 `5b44d976052bdfdd9fc88419e685dd4926caae47a43c3d7f9bde7e20fb90c4ca`, md5 `d32fe6d63f639e26e1098296a9a7ab2a`. Previous record (#10) below. |
| Submission #10 | uploaded 26 Sep 2026, ~09:55 IST (git tag `sub-10`) |
| Attempt | `attempts/018_lexical` (chain 016_stage2b → 017_route → 018_lexical); code commits `caa614e`, `0c0760a` |
| CV | 0.98618 full graph; paired vs 012 (009 + route + lexical): +0.00077, SE 0.00002, 5/5 folds (US +0.0007, India +0.0008); 016 vs 009 +0.00092 (5/5); route over 016 +0.00126 (sample, 5/5); lexical over 017 +0.00016 (full, nested, 5/5) |
| Public LB | **0.98393** (+0.003 vs 009) |
| matching_results.tsv | sha256 `efdafee034c499c80d2cbebd5c4db7ee08a15719fc0cf95c5749e026c27d2183` (md5 `04a02dcececc401e4ff2a93302f3b21b`); 1,732,544 rows, 5,769,711 matches (3.33 per S1), 5.8% empty |
| Validator | organisers' `validate_submission.py --check-ids`: PASS (matching + candidates) |
| Final-package form | `019_band`: same matches (byte-identical file), candidate set = the final models' band, 7.25 per S1 instead of 46.5; CV equal (0.986181); validator PASS; `submissions/archive/11_019_band.*` |

| # | Uploaded (IST) | Attempt | CV | Public LB |
|---|---|---|---|---|
| 1 | 25 Sep 05:54 | 001_baseline | 0.94292 | 0.929 |
| 2 | 25 Sep 07:08 | 002_namefreq | 0.96050 | 0.949 |
| 3 | 25 Sep ~11:00 | 004_coherence_full | - | not scored (portal failure) |
| 4 | 25 Sep 15:22 | 006_stage2 | 0.97423 | 0.963 |
| 5 | 25 Sep 23:00 | 008_stage3 | 0.98254 | 0.974 |
| 6 | 26 Sep 01:30 | 009_stage3m | 0.98498 | 0.981 |
| 7 | 26 Sep ~09:55 | 018_lexical (file 10) | 0.98618 | 0.98393 |
| **8** | **26 Sep ~11:15** | **020_gen** (file 12) | **0.98700** | **0.985** |

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

## 3. The 018 pipeline (all code in `src/`; settings in `attempts/<id>/train.py`)

1. **Normalization** (`er_dict.py`, `er_norm.py`):
   - a native-script→Latin token dictionary learned from train gold (1,347 tokens → 147 words);
   - multi-view names and addresses;
   - French street and particle rules for rows labelled France.
2. **Blocking per country** (`er_block.py`): word TF-IDF top-40 plus 6 exact keys. About 45 candidates per S1 (98M train pairs, 79M test).
3. **Stage 1** (`er_feat.py`, `er_feat2.py`): LightGBM (127 leaves, lr 0.05, 600 rounds) on ~48 pair features.
4. **Stage 2** (`er_feat4.py`): stage-1 features plus competition features from out-of-fold stage-1 probabilities (rank, margin, best other, sums).
5. **Pair features** (`er_feat5.py`, 34 label-free features, each comparing one pool record with its S1):
   - **name specificity (10):** IDF-weighted shared / S1-only / pool-only name-token mass, their fractions, max one-sided weight, weighted cosine, one-sided counts;
   - **address specificity (6):** the same over address tokens;
   - **house relations (11):** suffix (dropped leading digits), log |diff|, diff ≤ 10 (and on the same street), edit distance, same digits reordered, each side's house among the other side's numbers, unit/suite equal / conflict / one-sided;
   - **legal form (4):** equal, nested, disjoint, one-sided;
   - **char-3-gram name cosine (1)**; **signed house shift (2):** pool − S1, and whether it is in the decoy shift set.
   - **Name operators** (`er_feat6.py`): words added / dropped by the pool name, and label-free decoy-marker scores of the added, dropped and swapped words: (n_d + 1)/(n_d + n_0 + 10), where n_d counts candidate pairs in which the word is added and the pool house number is the S1's shifted by d ∈ {1,2,3,4,5,7,9,11,13,21}, and n_0 counts pairs with equal house numbers. Tables come from train pairs; France builds its own from its test pairs.
   - **Lift and French parser** (`er_lift.py`): marker scores become logit(score) − log(Σn_d/Σn_0) of their country, so France's test-built table sits on the train scale; France house columns are recomputed with a French fallback house parser (padded or parenthesised numbers, extra street types). In-country neutral (+0.00001); on a leave-one-country-out proxy the lift gains +0.0086 (5/5).
6. **Band stage 2** (`er_stage3.py 016_stage2b`, INIT = False, USE_LOGIT = False): a fresh LightGBM (127 leaves, lr 0.05, 600 rounds) on the pairs with stage-2 p ≥ 0.001 (12.9M train pairs, all positives) with every feature set (feat, feat2, feat4, feat5p, feat6l). Pairs below the band keep their stage-2 probability. +0.00092 over the earlier residual design (5/5).
7. **Retrieval route** (`er_retrieve.py`, `er_newpairs.py`): char-3-gram TF-IDF (min_df 2, sublinear tf) over name core and normalised address, top-40 per view within country; new pairs are scored by a separate 66-feature LightGBM (63 leaves, 300 rounds) trained on existing train pairs of a 10% owner sample; pairs with p > 0.001 are added (0.94 per test S1). +0.00126 (sample, 5/5).
8. **Lexical correction** (`er_lexical.py`): six token channels per pair (name words added / dropped, address words added / dropped, legal-word transition, character edits); cross-fitted likelihood ratios from true and false pairs of other folds, 42 summary features; LightGBM (100 rounds, 9 leaves, lr 0.04) boosted from logit(p) on pairs with stage-2 p ≥ 0.01. +0.00016 (full, nested, 5/5).
9. **Decision** (`er_model.decide`):
   - each pool record goes to its highest-probability S1;
   - per S1, keep the prefix k that maximises 5·Σp₁..ₖ/(Σp + 4k), unless P(no match) = Π(1−p) is larger;
   - then drop pairs with p ≤ 0.6.
   - This rule measured as already Bayes-optimal for calibrated probabilities; no decision tweak helps.
10. **Candidate set** (`019_band`, `er_predict.py` CAND_BAND): the final decision runs on the band (stage-2 p ≥ 0.001 plus route pairs), and that band is written as candidate_pairs.tsv.

**Reproduce 018 / 019:** the README inside `11_019_band_code.zip` (or `10_018_lexical_code.zip`) lists every command in order: stages 1-2 → `er_feat5.py`, `er_feat6.py` → `er_lift.py` → `er_stage3.py 016_stage2b` → retrieval (`er_retrieve.py`, `er_newpairs.py union/fit/score 017_route`) → `er_lexical.py 018_lexical` → `er_predict.py 019_band --from-probs 018_lexical`.

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

Research lanes running now (reports in `docs/research/r10.md`, `r11.md`, `r12.md`). Numbers are the lanes' own paired results vs 018 and still need re-verification on full data plus the test guard before any upload.

| # | Step | Evidence | Expected gain |
|---|---|---|---|
| 1 | **Generator residual features** (r12): 26 features that describe the generator's operators, in a residual model on top of 018. | +0.00035 (SE 0.00001, 5/5; US +0.00029, India +0.00043), measured by the lane; test guard pending. | LB +0.0003 (derived) |
| 2 | **Extra candidate keys** (r10): name_join[:6] + house, name_join[:8], skeleton, first+last token; new pairs scored by the route classifier. | name_join[:6] + house +0.00033 (5/5); others +0.00005 to +0.00013 each (measured by the lane). | LB +0.0003-0.0005 (derived) |
| 3 | **Second competition round, pool side** (r11): rank, best other S1 p, margin and counts recomputed from 018 probabilities, then a stage 3. | +0.00020 (SE 0.00001, 5/5); US/India guard PASS, France flagged. The S1-side variant fails the guard (asymmetric on test). | LB +0.0002 (derived) |
| 4 | **Where the rest is** (r12 oracles, set p = y and re-decide): all in-candidate errors +0.0080; empty-address "exchangeable twins" (S1s sharing the same name) +0.0033 and nearly irreducible without new evidence; everything except those twins 0.99093 (US 0.99217, India 0.98906). r10: never-retrieved gold +0.0043, gold lost to another owner +0.0036. | Measured (oracle). | Ceiling, not a plan |

## 6. Tried and rejected (do not retry)

- Coherence, house-agreement, source-balance and context stackers: large CV gains, but they emit decoys on test (Section 2).
- Decision-rule changes: exact expected F0.5, calibration, ratio rules, joint assignment. All within ±0.0001.
- Mapping the 171 residual native-script shop words: they appear only on decoys (0 of 64,020 train pool records match), so mapping would remove a perfect decoy signal.
- More training data or capacity: 5x more training entities adds only +0.0007.
- Hard house-shift veto rule: helps US, hurts India, where the parsed house is often a template number.
- France mass calibration on 018: France over-claims by only 0.014 per S1 now (0.18 under 006); expected ≤ +0.0002 LB with uncertain sign.
- Lexical correction off for France: rejected; what it removes in France looks like decoys (house conflict 5.5% vs 1.0%), and turning it off costs about −0.0001 to −0.0003 LB.
- Decision variants on 018 (floors, k rules, calibration): nothing to gain (r11).

## 7. Infrastructure

- **Code:** private repo `amazon-ml-2026` (this file is `LATEST_SUBMISSION.md` at its root). A self-contained copy of the current code is in the code zip `11_019_band_code.zip`. The dataset is not in git: download the official `student_resource` zip from the portal and unzip it to `data/student_resource/dataset/{train,test}/`.
- **Build machines (AWS; access through the repo owner):**
  - A: us-east-1 r7a.2xlarge, 64 GB. Holds all caches under `~/aml/cache` and the research kit `~/aml/rk`.
  - C: us-west-2 r7a.2xlarge, 64 GB.
  - D: us-east-2 r7a.xlarge, 32 GB.
  - F: us-east-2 r7a.xlarge, 32 GB (spot).
  - All auto-shut down 27 Sep 18:30 UTC. Spot capacity in us-east-1 was reclaimed repeatedly; use on-demand.
- Run heavy jobs through `~/aml/rk/run.sh <GB> <threads> <cmd>`, a memory-capped queue. One heavy job per machine. Under CPU contention, use 1 LightGBM thread per job.

## 8. Final package checklist (organiser rules)

- The zip contains:
  - `output/matching_results.tsv` and `output/candidate_pairs.tsv`, where candidate_pairs.tsv is exactly the set the final model scored and every match is inside it;
  - `code/business_entity_resolution/` with src, README and requirements;
  - the filled-in `Documentation_template.md`.
- Build it with `python src/er_package.py submissions/archive/11_019_band.tsv --final --team "<registered team name>"` (or the newer best attempt). The team name must match the portal registration. 019's candidates file is 184 MB (12.6M pairs, 7.25 per S1).
- Validate with the organisers' `utils/validate_submission.py --check-ids`; run it on a 64 GB machine.
- Country is an open set: every test S1, including France, must appear. No external data, lookups or APIs.
