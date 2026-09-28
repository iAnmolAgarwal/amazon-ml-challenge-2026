# Latest submission and current best approach (handoff)

Only the team leader uploads. This file is updated after every scored upload, so check it before starting new work. The same content is shared as the team handoff document.

As of 27 Sep 2026, ~18:45 IST. Deadline: 27 Sep 23:59 IST. Upload 21 used (last). Best public LB: 0.990203 (#20, 029f6_final); the final package is built on 029f6. The final package goes in separately if we are in the top 50.
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
| 12 `029b_final` | rebuilt chain (GCP) + final correction (023: generator-operator + pool-side features) + e5-small cross-encoder band stack (028) + cross-fitted meta-model over the union with the in-country stack model family (027d, `src/er_meta.py`, `src/stack/`) + France logit blend with the stack family (`src/er_france.py`) | 0.99046 (India 0.99009, US 0.99071) | 0.988787 |
| 13 `029e_final` | + extra key / reverse-text candidates on the train sample, so the meta-model (027e) learns them; France unchanged | 0.99051 (sample-paired +0.00045 vs 027d, 5/5) | 0.989022 |
| 14 `029h_final` | + our cross-encoder scores (e5-small, e5-base) and pair side-car features (feat5) as meta inputs (027h); test cross-encoder score from one half model per S1 (parity) | 0.99074 (India 0.99055, US 0.99086) | 0.989465 |
| 15 `029l_final` | + cross-encoder scores on the added (route / extra-key) pairs as meta inputs, 127 leaves (027l); empty-address correction (030l); France on the 024 base (extra-key pairs); owner-only candidate file (3.68 per S1) | sample-paired +0.00049 vs 027h (5/5) | 0.989727 |
| 16 `029n_final` | meta 027n: extra-key pairs for every train S1 (024f_keys, 028f_ce) with cross-encoder scores on all of them; fix 030n | sample-paired +0.00014 vs 027l (5/5) | 0.989801 |
| 17 `029f2_final` | + France rename rule: a renamed business at the S1's unique exact address takes max(blend, stack-family p) | LOCO +0.00033 / +0.00062, 5/5 both | 0.989886 |
| 18 `029f4_final` | + France cross-encoder third input (logit weight 0.1) | France-like LOCO +0.00053 / +0.00052 (5/5 both); flat with in-country parents | 0.98999 |
| 19 `029f5_final` | + single flat LightGBM (205 features, one model) as a meta-model input (027p, src/single); fix 030p | paired vs 030n +0.000015 (4/5, > 2 SE) | 0.989992 |
| **20 `029f6_final`** | **+ empty-address / name-twin correction fitted on US + India (no country flag) applied to France (030q_fix)** | **LOCO: US->India +0.000062 (5/5), India->US +0.000018 (2/5)** | **0.990203** |
| 21 `029fmax_final` (endgame bet) | France temperature 3 + floor 0.7 + empty ratio 3 + decoy-shift drops + house-number vetoes on top of 029f6 (28.5k fewer France pairs, 3.18 pairs/S1) | none (label-free profile only; LOCO +0.013 / -0.009) | 0.989057 (-0.0011: France under-recalls, so the aggressive precision moves hurt) |

*Full-graph CV with retrieval-route pairs on the 10% sample owners only; the route adds +0.00126 on those owners, so full-route CV is about 0.9873 (derived).

- The CV→LB gap shrank from −0.011 (006) to −0.0085 (008) to −0.004 (009) to −0.0023 (018). The new features reject the decoys that test has more of (measured).
- **Implied France** (LB minus US/India at CV weights 0.38/0.47): 0.902 (006) → 0.929 (008) → 0.960 (009) → **0.967-0.973** (018, from the exact LB 0.98393; the range covers the route correction) (derived). France is 15% of test and has no labels. The decoy-marker table France builds from its own test pairs rejects French decoys (word swaps like Club↔École, inserted Groupe/Holding/Développement/Participations, SARL→SCI swaps, shifted house numbers).
- 008 details (measured; 009 adds +0.00244 CV, SE 0.00002, 5/5):
  - pair precision 0.9966, recall 0.9548; blocking oracle 0.9940;
  - F0.5 by true-match count: 0 → 0.985, 1 → 0.944 (weakest), 2 → 0.980, 3+ → 0.985–0.987.

### Latest submission record

| | |
|---|---|
| Submission | #17 (file 13_029f2_final), uploaded 27 Sep 2026, ~18:40 IST; public LB **0.989886** (+0.000085 vs #16). 029n plus er_france RENAME: class R = pool name shares no core-name token with the S1, same house number, a shared street word, and the S1 is the only one of its country at that house and street-word set; p = max(p, pS). LOCO (r17 testbed, 10% sample): US held out +0.00033 (SE 0.00004, 5/5), India held out +0.00062 (SE 0.00003, 5/5), re-run independently; in-country neutral. France guard: +2,124 / -46 emitted pairs, pairs per S1 3.330 -> 3.338, added pairs 0% decoy house shift. US/India rows byte-identical to 029n. Validator PASS. matching sha256 `c96989ae5ba27ae09339b428a03b5233a7bba8283d003d08e320adb28243301e`, md5 `66e4ac51bd51ae6f1cdc5011c08052a8`. |
| Submission | #20 (file 18_029f6_final), uploaded 27 Sep 2026, ~21:20 IST; public LB **0.990203** (+0.000211 vs #19; first score above 0.99). 029f5 plus er_fix FRANCE option (030q_fix: fitted on US + India band pairs of 027p without the country flag, applied only to France pairs of 029f5; non-empty pairs can only go down). France: 3.337 -> 3.320 pairs/S1, +671 / -5,139 emitted; dropped pairs are decoy-enriched (house-conflict 9.5%, decoy shift 2.4% vs 1.0% / 0.25% for all emitted). Validator PASS. matching sha256 `2b885f1ef131538175b24f5b8082c61e049128dc3c06c6ba022d76fd17071f9e`, md5 `9261e5c681ddfb1b49ac5cc8822f737f`. |
| Submission #19 | (file 17_029f5_final), uploaded 27 Sep 2026, ~20:40 IST; public LB **0.989992**. 029f4 plus meta 027p = 027n + side-car r19a2 (logit of a single flat LightGBM on the stage-2 band, src/single/{build,fit,sidecar}.py; main-logit fill for pairs outside the band, built from 028f / 028k so train and test coverage match), fix 030p. Paired: 027p vs 027n +0.00002 (5/5), 030p vs 030n +0.000015 (4/5, 2 SE 0.0000105). Guard unchanged; France identical to #18. Validator PASS. matching sha256 `e5f91bd908bb185375b8e8ccfe8d04db35d9e369306ecf95d3e8650643e01ec3`, md5 `3d2ba36d4780708d6eccda2486ac30fb`. |
| Final package | `submissions/final_package/Apprentice_submission.zip` = `~/Desktop/Apprentice_submission.zip` (md5 `6816c7ee528d83bcb3276ff6e663599a`, 122 files, 659 MB; rebuilt 28 Sep ~22:55 IST: two rounds of evaluator audits applied, extras that evaluator A counted against reproducibility removed, rerun-robustness fixes): output/matching_results.tsv = the #20 file (029f6, md5 9261e5c6, LB 0.990203, byte-identical rerun), output/candidate_pairs.tsv = every pair the final models score (11,837,012 pairs, 6.83 per S1), second-family outputs in artifacts/stack, final probabilities in artifacts/final with src/verify_final.py (rebuilds both TSVs byte-identical in 18 s from the unpacked zip), run_manifest.json (shipped files only, commit be563bd), SHA256SUMS over all 121 files, rewritten Documentation. Validator PASS (--check-ids). Round-2 audit (17:07 build): A borderline (repro 6, README 7, pins 7, code 6, structure 7), B 8/10 advances, C minor risks, blocking 8/10. France chain-probability cut tried (6.59 per S1) but it changed 95 matched pairs, so not shipped. |
| Submission #16 | (file 12_029n_final), uploaded 27 Sep 2026, ~18:15 IST; public LB **0.989801** (+0.000074 vs #15). Meta 027n = 027l settings with MAIN 028f_ce (full-coverage extra-key train pairs, 2.68M, scored out-of-fold by the route classifier) and CE on all of them (Kaggle inference, held-out half models); fair-sample paired vs 027l +0.00014 (SE 0.00004, 5/5); fix 030n +0.00007 (5/5); like-for-like guard clean. Validator PASS. matching sha256 `eafcd7e6fc061cf1144531f51352c298e6ab6864688ee3fa27c948ca72c09542`, md5 `2b97855b57aa34c8d021a9b59a1a99ac`. |
| Submission #15 | (file 11_029l_final), uploaded 27 Sep 2026, ~17:30 IST; public LB **0.989727** (+0.000262 vs #14). Meta 027l = 027h settings with 127 leaves / 600 rounds and FEATS ce2 (CE scores filled in for route / extra-key pairs, research/r16 extra scoring; train out-of-fold by half model, test parity); paired on the fair 10% sample vs 027h +0.00044 (5/5); fix 030l on 027l +0.00005 (5/5); chain vs 027h on the sample +0.00049 (SE 0.00007, 5/5). France base 024_keys: +4,497 / -1,633 France pairs, added pairs cleaner than dropped (decoy-shift 0.7% vs 2.0%). Candidates: p > 0.001 plus owner-only, 6,372,176 pairs (3.68 per S1). Validator PASS; guard vs #14 clean. matching sha256 `3e7a12eb536d27cc1ab5a4ac96cda99fefae2f74b3c207db6b48d6610a7ec4d1`, md5 `745d02a2b30c8d882f15e44988ab29d4`. |
| Submission #14 | (file 08_029h_final), uploaded 27 Sep 2026, ~13:10 IST; public LB **0.989465** (+0.000443 vs #13; predicted +0.0001-0.0002). Meta 027h = 027e + FEATS ce (ce, ce_base) and feat5 (test feat5p, route pairs feat5_new) + parity test CE (er_extend.py parity); paired vs 027e +0.00023 (SE 0.00001, 5/5; US +0.00012, India +0.00038), re-verified independently; guard stable (small unexplained US excess of blocking-candidate additions, 1.5x train). Validator PASS. matching sha256 `8e66b88525c217f60945feda6030044e0f74b4454fd95966d861cb75003a9efe`, md5 `1b9a95fe6a67bdb5980e35b398bc5e40`. |
| Submission #13 | (file 07_029e_final), uploaded 27 Sep 2026, ~10:25 IST; public LB **0.989022** (+0.000235 vs #12, inside the predicted +0.0002-0.0004). 029b plus the extra key / reverse-text candidates (024) in the meta-model: train-side check on a label-free 10% S1 sample (er_extend.py sample 10 3), 024 vs 023 +0.00222 (5/5; added precision 95% US / 91% India), meta 027e vs 027d +0.00045 (SE 0.00004, 5/5). Validator PASS; guard vs #12 clean (France identical). matching sha256 `3b3d598732dee1e2b6eae50caeb523f588c34d5ebba403d7a172b80ffa7b8e54`, md5 `f7d0bd0fb63b5ccb496d183dc90b0c5d`. Top-50 cut at upload time 0.99002 (rank 50); #12 ranked 69. |
| Submission #12 | (file 06_029b_final), uploaded 27 Sep 2026, ~10:00 IST; public LB **0.988787** (exact). Chain: 016 band → 017 route → 018 lexical → 023_comb (CV 0.98742, +0.00138 vs 018, 5/5) → 028_ce cross-encoder stack (0.98779, +0.00037, 5/5) → 027d_meta meta-model (0.99046, +0.00267 vs 028, SE 0.00002, 5/5; re-verified independently; the train candidates lack the route pairs test has, so the expected LB gain was ~+0.001-0.0018) → 029b_final France blend (r17 rule, guard clean). Validator PASS; byte-identical rerun from the zip code; guard vs #11 PASS. matching sha256 `22510911e746bb01dbac0f6b2c53e1cb9af0064eb9b68db7ab703cbc722d78a4`, md5 `eb8d8813b2ab0eb0b7907aafc2e4b4a1`. Final package `submissions/final_029b/Apprentice_submission.zip` (md5 `089d3b12ea45c73337778842ac696fdb`). |
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
