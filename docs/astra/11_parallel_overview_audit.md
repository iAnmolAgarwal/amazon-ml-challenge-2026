# Audit 11: the parallel pipeline's overview (OOF 0.98082, LB 0.971) vs MAIN (006, OOF 0.97423, LB 0.963)

Auditor: independent (Opus), 2026-09-25. Inputs:
- the parallel pipeline's overview document (Downloads);
- its older code package (OOF about 0.9754), extracted in the scratchpad. Its newer name-specificity module is not available, so the overview's description is all there is for it;
- MAIN's `attempts/{002,005n,005f,006}` results and entity scores, `src/er_*.py` and `cache/raw` on the Mac;
- the second reviewer's review `docs/astra/11_codex_overview_review.md`, verified in §6.

Nothing was trained on EC2. No file in `src/` or `attempts/` was edited. Scripts and outputs are in the scratchpad under `audit11/`: `es.py`, `nn.py`, `build.py`, `desc*.py`, `resid*.py`, `addr.py`, `fuzz_feat.py`, `chain.py`, `swap.py`, `me.py`, and `*.txt`. Folds are MAIN's folds. Test data was used label-free only.

## 0. Bottom line

- **The overview is broadly sound.** It uses entity-grouped folds and out-of-fold scoring over every S1. Singletons are counted, and blocking misses count as false negatives. I checked this in its `metric.entity_scores` and `run_experiment`. Its optimism sources are real but small, and all of them also apply to MAIN:
  - the dictionary is mined from all train gold;
  - stage 2 has a second-order rival-probability path;
  - the older code early-stopped on the scored fold, but that switch was inert because the round cap was hit.

  Its LB−CV gap is −0.0098, against MAIN's −0.0112. Nothing suggests that its CV is more inflated than MAIN's.
- **Where the +0.0066 comes from.**
  - **By entity type:** 96% of the gap is on entities that have matches. Only 4% (+0.00026) is on singletons.
  - **By error type:** it is mostly recall. US mean R is 0.9456 in MAIN against 0.960; India is 0.9416 against 0.951. Precision is 0.002-0.003 lower in MAIN.
  - **Blocking is not the cause:** the oracle is 0.99398 in MAIN against 0.9943. The decision floor is not the cause either (±0.0003).
  - **Training size favours MAIN:** 40% of entities per fold model, against 20% in the parallel pipeline.

  The gap is information MAIN's features do not carry. The main piece is **how specific the words on each side of a name match are** (IDF mass of shared and one-sided name tokens). The rest is legal-form agreement and separate IDF-weighted name and address cosines.
- **Measured on MAIN's own out-of-fold probabilities.** This was a residual-information test on 240k S1, cross-fitted on MAIN's folds, with the decision re-run.
  - US, clean: token-IDF masses add **+0.0028** (5/5 folds). IDF masses + skeleton chain + legal-form agreement add **+0.0049**. Adding separate name/address cosines brings it to **+0.0061**.
  - A control that re-adds MAIN's existing name-frequency features adds +0.00002. The method therefore does not invent gains.
  - India shows +0.011 to +0.012, but that is an upper bound because the 002 base lacks the dictionary.
- **Implied France** (LB = 0.47·India + 0.38·US + 0.15·France, with US/India at CV):
  - MAIN: **0.902**;
  - parallel pipeline: **0.919**;
  - each is ±0.003 from LB rounding.

  About 0.0025 of the 0.008 LB lead comes from France or test transfer, which CV cannot see. France is the largest single loss for both pipelines: about 0.07-0.08 below the known countries, or about −0.011 on the LB.
- **Adopt, ranked:**
  1. a name-specificity side-car `feat5` (spec in §4.1): +0.003 to +0.005 CV, about 30 min per split plus one CV;
  2. stage-1 OOF trained on 100% of the other folds, which fixes the 40%-vs-100% train/test mismatch;
  3. size-invariant frequency features;
  4. a model-capacity check (the parallel pipeline uses 2.1× MAIN's learning-rate × rounds).

  All four fit into the v3 rebuild as side-cars and attempt settings. The combined plan is in §7.

## 1. Soundness of the overview's claims

| Claim | Check | Verdict |
|---|---|---|
| 5 folds by S1 entity (CRC32), every entity scored by a model that never saw its labels | `model.fold_of`, `predict_stream(row_fold=...)`, `run_experiment` | **Sound.** Training uses a sample of entities (`sample_rows`), and OOF predictions cover all rows |
| Metric: macro F0.5 per S1, singletons included, blocking misses = FN | `metric.entity_scores` takes n_true from the ground truth, and a singleton with an empty prediction scores 1 | **Sound**, identical to MAIN's `per_entity_f05` |
| Honest early stopping (off the scored fold) changed nothing | The older code early-stops on the sampled rows of the scored fold (`train_folds`: `valid_sets=[dva]`, `dva = full.subset(va)`) | **Plausible.** §10 says the stage-1 models hit the 800-round cap, so early stopping never fired and the switch could not matter. The newer code was not seen |
| Second-order stage-2 effect "shared by all configurations, so it does not affect gains" | Rival p1 values come from fold models that trained on the scored fold's pairs. MAIN has the same path (`er_feat4` on OOF from `er_eval`) | **Partly.** It affects both pipelines' levels. It is *not* guaranteed to cancel in a gain: a feature that changes p1 changes how much stage 2 leans on rival p1. Magnitude unmeasured, probably small. The 10_audit_codex nested outer-fold check answers it for both |
| India may be optimistic (dictionary from all gold) | Parallel: support ≥ 3, majority ≥ 0.6. MAIN: ≥ 5 owners. 06 audit: 100% of held-out Indic tokens are covered by the other folds at ≥ 5 owners | **Small** (≤ 0.001 India) in both. The parallel pipeline's support-3 tail is slightly more exposed |
| Name-specificity features computed from unlabeled records, identically on test | Label-free: yes. **But raw counts are not size-invariant** (§3.3): test US has half of train's S1 (663k vs 1.32M), so every chain looks half as common. 60.1% of test US names are unique, against 52.3% in train (p99 chain 150 vs 289). IDF is invariant: mean 7.94 vs 7.97 | **Transfer risk** for any raw-count feature, including MAIN's `nf_*`. Use IDF-style `ln((N+1)/(c+1))` |
| "Probabilities well calibrated; decision layer already right" | Same rule as MAIN. MAIN's floor sweep found 0.5 vs 0.6 worth −0.00026 on MAIN | Agree. The decision rule is not a gap source |
| LB will be somewhat lower (France) | LB−CV: parallel −0.0098, MAIN −0.0112 | Consistent. The gaps are similar, so there is no sign of extra CV optimism in either |

**Test-time structural difference (a transfer advantage, not a CV one).** The parallel pipeline builds its test stage-1 probabilities from its fold models, each trained on the same fraction as its OOF models, and averages them (`predict.py`). The stage-2 competition features therefore see the same kind of p1 on train and test. MAIN's stage 1 trains on 40% of entities (50% sample × 4/5) for OOF but on 100% for test, so the test p1 are sharper than anything stage 2 was trained on. That mismatch may explain part of MAIN's larger LB−CV gap (−0.0112 vs −0.0098). 10_audit_codex flagged it independently.

### 1.1 Implied France (LB rounded to 3 decimals, so ±0.0033 on France)

| | India CV | US CV | LB | France = (LB − .47·I − .38·U)/.15 |
|---|---|---|---|---|
| MAIN 006 | 0.97207 | 0.97567 | 0.963 | **0.9025** (0.899-0.906) |
| Parallel | 0.97801 | 0.98269 | 0.971 | **0.9194** (0.916-0.923) |

- **Split of the LB lead.** Of the parallel pipeline's +0.008 LB lead, the known-country CV difference accounts for 0.47·0.00594 + 0.38·0.00702 = **+0.00546**. The residual +0.0025 is equivalent to about +0.017 in France.
- **How to read that residual.** It is conditional. If test US/India score δ below CV (test has 1.9× the unmatched pool per S1), then France = implied + 5.67·δ for both pipelines. The absolute France level is therefore uncertain. The difference between the pipelines holds only if both have the same δ.
- **Plausible reasons for a France edge.** These are hypotheses, not measured:
  - fewer candidates (23.5 vs 50.6 per France S1), so fewer look-alikes to accept;
  - token-rarity features, which lower confidence on common French words (French mean token IDF is 6.65, against 7.9 for US);
  - model probabilities fed into stage 2 that match between train and test.
- **Consequence.** For MAIN, France at 0.90 costs about 0.011 LB against known-country quality. No CV-validated lever exists for it. Changes should be judged on known-country CV plus the label-free test guards, and France should not be tuned.

## 2. Where the +0.0066 CV comes from

### 2.1 Per-bucket comparison (parallel conventions: mean P with P = 1 when k = 0, mean R with R = 1 for singletons)

| | MAIN 006 India | parallel base India | parallel final India | MAIN 006 US | parallel base US | parallel final US |
|---|---|---|---|---|---|---|
| macro F0.5 | 0.97207 | 0.97076 | 0.97801 | 0.97567 | 0.97829 | 0.98269 |
| mean P | 0.9911 | 0.990 | 0.994 | 0.9938 | 0.993 | 0.996 |
| mean R | 0.9416 | 0.941 | 0.951 | 0.9456 | 0.953 | 0.960 |
| singleton accuracy | 0.9679 | 0.959 | 0.975 | 0.9769 | 0.971 | 0.980 |
| F on matched entities (derived) | 0.97231 | | 0.97818 | 0.97560 | | 0.98285 |
| pair-recall ceiling | 0.9741 | 0.975 | 0.975 | 0.9851 | 0.986 | 0.986 |
| candidates / S1 (test) | 45.8 | 22.7 | | 42.9 | 21.7 | |

- **Split of the gap, weighted by the training country shares (India 40% / US 60%):**
  - singletons: 0.4·0.0559·(0.975−0.9679) + 0.6·0.0558·(0.980−0.9769) = **+0.00026**;
  - matched entities: 0.4·0.94·0.00587 + 0.6·0.94·0.00725 = **+0.0063**.
- **Against the parallel *base*** (before its two changes): MAIN is equal or better in India on P, R and singletons, and better in US on P and singletons. MAIN trails only on **US recall (−0.0074 mean R)**.

MAIN's own loss taxonomy (006, macro-F mass):
- US 0.0243:
  - retrieved-or-missed FN only: 0.0136;
  - matched but predicted empty: 0.0051;
  - FP only: 0.0030;
  - FP and FN: 0.0013;
  - singleton FP: 0.0013.
- India 0.0279: the same order (0.0144 / 0.0054 / 0.0044 / 0.0019 / 0.0018).
- Recall-type losses are 77%.
- **The matched-but-empty entities** (6,750 US, 4,735 India) are mostly one-match entities (72%, against a 5.7% base rate) with chain names (60-63%, against 48-54%). For only 18% (US) / 32% (India) of them was no gold record retrieved at all.

### 2.2 Component by component

| Component | Parallel | MAIN | Contribution to the +0.0066 |
|---|---|---|---|
| Normalization | unidecode, mined dictionary (support ≥ 3), large abbreviation and singular/plural canon, stopwords, per-token skeleton | fold, dictionary (≥ 5 owners) + India spelling canon, France-gated street/postal rules | **≈0.** MAIN's India base is ahead (+0.0013). Plural/abbreviation swaps are < 2% of one-word swaps (§3.2) |
| Blocking | conjunction-key IDF vectors, both directions (record→top-3 S1, S1→top-15), about 22 candidates / S1 | word TF-IDF top-40 + 6 exact keys, about 44 candidates / S1 | **≤ +0.0003** (oracle 0.9943 vs 0.99398; pair-recall ceilings within 0.001) |
| Training size | 25% sample → 20% of entities per fold model | 50% sample → 40% per fold model | **MAIN ahead by about +0.0009** (one doubling, per MAIN's learning curve). The parallel pipeline's 7→25% (+0.0018) is not available to MAIN |
| Capacity | lr 0.08, up to 800 rounds (cap hit), min_data 100, max_bin 127 | lr 0.05, 600 fixed rounds, min_data 200 | **Unmeasured.** lr×rounds is 64 against 30 (2.1×), and the parallel pipeline still wanted more rounds on half MAIN's data. Plausible +0.0005-0.002 |
| Decision | expected-F prefix, min_p 0.5 | same, floor 0.6 | **±0.0003** (MAIN sweep: 0.6 is optimal for MAIN) |
| Stage 2 | 11 competition features | 11 equivalent features (MAIN copied the idea) | **≈0 in CV** (test parity favours the parallel pipeline, §1) |
| Base pair features | TF-IDF cosines on name words, name char-3grams, skeleton words, skeleton char n-grams and address words (IDF on the country's S1); legal-form shared/Jaccard; skeleton fuzz; both-direction blocking ranks | rapidfuzz views, house/street/postal parse, conservative names, `nf_*` whole-name counts, one **combined** name+address cosine | the parallel base's US lead of +0.0026 at 5.6% data, about +0.0035 at equal data. Measured pieces: legal +0.0010 US, cosines +0.001 on top of IDF (§3.4) |
| Name specificity | chain size (name, skeleton), name frequencies, IDF mass of shared vs one-sided tokens: **+0.0038** | `nf_*` = whole-name counts only | **+0.0038 reported.** On MAIN: US +0.0049 (IDF + skeleton + legal), clean |

Summary of the arithmetic:
- 0.00659 = +0.00105 (base, all in US) + 0.00177 (their 7→25%) + 0.00377 (name specificity).
- Correcting for MAIN's larger training fraction (≈ +0.0009 to MAIN), the like-for-like gap is **about +0.0075**.
- My attribution:
  - **token-rarity / name-specificity information: +0.004 to +0.005** (measured on MAIN: US +0.0049);
  - **legal-form and separate IDF cosines: about +0.001-0.002**, US-centred;
  - **model capacity or other: about +0.001**, unmeasured.

## 3. Measurements on this Mac

Protocol: 120k random S1 per country with MAIN's folds. I took their pairs from `cache/oof/002_namefreq.parquet` and fixed each pool record's owner on the **full** OOF graph (max probability). I then ran `er_model.decide` (floor 0.6) on the owned pairs, which gives 1.09M pairs. The sample reproduces 002 at floor 0.6: India 0.94508 vs 0.94500 full, US 0.97147 vs 0.97188. Names were normalized with MAIN's `er_norm` (dictionary on). Peak RSS was 2.0-3.7 GB. One build step briefly exceeded the 2.5 GB target (3.7 GB, 17 s); everything else stayed under.

### 3.1 What MAIN's false positives look like (002, floor 0.6; share of pairs)

| Pattern (core-name tokens A = S1, B = pool) | US FP | US missed gold (retrieved, not emitted) | US TP | India FP | India missed | India TP |
|---|---|---|---|---|---|---|
| same core name | **40.6%** | **54.9%** | 63.0% | 30.7% | 60.9% | 72.0% |
| one word swapped | **31.9%** | 23.2% | 16.0% | **32.0%** | 20.0% | 12.1% |
| S1 + 1 extra word | 2.9% | 8.2% | 2.1% | 6.9% | 10.7% | 4.4% |
| S1 − 1 word | 2.0% | 2.6% | 6.3% | 0.9% | 0.8% | 1.6% |
| no shared token | 16.4% | 7.5% | 9.7% | 26.5% | 4.8% | 8.7% |

- **"True name + one extra word" is not MAIN's typical distractor** (3-7% of FPs). The dominant ones are:
  - exact same core name elsewhere: 47% of those S1 names are chain names; the legal form conflicts in 14% (US) / 42% (India);
  - one word swapped ("Russell & Osborn California" vs "… Development").

  IDF masses target the second directly, and chain size and legal form the first.
- **Most missed true matches are exact same-name records** (55-61%), with a mean probability of 0.38-0.44. The model does not trust common names enough. That is the recall gap in §2.1.
- **Same-name pairs where the S1's name is unique** are still gold only 87.8% of the time (US). In train, decoys copy names too, so rarity must be combined with address evidence and cannot be a rule.
- **Mean pool-only IDF mass** (US): FP 8.66, TP 4.15, missed gold 5.23. Mean S1-only mass: FP 7.27, TP 4.05.

### 3.2 One-word swaps (typo vs genuine word)

- US: 72k different-word swaps (76% gold), 13k typos (80% gold), 0.5k plurals (85% gold), 1k prefix/abbreviation (34% gold).
- India: 69k (58%) / 10k (77%) / 0.6k (90%) / 0.9k (17%).
- Singular/plural canon ("service/services", "associate/associates") touches under 1% of swaps, so normalization is not a material gap source.
- A typo-aware variant of the IDF masses (tokens within edit distance 1-2 count as shared) adds only +0.0002 (§3.4).

### 3.3 Chain sizes (S1 per country sharing a space-free core name; label-free)

| split, country | S1 | unique name | chain 2-5 | 6-20 | > 20 | mean ln(1+chain) | p99 chain | mean token IDF |
|---|---|---|---|---|---|---|---|---|
| train India | 883,188 | 46.0% | 17.7% | 4.3% | 32.0% | 1.97 | 126 | 7.60 |
| test India | 809,986 | 46.8% | 17.4% | 3.9% | 32.0% | 1.94 | 117 | 7.59 |
| train US | 1,323,633 | 52.3% | 24.3% | 7.2% | 16.3% | 1.59 | 289 | 7.97 |
| **test US** | **663,106** | **60.1%** | 20.7% | 5.5% | 13.7% | **1.38** | **150** | 7.94 |
| test France | 259,452 | 52.4% | 21.8% | 12.3% | 13.6% | 1.52 | 269 | 6.65 |

- **Chain counts scale with block size; IDF does not.** Test US has half the S1, so its chains are half as long. MAIN's `nf_*` (log1p raw counts) and any raw chain-size feature shift by about −0.69 on test US, which makes names look rarer and matches more trusted. The same model on test will lean towards accepting same-name look-alikes.
- **Fix:** express every count as ln((N_c+1)/(count+1)), where N_c is the same-split S1 count (or pool count) of the country. CV cannot validate this; it is a transfer correction backed by the table above.
- **006 F by chain size:** US unique names 0.9814, chains of 2-20 about 0.969-0.971, > 20 0.9685. Chain-name S1 are 48% of US entities but **60% of US loss**. India: unique 0.9748, 2-20 0.960-0.965, > 20 0.9744.

### 3.4 Residual-information test (is the information already in MAIN's features?)

Method:
- Take the 002 OOF probability p, which already reflects all 48 MAIN features, including `nf_*`, conservative names and house/street/postal.
- Cross-fit a small LightGBM (31 leaves, 300 rounds) on [logit p] + candidate features, over MAIN's folds (train on 4 folds of the sample, predict the 5th).
- Re-run `decide` (floor 0.6) and score paired per fold against [logit p] + a control.
- The control recomputes MAIN's `nf_s1_s1` / `nf_p_s1` / `nf_s1_pool` / `nf_p_pool` equivalents.
- **US is clean**: the dictionary has no effect there. **India is contaminated**: 002 predates the dictionary, while these features use latinized names, so India is an upper bound.

| Added to [logit p + control] | US ΔF (SE, folds +) | India ΔF (upper bound) |
|---|---|---|
| control vs logit p only (MAIN's `nf_*` re-added) | **+0.00002** (0.00009, 1/5): the method works | +0.0051 (dictionary contamination) |
| token-IDF masses (shared, S1-only, pool-only, fractions, max one-sided, IDF cosine, counts) | **+0.00279** (0.00016, 5/5) | +0.00726 |
| skeleton chain sizes (S1 and pool record) + skeleton equality | −0.00001 (3/5) | +0.00182 |
| legal-form agreement / conflict | **+0.00104** (0.00016, 5/5) | +0.00135 |
| separate IDF cosines (name char-3gram, address word) alone | −0.00007 (2/5) | +0.00195 |
| IDF masses + skeleton + legal | **+0.00488** (0.00018, 5/5) | +0.01099 |
| + typo-aware one-sided masses | +0.00505 (5/5) | +0.01125 |
| + separate IDF cosines | **+0.00607** (0.00014, 5/5) | +0.01238 |

- **Mechanism in the US** (IDF + skeleton + legal vs control):
  - mean P 0.9913 → 0.9928;
  - mean R 0.9399 → 0.9490;
  - singleton accuracy 0.950 → 0.959;
  - false-positive pairs −17% and true-positive pairs +1.0%.

  This is the same signature the overview reports for its change: P, R and singletons all up.
- **Caveats:**
  - The base is 002 (one stage, 20% training). 006 adds stage 2, and part of the gain may overlap with it. Competition features describe rival probabilities, not name informativeness, so the overlap should be partial.
  - The stacker only re-scores owned pairs, with ownership frozen.
  - A full integration into MAIN's 127-leaf model can do more (interactions with address features) or less.

  **Forecast for 006-level MAIN: US +0.0025-0.004, India +0.003-0.006, CV +0.003-0.005.**

## 4. What MAIN should adapt (rebuilt in MAIN's harness, not copied)

### 4.1 Rank 1: `feat5` name-specificity side-car (new `src/er_feat5.py`; a new harness file, so it needs the human's approval)

- **Inputs:**
  - `cache/norm/{split}_{s1,pool}.parquet` (`country_n`, `name_core`, `name_join`, `name_skel`, `name_lat`, `addr_norm`);
  - `cache/cand/{split}_cand{tag}.parquet`.
- **Output:** `cache/feat/{split}_feat5{tag}.parquet`, keyed and ordered exactly like `{split}_feat{tag}` (write in candidate order, in 5M-pair batches like `er_feat2`).
- **Label-free, per split and per country c, identical code on train and test.** France gets its own statistics from test S1.

Definitions:
- N_c = number of S1 in c.
- df_c(t) = number of S1 in c whose `name_core` token set contains t.
- w_c(t) = 1 + ln((N_c+1)/(df_c(t)+1)). This is smoothed IDF: size-invariant, and bounded by 1 + ln(N_c+1) for tokens unseen in S1.
- IDF source: **S1 only**, not the pool. The pool carries decoys, and its density differs on test (5.76 vs 4.68 per S1).
- A = set of `name_core` tokens of the S1, B = the same for the pool record. Empty sets give −1 for ratios and 0 for masses.

| Feature | Definition | Measured role |
|---|---|---|
| `ns_sh` | Σ_{t∈A∩B} w(t) | strength of the shared evidence |
| `ns_a1` | Σ_{t∈A∖B} w(t) | S1 words the record lacks |
| `ns_b1` | Σ_{t∈B∖A} w(t) | extra or swapped words in the record (FP 8.66 vs TP 4.15, US) |
| `ns_a1_frac`, `ns_b1_frac` | ns_a1/(ns_sh+ns_a1), ns_b1/(ns_sh+ns_b1) | share of the name left unexplained |
| `ns_a1_max`, `ns_b1_max` | max w over the one-sided tokens (0 if none) | one rare distinguishing word |
| `ns_cos` | Σ_{A∩B} w² / √(Σ_A w² · Σ_B w²) | IDF-weighted name-word cosine |
| `ns_n_a1`, `ns_n_b1` | \|A∖B\|, \|B∖A\| | counts |
| `lg_eq`, `lg_conf`, `lg_any` | legal token sets from `name_lat`: tokenize as `er_norm` does (`merge_split_legal`, `NAME_ABBR`), then intersect with `er_norm.LEGAL`. Equal and non-empty / both non-empty and different / either non-empty | legal-form swap decoys (US +0.0010) |
| `cos_nc3` | cosine of char-3-gram hashed binary vectors of `name_join`, IDF on the country's S1, L2 | typo-robust name evidence |
| `cos_aw` | cosine of address-word hashed binary vectors (`addr_norm` tokens), IDF on the country's S1 | separates address from name (with `cos_nc3`, +0.001 US) |
| `sk_rar_a`, `sk_rar_b` | ln((N_c+1)/(n_c(skel)+1)) for the S1's and the pool record's `name_skel`, counted among S1; −1 if the skeleton is empty | India only, +0.0018 (upper bound) |
| `nf_*_inv` (4) | ln((N_c+1)/(count+1)) with the same counts as `nf_*` (S1 counts use N_c, pool counts use the country's pool size) | transfer fix for the size shift (§3.3). Keep `nf_*` in the first paired CV, then test dropping it |

- **Optional:** `ns_b1_hard` / `ns_a1_hard` excludes one-sided tokens that have a partner on the other side within Levenshtein 1 (2 if ≥ 8 characters, both sides ≥ 4 characters). It is worth +0.0002; skip it unless it is free.
- **Overlap with MAIN's features.**
  - The whole-name frequency part of the parallel pipeline's feature group already exists in MAIN as `nf_*`; the control shows it adds 0.
  - `word_sim` is IDF-weighted, but it is symmetric and mixes name with address; IDF for it is fitted on the pool.
  - `n_tset` hides extra words (a subset scores 100), and the rapidfuzz scores are rarity-blind.
  - `n_tok` / `n_tok_p` give token counts but not rarity.
  - None of these encode directional rarity. §3.4 measures the information as absent.
- **Cost.**
  - Coding: about 1.5 h.
  - Compute: about 20-30 min per split on 8 cores, using a pure-Python token loop over about 98M pairs in `multiprocessing` batches (mirror `er_feat2`), plus sparse row-wise dot products.
  - Output: 15-19 float32 columns, about 6-7 GB on disk. For the memory impact, see §6(d).
- **Fit with the v3 rebuild:** yes. It only needs norm + cand, so it can be computed on `_v3` candidates after the running v3 chain without touching its artifacts.
- **Expected:** CV +0.003-0.005 (US measured +0.0049/+0.0061 on 002, clean; India upper bound +0.011). LB +0.0025-0.004 from known countries. France is unmeasurable; the direction is plausibly positive because false positives fall 17%.

### 4.2 Rank 2: stage-1 OOF at 100% of the other folds (`TRAIN_FRAC = 1.0` for the stage-1 attempt)

- **Why:** it removes the 40%-OOF vs 100%-test mismatch in the p1 fed to stage 2. The parallel pipeline avoids the mismatch by averaging its fold models at test.
- **Expected:** +0.0003-0.0008 CV (learning curve) plus a transfer correction (sign unknown, −0.001 to +0.002).
- **Cost:** stage-1 CV about 1.5× longer. Memory is unchanged, because the full dataset is binned once and folds are subsets.
- It joins the next rebuild directly.

### 4.3 Rank 3: capacity check

- Stage-2 attempt with `ROUNDS = 1200` at lr 0.05 (or lr 0.08 with 800 rounds) and nothing else changed, as a paired CV.
- **Expected:** +0.0005-0.002, unmeasured.
- **Cost:** about +40 min per CV.
- Do it only after `feat5`, as its own attempt.

### 4.4 Not worth adapting now

| Item | Why not |
|---|---|
| Their blocking | MAIN's oracle is equal. v3 already adds reverse top-3 and skeleton keys |
| min_p 0.5 | MAIN's own sweep puts the optimum at 0.6. Re-check {0.5, 0.55, 0.6} free with `--from-oof` on the final OOF |
| Singular/plural and abbreviation canon | < 2% of swaps |
| Their proposed entity-consistency ("coherence") idea | MAIN's coherence failed the test-shift guard. It is untested in the parallel pipeline too |

## 5. Stage-2 competition features: same second-order issue in both pipelines

Both pipelines score rival S1 pairs with fold models that trained on the scored fold's labels. MAIN's +0.0034 (006) and the parallel pipeline's +0.0038 (name specificity inside a two-stage model) carry the same small optimism. The only direct measurement is the 10_audit_codex nested outer-fold check. It costs 2-3 h. Run it once on the final bundle, not per feature.

## 6. Verification of the second reviewer's findings (`11_codex_overview_review.md`)

| # | Finding | Verdict | Evidence |
|---|---|---|---|
| 1 | Missing component = rarity of unmatched name words; spec: shared / S1-only / pool-only IDF mass, fractions, max pool-only IDF; +0.001-0.003 LB | **AGREE, and measured larger.** (b) Its spec is exactly my §4.1 core (same smoothed IDF). No MAIN feature encodes it: `n_tset` gives 100 for "Rao Motors" vs "Rao Sunrise Motors"; `nf_*` are whole-name counts (control +0.00002); `word_sim` is symmetric and mixes in the address. On 002: US +0.0028 (IDF masses), +0.0049 with legal/skeleton. "Cap unseen weights": the smoothed formula is already bounded by 1+ln(N+1). "No hard rejection" is right: extra-word pairs whose extra word has IDF > 9 are gold 40% (US) / 57% (India) of the time | §3.1, §3.4 |
| 2 | (a) Principal cosine combines name + address; separate name-word / address-word TF-IDF cosines are missing; add skeleton ambiguity counts; skeleton definitions differ | **AGREE on the facts, PARTLY on value.** `er_norm` builds `text = core tokens + address tokens`. `er_block.vectorize` hashes word 1-2 grams of `text`, with IDF fitted on the country's **pool**. `word_sim`, `word_rank`, `word_gap`, `rev_rank` and `rev_gap` all derive from that one cosine. Separate cosines alone add −0.00007 US, but +0.001 on top of the IDF masses. Skeleton counts: 0 US, +0.0018 India (upper bound). "Skeleton absent from features": true today (`er_feat.COLS`), but in v3 the key flags `k_sk` / `k_skh` / `k_fl` / `k_rev` become features automatically (every `k_*` column enters `base`). MAIN's skeleton is whole-string, letters only, digits dropped; the parallel one is per token with letter substitutions: confirmed | `er_norm.py` `normalize_rows`, `er_block.py` `vectorize`, `er_feat.py` `build` |
| 3 | +0.00543 decomposition (+0.00035 / +0.00163 / +0.00345); the 7→25% gain is not available to MAIN; oracle 0.9943 vs 0.99398 | **AGREE.** With 4-decimal MAIN country scores: .47·(−.00134) + .38·(.00259) = +0.00035; .47·.00235 + .38·.00139 = +0.00163; .47·.0049 + .38·.00301 = +0.00345; total +0.00543 (+0.00546 unrounded). These are test-weighted over 0.85, i.e. their contribution to the LB. MAIN already trains on 40% per fold model against their 20% | §1.1, §2.2 |
| 4 | Transfer in both directions; the parallel archive lacks conservative names and house/street/postal parse; number equality uses the first number | **AGREE on the facts** (older archive). `features.py`: `num1_eq` uses the first token of `addr_nums`, which is in text order (`normalize.norm_addr`), and there are no house/postal/conservative-name features. It does have `nums_shared` / `nums_jac`, so number evidence is not absent. Whether to invest in the other branch is the human's call. Under this repo's rules MAIN is the submission line, so I rank it second to the `feat5` port. The blend advice is right: missing scores are not probability zero | parallel `features.py`, `normalize.py` |
| 5 | Validation claims do not justify adding gains mechanically; the dictionary uses all labels; the "shared stage-2 effect" claim is unsupported; do not copy coherence | **AGREE.** The dictionary effect is small (≤ 0.001 India) in both pipelines. The stage-2 claim is a level-vs-gain confusion (§5). Its diagnostics list is sound; the useful new one is "newly emitted test pairs with rare unmatched name tokens, by address support". Coherence: agree | §1, §5 |
| 6 | Order: features on current candidates → stage 2 only with stage 1 fixed → full two-stage. (d) Memory: 8 float32 columns = 3.14 GB; the 61.7 GB peak leaves no room | **(e) Order AGREE**, adapted to v3 (§7). A stage-2-only screen is the cheapest honest test; §3.4 is its miniature. **(d) Arithmetic right** (98,021,508 × 8 × 4 B = 3.14 GB). Re-estimate after `b6f0ccb`: `to_matrix` no longer holds the joined frame (about 25 GB for 64 columns) and the matrix (about 23 GB) at the same time, so that phase falls to about 25 GB. **The join phase remains** (`er_eval.train_oof`: side-car joins, then the fold join, then the gold join, each materialising a new full frame while the old one is alive), at roughly 2 × 25 = 50 GB transient; this is my estimate, not a measurement. 15-19 more columns add about 6-7 GB per frame copy, i.e. about 12-14 GB transient: **not safe**. Fix before `feat5` CV (harness change, human approval): read each side-car with `columns=`; since `feat2`, `feat4` and `feat5` are written in the `feat` row order, check that the key columns are equal and `hstack` instead of joining; attach fold and y as numpy arrays via one small key join; or fill preallocated `X` column slices file by file. Measure the peak once with `/usr/bin/time -v` | `er_eval.py` `train_oof`, `er_model.py` `to_matrix` |
| 7 | Implied France 0.9194; 0.98 requires France ≈ 0.979; balanced 0.98 target (.990/.986/.940) | **AGREE** (0.91942; (0.98 − .459665 − .373422)/.15 = 0.97942; .3762 + .46342 + .141 = 0.98062). The planning ranges are guesses; mine for MAIN are in §7 | §1.1 |

## 7. One combined recommendation for MAIN's next iteration

This combines 10_audit_codex (v3 plus skeleton features, stage-1 at 100%, memory gate, nested check, French repairs kept separate), the second reviewer (port name specificity first, stage-2 screen first) and this audit (measured feature value, size invariance, capacity). Machine: one 8-core / 64 GB. Heavy jobs run one at a time.

| Step | What | Compute (A) | Go / no-go |
|---|---|---|---|
| 0 (running, do not touch) | v3 chain: `007a` CV → `feat4` v3 → `007` CV (about 19:00) → v3 test → `007` predict (about 21:30) | as scheduled | `007` vs `006`: paired rule plus test guards (US house-conflict about ≤ 7.5%, empty share, mean k by country vs OOF). The winner's candidate tag becomes the base for step 2 |
| 1 (Mac, during step 0; human approval) | Write `er_feat5.py` (§4.1) and unit-test it on a sample (the scratchpad `audit11` scripts are the reference implementation). Make the side-car loading memory-lean (§6 d). Verify identical CV on the `_s20k` tag | 0 (2-3 h coding) | identical 006 result on `_s20k` before and after the loader change |
| 2 | `feat5` for train on the winning tag, then **`008s`: stage-2-only screen** = parent stage 2 (`007` or `006`) + `feat5`, with the stage-1 OOF fixed | about 0.5 h + 0.8 h | Paired gain ≥ +0.001, > 2 SE, ≥ 4/5 folds, positive in both countries. Also check the `feat5` distributions train vs test per country (IDF masses should match; the invariant counts should match better than raw `nf_*`) |
| 3 (if 2 passes; otherwise run it without `feat5` as the required control) | **`008a`**: stage 1 + `feat5`, `TRAIN_FRAC 1.0` → OOF; `feat4` from it; **`008`**: stage 2 + `feat5` + `feat4` | about 1.5 h + 0.2 h + 1.2 h | paired vs step-2 parent; peak RSS ≤ about 54 GB |
| 4 | Test: `feat5` test, `008a` predict (100%), `feat4` test, `008` predict, validate, package | about 2.5 h | Label-free guards against the parent: US house-conflict within +0.5 pt; predicted-empty share within ±1 pt of OOF; mean k per country within 0.1 of OOF; the new diagnostic of emitted pairs with `ns_b1_max` > 9 by address support |
| 5 (optional, before the final submission) | (a) `ROUNDS 1200` stage-2 attempt (about 1.3 h); (b) nested outer-fold check on the final bundle (2-3 h); (c) French parser repairs as a separate attempt (10_audit_codex §6); (d) blend with the parallel pipeline's OOF if keyed probabilities become available | 3-6 h | each under the paired rule; France changes by label-free incidence only |

- **Compute after step 0:** about 7 h for steps 2-4, plus 3-6 h optional.
- **Expected CV:**
  - v3: +0.0005-0.0015;
  - `feat5`: +0.003-0.005;
  - stage-1 at 100%: +0.0003-0.0008.

  Total about **0.978-0.981**.
- **Expected LB:** about **0.967-0.971** if LB−CV stays near −0.011. France stays about 0.90 unless the lower false-positive rate carries over to it.
- **What is measured and what is guessed.** The US `feat5` value is measured on 002 (clean). The India value, the overlap with stage 2, capacity and all LB and France numbers are estimates.
