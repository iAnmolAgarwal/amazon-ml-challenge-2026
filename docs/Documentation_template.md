# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Apprentice  
**Team Members:** Anmol Agarwal (team leader), Akshit Kumar Singh, Nitya Arora, Harsimran Singh  
**Submission Date:** 28 September 2026

---

## 1. Executive Summary
We model the generator behind the data: true matches are noisy copies of an S1 record, and unmatched pool records are near-copy decoys made by a few operators (a house number shifted by one of ten steps, one inserted marker word, a swapped legal form), which the test set contains at about 1.9x the train rate. Candidates come from TF-IDF retrieval and exact keys (train oracle F0.5 0.998); LightGBM models on pair-local features, including label-free decoy-marker scores that also work for unlabelled France, score each pair, a cross-fitted meta-model combines them with fine-tuned e5 cross-encoders and a second LightGBM family, and an expected-F0.5 rule with one owner per pool record picks each answer. US and India changes were accepted only on a paired 5-fold CV test plus a label-free test-shift guard (CV 0.99184); France rules were set by leave-one-country-out tests and label-free counts (public leaderboard 0.990203).

---

## 2. Methodology

### 2.1 Problem Analysis
- Each Source-1 (S1) record matches 0 to many pool records (sources 2/3); each pool record has at most one owner, never across countries. Train: 2.21M S1, 10.3M pool records, 3.46 matches per S1. Test: 1.73M S1 (India 47%, US 38%, **France 15% with no training labels**), 9.97M pool records.
- True-match noise: typos, truncated digits (1995 vs 995), fillers (center, services, inc), alias markers (aka, dba, formerly), website forms, token reorder and splits, acronyms, native-script (Indic) names, addresses that drop numbers, gain state codes or become "None".
- Unmatched pool records (**decoys**) are near-copies of real businesses: house number shifted by +1, 2, 3, 4, 5, 7, 9, 11, 13 or 21; one name word inserted from a per-country list; a swapped legal form; a changed unit number. Evidence: on name-similar train pairs, non-matches spike at exactly these ten shifts (13.6k-15.8k pairs each on a 10% sample, against 1k-3k at every other difference, including all negative ones). Decoys never have an empty address.
- Test has about 1.9x the decoy density of train. We measured this without labels: in US, candidate pairs whose house numbers differ by one of the ten shifts are 1.94x as frequent per S1 on test as on train, while every other difference keeps the train rate. Test decoys also come in clusters that share one perturbed house number.
- Consequence: features that judge a candidate by its agreement with the S1's other candidates win in CV and emit decoys on test, where each S1 has about twice as many decoy siblings. Both such families in Table B.3 (pool-record graph features, an S1-context stacker) were dropped. In our main family, context enters only through scores: the rank and margin of a pair's earlier-stage probability among the S1 that claim the same pool record and among the S1's own candidates. The second family keeps one content-context group, the similarity of a pool record to the S1's own confident records (stage-2 p ≥ 0.9); the meta-model that reads it passed the same label-free test guard before upload.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier (multi-stage LightGBM, fine-tuned cross-encoders, and a meta-model over two model families)  
**Core Innovation:** label-free decoy-marker scores and generator-operator features that model how the synthetic noise and decoys were produced, re-centred per country so the same features work on the unlabelled country.

**Validation protocol**
- 5-fold CV over the 2.21M train S1, stratified by country × match count. Every stacked input is out-of-fold. Final fold SD: 0.00003.
- Acceptance: paired per-fold gain over the parent > 2 SE and positive in at least 4 of 5 folds.
- Decoy-density stress test: features that could react to decoy density (graph features, the per-country re-centring of the marker scores) were re-scored with train decoys cloned to about 2x.
- Label-free test guard before every upload: per country, compare test and train on emitted pairs per S1, empty-answer rate and the share of emitted pairs with conflicting house numbers. A change that emits far more extra pairs on test than on train is rejected even if CV improves.
- Fair sample (glossary): changes that add candidate sources are also checked paired on it.
- Test data without labels: test records are used, label-free, in three places: the France decoy-marker table, the France word lists (`src/er_frwords.py`) and the test guard's summary counts. No test label or leaderboard score enters any feature.
- France: leave-one-country-out (LOCO) tests plus label-free counts on France's own test candidates (4.3).
- Public leaderboard: recorded after each upload. US/India changes were chosen by CV alone. France rules were fixed by LOCO and label-free counts before upload, but the leaderboard was the only check of the France result: one France change (the empty-address correction, which passed LOCO in one direction only) was kept because the leaderboard rose.

**Glossary.** *Band*: pairs whose stage-2 probability is ≥ 0.001 (the cross-encoder band is 0.003-0.997). *Route*: a candidate source (character retrieval, extra keys, reverse retrieval). *Name twin*: another S1 of the same country with the same normalized name. *LOCO*: fit on US and score India as if unlabelled, and the reverse. *Lift*: a country's marker scores re-centred by its own decoy base rate. *Generator operator*: one edit the data generator applies to make a noisy match or a decoy (house shift, truncated digits, word insertion or drop, legal-form swap, reorder, acronym); generator-operator features encode which operators link a pair. *Chain probability*: our family's probability after its last correction, before the meta-model. *Second family*: an independently built LightGBM and cross-encoder pipeline (`src/stack`) with its own blocking and folds. *Fair sample*: a fixed, label-free 10% of train S1 that gets the same extra candidate sources as test.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** per country, word TF-IDF (hashed name + address 1-2 grams), top 40 neighbours; exact keys on house + street, space-free name, sorted space-free name, name + house, initials + house (acronyms) and alternate ("formerly") name, skipping oversized key blocks; a character 3-gram TF-IDF route (top 40 per view over core name and normalized address); capped exact name keys (name prefixes, consonant skeleton with numbers, initials, word pairs, name word + house); and reverse character retrieval from unclaimed pool records back to S1 (top 5). Pairs added by the last three routes are scored by a route classifier and kept at p > 0.001. Normalization first maps native-script tokens to Latin with a dictionary learned from training pairs.
- **Candidate pairs generated:**

| Set | Train | Test |
|---|---|---|
| Word TF-IDF + exact-key blocking | 98.0M pairs (44 per S1), oracle F0.5 0.994, 98.1% of true pairs | 78.9M pairs (46 per S1) |
| Other routes (character retrieval, extra keys, reverse retrieval) | proposed pairs cut by the route classifier at p > 0.001 | same |
| Final candidate set (`candidate_pairs.tsv`) | 13.37M pairs (6.06 per S1), oracle F0.5 0.9980 | 11.84M pairs (6.83 per S1: US 5.63, India 6.06, France 12.31) |

  `candidate_pairs.tsv` is every pair the final models score: our pairs that pass the stage-2 filter (p ≥ 0.001) or were added after it (for US and India also chain probability ≥ 0.001), plus the second family's cascade (p > 0.02, top 10 per S1). 58,364 second-family pairs that our stage 2 scored below 0.001 are scored but never matched (README, "Candidate file"). France has twice the pairs per S1 because its pairs are not cut before the France blend.
- **How we ensured true matches were not lost:** oversized key blocks are skipped rather than truncated by row order. The character route, the extra keys and reverse retrieval recover matches that word retrieval misses; their added pairs are scored by the route classifier, and the test guard was run on every upload that added a source. With them, the oracle of the final train set is 0.9980, so unretrieved pairs cost at most 0.0020 F0.5. Further retrieval has little left to find: a wider reverse retrieval finds only 4.9% of the remaining misses.

---

## 4. Matching Model

**4.1 Pipeline**
```
S1 + pool records, per country
 -> normalize (native-script words -> Latin, dictionary learned from train pairs)
 -> blocking: word TF-IDF top 40 + exact keys + char 3-gram route + extra name keys + reverse retrieval
 -> stage 1 LightGBM -> stage 2 LightGBM (+ competition features); keep p >= 0.001
 -> band model (all features) -> lexical correction -> generator-operator correction
 -> e5 cross-encoder stack on the uncertain band
 ==> candidate_pairs.tsv: chain p >= 0.001 (France: stage-2 band) + second-family cascade; 11.84M pairs, 6.83 per S1
 -> meta-model over both model families (US, India) | France blend (4.3)
 -> empty-address correction -> expected-F0.5 decision (4.4)
```

**Features used:**
- Name features: fuzzy ratio / token-set / token-sort / partial / Jaro-Winkler on several name views (core, website-free, space-free, sorted, initials, alternate name); IDF-weighted shared and one-sided token mass; character 3-gram cosine; name-frequency counts; legal-form relations; **decoy-marker scores** for added, dropped and swapped words, (n_d + 1)/(n_d + n_0 + 10), where n_d counts candidate pairs whose pool name adds (drops, swaps) the word and whose house number is shifted by a decoy step, and n_0 counts the same with equal house numbers (no labels, so France gets its own table from its test candidates); name edit operations (typo position, confusable characters, acronyms, prefix drops, reorder / join / split, fillers, alias and web markers).
- Address features: token-set and partial scores; IDF-weighted token mass; house, street, postal and unit equality or conflict; house relations (dropped leading digits, near difference on the same street, digit reorder, signed shift and membership in the decoy shift set); number-multiset operation codes; a French house-number parser.
- Other: stage-2 competition features from out-of-fold probabilities (rank and margin of the pair among the S1 that claim the same pool record, and among the S1's own candidates); cross-fitted lexical likelihoods of added / dropped words and character edits (42 features); cross-encoder logits.

**Model type:** LightGBM (pair models, corrections, meta-model) with fine-tuned multilingual-e5 cross-encoders (small, base).

**4.2 Models** (LightGBM unless stated; every train input is out-of-fold)
1. *Stages 1-2 (blocking filter)* (127 leaves, 600 rounds). Stage 2 adds competition features from stage-1 out-of-fold probabilities.
2. *Band model:* one model on stage-2 pairs with p ≥ 0.001 and every feature family.
3. *Route classifier:* scores pairs added by the character route, extra keys and reverse retrieval.
4. *Two corrections boosted from the previous logit:* first the 42 lexical likelihoods, then generator-operator and pool-side competition features (255 leaves, 800 rounds).
5. *Cross-encoders:* intfloat/multilingual-e5-small and e5-base, fine-tuned as pair classifiers on two S1 halves split by fold, so every train score is out-of-fold (held-out AUC 0.9996); on test each S1 is scored by one half model, chosen by its id, as on train. A small LightGBM stacks the score on the cross-encoder band.
6. *Meta-model* (US, India; 127 leaves, 600 rounds, cross-fitted on our folds) over the union of our pairs and the second family's. The second family (src/stack) was built independently in its own environment: its own normalization and transliteration, blocking, a two-stage LightGBM with name-specificity features, label-free decoy features, and a stage-3 stack over three e5 cross-encoders (one cross-country, two in-country); for France it adds a cross-country e5-base cross-encoder, its own decoy stacker and a marker veto. Both families were written by the team during the challenge; the second was built on purpose in a separate environment (pandas, Python 3.11, its own folds) so that the meta-model combines two independent error patterns, and both use the same generator findings (the decoy shift set, generator-operator features). Inputs: both families' logits and their min / max / sum / difference, both families' cross-encoder scores, a few pair features, the logit of one flat LightGBM on all 205 band features, and pool-side competition. The second family uses its own entity folds (crc32 of the S1 id), and the cross-encoder halves are not nested in the meta-model folds, so this CV may be slightly optimistic.
7. *Empty-address correction* (63 leaves, 300 rounds): boosted from the meta logit using the name-twin structure among a pool record's claimers. Only empty-address pool records can gain probability (decoys never have an empty address); every other pair can only lose.

Pretrained: intfloat/multilingual-e5-small (MIT, 118M) and e5-base (MIT, 278M) at pinned revisions, fine-tuned on training pairs only (12 checkpoints); LightGBM 4.7.0 (MIT); no external data; all libraries permissive except Unidecode (GPL-2.0, second family only).

**4.3 France (no labels).** Nothing is trained on France labels. Each rule was fixed by LOCO or by label-free counts on France's own test candidates.
- *Scale:* decoy-marker scores are re-centred per country as logit(score) − log(Σn_d / Σn_0), so France's table, built from its own candidates, lands on the train scale.
- *Blend:* p = σ(0.5·logit p_ours + 0.5·logit p_second) on the union of both families' France pairs. In a LOCO grid, weights 0.5-0.6 were the only ones that did not lose in-country and gained in both LOCO directions (+0.0058 / +0.0024 for the full blend); union, temperature 2 and a 2x logit-scale error each flipped sign with the direction. The cross-encoder logit enters the blend with weight 0.1 on band pairs (LOCO +0.00053 / +0.00052).
- *Decoy words:* `src/er_frwords.py` derives the lists without labels from France's test candidates: pool names that add exactly one word to the S1 name on the same street, split by equal house number versus a decoy shift. *participations, holding, distribution, international* are added 17.6-17.7k times with a decoy shift and only 51-60 times at an equal house (shift share 0.997); pairs whose pool name adds one of them get p = 0. *groupe, developpement, france* have the same shifted counts but about 1.2k equal-house insertions (share 0.93), so they are fillers as well as markers and are not vetoed; the second family's zeros on them are treated as missing. On US "partners", the same dual-use case with labels, a veto would remove pairs that are 99.8% true (−0.0024 US).
- *Renamed business at the S1's own unique address* (no shared name word, same house number, a shared street word, the only S1 of its country at that address): max(blend, p_second) (LOCO +0.00033 / +0.00062).
- *Empty-address correction* fitted on US + India without the country flag and applied to France (transfer: fitted on US, applied to India +0.00006, 5/5 folds; fitted on India, applied to US +0.00002, not significant).

**4.4 Threshold selection method:** expected-F0.5 decision per S1. Each pool record goes to its highest-probability S1. Per S1, the prefix k maximizing 5·Σp(top k) / (Σp + 4k) is kept unless the probability of no match, Π(1 − p), is larger. Then pairs with p ≤ 0.6 are dropped. The prefix score is a plug-in approximation of expected F0.5 (a ratio of expectations with pairs treated as independent), and the one-owner step is greedy per pool record, so the rule is near-optimal rather than exact. The 0.6 floor is an empirical correction on top, chosen by paired CV: on the same stage-1 probabilities it raises CV from 0.97008 to 0.97083.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 5-fold CV on all 2.2M train S1: **0.99184** (US 0.99164, India 0.99213; fold SD 0.00003). France has no labels. If US and India held their CV on test, the public score of 0.990203 would imply France ≈ 0.981; the higher test decoy density probably lowers US and India too, so 0.981 is a lower bound for France (the public split's country mix may also differ).
- **Common false positives (wrong merges):** decoys that keep the name and address but shift the house number by a typical step or insert a generic business word; legal-form swaps; unit-number changes in the same building. Out-of-fold false-positive pairs fell from 12,987 (band model) to 5,564 (final), and pair precision is 0.9993.
- **Common false negatives (missed matches):** of the CV loss (1 − 0.99184 = 0.0082), at most 0.0020 comes from true pairs never retrieved (oracle 0.9980). The rest are true candidates that score below the floor (missed pairs fell from 302,721 to 186,753; pair recall 0.976). On the last analysed parent (CV 0.99051), two thirds of the in-candidate headroom (+0.0042 of +0.0064 oracle F0.5) was pool records with an empty or "None" address that a name twin also claims: with no address the twins are interchangeable. S1 records never have an empty address. S1 with a single true match score lowest (0.973, against 0.990-0.994 for two or more), because one miss costs the whole entity; S1 with no true match (5.6% of train) score 0.997, so the empty answer rarely fires wrongly.

| Error type (final out-of-fold decisions, US + India) | Pairs | Share |
|---|---|---|
| FP: pool record is a true match of another S1 (wrong owner) | 2,369 | 43% |
| FP: unmatched pool record, name adds words | 814 | 15% |
| FP: unmatched pool record, house number shifted by a decoy step | 303 | 5% |
| FP: unmatched pool record, legal-form swap or unit conflict | 77 | 1% |
| FP: unmatched pool record, other | 2,001 | 36% |
| FN: empty pool address, and the S1 has a name twin | 103,587 | 55% |
| FN: never retrieved | 51,473 | 28% |
| FN: other, scored below the floor or lost to another S1 | 31,693 | 17% |

Examples from train. False positive (p = 0.9993): S1 "Schmidt's Coastal Retail, 209 11th Street, Herrin, IL" vs pool "Schmidt Coastal (Retail), 210 11ND STREET, HERRIN, IL", a decoy one house number away. Missed match (p = 0.44): S1 "Legacy Learning Consultants LLC, 646 Pontiac Drive, Georgetown, IL" vs pool "legacy learning consultants llc" with no address, whose name another S1 shares.

---

## 6. Conclusion
Modelling the generator drove the largest single steps: pair features for name specificity, house-number relations and legal form (+0.009 CV), decoy-marker scores (+0.002 CV but +0.007 public LB) and generator-operator codes (+0.001). The second model family added +0.0027 CV through the meta-model. One LightGBM on all 205 features matches the whole correction chain (0.98779 both, Table B.2), so the gains came from the features, not the stage structure. A production version would keep one LightGBM on all features plus the meta-model, and drop the stage chain and the flat-model input (+0.00002 CV for about 32 GB and 2 hours). The most important lesson was to validate every idea against the test's higher decoy density with a label-free guard: features that exploit an S1's other candidates look excellent in CV and fail on test.

**Limitations.** (1) The pipeline has about 20 steps in two environments; two command lines of the second family were rebuilt from its run records, and the GPU steps are not bit-identical on rerun, so the second family's outputs are shipped with the code. (2) France rests on LOCO proxies and label-free counts; its true score is unknown. (3) Empty-address pool records shared by name twins cost about 0.004 F0.5 (measured on the 0.99051 parent); the empty-address correction recovers only a small part (0.99177 → 0.99184 CV), because with no address the twins carry no separating evidence. (4) Train candidates lack the character-route pairs that test has; the fair-sample checks cover this. (5) The France path is selected by the country name; a new unlabelled country would need the same switch. (6) The decoy-clone stress test, the test guard, the LOCO grids of our family and the ablations were run with analysis scripts that are not in the package; the package reproduces the pipeline, not these checks.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` contains `src/` (all pipeline modules; `src/ce/` the cross-encoder scripts; `src/single/` the flat LightGBM; `src/stack/` the second model family with its own README and requirements), `attempts/` (the settings of every stage used, one folder per stage), `artifacts/stack/` (the second family's outputs used by the submission), `config.yaml`, `README.md`, `requirements.txt`, `requirements-gpu.txt`, `run_manifest.json` and `SHA256SUMS`. The README lists every command in order, the expected CV after each stage, what was rerun to verify it, and each GPU run. Compute: about 30-55 CPU hours on 12-vCPU / 96 GB Linux machines, about 15-20 GPU hours on Kaggle 2x T4, and about 120 GB of disk for intermediate files; entry points: the README command list rebuilds both output files from the raw TSVs; its last command, `python src/er_predict.py 029f6_final --from-probs 030q_fix`, writes them in 19 s once the cached probabilities exist.

### B. Additional Results
**B.1 Progression** (5-fold CV macro F0.5 on US + India; public leaderboard where uploaded)

| Stage | CV | Public LB |
|---|---|---|
| Blocking + one LightGBM | 0.943 | 0.929 |
| + name frequency, conservative names | 0.961 | 0.949 |
| + native-script dictionary, floor, competition features | 0.974 | 0.963 |
| + pair features (specificity, house relations, legal form) | 0.983 | 0.974 |
| + decoy-marker name features | 0.985 | 0.981 |
| + band model, country lift, French parser, character retrieval, lexical correction | 0.986 | 0.98393 |
| + generator-operator correction | 0.987 | 0.985 |
| + pool-side competition features in the final correction | 0.98742 | - |
| + cross-encoder band score | 0.98779 | - |
| + meta-model with the second model family, France blend | 0.99046 | 0.988787 |
| + extra name keys / reverse retrieval learned on a 10% train sample | 0.99051 | 0.989022 |
| + cross-encoder and pair features as meta inputs | 0.99074 | 0.989465 |
| + cross-encoder on added pairs, empty-address correction, France on the extra-key base | 0.99079 (fair sample +0.00049) | 0.989727 |
| + extra keys for every train S1 with cross-encoder scores¹ | 0.99182 (fair sample +0.00014) | 0.989801 |
| + France rename rule | France only (LOCO +0.00033 / +0.00062) | 0.989886 |
| + cross-encoder as a third France input | France only (LOCO +0.00053 / +0.00052) | 0.989990 |
| + single flat LightGBM as a meta-model input | 0.99184 (+0.00002, passes the paired rule) | 0.989992 |
| + empty-address correction applied to France (final) | France only (transfer US→India +0.00006, 5/5; India→US +0.00002, not significant); kept on leaderboard evidence (+0.00021) | **0.990203** |

¹ From this row on, the train candidate set gains the pairs test already had, so CV rises by about +0.001 for reasons other than the model. The fair-sample figure is the comparable gain; it is measured on the 10% sample where the change acts, so the full-CV effect is about a tenth (+0.00049 → +0.00005 in the row above). Early rows were logged at three decimals. The CV-LB gap fell from 0.011 (third row) to 0.009 with the pair features and 0.004 with the decoy markers, and stayed between 0.0011 and 0.0021 afterwards, which fits the label-free finding that test is decoy-denser.

**B.2 Is the stage chain necessary?** (one-model ablation, same folds and decision rule)

| Variant | CV |
|---|---|
| Correction chain up to the cross-encoder stack (5 LightGBM stages) | 0.98779 |
| One LightGBM on all 205 features of the stage-2 band | 0.98779 (paired −0.00001: equal) |
| Same, without generator-operator and cross-encoder features | 0.98638 (−0.00141, 0/5 folds) |
| One XGBoost on the same 205 features | 0.98778 |
| Logistic GAM, 16 features | 0.98017 |

The learner does not matter; the information is in the features (largest gain shares: the probability margins, then the cross-encoder and the decoy markers). The GAM's gap shows that interactions matter: a decoy house shift counts only when the name matches.

**B.3 Negative results** (ideas that won somewhere and were dropped)

| Idea | Evidence for | Why dropped |
|---|---|---|
| Pool-record graph features | +0.0033 CV (5/5) | −0.037 (0/5) once train decoys were cloned to test density |
| S1-context stacker on stage-2 probabilities | +0.0028 CV (5/5) | On test it emitted 3.4x (US) and 5.8x (India) the train rate of extra pairs (label-free guard) |
| Broad veto of dual-use decoy words | 93% of France insertions carry a decoy shift | The labelled US analogue removes pairs that are 99.8% true (−0.0024 US) |
| Wider reverse retrieval (top 3 per pool record) | finds 17% of all misses | Only 4.9% of the misses left after the current routes, at 7.8 new pairs per train S1 |
