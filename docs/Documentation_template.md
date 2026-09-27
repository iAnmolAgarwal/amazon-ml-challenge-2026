# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Apprentice  
**Team Members:** Anmol Agarwal (team leader), Akshit Kumar Singh, Nitya Arora, Harsimran Singh  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
A multi-stage blocking + gradient-boosting pipeline built around the data generator's noise and decoy operators. Candidates come from word TF-IDF, exact keys and character 3-gram retrieval. LightGBM models then score pairs with label-free features that separate true-match noise (typos, truncated digits, fillers, aliases) from decoys (shifted house numbers, inserted words, swapped legal forms), and an expected-F0.5 decision rule turns the probabilities into per-entity answers. The key innovations are label-free "decoy-marker" statistics learned from the candidate graph itself (which transfer to France, a country with no training labels) and pair-local generator-operator features in a final correction model. A fine-tuned multilingual-e5-small cross-encoder adds a text-pair score on the uncertain band, and a meta-model combines the main pipeline with a second model family (an in-country LightGBM stack over its own candidates, stage-2 models and e5 cross-encoder scores) on the union of both candidate sets. France, which has no labels, uses a logit blend of the two families with a decoy-word veto.

---

## 2. Methodology

### 2.1 Problem Analysis
- Each Source-1 record matches 0 to many pool records (sources 2/3), and each pool record has at most one owner, never across countries. Train: 2.21M S1, 10.3M pool records, 3.46 matches per S1. Test: 1.73M S1, India 47%, US 38%, **France 15% with no training labels**.
- True-match noise: typos inside tokens, truncated digits (1995 vs 995), added fillers (center, services, the, inc), alias markers (aka, fka, dba, formerly), website forms, token reorder and splits, acronyms, native-script (Indic) names, addresses that drop numbers, gain state names or codes, or become "None".
- Unmatched pool records ("decoys") are near-copies of real businesses: house number shifted by +1, 2, 3, 4, 5, 7, 9, 11, 13 or 21; one name word inserted from a per-country list; a swapped legal form; a changed unit number.
- Test has about 1.9x the decoy density of train, and decoys come in clusters that share one perturbed house number. So any feature that looks at an S1's other candidates (coherence, sibling agreement) wins in CV and emits decoys on test; we measured this three times and banned such features. Every feature of the main pipeline compares one pool record with its own S1; competition features only ask how strongly other S1 claim the same pool record. The second model family's probabilities are out-of-fold on its own entity folds and enter the meta-model only as inputs.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier (multi-stage LightGBM with a retrieval route, residual corrections, a cross-encoder score and a meta-model over two model families)  
**Core Innovation:** label-free decoy-marker scores and generator-operator features that model how the synthetic noise was produced, with a country-level scale correction so the same features work on the unlabelled country.

Validation: 5-fold CV over S1 (stratified by country and match count), every stacked input of the main pipeline out-of-fold, and a change is kept only if it gains more than 2 standard errors and in at least 4 of 5 folds; before any upload, a label-free test guard compares emitted pairs per S1, empty-answer rate and house-conflict share per country.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** per country, word TF-IDF (hashed name + address 1-2 grams) top-40 neighbours; exact keys on house + street, space-free name, sorted space-free name, name + house, initials + house (acronyms) and alternate ("formerly") name, skipping oversized key blocks; plus a character 3-gram TF-IDF retrieval route (top 40 per view over core name and normalized address) whose new pairs are scored by a separate classifier and kept at p > 0.001. Normalization first maps native-script tokens to Latin with a dictionary learned from training pairs.
- **Candidate pairs generated:** about 98M train and 80.5M test pairs before filtering (about 46 per S1); the final model's band, written as candidate_pairs.tsv, is 12.6M test pairs (7.25 per S1).
- **How you ensured true matches were not lost:** the candidate union holds 98.1% of all true train pairs (oracle macro F0.5 0.994). Oversized key blocks are skipped rather than truncated by row order. The character route adds matches that word retrieval misses. The final candidate band (stage-2 probability >= 0.001 plus route pairs and the second model family's pairs) keeps 99.99% of the true pairs that the full set contains; for the band model, deciding on the band gave decisions identical to deciding on the full set.

---

## 4. Matching Model

**Features used:**
- Name features: fuzzy ratio / token-set / token-sort / partial / Jaro-Winkler on several name views (core, website-free, space-free, sorted, initials, alternate name, conservative raw form); IDF-weighted shared and one-sided name-token mass and weighted cosine; character 3-gram cosine; label-free name-frequency counts; legal-form relations; decoy-marker scores for added, dropped and swapped words ((n_d + 1)/(n_d + n_0 + 10), counted over candidate pairs with a decoy house shift versus equal houses, lifted per country); name edit operations (typo position, doubled letters, confusable characters, acronyms, prefix drops, reorder / join / split, fillers, honorifics, alias and web markers).
- Address features: token-set and partial scores; IDF-weighted address-token mass; house, street, postal and unit equality or conflict; house relations (suffix / dropped leading digits, log difference, near difference on the same street, digit reorder, house among the other side's numbers, signed shift and membership in the decoy shift set); number-multiset operation codes; a French house-number fallback parser.
- Other: graph ranks and gaps from both sides of the candidate set; stage-2 competition features from out-of-fold probabilities (pair rank and margin among the S1 that claim the same pool record, and among the S1's own candidates); cross-fitted lexical likelihoods of added / dropped words and character edits (42 features).

**Model type:** LightGBM throughout: stage 1 (127 leaves, 600 rounds), stage 2 with competition features, a band model on pairs with stage-2 p >= 0.001 using every feature family, the retrieval-route classifier, a lexical correction and a final correction boosted from the previous probability (255 leaves, 800 rounds) using generator-operator features and pool-side competition features; then a small LightGBM stack of that probability with a cross-encoder logit (multilingual-e5-small fine-tuned on training pairs, two halves split by folds so every training score is out-of-fold) on the uncertain band; finally a cross-fitted LightGBM meta-model over the union of these pairs and the pairs of the in-country stack model family (its out-of-fold probabilities and e5 cross-encoder scores, plus pool-side competition of both probabilities). France: logit blend (weight 0.5) of both families, with pure decoy-word pairs set to 0.  
**Threshold selection method:** expected-F0.5 decision per S1: each pool record goes to its highest-probability S1; per S1 the prefix k maximizing 5·Σp(top k)/(Σp + 4k) is kept unless the probability of no match is larger; then pairs with p <= 0.6 are dropped (floor tuned by paired CV).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 5-fold CV on all 2.2M train S1 (US + India; France has no labels): 0.99046 (US 0.99071, India 0.99009). The training candidates do not contain the retrieval-route pairs that the test candidates have, so part of the meta-model's +0.0027 CV gain over the single-family pipeline is recall that the route already provides on test; we expect about +0.001 on the leaderboard. Public leaderboard 0.985 for the single-family pipeline before the meta-model.
- **Common false positives (wrong merges):** decoys that keep the name and address but shift the house number by a small typical step, or insert a generic business word; legal-form swaps; unit-number changes on the same building. Out-of-fold false-positive pairs fell from 13,043 (band model) to 7,904 (final correction) and 7,373 (meta-model), and missed true pairs from 302,550 to 214,404.
- **Common false negatives (missed matches):** pairs never retrieved by blocking (the largest remaining source); true pool records claimed by another S1 with a near-identical name; S1 with an empty or "None" address whose name is shared by other S1 (no distinguishing evidence); records with one true match, which score lowest.

---

## 6. Conclusion
Modelling the generator, rather than generic string similarity, drove most of the gains: decoy-marker and generator-operator features lifted CV from 0.974 to 0.988 and the leaderboard from 0.963 to 0.985. The most important lesson was to validate every idea against the test's higher decoy density with a label-free guard, since features that exploit an entity's other candidates look excellent in CV and fail on test.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` contains `src/` (all pipeline modules), `attempts/` (the settings of every stage used, one folder per stage), `config.yaml`, `README.md` and `requirements.txt`. The README lists every command in order, from converting the organisers' TSVs to parquet through normalization, blocking, pair features, the stage models, the retrieval route, the lexical and final corrections, to the prediction step that writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. Run on 12-vCPU / 96 GB Linux machines with Python 3.12; the cross-encoder steps ran on Kaggle GPUs (2x T4), and their scripts and settings are included (src/ce, src/stack).

### B. Additional Results
| Stage | 5-fold CV macro F0.5 | Public LB |
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
| + meta-model with the in-country stack model family, France blend | 0.99046 | this submission |

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
