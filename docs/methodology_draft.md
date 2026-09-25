# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [registered team name]  
**Team Members:** [registered members]  
**Submission Date:** [date]

---

## 1. Executive Summary
Per-country blocking (word TF-IDF top-40 plus exact keys) feeds a LightGBM pair classifier. Its decisions are made per Source 1 entity by maximising expected F0.5 under the one-owner-per-record constraint. The main innovations are:
- a native-script to Latin token dictionary learned from aligned training pairs, which makes Indian-script names directly comparable;
- a second-stage model that sees each candidate's competition (the record's best rival entity, the entity's other candidates);
- a probability floor chosen to stay precise on a test set that is denser in near-duplicate decoys.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Data:** train 2.21M Source 1 entities and 10.32M Source 2/3 records; test 1.73M and 9.97M. Test adds France (15%), which is absent from training.
- **Matching structure:**
  - each Source 2/3 record matches at most one Source 1 entity, and never across countries;
  - 5.6% of entities are singletons; 3.46 matches per entity on average;
  - 26% of training pool records match nothing (decoys).
- **Noise comes from a small, fixed set of operators, each with a small vocabulary.** Rates per gold pair:
  - addresses: uppercased 30%; native-script address 9.0%; house numbers in ~20 templates (#N, no N, door no N, h.no N, plot no.); state codes vs names vs native script; city aliases; "None" addresses;
  - names: native-script name 7.4%; case-only change 5.9%; website form 5.2% (concatenated lowercase + ".com"); token reorder 5.0%;
  - also: acronyms, honorific and junk prefixes, "formerly"/DBA names, words split by a space ("Li mited").
- **Native-script names are word-by-word transliterations of a small vocabulary.** 1,347 native tokens map to 147 Latin words.
- **Test has about twice the decoys per entity of training** (pool/entity 5.76 vs 4.68). The decoys are near-duplicate businesses on the same street with nearby house numbers.

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage gradient-boosted classifier + expected-F0.5 decision  
**Core Innovation:**
- a learned native-script dictionary and operator-inverting normalisation;
- competition-aware second-stage features;
- decisions optimised for macro F0.5 per entity, with a precision floor validated against decoy density.

---

## 3. Candidate Generation (Blocking)
- **Normalisation:** several side-by-side views rather than one destructive string:
  - core name (legal forms, honorifics and junk removed; website form undone; split words re-joined; digit-for-letter undone);
  - space-free and sorted keys, initials, and the alternate ("formerly") name;
  - normalised address, house number (template-aware), first street word, postal code;
  - French street-type and particle rules applied to France-labelled rows.
  Learned native-script tokens are replaced by their Latin form first, and Latin spelling variants on Indian names are mapped to one canonical form.
- **Blocking keys used:**
  - word-view TF-IDF (hashed word 1-2 grams of name + address, sublinear, fitted on the pool), exact top-40 cosine neighbours per entity, chunked sparse top-n;
  - exact keys: house+street, space-free name, sorted space-free name, name+house, initials+house, alternate name. Keys shared by more than 50 records are skipped.
- **Candidate pairs generated:** about 98M train and 79M test (about 44 per entity).
- **How you ensured true matches were not lost:**
  - blocking recall is measured on training labels (oracle macro F0.5 0.990 before the dictionary);
  - the exact keys cover what TF-IDF misses (acronyms, alternate names, reordered names);
  - the dictionary recovers native-script matches that no Latin-text view can retrieve.

---

## 4. Matching Model

**Features used:**
- Name features:
  - rapidfuzz ratio / token_set / token_sort / partial / Jaro-Winkler on the core and space-free names, and on lightly normalised raw names;
  - acronym and alternate-name matches;
  - label-free name frequency (how many entities and records share the name).
- Address features: token-set similarity; gated ratio and partial ratio; house number equal / prefix / missing; street and postal equality and conflict; number Jaccard.
- Other:
  - graph features: TF-IDF cosine and its rank and gap from both sides; candidate and rival counts;
  - source, website, empty and non-Latin flags;
  - second stage: out-of-fold stage-1 probability, the record's best rival entity and margin, number of competitors, sum of probabilities, and the entity's rank, gap to its best candidate and expected match count.

**Model type:** LightGBM binary classifier (127 leaves, learning rate 0.05, 600 rounds), two stages (stage 2 stacks on out-of-fold stage-1 probabilities). MIT licence, far below the parameter limit. No external data or pretrained models.  
**Threshold selection method:**
- each Source 2/3 record is kept only for its highest-probability entity;
- per entity, the number of matches k maximises the expected F0.5, 5·Σp₁..ₖ / (E[T] + 4k), compared against predicting nothing (Π(1−p));
- selected pairs must also exceed a probability floor of 0.6, chosen on out-of-fold data and checked on a decoy-dense simulation.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [final 5-fold out-of-fold score] (5-fold CV stratified by country × match count; every entity scored by a model that never saw it; singletons included).
- **Common false positives (wrong merges):** near-duplicate decoys: the same street with a nearby house number and a name differing by one word ("… Deli" vs "… Hair Studio").
- **Common false negatives (missed matches):** records whose name was replaced by an unrelated "formerly" name; typo-heavy names with empty addresses; native-script names containing words outside the learned vocabulary.

---

## 6. Conclusion
The generated noise is largely invertible, so explicit normalisation, a learned native-script dictionary and precise blocking recover most matches. The remaining errors are decisions among near-duplicates, which competition-aware second-stage features and a precision floor reduce. A label-free check of emitted pairs on test (house-number agreement against training) caught a feature that validated well but inflated decoy matches on test.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/`:
- `prepare_raw.py`: TSV to parquet;
- `er_dict.py`: native-script token maps;
- `er_norm.py`: normalisation views;
- `er_block.py`: candidate generation;
- `er_feat.py`, `er_feat2.py`, `er_feat4.py`: pair and competition features;
- `er_model.py`: model and decision;
- `er_eval.py`: 5-fold CV;
- `er_predict.py`: test inference writing both output files;
- `er_package.py`: validation and packaging.

`README.md` lists the exact command sequence; `requirements.txt` pins versions (Python 3.12, x86-64 Linux, 64 GB RAM).

### B. Additional Results
[CV table by attempt; per-country scores; loss decomposition]
