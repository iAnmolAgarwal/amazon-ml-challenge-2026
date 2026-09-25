You are a senior engineer who has won large-scale entity-resolution competitions. Audit my current position and design the single best NEXT ITERATION: which improvements to combine and compute together, in what order, for the highest public/private leaderboard gain in the next submission. Be quantitative, rank by expected gain per hour, say what is measured and what is guessed, and flag risks, especially test-only shifts and France. The owner's stated target is 0.99+ macro F0.5 on the leaderboard in the next submission. First judge whether that is reachable, what the realistic ceiling is, and what it would take. No praise, no summary of my text.

# 1. Task
Amazon ML Challenge 2026, business entity resolution. For each Source 1 (S1) record, list all matching Source 2/3 ("pool") records.
- Fields: entity_id (the prefix gives the source), business_name, business_address, country. No other columns.
- Metric: F0.5 per S1, macro-averaged; a singleton scores 1 only for an empty prediction.
- Rules: no external data or lookups. Models must be MIT/Apache, at most 8B parameters. Code must be reproducible.
- Data: train 2.21M S1 and 10.32M pool; test 1.73M S1 and 9.97M pool. Test is India 47% / US 38% / France 15%; France has no training labels.
- Structure: each pool record matches at most one S1, never across countries. 5.6% of S1 are singletons; 3.46 matches per S1 on average.
- Test has about twice the near-duplicate decoys per S1 of train (pool per S1 5.76 vs 4.68).
- The noise is synthetic, from a fixed operator set:
  - native-script names (Indic, word-by-word transliteration of a small vocabulary) and native-script state names in addresses;
  - website form, token reorder, acronyms, prefixes, "formerly"/DBA names, word splits;
  - ~20 house-number templates; state codes and city aliases; "None" addresses.

# 2. Current pipeline (submission 4, public LB 0.963)
1. **Normalization.** Multiple views: core name, space-free, sorted, initials, alternate name, normalized address, house number, street, postal, numbers. Also:
   - France-labelled rows get French street-type and particle rules;
   - a native-script to Latin token dictionary learned from training gold pairs (1,347 tokens to 147 Latin words, positional alignment, fold-stable) is applied before all views;
   - Latin spelling-variant canon on India names (laxmi/lakshmi, jai/jay, sree/shree, shiv/shiva).
2. **Blocking per country.** Word TF-IDF top-40 (hashed word 1-2 grams of name + address) unioned with 6 exact keys (house+street, space-free name, sorted name, name+house, initials+house, alternate name). About 44 candidates per S1.
3. **Stage-1 model.** LightGBM (127 leaves, lr 0.05, 600 rounds) on about 48 pair features: rapidfuzz scores on several name views, address token-set, house/street/postal equality and conflict, cosine rank and gap from both sides, candidate counts, label-free name frequencies, conservative-name similarities. Trained on 50% of entities in CV and 100% for the test prediction.
4. **Stage-2 model.** Stage-1 features plus competition features from out-of-fold stage-1 probabilities:
   - pool side: rank, best other S1 p, margin, number of competitors, sum of p;
   - S1 side: rank, gap to the best candidate, best other candidate, count with p >= 0.5, sum of p.
5. **Decision.** Each pool record goes to its best S1. Per S1 choose the prefix k maximizing expected F0.5 (5·Σp/(E[T]+4k)) against k=0 (Π(1−p)), then drop pairs with p <= 0.6.
   - The ratio rule (predict non-empty only if best > 2.5·P(empty)) was worse on stage-2 probabilities (−0.00015, 0/5 folds).

# 3. Measured results (5-fold OOF CV, stratified by country × match count; paired rule: mean fold diff > 2 SE and ≥ 4/5 folds)
| Step | CV macro F0.5 | Paired gain | Public LB |
|---|---|---|---|
| baseline | 0.94292 | – | 0.929 |
| + name frequency, conservative names | 0.96050 | +0.01758 | 0.949 |
| + 50% training | 0.96109 | +0.00059 | – |
| + native-script dictionary, consistent train/test normalization | 0.97008 | +0.00899 | – |
| + floor 0.6 | 0.97083 | +0.00076 | – |
| + stage-2 competition features (submission 4) | 0.97423 | +0.00340 | **0.963** |

- LB minus CV is steady at about −0.012 across three scored submissions (Spearman +1.0).
- Submission 4 by country (CV): India 0.9721 (blocking oracle 0.9918), US 0.9757 (oracle 0.9954). Overall oracle 0.99398; pair precision 0.9932, recall 0.9399.
- F0.5 by true-match count t (oracle in brackets; mean k−t):

| t | F0.5 | oracle | mean k−t |
|---|---|---|---|
| 0 (singletons) | 0.9733 | [1.0] | – |
| 1 | 0.9151 | [0.981] | −0.04 |
| 2 | 0.9698 | [0.993] | – |
| 3 | 0.9777 | [0.994] | −0.15 |
| 4 | 0.9802 | [0.995] | – |
| 5 | 0.9811 | – | – |
| 6+ | 0.9815 | [0.995] | −0.41 |

  Many-match entities are still under-predicted.
- **Implied France.** If US and India score on test as in CV: (0.963 − 0.38·0.9757 − 0.47·0.9721)/0.15 ≈ **0.90**. France is the largest gap.
- Label-free test profile of submission 4:

| | matches per S1 | empty rate | house-conflict among emitted pairs |
|---|---|---|---|
| France | 3.45 | 5.0% | 1.8% |
| US | 3.33 | 5.9% | 7.8% |
| India | 3.32 | 5.9% | 16.8% |

  Train OOF comparison: US 7.1%.
- **A failure we caught:** a "coherence" side-car (agreement of each candidate with the S1's other strong candidates) gained +0.0024 in CV but doubled the test US house-conflict rate (7.0% → 14.7%), i.e. it emitted decoys on test. It was removed. Lesson: test is decoy-denser than train, so check every change with this label-free guard.

# 4. Candidate improvements available now
Status is built, measured or idea; gains are estimates unless marked measured.
| # | Item | Status | Expected gain |
|---|---|---|---|
| a | Blocking v3: reverse retrieval (each pool record's top-3 S1) + exact keys on consonant skeleton, skeleton+house, first|last token | built, unit-tested; rebuild and CV running tonight (~19:00) | retrieval +0.001-0.003 |
| b | Skeleton fuzzy features (ratio / JW on skeletons, skeleton char n-gram cosine) | not coded | +0.0005-0.002 |
| c | 171 test-only native business words (motors, stores, bakery, …; 14% of test native names unmapped): label-free crude transliteration + nearest frequent test-S1 word | validated 91% precision on held-out known words | India +0.001-0.003 if precision ≥ 97% |
| d | France: label-free diagnosis (legal forms SARL/SAS/SASU/EURL/SCI/Ets/Sté/Cie, accents, cedex, street-type order, "&"/"et"), targeted normalization fixes; leave-one-country-out (train US → score India and reverse) to tune decision strictness for an unseen country | idea | France +0.01-0.05 → LB +0.0015-0.0075 |
| e | Decoy-robust coherence (anchors require exact house number; no decoy mutual support) + decoy-dense validation (remove ~19% of S1 so their pool records become decoys, recompute rival features) | idea | uncertain |
| f | Stage-2 variants: more rounds / lower lr, 100% training in CV, stage-3, or a second model family (e.g. CatBoost/XGBoost) blended | idea | +0.0005-0.002 |
| g | Under-prediction for many-match entities (k−t = −0.41 at t≥6): cardinality features or learned E[T] | idea | +0.0005-0.002 |
| h | Gated cross-encoder (Multilingual MiniLM, MIT) on pairs with p in [0.05, 0.95] (~2.1M train pairs, 86% of the classification ceiling) | idea; no AWS GPU, Kaggle T4 possible | +0.001-0.0035, 10+ h |
| i | Train/test calibration shift: test decoy density ~2x; prior-shift correction of stage-2 logits, or stricter floor for the test only | idea; not validatable by ordinary CV | ± |
| j | Blend with an independent parallel pipeline (OOF 0.9754, India 0.9710, US 0.9784; different blocking with ~22 candidates per S1, two-stage LightGBM, floor 0.6) | would need its OOF probabilities on our folds; its test predictions exist | +0.001-0.004 |

# 5. Constraints
- About 56 hours left; 12 submissions left (2 today, 5 per day after).
- Compute: one on-demand 8-core/64 GB machine (spot capacity was reclaimed 8 times today, so do not count on a second machine). A full rebuild (normalize + block + features) takes about 1.2 h per split; a 5-fold CV about 55 min; a test prediction about 30-45 min. Stage-2 CV peaks at 61.7 of 64 GB RAM.
- No GPU on AWS; Kaggle T4 possible.

# 6. What I want
1. Is LB 0.99+ reachable next submission? If not, the realistic next-submission LB and the realistic final LB, with the arithmetic by country (France especially).
2. The best combined next iteration:
   - which items (from §4 or anything better) to compute together in ONE rebuild + CV + predict cycle;
   - which must stay separate so their effect can be attributed or they can be killed;
   - an order that fits the single-machine budget (hours);
   - expected gain per item and in total;
   - the go/no-go check for each (paired CV plus the label-free test house-conflict guard).
3. For France specifically: how to diagnose and fix an unseen country without labels in a few hours, and how to validate it (leave-one-country-out protocol, label-free test checks). Give concrete normalization rules you expect for French business names and addresses in a generated dataset like this.
4. Any cheap high-value idea I have missed.
