# Business Entity Resolution: Approach Reproduction Guide

Purpose: give another engineer or agent everything needed to rebuild our leaderboard solution from the raw data, using the approach only (no code). Our public leaderboard history, which gives checkpoints to verify a rebuild step by step:

| Checkpoint | What it adds | Our 5-fold CV (macro F0.5, train) | Public LB |
|---|---|---|---|
| A | normalization, blocking, one LightGBM | 0.943 | 0.929 |
| B | + name-frequency and conservative-name features | 0.961 | 0.949 |
| C | + native-script dictionary, 0.6 floor, second-stage competition features | 0.974 | 0.963 |
| D | + third-stage correction with 34 pair features | 0.983 | 0.974 |
| E | + label-free decoy-marker name features | 0.985 | 0.981 |
| F | band model with every feature, country-lifted markers, French house parser, character retrieval, lexical correction | 0.986 | **0.98393** |

Reproduce in this order and check each checkpoint's CV before moving on. The CV numbers are on the full training set with the fold scheme below; the LB is roughly 0.002-0.015 below CV, because test has more decoys and a third, unlabelled country.

---

## 1. The task and the data facts that matter

- For every Source-1 (S1) record, output every pool record (Sources 2 and 3) that is the same business. Records have only: entity id (its prefix gives the source), business name, business address, country.
- Metric: F0.5 per S1, then averaged over S1 (macro). An S1 with no true match scores 1 only for an empty prediction. F0.5 weights precision more than recall.
- Structure: each pool record belongs to at most one S1, never across countries. Train: 2.21M S1, 10.3M pool records, about 3.46 matches per S1, about 5.6% of S1 with no match. Test: 1.73M S1, about 10M pool records; India 47%, US 38%, **France 15% with no training labels at all**.
- The data is synthetic, from a fixed noise generator. Understanding it is the key to the score:
  - **True-match noise:** typos inside tokens, truncated digits (1995 vs 995), filler words added (center, services, the, inc), alias markers (aka, fka, dba, formerly, née, "ID: nnn"), website forms of names, token reorder, word splits and joins, acronyms, prefixes, native-script (Indic) versions of names and state names, addresses that drop numbers, gain state names or codes, or become "None".
  - **Decoys (pool records with no owner) are near-copies of real businesses:** the house number shifted by +1, 2, 3, 4, 5, 7, 9, 11, 13 or 21; or one name word inserted from a per-country list (US: group, holdings, associates, ventures, north/east/south/west, downtown, lakeside...; India: industries, ventures, public, group, exports, overseas, infratech...; France: participations, holding, distribution, associés, international, développement, groupe, et fils); or the legal form swapped (llc and inc, private and public, sarl and sas/sci...); or a unit/floor number changed while the house stays.
  - **Test has about twice the decoy density of train**, and France is the densest. Every design choice must hold up under more decoys than CV shows.
- **The trap that cost us most:** never use features that look at an S1's other candidates (coherence with its other strong matches, agreement among siblings, source balance, context stackers). On test, decoys come in clusters that share the same perturbed house number, so such features learn "shared deviation means true" on train and then emit decoys on test. They gain in CV and lose on the leaderboard. Every feature must compare one pool record with its own S1 only. The one allowed exception is competition features (section 5), which ask how strongly other S1 claim the same pool record, or how a pair ranks among its S1's candidates. They do not reward agreement between candidates.

## 2. Validation protocol (do this first)

- 5 folds over S1, stratified by country and by the number of true matches; all pairs of an S1 stay in its fold.
- Every model used as an input to a later model must be out-of-fold: each train pair is scored by a model that never saw its S1.
- A change counts only if its mean per-fold gain over the parent is more than 2 standard errors and it is positive in at least 4 of 5 folds (paired on the same folds).
- For fast iteration, use a fixed 10% sample of S1 (with all their candidate pairs), then confirm on full data.
- Before trusting any change for the leaderboard, run a label-free test guard: per country, the emitted pairs per S1, the share of S1 with an empty answer, and the share of emitted pairs whose two house numbers are both present and different. If test moves much more than train, the change is emitting decoys.

## 3. Normalization (checkpoint A onward)

Build several views per record side by side; never collapse a record into one destructive string.
- **Native-script dictionary:** learn native-script-to-Latin token maps from the training gold pairs. Names are transliterated word by word, so align tokens by position in gold pairs with the same token count, and fall back to co-occurrence (Dice) otherwise; addresses use co-occurrence only. Keep a mapping with at least 5 distinct S1 owners and a target share of at least 0.5. Latin spelling variants that compete for the same native tokens are merged into one canonical spelling (India names). About 1,350 native tokens map to about 150 words. Latinize before every other step.
- **Name views:** latinized raw name; core name (lowercased, accents folded, punctuation removed, legal words and website parts removed); website-free name; space-free join key in original and sorted token order; initials; alternate name from "formerly"/DBA/aka markers.
- **Address views:** normalized address; house number; street tokens; all numbers in the address; postal code; units (suite, apt, floor) removed for the house parse.
- **France rules:** French street types and particles, remove postal-box references (BP, CS, TSA, CEDEX nnn); a French house-number fallback is added at checkpoint F (section 7).

## 4. Blocking (candidate generation)

Per country:
- Word retrieval: TF-IDF over hashed word 1-2 grams of name plus address; for each S1 keep its top 40 pool records by cosine.
- Exact keys: house plus street token; space-free name; sorted space-free name; name plus house; initials plus house (acronyms); alternate name. Skip any key value whose block is too large on either side (never truncate by row order).
- The union gives about 45 candidates per S1 (about 98M train pairs). Measure the oracle F0.5 of the candidate set (the score if every candidate were classified perfectly); ours is about 0.994.

## 5. Stage 1 and stage 2 models (checkpoints A-C)

- **Stage 1 pair features (about 48):** fuzzy name scores (ratio, token set, token sort, partial, Jaro-Winkler) on several name views; address token-set and partial scores (set to -1 when either address is empty); house, street and postal equal or conflicting; numbers overlap; acronym and join-key equality; share of native-script characters; graph features from both sides (the pair's TF-IDF rank and gap among the S1's candidates and among the pool record's S1s, candidate counts).
- **Name frequency and conservative names (checkpoint B):** how many S1 and pool records in the same country share the S1's space-free name, and the same for the pool record's name (label-free counts per split); fuzzy scores on conservatively normalized raw names (keep legal words and digits).
- **Model:** LightGBM, binary, 127 leaves, learning rate 0.05, 600 rounds, min data in leaf 200, feature and bagging fraction 0.8, L2 1. No ids, row positions or country as features.
- **Stage 2 (checkpoint C):** from stage-1 out-of-fold probabilities, competition features for each pair. On the pool side: the pair's rank among all S1 that retrieved this pool record, the best other S1's probability, the margin, the number of other S1 above 0.1, and the sum. On the S1 side: rank, gap to the best, best other, number above 0.5, and the sum. Retrain the same LightGBM on stage-1 features plus these. Test uses stage-1 test probabilities from a model trained on all training data.

## 6. Decision rule (from checkpoint C)

1. Each pool record goes only to its highest-probability S1.
2. Per S1, sort its remaining pairs by probability. For k ≥ 1, the expected F0.5 is about 5·(sum of the top k probabilities) / (sum of all its probabilities + 4k); with k = 0 it is the product of (1 - p). Keep the best k.
3. Drop kept pairs with probability ≤ 0.6.

This rule measured as optimal for calibrated probabilities; we found no decision tweak that helps.

## 7. Pair features for the generator (checkpoints D-F)

All are label-free and pair-local (one pool record against its S1), with statistics computed per split and per country.
- **Name specificity:** token weights 1 + ln((N+1)/(df+1)) over core-name tokens, with df over the country's S1. Features: shared, S1-only and pool-only weight mass, their fractions, the largest one-sided weight on each side, weighted cosine, and counts of one-sided tokens.
- **Address specificity:** the same over address tokens.
- **House relations:** one house number is the suffix of the other (dropped leading digits); log of the absolute difference; difference ≤ 10, also restricted to the same street; edit distance; same digits reordered; each side's house among the other side's numbers; the signed difference (pool minus S1); and whether it is one of the decoy shifts {1,2,3,4,5,7,9,11,13,21}.
- **Units:** equal, conflicting, or on one side only.
- **Legal forms:** equal, nested, disjoint, or on one side only.
- **Character 3-gram cosine** of the space-free names.
- **Decoy-marker scores (checkpoint E), label-free:** for each word, count over all candidate pairs how often the word is added to (or dropped from, or swapped in) the pool name while the pool house is the S1's house shifted by a decoy shift (n_d), versus while the houses are equal (n_0). Score = (n_d + 1)/(n_d + n_0 + 10). Features: number of added and of dropped words, the max score over added words, the max over dropped words, and the score of the one-for-one swap. Train uses tables built from train pairs; for France, which has no train data, build the table from France's own test pairs.
- **Country lift (checkpoint F):** replace each marker score with logit(score) minus log(Σn_d / Σn_0) of its country and kind. This puts France's test-built table on the same scale as US and India. It is neutral within train countries and large for an unseen, decoy-denser country.
- **French house parser (checkpoint F):** when the normal parser finds no house number in a French address, take a 1-4 digit number (optionally followed by bis/ter/quater or a letter) that precedes a French street type. This includes abbreviations the standard list misses (ch, crs, cr, cour, q, blvd, bvd, parvis, esplanade, lotissement, chaussée, mail, venelle, ruelle, sente, digue, corniche, grande rue, rond-point). Recompute the house features for France pairs with it.

**Checkpoint D model:** a LightGBM correction boosted from logit(stage-2 probability) as its starting score, using the pair features plus logit(p), on pairs with stage-2 p ≥ 0.001 only; all other pairs keep their stage-2 probability. Settings: 31 leaves, learning rate 0.05, 300 rounds, min data 200, L2 1.

**Checkpoint F model (replaces D's correction):** a fresh LightGBM trained, without a starting score, on the same band of pairs (stage-2 p ≥ 0.001, about 13% of pairs holding essentially all true matches), using every feature family (stage 1, frequency, competition, pair features with the French parser, lifted markers). Settings: 127 leaves, learning rate 0.05, 600 rounds. This was +0.0009 over the residual design.

## 8. Character retrieval route (checkpoint F)

Some true matches are never retrieved by word TF-IDF.
- Per view (core name; normalized address), fit character-3-gram TF-IDF on the pool: terms in at least 2 pool records, sublinear tf, smoothed idf, L2-normalized rows. For each S1, take the 40 nearest pool records of its own country per view.
- New pairs = retrieved pairs not already candidates. Build the same pair features for them.
- Score them with a separate LightGBM (66 features: basic string, number and frequency features plus the pair features of section 7; 63 leaves, 300 rounds) trained on existing train candidate pairs of a label-independent 10% owner sample. No probability or rank inputs.
- Keep new pairs with p > 0.001 (about 0.9 per test S1) and add them to the candidate set with that probability. Gain: +0.0013 CV.

## 9. Lexical correction (checkpoint F)

- For each pair, six token channels from the raw records: name words added, name words dropped, address words added, address words dropped, the legal-word transition, and character edit operations between the two names (with one character of context).
- For each token, count distinct S1 owners among true matches and among non-matches in the training pairs. A token seen at least 5 times scores log((pos+0.5)/(owners of positives+1)) - log((neg+0.5)/(owners of negatives+1)), shrunk by n/(n+10).
- Each channel is summarised by 7 numbers (count, max, min, sum, mean, share of unseen tokens, max support): 42 features.
- Cross-fit the counts on train, so each pair's features come from other folds only. Train a small LightGBM correction (100 rounds, 9 leaves, learning rate 0.04) boosted from logit(p) on pairs with stage-2 p ≥ 0.01. Test uses counts from all train pairs. Gain: +0.0002-0.0003 CV. It also helps France.

## 10. Candidate file and final checks

- Write the candidate file as exactly the set the final model decided on: pairs with stage-2 p ≥ 0.001 plus the retrieval pairs. That is about 7 per S1 instead of 46, with the same matches, and smaller candidate sets rank higher under the organisers' rules. Every emitted match must be inside the candidate file.
- Run the organisers' validator with id checks on both files; every test S1, including France, must appear.
- Guard values for the final build: emitted pairs per S1 of about 3.26 (France), 3.33 (India), 3.36 (US); empty share about 5.8-5.9%; house-conflict share of emitted pairs about 1.0% (France), 17% (India, where the parsed house is often a template number), 8.6% (US).

## 11. What did not help (skip these)

- Coherence / sibling / context features, and a hard house-shift veto (hurts India). Mapping leftover native shop words: they occur only on decoys, so mapping them removes a perfect decoy signal.
- Decision tweaks (thresholds, calibration, joint assignment, other k rules), more training data (5x data +0.0007), naive synthetic positives or pseudo-labels on the target country, France mass calibration once the lift and parser are in place, and switching the lexical layer off for France.

## 12. Where the remaining loss is (for going beyond 0.984)

- Missed matches dominate (pair precision 0.998, recall 0.960). S1 with exactly one true match score lowest.
- About 0.0033 of CV comes from empty-address S1 whose names are identical to other S1; they carry no distinguishing evidence.
- A residual model on generator-operator features (number operation codes, name edit operations, confusable characters, legal-form ids, reorder/join, fillers, honorifics, alias and web markers, address-word differences), boosted from the final probability on pairs with 0.003 ≤ p ≤ 0.997 for US/India, adds +0.0008 CV (verified on full data, test guard clean).
