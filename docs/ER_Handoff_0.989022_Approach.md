# Business Entity Resolution: Reproduction Guide for the 0.989022 Solution

Purpose: everything needed to rebuild our best leaderboard solution (public LB **0.989022**, upload 13, 27 Sep 2026) from the raw data. It describes the method, settings and checks only (no code). Rebuild in checkpoint order and confirm each checkpoint's CV before moving on.

| Checkpoint | What it adds | 5-fold CV macro F0.5 (train, US + India) | Public LB |
|---|---|---|---|
| A | normalization, blocking, one LightGBM | 0.943 | 0.929 |
| B | + name-frequency and conservative-name features | 0.961 | 0.949 |
| C | + native-script dictionary, 0.6 floor, second-stage competition features | 0.974 | 0.963 |
| D | + third-stage correction with 34 pair features | 0.983 | 0.974 |
| E | + label-free decoy-marker name features | 0.985 | 0.981 |
| F | band model with every feature, country-lifted markers, French house parser, character retrieval, lexical correction | 0.98604 | 0.98393 |
| G | + generator-operator correction (first version) | 0.987 | 0.985 |
| H | final correction: generator-operator + pool-side competition features (section 10) | 0.98742 | - |
| I | + cross-encoder score stacked on the uncertain band (section 11) | 0.98779 | - |
| J | + meta-model over our pipeline and a second model family, France blend (sections 13-15) | 0.99046 | **0.988787** |
| K | + extra key / reverse-retrieval candidates, also on a 10% train sample so the meta-model learns them (sections 12, 14) | 0.99051 | **0.989022** |

The LB sits below CV for two reasons: test has about twice the decoy density of train, and France (15% of test) has no labels, so it is scored only on the leaderboard. From checkpoint J on, part of the CV gain is recall that test already gets from the retrieval route, because the train candidates lack those pairs; expect the LB to move by about half of the CV gain.

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

## 10. Final correction: generator-operator and pool-side competition features (checkpoint H)

Applies to US and India pairs whose checkpoint-F probability p is between 0.003 and 0.997; all other pairs and all of France keep p.
- **Generator-operator features (about 86, label-free, pair-local):** how the pool record was derived from the S1 record.
  - Numbers: multiset differences, typical decoy shifts, +-1/2, truncated or dropped digits, the operation that turns the S1 house number into the pool one, the share of S1 numbers kept, and first/last number equality.
  - Name edits: the best edit between one-sided tokens (distance, position, digit-for-letter, doubled letter, transposition, confusable characters such as i/l, o/0, s/5), edit-operation counts over the core names, prefix drop or add, reorder, join or split, acronyms, legal-form identities, fillers, honorifics, alias markers, web forms, id tags and native script.
  - Raw cues: exact and case-insensitive equality, upper case, an empty or placeholder address, '#', PO box, fractions.
  - Address words present on one side only; street equality and overlap.
  - Fold accents before comparing.
- **Pool-side competition features:** from p, per pair: rank of p among all S1 that claim the same pool record (capped at 20), the best other S1's p, the margin, the sum of the others' p, and the number of S1 above 0.01 and above 0.5. Also pool-side contrasts for 11 key similarity features (name token-set, name Jaro-Winkler, address token-set, number overlap, house equal, street equal, character Jaro-Winkler, name and address specificity cosine, legal-form equal, added-word marker). Each contrast is the pair's value minus the best value among the other S1 of the same pool record; missing counts as -1. Nothing compares a pair with its own S1's other candidates.
- **Model:** LightGBM started from logit(p) as the initial score. Inputs: logit(p), the pool-side features, an India flag, the generator features, every earlier feature family (except the plain address ratio and partial scores) and the contrasts. Settings: 255 leaves, learning rate 0.05, 800 rounds, min data 300, feature fraction 0.9, bagging 0.8, L2 10. Pairs added by the retrieval route have no earlier features, so they use a second model without them (127 leaves, 400 rounds). Cross-fit on the 5 folds.
- Measured: 0.98742, paired +0.00138 over F (5/5 folds). Out-of-fold false-positive pairs fell from 13,043 to 7,904.
- Performance note: compute "best other S1" features by first computing the pool record's maximum as a column and then windows over it. A nested window expression is quadratic and ran for hours on 98M pairs; the two-step form takes 30 seconds.

## 11. Cross-encoder score on the uncertain band (checkpoint I)

- **Model:** multilingual-e5-small (MIT licence), fine-tuned as a one-logit pair classifier with BCE loss. Each side is its raw "name ; address" string, max length 96. Batch 128, AdamW lr 5e-5, weight decay 0.01, 3% linear warm-up then linear decay, gradient clip 1.0, fp16, one epoch. It took about 170 minutes per model on one T4 GPU (Kaggle, 2 x T4 in parallel).
- **Training pairs:** split S1 into two halves by the CV folds (folds 0-1 and folds 2-4). From each half, take 470k random S1. Per S1, take every true candidate pair, the top 8 non-matching candidates by word similarity and 2 random non-matching candidates, about 6.1M pairs per model. Model A trains on half 0, model B on half 1.
- **Scoring:** the band is pairs with checkpoint-F probability between 0.003 and 0.997 (3.0M train, 2.8M test pairs). Every train pair is scored by the model trained on the other half, so it is out-of-fold. Test is scored by both models and averaged.
- **Stack:** a small LightGBM per fold on [logit(checkpoint-H probability), cross-encoder logit] over band pairs (31 leaves, 300 rounds, min data 500, L2 1); other pairs keep their probability.
- Measured: alone, the cross-encoder is weaker than the tree model (AUC 0.947 vs 0.962 on the band), but the stack gains +0.00037 over H (+0.00106 over F), 5/5 folds. The gain is largest on S1 whose pool records the cross-encoder never saw, so it is not memorisation. A larger model (e5-base, one third of an epoch in the same time) added only +0.00001 on top, so it is not used.

## 12. Extra candidates: exact name keys and reverse character retrieval (checkpoint K)

- **Exact name keys** (per country; a key value is used only if at most 50 S1 and at most 50 pool records carry it):
  - the first 6 characters of the space-free name plus any number of the record;
  - the space-free name plus any number;
  - the first 8 characters of the space-free name;
  - the consonant skeleton of the name;
  - the first letters of the name words;
  - the skeleton plus the house number;
  - any two distinct core-name words;
  - a name word plus the house number (India);
  - a name word plus any number (US).
- **Reverse character retrieval:**
  - View: core name plus " | " plus normalized address.
  - TF-IDF: character 3-grams fitted once on all pool records of the split (terms in at least 2 records and at most 2% of them, sublinear tf, smoothed idf, L2 rows).
  - Queries: pool records whose best probability under checkpoint H is at most 0.5 ("unclaimed").
  - Search: against all S1 of the same country, keeping the 5 best matches.
- **New pairs:** retrieved pairs that are neither candidates nor checkpoint-H pairs. Build the section 8 features for them and score them with the section 8 route classifier as it is (no refit), keeping p > 0.001. On test this adds about 1.85M pairs (1.06 per S1).
- **Train side:** run the same keys and reverse retrieval for a fixed, label-free 10% of train S1 (row index mod 10 = 3). Score them out-of-fold with the route classifier's fold models, and add them to the train file that the meta-model reads (section 14). Without this, the meta-model never sees such pairs in training.
- Measured on the sample: adding them to H gains +0.00222 (5/5; the added emitted pairs are 95% correct in US and 91% in India). On top of the meta-model it still gains +0.00045 (5/5). LB +0.000235.
- Note: for these pairs, a different house number is not a decoy signal. Among the added pairs on train, the true ones have more house conflicts (43-45%) than the false ones (23%), because true matches often have a changed address.

## 13. The second model family (in-country stack)

A second, independently built pipeline gives the meta-model a different view of the same pairs. Its recipe:
- **Candidates and stage 1:**
  - Mine its own transliteration dictionaries from train pairs, then normalise.
  - Block per country with word TF-IDF neighbours (its recorded blocking settings: k_r = 3, k_a = 15, max document frequency 3,000, min similarity 0.1).
  - Build pair and name-specificity features, then train stage-2 LightGBMs over 5 entity folds. Its folds are the CRC32 of the S1 id mod 5, which differ from ours; that is fine, because its outputs enter our meta-model only as inputs.
- **Cross-encoders on its uncertain band (0.02 < p2 < 0.98), Kaggle 2 x T4:**
  - e5-small per country;
  - e5-small in-country, trained by halves (fold parity);
  - e5-base in-country, trained by halves (max length 128, batch 64, lr 3e-5, about 200 minutes);
  - an e5-base cross-country model for France.
  Each is one epoch with fp16, 5% warm-up and a time budget.
- **Stage 3:** a LightGBM stack (127 leaves, min data 100, 800 rounds) over the stage-2 probabilities, the cross-encoder scores, band features, decoy features and operation/pair features, giving the probability q.
- **Kept pairs:** stage-2 p2 > 0.02 and the top 10 per S1.
- **France:** a blend with a France-specific stage-2 model, the cross-country cross-encoder, a decoy stacker, a stricter decision for the unseen country (T = 2, min p 0.6) and a marker veto at lift 1.5.
- **Outputs for the meta-model:**
  - out-of-fold q and p2 for every kept train pair (US, India);
  - test probabilities for kept pairs in all three countries;
  - the in-country e5-small and e5-base cross-encoder scores (held-out on train, per half-model on test).
- Its own in-country CV is 0.98783. Its leaderboard alone was 0.9849.

## 14. Meta-model (checkpoints J and K)

- **Rows:** the union of our pairs with probability >= 0.001 and the second family's kept pairs. Apply its kept-pair rule (p2 > 0.02, top 10 per S1) to its train pairs too, so train and test hold the same kinds of pairs. US and India only.
- **Inputs:**
  - the logit of each family's probability (missing when that family lacks the pair), plus their min, max, sum and difference;
  - the second family's two in-country cross-encoder scores;
  - country;
  - pool-side competition of each probability: the pair's rank among the S1 claiming the pool record, the number of claimers, the number above 0.5, the best other claimer and the margin.
- **Train/test parity for the cross-encoder inputs:** on train each pair's score comes from one held-out half-model. On test, use one half-model per S1 as well, chosen by a hash of the S1 id mod 2. Averaging both models on test would give a smoother distribution than train.
- **Model:** LightGBM, 63 leaves, lr 0.05, 300 rounds, min data 500, feature fraction 0.9, bagging 0.8, L2 10, deterministic mode. Cross-fit on our 5 folds; test is the mean of the 5 fold models.
- Measured: CV 0.99046 (US 0.99071, India 0.99009), paired +0.00267 over I (5/5), +0.00264 over the second family alone and +0.00142 over the best plain logit blend of the two. Most of the gain comes from pairs only one family finds, which a plain blend discards. The test guard held: added pairs per S1 on test were at or below train.
- Honest caveat: our train candidates do not contain the retrieval-route pairs that test has. So 86% of the meta-model's CV gain is recall that, on test, the route and extra keys partly deliver already. Against a train set that already held those pairs, the gain would be about +0.0008. The LB moved +0.0038 from the previous upload, which also includes the France change.

## 15. France rule

- **Our France probability:** from checkpoint H (unchanged since F).
- **The second family's France probability:** its final France output. Its zeros on pairs whose pool name adds "developpement", "france" or "groupe" count as missing; its broad veto kills true pairs that carry these words as fillers.
- **Per France pair** (union of both sets): p = sigmoid(0.5 · logit(ours) + 0.5 · logit(theirs)) when both exist, otherwise the one that exists.
- **Veto:** pairs whose pool name adds "participations", "holding", "distribution" or "international" (and the S1 name lacks it) get p = 0.
- **Decision:** as everywhere, with floor 0.6.
- **Validation:** there are no labels, so the weight came from leave-one-country-out tests (train on one labelled country, evaluate on the other). A 50/50 blend gains in both directions, while union, intersection and the stricter T = 2 rule each hurt in one direction. The France guard held: pairs per S1 rose from 3.258 to 3.319 (US/India 3.37-3.39), the empty rate stayed at 5.8%, and the share of emitted pairs with a decoy house shift fell.

## 16. Candidate file and final checks

- The candidate file is exactly the set the final decision ran on: pairs with stage-2 p >= 0.001, the route pairs, the extra-key / reverse-retrieval pairs and the second family's pairs. That is about 13 per S1 instead of 46.5. Every emitted match must be inside it.
- Run the organisers' validator with id checks on both files; every test S1 (1,732,544), including France, must appear.
- Guard values of the 0.989022 file:

  | Country | Emitted pairs per S1 | Empty share | House-conflict share |
  |---|---|---|---|
  | US | 3.386 | 5.76% | 8.77% |
  | India | 3.374 | 5.76% | 17.4% |
  | France | 3.319 | 5.79% | 1.04% |

  The final file has 5,838,977 matched pairs.
- Reproducibility: every layer should be deterministic (fixed seeds, deterministic tree mode). Our rebuild on new machines reproduced the original stage CVs to within 0.00003, and the final file byte for byte.

## 17. What did not help (skip these)

- Coherence / sibling / context features, and a hard house-shift veto (hurts India). Mapping leftover native shop words: they occur only on decoys, so mapping them removes a perfect decoy signal.
- Decision tweaks (thresholds, calibration, joint assignment, other k rules), more training data (5x data +0.0007), naive synthetic positives or pseudo-labels on the target country, France mass calibration once the lift and parser are in place, and switching the lexical layer off for France.
- Later layers:
  - a second, larger cross-encoder on top of the first (+0.00001);
  - dropping the second family's cross-encoder inputs from the meta-model (-0.00006);
  - raw cross-encoder columns of our own as meta inputs (+0.00009, but missing for extra-key pairs, so not used);
  - a plain logit blend of the two families instead of the meta-model (-0.0014);
  - union, intersection or the stricter decision for France;
  - the second family's wider France veto list (in the US, the equivalent veto on "partners" would cost -0.0024).
- House-conflict share is not a precision signal for extra-key pairs (section 12).

## 18. Where the remaining loss is

- The candidate ceiling (every candidate classified perfectly) is now 0.99687 on train (0.99797 on the sample that also has the extra-key candidates). The best model is at 0.99051, so the remaining loss is mostly classification.
- On train the meta-model still misses about 214k true pairs, of which about 136k are in the candidates but rejected; false positives are only about 7.4k. S1 with exactly one true match score lowest.
- France (no labels) is probably still the weakest country on the leaderboard.

## 19. Compute used

CPU steps ran on 12 vCPU / 96 GB Linux machines, with long retrieval steps sharded across several machines by country. The heaviest steps are blocking, stage-1/2 CV and address retrieval (1-2.5 hours each on one machine), then the generator features (about 30 minutes) and each correction model (15-30 minutes). The cross-encoders ran on Kaggle GPUs (2 x T4; about 3 hours of training plus 22 minutes of scoring for ours).
