# Gate 6 audit: Astra "breakthrough" answer (06_breakthrough_feedback.md)

Auditor: independent (Opus), 2026-09-25 ~09:30 IST. All numbers below were measured on training labels only (002_namefreq OOF, 97,811,967 pairs; `decide()` reproduced exactly: India 0.944383, US 0.971258). Test files were used only label-free (script shares, pool/S1 ratios, the already-submitted sub-02 predicted-k profile). Scripts: scratchpad `audit06/{oof,sib,indic,dict_test,alias,alias2,drop,france}.py`. "Test-weighted" = 0.47 x India delta + 0.38 x US delta (France contributes 0 unless stated).

## 0. Bottom line

- Astra's arithmetic is right (two rounding slips). Its ranking of bets is wrong. The measurements point to one cheap move that was underrated: an **Indic-to-Latin token dictionary learned from gold pairs**. 1,364 Indic name-token types map onto only 151 Latin words. Every held-out Indic token is covered by the other folds with at least 5 owners. Mapped tokens hit the S1 name 96% of the time. The India ceiling is +0.024 (+0.0115 test-weighted), and a realistic proxy gives +0.0177 India (+0.0083 test-weighted).
- The cross-encoder gate is compact. Pairs with p in [0.05, 0.95] are 2.1% of pairs (2.10M train) and hold 86% of the known-country classification ceiling (+0.0229 of +0.0266). The bet is therefore computationally cheap, but how much of that ceiling it recovers is unknown. Astra's 10M-pair OOF / 54 GPU-h framing overstates the cost.
- The alias catalogue (finding 7) is dead. 0 of 1,732,544 test S1 share an exact normalized name+address with any train record.
- Pool-sibling expansion has a large oracle ceiling (+0.0075 test-weighted). 56% of India's part of it is the same Indic-name misses that the dictionary fixes directly. For Latin-name misses, sibling links are barely better than direct S1 links. What remains is about +0.001-0.002 realistic.
- +0.025 is not reachable from the measured headroom in about 50 hours. A realistic stack is 004 (+0.004-0.005) + Indic dictionary (+0.004-0.008) + possibly a gated cross-encoder (+0.001-0.004), which puts the LB around 0.958-0.966.

## 1. Arithmetic check (finding 1, 3, 4, 7, allocation)

| Astra item | Astra value | Recomputed | Verdict |
|---|---|---|---|
| Perfect US/India classification | 0.02660 | 0.47x(0.981453-0.944383)+0.38x(0.995417-0.971258) = **0.026603** | right |
| Perfect US/India retrieval | 0.01044 | 0.47x0.018547+0.38x0.004583 = **0.010459** | slip (+0.00002) |
| India retrieval ceiling | 0.00870 | **0.008717** | slip, immaterial |
| France 0.9069 -> 0.96 | 0.00796 | 0.15x0.0531 = **0.007965** | right |
| France perfect | 0.01396 | **0.013965** | right |
| 0.025 / 0.0266 | 94% | 94.0% | right |
| Route: 0.005 / 0.01044, 0.012 / 0.0266, 0.008 / 0.15 | 48%, 45%, 0.053 | 47.8%, 45.1%, 0.0533 | right |
| Native-name / native-address gold edges | 565k / 687k | 0.074x7,638,365 = 565k; 0.09x = 687k | right |
| CE sensitivity 10-30%: own / sel / FP | 0.00046-0.00138 / 0.00136-0.00408 / 0.00112-0.00336 | same | right, but **train-weighted** (US 60%). Test-weighted, 10-30% of the known-country classification loss is 0.0027-0.0080 overall |
| 6x500k at 100/s; 15M at 500/s; 98M at 500/s; 10M at 50/s | 8.33 / 8.33 / 54.4 / 55.6 h | same | arithmetic right; the rates are assumptions and the 10M-pair OOF premise is wrong (see §2a) |
| 10M x 384 float32 | 15.36 GB | same | right |
| Pairs inside 44-candidate lists | ~2.1B | 2.21M x 44 x 43 / 2 = 2.09B | right |
| Alias table (5% x 0.10 etc.) | 0.005 | same | right (moot, §2d) |
| 3+6+2+2 = 13; 3/5/5 by day | 13 | right | |

Caveat on "implied France 0.907": this assumes US/India transfer at CV. LB is rounded to 3 decimals, so France lies in [0.904, 0.910] even under that assumption. Sub-01 implies 0.883, so both submissions show about the same gap. Test has 5.76 pool records per S1 against 4.68 in train (US 5.76, India 5.82, France 5.53), so about 40% of test pool records are unmatched against 26% in train (if t is unchanged). Predicted k per S1 on test matches OOF (US 3.29 vs 3.25; India 3.16 vs 3.13), so the extra pool is mostly unmatched records. A fixed-probability simulation that drops 19% of train S1 from the graph (their gold records become orphans) moves macro by only -0.0002 (India) and -0.00014 (US). A logit shift (delta 0.25-1.0) never helps (India -0.0002 to -0.0049). The feature-level effect (rival counts, reverse ranks, name frequencies on a graph with orphans) is unmeasured. Until it is measured, part of the CV-LB gap could be test shift rather than France.

## 2. Measurements

### 2a. Classification headroom and "uncertain gate" (optimistic correction, ownership re-run)

Gated pairs get p := y; `decide()` then re-runs on the full country graph, so ownership rivals are handled.

| Gate | India pairs | US pairs | India dF | US dF | Test-weighted | Share of 0.0266 ceiling |
|---|---|---|---|---|---|---|
| all pairs perfect | 40.4M | 57.4M | +0.03707 | +0.02416 | +0.02660 | 100% |
| p in [0.01, 0.99] | 2.42M | 3.15M | +0.03634 | +0.02318 | +0.02589 | 97% |
| p in [0.02, 0.98] | 1.77M | 1.99M | +0.03538 | +0.02207 | +0.02502 | 94% |
| **p in [0.05, 0.95]** | **1.01M** | **1.10M** | **+0.03252** | **+0.01992** | **+0.02286** | **86%** |
| p in [0.1, 0.9] | 0.66M | 0.73M | +0.02921 | +0.01777 | +0.02048 | 77% |
| p in [0.2, 0.8] | 0.39M | 0.41M | +0.02286 | +0.01382 | +0.01600 | 60% |
| [0.05, 0.95] or ownership contest (2nd S1 p > 0.05) | 1.03M | 1.12M | +0.03264 | +0.01996 | +0.02293 | 86% |
| entity gate (all pairs of S1 with any p in band) | 26.8M | 32.2M | +0.03519 | +0.02195 | +0.02488 | 94% |

- The positive rate inside the [0.05, 0.95] gate is 0.46 (India), so the gate is a balanced training distribution for a re-ranker.
- Confident errors outside the gate: gold with p < 0.05 is 22,903 (India) / 37,154 (US); non-gold with p > 0.95 is 8,593 / 9,055. A gated model cannot fix them.
- 27.8% of India gate pairs have an Indic-script pool name. The dictionary (§2c) targets those first.
- Reading: the whole known-country classification ceiling sits in about 2.1M train pairs (about 1.4M on test for US+India). Any re-scorer only needs OOF over those pairs. The ceiling is not a forecast. It assumes perfect decisions on a set where LightGBM is by construction unsure.

### 2b. Pool-sibling expansion ceiling (finding 4)

| | India | US |
|---|---|---|
| Unretrieved gold edges | 153,542 | 68,447 |
| ... belonging to S1 with 0 retrieved gold | 8,357 (5.4%) | 1,487 (2.2%) |
| S1 with t>0 and 0 retrieved | 4,932 | 1,250 |
| S1 with >=1 retrieved and >=1 missed | 113,813 (12.9% of India S1) | 62,294 (4.7%) |
| Oracle F0.5: current -> all siblings retrieved | 0.98145 -> 0.99442 (+0.0130) | 0.99542 -> 0.99906 (+0.0036) |
| Decision-level, missed siblings added with the S1's best gold p | 0.94438 -> 0.95955 (+0.0152) | 0.97126 -> 0.97544 (+0.0042) |
| Test-weighted | oracle +0.0075; decision +0.0087 | |

Linkability of missed records (c>=1) to their retrieved siblings, compared with the S1 itself (conservative text, token_set_ratio):

| Missed-record group | n | sibling name>=90 | sibling addr>=90 | sibling either | S1 either>=90 | sibling name exact |
|---|---|---|---|---|---|---|
| India, Indic-script name | 80,303 | 0.585 | 0.516 | 0.765 | 0.377 | 0.564 |
| India, Latin name | 64,882 | 0.311 | 0.375 | 0.577 | 0.508 | 0.020 |
| US (all Latin) | 66,960 | 0.392 | 0.309 | 0.599 | 0.463 | 0.037 |

- The sibling route mainly helps Indic-name misses, and the dictionary solves those directly.
- For Latin misses, siblings beat the S1 link only modestly (0.58 vs 0.51 India, 0.60 vs 0.46 US). Link precision (contamination) was not measured. Sampled misses are typo'd names with empty addresses, website forms and unrelated DBA names ("iriarc" for "keystone hospitality ventures llc").
- The residual sibling ceiling after the dictionary is roughly India +0.006 and US +0.004, about +0.004 test-weighted. Expect 20-40% of that.

### 2c. Cross-script retrieval and dictionary coverage (finding 2)

- India unretrieved gold edges: 56.2% have an Indic-script pool name, 22.2% an Indic address and 63.5% either. Among retrieved gold edges: 16.0% name, 22.6% address. **Name script, not address script, drives misses.** Among misses of S1 with zero retrieved gold: 71.7% Indic name.
- Indic scripts appear only in India. Test Indic shares: US 0.0 / France 0.0 / India 15.7% names and 19.9% addresses. Train India pool: 18.3%. A script-gated view is automatically a no-op for France.
- Dictionary coverage. Folds are the 002 CV folds. For each fold, the dictionary is built from other-fold gold pairs, and the test set is the held-out fold's *missed* Indic-name records. Mean over 5 folds:

| Metric | Names | Addresses |
|---|---|---|
| Held-out Indic token occurrences (per fold) | 65.4k in 17.3k records | 7.5k in 6.8k records |
| Token covered by >=1 / >=5 other-fold owners | 1.000 / 1.000 | 1.000 / 1.000 |
| Records with all Indic tokens covered (>=5 owners) | 1.000 | 1.000 |
| Dice-mapped token hits a token of the S1 name | 0.960 | 1.000 |
| Records with >=50% / 100% of mapped tokens hitting the S1 | 0.995 / 0.900 | 1.000 / 1.000 |
| Whole Indic name seen as a gold name in other folds (>=1 / >=5 owners) | 0.659 / 0.306 | n/a |

- The vocabulary is tiny: 1,364 Indic name-token types, 151 Latin targets, median support 228 owners, 95.7% of types with a top-target share >= 0.9. Examples: लिमिटेड -> limited, प्राइवेट -> private, प्रा/लि -> pvt/ltd, and one fixed spelling per script.
- Dice co-occurrence confuses some always-together pairs (Malayalam "limited" -> "private" at 0.72). Use positional alignment when token counts match.
- Test India pool: 96.4% of Indic token occurrences are in the train dictionary (distractors included), and 86.0% of Indic-name records are fully covered. Keep the original token as well for the rest.
- Ceilings and proxy (India, full graph re-decided):

| Scenario | India F0.5 | India d | Test-weighted |
|---|---|---|---|
| base | 0.94438 | | |
| perfect decisions on all Indic-name pairs (classification only) | 0.95626 | +0.0119 | +0.0056 |
| add the 86,296 missed Indic-name gold edges with inherited best p | 0.95342 | +0.0090 | +0.0042 |
| both | 0.96880 | +0.0244 | +0.0115 |
| proxy: Indic gold pairs and added misses get the Latin-gold p distribution; negatives unchanged | 0.96212 | +0.0177 | +0.0083 |
| India oracle with those misses retrieved | 0.98145 -> 0.99333 | | |

- Current handicap: retrieved Indic-name gold edges are emitted 88.8% of the time against 94.4% for Latin. Emitted precision is 0.965 against 0.987. Indic-name records are 33% of India FP edges.
- The proxy is optimistic because it assumes every mapped miss is retrieved and scored like a typical gold pair. It is conservative because it gives no FP reduction. Planning range: **+0.004 to +0.008 test-weighted**, France 0.

### 2d. Alias catalogue (finding 7), about 10 minutes

- Test S1 whose exact lowercase-alphanumeric name+address equals any train record (S1 or pool, same country): **0 of 1,732,544**. Members of train gold clusters matching a test S1: 0. Bridges: 0.
- Name-only: 46% (India) / 40% (US) of test S1 names occur in train. 78k names are unique in both train S1 and test S1. Their addresses are unrelated: token_set >= 80 in 0.08% of cases against 0.01% for shuffled pairs, mean 36 vs 32.
- Train and test businesses are disjoint and built from a shared vocabulary. The shared vocabulary is exactly why the token dictionary transfers and a business catalogue does not. **Kill finding 7.**

## 3. Verdict per finding

1. **AGREE (arithmetic), PARTLY (framing).** Numbers are right apart from two 0.00002 slips. The framing misses two things. The classification loss is 86% concentrated in 2.1% of pairs. Implied France rests on CV transfer, which the 5.76 vs 4.68 pool/S1 shift puts in doubt. The illustrative route should not assume retrieval and classification are separate levers: the Indic dictionary moves both (64% of India's retrieval loss plus the Indic classification handicap).
2. **AGREE on the direction, DISAGREE on the design and the rank.** A learned correspondence is the right idea, but the measured coverage (100% held-out, 96% of test tokens, 151 targets) makes the transducer, the P(record|channel)/background likelihood ratio and the operation-specific residual scores unnecessary within 50 hours. A token dictionary plus a mapped name view in retrieval and features captures it. Astra's 0 to +0.010 (successful +0.003-0.010) matches my +0.004-0.008. Astra ranks it first and I agree it is first, but at about 3 engineering hours, not 5-10.
3. **PARTLY.** The CE mechanism is plausible and the gate measurement shows it is cheap to deploy: about 2.1M train + about 1.4M US/India test (+ France) pairs, not 10M-98M. The recoverable fraction is unmeasured. LightGBM already sees about 60 features that encode most generator operators, and the dictionary will take the Indic share (27.8% of India gate pairs) first. Expect 5-15% of the gate ceiling: +0.001 to +0.0035 test-weighted. Astra's throughput assumptions (100 train / 500 infer pairs/s) are unbenchmarked; MiniLM-L12-H384 fp16 on a T4 at 128 tokens is probably several times faster (not verified). Its "do not score only top-3" and "calibrate on natural distribution" points are correct.
4. **PARTLY.** The oracle is large (+0.0075 test-weighted), but 56% of India's part is Indic-name misses, which the dictionary handles. For Latin misses, sibling links are only slightly better than S1 links and their precision is unmeasured. The residual realistic gain is +0.001-0.002 for 6-12 hours. Astra's 0 to +0.008 upper bound double-counts the dictionary.
5. **PARTLY.** Validating by country transfer (train without one country, test on it) is the right test and is the only CV-grounded France evidence available. A synthetic French generator costs 4-8 hours for 0 to +0.005 that cannot be validated, so it is not a priority. Label-free test profile of sub-02: France predicted k=0 on 4.2% of S1 against 5.5% for US/India (the singleton prior is 5.6%), and k>=5 on 25.2% against 22.5% / 20.4%. That leans towards over-merging in France. A leave-one-country-out check of decision strictness is cheaper than synthesis. Agree that pseudo-labels will not give a step change.
6. **AGREE.** The decision layer is exhausted. The prior-shift test here (logit - delta) also only loses (-0.0002 to -0.005).
7. **AGREE to check, measured DEAD** (§2d).
- **Rules (pretrained weights): DISAGREE with Astra's doubt.** README constraint 5 says "Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters". That rule only makes sense if pretrained models are expected, and the template lists "Siamese Network, Transformer" as model types. The prohibition covers *external data lookup*: entity-resolution APIs, government registries, geocoding, "external data augmentation from internet sources". Pretrained MIT/Apache weights of 8B parameters or fewer are allowed. Name the model and licence in the methodology. The dictionary needs no external data at all.
- **Submission allocation: PARTLY.** 6 large-bet slots is too many. No large bet will be CV-accepted before 26 Sep, and the LB shows 3 decimals, so gains under about 0.001 are invisible. Suggested: 25 Sep 1-2 slots (003 now, 004 when accepted); 26 Sep 2-3 (dictionary build, dictionary + CE if accepted, at most one France-only probe with US/India predictions fixed); 27 Sep 2-3 (final combination plus a reproducible fallback). Keep 3 or more unused as reserve. Leaving slots unused costs nothing.

## 4. Compute realism: cross-encoder

Resources: no AWS GPU; Kaggle T4/P100 about 30 GPU-h/week (the weekly reset is reportedly Saturday 00:00 UTC, i.e. 26 Sep 05:30 IST; not verified); 2x r7a.2xlarge busy for about 2 h; budget under $100.

| Step | Pairs | Rate (assumed, benchmark first) | GPU time |
|---|---|---|---|
| Export gate pairs (002/004 OOF p in [0.05, 0.95]) + text | 2.1M train, ~1.4M test US/India, France ~0.3-0.6M (unknown) | CPU | 0 |
| 5 fold models x 300k gate pairs x 1 epoch | 1.5M | 300-800/s train | 0.5-1.4 h |
| OOF inference on train gate | 2.1M | 1,500-3,000/s | 0.2-0.4 h |
| Test inference | ~2M | same | 0.2-0.4 h |
| Total | | | **about 1-2.5 GPU-h, 5 GPU-h at Astra's rates** |

- Integration: blend in logit space inside the gate, fitted on out-of-fold data (3 parameters), then `decide()`. Adding the CE score as a LightGBM feature would need a full CV rerun (2 h on EC2) and stacked-leakage care.
- One-person time: Kaggle setup and dataset upload 1-2 h, pilot 3-4 h, full folds + test + CV + packaging 5-7 h. **About 10-13 hours for +0.001-0.0035 expected.** Feasible in 50 h only after the dictionary and 004 are done.
- CPU-only is not viable for training. CPU inference of about 3.5M pairs with ONNX int8 at an estimated 200-500 pairs/s is 2-5 h per box, which works as a fallback for inference only.
- Smallest honest version: one CE trained on 300k gate pairs from folds 1-4 (US+India), scoring the fold-0 and fold-1 gates. Fit the blend on fold 1, evaluate `decide()` on fold 0 with the rest of the graph unchanged. Go if fold 0 test-weighted gains >= +0.002 (per-fold SE is about 0.0001) and measured T4 throughput projects to under 10 GPU-h in total. Otherwise stop.

## 5. Ranked plan (expected gain per hour)

| Rank | Move | Mechanism | Expected test-weighted gain | Cost | Smallest go/no-go pilot | Leak / France risk |
|---|---|---|---|---|---|---|
| 1 | **Indic -> Latin token dictionary view** | Learn token map from gold (positional alignment when counts match, else Dice; support >=5 owners). Add a mapped pool-name view to word TF-IDF + exact keys + conservative-name features (c_ratio/c_tset/c_jw on mapped names) + a mapped-fraction flag. Keep original tokens. | **+0.004 to +0.008** (ceiling +0.0115; proxy +0.0083) | 2-3 eng h + ~4-5 h EC2 (norm pool, block 1 h, feat 20-30 min, CV 2 h). Harness edits need human approval | (a) 30-60 min: recompute c_* features with mapped names for existing India candidates and rerun CV on the `_s20k` sample. Go if India improves >= +0.003. (b) Retrieval oracle on 50k held-out India S1 against the full pool. Go if India oracle gains >= +0.006 (max +0.0119) | Build 5 fold-local dictionaries and show they equal the global one for all tokens with support >= 20; expected identical at median support 228. France/US have 0 Indic characters: no-op, verify a byte-identical France output |
| 2 | Finish 004 (coherence + 100% train) -> submission | already running | +0.004-0.005 CV (gate 5) | running | its CV | as audited in gate 5 |
| 3 | Leave-one-country-out transfer check for France | Train on US and evaluate India, then the reverse. Test decision strictness (delta, p0 prior) and dropping country-sensitive features on the unseen country. Apply to France only if the effect agrees in both directions | 0 to +0.003 (France only; the test profile suggests over-merging) | 1 eng h + ~1 h EC2 | one LightGBM fit per direction on existing features plus `decide()` sweeps | CV-grounded but only a proxy for France; one France-only LB probe, US/India fixed |
| 4 | Gated cross-encoder (after 1) | Multilingual-MiniLM re-scorer on p in [0.05, 0.95], blended in logit space | +0.001 to +0.0035 (ceiling +0.0229) | 10-13 h, 1-5 GPU-h Kaggle | §4 pilot | Stacking leak if used as a LightGBM feature; untrained on French. Option: apply to US/India only |
| 5 | Latin-residual sibling expansion | Pool-pool rare name/address links from high-confidence gold candidates to unretrieved records | +0.001 to +0.002 | 6-10 h | measure precision of sibling links (token_set >= 90 on name or non-trivial address) against retrieved non-gold records on one fold | Contamination; French shared premises |

Cheap items not raised by Astra:
- Test has 5.76 pool/S1 against 4.68 in train (about 40% vs 26% unmatched). The first-order ownership effect is negligible (-0.0002), but a *re-featurized* drop simulation (remove 19% of train S1, rebuild rival/frequency features, rescore with the same model) takes about 1 h on EC2. It tells whether the CV-LB gap is test shift rather than France before any France work is funded.
- Items 1 and 3 are the only ones whose measured evidence clears a 2-3 hour pilot. The remaining +0.025 target has no measured path. The cross-encoder is the only move whose ceiling is big enough, and its realistic fraction is unknown.
