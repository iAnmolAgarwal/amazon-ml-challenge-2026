# Evaluation B (round 2): methodology and documentation quality

Package: `submissions/final_package/Apprentice_submission.zip`, public LB 0.990203. I unzipped it to scratch and did not touch the zip or the repo.
Lens: methodology and documentation. Bar: top 50 of about 1,000 teams (top 5%).
Main document: `Documentation_template.md`, 3,347 words. I cross-checked it against `code/business_entity_resolution/README.md`, `src/stack/README.md`, every shipped `attempts/*/results.json` and the code (`er_model.py`, `er_feat5/6.py`, `er_gen.py`, `er_lexical.py`, `er_france.py`, `er_frwords.py`, `er_meta.py`, `er_fix.py`, `er_newpairs.py`, `src/single/fit.py`, `src/stack/src/stack_ce.py`, `stage3_feats.py`, `decoy_feats.py`).

## Scores and verdict

| Criterion | Score | Reason |
|---|---|---|
| Completeness | 9 / 10 | Every template section has real content. It also has a pipeline sketch, a glossary, limitations, compute, a progression table, an ablation and negative results. Missing: a one-line "Model type" field, error-analysis examples, and an explicit reproduction entry point. |
| Technical depth and soundness | 9 / 10 | Stratified entity folds, out-of-fold inputs at every stacking level, a paired acceptance rule, a decoy-density stress test, a label-free test guard, fair-sample checks and LOCO for France. Soft spots: the decision rule is overclaimed as exact, and the second family gets one clause even though it adds +0.0027 CV. |
| Originality / insight | 8 / 10 | Several parts are original: the reverse-engineered decoy generator (ten house shifts, marker words, "decoys never have an empty address"), label-free decoy-marker scores with a per-country lift, and the finding that context features win CV but emit decoys on test. The rest is standard: TF-IDF blocking, LightGBM, e5 cross-encoders, stacking. |
| Clarity and structure | 6.5 / 10 | Well sectioned, with tables and a glossary. But the text is dense and full of numbers. The executive summary is 5 sentences (the template asks for 2-3). "Generator-operator" appears 6 times and is never defined. The candidate-file paragraph and Section 4.3 take several readings. |
| Honesty and credibility | 8 / 10 | Every number I checked matches the shipped results exactly (table below), and the disclosures are unusually candid. There are four statements a careful judge can catch (HIGH 1-3 below): "every change passed", "no feature compares ... the S1's other candidates", "three times", and "maximizes expected F0.5". |
| **Overall (this lens)** | **8 / 10** | **Clears the top-50 bar on methodology. In substance it is top-10 grade; on readability it is not yet. About 8.5-9 after the HIGH edits.** |

**Verdict.** On this lens the package advances. Its validation discipline is stronger than what most top-5% write-ups show: paired CV, a label-free measurement of the test shift, a guard that rejected CV winners, and a negative-results table. So is the evidence that the gains come from features, not stage structure (B.2). The risks are credibility and readability:
- Three sentences overclaim or contradict the shipped code or tables. A judge who reads the code or checks Table B.3 will find them.
- The document reads like a dense lab log. A judge with five minutes gets the idea from the summary and Table B.1, but will not trust the details without effort.

## Numbers verified

| Claim in the document | Source in the zip | Status |
|---|---|---|
| CV 0.99184, US 0.99164, India 0.99213, fold SD 0.00003 | `attempts/029f6_final/results.json` (mean 0.991836, std 3.35e-5) | exact |
| Train oracle 0.9980, 13.37M pairs, 6.06 per S1 | same file (oracle 0.997985, n_pairs 13,373,981) | exact |
| Blocking 98.0M pairs, oracle 0.994 | `005f_floor/results.json` (98,021,508; 0.99398) | exact |
| Stage CVs 0.97008 / 0.97423 / 0.98589 | `005n`, `006`, `016` results.json | exact |
| FP 12,987 → 5,564; FN 302,721 → 186,753; precision 0.9993; recall 0.976 | derived from pair precision and recall in `016` and `029f6` (about 7.64M true pairs) | consistent |
| Single-match S1 0.973, 2+ matches 0.990-0.994 | `by_true_matches` in `029f6` | exact |
| Decision rule 5·Σp(top k)/(Σp + 4k) vs Π(1 − p), floor 0.6, one owner per pool record | `er_model.decide` | exact |
| Shift set {1,2,3,4,5,7,9,11,13,21} | `er_feat5.py`, `er_feat6.py`, `er_frwords.py`, `stack/decoy_feats.py` | exact |
| France W = 0.5, C = 0.1, veto and broad word lists | `attempts/029f5_final/train.py`, `er_france.py` | exact |
| France ≈ 0.981 implied | 0.47·0.99213 + 0.38·0.99164 + 0.15·F = 0.990203 gives F = 0.9805 | arithmetic correct; it is a lower bound (LOW 1) |
| Decoy-clone stress test, test guard, main-family LOCO grids, GAM ablation | no script in the zip | not reproducible from the package (MEDIUM 7) |
| "No feature compares a candidate's name or address with the S1's other candidates" | `src/stack/src/stage3_feats.py`: "similarity of r's address/name to the entity's OTHER confident records (anchors, p2 >= 0.9)". Used through `--extra` in steps 7-8 of `src/stack/README.md` (`stack_ce.py` lines 44, 194, 411). | **false for the submitted system** (HIGH 2) |

## What a judge sees in five minutes

**Impressive (keep it prominent):**
- Treating the data as a generator, with evidence: non-matches spike at exactly ten house shifts. The test's 1.94x decoy density was measured without labels.
- Label-free decoy-marker scores (n_d + 1)/(n_d + n_0 + 10), re-centred per country so they transfer to France.
- The CV-vs-test trap: context features that win CV (+0.0033, 5/5 folds) lose −0.037 at test decoy density. This is the most memorable lesson in the document.
- B.2: one LightGBM on all 205 features equals the 5-stage chain (0.98779 both), so the gains come from information, not architecture. Judges rarely see this kind of self-critique.
- B.3 negative results, and honest disclosure that the leaderboard decided one France change.

**Generic (fine, but it should not take most of the words):** TF-IDF plus key blocking, fuzzy-ratio features, e5 cross-encoders, LightGBM stacking, a meta-model over two pipelines.

**Red flags a judge could raise:**
1. The summary says "Every change had to pass a paired 5-fold CV test", but 2.2 says one France change was kept because the leaderboard rose.
2. 2.1 says no feature compares a candidate with the S1's other candidates. The second family's stage-3 anchor features do exactly that, and the meta-model reads them.
3. 2.1 says "We saw this three times ... (Table B.3)", but B.3 shows only two context-feature cases.
4. 4.4 says the rule "maximizes expected F0.5". It is a ratio-of-expectations approximation with greedy one-owner assignment and an empirical 0.6 floor on top.
5. The late rows of B.1 add +0.00002 CV and +0.000002 LB, and one of them keeps a 32 GB, 2-hour component. Without a cost-benefit sentence this reads as leaderboard grinding.
6. Test records are used transductively (France marker table, France word lists), which is label-free but never stated as a principle. A judge who finds it alone may suspect test tuning.

## Against a strong top-10 write-up

A top-10 write-up usually has: a 3-sentence summary, a diagram near the top, the one key insight stated early, an ablation of the final model, error analysis with categories and 2-3 real examples, runtime and cost, and limitations.

This one beats the typical top-10 write-up on validation rigor, negative results and honest disclosure. It matches it on ablation (B.1, B.2), compute and limitations. It falls short on three things:
- **Readability:** long sentences, jargon, 5-decimal numbers inside prose.
- **Error analysis:** false positives are described but not counted by type, and there are no concrete examples.
- **The second family:** a +0.0027 CV component described in one clause.

Among the write-ups of the top-50 LB teams I would expect this one in the top quarter on methodology. It becomes a top-10 contender if the HIGH edits land.

## Prioritised edits to Documentation_template.md

Length budget: the document is 3,347 words. The edits below add about 350 words and remove about 150 (MEDIUM 6 and LOW 4). Keep the result at or below about 3,500 words.

### HIGH (credibility and the first five minutes; about 2-3 hours in total)

**HIGH 1. Executive summary: fix the "every change" overclaim and cut it to 3 sentences.**
Current (whole paragraph):
> We treat the data as the output of a generator that adds noise to true matches and plants near-copy decoys, and we model that generator. Candidates come from word and character TF-IDF plus exact keys (train candidate oracle F0.5 0.998). LightGBM models score each pair with pair-local features, including label-free decoy-marker statistics that transfer to the unlabelled country (France), and a cross-fitted meta-model combines them with fine-tuned e5 cross-encoders and a second LightGBM model family. An expected-F0.5 rule with one owner per pool record picks each answer. Every change had to pass a paired 5-fold CV test and a label-free test-shift guard: CV 0.99184 (US and India), public leaderboard 0.990203.

Replace with:
> We model the generator behind the data: true matches are noisy copies of an S1 record, and unmatched pool records are near-copy decoys made by a few operators (a house number shifted by one of ten steps, one inserted marker word, a swapped legal form), which the test set contains at about 1.9x the train rate. Candidates come from TF-IDF retrieval and exact keys (train oracle F0.5 0.998). LightGBM models on pair-local features score each pair, including label-free decoy-marker scores that also work for unlabelled France; a cross-fitted meta-model combines them with fine-tuned e5 cross-encoders and a second LightGBM family, and an expected-F0.5 rule with one owner per pool record picks each answer. US and India changes were accepted only on a paired 5-fold CV test plus a label-free test-shift guard (CV 0.99184); France rules were set by leave-one-country-out tests and label-free counts (public leaderboard 0.990203).

(The second sentence drops "character" because train candidates have no character-route pairs; see Limitation 4.)

**HIGH 2. Section 2.1, last bullet: make the central lesson match the shipped system and Table B.3.**
Current:
> - Consequence: features that judge a candidate by its agreement with the S1's other candidates win in CV and emit decoys on test. We saw this three times and dropped each family (Table B.3). What remains is pair-local content features plus score-based context: the rank and margin of a pair's earlier-stage probability among the S1 that claim the same pool record and among the S1's own candidates. No feature compares a candidate's name or address with the S1's other candidates.

Replace with:
> - Consequence: features that judge a candidate by its agreement with the S1's other candidates win in CV and emit decoys on test, where each S1 has about twice as many decoy siblings. Both such families in Table B.3 (pool-record graph features, an S1-context stacker) were dropped. In our main family, context enters only through scores: the rank and margin of a pair's earlier-stage probability among the S1 that claim the same pool record and among the S1's own candidates. The second family keeps one content-context group, the similarity of a pool record to the S1's own confident records (stage-2 p ≥ 0.9); the meta-model that reads it passed the same label-free test guard before upload.

Before using this text, the team should confirm that the meta-model upload passed the guard. If a third dropped case exists, add it as a row in B.3 and keep "three".

**HIGH 3. Section 4.4: do not claim exact optimality, and show the floor's evidence.**
Current:
> Then pairs with p ≤ 0.6 are dropped (floor tuned by paired CV). The rule maximizes expected F0.5 per S1 under the model's marginals.

Replace with:
> Then pairs with p ≤ 0.6 are dropped. The prefix score is a plug-in approximation of expected F0.5 (a ratio of expectations with pairs treated as independent), and the one-owner step is greedy per pool record, so the rule is near-optimal rather than exact. The 0.6 floor is an empirical correction on top of it, chosen by paired CV: on the same stage-1 probabilities it raises CV from 0.97008 to 0.97083 (`attempts/005n_indic_nocoh` vs `005f_floor`).

**HIGH 4. Glossary: define the terms the document actually uses.**
Current:
> **Glossary.** *Band*: pairs whose stage-2 probability is ≥ 0.001 (the cross-encoder band is 0.003-0.997). *Route*: a candidate source (character retrieval, extra keys, reverse retrieval). *Name twin*: another S1 of the same country with the same normalized name. *LOCO*: fit on US and score India as if unlabelled, and the reverse. *Lift*: a country's marker scores re-centred by its own decoy base rate.

Append:
> *Generator operator*: one edit the data generator applies to make a noisy match or a decoy (house shift, truncated digits, word insertion or drop, legal-form swap, reorder, acronym); generator-operator features encode which operators link a pair. *Chain probability*: our family's probability after its last correction, before the meta-model. *Second family*: an independently built LightGBM and cross-encoder pipeline (`src/stack`) with its own blocking and folds. *Fair sample*: a fixed, label-free 10% of train S1 that gets the same extra candidate sources as test; a gain there is about 10x its effect on full CV.

Then shorten the "Fair sample" bullet in 2.2 to: "Fair sample (glossary): changes that add candidate sources are also checked paired on it."

**HIGH 5. Section 5: quantify the errors and show real examples.**
Current:
> - **Common false positives (wrong merges):** decoys that keep the name and address but shift the house number by a typical step or insert a generic business word; legal-form swaps; unit-number changes in the same building. Out-of-fold false-positive pairs fell from 12,987 (band model) to 5,564 (final), and pair precision is 0.9993.

Keep that text and add a compact table computed once from the final out-of-fold predictions (`030p_fix` with gold and the `feat5` / `gen` columns). **Fill it only with computed counts; do not estimate.**

> | Error type (final out-of-fold, US + India) | Pairs | Share |
> |---|---|---|
> | FP: house number shifted by a decoy step | [count] | [%] |
> | FP: pool name adds one word | [count] | [%] |
> | FP: legal-form swap or unit conflict | [count] | [%] |
> | FP: other | [count] | [%] |
> | FN: empty or "None" pool address with a name twin | [count] | [%] |
> | FN: other, scored below the floor | [count] | [%] |

Add two short real train examples, quoted verbatim from the data, each with S1 name and address, pool name and address, p, and the reason: one decoy false positive and one missed empty-address twin.

Also add one line: "S1 with no true match (5.6% of train) score 0.997, so the empty-answer option rarely fires wrongly." (`by_true_matches`: 123,247 S1, F0.5 0.99697.)

### MEDIUM (consistency and depth; about 1.5-2 hours)

**MEDIUM 1. Section 6: the +0.009 row is not only house shifts.**
Current:
> Modelling the generator drove the largest single steps: pair features with house-shift relations (+0.009 CV), decoy-marker scores (+0.002) and generator-operator codes (+0.001).

Replace with:
> Modelling the generator drove the largest single steps: pair features for name specificity, house-number relations and legal form (+0.009 CV), decoy-marker scores (+0.002 CV but +0.007 public LB) and generator-operator codes (+0.001).

**MEDIUM 2. Add the CV-LB gap story to B.1. It is free, verifiable evidence for the decoy-shift hypothesis.**
After the B.1 footnote, add:
> The CV-LB gap fell from 0.011 (third row) to 0.004 when the decoy markers were added (+0.002 CV, +0.007 LB), and stayed between 0.0011 and 0.0021 afterwards. This fits the label-free finding that test is decoy-denser; France, which gains from the same label-free features, also contributes.

**MEDIUM 3. Limitation 3 contradicts the empty-address correction.**
Current:
> (3) Empty-address pool records shared by name twins are a ceiling of about 0.004 F0.5 that no feature here resolves.

Replace with:
> (3) Empty-address pool records shared by name twins cost about 0.004 F0.5 (measured on the 0.99051 parent). The empty-address correction recovers only a small part (0.99177 → 0.99184 CV), because with no address the twins carry no separating evidence.

**MEDIUM 4. B.1: label the leaderboard-kept France row and explain the fair-sample scale.**
Current row:
> | + empty-address correction applied to France (final) | France only (transfer +0.00006, 5/5 / +0.00002) | **0.990203** |

Replace with:
> | + empty-address correction applied to France (final) | France only (transfer US→India +0.00006, 5/5; India→US +0.00002, not significant); kept on leaderboard evidence (+0.00021) | **0.990203** |

Append to footnote ¹ (current text: "The fair-sample figure is the comparable gain."):
> Fair-sample gains are measured on the 10% sample where the change acts, so the full-CV effect is about a tenth (+0.00049 → +0.00005 in the row above).

**MEDIUM 5. Section 4.2, item 6: describe the second family. It is the largest late gain.**
Current:
> The second family (src/stack) is a separately built LightGBM pipeline with its own blocking, stage-3 stack and in-country e5 cross-encoders.

Replace with:
> The second family (src/stack) was built independently in its own environment. It has its own normalization and transliteration, blocking, a two-stage LightGBM with name-specificity features, label-free decoy features with the same country lift, and a stage-3 stack over three e5 cross-encoders (one cross-country, two in-country). For France it adds a cross-country e5-base cross-encoder, its own decoy stacker and a marker veto.

In the same item, extend the fold caveat. Current:
> The second family uses its own entity folds (crc32 of the S1 id), so this CV may be slightly optimistic.

Replace with:
> The second family uses its own entity folds (crc32 of the S1 id), and the cross-encoder halves ({0,1} / {2,3,4}) are not nested in the meta-model folds, so this CV may be slightly optimistic.

**MEDIUM 6. Section 3: move the candidate-file rule to the README, which already has it verbatim.**
Current (the paragraph under the table):
> `candidate_pairs.tsv` lists every pair the final models run inference over: our pairs that pass the stage-2 filter (p ≥ 0.001) or were added after it (for US and India also chain probability ≥ 0.001, the meta-model's input), plus every candidate of the second model family (its cascade: p > 0.02, top 10 per S1). The decision uses the same set except 58,364 second-family pairs that our stage 2 scored below 0.001 (scored, never matched). France has twice the pairs per S1 because its pairs skip the chain-probability cut before the France blend.

Replace with:
> `candidate_pairs.tsv` is the union of both families' pairs that the final models score; the exact rule, including the 58,364 scored-but-never-matched pairs, is in the README ("Candidate file"). France has twice the pairs per S1 because its pairs are not cut before the France blend.

(This saves about 60 words. If the compliance review wants the full rule in this document, move it to Appendix A rather than deleting it.)

**MEDIUM 7. State the transductive use of test data, and admit that the validation checks are not shipped.**
In 2.2, after the "France: leave-one-country-out ..." bullet, add:
> - Test data without labels: test records are used, label-free, in three places: the France decoy-marker table, the France word lists (`src/er_frwords.py`) and the test guard's summary counts. No test label or leaderboard score enters any feature.

After Limitation (5), add:
> (6) The decoy-clone stress test, the test guard, the LOCO grids of our family and the GAM ablation were run with analysis scripts that are not in the package; the package reproduces the pipeline, not these checks.

**MEDIUM 8. Cost-benefit sentence, to defuse the leaderboard-grinding reading of B.1's late rows.**
Add to Section 6, after the B.2 sentence:
> A production version would keep one LightGBM on all features (B.2) plus the meta-model, and drop the stage chain and the flat-model input (+0.00002 CV at about 32 GB and 2 hours).

**MEDIUM 9. Appendix A: name the entry points, as the template asks.**
Current:
> the final decision step takes 19 s from the saved probabilities.

Replace with:
> Entry points: the README command list rebuilds both output files from the raw TSVs. Its last command, `python src/er_predict.py 029f6_final --from-probs 030q_fix`, writes them in 19 s once the cached probabilities exist (the cache is not shipped).

### LOW (polish; under 1 hour)

1. Section 5. Current: "the higher test decoy density probably lowers US and India too, so this is only a bound." Replace with: "the higher test decoy density probably lowers US and India too, so 0.981 is a lower bound for France (the public split's country mix may also differ)."
2. In prose, round to 3-4 decimals ("+0.0003 / +0.0006"). Keep 5 decimals in tables only. Six "0.0000x" figures in running text read as leaderboard arithmetic.
3. Section 4: add the template's field after the 4.2 heading: "**Model type:** LightGBM (pair models, corrections, meta-model) with fine-tuned multilingual-e5 cross-encoders (small, base)."
4. Section 4.2, "Pretrained models" paragraph: cut to one sentence, since the README carries the revisions and licences: "Pretrained: multilingual-e5-small and e5-base (MIT, pinned revisions, fine-tuned on training pairs only; 12 checkpoints); LightGBM 4.7.0; no external data; all libraries permissive except Unidecode (GPL-2.0, second family only)."
5. Section 4.2, item 5: add "on test each S1 is scored by one half model, chosen by its id, as on train". This shows care about train/test score distributions (`er_extend.py parity`).
6. The Section 2.1 evidence bullet (the "13.6k-15.8k pairs" histogram) is the best single piece of evidence in the document. If there is room, add it as a 12-number table B.4 (count per house-number difference).
