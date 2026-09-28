# Evaluation B (round 3): methodology and documentation quality

Package: `submissions/final_package/Apprentice_submission.zip` (team Apprentice, public LB 0.990203). I unzipped it to scratch without the parquet files and `candidate_pairs.tsv`. I did not modify the zip or the repo, apart from this file.
Lens: methodology and documentation. Bar: top 50 of about 1,000 teams (top 5%).
Main document: `Documentation_template.md`, 4,087 words, 178 lines. I cross-checked it against the organiser template and README, `code/business_entity_resolution/README.md`, `src/stack/README.md`, every `attempts/*/results.json` and `train.py`, `run_manifest.json`, the shipped `output/matching_results.tsv` and the code (`er_model.py`, `er_eval.py`, `er_feat6.py`, `er_lift.py`, `er_frwords.py`, `er_france.py`, `er_comb.py`, `er_fix.py`, `er_meta.py`, `er_lexical.py`, `er_norm.py`, `ce/ce_train.py`). I checked the dataset counts against the organiser TSVs.

## Scores and verdict

| Criterion | Score | Reason |
|---|---|---|
| Completeness | 9 / 10 | Every template section has specific content. The document also has a pipeline sketch, a glossary, compute, a progression table, an ablation, negative results, an error taxonomy with examples, and limitations. Gaps: the "Other routes" row has no numbers, and the document gives no candidate pair recall or reduction ratio, although the organisers say they analyse both. |
| Technical depth and soundness | 9 / 10 | The CV folds are stratified by entity. The acceptance rule is a paired test. Decoy density is measured without labels and backed by a stress test, a test-shift guard and LOCO for France. The decision rule is derived correctly and its approximations are stated. Soft spots: the meta-model CV caveat is placed far from the claim it qualifies, selection optimism from reusing the same folds is not mentioned, and one France change was kept on leaderboard evidence (this is disclosed). |
| Originality / insight | 8 / 10 | The distinctive parts are the reverse-engineered generator (ten house shifts, marker words, "decoys never have an empty address"), decoy-marker scores that use a house shift as a label-free noisy label, the 1.94x test decoy density measured without labels, and the finding that context features win CV but emit decoys on test. The blocking, LightGBM, e5 cross-encoders and stacking are standard, but well executed. |
| Clarity and structure | 6 / 10 | The document is correct but hard to skim. The executive summary is 146 words of stacked clauses. The glossary and pipeline appear after the terms they explain. Section 4.2 item 6 is one 200-word paragraph. The false-positive narrative contradicts its own table. A judge gets about two thirds of the solution in 5 minutes. |
| Honesty and credibility | 8 / 10 | Every number I could check matches (table below). The disclosures are candid: fold misalignment, the leaderboard-kept France change, unshipped analysis scripts, GPU non-determinism. Points a careful reviewer could catch: the Core Innovation wording and the pipeline sketch imply that the generator-operator correction also serves France, which the code does not do; the France path is hard-coded by country name, against the organisers' "open set" instruction, and the document mentions this only as a limitation; 19 s in the text versus 38 s in the manifest; and a CV-LB gap argument that ignores France's share. |
| **Overall** | **8 / 10** | |

**Verdict.** On methodology this package is clearly top-50 and plausibly top-10: few teams will show a label-free measurement of train/test shift, a paired acceptance rule, a decoy-cloning stress test, negative results with evidence, and a one-model ablation showing that the stage chain is not needed. What separates it from a polished top-10 write-up is readability, not substance. The document reads like an audit log: it is precise and dense, and it is organised for a verifier, not for a judge with 5 minutes. The top five edits below fix that without new experiments, in about 2-3 hours, and should leave the document shorter than it is now.

## Numbers I verified

| Claim in the document | Check | Result |
|---|---|---|
| Train 2.21M S1, 10.3M pool, 3.46 matches per S1; 5.6% with no match | organiser TSVs | 2,206,821; 10,320,219; 3.461; 5.58%: match |
| Test 1.73M S1 (India 47%, US 38%, France 15%), 9.97M pool | organiser TSVs | 1,732,544; 46.8 / 38.3 / 15.0%; 9,969,589: match |
| CV 0.99184, US 0.99164, India 0.99213, fold SD 0.00003, oracle 0.9980, precision 0.9993, recall 0.976 | `029f6_final/results.json` | match |
| FP 12,987 → 5,564; FN 302,721 → 186,753; error table sums | derived from precision, recall and 7,638,365 true pairs | match (FP 5,564 = 2,369+814+303+77+2,001; FN 186,753 = 103,587+51,473+31,693) |
| Floor 0.6 raises 0.97008 → 0.97083 | `005n` / `005f` results | match |
| Decision rule 5·Σp/(Σp+4k) vs Π(1−p), one owner, floor | `er_model.decide` | match; the expected-F0.5 algebra is correct |
| Folds stratified by country × match count | `er_model.folds` | match (0, 1, 2, 3, 4+) |
| Marker score (n_d+1)/(n_d+n_0+10); lift logit − log(Σn_d/Σn_0); ten shifts | `er_feat6.py`, `er_lift.py` | match |
| Cross-encoder held-out AUC 0.9996 | `ce/ce_train.py` docstring | supported |
| France ≈ 0.981 implied | exact test shares | 0.9805 (per-country CV) to 0.9809 (pooled CV): fine |
| 11.84M test candidates, 6.83 per S1; 58,364 scored but never matched | README, manifest | match (11,837,012 − 11,778,648) |
| Final step "writes them in 19 s" | `run_manifest.json` | **38 s recorded** |
| Generator-operator features work "on the unlabelled country" (Core Innovation) | `er_comb.py` line 28 filters `country_n != "france"` | **not applied to France** in our family |

Extra numbers the team can use (computed from the shipped files): the pair recall of the final train candidates is 1 − 51,473 / 7,638,365 = **99.33%**. The reduction ratio against all within-country pairs is 1 − 11.84M / 6.72·10¹² = **99.9998%** on test. For the submitted file, emitted pairs per S1 are **US 3.39, India 3.38, France 3.32**, against 3.38 for train out-of-fold decisions, and the empty-answer rate is **5.76% / 5.76% / 5.80%**, against a true train rate of 5.58%.

## Findings by question

1. **Completeness.** All template fields are filled with specifics, none generic. The template's "Candidate pairs generated: [total]" is answered with a table, but the middle row has no numbers. The organisers say they use `candidate_pairs.tsv` to judge "recall ceiling, reduction ratio"; the document gives oracle F0.5 but no pair recall of the final set and no reduction ratio.
2. **Technical depth.** This is strong throughout. Problem analysis (2.1) is the best section: the shift histogram argument and the label-free 1.94x measurement are exactly what a judge wants. The validation protocol is above top-team norms. Weak points: (a) "Every stacked input is out-of-fold" (2.2) is qualified 60 lines later in 4.2 item 6; (b) many experiments were accepted on the same five folds, so some selection optimism is unavoidable, and the document does not mention it; (c) the France path is described rule by rule (4.3), but the pipeline sketch does not show which stages France skips (the meta-model and the generator-operator correction are US/India only).
3. **Originality.** The generator framing is original and well evidenced. Two ideas would impress a judge but are under-sold: using a decoy house shift as a *label-free noisy label* for word markers (weak supervision), and cloning decoys to test density to stress-test features. The second-family ensemble and the e5 cross-encoders read as standard; that is fine.
4. **Clarity.** The main problems are the dense executive summary; the 13 uses of "band", 13 of "route" and 11 of "second family" before or without an early definition (the glossary sits at the end of 2.2); the pipeline sketch sitting in 4.1 rather than on page 1; and 4.2 item 6 mixing architecture, motivation, provenance and a leakage caveat. There is also one internal inconsistency: the false-positive bullet in Section 5 lists decoy types first, but the table below it shows that ownership errors (43%) and "other" (36%) dominate, with decoy shifts at only 5%. CV figures for the empty-address correction also differ by section: 0.99177 → 0.99184 in Limitations (3), versus its introduction at the 0.99079 row in B.1, where the flat LightGBM takes 0.99182 → 0.99184. Both are true in different orders, but a reader cannot reconcile them.
5. **Credibility.** The numbers are exact and the caveats honest. Flags: see the last two rows of the verification table; Limitation (5), which admits the France path is keyed on the country name without framing it against the organisers' explicit "treat country as an open set" instruction; and the B.1 footnote's reading of the late CV-LB gap (0.0011-0.0021) as decoy-density evidence, when France alone at about 0.981 and 15% of test accounts for about 0.0016. The text does not read as boilerplate or auto-generated: it has no filler or stock phrases. Its uniform, clipped register and dense cross-references can still feel machine-edited, and shorter sentences on page 1 fix most of that.
6. **Against a top-10 write-up.** Present: pipeline description, ablation (B.2), progression (B.1), negative results (B.3), error taxonomy with examples, runtime/compute, limitations, reproducibility notes. Missing or weak: a key-numbers box on page 1; a pipeline diagram near the top; a per-idea contribution view (B.1 is cumulative and 15 rows long); a characterisation of the 36% "other" false positives; the label-free guard shown as an actual table for the final file; candidate pair recall and reduction ratio. The heavy compute (30-55 CPU h, 15-20 GPU h, 120 GB) is disclosed and partly defended by the "production version" sentence.

## Prioritised edits to Documentation_template.md

### HIGH (page 1 and consistency; about 1.5 hours)

**H1. Executive summary: 3 plain sentences plus a key-numbers box.** Replace the whole paragraph under "## 1. Executive Summary":

> Current: "We model the generator behind the data: true matches are noisy copies of an S1 record, and unmatched pool records are near-copy decoys made by a few operators (a house number shifted by one of ten steps, one inserted marker word, a swapped legal form), which the test set contains at about 1.9x the train rate. Candidates come from TF-IDF retrieval and exact keys (train oracle F0.5 0.998); LightGBM models on pair-local features, including label-free decoy-marker scores that also work for unlabelled France, score each pair, a cross-fitted meta-model combines them with fine-tuned e5 cross-encoders and a second LightGBM family, and an expected-F0.5 rule with one owner per pool record picks each answer. US and India changes were accepted only on a paired 5-fold CV test plus a label-free test-shift guard (CV 0.99184); France rules were set by leave-one-country-out tests and label-free counts (public leaderboard 0.990203)."

> Proposed: "We treated the data as the output of a generator: true matches are noisy copies of a Source-1 (S1) record, and unmatched pool records are decoys made by a few fixed edits (house number shifted by one of ten steps, one inserted word, a swapped legal form), which the test set contains at about 1.9x the train rate (measured without labels). The pipeline is TF-IDF and exact-key blocking, LightGBM pair models with label-free decoy-marker features, fine-tuned multilingual-e5 cross-encoders, a meta-model over two independently built model families, and an expected-F0.5 decision that gives each pool record at most one owner. Every US/India change had to beat its parent on paired 5-fold CV and pass a label-free test-shift check; France, which has no labels, was set by leave-one-country-out tests and label-free counts.
>
> | Key number | Value |
> |---|---|
> | 5-fold CV macro F0.5, US + India (fold SD) | 0.99184 (0.00003) |
> | Public leaderboard | 0.990203 |
> | Recall ceiling: oracle F0.5 of the final train candidates | 0.9980 |
> | Test candidate pairs | 11.84M (6.83 per S1) |"

**H2. Make the false-positive text agree with the table.** In Section 5:

> Current: "**Common false positives (wrong merges):** decoys that keep the name and address but shift the house number by a typical step or insert a generic business word; legal-form swaps; unit-number changes in the same building. Out-of-fold false-positive pairs fell from 12,987 (band model) to 5,564 (final), and pair precision is 0.9993."

> Proposed: "**Common false positives (wrong merges):** out-of-fold false-positive pairs fell from 12,987 (band model) to 5,564 (final; pair precision 0.9993). The largest group (43%) is ownership errors: the pool record is a true match of another S1, usually a name twin, and the one-owner step gives it to the wrong S1. Surviving decoys mostly carry an inserted business word (15%) or a decoy house shift (5%); legal-form and unit conflicts are 1%. The other 36% are unmatched records with none of these edits, mostly [one phrase from a 50-pair sample]."

Only the team can fill the bracket. If there is no time to sample, write "not characterised further".

**H3. State exactly what France gets (Core Innovation and pipeline sketch).** In `er_comb.py` the generator-operator correction filters out France, and in `er_meta.py` the meta-model is US/India only. The current wording implies otherwise.

> Current: "**Core Innovation:** label-free decoy-marker scores and generator-operator features that model how the synthetic noise and decoys were produced, re-centred per country so the same features work on the unlabelled country."

> Proposed: "**Core Innovation:** modelling the data generator: (1) decoy-marker scores learned without labels, using a decoy house shift as a noisy label, and re-centred per country so they also serve France; (2) generator-operator features that encode which edits link a pair (US/India correction); (3) label-free checks of every change against the test's higher decoy density."

In the 4.1 sketch, replace

> "-> band model (all features) -> lexical correction -> generator-operator correction"

with

> "-> band model (all features) -> lexical correction -> generator-operator correction (US, India)"

and add one line after the meta-model line (the team should confirm it against `029f5_final` / `024_keys` before pasting):

> " France: band model -> lexical correction -> + added-route pairs -> blend with the second family and the cross-encoder logit -> rename rule, word veto -> empty-address correction (4.3)"

**H4. Frame country handling against the organisers' "open set" rule.** The organiser README says to treat country as an open set and not to hard-code the pipeline to {US, India}. Limitation (5) currently reads like an admission, with no design rationale:

> Current: "(5) The France path is selected by the country name; a new unlabelled country would need the same switch."

> Proposed: "(5) Countries with training labels use the label-trained models; a country without labels takes the unlabelled path of 4.3 (label-free marker tables built from its own candidates, the two-family blend, corrections fitted without the country flag). Blocking and marker tables are built per country for whatever countries appear, and every test S1 gets a row. In the code the unlabelled path is switched on by the country value `france` (in `er_lift`, `er_meta`, `er_comb`, `er_fix`, `er_france`, `er_frwords`), and French address rules in `er_norm` (street types, BP/CEDEX) are language knowledge, not labels; another unlabelled country would need its value added there."

Verify the module list and the "every country" claim before pasting. I found `france` literals in exactly these modules.

**H5. Put the map before the territory.** Two moves, with no new text: move the 4.1 pipeline code block to directly under "**Approach Type** / **Core Innovation**" in 2.2, and move the "**Glossary.**" paragraph to directly under it, so both come before the validation protocol. In the glossary, drop "*Chain probability*" and write "our family's final probability" at its two uses. Readers then meet "band", "route", "LOCO" and "name twin" after their definitions.

### MEDIUM (evidence and precision; about 1-1.5 hours)

**M1. Show the label-free guard for the submitted file.** This is the cheapest strong evidence for France. Insert after the F_0.5 bullet in Section 5:

> "**Label-free check of the submitted file** (no labels used; train = out-of-fold decisions):
>
> | | Train OOF (US + India) | Test US | Test India | Test France |
> |---|---|---|---|---|
> | Emitted pairs per S1 | 3.38 | 3.39 | 3.38 | 3.32 |
> | Empty answers | [x]% (true 5.58%) | 5.76% | 5.76% | 5.80% |
>
> France, built without labels, emits at the rate of the labelled countries."

I computed the test values from `output/matching_results.tsv`, and the train 3.38 from recall × 7,638,365 true pairs + 5,564 FP. The team should fill in the train empty rate, and per-country train values if they have them.

**M2. Complete the blocking numbers (organisers: "recall ceiling, reduction ratio").** Replace the row

> "| Other routes (character retrieval, extra keys, reverse retrieval) | proposed pairs cut by the route classifier at p > 0.001 | same |"

with the actual proposed and kept pair counts and the oracle gain (0.994 → 0.998). After the table, add:

> "Pair recall of the final train candidates: 99.33% (51,473 of 7,638,365 true pairs missing). Reduction ratio against all within-country S1 × pool pairs: 99.9998% on test (11.84M of 6.72·10¹²)."

Also relabel the row "Final candidate set (`candidate_pairs.tsv`)" as "Final candidate set (test: `candidate_pairs.tsv`)"; the train column is not in that file.

**M3. Put the CV caveats next to the CV claim.** In 2.2:

> Current: "- 5-fold CV over the 2.21M train S1, stratified by country × match count. Every stacked input is out-of-fold. Final fold SD: 0.00003."

> Proposed: "- 5-fold CV over the 2.21M train S1, stratified by country × match count (0-4+). Each stacked input is out-of-fold, but the second family uses its own folds and the cross-encoder halves are not nested in the meta-model folds, so the meta-model CV may be slightly optimistic; the CV-LB gap did not widen when the meta-model arrived (about 0.002 before, 0.0017 after, B.1). All decisions reused the same folds, so the final CV also carries some selection optimism. Fold SD: 0.00003."

Then delete from 4.2 item 6: "The second family uses its own entity folds (crc32 of the S1 id), and the cross-encoder halves are not nested in the meta-model folds, so this CV may be slightly optimistic."

**M4. Split 4.2 item 6** (about 200 words) into two items. Keep "*Meta-model* (US, India; 127 leaves, 600 rounds, cross-fitted on our folds) over the union of our pairs and the second family's. Inputs: ..." as item 6. Make the second family its own item: "*Second family* (`src/stack`): an independent pipeline the team built in a separate environment (pandas, Python 3.11, own folds) so that the meta-model combines different errors; it has its own normalisation, transliteration and blocking, a two-stage LightGBM with name-specificity and label-free decoy features, and a stage-3 stack over three e5 cross-encoders; for France it adds a cross-country e5-base cross-encoder, a decoy stacker and a marker veto." Cut the sentence "Both families were written by the team during the challenge; the second was built on purpose ... generator-operator features)", which the new item now covers.

**M5. Explain why within-S1 rank/margin features survive the "no S1 context" rule.** 2.1 says S1-context features were dropped, then keeps "rank and margin ... among the S1's own candidates". Replace the final bullet of 2.1 (from "Consequence: features that judge a candidate ..." to the end) with:

> "- Consequence: features that judge a pair by the *content* of its S1's other candidates win in CV but emit decoys on test, where each S1 has about twice as many decoy siblings; both such families were dropped (B.3). Context enters our family only through earlier-stage *probabilities* (a pair's rank and margin among its S1's candidates and among the S1 that claim the same pool record), [which passed the same label-free guard]. The second family keeps one content-context group, a pool record's similarity to the S1's confident matches (stage-2 p ≥ 0.9); the meta-model that reads it passed the guard before upload."

Keep the bracketed clause only if it is true.

**M6. Correct the CV-LB gap argument.** In the B.1 footnote:

> Current: "The CV-LB gap fell from 0.011 (third row) to 0.009 with the pair features and 0.004 with the decoy markers, and stayed between 0.0011 and 0.0021 afterwards, which fits the label-free finding that test is decoy-denser."

> Proposed: "The CV-LB gap fell from 0.011 (third row) to 0.009 with the pair features and 0.004 with the decoy markers, which fits the label-free finding that test is decoy-denser. The later gap of 0.0011-0.0021 is about what a lower France score alone produces (15% of test at about 0.981 costs 0.0016), so it cannot separate the two causes."

**M7. Reconcile the order-dependent gains.** Add to the B.1 footnote: "Rows are in acceptance order and gains depend on order: in the final build order (README, Verification) the meta-model, including the flat LightGBM input, gives 0.99177 and the empty-address correction adds +0.00007 (0.99184)."

**M8. Describe the stress test and the France word thresholds**, the two places where a judge may suspect hand-tuning.

(a) In 2.2, after "were re-scored with train decoys cloned to about 2x.", add "[how a clone is made, e.g. each unmatched pool record duplicated under a new id with a fresh decoy shift]. Graph features lost 0.037 (B.3); the re-centred marker scores [result]."

(b) In 4.3, after "split by equal house number versus a decoy shift.", add "Words with at least 300 shifted insertions are vetoed at a shift share ≥ 0.97 and treated as dual-use at 0.60-0.97 (thresholds fixed in `er_frwords.py`)." These thresholds are the script's defaults, which I verified.

**M9. Conclusion to 3 sentences** (the template asks for 2-3). Keep the first sentence (idea → gain), the lesson sentence ("The most important lesson ... fail on test.") and the production sentence. Cut "One LightGBM on all 205 features matches the whole correction chain ... not the stage structure." (B.2 already says it). Optionally add a "Δ CV" column to B.1 and bold its four largest rows, so a judge sees the contribution of each idea without reading prose.

### LOW (polish; about 30 minutes)

- **L1.** Appendix A: "writes them in 19 s once the cached probabilities exist" → "writes them in under a minute (38 s in `run_manifest.json`) once the cached probabilities exist".
- **L2.** 4.4: add the derivation in one line: "With E[TP] = Σp(top k) and E[T] = Σp, F0.5 = 1.25·TP / (0.25·T + k) becomes 5·Σp(top k) / (Σp + 4k)." This makes the rule visibly principled.
- **L3.** 4.2 item 1: add "each fold model is trained on a 50% entity sample of the other folds" (`TRAIN_FRAC = 0.5` in `005n`/`006`).
- **L4.** 2.1: "Decoys never have an empty address." → "In train, no unmatched pool record has an empty address." Qualify similar absolute claims (for example "S1 records never have an empty address") with "in train", or with "in train and test" if you checked both.
- **L5.** 2.1: the 1.94x figure is given for US only; add the India figure, or say why India is not measured.
- **L6.** 5: "would imply France ≈ 0.981" → "would imply France ≈ 0.980-0.981" (exact test shares give 0.9805 with per-country CV, 0.9809 pooled).
- **L7.** 2.2: "Label-free test guard before every upload" is only safe if the guard existed from the first upload; otherwise write "from upload N on".
- **L8.** Replace "pair-local features" and "cross-fitted" in the page-1 text with plain words; keep the technical terms for Sections 3-4.

Net effect: H1 and M4 cut about 120 words, M9 about 60, M5 is length-neutral, and M1/M2 add about 90. The document ends slightly shorter and much easier to skim.

## Package notes outside the document (only if there is time)

- `README.md`, "Quick check": "the last command of the list below" should say "above".
- `run_manifest.json` has `"n_entities": 0`, `"n_train_pairs": 0` and `sample_sha256` equal to the hash of an empty string. This is correct for a `--from-probs` run, but it looks like a glitch. A one-line README note ("fields for training are empty because the final step only decides on cached probabilities") pre-empts the question.
- Limitation (6): the scripts behind 1.94x, the decoy-clone result (−0.037) and the guard (3.4x / 5.8x) are not shipped. Shipping just the guard script, which is short and label-free, would make the headline shift claim checkable by a judge.
