# Evaluation B: methodology and documentation quality

Package: `submissions/final_package/Apprentice_submission.zip` (unzipped to scratch, zip and repo untouched). Public LB 0.990203.
Lens: methodology and documentation. Bar: top 50 of about 1,000 teams (top 5%), so the scoring is strict.
Main document: `Documentation_template.md` (2,169 words). I checked it against `code/business_entity_resolution/README.md`, `src/stack/README.md`, the `attempts/*/results.json` files, the code, and the organisers' data where a claim needed it.

## Verdict

**Does it clear a top-50 bar? Not yet. It is borderline as written. The methodology clears the bar; the write-up does not.**
The package has three things that most top-5% write-ups lack: label-free decoy statistics learned from the candidate graph, a paired significance rule for every change, and a label-free test-shift guard that rejected features which won in CV. A harsh reviewer can still cut it because of:
(1) a factual error in the error analysis;
(2) a contradiction between Section 2.1 and Section 4 about which features look at an S1's other candidates;
(3) a 363-word Section 4 paragraph that no judge will parse in five minutes;
(4) France rules that read as hand-tuned on the test set, because the document never says how the word lists and weights were chosen;
(5) a README step that fails as shipped.
Much of the strongest evidence exists in the team's research notes but not in the document: the decoy-dense stress test, the one-model ablation, the count estimator validated on US data, and the loss decomposition.
After the HIGH edits below (about 3-4 hours), this write-up should sit safely inside the top-50 tier on this lens.

| Criterion | Score | One-line reason |
|---|---|---|
| Completeness | 8 / 10 | Every template section has real content. Missing: limitations, runtime/cost, a pipeline figure, negative results. |
| Technical depth and soundness | 8 / 10 | Out-of-fold stacking, stratified entity folds, a correct expected-F0.5 rule with one owner per pool record, and leave-one-country-out (LOCO) tests for France. Weak points: the France rules are not justified in the text, and the two families use different fold schemes. |
| Originality / insight | 7 / 10 | The label-free decoy markers and the reverse-engineered decoy shift set are genuinely original. The cross-encoder, the LightGBM stack and the meta-ensemble are standard. The best insights are not in the document. |
| Clarity and structure | 4.5 / 10 | Dense log-style prose (42 semicolons, 72 parentheses). The executive summary is 158 words against the requested 2-3 sentences. There is no diagram, and terms like "band", "route", "lifted", "fair-sample", "France-like parents" and "name-twin" are never defined. |
| Honesty and credibility | 6 / 10 | Good: the CV-LB gap is explained, the fair-sample checks are shown, and the reconstructed steps are disclosed. Bad: one factual error, one internal contradiction, one stale oracle number, one conclusion that overclaims, and one broken reproduction step. |
| **Overall (this lens)** | **6.5 / 10** | **Borderline for the top 50. About 8 / 10 after the HIGH and MEDIUM edits.** |

## What a judge sees in five minutes

**What impresses (keep it, and make it more visible):**
- **Label-free decoy-marker scores.** A word is scored (n_d + 1)/(n_d + n_0 + 10). Here n_d counts candidate pairs whose pool name adds or drops the word and whose house number is shifted by a decoy step; n_0 counts the same with equal house numbers. This is weak supervision built from the generator's own signature. It needs no labels, so it transfers to France, and the per-country lift puts all countries on one scale (verified in `src/er_feat6.py` and `src/er_lift.py`). This is the strongest original idea in the package, but in Section 4 it is buried in a parenthesis.
- **Reverse-engineering the decoy generator:** the shift set {1,2,3,4,5,7,9,11,13,21}, and the fact that decoys never have an empty address (`src/er_fix.py` docstring).
- **The CV-vs-test trap.** Features that use an S1's other candidates win in CV but emit decoys on test, where decoys are 1.9x denser. The team caught this with a label-free guard. This is the kind of lesson judges remember.
- **Validation discipline:** a paired per-fold rule (> 2 SE and at least 4 of 5 folds), stratified entity folds, out-of-fold stacked inputs throughout, and LOCO for France.
- **Correct decision theory.** `decide()` in `src/er_model.py` implements expected F0.5 = 5·Σp_top-k/(Σp + 4k) against P(no match) = Π(1 − p), with one owner per pool record. The document states this formula correctly.

**What reads as generic:** TF-IDF plus exact-key blocking, fuzzy-ratio features, a fine-tuned e5 cross-encoder, LightGBM stacking and a meta-model over two families. That is fine, but it should not take up most of the words.

**What reads as a red flag:**
- A 17-row progression table in which the late rows gain +0.000015 CV and +0.000002 LB. That looks like leaderboard chasing unless the text says the LB was not used to choose changes.
- France handled by named word lists and fixed weights with no stated derivation.

## Evidence checks (claims against the package)

| Claim in the document | Source checked | Status |
|---|---|---|
| CV 0.99184 (US 0.99164, India 0.99213) | `attempts/029f6_final/results.json`: 0.991836, fold SD 0.000033 | OK |
| 11.78M test candidates, 6.80 per S1 | counted in `output/candidate_pairs.tsv`: 11,778,648, 6.798 per S1 | OK |
| Blocking recall 98.1%, oracle 0.994 | `005f`, `006`, `016` results: oracle 0.99398 | OK |
| "oracle macro F0.5 of 0.9969" after the extra keys, reverse retrieval and second-family pairs | `029f6_final/results.json`: oracle **0.99798** on 13.37M train pairs | **Stale.** 0.9969 matches an earlier parent. |
| FP 13,043 → 5,582 and missed 302,550 → 187,067 | Derived from pair precision/recall in `016` and `029f6` results, with 7,638,365 gold pairs: 016 about 12,990 FP / 302,700 missed; final about 5,560 FP / 186,700 missed | Close, but they do not reproduce from the shipped results. The source is not stated. |
| FN: "S1 with an empty or "None" address whose name is shared by other S1" | Train S1 has **0** empty or "None" addresses out of 2,206,821; S2 has 168,967 | **Wrong.** The empty address is on the pool record. The document's own Section 4 says so ("empty-address pool records"). |
| FN: "pairs never retrieved by blocking (the largest remaining source)" | Oracle 0.99798 means unretrieved pairs cost at most 0.0020 of the 0.0082 loss | **Contradicted** by the shipped results. |
| 2.1: "Every feature of the main pipeline compares one pool record with its own S1" | `src/er_feat4.py` has S1-side rank, gap to best candidate, count with p ≥ 0.5, and Σp; Section 4 lists "among the S1's own candidates" | **Internal contradiction.** The banned family is content coherence (`er_feat3.py`), not score ranks. |
| Cross-encoder = multilingual-e5-small | The README and `027p_meta` inputs also use **e5-base** (`ce_base`, through `ce2`) | Incomplete |
| Conclusion: decoy-marker and generator-operator features "lifted CV from 0.974 to 0.988" | Table B: that span also includes the band model, char retrieval, lexical correction, French parser, pool-side competition and the cross-encoder | Overclaim |
| README: `python src/er_france.py 029f5_final` | `attempts/029f5_final/` is **not in the zip**. `load_attempt` fails, and `030q_fix` reads its output. | **The reproduction chain breaks.** The folder exists in the repo; add it. |
| Appendix A: "`attempts/` (the settings of every stage used, one folder per stage)" | 029f5_final is missing; `027p_meta` says `PREDICT ... 027l_meta` and `PRE_STEPS ... 029f2_final` | Stale strings |
| Two families' folds | Main family: StratifiedKFold (country × match count). Second family: `crc32(S1 id) % 5`. | Disclosed only as "its own entity folds". Meta-model CV may be slightly optimistic. |

## Cut risks a harsh top-50 reviewer would use (ranked)

1. **The factual error in the FN analysis** (S1 empty addresses do not exist). It suggests the authors do not know their own error profile. It also contradicts the document's own Section 4.
2. **The sibling-feature contradiction** (2.1 against Section 4 and the code). The document's central lesson is "we banned features that use other candidates", and then Section 4 lists such features.
3. **The Section 4 wall of text.** The judge cannot find the pipeline, the France logic or the decision rule without reading 363 words in one paragraph.
4. **France looks test-tuned.** VETO_WORDS, W = 0.5, C = 0.1, the rename rule and the country lift appear without derivation. France is 15% of test and the only unlabelled country, so every judge will look here.
5. **Reproducibility.** The missing attempt folder breaks the documented chain. There are "reconstructed" steps and two Python environments. A reviewer who tries step 105 of the README stops there.
6. **Micro-gain rows and one LB number per row** read as leaderboard-driven iteration, and there is no ablation that isolates components.
7. **Numbers that disagree with the shipped `results.json`** (oracle, FP/FN counts).
8. **No limitations, runtime or cost, and a leftover template instruction** at the end ("**Note:** Teams can modify sections…"). These look unfinished.

## Against a strong top-10 write-up: what is missing

| Top-10 write-ups usually have | This document |
|---|---|
| One pipeline figure or diagram | None |
| A 2-3 sentence summary with the headline numbers | 158 words, no oracle or recall headline |
| A clean ablation (remove one component, show the delta) | Cumulative progression only. The one-model ablation exists in the research notes (r19) but not here. |
| A dedicated section on the unseen country | France is spread over Sections 2.1, 4 and 5 and the table, with no derivation |
| Quantified error buckets | Qualitative bullets, one of them wrong. A loss decomposition exists (r20) but is not used. |
| Negative results | One sentence ("measured this three times") with no numbers |
| Runtime, compute and cost | In the README, not in the document. No totals. |
| Limitations and future work | None |

---

## Prioritised edits to `Documentation_template.md`

Numbers marked **[notes]** come from the team's research notes in `docs/research/`, outside the zip. Check each against its log before pasting. All other numbers are verifiable inside the zip.
Length budget: aim for at most 3,000 words (now 2,169). The HIGH edits roughly replace what they remove. The MEDIUM additions add about 600 words.

### HIGH (do all; about 3-4 hours)

**H1. Fix the false-negative bullet (factual error, and it contradicts the oracle).** Section 5, current:
> - **Common false negatives (missed matches):** pairs never retrieved by blocking (the largest remaining source); true pool records claimed by another S1 with a near-identical name; S1 with an empty or "None" address whose name is shared by other S1 (no distinguishing evidence); records with one true match, which score lowest.

Replace with:
> - **Common false negatives (missed matches):** of the CV loss (1 − 0.99184 = 0.0082), at most 0.0020 comes from true pairs never retrieved (candidate oracle 0.9980). The rest are true candidates that score below the floor. The dominant case is a pool record with an empty or "None" address whose normalized name is also claimed by another S1 of the same country (a *name twin*). With no address the twins are interchangeable. On the last analysed parent (CV 0.99051), these records held two thirds of the in-candidate headroom (+0.0042 of +0.0064 oracle F0.5) **[notes r20]**. S1 records never have an empty address. S1 with a single true match score lowest (0.973, against 0.990-0.994 for two or more), because one miss costs the whole entity.

**H2. Remove the sibling-feature contradiction and back "three times" with numbers.** Section 2.1, current:
> So any feature that looks at an S1's other candidates (coherence, sibling agreement) wins in CV and emits decoys on test; we measured this three times and banned such features. Every feature of the main pipeline compares one pool record with its own S1; competition features only ask how strongly other S1 claim the same pool record.

Replace with:
> Features that judge a candidate by its agreement with the S1's other candidates therefore win in CV and emit decoys on test. We saw this three times and dropped each family (Table B.3). A pool-record graph gained +0.0033 in normal CV but lost 0.037 (0/5 folds) once decoys were duplicated to test density. An S1-context stacker gained +0.0028 CV but emitted 3.4x (US) and 5.8x (India) the train rate of extra pairs on test **[notes r7, r4]**. What remains is pair-local content features plus score-based context: the rank and margin of a pair's earlier-stage probability among the S1 that claim the same pool record and among the S1's own candidates. No feature compares a candidate's name or address with the S1's other candidates.

**H3. Add evidence for the two key EDA claims.** Section 2.1: after the sentence ending "a changed unit number.", add:
> On name-similar train pairs, non-matches spike at exactly these ten shifts (13.6k-15.8k pairs each on a 10% sample, against 1k-3k at every other difference, including all negative ones) **[notes r9]**. Decoys never have an empty address.

Then replace "- Test has about 1.9x the decoy density of train, and decoys come in clusters that share one perturbed house number." with:
> - Test has about 1.9x the decoy density of train. We measured this without labels: per S1, candidate pairs whose house numbers differ by one of the ten shifts are 1.94x as frequent on test as on train (US), while every other difference keeps the train rate **[notes r9]**. Test decoys also come in clusters that share one perturbed house number.

**H4. Split Section 4 into readable parts, and add a diagram and a France subsection.** Replace the whole "**Model type:** LightGBM throughout: stage 1 (127 leaves, 600 rounds), … so it mainly removes decoy-like France pairs." paragraph (363 words) with the following.

> **4.1 Pipeline**
> ```
> S1 + pool records, per country
>  -> normalize (native-script words -> Latin, dictionary learned from train pairs)
>  -> blocking: word TF-IDF top-40 + exact keys + char 3-gram route + extra name keys + reverse retrieval
>  -> stage 1 LightGBM -> stage 2 LightGBM (+ competition features); keep p >= 0.001
>  -> band model (all features) -> lexical correction -> generator-operator correction
>  -> e5 cross-encoder stack on the uncertain band
>  -> meta-model over both model families (US, India) | France blend (4.4)
>  -> empty-address correction -> expected-F0.5 decision (4.5)
> ```
> **4.2 Models** (LightGBM unless stated; every train input is out-of-fold)
> 1. *Stages 1-2* (127 leaves, 600 rounds). Stage 2 adds competition features from stage-1 out-of-fold probabilities.
> 2. *Band model.* One model on stage-2 pairs with p ≥ 0.001 and every feature family.
> 3. *Retrieval-route classifier.* It scores pairs added by char retrieval, extra keys and reverse retrieval; pairs are kept at p > 0.001.
> 4. *Two corrections boosted from the previous logit.* First 42 cross-fitted lexical likelihoods, then generator-operator plus pool-side competition features (255 leaves, 800 rounds).
> 5. *Cross-encoder.* multilingual-e5-small and e5-base, fine-tuned as pair classifiers on two S1 halves split by fold, so every train score is out-of-fold (held-out AUC 0.9996). A small LightGBM stacks the score on 0.003 ≤ p ≤ 0.997.
> 6. *Meta-model* (US, India; 127 leaves, 600 rounds, cross-fitted on our folds). It runs over the union of our pairs and a second LightGBM family's pairs (src/stack). Inputs: both logits, their min / max / sum / difference, both families' cross-encoder scores, a few pair features, the logit of one flat LightGBM on all 205 band features, and pool-side competition. The second family uses its own entity folds (crc32 of the S1 id), so this CV may be slightly optimistic. The LB gain confirmed it.
> 7. *Empty-address correction.* Boosted from the meta logit using name-twin structure among a pool record's claimers. Only empty-address records may gain probability (decoys never have an empty address); every other pair can only lose.
>
> **4.4 France (no labels).** Nothing is trained on France. Each rule was chosen by LOCO (fit on US, score India as if it were unlabelled, and the reverse) or by label-free counts on test.
> - *Scale:* decoy-marker scores are re-centred per country as logit(score) − log(Σn_d/Σn_0). France's table is built from its own test candidates.
> - *Blend:* p = σ(0.5·logit p_main + 0.5·logit p_second) over both families' pairs [state how W = 0.5 was fixed]. The cross-encoder logit enters with weight 0.1 on band pairs (LOCO +0.00053 / +0.00052).
> - *Decoy-word veto:* pairs whose pool name adds *participations, holding, distribution* or *international* get p = 0. Each word is added about 34k times in pool names but appears 1-203 times in S1 names, and 94% of its insertions carry a decoy house shift **[notes r1, r17]**. The dual-use words *groupe, developpement, france* are not vetoed. A count estimator gives them a 0.93 truth rate at equal house numbers; on US "partners" the same estimator predicts 0.84 against a labelled 0.835 **[notes r17]**.
> - *Renamed business at the S1's own unique address:* max(blend, p_second) (LOCO +0.00033 / +0.00062).
> - *Empty-address correction* fitted on US + India without the country flag (transfer +0.00006 US→India, 5/5; +0.00002 India→US).

Keep the current Threshold line as **4.5**, and add: "The rule maximises expected F0.5 per S1 under the model's marginals. P(no match) is Π(1 − p)." (verified against `decide()`).

**H5. Make the validation protocol a visible block.** Section 2.2, current:
> Validation: 5-fold CV over S1 (stratified by country and match count), every stacked input of the main pipeline out-of-fold, and a change is kept only if it gains more than 2 standard errors and in at least 4 of 5 folds; before any upload, a label-free test guard compares emitted pairs per S1, empty-answer rate and house-conflict share per country.

Replace with:
> **Validation protocol**
> - 5-fold CV over the 2.21M train S1, stratified by country × match count. Every stacked input is out-of-fold. Final fold SD: 0.00003.
> - Acceptance: paired per-fold gain over the parent > 2 SE and positive in at least 4 of 5 folds.
> - Stress test at test density: features that could react to decoy density were re-scored with train decoys duplicated to about 2x [confirm scope].
> - Label-free test guard before every upload: per country, compare emitted pairs per S1, empty-answer rate and the share of emitted pairs with conflicting house numbers between test and train. A change that emits far more extra pairs on test than on train is rejected even if CV improves.
> - France: LOCO plus label-free counts (4.4).
> - The public LB was recorded after each upload and never used to choose a change. [Keep this sentence only if it is strictly true.]

**H6. Make the shipped numbers agree.** Section 3, current: "the candidate-recall ceiling on train rises to an oracle macro F0.5 of 0.9969." Replace with: "…rises to an oracle macro F0.5 of 0.9980 (13.37M train pairs; `attempts/029f6_final/results.json`)."
Section 5, current: "Out-of-fold false-positive pairs fell from 13,043 (band model) to 7,904 (final correction) and 5,582 (final), and missed true pairs from 302,550 to 187,067." Recompute these from the shipped files (they imply about 12,990 → 5,560 FP and 302,700 → 186,700 missed), or name the run they came from.

**H7. Package fix (outside this document, but it breaks the README chain the document points to).** Add `attempts/029f5_final/train.py` to the zip. The repo copy matches 029f6_final except for DEPENDS. In Appendix A, "the settings of every stage used" is only true after that.

### MEDIUM (about 2 hours; these lift originality and credibility)

**M1. Rewrite the executive summary to three sentences with numbers.** Replace the paragraph "A multi-stage blocking + gradient-boosting pipeline built around the data generator's noise and decoy operators. … uses a logit blend of the two families with a decoy-word veto." with:
> We treat the data as the output of a generator that adds noise to true matches and plants near-copy decoys, and we model that generator. Candidates come from word and character TF-IDF plus exact keys (train candidate oracle 0.998). LightGBM models score each pair with pair-local features, including label-free decoy-marker statistics that transfer to unlabelled France, and a cross-fitted meta-model combines them with fine-tuned e5 cross-encoders and a second LightGBM family. An expected-F0.5 rule with one owner per pool record picks each answer. Every change had to pass a paired 5-fold CV test and a label-free test-shift guard (CV 0.99184, public LB 0.990203).

This also fixes "score pairs with label-free features". Some features are cross-fitted on labels (lexical likelihoods, the native-script dictionary).

**M2. Fix the conclusion's attribution and add limitations.** Current:
> Modelling the generator, rather than generic string similarity, drove most of the gains: decoy-marker and generator-operator features lifted CV from 0.974 to 0.988 and the leaderboard from 0.963 to 0.985.

Replace with:
> Modelling the generator drove the largest single steps: pair features with house-shift relations (+0.009 CV), decoy-marker scores (+0.002) and generator-operator codes (+0.001). The second model family added +0.0027 CV (+0.0038 LB with the France blend). One LightGBM on all 205 features matches the whole correction chain (0.98779 both) **[notes r19]**, so the gains came from the features, not the stage structure.

Keep the "most important lesson" sentence, then add **Limitations**:
> (1) The pipeline has about 20 steps in two environments. Two steps of the second family were rebuilt from logs, and the GPU steps are not bit-identical on rerun.
> (2) France rests on LOCO proxies and label-free counts; its true score is unknown.
> (3) Empty-address pool records shared by name twins are a ceiling of about 0.004 F0.5 that no feature here can resolve.
> (4) Train candidates lack the character-route pairs that test has.

**M3. Add B.2, a one-model ablation (about 80 words and a 5-row table)** **[notes r19]**:

| Variant (US + India CV) | Macro F0.5 |
|---|---|
| Correction chain up to the cross-encoder stack (5 LightGBM stages) | 0.98779 |
| One LightGBM, all 205 features | 0.98779 |
| Same, without generator-operator and cross-encoder features | 0.98638 (−0.00141, 0/5) |
| One XGBoost, same 205 features | 0.98778 |
| Logistic GAM, 16 features | 0.98017 |

Add one sentence: the top gain shares are pool-side and S1-side margins, then the cross-encoder and decoy markers. This answers "is this complexity necessary?" before a judge asks.

**M4. Add B.3, negative results (4-5 rows, about 90 words).** Include the pool-record graph (+0.0033 CV, −0.037 at doubled decoys), the S1-context stacker (+0.0028 CV, 3.4x / 5.8x extra test pairs), the coherence features [fill from your log], broader reverse retrieval (it finds only 4.9% of the remaining misses **[notes r20]**), and the broad France veto (−0.0024 US for the same kind of rule in-country **[notes r17]**). Judges value this far more than +0.000015 rows.

**M5. Clean up Table B.1.**
(a) Add a footnote: "From the row '+ extra keys for every train S1', the train candidate set gains the pairs test already had, so CV rises by +0.001 for reasons other than the model. The fair-sample figure (paired CV on a fixed label-free 10% S1 sample where train and test candidates match) is the comparable gain."
(b) Merge the "+ single flat LightGBM as a meta-model input | 0.99184 (+0.000015) | 0.989992" row into the one above, or mark it "neutral (+0.000015, passes the paired rule)".
(c) Current: "| + empty-address / name-twin correction applied to France (final) | France only | 0.990203 |". In the CV column, write "transfer +0.00006 (5/5) / +0.00002", as Section 4 already says.
(d) Use one precision throughout: 5 decimals for CV, 6 for LB.

**M6. Add a runtime and cost line to Appendix A.** Example: "End to end: about N CPU hours on 12-vCPU / 96 GB machines (blocking, stage 1-2 CV and address retrieval 1-2.5 h each), plus about M Kaggle 2x T4 session hours (e5-small 170 + 22 min, e5-base 140 min, second-family cross-encoders about 47 min + 2 h + 4.7 h + 2 × 75 min). Test inference alone: X h." Fill N, M and X from your logs.

**M7. Define jargon on first use, or add a four-line glossary under 2.2.** Terms: *band* (pairs with 0.003 ≤ p ≤ 0.997), *route* (character-retrieval candidate source), *name twin*, *LOCO*, *fair sample*, *lifted* (re-centred per country). Replace "France-like parents" with "parents built as they are for France (country tables from their own pairs)".

### LOW (15-30 minutes)

- **L1.** Delete the leftover template line at the end: "**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth."
- **L2.** In Section 3, turn "**Candidate pairs generated:** …" into a 3-row table: raw union, train 98.0M (44 per S1) / test 80.5M (46 per S1); final candidate set, train 13.37M / test 11.78M (6.80 per S1); oracle 0.994 → 0.998. Keep the one-line definition of candidate_pairs.tsv.
- **L3.** Section 5, current: "France (15% of test) has no labels and scores lower, so the leaderboard sits below CV." "Scores lower" cannot be verified. Write instead: "If US and India held their CV on test, the public LB would imply France ≈ 0.981. The 1.9x test decoy density probably lowers US and India too, so this is only a bound."
- **L4.** Code hygiene a judge may see: `attempts/027p_meta/train.py` has `PREDICT ... 027l_meta` and `PRE_STEPS` citing `029f2_final`; `src/common.py` carries unused generic metrics (smape, mae, map@k) and an `mps` device default. Tidy these, or ignore them if time is short.

## Recommended order

H1, H2, H6 (30 min: removes the errors) → H4 (90 min: the biggest clarity gain) → H5, H3 (40 min) → H7 (5 min, package) → M1, M2 (40 min) → M3, M4, M5 (60 min) → the rest.
