VERDICT: REVISE — The largest risks are an inadequate blocking target, leakage through learned preprocessing and competing
  scores, and a decision rule that does not optimize macro F0.5.

  References below identify sections of your supplied plan; no implementation was supplied for file:line verification. Numerical
  effects are conditional score sensitivities, not measured gains. Costs are engineering estimates.

  1. HIGH — The blocking design can miss entire entities while exceeding 97% pair recall.
     Reference: §4 Pipeline 2; §2 match-count distribution and noise.

     Claim: Every proposed blocking key can fail on the stated corruptions. Fuzzy matching restricted to surviving blocks cannot
     recover a pair excluded by every key.

     Why: A changed house number disables two keys; a name typo defeats the exact space-free key; missing or misspelled locality
     tokens defeat the contextual key. Arbitrary block caps further penalize common names and addresses.

     More importantly, 97% pair recall can coexist with an oracle macro F0.5 of approximately 0.921: miss every one-match entity,
     then both matches for another 2.49% of entities. That consumes approximately 3% of gold pairs while losing 7.89% of entity
     scores.

     Change: Add an independent, country-wide retrieval pass using informative character-gram/token postings and bounded top-k
     retrieval. Measure its incremental recovery. Count join products before materializing them; refine or similarity-rank
     oversized blocks instead of dropping them or truncating by row order.

     Make macro oracle loss the blocking acceptance criterion:
     [
     F_{\mathrm{oracle},i}=\frac{1.25c_i}{0.25t_i+c_i},
     ]
     where (t_i) is truth count and (c_i) is retrieved truth count; true singletons score 1. Report losses by match count,
     country, source, corruption and cap exclusion. Start with an oracle-loss budget such as 0.005, then measure its compute cost.

     Expected impact: Potentially several absolute score points. The 0.079 example is a demonstrated possibility, not a forecast.
     Confidence: high in the failure mechanism; actual prevalence unknown.

     Cost: Approximately 4–8 hours for retrieval and diagnostics. Full indexes should be built on Kaggle; the Mac can run bounded
     query batches.

  2. HIGH — Gold-derived normalization needs the same fold isolation as the classifier.
     Reference: §4 Pipeline 1, “mappings learned from training gold pairs”; Validation.

     Claim: “Labels … are never used as features” does not prevent leakage through normalization, blocking or generator rules. The
     plan does not specify when those resources are fitted.

     Why: Learning a rare alias from a validation pair can make that very pair retrievable. The resulting blocking recall and
     classifier score both benefit from held-out supervision.

     Change: Split identities before mining mappings. Fit every gold-derived alias, substitution rule and operator statistic using
     outer-training labels only; freeze them before generating validation candidates. Apply the same boundary to inner threshold
     selection. Fit final resources on all training labels only after selecting the pipeline.

     Keep supervised mappings separate from label-free token frequencies. Record the training-ID split associated with each cached
     mapping and candidate artifact.

     Expected impact: Unknown. If leaked mappings rescue otherwise unretrievable matches for 1% of entities, CV inflation can
     approach 0.01. Confidence: high in the dependency; implementation unverified.

     Cost: Approximately 2–4 hours to establish fit/apply boundaries. Rebuilding candidates for every fold is expensive; use one
     clean development split initially.

  3. HIGH — Ordinary entity OOF predictions are insufficient for validating reverse ranks and exclusive assignment.
     Reference: §4 Pipelines 3 and 5; Scale plan; Validation.

     Claim: The plan does not define which competing S1 entities participate in validation or how their scores are produced. That
     omission can invalidate the reported effect of assignment.

     Why: With 10% of S1 active, approximately 92.6% of the full pool has no true owner among active queries, versus 26% in the
     complete training problem. Keeping the full pool preserves retrieval difficulty for individual queries but does not preserve
     reference competition.

     Fold-only assignment removes rivals. Adding training references with in-sample scores introduces unusually strong rivals.
     Pooling ordinary OOF scores creates another dependency: the model scoring a rival may have trained on the evaluated entity’s
     labels. These effects can bias scores in either direction.

     Change: Initially use deterministic, label-free similarity ranks. For a practical joint-decoder check, train on 10–20% of
     identity clusters and evaluate on a large, untouched graph containing the remaining identities, their gold-linked pool
     records and independently allocated distractors. Keep all competing validation references unseen by the scorer. Report its
     pool-size mismatch separately from full-pool retrieval diagnostics.

     If retaining full-graph outer CV, every model contributing competing scores or learned ranks must exclude that outer fold’s
     labels; training-reference rivals also need inner cross-fitting. Do not call pooled OOF assignment leakage-free without
     tracing these dependencies.

     Expected impact: Unquantified; correcting decisions on 1% of entities changes macro F0.5 by at most 0.01. Confidence: high
     that the validation graph must be specified.

     Cost: Approximately 3–6 hours for a large held-out graph evaluator. Nested graph cross-fitting is substantially more
     expensive; full graph work belongs on Kaggle.

  4. HIGH — The full-scale execution plan leaves predictable memory failures until too late.
     Reference: §3 compute constraints; §4 Pipelines 3 and 7.

     Claim: Chunking S1 rows does not bound pool-index memory, join expansion or global ranking state. “Move to Kaggle if the Mac
     runs out” risks losing the first useful experimentation cycle.

     Why: At just 20 candidates per test S1, there are 34.65 million pairs. An illustrative layout containing two uint32 IDs and
     forty float32 features occupies 5.82 GB, before strings, indexes, temporary allocations or LightGBM. At 50 candidates, it
     occupies 14.55 GB. Seven GB free disk also cannot accommodate that feature cache.

     Streaming is not an automatic guarantee: Polars documents that some operations fall back to in-memory execution. Polars
     streaming documentation (https://docs.pola.rs/user-guide/concepts/streaming/)

     Change: Establish the Kaggle execution path on day one. Use integer IDs, bounded candidate batches and streamed feature
     scoring. Discard features after scoring. Compute reverse ranks and assignment in a separate pass over compact pair scores,
     globally by pool ID—not independently inside S1 chunks.

     Benchmark representative queries against the full pool, including large blocks. Record peak memory, output bytes, candidate-
     count tails and end-to-end throughput before extrapolating runtime.

     Expected impact: Prevents a potentially submission-ending failure; no meaningful numeric F0.5 forecast. Confidence: high in
     the memory arithmetic.

     Cost: Approximately 2–4 hours for integration and measurement, plus implementation-dependent streaming work. Restrict the Mac
     to small feature batches and inspections.

  5. HIGH — France requires a separate uncertainty assessment; an automatic higher threshold is unjustified.
     Reference: §4 Pipeline 6; §2 country proportions.

     Claim: US→India does not establish France’s calibration shift. Raising the threshold can worsen performance if unfamiliar
     French text already depresses positive scores.

     Why: France is 15% of test: a 0.10 France-only score loss costs 0.015 globally. Retrieval failure, calibration shift and
     changed match prevalence require different responses.

     Ordinary CV also uses the wrong known-country mixture. A change gaining 0.01 in US and losing 0.01 in India improves the
     training-weighted score by 0.002, but reduces the known-country contribution to test score by 0.0009.

     Change: Run both US→India and India→US, with all supervised normalization fitted only on the training country. Separate
     candidate oracle loss from classifier/decoder loss. Compare the global threshold against a small country-agnostic confidence
     rule; require transfer evidence before adding an unseen-country offset.

     Report US and India changes using test weights 0.38 and 0.47, with France represented explicitly as an unknown sensitivity.
     Inspect France’s missingness, script, zero-candidate rates and block-cap losses before inference. Those diagnostics cannot
     establish French accuracy.

     Expected impact: A 0.02 French-only improvement contributes 0.003 globally. Actual direction remains unknown. Confidence:
     high that the proposed offset lacks evidence.

     Cost: Approximately 2–4 hours plus two bounded model fits. Small sampled feature matrices fit on the Mac; full retrieval
     diagnostics should run on Kaggle.

  6. HIGH — A global pair threshold and highest-probability assignment do not optimize macro F0.5.
     Reference: §1 Metric; §4 Pipeline 5.

     Claim: A pair’s value depends on the entity’s other predictions and possible truth count. Its probability alone is
     insufficient.

     Why: For a nonempty truth set,
     [
     F_{0.5}=\frac{5TP}{T+4K},
     ]
     where (T) is truth count and (K) prediction count.

     Consider a calibrated candidate with probability 0.6:
      - If it is the only possible match, predicting it has expected score 0.6, versus 0.4 for empty.
      - With three certain matches and this optional fourth, including it gives expected score 0.91579; excluding it gives 0.9625.

     No single probability cutoff produces both optimal decisions. Raw-probability conflict resolution likewise ignores different
     marginal gains across entities.

     Change: On honest held-out scores, compare empty and top-k predictions using a small rule family conditioned on top scores,
     confident-match count and ambiguity. Include explicit singleton probability. Resolve contested pool IDs using estimated
     marginal entity-score gain, with abstention available. Do not force predictions toward the mean count of 3.46.

     If negatives are sampled or class weights introduced, calibrate on the natural candidate distribution before interpreting
     outputs as probabilities. LightGBM explicitly warns that imbalance weighting can degrade probability estimates. LightGBM
     parameters (https://lightgbm.readthedocs.io/en/latest/Parameters.html)

     Expected impact: The fourth-match example gains 0.04671 per affected entity; at 10% prevalence, 0.00467 overall. Prevalence
     is unmeasured. Confidence: high in the objective mismatch; medium in achievable gain.

     Cost: Approximately 3–6 hours. Threshold and top-k experiments are inexpensive on compact scores; exact probabilistic
     optimization is unnecessary initially.

  7. HIGH — Assignment inheritance can amplify false positives and violate the candidate-output contract.
     Reference: §4 Pipeline 5, “inherits its assignment”; §1 output requirements.

     Claim: A strong pool-to-pool match does not justify unconditional assignment inheritance. Propagation may also introduce S1–
     pool pairs absent from the final scored candidate set.

     Why: A mistaken anchor can contaminate several predictions. With four true matches, one additional false positive reduces an
     otherwise perfect entity score to 0.8333. Shared addresses and generic names are particularly dangerous bridge evidence.

     Change: Treat pool-to-pool evidence as a candidate-expansion mechanism. Add the implied S1–pool edge, score it explicitly
     with direct and cluster-consistency features, and apply the normal acceptance/conflict rule. Start with one expansion hop
     from high-confidence anchors; do not repeatedly propagate newly inferred assignments.

     Generate candidate_pairs.tsv from the actual final scoring input. Define equivalent candidate provenance for the ensemble,
     then verify that every prediction belongs to that set. Run the official validator on the exact packaged outputs.

     Expected impact: Preventing one extra false positive on 1% of four-match entities recovers approximately 0.00167. Unbounded
     propagation can affect more entities; magnitude unknown. Confidence: high in the failure mechanism.

     Cost: Approximately 2–4 hours for bounded expansion and provenance checks. Pool-to-pool retrieval itself can be expensive; do
     not make it a baseline dependency.

  8. MEDIUM — Co-occurrence is not alias equivalence, and destructive normalization can erase the evidence needed to distinguish
     entities.
     Reference: §4 Pipeline 1; §2 generator examples.

     Claim: Bare token co-occurrence cannot distinguish aliases from different address components. The normalization plan also
     lacks explicit treatment of website hosts, alternate names and native-script preservation.

     Why: A city and state co-occur because they belong to the same address, not because they are interchangeable. Removing
     punctuation from starswelfaresociety.com leaves a trailing com, so space removal alone does not produce the intended match.

     “Strip accents” also needs a precise implementation. In a read-only Python check, removing all Unicode marks changed किरण to
     करण. Indic vowel signs and other marks carry textual information. Unicode character and combining-mark guidance
     (https://unicode.org/faq/char_combmark.html)

     Change: Preserve separate raw Unicode, conservative normalized, Latin-accent-folded, domain-name, acronym and alternate-name
     views. Parse website forms explicitly. Retain both sides of DBA/“formerly” expressions.

     Exploit the generator by mining aligned differences across gold pairs: repeated affix insertion, domain wrapping, number
     formatting and supported span substitutions. Require support across distinct training identities and check collisions against
     hard negatives. Prioritize operators responsible for held-out errors; do not equate normalization coverage with accuracy.

     Expected impact: Unmeasured. Recovering all matches for an additional 1% of previously unretrievable entities can contribute
     up to 0.01. Confidence: high in the specific normalization risks; medium in attainable recovery.

     Cost: Approximately 3–6 hours. Mapping statistics and focused checks can stream on the Mac; additional full-pool indexes
     consume substantial memory.

  9. MEDIUM — One fold standard deviation is the wrong threshold for accepting changes.
     Reference: §4 Validation and Submission plan.

     Claim: Variation in absolute fold scores is not uncertainty in the difference between two models evaluated on the same
     entities.

     Why: Fold difficulty affects both models together. A small gain can be consistent across every fold despite large absolute
     fold variation. Conversely, a mean gain exceeding that variation does not establish robustness to France or repeated model
     selection.

     Change: Save per-entity scores for parent and candidate and examine paired differences. Estimate uncertainty by resampling
     identities, or assignment-connected groups when conflicts couple entities. Report country/cardinality regressions separately.
     Keep one holdout untouched by iterative selection, and use the paired interval alongside effect size and runtime—not as
     another automatic veto.

     Expected impact: No defensible numeric forecast; the current rule can reject real gains below unrelated fold variation.
     Confidence: high.

     Cost: Approximately 1–2 hours. A few numeric arrays over two million entities fit comfortably on the Mac.

  10. MEDIUM — The submission schedule does not specify which uncertainty each upload resolves.
     Reference: §4 Submission plan; §1 ranking rules.

     Claim: Near-equal CV scores can represent meaningfully different France behavior or complementary errors. A late ensemble and
     fallback also depend on unverified final-submission selection semantics.

     Why: One baseline upload cannot establish a CV-to-private-score relationship when France is absent from CV and public/private
     composition is unspecified. Repeatedly selecting on public scores can favor public-specific noise. An ensemble can also
     change calibration, candidates and assignment behavior.

     Change: Give each upload a concrete purpose: independent retrieval coverage, transfer behavior, entity-aware decisions or
     demonstrated error complementarity. Permit near-tied candidates when they test different hypotheses. Validate the ensemble as
     a complete pipeline before its submission slot.

     Check whether final ranking uses a selected submission, the latest submission or another rule, and which timestamp breaks
     ties. Submit the strongest validated version early enough to recover from packaging or inference failures; do not assume the
     fifteenth slot should contain a new model.

     Expected impact: Unknown; avoiding a 0.02 France-only regression preserves 0.003 overall. Confidence: medium.

     Cost: Under an hour for the submission decision log and portal-rule check; no substantial Mac compute.

  Rule status: No definite external-data or licence violation is established by the written plan. Downloading geographic
  dictionaries, registry data or externally sourced alias tables would breach the supplied data restriction. The provenance of the
  handwritten French layer is unspecified; general language rules alone are not evidence of a breach.

  LightGBM’s code is MIT-licensed, but any added pretrained checkpoint requires its own licence and parameter-count verification;
  installing sentence-transformers establishes neither. LightGBM licence
  (https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE) Actual artifacts were not supplied, so model size, TSV validity and
  package reproducibility remain unverified. The concrete output-contract risk is finding 7.