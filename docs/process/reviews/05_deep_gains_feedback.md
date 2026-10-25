VERDICT: Prioritize richer matching evidence and broader training coverage. Another decoder rewrite has weak support from your
  measurements. A 0.98 private score would require substantial France improvement that current CV cannot establish.

  References below identify your supplied sections; I have not inspected the implementation and cannot provide verified file:line
  findings. Projected gains are absolute full-test F0.5, assuming known-country improvements transfer and France remains unchanged
  unless specified. They are hypotheses, overlap, and must not be added together. Engineering and compute estimates are
  assumptions.

  1. HIGH — The pair-loss breakdown can send you toward the wrong experiments.
     Reference: §3, diagnostics.

     The 4.2% selection loss is not 0.042 recoverable macro score. Your entire gap between the retrieval oracle and achieved
     training score is:

     [
     0.98983-0.96050=\mathbf{0.02933}.
     ]

     For an entity with four true matches, recovering its fourth match improves F0.5 by 0.0625. Adding a false positive to four
     correct predictions loses 0.1667. Systematic underprediction is therefore not, by itself, evidence of a bad decision rule.

     Change: Spend the first 30 minutes computing these counterfactual scores on the same graph:

      Score    Predictions
     ━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      A        Every retrieved gold edge, no false positives
     ───────  ──────────────────────────────────────────────────────────
      B        Every gold edge surviving current ownership
     ───────  ──────────────────────────────────────────────────────────
      C        Only the true positives among actual emitted predictions
     ───────  ──────────────────────────────────────────────────────────
      D        Actual predictions

     Then retrieval loss is (1-A), ownership loss (A-B), selection loss (B-C), and false-positive damage (C-D). These telescope
     exactly. Report them by country, true cardinality, script mismatch, missing address and name frequency.

     Singletons alone account for 0.00515 training macro loss, derived from (0.056(1-0.908)). Using your earlier 5.4% one-match
     frequency, one-match entities contribute another 0.00475. Those two groups warrant at least as much attention as high-
     cardinality underprediction.

     Impact/confidence: High confidence in the accounting; no direct score gain. It determines which recoverable errors deserve
     engineering time.
     Cost: Approximately 30 minutes engineering and a few score-table scans.

  2. HIGH — New stacked models could appear to improve CV through indirect label dependence.
     Reference: §2, full OOF graph and proposed second stages.

     Ordinary pooled OOF predictions do not make the joint decision graph strictly held out. When evaluating fold A, a rival in
     fold B receives predictions from a model that may have trained on A’s labels. That establishes a dependency; it does not
     establish its direction or magnitude.

     The same problem becomes more direct when training calibration, coherence or cardinality models on one global OOF file.
     Cross-fitting only the second stage does not remove validation-label influence through its input models.

     Change: Audit one outer fold A before promoting a learned second stage:
      - Score A with a model excluding A.
      - Score rival fold B with a model excluding both A and B.
      - Fit every supervised mapping, calibrator and second stage without A.
      - Keep supervised training sample sizes comparable between the ordinary and audit models.

     This needs four additional base-model fits for one outer-fold audit. Static cosine ranks and frequencies computed on the full
     unlabeled graph do not require this treatment; probability-derived features do.

     Impact/confidence: Potential CV correction is unknown. High confidence in the dependency, low confidence that existing gains
     are materially inflated.
     Cost: Approximately 1–2 engineering hours; four fits plus graph scoring. Reuse this outer-fold framework for subsequent
     stacking experiments.

  3. HIGH — Experiment priority 1: measure the training-data learning curve before changing model families.
     Reference: §2, training on 20% of entities.

     You already have 98M cached pairs but expose each model to approximately 15.6M. The cheapest unresolved question is whether
     the remaining training entities improve coverage of rare corruption combinations and difficult name collisions.

     Change:
      - Compare 20% versus 40% of training-fold entities, using nested samples, identical features and identical parameters.
      - If 40% improves, test 100% of training-fold entities.
      - Only then compare the incumbent with one larger-capacity configuration: for example, 255 leaves and up to 1,200 rounds,
        with stopping/model selection performed inside the training portion.

      - Do not simultaneously change class weighting, sampling and calibration; you would lose attribution.

     Your measured 25-minute CV implies approximately 50 minutes at 40% and 125 minutes at 100% under linear scaling. Those are
     extrapolations, not runtime guarantees.

     Memory needs attention: 98M × 48 float32 features is 18.8 GB before copies. Train folds sequentially, release retrieval
     indexes, and avoid holding both the full matrix and another full floating-point fold copy. LightGBM documents that row-wise
     construction doubles Dataset memory; benchmark column-wise construction if memory approaches the limit. LightGBM parameters
     (https://lightgbm.readthedocs.io/en/stable/Parameters.html)

     Where points come from: Selection misses, ownership mistakes and false positives whose feature patterns are underrepresented
     in the current sample. Retrieval misses remain unchanged.

     Expected impact: 0 to +0.0015, medium confidence that this is worth testing, low confidence in magnitude. With 1.19M training
     positives already, a flat learning curve is plausible.
     Cost: 15–45 minutes engineering; approximately 1–3 compute hours for the useful comparisons. Start this while coherence
     features are being implemented.

  4. HIGH — Experiment priority 2: add bounded pool-to-pool coherence features.
     Reference: §2, exclusively direct pair features; §3, selection and ownership losses.

     This is the strongest new-information hypothesis. A corrupted pool name may compare poorly with S1 while agreeing strongly
     with another pool record that has convincing direct evidence for that S1.

     Change: Start with at most three deterministic anchors per S1, chosen using conservative name agreement plus meaningful,
     nonmissing address evidence. A high cosine or matching house number alone is insufficient.

     For each candidate, compare it with the other anchors, excluding its own pool ID. Add approximately 10–12 features:
      - Best and second-best conservative-name similarity.
      - Best address similarity and number agreement.
      - Support from a compatible house number and suffix.
      - Number of supporting distinct normalized records.
      - Support across S2/S3.
      - Strong address or postal contradictions.
      - For contested candidates, support from the strongest rival’s anchors.

     Preserve the direct S1 features. Repeated identical pool strings should not count as several independent confirmations.
     Coherence should influence the classifier; it should not automatically propagate an assignment.

     With three anchors, the upper bound is 294M candidate–anchor comparisons for 98M train edges. Stream these comparisons and
     retain aggregates. Do not materialize the expanded edge table.

     Deterministic anchors avoid adding a supervised stacking layer. If probability-selected anchors later prove necessary, use
     the outer-fold construction in finding 2.

     Where points come from: Low-scoring true candidates among the 320,470 selection misses; wrong owners among the 104,053
     ownership misses; false matches supported only by a common name.

     Expected impact: A working hypothesis of +0.0008 to +0.0034, with zero gain possible. Medium confidence in the mechanism.
     Cost: 2–4 engineering hours; approximately 30–90 minutes for feature building and CV. Twelve additional float32 columns cost
     4.7 GB on the current train graph.

     Promotion: Require your paired criterion, positive India improvement, and no concentrated deterioration among singletons or
     common-name entities.

  5. HIGH — Experiment priority 3: India top-80 is justified, but its upside is bounded and its downstream cost is substantial.
     Reference: §3, measured retrieval rescue.

     The measured India oracle improvement translates to:

     [
     \text{top-80: }0.47(0.003)=\mathbf{0.00141}
     ]
     [
     \text{top-160: }0.47(0.006)=\mathbf{0.00282}.
     ]

     These are approximate full-test oracle gains, before classification losses. Top-80 cannot reasonably be forecast as a
     several-point improvement.

     Change: Run top-80 for all India queries first, retaining the existing candidate union. Rebuild ranks, counts and features on
     the expanded graph, then retrain.

     Avoid gating retrieval only on weak existing matches: an entity can have three convincing matches and still need a fourth.
     Likewise, filtering queries to Indic S1 names misses the important Latin-S1/Indic-pool case.

     Forty extra retrieval slots add up to approximately 35.3M train pairs and 32.6M test pairs, before overlap removal. This is
     why candidate generation time alone understates the cost.

     Where points come from: The measured 616 recovered gold edges per 3,426 sampled India misses. Newly retrieved difficult
     negatives can also change existing decisions.

     Expected impact: +0.0003 to +0.0012 if a useful fraction of the oracle gain survives classification. Medium confidence;
     actual regression remains possible.
     Cost: 15–45 minutes engineering; budget 1–2 additional compute hours initially, then replace that estimate with measurements.

     Promotion: Positive full-pipeline macro gain, not merely recall. Try top-160 only after top-80 demonstrates that recovered
     candidates can be classified accurately.

  6. HIGH — Experiment priority 4: learn a restricted cross-script alias view from training clusters.
     Reference: §3, 56% of India retrieval misses have Indic pool names; character retrieval recovers none of those misses.

     More Latin character retrieval cannot bridge disjoint scripts. A learned cross-script view can affect both retrieval and the
     confidence of already retrieved pairs.

     Change: Build a conservative token/phrase mapping before attempting a general transliterator:
      - Within training-fold gold clusters, collect Latin and Indic variants of the same field, including pool-to-pool variants.
      - Keep name and address mappings separate.
      - Generate token-alignment candidates from those paired fields.
      - Initially retain mappings supported by at least five distinct S1 owners, with a dominant destination and a substantial
        margin over alternatives. These thresholds are proposed starting settings.

      - Reject mappings explained merely by common legal words or locality co-occurrence.
      - Preserve original text; add mapped text, mapping coverage and ambiguity indicators as separate views.

     Index mapped Indic pool records and query that index with all India S1 records. Add a bounded top-20 rescue set, then compute
     mapped-name/address similarities for the classifier.

     Measure separately:
      - Coverage on held-out entities.
      - Performance on tokens observed versus unseen during mapping induction.
      - Retrieval oracle gain.
      - Final macro gain after new negatives and ownership competition.

     Learn every dictionary without the outer validation fold. A dictionary learned from all gold pairs would directly contaminate
     blocking recall.

     If the dictionary succeeds but unseen words remain the main limitation, a small character transducer learned from reliable
     token alignments becomes reasonable. A phonetic or character view is more useful after conversion to comparable scripts.

     Where points come from: Indic retrieval misses and cross-script selection/ownership errors.

     Expected impact: +0.0005 to +0.003 as a hypothesis; low confidence until held-out coverage is measured.
     Cost: 4–8 engineering hours and approximately 2–4 compute hours including fold-specific resources and validation. Time-box
     the first held-out pilot to four hours; stop if it provides little reusable coverage.

  7. HIGH — Experiment priority 5: France needs a separate transfer check; no score forecast is identifiable from current CV.
     Reference: §1 country mix; §3 public score and inferred France performance.

     From the rounded submission-001 numbers, the inferred France score is approximately 0.883, conditional on public composition
     and US/India transfer. It is not a measurement.

     Submission 002’s known-country contribution is:

     [
     0.38(0.9713)+0.47(0.9444)=0.812962.
     ]

     Holding that inferred France score constant gives approximately 0.9454 overall. Even reaching today’s US/India retrieval
     oracles gives only approximately 0.972 overall under the same assumption. Reaching 0.98 with those oracles requires France
     near 0.9363.

     Change: Beyond the French normalization already scheduled:
      - Keep 12, 12 bis, 12 ter and 12 B distinguishable through explicit suffix features.
      - Treat BP/CS/TSA numbers as separate address components; retain CEDEX routing indicators when interpreting postal
        disagreement.

      - Apply particle removal and st/ste expansion only in contextual alternate views. Preserve conservative names and addresses.
      - Add explicit ligature alternatives such as œ → oe; do not assume ordinary accent folding covers them.
      - Evaluate the new name-frequency and rival-count features under both US→India and India→US transfer. Removing country does
        not make these features invariant to graph size or naming distributions.

      - Compare the full model with one ablation of raw frequency/count and script indicators, retaining direct matching evidence.
        Add frequency percentiles as alternatives rather than blindly replacing absolute counts.

     Choose any generalization-oriented model or blend using held-out-country results before examining its France leaderboard
     effect. Do not impose an arbitrary French cardinality target or probability offset.

     Expected impact: Not estimable from current evidence. A +0.01 France improvement contributes +0.0015 overall; the same
     deterioration costs that amount. Confidence in any positive forecast is low.
     Cost: 1–3 engineering hours; two transfer fits plus the chosen ablation, approximately 1–2 compute hours at current sample
     sizes.

  8. MEDIUM — Calibration deserves one bounded experiment; learned action utility is preferable to forcing cardinality.
     Reference: §2 decision audit; §3 calibration and underprediction.

     Your audit already limits the case for another formula rewrite. The remaining selection loss should first be treated as
     insufficient evidence, rather than assumed decoder failure.

     Change: Allow 60–90 minutes for global sigmoid/isotonic calibration, evaluated independently of calibration fitting. Inspect
     reliability specifically for ownership winners and candidates at the selection boundary.

     For isotonic calibration, preserve the original probability as a secondary ordering key: its flat regions introduce ties that
     could otherwise change ownership arbitrarily. Scikit-learn calibration documentation
     (https://scikit-learn.org/stable/modules/calibration.html)

     Country-specific calibration needs a global fallback for France. Cardinality conditioning must use predicted cardinality or
     score summaries, never true match count. Also, comparing India’s mean sum of probabilities with the overall 3.46 truth mean
     is inconclusive without India’s actual truth mean.

     If structured errors remain, train a small action-value model. For each S1 and candidate action (k), the supervised target
     is:

     [
     U_i(k)=
     \begin{cases}
     \mathbf{1}[T_i=0],&k=0\
     5TP_i(k)/(T_i+4k),&k>0.
     \end{cases}
     ]

     Use top scores, gaps, prefix sums, pre-ownership mass, coherence, missingness and rarity. Start with actions around the
     existing choice plus zero; preserve the baseline action. Group all actions from one S1 together during splitting.

     This directly learns expected decision utility. Rounding a predicted match count to (k) does not optimize F0.5 and cannot
     identify which candidates are false.

     Expected impact: Calibration 0 to +0.0008; learned utility 0 to +0.0017, low confidence beyond your existing audit.
     Cost: Calibration approximately one engineering hour and score scans. Learned utility approximately 3–5 engineering hours
     plus honest nested fitting. Defer the latter until coherence is evaluated.

  9. MEDIUM — Global clustering and Hungarian matching are poor immediate investments.
     Reference: §3 ownership losses and rejected-owner diagnostics.

     Standard Hungarian matching imposes the wrong one-to-one structure. With independent edge utilities and only pool-side
     exclusivity, choosing the best owner already solves the separable assignment problem. A joint optimizer helps only through
     entity-level utility or additional evidence.

     The 1.37 rejected owned records per S1 exaggerates actionable opportunity. You identified 31,119 gold runner-up cases, with
     median runner-up probability 0.28. If every case restored the last missing match of a distinct four-match entity, the gain
     would be only 0.00088 training macro. That is a sensitivity calculation, not an upper bound.

     Change: After adding coherence, revisit only small conflict components. Retain two plausible owners, propose transfers using
     receiver utility gain minus donor loss, include abstention, and bound iterations. Keep baseline behavior for large ambiguous
     components.

     Avoid unrestricted pool connected components: one mistaken bridge can merge several businesses and amplify false positives.

     Expected impact: 0 to +0.0007 beyond the already-tested release pass, low confidence.
     Cost: 1–2 engineering hours plus score-table scans. Lower priority than the five experiments above.

  10. LOW — A transformer or new model family has to earn entry through a small pilot.
     Reference: §4, model-family and GPU alternatives.

     Your current evidence does not justify encoding the entire pool or cross-encoding every candidate pair.

     Change: First try a second LightGBM configuration or feature subset and blend probabilities before ownership and selection.
     Do not union emitted match lists.

     For a neural pilot, restrict computation to a predetermined ambiguous subset and measure the incremental gain when its score
     is added to the existing classifier. A model such as multilingual E5-small provides embeddings, not calibrated entity-match
     probabilities; its model card specifies MIT licensing. Model card (https://huggingface.co/intfloat/multilingual-e5-small)

     At an assumed 500 pairs/second, scoring 2M ambiguous pairs takes approximately 67 minutes; scoring 98M takes approximately 54
     hours. Benchmark actual tokenization and inference throughput before allocating training time.

     Require an honest gain of at least 0.001 known-country macro and an acceptable complete submission runtime before expanding
     the pilot. A reranker also cannot recover pairs absent from blocking.

     Expected impact: No defensible positive forecast.
     Cost: A strict 4–6-hour pilot budget, only after the earlier experiments. Verify that pretrained checkpoints are permitted
     separately from their licence; permissive licensing alone does not establish compliance with an “only provided training data”
     restriction.

  For the next five experiments, use this execution order. The promotion floors are proposed time-management gates, additional to
  your paired-fold criterion.

   Order    Experiment                                            Promotion criterion
  ━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1        20% → 40% training coverage                           Known-country macro gain ≥0.0003; proceed to 100% only if the
                                                                  learning curve remains useful
  ───────  ────────────────────────────────────────────────────  ─────────────────────────────────────────────────────────────────
   2        Three-anchor coherence features                       Gain ≥0.0005; India positive; inspect singleton/common-name
                                                                  regressions
  ───────  ────────────────────────────────────────────────────  ─────────────────────────────────────────────────────────────────
   3        India word top-80                                     Full-pipeline gain ≥0.0004 known-country macro; oracle gain
                                                                  alone is insufficient
  ───────  ────────────────────────────────────────────────────  ─────────────────────────────────────────────────────────────────
   4        Fold-local cross-script mapping                       First demonstrate held-out oracle improvement; then require
                                                                  final gain ≥0.0005
  ───────  ────────────────────────────────────────────────────  ─────────────────────────────────────────────────────────────────
   5        Transfer ablation plus French normalization checks    Improve held-out-country behavior without material known-
                                                                  country regression; treat French accuracy as unmeasured

  Run fits and final builds on the 64 GB on-demand machine. Use Spot for restartable feature shards, mapping statistics and
  bounded retrieval jobs. Do not make completion of the next submission depend on Spot survival.

  For the next four submissions:

  1. Submit 002 promptly. Its measured US/India changes imply approximately +0.01639 full-test contribution if they transfer,
     before any France change. Do not delay it for speculative work.

  2. Submit the strongest validated classifier improvement from training coverage and coherence, keeping retrieval fixed for
     attribution.

  3. Submit the validated India retrieval improvement, with regenerated graph features and retraining. Include transliteration
     only if its held-out benefit survives the complete pipeline.

  4. Use one controlled France/generalization comparison, holding US/India predictions fixed where practical. This makes the
     public comparison interpretable, but it still does not establish private France performance.

  Do not spend all remaining time searching for a supposed “typical 0.98 solution.” The supported missing capabilities are cross-
  record corroboration, reusable cross-script transformations and a measured training-data learning curve. Preserve the last 8–10
  hours for combining accepted changes, complete inference, independent audit and reproduction. Every rescue stage must appear in
  the exported candidate set, and every gold-derived mapping must be included with its training provenance.