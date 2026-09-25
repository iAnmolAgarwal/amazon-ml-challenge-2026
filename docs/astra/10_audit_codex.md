VERDICT: 0.99 next submission is not a credible target. Aim for roughly 0.965–0.970. The best next iteration is v3 retrieval + a
  few skeleton features + verified French normalization repairs, while correcting the stage-1 training-size mismatch.

  I inspected the current code. Gains below are forecasts unless explicitly marked measured or derived.

  1. HIGH — Your attainable next-submission gain is much smaller than +0.027.

     Using the measured country scores and blocking oracles in attempts/006_stage2/results.json:25:

     [
     F_{\text{France}}\approx
     \frac{0.963-0.38(0.9757)-0.47(0.9721)}{0.15}
     =0.9023.
     ]

     This remains conditional on US/India transferring unchanged and the public subset having the full-test country proportions.

     The following are derived, conditional bounds, not measured test ceilings:

      Scenario                                                     Overall score
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━
      Current US/India; France becomes perfect                           0.97765
     ───────────────────────────────────────────────────────────  ───────────────
      US/India reach current blocking oracles; France unchanged          0.97975
     ───────────────────────────────────────────────────────────  ───────────────
      US/India reach current blocking oracles; France perfect            0.99440

     Reaching 0.99 would recover 86% of the remaining loss below that last bound. Even with perfect classification of all
     retrieved US/India matches, France must reach 0.9707.

     My planning scenarios—not statistical confidence intervals—are:

      Scenario                            US    India    France    Weighted score
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━  ━━━━━━━  ━━━━━━━━  ━━━━━━━━━━━━━━━━
      Next iteration, modest success    .977     .973      .910             .9651
     ────────────────────────────────  ──────  ───────  ────────  ────────────────
      Next iteration, strong success    .980     .978      .925             .9708
     ────────────────────────────────  ──────  ───────  ────────  ────────────────
      Final, moderate outcome           .980     .979      .925             .9713
     ────────────────────────────────  ──────  ───────  ────────  ────────────────
      Final, strong outcome             .988     .986      .950             .9814

     Forecast: next submission 0.965–0.970, low-to-medium confidence; final 0.971–0.981, low confidence. Regression remains
     possible. A 0.99 outcome requires a substantial, demonstrated French breakthrough and near-elimination of known-country
     classification errors—not the currently estimated increments.

  2. HIGH — The current stage-2 CV contains an indirect validation-label path.

     Pool competition features combine probabilities from different fold models: src/er_feat4.py:28. For an entity in fold F, a
     rival’s prediction can come from a model trained on F’s labels: src/er_eval.py:49. The globally generated sidecar is then
     loaded before stage-2 splitting: src/er_eval.py:29.

     Consequently, the measured +0.00340 is not a fully independent estimate. This does not invalidate the actual leaderboard
     improvement, but it weakens forecasts based on further stacking gains.

     Minimal next-iteration check: reserve one outer fold whose labels are excluded from dictionary learning, stage 1 and stage 2.
     Generate inner OOF predictions within the remaining entities, train stage 2 there, and evaluate parent and proposed bundle
     identically on the outer fold after global assignment.

     Budget 2–3 hours, assumed, using two inner folds. This is a cheap falsification test: scoring the complete graph still
     includes in-sample rival entities, so it does not perfectly reproduce test competition. A material reversal kills the change;
     a positive result supplements ordinary CV rather than proving transfer.

     Also, --folds 0 currently does not save fitting time: src/er_eval.py:47 still trains every fold. The option only restricts
     reporting.

  3. HIGH — Correct the stage-1 probability shift before adding another model layer.

     Stage-1 OOF models currently train on:

     [
     50%\times80%=40%\text{ of all training entities}.
     ]

     Test stage-1 models train on 100%: attempts/007_stage2/train.py:7. Stage 2 therefore learns probability sums, margins and
     threshold counts from a base model trained on 2.5× less data.

     Change: generate the next stage-1 OOF predictions using 100% of the other folds, rebuild competition features, then train
     stage 2. Keep stage-2 capacity and the 0.6 floor unchanged initially. Increasing only stage-2 training fraction would leave
     the main feature-distribution mismatch intact.

     Expected LB effect: −0.001 to +0.002, guessed. This is primarily a transfer-risk correction, not a promised gain. Measure it
     separately on the cached parent candidates before combining it with v3.

  4. HIGH — V3 needs a memory gate and a correctly constructed retrieval benchmark.

     Reverse top-3 can introduce up to:

     [
     3\times10.32\text{M}=30.96\text{M}
     ]

     additional train edges before deduplication; new exact keys can add more. A purely linear extrapolation of the existing peak
     gives:

     [
     61.7\times(98+31)/98\approx81.2\text{ GB}.
     ]

     That is an illustrative risk calculation, not a RAM forecast. Your current evaluator materializes both joined features and a
     dense matrix: src/er_eval.py:28, line 40 (src/er_eval.py:40).

     Before the full run: release joined frames promptly, stream prediction features in smaller batches, and avoid retaining full
     dense feature matrices alongside their dataframe copies. Try force_col_wise=true; LightGBM documents its memory advantage
     over row-wise construction. Require a measured projected peak below approximately 54 GB—an operational headroom target, not a
     model requirement. LightGBM documentation (https://lightgbm.readthedocs.io/en/stable/Parameters.html)

     There is also a benchmark trap: --sample samples S1 before reverse retrieval at src/er_block.py:144. Reverse top-3 then
     searches an artificially small S1 index, inflating recall.

     Fix: sample the entities being evaluated or the pool queries, while retaining the full country S1 index for reverse
     retrieval.

  5. MEDIUM — Run this specific bundle; do not combine all ten proposals.

     The ranking below uses expected marginal gain per implementation/compute hour. Shared rebuild costs are listed separately
     afterward.

      Priority                    Component                Expected overall LB gain       Incremental cost    Promotion gate
     ━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━
      1                           d: demonstrated        0 to +.004, low confidence       1–2 h diagnosis/    Frequent affected
                                  French parser/view                                        implementation    records, preserved
                                  repairs                                                                     distinguishing
                                                                                                              information, no
                                                                                                              unexplained
                                                                                                              collision growth
     ──────────────────────────  ─────────────────────  ────────────────────────────  ─────────────────────  ─────────────────────
      2                           a: v3 retrieval           +.0005–.002, medium-low         Already built;    Weighted oracle
                                                                                                additional    gain ≥.001;
                                                                                         retrieval runtime    realized paired
                                                                                                unmeasured    gain; memory passes
     ──────────────────────────  ─────────────────────  ────────────────────────────  ─────────────────────  ─────────────────────
      3                           b: skeleton ratio/                0 to +.001, low                0.5–1 h    Incremental
                                  JW plus collision                                                           improvement on v3
                                  information                                                                 candidates; no
                                                                                                              decoy-stress
                                                                                                              reversal
     ──────────────────────────  ─────────────────────  ────────────────────────────  ─────────────────────  ─────────────────────
      Required control            100% stage-1 OOF              −.001 to +.002, low     Approximately 0.5–    Better transfer
                                  training                                                     1.5 h extra    diagnostics and no
                                                                                                              material validation
                                                                                                              regression
     ──────────────────────────  ─────────────────────  ────────────────────────────  ─────────────────────  ─────────────────────
      Next separate experiment    j: independent-           +.0005–.002, low-medium    1–3 h if usable OOF    Complementary
                                  pipeline blend                                                is readily    errors and an
                                                                                                 available    honest OOF
                                                                                                              improvement

     These estimates overlap. Budget +0.002–0.006 for the combined next iteration, rather than summing optimistic endpoints.

     For skeleton features, add only ratio, Jaro–Winkler and a short/common-skeleton indicator initially. Keep conservative names.
     Skip another char-ngram index for this submission. Skeleton equality without collision information can strengthen the wrong
     same-name business.

     Compute a candidate-provenance mask for every edge: original, reverse-only, skeleton-only, other-new. This permits source-
     specific diagnostics and cached ablations. Recompute graph features whenever an ablation changes the candidate set.

     Keep separate:
      - c: reject unconditional substitution at 91% measured word precision. Validate whole held-out vocabulary families,
        frequency-weighted accuracy and abstention coverage. Accepted guesses should first be an additional retrieval view, not
        destructive replacement.

      - e: implement the decoy stress test, but do not reintroduce coherence in this submission.
      - f: no stage 3 or second model family yet; fix stage-1 OOF size first.
      - g: diagnose the floor’s losses before building cardinality machinery.
      - h: no cross-encoder on this iteration; its setup/inference cost exceeds the better-supported CPU experiments.
      - i: no test-only logit or floor adjustment justified by row counts.

     The independent pipeline’s measured country advantage is only:

     [
     .38(.9784-.9757)+.47(.9710-.9721)=.000509
     ]

     before France. Its larger value would have to come from complementary errors. Obtain keyed OOF probabilities, compare error
     overlap, and blend on the candidate union. Missing scores are missing, not probability zero. Do not blend emitted lists.

     Single-machine execution order:

      Order    Work                                                                                                        Budget
     ━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      1        Freeze inputs; memory gate; French incidence/collision                                              1–2 h, assumed
               scan; small feature implementation
     ───────  ─────────────────────────────────────────────────────────  ─────────────────────────────────────────────────────────
      2        One train/test normalization–candidate–feature build         2.4 h measured baseline; 3–4 h allowance with reverse
               for the frozen bundle                                                                                    retrieval
     ───────  ─────────────────────────────────────────────────────────  ─────────────────────────────────────────────────────────
      3        Stage-1 OOF at full fold fraction → competition                          2–3 h assumed; these are two model stages
               features → stage-2 CV
     ───────  ─────────────────────────────────────────────────────────  ─────────────────────────────────────────────────────────
      4        Paired blinded-outer check and decoy stress diagnostics                                             2–3 h, assumed
     ───────  ─────────────────────────────────────────────────────────  ─────────────────────────────────────────────────────────
      5        Both test scoring stages, changed-pair inspection,                                                1.5–2 h, assumed
               validation and packaging

     Total: approximately 10–14 hours, less any completed work. Run heavy jobs sequentially.

     If tonight’s v3 job has already started, do not mutate its normalization inputs. Current normalization paths are shared
     across candidate tags: src/er_block.py:138. Either reuse its frozen artifacts or treat it as the a-only experiment before
     creating a separate complete cache snapshot.

  6. MEDIUM — France has concrete representation defects, but their incidence must justify the work.

     The following are visible in the implementation:

      Defect                                                         Exact change
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      & becomes and, which is removed; French et remains. src/       Add a French conjunction-equivalent view for &/and/et;
      er_norm.py:137                                                 preserve conservative text.
     ─────────────────────────────────────────────────────────────  ──────────────────────────────────────────────────────────────
      Sté Dupont and Société Dupont remain different. src/           Prefix-context aliases: sté/société, ets/établissements,
      er_norm.py:138                                                 cie/compagnie. Do not globally interpret ste as sainte in
                                                                     business names.
     ─────────────────────────────────────────────────────────────  ──────────────────────────────────────────────────────────────
      12 bis and 12 ter expose the same house-number feature.        Preserve base number and suffix separately; distinguish
      src/er_norm.py:178                                             agreement, conflict and absence.
     ─────────────────────────────────────────────────────────────  ──────────────────────────────────────────────────────────────
      BP/CS/TSA numbers are deleted before address features are      Extract postal-reference type/number before removal from the
      computed. src/er_norm.py:167                                   house parser. Keep them as address evidence.

     Accent folding, ligatures and many French legal forms already exist. Reimplementing them offers no gain.

     Further rules worth testing only where observed:
      - Recognize spaced/dotted legal forms such as S.A.R.L. in an alternate view.
      - Normalize apostrophe variants; retain l’/d’ and de/du/des/le/la/les in conservative names.
      - Preserve five-digit postal codes as strings, including leading zeros.
      - Preserve arrondissement and CEDEX delivery information separately from house numbers.
      - Support observed street-before-number formats with contextual parsing; never take an arbitrary later number as the house.
      - Preserve distinctions such as 12, 12 bis, 12 ter while allowing missing suffixes to remain uncertain.

     Two-hour checkpoint: count affected French records and S1 candidate groups, inspect transformation examples, and measure
     newly created canonical collisions. If a rule changes predictions for only 0.2% of French entities, even perfect correction
     there contributes at most 0.0003 overall.

     New France-only feature columns that are constant throughout training will not teach LightGBM anything. Express repairs
     through existing comparable features, or train generic features on genuine US/India analogues.

     LOCO validation: train US→India and India→US with the held-out country excluded from all supervised mappings and both model
     stages. Use this to reject brittle generic decisions. It cannot validate French-specific rules or determine a French
     probability threshold. Two stage-1-only fits are a cheaper diagnostic, but their calibration must not be transferred directly
     to stage 2.

  7. MEDIUM — The decoy and house-conflict checks need stricter interpretation.

     Pool/S1 increases by 23.1%, not 100%. Assuming unchanged true cardinality:

     [
     \text{unmatched pool/S1}: 4.68-3.46=1.22
     \quad\rightarrow\quad
     5.76-3.46=2.30.
     ]

     That is 1.89×, but it does not establish twice as many near-duplicate decoys.

     Removing 18.75% of S1 matches the aggregate pool/S1 ratio. Use it as a stress test, not as an estimated test prior. Recompute
     name frequencies, competition features, reverse retrieval and key-cap eligibility; filtering the old graph alone is
     incomplete.

     For promotion:
      - Require your paired five-fold rule, with at least +0.001 overall bundle gain as a practical next-submission target.
      - Require no material reversal in the blinded outer check.
      - Require nonnegative aggregate gain in the decoy stress test.
      - Inspect test additions/removals separately by country and candidate source.

     Compute house conflicts using a frozen diagnostic parser for both submissions. Otherwise a parser change can manufacture an
     apparent improvement. Report parsing coverage, missingness, postal conflicts and conflicts among newly emitted edges, not
     just the overall average.

     As a proposed review trigger, investigate a US increase above 0.5 percentage points, or a country-relative increase above
     20%. These are operational thresholds, not accuracy estimates. A falling conflict rate can also mean lost recall.

  8. LOW — The cheapest overlooked diagnostic is the one-match failure slice, not another cardinality model.

     Your one-match entities have substantial recoverable classification loss: attempts/006_stage2/results.json:47.

     Using the supplied population shares:

     [
     0.054(0.981-0.9151)=\mathbf{0.00356}
     ]

     is the train-distribution headroom from bringing that slice to its current retrieval oracle. Perfecting singletons offers
     only:

     [
     0.056(1-0.9733)=\mathbf{0.00150}.
     ]

     Split one-match failures into retrieval misses, wrong owners, empty decisions and post-selection floor rejection. The floor
     is applied after choosing k at src/er_model.py:80; a cardinality model cannot recover edges subsequently discarded by that
     floor.

     Also cross-tab these failures against common names, rare address-token agreement and candidate provenance. One or two generic
     address-rarity features may distinguish a true business from its same-name decoys more directly than another layer of
     probability aggregation. Treat that as a subsequent 0–0.0015 guessed opportunity, unless the diagnostic reveals a frequent,
     specific failure suitable for this bundle.
