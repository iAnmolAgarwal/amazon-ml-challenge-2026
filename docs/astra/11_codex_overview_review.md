VERDICT: Port the missing name-specificity and separate text-similarity features first. This is a better next experiment than
  another blocking rebuild. A move beyond 0.98 is possible, but the overview does not establish that it is achievable.

  I inspected the overview, your current features, and the nearby parallel_package.zip. That zip lacks the overview’s
  feat_spec.py, so I cannot verify the exact 0.971 implementation. I treat 0.971 as your reported leaderboard result; the overview
  itself reports 0.98082 CV.

  All forecast gains and implementation times below are estimates, not reproduced measurements.

  1. HIGH — The most valuable missing component is the rarity of unmatched name words.

     The overview reports +0.00377 CV from its specificity features, with India improving +0.00490 and US +0.00301. /Users/
     anmolagarwal/Downloads/OVERVIEW.md:65

     Your pipeline already has the four exact-name frequency features. src/er_feat2.py:46 What it lacks is the distinction
     between:
      - Matching on common words versus rare words.
      - A candidate omitting an important S1 word.
      - A candidate adding an important distinguishing word.

     Your current fuzzy scores do not encode that distinction directly. src/er_feat.py:48

     For example, Rao Motors versus Rao Sunrise Motors can receive a perfect token-set score. Whether Sunrise is rare and
     distinguishing matters more than its character count.

     Implement this feature group on your existing candidates:

     Let (A) and (B) be the sets of normalized name tokens for S1 and the pool record. Compute country-specific token document
     frequencies from the unlabeled S1 records:

     [
     w(t)=1+\log\frac{N+1}{df(t)+1}.
     ]

     This is the standard smoothed IDF convention. scikit-learn documentation
     (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)

     Add:
      - Shared IDF mass: (\sum_{A\cap B}w(t)).
      - S1-only IDF mass: (\sum_{A\setminus B}w(t)).
      - Pool-only IDF mass: (\sum_{B\setminus A}w(t)).
      - S1-only mass divided by total S1 mass.
      - Pool-only mass divided by total pool mass.
      - Maximum IDF of a pool-only token.

     Use token sets, handle empty names explicitly, and cap unseen-token weights. Preserve your conservative-name features. Do not
     turn “extra rare word” into a hard rejection: the generator also inserts legitimate filler words.

     These features can be computed transductively on each split without labels, including France.

     Expected incremental LB gain on your 0.963 pipeline: +0.001–0.003, medium-low confidence.
     Cost: approximately 1–2 hours implementation, followed by a benchmarked feature pass and CV.
     Why not predict the full +0.00377? Their feature bundle includes exact-name frequencies that you already possess.

  2. HIGH — Add separate name/address similarities; your combined cosine conceals disagreements.

     The inspected archive contains distinct TF-IDF similarities for name words, name character trigrams, skeleton words, skeleton
     character n-grams and address words. See the archive (/Users/anmolagarwal/Downloads/parallel_package.zip), member code/
     business_entity_resolution/src/features.py:15.

     Your principal cosine combines name and address into one text field. src/er_block.py:27 A strong address match can therefore
     conceal a materially different name, and vice versa.

     First add only:
      - Name-word TF-IDF cosine.
      - Address-word TF-IDF cosine.

     Compute them only for existing candidate pairs. This needs sparse row-wise dot products, not another nearest-neighbour
     search.

     Then add two skeleton ambiguity features: the number of S1 records sharing each side’s nonempty skeleton. Your skeleton
     currently participates in blocking but is absent from the feature input columns. src/er_norm.py:214, src/er_feat.py:22

     Preserve conservative names and digits. The other implementation uses token-wise skeletons; yours skeletonizes the whole
     joined name and drops digits. Do not silently substitute one definition for the other.

     Expected incremental gain: +0.0005–0.002, low confidence; overlaps with item 1.
     Cost: approximately 1–2 hours implementation plus feature generation.

     Combined forecast for items 1–2: approximately +0.002–0.005 on your current model, with zero or negative gain still possible.
     This is the first experiment I would authorize.

  3. HIGH — Do not attribute the 0.971 result to a wholesale architectural advantage.

     Applying test country weights to the overview’s ablation table gives:

      Difference                                        Derived contribution to overall test score, assuming transfer
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      Their original model versus your current model                                                         +0.00035
     ────────────────────────────────────────────────  ───────────────────────────────────────────────────────────────
      Their training sample increase, 7% → 25%                                                               +0.00163
     ────────────────────────────────────────────────  ───────────────────────────────────────────────────────────────
      Their specificity features                                                                             +0.00345
     ────────────────────────────────────────────────  ───────────────────────────────────────────────────────────────
      Total known-country improvement                                                                        +0.00543

     Source: /Users/anmolagarwal/Downloads/OVERVIEW.md:64. This is arithmetic across reported results, not a controlled comparison
     between implementations.

     Your model already trains on 50% in CV and 100% for test. Their 7%→25% improvement is not another +0.0018 available to you.

     Their reported blocking oracle is 0.9943, versus your 0.99398—only 0.00032 apart overall. /Users/anmolagarwal/Downloads/
     OVERVIEW.md:80 That supports prioritizing discrimination.

     Different blockers may miss different matches, so candidate union remains useful. Measure your oracle, their oracle and the
     union oracle on identical entities before replacing blocking. Candidate count alone—21 versus 44—does not establish which
     system will score better.

  4. HIGH — For maximum score, transfer improvements in both directions and retain the verified stronger incumbent.

     Once the exact 0.971 artifact is identified and reproducible, use it as the benchmark every new candidate must beat.

     The inspected earlier archive suggests useful transfers from your pipeline into theirs:
      - Conservative-name ratio, token-set and Jaro–Winkler.
      - Explicit house-number, street and postal agreement/conflict.
      - Missingness-aware numerical evidence.

     Your implementation already exposes these. src/er_feat2.py:72, src/er_feat.py:65

     The archive instead uses the first numerical address token as its main number-equality feature: member src/features.py:109.
     Its normalizer retains numbers in encountered order: src/normalize.py:174. After component reordering, that first number may
     be a postcode or unit number.

     Check whether the actual 0.971 branch still has these omissions. If so, adding your stronger raw-name and parsed-address
     evidence to that branch may offer a faster route upward than rebuilding all its advantages inside your 0.963 branch.

     Forecast: 0–0.003 above 0.971, low confidence. These additions require an ablation; they are not independently measured.

     Afterward, test a small probability-blend grid between the improved branches. Require OOF predictions with verified training
     exclusions, key pairs by actual IDs, and rerun ownership and set selection. Rescore missing edges or represent missingness
     explicitly; absence from one blocker is not a calibrated probability of zero.

     Blend forecast: 0–0.002 above the stronger model, depending entirely on error complementarity.

  5. HIGH — The overview’s validation claims do not justify adding its gains mechanically.

     It explicitly admits that the transliteration dictionary uses all training labels. /Users/anmolagarwal/Downloads/
     OVERVIEW.md:112

     It also asserts that a shared stage-2 competition effect cannot affect measured gains. /Users/anmolagarwal/Downloads/
     OVERVIEW.md:116 That is unsupported: new features can change how strongly stage 2 exploits contaminated rival probabilities.

     Minimum validation for the transfer:
      - Hold candidates, normalization, folds and decision floor fixed for the first comparison.
      - Apply your paired-fold promotion rule.
      - Check the improvement on an outer-held-out set excluded from every supervised mapping and model.
      - Report results for common names, extra-word candidates, empty addresses, singletons and one-match entities.
      - Compare test additions using the same frozen house parser as the parent.
      - Add a new diagnostic: newly emitted pairs containing rare, unmatched name tokens, broken down by address support.

     The last diagnostic matters because a house-conflict check cannot detect all same-address or missing-address decoys.

     Do not immediately copy their proposed coherence experiment. It is explicitly untested in the overview, and your own
     coherence attempt already failed the test-shift check. /Users/anmolagarwal/Downloads/OVERVIEW.md:137

  6. MEDIUM — The fastest next experiment needs no normalization or blocking rebuild.

     I would give the coding agents this sequence:

      Order    Experiment                                                       Decision
     ━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      1        Add directional token rarity, skeleton counts and separate       Keep stage-1 predictions fixed initially
               word cosines on the current candidate cache
     ───────  ───────────────────────────────────────────────────────────────  ───────────────────────────────────────────────────
      2        Train stage 2 with those added features                          Fastest test of whether the new information helps
     ───────  ───────────────────────────────────────────────────────────────  ───────────────────────────────────────────────────
      3        If promising, add the features to stage 1, regenerate OOF        Produces the complete candidate
               predictions, rebuild competition features and retrain stage 2
     ───────  ───────────────────────────────────────────────────────────────  ───────────────────────────────────────────────────
      4        Compare against the exact 0.971 branch; test complementary       Promote only against the stronger incumbent
               feature transfer or blending

     Keep the 0.6 floor fixed during feature comparisons. Subsequently test a small predeclared floor grid on OOF predictions; the
     other model’s 0.5 floor is not automatically appropriate for yours.

     Estimated first screening cycle: 3–5 hours including implementation, feature construction and one stage-2 CV.
     Estimated complete two-stage candidate: another 2–3 hours, assuming existing reported training speeds.

     Memory must be fixed before adding columns. At your measured 98,021,508 pairs, eight float32 features add 3.14 GB before
     copies. Your previous 61.7 GB peak leaves no room.

     Stream feature creation and prediction, and avoid keeping the complete joined dataframe and dense matrix simultaneously. The
     archive’s streamed row-group loading is worth adapting independently of its model design: member src/model.py:47.

  7. MEDIUM — Above 0.98 requires substantial additional gains after reaching 0.971.

     Under the same transfer assumptions:

     [
     F_{\text{France,new}}
     =\frac{0.971-0.38(0.98269)-0.47(0.97801)}{0.15}
     \approx0.9194.
     ]

     Thus the observed +0.008 leaderboard difference decomposes conditionally into:
      - +0.00543 from known-country CV improvements.
      - +0.00257 from everything else, equivalent to France improving about 0.0171.

     That residual is not proof of a French improvement; public composition and US/India transfer remain unknown.

     With their US/India scores unchanged, 0.98 requires France ≈0.9794. That is not a reasonable forecast from minor French
     normalization changes.

     A more balanced target is:

     [
     0.38(0.990)+0.47(0.986)+0.15(0.940)
     =\mathbf{0.98062}.
     ]

     This requires meaningful improvements in all three countries. The plausible route is better name discrimination,
     complementary candidate/model evidence, and demonstrated French parser repairs together.

     My planning estimate is 0.965–0.969 for the feature port into your 0.963 branch, and 0.973–0.977 for a successful adapted/
     combined approach starting from the verified 0.971 incumbent. Both are guesses. Treat >0.98 as a stretch outcome requiring
     new measured evidence, not the expected result of copying the overview.
