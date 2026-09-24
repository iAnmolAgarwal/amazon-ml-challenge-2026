VERDICT: REVISE — The decoder is mathematically mis-specified, and ordinary pooled OOF predictions do not establish leakage-free
  evaluation of the coupled assignment rule.

  References identify your supplied sections; implementation-specific risks remain unverified without code. Impact numbers below
  are sensitivities, not measured improvements. Costs are estimates for the stated machine.

  1. HIGH — The decoder estimates neither expected F0.5 nor singleton probability correctly.
     Reference: §3 Decision rule.

     Claim: There are three distinct errors: a ratio of expectations replaces expected utility; retained probability mass replaces
     total truth count; and singleton probability assumes independent candidate labels.

     Why: With one possible match at probability 0.48, your formula assigns predicting it utility
     [
     \frac{5(0.48)}{0.48+4}=0.5357,
     ]
     exceeding the empty utility of 0.52. But the actual expected utilities are 0.48 for predicting and 0.52 for empty. Brute-
     forcing the implemented formula would not detect this error.

     Also, sum(kept probabilities) estimates true matches among retained candidates, excluding matches lost through both blocking
     and ownership. Dividing by aggregate retrieval recall cannot correct entity-specific missing mass.

     Finally, consider three candidates that are either all correct or all incorrect, with probability 0.4 of being correct. Their
     marginal probabilities are perfectly calibrated. Your singleton estimate is (0.6^3=0.216), whereas the true singleton
     probability is 0.6. Multiple records of the same competing business make such dependence plausible.

     Change: First compare against an empirically tuned pair threshold and an explicit entity-level singleton classifier. Train
     the latter against zero total gold matches, using candidate aggregates before ownership.

     Calibrate pair scores using held-out predictions, checking ownership winners separately. No class weighting does not
     guarantee calibration. Calibration fitting must itself exclude evaluation entities. Calibration guidance
     (https://scikit-learn.org/stable/modules/calibration.html)

     The correct nonempty utility is
     [
     U(S)=5\sum_{i\in S}\sum_{t\ge1}
     \frac{P(Y_i=1,T=t)}{t+4|S|};
     \qquad U(\varnothing)=P(T=0).
     ]
     Pair marginals alone do not determine these quantities. An independent-Bernoulli calculation is useful as a diagnostic, but
     retains the dependence and missing-truth assumptions. A practical learned alternative predicts each candidate action’s
     realized (F_k) from honest OOF score/count/conflict aggregates.

     Expected impact: Potentially material; magnitude unknown. The correlated example loses 0.2 per affected entity—0.002 overall
     if it affects 1% of entities. Confidence: high in the mathematical errors.

     Cost: Approximately 1–3 hours for calibration, threshold and singleton comparisons using cached scores; 3–5 hours for an
     honestly evaluated learned utility model. No retrieval rerun.

  2. HIGH — India’s blocking ceiling is the largest quantified recoverable loss.
     Reference: §3 Blocking.

     Claim: Word-only retrieval and exact keys lack an independent recovery mechanism for corrupted words. The hard key cap also
     discards some cheap joins.

     Why: India’s measured oracle deficit contributes
     [
     0.47(1-0.9820)=\mathbf{0.00846}
     ]
     to potential full-test score loss, if the measured country behavior transfers. US contributes another 0.001824. The
     classifier cannot recover these missing edges.

     The cap rejects a key with one S1 and 51 pool records, while admitting a key with 50 S1 and 50 pool records. Those produce 51
     versus 2,500 pairs. Per-side counts are a poor proxy for join cost.

     Change: On the existing audit sample, compare:
      - Word top-40 versus top-80/top-160.
      - Adding a name-character retrieval view.
      - Replacing the per-side cap with a bounded join-product budget, followed by similarity filtering where necessary.

     Select by incremental macro oracle gain per added candidate and runtime. Do not automatically widen the full graph.

     Verify the acronym join specifically: computing initials on both sides maps Stars Welfare Society to SWS, but SWS to S. The
     intended retrieval needs S1 initials ↔ pool compact name, and the reverse direction when applicable, combined with address
     evidence.

     Expected impact: Raising India oracle by 0.005 creates room for 0.00235 global gain, before classification errors.
     Confidence: high in the ceiling; medium in the proposed recovery mechanisms.

     Cost: Approximately 2–4 engineering hours, plus a sampled retrieval benchmark. Full-run cost must be measured for the chosen
     additional view; existing word throughput does not establish character-view throughput.

  3. HIGH — France needs guarded alternate views and transfer checks; generic normalization alone is not evidence of robustness.
     Reference: §3 Normalization, Features and France.

     Claim: Several existing transformations are ambiguous in French, especially st→street, legal-word deletion and numeric
     extraction. Removing country from the feature list does not remove country-dependent feature distributions.

     Why: France accounts for 15% of test. A 0.05 France-only regression costs 0.0075 globally. There are no French labels to
     validate aggressive normalization or a special threshold.

     Change: Implement the following as additional comparison views, preserving conservative text and distinguishing tokens. These
     are general format rules, not geographic mappings.

      Area                       Concrete rule                                     Guard
     ━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      Legal forms                Recognize dotted/spaced variants of SARL,         Remove only in a legal-form view. SA/SAS can
                                 EURL, SAS, SASU, SA, SNC, SCI, SCP, SELARL,       also be name tokens
                                 SELAS; optionally recognize their spelled-out
                                 forms where known
     ─────────────────────────  ────────────────────────────────────────────────  ────────────────────────────────────────────────
      Business descriptors       Alternate expansions ets→etablissements, ste/     Preserve the original view; do not blanket-
                                 sté→societe, cie→compagnie                        delete these words
     ─────────────────────────  ────────────────────────────────────────────────  ────────────────────────────────────────────────
      Street types               In address context: av/ave→avenue, bd/            Short ambiguous forms such as r and ch require
                                 boul→boulevard, rte→route, chem→chemin,           street-component context
                                 imp→impasse, pl→place, all→allee, sq→square,
                                 fg/fbg→faubourg
     ─────────────────────────  ────────────────────────────────────────────────  ────────────────────────────────────────────────
      Saint/Sainte               Preserve st/ste; optionally compare with          Do not globally expand st to street
                                 saint/sainte in a separate view
     ─────────────────────────  ────────────────────────────────────────────────  ────────────────────────────────────────────────
      House suffixes             Parse 12bis, 12 bis, 12-bis into base 12,         Keep suffix agreement/conflict separate. 12,
                                 suffix bis; likewise ter, quater and letter       12 bis and 12 B are not automatically
                                 suffixes                                          equivalent
     ─────────────────────────  ────────────────────────────────────────────────  ────────────────────────────────────────────────
      Postal routing             Recognize CEDEX and its following routing         Keep a routing flag and original text. Do not
                                 suffix; mark BP, CS, TSA numbers as postal-       treat these numbers as house numbers or
                                 service references                                automatically as postcodes
     ─────────────────────────  ────────────────────────────────────────────────  ────────────────────────────────────────────────
      Postal/district formats    Preserve five-digit postcodes as strings,         Do not infer arrondissement from arbitrary
                                 including leading zeros. Normalize 15e, 15ème,    postcode endings or convert Roman-looking
                                 15eme; parse Roman numerals only in explicit      business names
                                 arrondissement context
     ─────────────────────────  ────────────────────────────────────────────────  ────────────────────────────────────────────────
      Elisions/particles         Normalize straight/curly apostrophes; retain      Never use particle deletion as the sole
                                 both l etoile and letoile, similarly d …. Add     blocking representation
                                 a particle-light comparison view for de/du/
                                 des/le/la/les
     ─────────────────────────  ────────────────────────────────────────────────  ────────────────────────────────────────────────
      Accents/ligatures          Keep accent-folded and original views;            NFKC plus accent folding does not supply every
                                 explicitly add œ↔oe, æ↔ae alternatives            such equivalence
     ─────────────────────────  ────────────────────────────────────────────────  ────────────────────────────────────────────────
      Street-name numbers        Preserve numeric street-name components, such     Do not identify its first number as the house
                                 as rue du 8 mai 1945                              number merely because no house was found

     Add presence-conditioned comparison features, house-suffix features, field-specific token coverage and rarity-weighted
     address agreement. Compute both-side statistics on the full unlabeled graph before supervised sampling.

     Run US→India and India→US transfer evaluations, including a reverse-rank/degree ablation. They can expose brittle feature
     behavior; neither provides a French accuracy estimate. Do not introduce a France threshold offset without evidence.

     Expected impact: Unknown; 0.02 French-only improvement contributes 0.003 globally. Confidence: high in the ambiguity risks,
     low in any forecast French gain.

     Cost: Approximately 3–5 engineering hours, normalization checks and two bounded transfer fits. Retrieval-affecting changes
     require a fresh candidate audit.

  4. HIGH — The pooled OOF graph still contains a held-out-label path through rivals.
     Reference: §3 CV and both-side rank features.

     Claim: An entity’s own OOF score does not make the final competition outcome OOF.

     Why: Let evaluated entity A belong to fold 1, and its rival B to fold 2. B’s score comes from a model that excludes fold 2
     but may have trained on A’s labels. A’s final assignment consumes B’s score. The supervised subsample reduces how frequently
     this occurs; it does not remove the dependency.

     This establishes a leakage route, not its direction or magnitude. It might be small at this scale.

     Static cosine ranks computed over the full unlabeled graph do not create this label path. Prediction-derived ranks do.
     Computing either ranks or rival counts on only the supervised sample creates a separate train/inference mismatch.

     Change: Audit one outer fold A. Score A with its existing exclusion model; score each rival fold B with a model excluding
     both A and B. Keep supervised training sample counts comparable. This requires four additional fits for one outer-fold check,
     followed by scoring and assignment.

     Compare ordinary OOF versus this audit on assignment gains, conflict winners and selected cardinalities. Any gold-learned
     normalization must follow the same exclusion boundary. Use unconstrained OOF predictions as an additional diagnostic
     meanwhile.

     Expected impact: Unknown, possibly either direction. Confidence: high in the dependency, low in material inflation.

     Cost: Approximately 1–2 hours to implement the audit, plus four model fits and a graph scoring pass. No retrieval rebuild if
     preprocessing and ranks are label-free.

  5. HIGH — Ownership can reserve a record for an S1 that subsequently refuses to predict it.
     Reference: §3 Decision rule, steps a–b.

     Claim: Highest-probability ownership followed by independent top-k selection can strand useful candidates.

     Why: An S1 with three certain matches should reject an optional fourth at probability 0.6: exact expected scores are 0.9625
     without it versus 0.91579 with it. Another S1 can still benefit from that record at a lower probability, depending on its
     other candidates and singleton probability. Your initial ownership step prevents reconsideration.

     Change: Preserve competing edges. After selection, identify pool records whose nominal winner does not emit them. Perform a
     bounded release-and-reconsideration pass for alternative S1 entities. Eventually compare ownership by marginal entity
     utility, including abstention.

     Keep total-truth and singleton estimates based on pre-ownership information; otherwise reallocation changes the decoder’s
     supposed truth distribution.

     Expected impact: Depends on conflict frequency and the number of stranded gold edges. Confidence: high in the failure
     mechanism; population gain unknown.

     Cost: Approximately 1–3 engineering hours and compact score-table scans. No new retrieval or pair features.

  6. MEDIUM — The feature set can reward information destroyed by normalization and shared missing values.
     Reference: §2 common names/None; §3 Normalization and Features.

     Claim: Your listed name features primarily compare aggressively normalized names. They cannot reliably recover distinctions
     erased by digit substitution or legal-word removal.

     Why: Mixed alphanumeric tokens can legitimately contain digits—3M, B2B, 7Eleven are format examples. Similarly, identical
     common names require stronger address evidence than rare names. Candidate counts alone do not encode name frequency.

     Literal None also needs explicit treatment: comparing "none" with "none" must not produce strong address-agreement evidence.
     Empty house/postal/street values must not count as agreement.

     Change: Add conservative-name similarity and exactness alongside core-name features. Add label-free name frequencies and
     rarity-weighted address overlap. Explicitly canonicalize missing-value sentinels; gate every equality and fuzzy-agreement
     feature on field presence, retaining separate missingness flags.

     Expected impact: Unmeasured. Preventing a false positive on 1% of true singletons would recover 0.00056 globally, assuming
     the training singleton rate transfers. Confidence: medium; missing-value handling is implementation-dependent.

     Cost: Approximately 1–3 engineering hours, one feature-generation pass and a model comparison.

  Before selecting improvements, run the outer-fold dependency check and verify the vectorizer configuration. For hashed sublinear
  TF-IDF, the intended input is nonnegative raw counts: HashingVectorizer(norm=None, alternate_sign=False), followed by the TF-IDF
  transformer. The defaults are normalized, signed values; applying (1+\log(tf)) to those is not ordinary sublinear count
  weighting. Your exact constructor settings were not supplied. HashingVectorizer defaults
  (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html), TF-IDF definition
  (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)

  The next five improvements, ordered by provisional gain per engineering hour:

   Order    Change                                                      Promotion criterion
  ━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       1    Calibrated threshold comparator plus explicit singleton     Beats the current decoder on honest held-out entities;
            model                                                       inspect singleton and low-cardinality changes
  ───────  ──────────────────────────────────────────────────────────  ───────────────────────────────────────────────────────────
       2    Missingness gates, conservative-name features and name/     Reduces common-name false positives without hurting
            address rarity                                              missing-address positives
  ───────  ──────────────────────────────────────────────────────────  ───────────────────────────────────────────────────────────
       3    Release rejected ownership winners                          Recovers stranded gold edges with positive paired macro
                                                                        gain
  ───────  ──────────────────────────────────────────────────────────  ───────────────────────────────────────────────────────────
       4    Targeted India character retrieval, corrected acronym       Positive incremental oracle gain that survives
            joins and join-product cap                                  classifier/decoder evaluation
  ───────  ──────────────────────────────────────────────────────────  ───────────────────────────────────────────────────────────
       5    Guarded French variants and transfer-tested feature         Preserves known-country performance and improves relevant
            ablations                                                   format invariants; French gain remains unverified

  Include these diagnostics with the CV follow-up:

  - Scores and denominators: per-fold US/India F0.5 and oracle, S1 counts, gold-pair counts, and truth-count bins including zero.
    Your reported 0.9886 oracle is the equal-country audit mean. The normalized known-country test-weighted oracle is
    approximately 0.98790. Report France separately as unknown.

  - Stage losses: gold edges lost to retrieval, ownership and final selection; macro score before/after ownership; counts of
    reserved-but-unemitted pool records and how often their runner-up is the gold owner.

  - Decoder comparisons: current rule, tuned threshold, calibrated threshold, singleton-aware rule and any corrected utility rule,
    evaluated on identical candidates and score provenance.

  - Calibration: reliability tables for all candidates and ownership winners; actual singleton frequency versus prod(1-p); total
    truth count versus summed probabilities before/after ownership; competing probabilities summing above one per pool record.

  - Retrieval failure slices: script, missing address, common-name frequency, corruption operator, cap exclusion and word top-k
    exclusion. Include incremental oracle and candidate counts for each proposed rescue.

  - Graph provenance: whether ranks/gaps use cosine or model scores; whether degrees were computed before supervised sampling; the
    one-outer-fold exclusion comparison.

  - France without labels: empty/zero-vector rates, house/postal parse rates, normalization collisions, cosine and rival-degree
    distributions, predicted-empty rate and predicted cardinality distribution. Do not force these to match training.

  - Uncertainty: the five paired fold differences, changed-entity count, country regressions and number of variants tried. Five
    correlated fold estimates and a two-standard-error rule are not a reliable guarantee against selection noise.