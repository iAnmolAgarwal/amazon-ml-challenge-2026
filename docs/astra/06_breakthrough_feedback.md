VERDICT: No evidence supports expecting +0.025 from any single proposed method. The three credible bets are learning the
  corruption process, a multilingual cross-encoder, and pool clustering that expands retrieval. A set model using the existing
  features is a weaker bet.

  References below are to your supplied sections, not verified code locations. Gain ranges are speculative planning ranges, not
  measured forecasts; zero gain or regression remains possible.

  1. HIGH — Your target requires recovering several distinct losses, not replacing the decoder.
     Reference: §3, country scores and loss decomposition.

     Assuming the stated country mix and CV transfer:

      Opportunity                                                  Available full-test score gain, derived
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      Perfect US/India classification within current candidates                                    0.02660
     ───────────────────────────────────────────────────────────  ─────────────────────────────────────────
      Perfect US/India retrieval, beyond current oracles                                           0.01044
     ───────────────────────────────────────────────────────────  ─────────────────────────────────────────
      France improves from inferred 0.9069 to 0.96                                                 0.00796
     ───────────────────────────────────────────────────────────  ─────────────────────────────────────────
      France becomes perfect                                                                       0.01396

     Thus +0.025 with unchanged France and retrieval would require recovering 94% of the remaining known-country classification
     loss. That is an implausibly demanding expectation for a replacement classifier.

     An illustrative route to +0.025 is:
      - +0.005 from US/India retrieval: recover approximately 48% of its remaining oracle loss.
      - +0.012 from US/India classification: recover approximately 45% of its remaining classification loss.
      - +0.008 from France: improve it by approximately 0.053.

     These are required outcomes, not forecasts. They explain why the strongest bets must add information and address France. The
     public leader’s score does not identify its architecture or establish its private score.

  2. HIGH — Bet 1: learn a probabilistic corruption model, especially cross-script correspondences.
     Reference: §1 operator catalogue; §2 normalization and retrieval.

     Mechanism. A deterministic inverse is impossible for acronyms, missing addresses, arbitrary DBA names and ambiguous
     transliterations. Also, “deduplicated S1” does not establish that S1 is the original uncorrupted record.

     The missing component is an ambiguity-preserving model of how one record becomes another, rather than more destructive
     canonicalization. It should recognize that a particular difference is an observed generator operation and distinguish that
     explanation from accidental agreement between common businesses.

     Conceptually, score:

     ## [
     \log P(\text{pool record}\mid\text{reference},\text{corruption model})

     \log P(\text{pool record}\mid\text{background}).
     ]

     The background term matters: an acronym or common name admitting many possible originals is weak evidence.

     Concrete implementation:
      - Build training-fold gold clusters containing S1 and its pool variants.
      - Extract same-field Latin/Indic parallel variants, including pool-to-pool pairs.
      - Mine whole-name/phrase correspondences, then token correspondences. Keep name and address mappings separate.
      - Count support by distinct S1 owners, not repeated edges from one business.
      - Initially retain mappings with support from at least five owners and strong destination dominance. Preserve several
        alternatives when ambiguous.

      - Use reliable token alignments to train a small many-to-many character transducer if dictionary coverage is insufficient.
        Model Indic grapheme sequences to Latin character segments; retain a small beam of alternatives.

      - Add mapped name/address views and mapping-confidence features. Preserve original strings.
      - Retrieve against mapped Indic pool records using all India queries, including Latin-only S1 records.
      - Learn operation-specific residual scores for token splits, prefix insertion, house-number edits and alternate-name
        patterns. Evaluate these on ambiguous candidates rather than enumerating every possible transformed record.

     There are approximately 565,000 native-name gold edges and 687,000 native-address edges, derived from your rates and gold-
     pair count. These overlap and are not independent parallel examples, but they justify testing a learned channel.

     Loss buckets and magnitude. India retrieval alone has a full-test ceiling of 0.00870. The 56% Indic share is pair-weighted;
     multiplying that percentage into the macro ceiling would be invalid. Better cross-script evidence can additionally recover
     selection and ownership losses.

     A successful version might recover roughly 0.002–0.005 training macro from retrieval, plus 0.001–0.005 from classification.
     Those are guesses about recoverable fractions, not consequences of the operator frequencies.

     Expected overall gain: 0 to +0.010, low confidence; approximately +0.003–0.010 is the successful-pilot scenario.
     Cost: Dictionary pilot 2–3 engineering hours; useful complete implementation 5–10 hours. A general character transducer adds
     approximately 6–12 hours. CPU is sufficient.

     Validation and risks: Every mapping, alignment model and operation estimate must exclude the outer fold’s labels. Report
     performance on unseen entities and separately on previously unseen tokens. Do not mistake a city/business-word co-occurrence
     for a translation. Script-gate the Indic component so it cannot corrupt France.

  3. HIGH — Bet 2: train a multilingual cross-encoder on difficult identity distinctions.
     Reference: §2, compression of raw fields into approximately 60 similarities.

     Mechanism. Your classifier sees similarity summaries. It cannot learn every distinction between which tokens differ, their
     roles, and whether the discrepancy is a plausible corruption. A cross-encoder sees both raw records jointly and can learn
     those interactions.

     This is the strongest neural bet because the known-country classification gap is larger than the retrieval gap. Paired
     transformer classification has primary research precedent in entity matching, but published benchmark improvements do not
     predict your gain. Ditto paper (https://arxiv.org/abs/2004.00584)

     Model and training recipe:
      - Start with microsoft/Multilingual-MiniLM-L12-H384, approximately 117M parameters, MIT licensed, with a new binary
        classification head. Its model card specifies XLMRobertaTokenizer; it warns against assuming ordinary AutoTokenizer
        compatibility. Microsoft model card (https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384)

      - Serialize each side as field-labelled conservative raw name and address. Preserve digits and suffixes.
      - Start at 192 total subword tokens per pair. Allocate space explicitly to both names and addresses; measure truncation by
        script. Increase to 256 only where necessary.

      - Pilot on 300,000 pairs, then approximately 500,000 per fit if useful.
      - Include positives, difficult negatives sharing names or addresses, and ordinary negatives—including candidates of
        singleton S1 entities.

      - Use real different-owner pairs for hard negatives. Simply changing a house number and calling the result negative is
        unsafe when house-number corruption is a positive generator operation.

      - Combine the neural score with existing number-sensitive evidence. Calibrate on a naturally distributed inner holdout;
        probabilities from balanced pair sampling must not directly replace calibrated matching probabilities.

     Do not score only the top three candidates. That could exclude the low-scoring positives responsible for selection loss.
     Define a label-free gate covering uncertain pairs, cross-script discrepancies, close ownership contests, and risky emitted
     matches.

     Before training, run an optimistic correction experiment on that gate: determine how much score could be recovered if its
     eligible decisions were corrected, accounting for affected ownership rivals. A gate with insufficient recoverable loss cannot
     support a large neural gain.

     Loss buckets and magnitude. It cannot recover unretrieved edges. As a sensitivity calculation, recovering 10–30% of the
     reported training losses would recover:
      - Ownership: 0.00046–0.00138.
      - Selection: 0.00136–0.00408.
      - False positives: 0.00112–0.00336.

     The assumed 10–30% recovery is unmeasured. France could benefit separately through multilingual representations and
     adaptation.

     Expected overall gain: 0 to +0.010, low confidence. This is a plausible broad classifier improvement, not an evidence-backed
     +0.025 solution.

     Compute budget: The following rates are assumed benchmark targets, not claims about T4 performance:

      Work                                                                 Arithmetic
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━
      Six fits × 500k pairs at 100 training pairs/s                    8.33 GPU-hours
     ──────────────────────────────────────────────────────────────  ─────────────────
      10M total OOF pairs + 5M test pairs at 500 inference pairs/s     8.33 GPU-hours
     ──────────────────────────────────────────────────────────────  ─────────────────
      Total before setup, calibration scoring and extra audit         16.67 GPU-hours

     By contrast, scoring all 98M training pairs at 500/s costs 54.4 GPU-hours. Full scoring is incompatible with the stated
     budget at that rate.

     Cost: Approximately 4–8 engineering hours for a usable pipeline, plus measured GPU time. Do not plan large CPU inference: at
     an illustrative 50 pairs/s, 10M pairs require 55.6 hours.

     Validation and France risks: Train each neural model without the evaluated S1 fold; keep calibration/blending inside the
     outer training data. Evaluate number-sensitive hard negatives explicitly. Multilingual pretraining does not guarantee
     identity matching or preservation of French address distinctions.

     Bi-encoder alternative: Use intfloat/multilingual-e5-small, MIT licensed with 384-dimensional embeddings, only as an initial
     Indic retrieval rescue: 96–128 tokens per record, 100k–300k fine-tuning pairs, and top-20 union with existing candidates.
     Mask other same-owner positives from in-batch negatives. E5 model card
     (https://huggingface.co/intfloat/multilingual-e5-small)

     Its plausible gain is 0 to +0.005, low confidence. Full-pool deployment requires fold-specific encoding/indexing: 10M float32
     embeddings alone occupy 15.36 GB, before the ANN index. Do not budget only the final-model index.

     A byte model is a lower-priority deadline bet. ByT5-small is Apache-2.0, but byte sequences are longer and the model requires
     task fine-tuning. Prefer using a character model for short transliteration units rather than full-record generative matching.
     ByT5 model card (https://huggingface.co/google/byt5-small)

  4. HIGH — Bet 3: build pool components that recover missing candidates and provide a better reference representation.
     Reference: §2, candidate-local coherence.

     Mechanism. Pool clustering becomes fundamentally different from coherence features only if it does something your current
     candidate list cannot:
      - Connect a retrieved pool record to an unretrieved sibling.
      - Recover a useful name/address variant absent from S1.
      - Evaluate a coherent group jointly rather than independently rejecting its weaker members.

     If clustering merely groups the existing 44 candidates, much of its value may already be captured by coherence.

     Concrete implementation:
      - Train a separate pool-to-pool matcher using same-owner gold pairs and difficult different-owner pairs. Both inputs are
        noisy, unlike S1-to-pool matching.

      - Do not treat all unmatched pool records as one class. Two records unmatched to S1 may belong to the same unknown business;
        their mutual label is unavailable.

      - Build high-precision seed components using rare, compatible name-and-address evidence.
      - Attach uncertain records softly to seed components. Do not let a weak attachment merge two established components.
      - Maintain several actual representative records and alternative name/address views per component.
      - Retrieve S1 candidates against those representatives, then expand to component members.
      - Score component-to-S1 attachment, retaining rejection and ambiguous-member handling. Permit several components to attach
        to one S1.

      - Export every expanded real pool candidate considered by the final matcher.

     Avoid enumerating every pair inside every S1 candidate list. At 44 candidates each, that is already approximately 2.1 billion
     comparisons on training queries. Use sparse global keys and bounded neighbors.

     First compute a diagnostic ceiling: using gold groups only for analysis, expand each retrieved true pool record to all its
     true siblings. This measures how much retrieval loss pool linkage could potentially eliminate. Never use those gold
     components as validation inputs.

     Loss buckets and magnitude. A successful implementation might recover approximately:
      - Retrieval: 0.001–0.003 training macro.
      - Ownership: 0.0005–0.0015.
      - Selection: 0.001–0.003.
      - False positives: uncertain; bad merges can make this bucket worse.

     These are speculative allocations. They overlap substantially with the anticipated coherence gain.

     Expected overall gain: 0 to +0.008 versus submission 2, low confidence. The incremental gain after coherence could be much
     smaller.
     Cost: Approximately 6–12 engineering hours and 3–8 CPU-hours for a bounded implementation. Avoid an unrestricted all-pool
     nearest-neighbor project.

     Validation and France risks: Learn pool-pair models only from training-fold owners. Build held-out components using text and
     model predictions, never held-out gold links. Measure contaminated components and their actual macro damage—not just edge
     precision. A few erroneous bridges can erase many correct attachments. French common names and shared premises may produce a
     different contamination rate.

  5. HIGH — France synthetic adaptation is a distinct component; ordinary pseudo-labeling is unlikely to supply a step change.
     Reference: §4, explicitly permitted transductive use.

     Mechanism. Generate training pairs in the target vocabulary and address format using provided French S1 records, then teach
     the model to recognize the observed corruption process.

     Concrete implementation:
      - Generate several variants per French S1 using operation recipes and compositions learned from training.
      - Create hard negatives from different French S1 identities, especially shared-name/different-address cases.
      - Mix synthetic examples with real US/India examples, or train a separate adaptation model.
      - For the neural bet, use this as a second training phase or a controlled mixed-data experiment.
      - Preserve real-label calibration initially. Synthetic positive/negative proportions do not establish French match
        prevalence.

      - If training a graph-level model, explicitly include synthetic singletons and distractors. Giving every synthetic S1 four
        positives teaches the wrong empty-set behavior.

     Do not label unassigned real French pool records negative. Some are precisely the matches you need to recover.

     Validation: Run the complete adaptation procedure twice:
      - Train without India labels, synthesize from India S1 text, evaluate on real India labels.
      - Train without US labels, synthesize from US S1 text, evaluate on real US labels.

     Learn operation statistics without the held-out country. Strong performance on generated validation pairs only demonstrates
     agreement with your own generator.

     Pseudo-labels: Prefer one conservative round, low weight, and agreement between genuinely different evidence sources—such as
     rare name/address agreement and pool consistency. Two related classifiers agreeing is weaker evidence. High-confidence
     pseudo-labels mostly reproduce already-correct decisions unless you demonstrate coverage of the difficult tail.

     Loss buckets: Potentially all French buckets, but their split is unknown. No defensible bucket-specific recovery estimate
     exists without French labels.

     Expected overall gain: 0 to +0.005, low confidence. Reaching France 0.96 would add approximately 0.008, but that is an
     ambitious target, not a prediction.
     Cost: A 2–3-hour country-transfer pilot; approximately 4–8 engineering hours for a complete experiment. CPU suffices for a
     feature model; neural adaptation shares the GPU budget above.

     Under your explicitly allowed transductive setting, using supplied test text for synthetic generation and pseudo-labeling
     stays within the stated data scope. External dictionaries, translation services and entity lookups do not.

  6. MEDIUM — A set/listwise model over the same evidence is not a strong step-change candidate.
     Reference: §3, learned utility and joint-assignment results.

     A small set transformer could model correlated candidates, but you have already tested learned utility and joint assignment,
     and coherence is being added. Changing the architecture does not restore missing textual information.

     A genuinely new set model would need raw-text embeddings or pool-component structure. Its gains should then be attributed to
     those inputs and compared against simply giving the same information to the existing classifier.

     Expected gain from a set model using existing features: 0 to +0.002, low confidence, with approximately 4–8 engineering hours
     plus nested validation. It is not one of the three main bets.

     If tested, use a small permutation-equivariant model with candidate masks, an explicit no-match output and pool-side
     exclusivity afterward. Split by S1; supervised inputs from base models must also exclude the outer fold. France cardinality
     priors remain unvalidated.

  7. MEDIUM — Check one cheap wildcard: whether training gold clusters provide a useful test-time alias catalogue.
     Reference: §1, provided data; train/test business overlap is unspecified.

     Spend approximately 15 minutes measuring rare, country-consistent conservative-name plus complete-address overlap between
     train and test. Then measure useful alias bridges: cases where a training gold cluster connects a test S1 to a difficult pool
     variant.

     Shared easy exact matches provide little additional value. Identical fields also do not prove identical business identity, so
     use catalogue evidence for retrieval/scoring rather than unconditional assignments.

     For +0.005 overall, the required arithmetic includes:

      Fraction of all test entities helped    Average improvement on those entities
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                        5%                                     0.10
     ──────────────────────────────────────  ───────────────────────────────────────
                                       10%                                     0.05
     ──────────────────────────────────────  ───────────────────────────────────────
                                       20%                                    0.025

     No positive forecast is justified before measuring useful coverage. France has no same-country training catalogue.

     Ordinary entity-disjoint CV cannot validate a known-business memory regime. Add a separate record/variant-held-out diagnostic
     and label it clearly; never populate ordinary OOF memory with held-out gold clusters.

  For all large bets, use a representative held-out S1 sample for the pilot, retaining full-country retrieval and relevant
  ownership competition. Error-enriched samples are useful for diagnosis, but their unweighted gains are not population gains.
  Final promotion requires your five-fold criterion and comparison against the accepted coherence incumbent, not just submission
  2.

  A clean outer-fold audit remains necessary for learned stacked features: ordinary pooled OOF scores can contain indirect label
  paths through rival models. Fit dictionaries, neural models, calibrators and cluster models without the outer fold’s labels.

  The three highest-value bets and their 2–3-hour checkpoints are:

   Rank                                       Smallest useful experiment                 Go/no-go checkpoint
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1. Learned corruption model                Fold-local dictionary/channel pilot on     Continue as a major bet if final India
                                              approximately 50k held-out India S1        macro improves approximately 0.006, or
                                              against the full pool                      oracle improves 0.005 with convincing
                                                                                         additional classification headroom.
                                                                                         Smaller gains can be kept cheaply, but
                                                                                         do not justify a long transducer
                                                                                         project.
  ─────────────────────────────────────────  ─────────────────────────────────────────  ──────────────────────────────────────────
   2. Multilingual cross-encoder              Benchmark actual tokenization/             Gate must contain substantial
                                              inference; train on 300k pairs;            recoverable loss; require approximately
                                              evaluate a representative 50k–100k S1      +0.004 known-country macro in the pilot
                                              slice                                      and a complete fit/OOF/test/audit
                                                                                         projection below 24 GPU-hours. Stop if
                                                                                         gains mainly duplicate coherence or
                                                                                         weaken number discrimination.
  ─────────────────────────────────────────  ─────────────────────────────────────────  ──────────────────────────────────────────
   3. Pool reconstruction and retrieval       Build conservative seed components         Require approximately +0.003 macro
   expansion                                  globally; test bounded attachment and      beyond coherence, including recovered
                                              expansion on held-out queries              out-of-list matches. Reject if component
                                                                                         contamination causes more than
                                                                                         approximately 0.0005 macro damage in the
                                                                                         pilot.

  These thresholds are proposed resource-allocation gates, not statistical facts. A pilot can kill a weak hypothesis; it cannot
  prove a private-France improvement. Add French synthetic adaptation to the neural bet only after its country-transfer check
  passes.

  For the 13 submissions, allocate ceilings rather than treating every slot as mandatory:

   Purpose                                                              Slots
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━
   Validated safe increments already underway                               3
  ───────────────────────────────────────────────────────────────────  ───────
   Large-bet submissions, including one controlled France adaptation        6
  ───────────────────────────────────────────────────────────────────  ───────
   Combinations of independently validated, complementary bets              2
  ───────────────────────────────────────────────────────────────────  ───────
   Final recovery/reproduction reserves                                     2

  With two Day-1 submissions already used, that fits 3 / 5 / 5 remaining daily slots. Use the first available slot for the
  strongest completed incumbent. For the France experiment, hold US/India predictions fixed where practical: any public change
  then comes from changed French predictions, although its private transfer remains uncertain.

  Reserve the final 8 hours for complete inference, audit and packaging. All new candidates must be reflected in
  candidate_pairs.tsv; synthetic records must never appear in either output. Pin learned mappings and model revisions. The model
  cards verify licences, but if the competition’s external-data prohibition also forbids pretrained weights, drop the pretrained
  neural branch—the licence alone does not authorize it.
