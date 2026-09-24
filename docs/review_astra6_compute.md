VERDICT: Use an on-demand r7a.2xlarge plus a Spot r7a.2xlarge, run country-wide sparse retrieval once, and cache candidates and
  features. State partitioning is excluded.

  D = supplied measurement/specification or direct arithmetic from it. A = assumption, proposed configuration, target, or forecast
  requiring measurement. References below refer to your supplied sections; no implementation was audited.

  1. HIGH — Choose the existing x86 stack; move all substantial computation off the Mac.
     Reference: §4 compute options.

      Machine                  Assigned work
     ━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      On-demand r7a.2xlarge    Initial retrieval; then development, final training, submission scoring and global decisions
     ───────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────
      Spot r7a.2xlarge         Initial retrieval in parallel; remaining training candidates, feature caches and independent
                               experiments
     ───────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────
      Mac                      Editing, agents, orchestration and small inspections
     ───────────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────
      Kaggle CPU               Optional additional retrieval worker after the EC2 pipeline works; never required for a submission

     Each EC2 host provides 8 physical cores and 64 GiB RAM [D]. AWS confirms the physical-core mapping. This does not establish
     its speed relative to the Mac. AWS R7a specifications (https://aws.amazon.com/ec2/instance-types/r7a/)

     Choose R7a over M7a because the additional memory costs only $9.43 [D] over the remaining 65 hours [D] at your rates. That is
     cheaper than spending competition time eliminating intermediate allocations.

     Use the same x86 Linux environment on both hosts, with 200 GB gp3 each [A]. Allow 45–75 minutes [A] for the IAM change,
     access setup, environment, data transfer and initial benchmark. Use the existing S3 bucket for code and artifacts. A GitHub
     token is unnecessary if you transfer a code zip.

     Graviton is not the fastest-setup default: the pinned sparse_dot_topn release publishes Linux x86 wheels but no Linux ARM
     wheel. Its working Apple Silicon installation does not establish Linux ARM compatibility. Pinned release files
     (https://pypi.org/project/sparse-dot-topn/1.2.0/#files)

  2. HIGH — Keep country-wide sparse_dot_topn; parallelize query shards and prioritize the useful training subset.
     Reference: §3 retrieval measurements; §4 algorithmic options.

     Use the existing sparse implementation with:
      - Float32 sparse values and compatible integer indices.
      - Pool transposes prepared once per country/view.
      - Query shards sized for approximately 5 minutes [A] of work.
      - A single native parallel job per host, initially using 8 threads [A]; avoid nested multiprocessing and OpenMP
        oversubscription.

      - Persistent workers that reuse indexes between shards.
      - Explicit sorting if downstream features consume retrieval ranks.

     Shard queries within countries. Do not make a whole country an indivisible task. Both machines must be able to help finish
     the largest remaining job.

     The underlying extrapolation is:

     [
     T_{\text{view}}\approx
     \frac{\sum_c Q_cP_c}{505\times10^6}\ \text{seconds}.
     ]

     Here 505 queries/s against a million-row pool [D] is your measurement. Linear scaling with pool size is an assumption [A]:
     posting frequencies and vocabulary skew matter.

     Your stated whole-country estimates imply this work:

      Retrieval work              Mac-equivalent node-hours [D arithmetic from your estimates]
     ━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      Full training, all views                                                           19.50
     ──────────────────────────  ──────────────────────────────────────────────────────────────
      Full test, all views                                                               11.10
     ──────────────────────────  ──────────────────────────────────────────────────────────────
      Combined                                                                           30.60

     Do not wait for every training S1 before producing the first useful model. Prioritize a representative 10% training-query
     subset [A], always against the full training pool, plus the full test set.

     Let (h) be the measured EC2/Mac throughput ratio. Assuming query-linear scaling and balanced workers:

     [
     T_{\text{initial retrieval}}
     =\frac{0.1(19.5)+11.1}{2h}
     =\frac{6.525}{h}\text{ hours [A]}.
     ]

     [
     T_{\text{all retrieval}}
     =\frac{30.6}{2h}
     =\frac{15.30}{h}\text{ hours [A]}.
     ]

      Assumed EC2/Mac ratio [A]    Initial subset + test [A]    All train + test [A]
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━
                          0.75×                       8.70 h                 20.40 h
     ───────────────────────────  ───────────────────────────  ──────────────────────
                          1.00×                       6.53 h                 15.30 h
     ───────────────────────────  ───────────────────────────  ──────────────────────
                          1.25×                       5.22 h                 12.24 h

     These exclude setup, merging and feature production.

     Measure first: approximately 20,000 representative queries [A] against actual country pools, covering every view, Indic
     scripts, short names and common tokens. Measure index construction, retrieval, deduplication, serialization and peak memory.
     Name-trigram timing cannot establish address-trigram timing.

     Go/no-go: retain the schedule above only if measured aggregate (h\geq0.93) [A target]. That supports the initial retrieval
     tranche within approximately 7 hours [A]. Below that, revise the schedule immediately; defer full training-query coverage
     while continuing with the development subset.

     Rare-gram pruning is an optional bounded experiment—not assumed acceleration. Give it 30 minutes of investigation [A] and
     adopt it only with at least 2× measured speedup [A target] and at most 0.001 additional held-out macro-oracle loss [A
     tolerance], with uncertainty reported. Keeping only rare grams can discard the correctly shared grams when a rare token is
     corrupted.

     Raising the similarity threshold or lowering retained top-k does not proportionally reduce posting traversal. The library
     specifically cautions that threshold changes have little effect on its computational cost. Sparse retrieval implementation
     (https://github.com/ing-bank/sparse_dot_topn)

  3. HIGH — Cache features; otherwise the submission target depends on an unmeasured feature pipeline.
     Reference: §3 development loop and per-submission build.

     A fresh inference run does not require recomputing unchanged deterministic retrieval and features under the rules you
     supplied. For classifier changes, rerun model prediction, global decisions and output generation.

     Cache candidate IDs, static features and retrieval scores. Recompute learned ranks or score-gap features when their producing
     model changes. Include preprocessing, vocabulary/IDF, candidate definition, feature schema and fold provenance in cache
     identities.

     These are the planning envelopes at (h=1) [A]:

      Deliverable                                                                        Wall-clock estimate [A]
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      Cloud environment and benchmark ready                                                            45–75 min
     ─────────────────────────────────────────────────────────────────  ─────────────────────────────────────────
      Development-subset candidates/features available, prioritized                               2–3 h from now
     ─────────────────────────────────────────────────────────────────  ─────────────────────────────────────────
      Full test candidates, shared across both hosts after the subset                             7–9 h from now
     ─────────────────────────────────────────────────────────────────  ─────────────────────────────────────────
      First complete scored build using the subset-trained model                                 8–10 h from now
     ─────────────────────────────────────────────────────────────────  ─────────────────────────────────────────
      Full train and test candidate generation                                                  16–20 h from now
     ─────────────────────────────────────────────────────────────────  ─────────────────────────────────────────
      Development experiment on cached, bounded data                                                   10–20 min
     ─────────────────────────────────────────────────────────────────  ─────────────────────────────────────────
      Submission with cached features                                                                  20–35 min
     ─────────────────────────────────────────────────────────────────  ─────────────────────────────────────────
      Submission requiring feature recomputation                         35–45 min, only if the gates below pass

     The full-candidate estimate assumes both hosts remain mostly assigned to retrieval. If on-demand switches to continuous
     experiments after the first build, remaining training generation moves closer to 24–28 hours elapsed [A]. That can be the
     better competition schedule; full training-query coverage need not block experiments.

     For the submission build, benchmark the actual final model, not an arbitrary small forest:
      - Feature production at 100,000 pairs/s [A target] processes the supplied test candidate count in 11.7 minutes [A].
      - Prediction at 60,000 pairs/s [A target] takes 19.4 minutes [A].
      - Reserve 10 minutes [A] for remaining I/O, decisions, TSV generation and validation.
      - Combined uncached build: approximately 41 minutes [A].

     Failure to meet either throughput gate invalidates the uncached build target. Cached builds remain substantially easier.

     Your RapidFuzz measurement establishes only 29.2 seconds per token-set scorer over the test pairs [D]. It does not include
     other scorers, string gathering, number extraction, joins or model prediction.

     For development, initially use approximately 2 million candidate rows [A] and require each fold’s fit-and-predict to finish
     within 3 minutes [A]. Larger datasets need their own timings. Five-fold validation does not require deploying a five-model
     ensemble: at the prediction target above, that ensemble would take approximately 97 minutes [A] just for scoring.

  4. HIGH — The global decision pass does not require every triple in RAM.
     Reference: §3 memory estimates and global decision pass.

     The supplied test candidate count implies:

      Representation                                          Size [D]
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━
      Forty float32 features                                  11.20 GB
     ──────────────────────────────────────────────────────  ──────────
      Two uint32 IDs plus float32 probability                  0.84 GB
     ──────────────────────────────────────────────────────  ──────────
      Probabilities alone                                      0.28 GB
     ──────────────────────────────────────────────────────  ──────────
      Best owner and best probability per test pool record    79.76 MB

     For the stated “best S1, then per-S1 selection” decoder, use a streaming pass to update the best owner of each pool record
     while writing probabilities alongside cached candidates. Traverse the candidates again in S1 order to apply the selection
     rule. This reproduces the stated decoder without retaining all triples simultaneously.

     A more complex iterative reassignment optimizer would need a different analysis.

     Plan these stage peaks; all peak ranges are A, not measured:

      Stage                                                   Planned peak RAM per worker [A]
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      Normalization and bounded key joins                                             8–16 GB
     ──────────────────────────────────────────────────────  ─────────────────────────────────
      One country/view retrieval index and working buffers                            8–28 GB
     ──────────────────────────────────────────────────────  ─────────────────────────────────
      Feature production with bounded batches                                         8–20 GB
     ──────────────────────────────────────────────────────  ─────────────────────────────────
      LightGBM training on the proposed pair counts                                  12–32 GB
     ──────────────────────────────────────────────────────  ─────────────────────────────────
      Cached prediction and decision                                                  4–16 GB

     Set a 48 GiB peak-memory gate [A] on EC2. Release retrieval indexes before starting memory-heavy training.

     Additional corrections:
      - One full training-pool name CSR at the measured sparsity needs approximately 1.78 GB [D] before transpose copies and
        vocabulary structures. Address sparsity remains unmeasured.

      - The retrieval views can produce 103.95 million test edges [D] before deduplication and before exact-key candidates. The
        assumed final candidate count must be measured.

      - The dataset table totals 24.23 million records [D]. At the supplied normalization rate, processing both fields once
        implies approximately 108 core-seconds [D] before overhead. Verify that the benchmark includes the complete normalizer;
        normalization should not be the optimization priority if it does.

      - Keep full feature caches on EC2/S3. The Mac’s free disk is insufficient.

  5. MEDIUM — Budget for actual Spot pricing, storage and retained artifacts; alerts are not a spending cap.
     Reference: §4 AWS budget and price table.

     Conservatively budgeting both machines for the entire remaining window:

      Item                                             Budget
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━
      On-demand compute at supplied rate           $39.59 [A]
     ───────────────────────────────────────────  ────────────
      Spot compute at $0.25/hour [A]               $16.25 [A]
     ───────────────────────────────────────────  ────────────
      Both gp3 volumes                              $2.85 [A]
     ───────────────────────────────────────────  ────────────
      Public addresses                              $0.65 [A]
     ───────────────────────────────────────────  ────────────
      S3, requests and small-transfer allowance     $3.00 [A]
     ───────────────────────────────────────────  ────────────
      Total before credits and tax                 $62.33 [A]

     Storage assumes $0.08/GB-month [A pricing input] and the address calculation uses $0.005/address-hour [A pricing input],
     consistent with AWS’s published prices. EBS pricing (https://aws.amazon.com/ebs/pricing/), VPC pricing
     (https://aws.amazon.com/vpc/pricing/)

     At your observed Spot minimum, compute alone would total $50.38 [D conditional arithmetic]. That minimum is not a guaranteed
     available rate.

     Allocate $60–70 [A] gross for the plan. Credits may cover eligible charges; their applicability and existing account spending
     are unverified. Schedule shutdown at the competition deadline and remove unneeded volumes afterward. AWS Budgets can alert
     after costs have already accrued. AWS Budgets behavior
     (https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html)

  6. MEDIUM — Make the fallback a surviving CPU pipeline, not a new GPU stack.
     Reference: §4 compute options.

     Fallback: continue on the existing on-demand R7a, using the cached development subset and test artifacts. Replace Spot
     opportunistically; do not delay a submission waiting for it.
      - Spot interruption: upload completed shards continuously, with completion manifests. The proposed shard duration limits
        lost work to approximately 5 minutes [A], but replacement delay is unbounded. AWS interruption notices are best-effort.
        Spot interruption notices (https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-instance-termination-notices.html)

      - Quota limitation: you cannot replace Spot with another full-sized on-demand host while the existing one occupies the
        supplied quota. Purchasing-option quotas are separate. EC2 quotas
        (https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html)

      - Mac memory pressure: remote workers continue independently. Run them in persistent remote processes; the Mac holds neither
        indispensable data nor job state.

      - Kaggle timeout: assign only independently checkpointed retrieval shards. Resume unfinished shards elsewhere; never make
        Kaggle responsible for the only copy of a cache.

      - Missing ARM wheel: stay on x86. Do not spend the competition porting dependencies merely to reduce the hourly rate.

     After EC2 is running, adding Kaggle CPU can improve wall-clock. Time-box its setup to 30 minutes [A] and assign a memory-
     tested name-retrieval view. If it delivers 0.35× one EC2 host’s throughput [A], joining shortly after startup could save
     approximately 0.9 hours on the initial tranche [A] and 2.1 hours on full retrieval [A]. Those gains are excluded from the
     committed schedule.

     Alternatives rank as follows:

      Alternative                             Decision
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      Additional Kaggle CPU worker            Useful after setup and shard throughput pass the gate
     ──────────────────────────────────────  ──────────────────────────────────────────────────────────────────────────
      M7a Spot                                Accept for jobs demonstrated to fit its memory
     ──────────────────────────────────────  ──────────────────────────────────────────────────────────────────────────
      Graviton                                Consider only after a successful environment and throughput probe
     ──────────────────────────────────────  ──────────────────────────────────────────────────────────────────────────
      MinHash/custom postings engine          Experimental replacement; requires recall and end-to-end timing evidence
     ──────────────────────────────────────  ──────────────────────────────────────────────────────────────────────────
      SVD or multilingual encoder plus ANN    Adds representation, indexing and recall-validation work
     ──────────────────────────────────────  ──────────────────────────────────────────────────────────────────────────
      Kaggle GPU sparse stack                 Unverified setup and kernels; unsuitable as the deadline fallback

  7. LOW — Correct the licence assumption and make caches optional for reproduction.
     Reference: §1 rules; §4 libraries.

     sparse_dot_topn is Apache-2.0, not MIT. LightGBM is MIT-licensed. Both fit the supplied licence categories. Sparse library
     licence (https://github.com/ing-bank/sparse_dot_topn/blob/master/LICENSE), LightGBM licence
     (https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE)

     This plan needs no external entity data or pretrained encoder. Keep normalization resources derived from permitted inputs;
     cloud infrastructure and software packages do not supply additional business records.

     The zip must reproduce caches from the provided inputs without your cloud credentials. Include exact preprocessing and
     feature definitions, model artifacts or training commands, environment pins, deterministic ID mappings and tie-breaking, and
     the final candidate definition. Cached retrieval must remain the candidate set actually scored and reported; subsequent
     expansion must update that artifact.
