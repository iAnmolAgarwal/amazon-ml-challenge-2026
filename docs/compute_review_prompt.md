You are a senior engineer specialising in high-throughput record linkage and cost-efficient cloud compute. I need the **fastest end-to-end compute plan** (wall-clock, including setup time) for the workload below, under hard constraints. Recommend one plan, with a fallback. Be concrete and numeric. Challenge any assumption below that is wrong.

# 1. Context

Solo entry in a 72-hour ML competition (Amazon ML Challenge 2026, business entity resolution). About 65 hours remain. 15 leaderboard submissions in total, 5 per day. Every submission needs a fresh test-set inference run. Final ranking is on a private leaderboard.

**Task.** For each Source 1 (S1) business record, find all matching records in Source 2 and Source 3 (the "pool"). Records have `entity_id`, `business_name`, `business_address`, `country`. Output: one row per test S1 with a comma-separated list of matched pool ids, plus the candidate set the model scored. Metric: macro F0.5 per S1 entity.

**Rules that constrain compute choices.** No external data lookups (no geocoding, registries, internet data). Any model used must be MIT or Apache-2.0 licensed and at most 8B parameters. Code must be reproducible from a zip (README + pinned requirements).

# 2. Data size (measured)

| | S1 | Pool (S2+S3) |
|---|---|---|
| Train | 2,206,821 | 10,320,219 |
| Test | 1,732,544 | 9,969,589 |

Countries: train US 60% / India 40%; test India 47% / US 38% / France 15%. Matches never cross countries. Each pool record matches at most one S1. Average 3.46 matches per S1. 18% of India names and 22% of India addresses contain Indic scripts. Raw TSV 2.5 GB; zstd parquet 0.97 GB (loads in 2-5 s per file). The full train pool as a polars DataFrame uses 0.8 GB RAM.

# 3. Pipeline and measured costs

Stages (all CPU; no deep learning required so far):
1. **Normalization** of ~26M records (name + address views). Measured: ~450k strings/s per core in Python.
2. **Exact-key blocking** (polars joins on keys such as house number + street token, space-free name, initials + house number). Cheap; minutes.
3. **TF-IDF top-k retrieval**, 3 views (name char 3-grams, name+address words, address char 3-grams), per country, k about 20. Measured on the Mac:
   - Index build for 1M pool rows: 10 s; ~21 non-zeros per row for name char 3-grams.
   - Brute force (scipy sparse product + dense argpartition): 28 queries/s per 1M pool rows, single process.
   - `sparse_dot_topn` 1.2 (MIT, `sp_matmul_topn`, 8 threads): 505 queries/s per 1M pool rows.
   - Implied whole-country cost with sparse_dot_topn on the Mac: about 3.7 h per view for test, 6.5 h per view for train (3 views: about 30 h). Too slow.
   - Same-state partitioning was **measured and performs poorly** (train, gold pairs):
     - Last address segment as the state: 81.6% (US) / 80.9% (India) of gold pairs fall in the same partition; work 11x / 19x lower.
     - State vocabulary (60 US / 40 India most frequent S1 address segments) plus pool-segment aliases learned from gold pairs, unknown-state records searched country-wide: 96.7% (US) / 95.8% (India) of gold pairs recoverable; work only 6x / 3x lower. 14% of US and 36% of India pool records get no state.
     - Conclusion: geography is too noisy (codes, native scripts, reordered segments, cities mixed with states) to be a safe first filter. A different retrieval design is needed.
4. **Pair features** (rapidfuzz scorers, number/postal agreement, ranks): ~70M test pairs, ~40M+ train pairs if all train S1 are used. Measured: rapidfuzz `cpdist` token_set_ratio 2.4M pairs/s on 8 cores.
5. **LightGBM** binary classifier on ~5-20M labelled pairs, ~40 features.
6. **Global decision pass**: assign each pool record to its best S1, then per-S1 top-k selection maximising expected F0.5. Needs all (s1, pool, prob) triples at once: ~70M rows x 12 bytes ≈ 0.85 GB.
7. **Development loop**: 5-fold CV on a subsample of S1 entities using cached candidates; target 10-20 min per experiment.
8. **Per-submission build**: features + scoring + decision on the full test set; target under 45 min.

Memory estimate if all test pair features are materialised at once: ~6 GB at 20 candidates per S1, ~15 GB at 50. Streaming in batches is planned.

# 4. Compute options available right now

**Local**
- MacBook Air M3: 8 cores (4 performance + 4 efficiency), 8 GB unified memory, about 1-2 GB free while working, ~7 GB free disk. Also runs the coordinating AI sessions (3 coding agents + browser), which compete for RAM.

**Free notebooks**
- Kaggle CPU notebook: ~4 vCPU, ~30 GB RAM, 12-hour session limit, ~20 GB disk. Requires uploading data as a private Kaggle dataset (~1 GB) and a GitHub token in Kaggle Secrets. Not set up yet (~15 min of human setup).
- Kaggle GPU notebook: 2x T4 or 1x P100, ~30 GB RAM, ~30 GPU-hours per week. RAPIDS / cupy / cuML availability not verified.
- Google Colab free: ~2 vCPU, ~12 GB RAM, session limits. Colab paid tiers not purchased.

**AWS (account already set up; budget alerts at a 100 USD/month budget; about 100 USD in promotional credits)**
- EC2 quotas in us-east-1: **Standard (A, C, D, H, I, M, R, T, Z) on-demand = 8 vCPU; Standard spot = 8 vCPU** (both usable now, and they are separate pools, so one 8-vCPU on-demand plus one 8-vCPU spot instance can run at the same time). GPU (G/VT) on-demand quota request was **denied**; the spot GPU request is pending. SageMaker notebook quotas: only ml.t3.medium (2 vCPU, 4 GB); all other types 0.
- A private S3 bucket in us-east-1 exists for data and artifacts. An IAM user with a guardrail policy exists (currently allows only g5.xlarge, g5.2xlarge, t3.medium; would need a policy change for other types). Launch requires a key pair and security group (not created yet; ~15 min).
- Candidate instance types (measured specs and prices, us-east-1, Linux):

| Type | vCPU | Physical cores | RAM GB | Arch | On-demand $/h | Spot min $/h |
|---|---|---|---|---|---|---|
| c7a.2xlarge | 8 | 8 | 16 | x86_64 (AMD) | 0.411 | 0.167 |
| m7a.2xlarge | 8 | 8 | 32 | x86_64 (AMD) | 0.464 | 0.196 |
| r7a.2xlarge | 8 | 8 | 64 | x86_64 (AMD) | 0.609 | 0.166 |
| c7i.2xlarge | 8 | 4 | 16 | x86_64 (Intel) | 0.357 | 0.184 |
| m7i.2xlarge | 8 | 4 | 32 | x86_64 (Intel) | 0.403 | 0.163 |
| r7i.2xlarge | 8 | 4 | 64 | x86_64 (Intel) | 0.529 | 0.191 |
| c7g.2xlarge | 8 | 8 | 16 | arm64 (Graviton3) | 0.290 | 0.120 |
| m7g.2xlarge | 8 | 8 | 32 | arm64 (Graviton3) | 0.326 | 0.147 |
| r7g.2xlarge | 8 | 8 | 64 | arm64 (Graviton3) | 0.428 | 0.129 |
| c8g.2xlarge | 8 | 8 | 16 | arm64 (Graviton4) | 0.319 | 0.145 |
| r8g.2xlarge | 8 | 8 | 64 | arm64 (Graviton4) | 0.471 | 0.178 |
| g5.xlarge (GPU A10G) | 4 | 2 | 16 | x86_64 | 1.006 | 0.430 (quota 0) |

**Algorithmic options that change the compute bill** (all libraries MIT/Apache/BSD unless noted; verify):
- Same-state/region partitioning of TF-IDF retrieval (planned, unmeasured).
- Pruned sparse vectors: keep only the top-n highest-IDF n-grams per record before the sparse product.
- A hand-built inverted index over rare n-grams/tokens with posting-list caps, joined in polars (no matrix product).
- MinHash LSH (e.g. datasketch, MIT) on name/address shingles.
- Dense ANN: TruncatedSVD of char TF-IDF, or a small multilingual sentence encoder (e.g. multilingual-e5-small, MIT, ~118M params), indexed with FAISS (MIT) or hnswlib (Apache-2.0). Encoding 20M strings with a small encoder on CPU vs GPU is a cost question.
- GPU sparse/dense top-k (cupy / cuML on Kaggle GPUs), if available there.
- Key-only blocking first, then TF-IDF retrieval only for S1 entities whose key candidates are weak (reduces queries).
- Caching: candidate sets and features computed once and reused across experiments; only decision-rule and model changes re-run per experiment.

# 5. What I want from you

1. **Recommended topology**: which machine(s) run which stages (one-time candidate generation for train and test, development/CV loop, per-submission builds), and whether to run machines in parallel (e.g. Mac for development + one on-demand and one spot EC2 instance). Include setup time in the wall-clock.
2. **Recommended retrieval algorithm** for stage 3 at this scale, with an estimated throughput and total time on your recommended hardware, derived from the measured numbers above (show the arithmetic). Say what must be measured first to confirm it, and the go/no-go threshold.
3. **Estimated wall-clock** for: one-time train candidates, one-time test candidates, one development experiment, one submission build.
4. **Cost** in USD for the whole competition under your plan (on-demand vs spot, with interruption handling for spot).
5. **Memory plan**: peak RAM per stage and whether it fits each machine.
6. **Risks and fallback**: what happens if spot is interrupted, the Mac runs out of memory, Kaggle sessions time out, or an arm64 wheel is missing.
7. Anything in this setup that would breach the rules (external data, licences, reproducibility).

Rank alternatives briefly if the choice is close. Do not praise or summarise the question. Mark every number as either derived from the measurements above or assumed.
