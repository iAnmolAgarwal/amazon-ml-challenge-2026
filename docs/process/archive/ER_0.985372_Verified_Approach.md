VERDICT: **0.985372 macro F0.5 verified on a 17,241-entity conditional holdout. This is not a leaderboard score or a clean end-to-end CV result.**

# Exact approach behind 0.985372

Written 25 September 2026. Experiments ran on the existing AWS CPU machine. The competition repository was read-only; all experiment code, models and predictions were written to scratch directories. This document is an implementation handoff, not a claim that the whole pipeline already scores 0.985372 on test.

**Does this establish 0.98 on the leaderboard? No.** If the measured US/India scores transferred unchanged, the supplied test weights give `0.38*0.987085645 + 0.47*0.982803577 + 0.15*France`. France at 0.900 would imply only **0.972010 overall**. France would need approximately **0.953265** to reach 0.980. Neither that French score nor unchanged US/India transfer has been verified. This file is a tested-method handoff, not a promised leaderboard score.

## 1. Measured result and its limits

| Evaluation slice | Strong old-candidate model | Same model plus new retrieval | Gain |
|---|---:|---:|---:|
| All 17,241 evaluated S1 | 0.984105467 | **0.985371675** | **+0.001266208** |
| US | 0.985656157 | **0.987085645** | +0.001429488 |
| India | 0.981782017 | **0.982803577** | +0.001021560 |

The incremental retrieval comparison is positive in all five inherited folds. Paired fold SE is **0.000152414**; the conditional bootstrap 95% interval is **[0.00092590, 0.00161708]**. Within the evaluated cohort it adds **247 true matches and six false matches**: 246 added true matches are NEW edges, and one is an OLD-edge decision change. Outside the cohort, one entity improves, none worsen, and one false match is removed.

The original 006 model scores **0.974772783** on these same entities. That is the matched starting point; do not subtract scores from a different cohort or add this improvement to the separate 0.971 public submission.

Important scope:

- The cohort is the 17,241 owners with `s1_idx % 128 == 6`. Indices choose evaluation partitions only; they are not predictive features.
- All **98,021,508 original candidate edges** participate in global ownership. Only cohort owners receive replacement old-edge probabilities and additional edges; outside owners retain original 006 scores.
- Correction fitting and lexical statistics exclude owner residues 2–6 before fitting. The cohort-6 models, retrieval candidates and predictions were frozen before its evaluation outcomes were disclosed.
- Upstream 006 OOF predictions, the normalization dictionary, and parts of the R5 training provenance are inherited. This is an independently held-out correction/retrieval intervention, **not fully nested end-to-end validation**.
- A clean US outer validation is being built separately. France has no labels and is **not included in the 0.985372 measurement**.
- The score on all 2.21M training entities after changing only this small cohort is not 0.985372. The number above is the evaluated cohort's score with full-graph competition.

## 2. Architecture to implement

The method has two routes:

1. **OLD pairs:** original candidate graph → original stage-2 probability → R5 residual correction → lexical corruption residual correction.
2. **NEW pairs:** additional name/address character retrieval → independent 66-input pair classifier.

Union both scored routes, resolve pool ownership globally, then use the existing expected-F0.5 prefix decision and probability floor 0.6.

Do not replace the strong OLD scores with the new-pair classifier. That classifier alone scores only **0.976343840 on OLD / 0.977640304 on UNION** on this cohort. It is useful for newly retrieved edges, not as a replacement for the stronger old-edge model.

## 3. OLD route: specificity plus learned corruption likelihoods

### 3.1 Preserve the existing original pipeline

Use the original normalization, word-TF-IDF top-40 plus exact-key candidate set, and 006 stage-2 probabilities. Preserve original strings alongside normalized views. Keep countries open-ended; retrieval comparisons stay within country.

### 3.2 R5: 33 inputs, residual LightGBM

Inputs are `logit(p006)` plus these 32 features:

| Family | Exact input names |
|---|---|
| Name token specificity, shared and one-sided IDF mass | `ns_sh, ns_a1, ns_b1, ns_a1_frac, ns_b1_frac, ns_a1_max, ns_b1_max, ns_cos, ns_n_a1, ns_n_b1` |
| Address token specificity | `as_sh, as_a1, as_b1, as_a1_max, as_b1_max, as_cos` |
| House and unit relationships | `h_suffix, h_diffl, h_near, h_near_st, h_lev, h_swap, h_in_p, h_in_s, u_eq, u_conf, u_one` |
| Retained legal-form relationships | `lg_eq, lg_sub, lg_disj, lg_one` |
| Name character-trigram cosine | `cos_nc3` |

Use the frozen feature implementation rather than reconstructing meanings from abbreviated names. In particular, `h_diffl` is based on **absolute** house difference; it is not the separate signed all-number channel.

R5 model: binary LightGBM, **300 rounds, 31 leaves, learning rate 0.05, min_data_in_leaf 200, feature_fraction 0.8, bagging_fraction 0.8, bagging_freq 1, lambda_l2 1, seed 42**.

Fit with `init_score = logit(p006)`. At inference:

```text
p_r5 = sigmoid(logit(p006) + r5_model.predict(features, raw_score=True))
```

**The initial logit is not automatically restored by ordinary booster inference. Omitting it produces the wrong model.** Clip probabilities to `[1e-7, 1-1e-7]` before taking logits.

### 3.3 Lexical corruption model: 75 inputs, residual LightGBM

Add **42 learned soft features** to the R5 logit and 32 R5 features. The six channels are:

- inserted name tokens: `nadd`;
- deleted name tokens: `ndel`;
- inserted address tokens: `aadd`;
- deleted address tokens: `adel`;
- legal-form transitions: `legal`;
- contextual character edits in names: `nedit`.

For each channel, encode seven summaries: **count, maximum, minimum, sum, mean, unseen fraction, maximum support**. Values derive from positive-versus-negative token/edit likelihoods learned from actual candidate pairs. Minimum support **5** and shrinkage **n/(n+10)** are part of the frozen encoder. Preserve its exact smoothing and token extraction from the referenced source; do not substitute unsmoothed token target means.

Counts deduplicate **(owner, label, channel, token)**. Training features use statistics excluding that owner's whole fold. Inference statistics use eligible training owners only. Never let an evaluated owner's labels create its own token likelihoods.

Lexical model: binary LightGBM, **100 rounds, 9 leaves, min_data_in_leaf 100, learning rate 0.04, lambda_l2 3, seed 42**, deterministic column-wise fitting.

Fit with `init_score = logit(p_r5)`. Apply only where **original `p006 >= 0.01`**:

```text
p_old = sigmoid(logit(p_r5) + lexical_model.predict(75_inputs, raw_score=True))
```

Outside this gate, keep `p_r5` unchanged. The gate is based on original p006, not the newly corrected probability.

The lexical addition replicated over R5 on separate 17,241-owner cohorts:

| Cohort | R5 | R5 + lexical |
|---|---:|---:|
| 3 | 0.981284031 | 0.983410228 |
| 4 | 0.981063616 | 0.983076195 |
| 5 | 0.981673693 | 0.983593694 |

These remain conditional correction holdouts with inherited upstream limitations.

## 4. NEW route: two character retrieval views

Retrieve separately over **normalized core name** and **normalized address**. Each query searches the complete pool for its country; there is no state partition.

The frozen implementation uses character **3-grams**, float32, `min_df=2`, sublinear TF `1 + log(tf)`, smoothed IDF, and L2 normalization. Vocabulary and IDF are fit over the complete provided training pool across countries; candidate comparisons are restricted by country. This detail matters for exact reproduction.

Pool chunks contain **125,000 rows**. Query batches contain **1,024 rows**. Retrieve **top 40 per view** using `sparse_dot_topn`, and merge per-chunk candidates using global pool indices. Empty or zero-similarity vectors may produce fewer than 40 candidates. The measured implementation used two threads.

Ties are score-valid at each library cutoff; retained tied results are ordered by global pool index. Do not claim bitwise equivalence to a different top-k implementation when cutoff ties exist.

Keep the union of both retrieval views, then perform an exact anti-join on `(s1_idx, pool_idx)` against OLD:

```text
retrieval = unique(name_top40 UNION address_top40)
NEW = retrieval ANTI_JOIN OLD ON (s1_idx, pool_idx)
FINAL = OLD UNION NEW
```

Measured cohort counts:

| Set | Pairs |
|---|---:|
| OLD | 766,951 |
| Union of both retrieval sources, before subtracting OLD | 1,335,648 |
| Genuinely NEW | **965,732** |
| FINAL | **1,732,683** |

This recovers **384 of 1,217 previously unretrieved gold edges**. Candidate oracle macro F0.5 rises **0.99374909 → 0.99568006**. The realized decision gain is smaller, as reported above. Do not substitute oracle gain for model gain.

## 5. Score NEW pairs with the fixed absolute66 model

Train a separate binary classifier using **34 existing pair-local features + the 32 R5 features**. The 34 are:

```text
is_s3, a_empty, a_empty_p, n_tok, n_tok_p, is_web_p,
n_ratio, n_tset, n_tsort, n_partial, n_jw, j_ratio, j_partial,
alt_ratio, a_tset, house_eq, house_prefix, street_eq, postal_eq,
postal_conflict, num_jacc, nonlatin_name_p, nonlatin_addr_p,
acronym, join_eq, nf_s1_s1, nf_s1_pool, nf_p_s1, nf_p_pool,
c_ratio, c_tset, c_jw, a_ratio_g, a_partial_g
```

No original probability, residual init score, graph rank/gap/count, word cosine, retrieval-source flag, or p006 gate enters this classifier. Reference/pool name-frequency inputs remain included.

Frozen model: binary LightGBM, **300 rounds, 63 leaves, min_data_in_leaf 200, learning rate 0.05, lambda_l2 1, feature_fraction 0.8, bagging_fraction 0.8, bagging_freq 1, seed 42**, deterministic column-wise fitting; no class weights.

The actual experiment trained on **9,418,424 eligible existing candidate pairs / 212,126 owners / 720,389 positives**, excluding residues 2–6. “All training pairs” here means all eligible rows of that cached training sample, not all 98M original graph edges.

Routing must be exact:

```text
score(OLD pair) = strongest75 probability, unchanged
score(NEW pair) = absolute66 probability
```

The hybrid preserves OLD probabilities bitwise. The two score sources have different training distributions; the matched holdout establishes this particular routing, not universal calibration under arbitrary retrieval expansion.

## 6. Global decision and evaluation

Append genuinely NEW pairs before ownership resolution. For each pool record, keep the highest-probability S1 with the existing deterministic tie rule. Within each S1, sort retained probabilities descending and select the prefix maximizing:

```text
5 * sum(p[:k]) / (sum(all_owned_p) + 4*k)
```

Compare against empty prediction utility `product(1-p)`, then remove selected pairs with **p <= 0.6**. Preserve the existing order of these operations for reproduction.

An evaluator that left-joins predictions onto OLD silently discards the new candidates. The correct evaluator removes the entire cohort's OLD portion, appends its complete replacement graph, and retains every outside edge. Evaluate both the cohort and any affected outside owners. Unretrieved gold matches remain in metric denominators.

For a production package, `candidate_pairs.tsv` must contain the exact scored OLD∪NEW set, including rejected pairs; all emitted matches must belong to it. Include every test S1, including France and empty predictions.

## 7. Compute and scaling

Measured for this 17,241-owner experiment on the AWS CPU machine:

- Name retrieval after index preparation: approximately **116.5 seconds**.
- Address retrieval after index preparation: approximately **400.1 seconds**.
- Absolute66 fitting: **186.7 seconds**.
- Strongest75 rebase: **255.0 seconds**.
- Union feature generation and scoring: **133.0 seconds**.
- Independent four-arm full-graph evaluation: **201.4 seconds**, approximately **5.59 GB peak**.

These are separate measured stage timings, under varying shared-machine load; they are not a guaranteed full-run schedule and exclude index preparation where stated.

Scaling this candidate density to all training owners projects approximately **221.8M total pairs**. The extra 66-feature float32 matrix alone would be about **32.6 GB**. Stream feature/scoring batches and persist compact `(s1, pool, probability)` triples. Do not materialize OLD and NEW dense matrices simultaneously on a 64GB machine.

Cheaper retrieval depths have only oracle results so far: top-10 retains 46.7% of top-40 oracle gain with 14.0% of its additional pairs. Reverse retrieval is under investigation. Neither is part of the verified 0.985372 result.

## 8. What must not be bundled into this score claim

- No French parser repair, pseudo-label adaptation or synthetic French training is included.
- No 18-field signed numeric extension: its latest trial helps US but harms India.
- No extra raw formatting features, address character-cosine feature, joint alias feature, or Platt calibration: these failed meaningful incremental validation.
- No hard decoy-offset veto, native-word deletion, forced cardinality, or automatic pool-cluster propagation.
- No external entity data, web lookup, pretrained neural model, or GPU is needed for this method.

## 9. Evidence, source files and reproducibility handoff

Local research root: `/private/tmp/aml_perfect_round2/`.

| Purpose | Local path relative to research root |
|---|---|
| Independent exact evaluation | `audit/cohort6/evaluation.json` |
| Frozen prediction paths/checksums | `r5_blind/retrieval6/all_predictions_frozen.json` |
| Absolute66 exact features and training manifest | `r5_blind/retrieval6/train_manifest.json` |
| Strongest75 exact parameters and provenance | `r5_blind/retrieval6/rebase75/manifest.json` |
| Absolute66 training code | `r5_blind/retrieval6/train_absolute.py` |
| NEW feature/scoring code | `r5_blind/retrieval6/score_union.py` |
| Exact hybrid routing | `r5_blind/retrieval6/freeze_hybrid.py` |
| Strongest75 cohort-6 rebase | `r5_blind/retrieval6/run_rebase6.py` |
| Lexical statistics/cross-fitting reference | `r5_blind/run_scaled.py` |
| Character retrieval/index implementation | `transductive_indic/retrieval_cohort6/screen.py` |
| Retrieval oracle, misses and depth tradeoffs | `tail/cohort6/SEALED_REPORT.md` and `oracle.json` |

Remote retrieval/scoring artifacts: `/tmp/aml_perfect_round2_r5_blind/retrieval6/`.

Frozen hybrid prediction SHA256:

```text
8db23c4f868d980cc4692bd5328bda717d5f90b60b4e1e21da1bb9afe579a6de
```

Frozen absolute66 model SHA256:

```text
e44ab782bc69d3d10bde4db45bb720246ccd9ea146a20d663d5184723b4f5997
```

These scripts depend on existing source modules and cached artifacts; this Markdown file is **not a standalone reproducible submission package**. Before production use, the coding team must integrate the frozen definitions, replace scratch-specific paths/cohort filters with explicit configuration, generate honest training predictions for every residual layer, rebuild train/test consistently, run complete validation, and package all learned resources with exact reproduction instructions. Scratch files may be ephemeral; preserve the required sources, models, learned statistics and manifests in the team's authorized artifact storage before cleanup.

## 10. Next verification priority

First verify the improvement on a complete held-reference graph whose upstream models exclude every evaluated owner's labels. Then measure reverse retrieval and any new scorer against a matched frozen control. A higher oracle or a sum of improvements from different cohorts is not evidence of 0.99.
