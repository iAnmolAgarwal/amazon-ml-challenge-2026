# Audit: the 0.985372 cohort claim (25 Sep)

Tags: **M** = measured in this audit, **R** = their number re-derived exactly, **D** = derived arithmetic, **G** = guess.
Nothing under `/private/tmp/aml_perfect_round2/` or the remote `/tmp/aml_perfect_round2_*` directories was modified.
Audit code/outputs: `amlc-a:/tmp/audit0985/` (audit*.py, lex_s10.py, *.json, logs); jobs via `rk/run.sh`, 5-8 GB, 2 threads.

## Verdict
- **0.985372 is measured correctly:** an honest held-out partial intervention; its caveats are second-order and shared by our CV.
- **As a package it is not the best option for us.** On the same owners, our next-build stage 3 (R5 + decoy markers)
  beats their old-edge stack by +0.0009 to +0.0013 (M). Their lexical layer adds only +0.0003 on top of our build (M).
- **Their new retrieval route is the part worth taking:** +0.00127 over 008 (M), the same gain as on their own stack.
- **Recommended:** our next build, then the new route behind a candidate filter, then lexical only if time allows.

## 1. Is 0.985372 correctly measured?
- I verified the SHA256 of the frozen files `hybrid_predictions.parquet` (8db23c4f...) and `strong_old_predictions.parquet`.
  Gold was mapped from `train_ground_truth.parquet` through the row order of `norm/train_{s1,pool}` (7,638,365 pairs, all mapped).
- **Decoder check:** I wrote an independent numpy version of the rule: owner = best S1 per pool record (ties go to the
  lowest s1_idx), then the best prefix by 5*sum(p1..k)/(sum p+4k) against prod(1-p), then drop p <= 0.6. It runs on the
  subgraph of every outside edge that touches the cohort's pool records, which gives exact cohort scores. It gives
  **hybrid = 0.9853716750** (R) and strong75 = 0.9841054673 (R). It also gives 006 = 0.974773 and 008 = 0.982474 on
  cohort 6, and the harness `er_model.decide` returns identical values.
- The saved 006 and 008 per-entity (t, k, tp, F0.5) values are reproduced with 0 mismatches on cohorts 3-7 (M).
- The paired hybrid vs strong75 result is +0.001266 (SE 0.000152, 5/5): 247 TP added, 6 FP added (R).
- Reproduction note: the handoff says clip to [1e-7, 1-1e-7]; the R5 code clips p006 at 1e-6 (lexical layer: 1e-7).

## 2. Leakage and optimism
- **No first-order leak found.** Code checked: `run_rebase6.py`, `train_absolute.py`, `score_union.py`, `freeze_hybrid.py`,
  `run_scaled.py`. The R5 refit, the lexical token statistics and fit, and the absolute66 fit all filter out
  residues 2-6 before labels are read, with asserts. The manifests record `labels_read_for_new_cohort: false`.
  - Lexical statistics are cross-fitted by harness fold on training owners; retrieval (char-3-gram TF-IDF) uses no labels.
- **Second-order paths (not measured, all small, G):**
  - The lexical layer's training init is our own research R5 5-fold OOF over the 10% sample
    (`research/r5/p3_s_...parquet`). Its fold models saw the roughly 1.7k sample owners in each evaluated residue.
    This affects only other owners' training offsets.
  - p006 is harness OOF. This is the same stacking structure as 008.
  - The Indic dictionary (`frozen_indic.json` = `er_dict`) was fit on all train gold, including evaluated owners
    (min support 5 owners). It feeds normalization, blocking, R5 features and lexical tokens. **Our CV has exactly
    the same dictionary caveat.**
- **Partial vs full rollout:** in the cohort protocol only the cohort is rescored and outside S1 keep their 006
  scores. I measured this effect with 008 (cohort from 008, outside from 006 vs the full 008 graph): +0.00003
  (SE 0.00003, 3/5) over 86,205 S1 (M). **It is negligible, so cohort numbers are comparable to our full OOF CV,**
  though noisier (SE 0.00015-0.0006 vs 0.00002).
- **Their correction models trained on less data:** roughly 206-217k owners, the 10% sample minus held residues,
  vs about 1.77M owners per fold for 008. Their gains are therefore conservative on that axis.
- **One unmeasured optimism:** only cohort S1 get new edges in the evaluation. In a full rollout, outside S1 would
  also claim those pool records. With 246 TP and 6 FP from new edges, the effect is bounded by the +0.00127 and is
  probably small (G).

## 3. Matched comparison with 008
All numbers below are M, paired per entity on the same S1, with outside S1 taken from the 008 graph (their arms move
by at most 0.00008 when outside S1 use 006 instead).

| cohort | 006 | 008 | their R5 | their R5+lexical | lexical - 008 (SE, folds +) | their hybrid |
|---|---|---|---|---|---|---|
| 3 | 0.974159 | 0.981865 | 0.981301 | 0.983443 | +0.00158 (0.00049, 5/5) | - |
| 4 | 0.973686 | 0.982798 | 0.981168 | 0.983158 | +0.00036 (0.00030, 4/5) | - |
| 5 | 0.973868 | 0.982773 | 0.981625 | 0.983588 | +0.00082 (0.00032, 4/5) | - |
| 6 | 0.974773 | 0.982474 | - | 0.984132 | +0.00166 (0.00025, 5/5) | 0.985399 (+0.00293, SE 0.00022, 5/5) |
| 7 | 0.973388 | 0.981976 | 0.980781 | 0.982983 | +0.00101 (0.00028, 5/5) | - |

- **Old-edge models (M):**
  - Pooled over cohorts 3/4/5/7 (68,964 S1), their R5+lexical beats 008 by **+0.00094 (SE 0.00012, 5/5)**.
    Nearly all of it is India (+0.0021; US +0.0002).
  - Their R5 alone: -0.00113 vs 008 (SE 0.00011, 0/5). A 50/50 logit blend with 008 gains +0.00094 over 008 across 5
    cohorts, no more than their stack alone: 008 adds nothing their stack lacks.
- **Our next build vs their stack, 10% sample (220,678 S1, `rk` rival protocol, 5-fold; `lex_s10.py`, M):**
  - Their recipe re-run on the sample (R5 stage 3 + lexical residual, nested-cross-fitted token statistics):
    0.98368, which is +0.00203 over R5. This replicates their +0.0020.
  - Our r5+r9 build (`v6` p59): **0.98457, +0.00089 over their stack (SE 0.00010, 5/5).**
  - The lexical residual on top of our build: 0.98489, **+0.00032 (SE 0.00003, 5/5)**, or +0.00031 against the
    same-layer control. About 85% of the lexical gain overlaps our label-free decoy markers.
- **Direct check against their frozen predictions (M):** on the 8,417 owners shared by the sample and cohorts 3-7,
  our p59 beats their frozen R5+lexical by **+0.00131 (SE 0.00021, 5/5)**.
- **New retrieval route on top of 008 (cohort 6, M):**
  - 008 + new pairs = 0.983745, **+0.00127 (SE 0.00015, 5/5; India +0.0010, US +0.0014)**, identical to its gain over
    their stack: it adds to any old-edge model. Top-10 per view only: +0.00073 (SE 0.00004); top-20: +0.00095.
- **Projection, rival protocol (D; assumes additivity, which holds for the new route):**
  - Our build + new route + lexical ≈ 006 + 0.0101 + 0.0013 + 0.0003 = **006 + 0.0117**.
  - Their hybrid on cohort 6 is 006 + 0.0106.
  - Full-CV equivalent of our build ≈ 0.9845-0.9855. With the new route, about 0.986-0.987 (D/G).

## 4. Costs
- **Candidates per S1:**
  - Ours today: test 45.55 (78,924,662 / 1,732,544, M); train cohort 6 old graph 44.5. Their final set: **100.5 on
    cohort 6 (+56.0), about 2.2x (M)**; about 175M test pairs (D). A large penalty under the new rule unless filtered.
- **Filter option:** 965,732 new pairs, of which 10,861 have absolute66 p > 0.001. That is **0.63 per S1, and those
  pairs keep all 384 new gold matches and the full +0.00127 (M).** At p > 0.05 it is 0.05 per S1 with the same gain.
  - The thresholds were read after unblinding, so re-confirm on fresh owners.
  - This counts only if a learned filter is accepted as candidate generation, which is the same assumption as the r6
    stage-1 filter (45.6 -> about 7 per S1 on test at zero loss, measured in r6). Together: about 7.6 per S1 (D).
- **Compute on one 8-core / 64 GB machine:**
  - Retrieval, their measurements: 116.5 s (name) + 400.1 s (address) for 17,241 queries at 2 threads, excluding
    index build. That scales to about 14.4 h at 2 threads for the 1.73M test S1 (D), or 3.6-5 h at 8 threads (G).
  - Features and scoring for about 97M new test pairs: about 2 h at 2 threads (D, from their 133 s per 1.73M pairs);
    a full 66-float32 matrix is 25.6 GB (D), so stream in chunks. Absolute66 fit: 187 s on 9.4M pairs (theirs).
  - CV of the route: on the 10% sample (about 1.8 h retrieval at 2 threads, D), not all 2.2M train S1 (18.4 h, D).
  - Lexical on the sample: 450 s including tokenization of 1.0M gated pairs (M). Full data means about 10M gated
    pairs held as Python token sets, so it must be chunked (G).
- **Test-shift risk (M on train cohort 6):**
  - New-route emitted pairs: precision 0.976 vs 0.9966 for old-graph emitted pairs. House-conflict share 0.31 vs
    0.115. **All 6 new-route false positives are house conflicts.**
  - With about 1.9x test decoys, expect roughly double the new-route false positives: about -0.0001 at 1.2e-5 each
    (D), small next to +0.0013. The absolute66 scorer lacks our decoy features; adding hd/dm is a likely fix (G).
- **France:** no labels anywhere.
  - Absolute66 and the lexical tables are India/US-only (French tokens mostly "unseen"); our r9 markers are label-free
    and can be fit on French test text. Run the new route for France only if its test house-conflict rate matches train.

## 5. Recommendation for the next submission
1. **Our next build first:** r9 decoy markers + hd into stage 3, full data, full 5-fold CV and the test guard.
   Expected +0.002 to +0.003 over 008 (D). It needs approval for the `src/` changes.
2. **Then the new retrieval route:**
   - Retrieval: their char-3-gram recipe, top-40 per view for name and address.
   - Scoring: a separate pair-local LightGBM, their 66 features, possibly plus hd/dm. Train it on old-graph train pairs
     with CV folds excluded. Old-graph scores stay unchanged.
   - Filter new pairs at p > 0.001 as candidate generation.
   - Validate on the 10% sample with fresh owners and the rival protocol.
   - Expected **+0.0012 to +0.0013 (M on 008)**. France is gated by the guard.
   - If a learned filter is not accepted as candidate generation, use top-10 instead: +7.8 pairs per S1 for +0.0007 (M).
3. **Lexical layer last, only if time allows:** +0.0003 over our build (M). Label-derived and France-blind.
4. None of this reaches 0.99 on the train cohort; France (15% of test) remains the largest unknown.
