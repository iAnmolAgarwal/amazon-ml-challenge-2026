# Audit of the teammate's working directory (read-only, 26 Sep ~21:50-22:40 IST)

Tags: **M** measured (their logs/files, not re-run by us), **D** derived (arithmetic on M), **G** guess.
Source: the SMB share of their repo working copy. Nothing on the share was modified, run or copied. Their caches, model outputs and Kaggle scores are **not on the share**. They live in the separate working copy their scripts run from (`~/amlc/tsk/cache`, `~/amlc/tsk/output*`, `~/amlc/kaggle/{band,out}` on their laptop) and in their private Kaggle kernels.
"P" = their pipeline, "O" = ours.

## 0. Bottom line
1. **No per-pair probabilities, OOF files, CE scores or model weights are on the share (M).** The only prediction artifacts there are the v08 `matching_results.tsv` and the v08 zip, which holds `output/candidate_pairs.tsv` and code but no probabilities. For the ensemble, they need to export 3 small file sets from their laptop (§2.3). That is about 15-30 min of their time. They have already started building the same thing (see 2).
2. **They are preparing a two-pipeline blend tonight (M).**
   - `tsk/src/team_blend.py` (commit 20:59) blends in-country at pair level on the union of candidates, keyed `(a_id, r_id, p)`, evaluated on their folds.
   - `tsk/src/tm_verify.py` (commit 21:15) re-scores "a teammate pipeline's" OOF and writes it as `theirs_train_{India,US}.parquet`.
   - Their design has a flaw: missing pairs are filled with p = 0, then logit-blended with a 1e-6 clip. A pair only one pipeline proposes then collapses to about p 0.01 at w = 0.5. So their logit variants act like an intersection and throw away the recall gain from the union (D). Only their `max` variant keeps union pairs.
   - **Coordinate so that the blend is run once, on our side, with our rule and our paired gate (§2.4).**
3. **The biggest reusable asset is their in-country cross-encoder scores.**
   - Models: e5-small `e5i` and e5-base `e5b`, trained on two entity halves. Every train band pair is scored by the half-model that never saw its S1, and every test band pair by both half-models.
   - On their stack these CEs gave **+0.0041 OOF** (tsk-c9 e5i: India +0.0051, US +0.0032, 10/10; M) and a further +0.0003 with e5b (tsk-c16, M). CE band AUC is 0.953-0.963 vs 0.925 for their LightGBM (M).
   - The scores are keyed by entity ids (`a_id`, `r_id`), so they drop straight into our comb as features. This replaces the e5-small CE we are training on Kaggle. **No fine-tuned weights exist:** `ce.py` never saves the model (M), so pairs outside their band cannot be re-scored.
4. **Compliance: the evidence on the share contradicts "same registered team" (§4).** Their upload log plus ours exceeds the 5/day cap on both 25 and 26 Sep, and their package zip carries a different team-name prefix. The coordinator relayed the human's confirmation that they are on our team. I still recommend the human check the portal team page before any of their outputs go into an upload.

## 1. What exists there that we lack (>= +0.0005 LB candidates)

| # | Asset / idea | Evidence | For O | Cost |
|---|---|---|---|---|
| 1 | **Pair-level ensemble with P** (in-country) | P in-country OOF 0.98782 (India 0.98640, US 0.98877; M, v08 manifest). O 023 ~0.9876 (full-route ~0.9885, D). Different blocking: P uses conjunction TF-IDF at 20.6/S1; O uses 45/S1 + routes + K9. Different models: P has CEs, O has generator/pool features. | **+0.0003 to +0.0010 LB (G)**; measure it | their export 15-30 min; our blend and eval ~1 h CPU |
| 2 | **Their e5i/e5b in-country CE scores as comb features** | +0.0041 / +0.0003 OOF on P (M); P's v08 generator features added only +0.00004 on top (M). That overlap runs the other way from ours, so the CE signal is probably not in O. | **+0.0003 to +0.0012 LB (G)**; overlaps with #1 | their export; our comb re-fit ~1.5 h |
| 3 | France: P's full France path | P France estimate ~0.971 (P's calibration). Implied from LB 0.984866: 0.9686-0.9710 (D). Their note says our France build had a lower emitted house-conflict rate (~1.0%) than theirs (1.42%) before their veto (M). | 0 to +0.0006 LB (G); only via LOCO, which we cannot run on O's full stack tonight | high |
| 4 | e5bc cross-country France CE scores (US and India models on the France band) | LOCO +0.0044 India->US, +0.0026 US->India on P's France stack (tsk-c26, M) | only useful inside a France stacker we do not have; skip | - |
| 5 | bge-reranker-v2-m3 and e5-large cross-country France CEs | Kaggle kernels launched 19:59 and 20:09 (M). **No results yet** (EXPERIMENTS.md ends "pending"). bge settings: batch 32, lr 2e-5, 150 min per GPU. | skip for us | - |

Already covered or rejected (no new value, M on P):
- The P France veto word list (developpement, distribution, france, groupe, holding, international, participations) is already in our 025 (pure words only).
- The utility-weighted stacker (-0.00012), count-prior features (0), house-shift veto (-0.0045 to -0.0005 LOCO), exact-EF decision (-0.0003), and in-country rule grids (0) do not need repeating.

Newest notes (cguide7 17:53, cguide8 18:22, cguide9 19:15, overview 20:38, EXPERIMENTS 20:51) hold no new measured gain beyond the summary. They are external review prompts (blocking-pilot validity, operator matcher, inference parity), and plans.
- cguide9 ranks "teammate/current-model complementarity" at "≥0.0005 LB only if disagreement is complementary". That matches item 1.
- Their plan note records the public LB at 19:10: **rank 25 = 0.987886, leader = 0.990748** (M). 0.99 is therefore top-3 territory.

## 2. Ensemble: what exists and the exact recipe

### 2.1 Their per-pair probability files (on their laptop, not on the share)
| File (under their `~/amlc/tsk/`) | Content | Rows | Keys |
|---|---|---|---|
| `cache/ic_q/<tag>_{India,US}.npz` (written by `stack_ce.py eval` with `SAVE_IC=<tag>`; `team_blend.py` defaults to tag `e1` = the v08 in-country stack) | **train in-country OOF**: `q` = final OOF p over ALL their candidate pairs (band pairs p3, others stage-2 p2), plus `p2`, `y`, `n_true`, `fold` (CRC32(id) % 5), `a`, `r` | ~20.6/S1: India ~18M, US ~27M (D) | row-aligned to `cache/pairs_train_<md5 of k_r3/k_a15/maxdf3000/min0.1>/{India,US}.parquet`, which holds `a_id` (S1 entity id) and `r_id` (pool entity id) |
| `output/final_p_{France,India,US}.npz` (the v08 build output dir) | **test** final p of the cascade-kept pairs (p2 > 0.02 and top 10). France includes the veto (p = 0). | ~4.2/S1, ~7.3M total (D from the 113.8 MB candidate file) | `a` = index into that country's test S1 ids in `load_norm` order; `r` = integer code, **not an id**. Needs the test pair table or re-keying. |
| `~/amlc/kaggle/out/scores_{e5i,e5b}_{0,1}_heldout.parquet` | train band CE logit per half-model; each pair scored by the model that did not train on it | train band (0.02 < p2 < 0.98), about half per file | `a_id`, `r_id`, `ce` |
| `~/amlc/kaggle/out/scores_{e5i,e5b}_{0,1}_test.parquet` | test band CE logits from each half-model (production uses the mean of the two) | 1,786,309 test band pairs, all 3 countries (M) | `a_id`, `r_id`, `country`, `ce` |
| `scores_e5s_{US,India}_*`, `scores_e5bc_*_france/heldout` | cross-country CEs | - | same keys |

- No France OOF exists; France can be checked only through their LOCO caches (`cache/loco_q/*`).
- Their OOF carries a self-reported optimism of ~0.001-0.004 (their leakage review, EXPERIMENTS.md l.184: selection on the same OOF, and nested-stacking contamination from CE halves trained on fold parity; M).
  - **Consequence:** a blend weight fitted on train will overweight P.
  - Cap w at 0.5, and prefer the weight that also wins on folds that were not used to choose it.

### 2.2 What is on the share now (M)
- `submission_7/v9_incountry_rich_stacker/matching_results.tsv`: v08, 96.9 MB, `source1_entity_id, matched_entity_ids`, sha256 `c1b0aaf2…`.
- the v08 package zip (its file name carries their team-name prefix): `output/candidate_pairs.tsv` (113.8 MB) plus code; no probabilities, no weights.
- These are usable only for a label-free agreement check against our test output (emitted-pair Jaccard per country, which bounds how much a blend can change). That check costs a 97 MB LAN copy.

### 2.3 What to ask them to export (exact)
Ask them to drop the files into a new folder on their side (for example the share's `handoff/`), all parquet, zstd:
1. `p_train_{India,US}.parquet`: `a_id, r_id, q, p2`, rows with `q >= 0.001` **or** cascade-kept (`p2 > 0.02` and top 10 by p2 within S1).
   - Source: `cache/ic_q/e1_{c}.npz` plus the pair table's `a_id, r_id` (same row order). `tm_verify.py` already streams the pair table this way.
   - About 8-12M rows, ~150-250 MB (G).
2. `p_test_{France,India,US}.parquet`: `a_id, r_id, p` for the v08 kept pairs, plus `p_preveto` for France if easy.
   - Simplest route: add `a_id = ids[A_]` and `r_id = RID_` to the `np.savez_compressed(... final_p ...)` line (stack_ce.py l.533) and rerun the 10-20 min final build.
   - Fallback with no rerun: `write_lists` (predict.py l.28) keeps pair order within each S1, and v08 has no cr pairs. So stable-sorting `final_p_{c}.npz` by `a` lines up row by row with the exploded `candidate_pairs.tsv` of that country; check that the row counts match.
   - About 7.3M rows, ~100-150 MB (G).
3. `scores_{e5i,e5b}_{0,1}_{heldout,test}.parquet` as they are, 8 files, ~300 MB total (G). Optionally `e5s`.

Transfer: SMB/LAN is fine at these sizes.

### 2.4 Blend recipe on our side (our rule, our folds, our gate)
- **Align.**
  - `s1_idx` = row index of `entity_id` in `cache/norm/{split}_s1.parquet`.
  - `pool_idx` = row index of `entity_id` in `cache/norm/{split}_pool.parquet` (the same indexing `er_block.py` uses).
  - Join `a_id -> s1_idx` and `r_id -> pool_idx`. Assert 100% resolve and that the country matches.
- **Candidate parity.** Their test file holds only cascade-kept pairs, so restrict their train OOF to the same cascade (p2 > 0.02 and top 10). They measured in-country OOF as identical with and without the cascade (M).
- **Union with our OOF.**
  - Our side is `cache/oof/<final attempt>.parquet` (`s1_idx, pool_idx, prob`) for train and `cache/oof/test_<attempt>.parquet` for test.
  - Caveat: our train OOF has route/K9 pairs only for the 10% sample owners, while test has them for every S1. Evaluate the blend on the full OOF and also on the sample-owner subset.
- **Variants.** Missing-pair handling is the crux:
  - (a) arithmetic `w·p_t + (1-w)·p_o`, with a missing side counted as 0;
  - (b) logit blend where both sides are present, and the present side's p when only one is;
  - (c) logit blend with a missing side clipped to 0.01, not 1e-6;
  - (d) max; (e) min;
  - w in {0.3, 0.4, 0.5}.
- **Decide** with our rule: one owner, EF prefix, floor 0.6. Use `er_eval --from-oof` on a blended OOF parquet written as an attempt (a `fit` in `attempts/026_blend/train.py`; no src change needed).
- **Gate.** Paired against our parent on `data/folds.csv`: mean > 2 SE and >= 4/5 folds, per country.
  - Choose the variant and weight on folds 0-2, then confirm on folds 3-4. This limits the P-optimism bias.
- **France.** No labels and no O LOCO, so keep **our 025 France unchanged** unless a both-direction LOCO blend exists. P's LOCO caches cannot be paired with ours.
- **Test.** Apply the chosen variant in-country, then `er_predict --from-probs`.
  - Run the guard: emitted pairs/S1 and house-conflict rate per country vs the parent.
  - Run the validator.
- **Packaging.** The final zip must regenerate the output "using only what is in this folder". It therefore has to contain P's code path (their zip `code/` tree plus the Kaggle CE scripts) and document the GPU steps. Budget 1-2 h for this tomorrow.

### 2.5 CE scores as comb features (alternative or complement to 2.4)
- Merge `ce_i = mean` of the e5i half-model scores and `ce_b` likewise.
  - Train: the heldout file already holds the single out-of-half score, so no averaging.
  - Test: average the two halves (P's production does the same; the train/test mismatch changes 0.13-0.14% of S1 decisions, M).
- Join on `(a_id, r_id)` to our band; missing -> NaN.
- Coverage: their band is `0.02 < p2 < 0.98` on their candidates, while ours is `0.003 <= p018 <= 0.997`. Overlap is partial; measure it first.
- Add pool-side CE context only: rank among the pool record's S1 and margin to the best other S1. Do not add P's within-S1 rank or gap, which is our trap family.
- Same paired gate as 2.4.

## 3. Their latest scores and current activity (M, share timestamps are IST)
- Uploaded: v01 0.970504, v03 0.975, v05 0.981, v06 0.983, **v07 0.984866** (logged 18:54, 26 Sep).
- **v08 built 19:52** (`submission_7/v9_incountry_rich_stacker`, validator PASS, not yet tagged). It is v07 plus the rich in-country stacker at +0.00009 OOF. Their plan uploads it about 23:50; expected ~0.98494 (D).
- Running at 20:09-21:15:
  - Kaggle `ce_france.py` with bge-reranker-v2-m3 (19:59) and with multilingual-e5-large (20:09). Watchers then run their France LOCO automatically.
  - AWS `specdk_full` (spec + decoy-aware stage 2, launched 20:48). Its stage-2 level so far: US->India +0.0019 in the blend (M).
  - `team_blend.py` / `tm_verify.py` for the two-pipeline blend (last commit 21:15).
- Tomorrow's plan (plan_0989_claude.md):
  - k_a 40 re-block: recovers 19.3% of India misses for +18.8 pairs/S1; ceiling ~+0.0008 LB (M/D).
  - Operator-explanation matcher v2.
  - Chain-branch features.
  - An France-empty LB probe (their reviewer advised dropping it).

## 4. Compliance evidence (for the human; ideas are fine either way)
- **Upload counts (D).**
  - Their `submissions/log.csv` shows scored uploads on 25 Sep (v01, v03) and 26 Sep (v05, v06, v07).
  - Ours (LATEST_SUBMISSION.md) shows 5 attempts on 25 Sep (4 scored plus 1 portal failure, which the portal appeared to count) and 3 on 26 Sep by 11:15.
  - As one team entry, that would be 6-7 uploads on 25 Sep and 6 on 26 Sep, above the 5/day cap the portal enforced on us.
  - So their uploads are not counted against our quota. They are a separate entry, or the cap is per uploader.
- **Package name (M).** Their v08 zip is `<team name>_submission.zip` with a team name different from the one in our documentation; the README requires `<team_name>_submission.zip`.
- **Their own docs (M).** AGENTS.md says "Four teammates, one laptop and one Portal". The file `ER_Approach_Reproduction_Guide_tm.md` on the share (10:32, 26 Sep) is our approach guide; its checkpoints A-F match our LB history. `team_blend.py` / `tm_verify.py` describe blending with "a teammate pipeline".
- **Risk.** If two portal entries exist, two near-identical ensembled packages among the top teams would draw the organisers' fair-play review. **Before any P output enters an upload, the human should confirm on the portal that one team lists both people and that only one designated person uploads.**
- No team or member names are written in this file.

## 5. Recommended order tonight and tomorrow
1. Now: ask them for the 3 exports in §2.3 and ask them not to run a separate blend upload. (Their plan uploads v08 at ~23:50.)
2. On GCP, beside the rebuild: build the §2.5 CE-feature comb and the §2.4 blend on train OOF. Each needs the paired gate; they overlap, so test the blend on top of the CE-feature comb as well.
3. Upload only a gated winner, with France = our 025.
4. Expected: blend and/or CE +0.0003 to +0.0012 LB on top of our ~0.9865-0.987 (G). Reaching 0.99 still needs about +0.002 more, and nothing measured on either side supports that.
