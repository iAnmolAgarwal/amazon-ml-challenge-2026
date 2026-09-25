# Live status / handoff (READ FIRST after any compaction)

## 25 Sep ~23:00 IST: submission 5 = 008_stage3 (uploaded by the human; score pending)
- 008 = 006 + stage-3 correction (src/er_stage3.py) boosted from logit(p006) with 34 label-free pair features (src/er_feat5.py). Full CV **0.98254** (+0.00831 vs 006, SE 0.00002, 5/5; US +0.0084, India +0.0082); pair P 0.9966 R 0.9548; test guard clean (house conflict US 8.4% / India 17.0% / France 1.0%). Expected LB ~0.970-0.974. Commit 5cea04d.
- Upload file: submissions/matching_results.tsv (md5 c674f878...). Archive: submissions/archive/05_008_stage3.* (validation + code zip being produced).
- Next-build evidence (10% sample, same protocol): + label-free decoy markers and signed house delta (r9) = +0.01014 vs 006 (r9 add-on +0.0029, 5/5); gains grow under doubled decoys (+0.0125, r7).
- A cohort experiment from the parallel line claims 0.985372 on s1_idx%128==6 (006 0.974773 there; our 008 0.982474 there) with a label-derived lexical residual + extra char-3-gram retrieval (~56 extra pairs/S1). An independent audit is running (report: docs/research/audit_0985.md).
- Research agents r1-r9: PAUSED by the human (resume only when told); their jobs frozen (SIGSTOP) on A/D.
- Compute: A (us-east-1, r7a.2xlarge, shared with the human's other session), C (us-west-2 on-demand r7a.2xlarge, 64 GB, ~/aml with sample + norm + some caches), D (us-east-2 on-demand r7a.xlarge, 32 GB). Guardrail now allows us-east-1/us-west-2/us-east-2 and r7a/m7a.xlarge (human-approved 25 Sep).
- Rules update: candidate_pairs.tsv is reviewed for final ranking; smaller candidate sets per S1 rank higher.


## 25 Sep ~18:30 IST: RESEARCH PHASE, paused by the usage limit until ~21:30 IST
- 9 research lanes ran on a shared 10% train sample. Reports: docs/research/r1..r9.md; each has a "Pending when resumed" section listing jobs and logs on A (~/aml/research/rN/).
- Kit: ~/aml/rk/ on A (rk.py = exact 006 decision on the sample, 0 mismatches; run.sh = memory-capped queue; brief in the session scratchpad brief.md).
- **Converged finding (r1, r7, r8 and r9 independently):**
  - Decoys are near-copies of real entities:
    - house number shifted by d in {1,2,3,4,5,7,9,11,13,21} (test has 1.94x these);
    - an inserted word from a fixed list (group/holdings/ventures/downtown/lakeside...; France: participations/holding/groupe/et fils);
    - a legal suffix swap.
  - True-match noise instead truncates digits, adds filler words (center/services) and uses aka/fka/dba.
  - Test decoys come in clusters sharing the perturbed house number (train: none). Never use sibling/support/coherence features; judge each pair against the S1 alone.
- **Best measured lead (r5):** stage-3 correction boosted from logit(p006) with 32 pair features (name/address IDF masses, house relations, legal forms, char-3-gram). Quick subset: **+0.0066** (5/5, SE 0.0003; India +0.0064, US +0.0067). The full-sample check was lost when spot B was reclaimed.
- **Other results:**
  - r2 (house relation / house agreement / source balance): +0.0069 on the quick subset. House agreement and source balance are sibling features, so they risk the decoy trap.
  - r4: the decision rule is already optimal, and the context stacker emits decoys on test.
  - r8: retrieval is worth +0.001; India native words are worth 0.
  - r4 side finding: name-frequency counts are raw per split; US test has half the S1, so normalize per country size.
- **UNATTENDED CHAIN RUNNING ON A** (started 12:56 UTC): `~/aml/build/chain.sh`, log build/chain.log.
  - It runs feat5 (build/er_feat5.py = proposed src/er_feat5.py) for train us/india and test india/us/france, then merges, then build/stage3.py.
  - stage3.py is 5-fold CV of arms ctrl / r5 / all (+hd, hd_shift) on the full train (98M pairs, band p006 >= 0.001).
  - Outputs:
    - build/stage3.json: CV, paired vs 006, test guard;
    - cache/oof/008s_<arm>.parquet and test_008s_<arm>.parquet;
    - build/matching_results_008s_<arm>.tsv: a candidate upload, NOT yet validated or packaged.
  - Proposed harness files are in the scratchpad src_proposed/ (er_feat5.py, stage3.py). src/ is untouched: it needs the human's approval before integration.
- Spot B was reclaimed twice today (r6a 1d, m7a 1b); do not rely on spot. Only A exists (on-demand).
- **At resume:**
  1. Read build/stage3.json and chain.log.
  2. If the r5/all arm passes vs 006 and the guard holds, ask the human to approve integrating er_feat5 + stage 3 into src + attempt 008.
  3. Validate and package; that is the upload candidate (2 uploads left on 25 Sep until 23:59 IST).
  4. Then read the lanes' pending results for the full-integration build tomorrow.


Last update: 2026-09-25 ~16:55 IST. **ALL COMPUTE STOPPED** at the user's request: A is idle, no spot instance exists, and no Mac watchers are running.
Next phase (user's decision): a deep research phase. 10 independent parallel agents (Opus 5.5, max effort) explore every configuration for a >0.98 breakthrough. Nothing gets built or run at scale until there is evidence. See docs/research_brief.md.

## Scores (public LB = the portal; CV = 5-fold OOF macro F0.5, folds data/folds.parquet)
| sub | attempt | CV | test-weighted CV (US .38 / India .47) | public LB |
|---|---|---|---|---|
| 1 | 001_baseline | 0.94292 | 0.93714 | 0.929 |
| 2 | 002_namefreq | 0.96050 | 0.95640 | 0.949 |
| (upload failed on portal) | 004_coherence_full | 004b gate 0.96351 | 0.95947 | none; coherence later found to emit decoys on test |
| 4 | **006_stage2** | **0.97423** | 0.97368 (India 0.9721, US 0.9757) | **0.963** |

- LB−CV gap is steady at about −0.011; Spearman CV vs LB +1.0.
- Implied France (if US/India transfer): **0.902**.
- A parallel pipeline built separately (not in this repo; `~/Downloads/OVERVIEW.md`) reports OOF 0.98082 (India 0.97801, US 0.98269) → **LB 0.971**, implied France 0.919.
- Budget: 25 Sep used 3 of 5 (2 left today); 12 of 15 overall. Log: submissions/archive/log.csv; `python src/sub_status.py`.
- `submissions/` top level holds only the current upload (`matching_results.tsv` = 006, `04_006_stage2_code.zip`). Everything else is in `submissions/archive/`.

## Attempt lineage (all in attempts/, each with proposal.md + results.json)
003 (50% training) → 005n (native-script dictionary, consistent train/test normalization, no coherence; +0.00899) → 005f (floor 0.6 via --from-oof; +0.00076) → 006 (stage-2 competition features er_feat4 from 005n OOF; +0.00340).
- Rejected: 004/004b coherence (test decoy shift), 006r ratio rule (−0.00015).
- Defined, not run: 007a_stage1 / 007_stage2 (blocking v3 tag _v3).

## Harness (src/), key facts
- **er_dict:** native-script→Latin token maps from gold (1,347 name tokens → 147 words, positional alignment; 17 state tokens; 4 variant classes). Fold-stable; the build is byte-reproducible (md5 8d834aeb…).
- **er_norm:** multi-view normalization. latinize() runs first; name_lat is kept for feat2/feat3. Also name_skel and name_fl views (consonant skeleton; first|last token).
- **er_block:**
  - word TF-IDF top-40 + 6 exact keys;
  - v3 options: `--rev-k 3` (reverse retrieval, flag k_rev) and `--keys` incl. k_sk, k_skh, k_fl. The default reproduces the old candidates.
- **er_feat** (base), **er_feat2** (name frequency, conservative names), **er_feat3** (coherence: DO NOT USE, it emits decoys on test), **er_feat4** (stage-2 competition features), **er_feat5** (DRAFT, toy-checked, never run: name-specificity IDF masses, name/address cosines, skeleton chain size).
- **er_model.decide(pairs, cfg, floor, empty_ratio):** attempt-level FLOOR / EMPTY_RATIO.
- **to_matrix** consumes a one-element list, so the frame is freed.
- **er_eval:** `--from-oof ATTEMPT` re-decides without training. **er_predict:** `--from-probs ATTEMPT` re-decides saved test probabilities; it also writes a run manifest JSON and cache/oof/test_<attempt>.parquet.
- **er_package:**
  - strict validation (`--check-ids`, fails on any WARNING);
  - `--final --team` builds a safe final zip;
  - attempt DEPENDS/PRE_STEPS go into the README.
- **Memory warning:** stage-2 CV at 98M pairs × 60 features peaked 61.7 of 64 GB. Each extra side-car join copies the frame (~+25 GB transient). Fix the loader (gather columns by key index instead of joining) before adding feature sets.

## Machines / caches (EC2-A = amlc-a, on-demand r7a.2xlarge i-038c4d2ef3e919514, us-east-1d)
- Spot B was reclaimed 8 times on 25 Sep; don't rely on spot. IAM allows g5.xlarge/2xlarge, t3.medium, r7a/m7a.2xlarge, r6a/r7i.2xlarge. Quota increase to 32 vCPU is pending.
- **A ~/aml/cache, dictionary versions:**
  - norm test_*; train_* (rewritten 09:37 UTC with extra skeleton columns only);
  - cand train_cand / test_cand;
  - feat train/test _feat, _feat2, _feat4 (from 005n OOF / 005f test probs);
  - oof 002…006 + test_005f_floor, test_006_stage2.
  - Old 004 caches are in ~/aml/cache_v004. v3 blocking was stopped midway (no _v3 files).
- **Mac:** cache/raw (all raw parquet), cache/oof/002_namefreq.parquet, cache/norm/train_s1.parquet, cache/dict/indic.json.

## Evidence base (docs/astra/)
- **06_audit** (India dictionary; decoy stats).
- **07_audit** (compliance fixes, applied).
- **08** (parallel pipeline audit).
- **09_floor_audit** (floor 0.6; 100% singletons not worth it).
- **10_next_iteration_prompt + 10_audit_codex + 10_audit_opus:**
  - 0.99 unreachable; realistic 0.965–0.972;
  - loss buckets: 0.0164 of 0.0224 is missed matches, 86% of under-predicting entities miss exactly one;
  - French parser bugs: 118k pool house numbers dropped, N°→north, et, EI, department vs region;
  - house-number relation feature; f/k/a–a/k/a–dba regex gaps; stage-1 40% vs test 100% mismatch; stage-2 rival-probability leak.
- **11_parallel_overview_audit + 11_codex_overview_review:** measured on our OOF, US: +0.0028 (word rarity) / +0.0049 (+ legal form, skeleton) / +0.0061 (+ separate name/address cosines).
- **Label-free test guard (memory: test-shift-guard):** house-conflict share of emitted pairs, test vs train OOF. Sub-2 US 7.0%, 006 US 7.8%, failed coherence 14.7%. Newer guard: nearby-house-number share (US ≤0.8%, India ≤1.2%).

## Rules (user)
- Show and approve every AWS change; the user pre-approved compute under $100 total.
- One heavy job per machine.
- No AI mentions, team names or member names in any artifact.
- Only the human submits.
- Paired CV rule plus the label-free test guard before any upload.
- Ideas from the parallel pipeline are adapted into this harness, never rewritten from it.
