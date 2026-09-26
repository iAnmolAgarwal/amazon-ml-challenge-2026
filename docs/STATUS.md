# Live status / handoff (READ FIRST after any compaction)

## 26 Sep 16:40 IST: AWS ACCOUNT SUSPENDED (identity verification); documents uploaded by the human, awaiting AWS
- Symptoms from ~15:56 IST: amlc key InvalidClientTokenId, all machines unreachable. AWS notice: KYC documents required; resources subject to termination; closure 11 Oct if not verified.
- Scores: upload 10 (018) 0.98393; upload 11 (020_gen, file 12) 0.985. Uploads left 8 of 15 (26 Sep: 3 of 5 used).
- READY BUT STRANDED ON A: upload 12 = 021_comb (018 + generator features gf1-6 + base band features + pool-side competition features, 255 leaves, 800 rounds, boosted from logit p018, band 0.003-0.997): full CV 0.98758; paired vs 018 +0.00140 (5/5), vs 020_gen +0.00058 (5/5), re-verified with build/paired_full.py; guard clean (France identical); validator PASS; A:~/aml_stage/submissions/archive/13_021_comb.tsv (sha 802685c7...). submissions/matching_results.tsv on the Mac is still upload 11: do not re-upload it.
- AT RISK: r12 code (A:~/aml/research/r12/*.py: gf.py..gf6.py, gen_step.py, pf.py, pf2.py, comb) and r11 code (C:~/aml/research/r11/, copy of t_final.py on A:~/aml/research/r11x/) are not in git; 020/021 cannot be packaged without them.
- On recovery: scratchpad/recover.sh (read-only check, prints start commands for approval), then `recover.sh pull` (fix SSH IPs, pull r11/r12 code into research/, pull 13_021_comb) -> port gen features into src, package 021, commit; resume r12 + a blocking-miss lane (max 4 agents; 2 while usage is tight).
- Final-day plan (27 Sep, 5 uploads): upload 12 at the first slot; later uploads only for re-verified gains; choose the final package by ~20:00 IST from the best verified build whose code is in src (fallback 019_band = upload 10 code, fully in git); `er_package.py --final --team "Apprentice"`; docs/Documentation_template.md is filled (update the final LB row).
- If AWS is not back by 27 Sep ~12:00 IST: fallback final = 019_band (LB 0.98393, fully reproducible), or rewrite the generator features from r12.md on a teammate's account.

## 26 Sep 07:45 IST: upload 10 = 018 in submissions/ (direct paired vs 012: +0.00077, 5/5); code ported (caa614e); final-package form 019_band
- submissions/matching_results.tsv = 10_018_lexical (sha efdafee0..., validator PASS) + 10_018_lexical_code.zip. Upload 8 stays in archive.
- Re-verification of 018: route over 016 +0.00126 (sample, 5/5); lexical over 017 +0.00016 (full nested, 5/5); 018 vs 012 +0.00077 (full, 5/5). Full-graph CV 0.98618 (US 0.98695, India 0.98502; route pairs only on the 10% sample owners).
- src: er_lift.py (feat6l + feat5p, byte-identical to the staging outputs), er_stage3 INIT/USE_LOGIT/EXCLUDE3, er_predict CAND_BAND; attempts 013-019. The human gave standing approval for src edits (26 Sep).
- 019_band = 018 decided on the band (006 p >= 0.001 + route pairs): candidates 7.25/S1 instead of 46.49 (train 5.93 vs 44.48), CV equal (0.986181, 1 S1 differs), test matches identical (sha efdafee0...), validator PASS. Files: submissions/archive/11_019_band.* (+ code zip). Use 019 for the final package (--final).
- France lane (r3, done): no France-only change to ship on 018 (mass excess only +0.014; lexical stays on for France: LOCO +0.0037 mean). Lead: France gray zone = same-address name changes with French vocabulary; untestable without French labels.
- Running: r10 recall/retrieval (F, C), r11 final model/decision (C), r12 US/India breakthrough (A, D). Disk: Mac 19 GB free after approved cleanup; A ~9.6 GB free (A safe set NOT approved).

## 26 Sep 06:30 IST: 016_stage2b VERIFIED BEST CV 0.98589 (+0.00092 vs 009, SE 0.00002, 5/5; US +0.0008, India +0.0011)
- Upload 10 candidate = 10_018_lexical (016 + route 017 + lexical 018) built on A (staging); guard vs 015 clean (France conflict 1.08% -> 0.99%).
- Re-verification running (~/aml/build/chain_v10.sh, log chain_v10.log): validator; route gain over 016 on the sample owners (val017.py); lexical --cv on 017 + paired_full 017 vs 018.
- Needs the human's approval for src: er_stage3 INIT/USE_LOGIT/EXCLUDE3 options (staging patch; backup er_stage3.py.bak), lift + French parser step (build/lift_parser.py), attempts 013-018.

## 26 Sep 05:45 IST: upload-10 chain running on A (~/aml/build/chain_16.sh, log chain_16.log)
- 016_stage2b = full-integration via the band: staging er_stage3 with INIT=False, USE_LOGIT=False, FEATS3 feat+feat2+feat4+feat5p+feat6l (EXCLUDE3 a_ratio, a_partial), 127 leaves, 600 rounds, band p006 >= 0.001 (12.9M train pairs). Staging er_stage3.py patched (backup er_stage3.py.bak). Then 017_route (OLD 016) and 018_lexical (SRC 017) -> 10_018_lexical.tsv; paired_full vs 009 and vs 013; guard vs 015.
- If 016 beats 013 (paired, full train) and the guard is clean, upload 10 = 018 (needs src approval together with upload 9's lift/parser code).

## 26 Sep 05:55 IST: full-integration measurement (next build candidate)
- 10% sample, in-sample stage 2 retrained with BASE_FEATS + r5 32 + r9 (hd, hd_D, nA, nB, dm_*): 0.98502 vs pb 0.98373 (+0.01129, 5/5); vs the stage-3 route p59 (0.98457): +0.00045 (SE 0.00003, 5/5) despite 5x less training data. Expected full-data gain over the stage-3 approach ~+0.0005-0.001 (derived). Needs the join-free loader (scratchpad loader_draft.py) + stage-1/2 rebuild ~3-4 h. Script: C ~/aml/rk/integ.py, log ~/aml/research/v/integ.log.

## 26 Sep 05:30 IST: upload 9 READY (pending the human's approval of src changes)
- 09_015_lexical = upload 8 + France-only fixes: lifted decoy-marker scores (feat6l) + repaired French house parser (test_feat5p), stage 3 retrained (013), route (014) and lexical (015) rebuilt on top.
- In-country CV 013 vs 009: +0.00001 (SE 0.00001, 3/5): neutral, as intended. Guard 8 -> 9: France pairs/S1 3.232 -> 3.275, empty 5.95% -> 5.81%, conflict 1.06% -> 1.08%; US/India unchanged. Validator PASS; files in submissions/archive/09_015_lexical.* (tsv sha 0215707d...).
- To ship: port build/lift_parser.py logic (scratchpad lift_parser_used.py) into src (lift tag in er_feat6 + French fallback parser step), attempt folders 013/014/015 into the repo, package zip. Recommended order: upload 8 first (measured layers), then 9 (France-only, simulation-validated).

## 26 Sep 05:05 IST: optional upload 9 building (France lift + parser), awaiting the human's approval
- Chain on A: ~/aml/build/chain_lp.sh (log ~/aml/build/chain_lp.log), staging only (no repo src edits):
  build/lift_parser.py -> cache/feat/{train,test}_feat6l (dm_* = logit(score) - log(sum nd / sum n0) per country and kind; tables in use: train_us, train_india, test_france) and test_feat5p (France house columns with the repaired French house parser; train_feat5p = symlink to train_feat5);
  013_stage3lp (FEATS3 feat5p + feat6l) CV + paired_full vs 009 (expect ~neutral in-country);
  014_route (OLD 013, models symlinked from 010) train + test; 015_lexical (SRC 014) -> 09_015_lexical.tsv; guard vs 012.
- Decision rule for upload 9: in-country CV of 013 vs 009 within +-0.0001 (no harm), and France guard moving as the review predicted (pairs/S1 ~3.23 -> ~3.27, empty ~5.9% -> ~5.8%, decoy-shift emissions down). Then ask the human to approve adding lift_parser logic to src (er_feat6 lift tag + er_norm French fallback) and upload.
- Reminder: upload 8 (012_lexical) is the file in submissions/ now; the user wakes ~06:00 to upload it.

## 26 Sep 05:00 IST: MORNING HANDOFF (uploads 7 and 8 built overnight)
- **Recommended upload: 08_012_lexical** = 009 + char-3-gram retrieval route (010) + lexical correction (012).
  File: submissions/matching_results.tsv (md5 6da271f4...), code zip submissions/08_012_lexical_code.zip; validator PASS on A (--check-ids).
- Measured gains (paired, 5/5 folds):
  - route vs 009: +0.00128 (10% sample, 220,678 S1; SE 0.00005);
  - lexical vs 010: +0.00031 (full 2.2M train S1, nested cross-fitting; SE 0.00001).
  Expected LB ~0.983-0.985 (derived from 009's 0.981).
- Fallback: 07_010_route (archive; validator PASS).
- Guard: US/India unchanged; France pairs per S1 3.215 (009) -> 3.237 (010) -> 3.232 (012), empty rate 6.0% -> 5.9%, house-conflict share unchanged.
- Codex review (docs/research/codex_leads.md): nothing big left in the parallel line (its neural France pilot has no result). New lead: 'lifted' decoy-marker scores (subtract each country's base rate; France's test-derived table sits on a shifted scale) + a French house-number parser repair; LOCO +0.0086 (5/5, agent-measured, not yet re-verified), estimated +0.001-0.002 LB; needs a stage-3 retrain + rebuild of route/lexical on top (~2 h) and the human's approval.
- Machines: A (on-demand) has everything; C (on-demand, idle); D (on-demand; France-lane jobs frozen); F (spot, idle); E and G were reclaimed (spot). Codex session's chain on A is still SIGSTOPped (resume: ssh amlc-a 'kill -CONT $(cat ~/codex_paused_pids.txt)').
- Agents: France lane paused; Codex-review done.
- Uploads: 26 Sep 1 of 5 used; 10 of 15 left overall.

## NIGHT PLAN 26 Sep 01:25 -> 06:00 IST (user asleep; wakes ~06:00 to upload)
- Mac caffeinated 12 h. Orchestrator scratchpad/orch010.sh runs unattended (logs: scratchpad/orch010.log, /tmp/orch010_train.log).
- Deliver by 06:00: upload 7 = 010_route (009 + retrieval route) as submissions/matching_results.tsv + code zip, validated (organiser validator on A), guard checked, plus the train-sample paired CV of the route (build/val010.py).
  - Integrate er_retrieve.py, er_newpairs.py, attempt 010_route into the repo src/ (approved: merged build) and er_package SRC_FILES; commit, push.
  - France mass calibration (er_masscal.py, attempt 011_route_frcal): NOT in upload 7 (009 already calibrated France: mass 3.33 vs target 3.29; the shift changes ~5k pairs).
- Upload 8 in progress: lexical layer (Codex recipe: 6 token channels x 7 summaries = 42 cross-fitted likelihood features + residual LightGBM 100 rounds / 9 leaves / lr .04 on p006>=0.01 pairs); auditor measured +0.00032 over our build on the 10% sample.
- France agent PAUSED (user); its D jobs SIGSTOPped. Codex-review agent on G running -> report docs/research/codex_leads.md; summarise for the user in the morning.
- Morning report: upload-7 file + measured route gain + guard; Codex-review findings; upload-8 status.

## 26 Sep ~01:55 IST: submission 6 = 009_stage3m, public LB 0.981 (008: 0.974). Upload 7 building.
- 009 = 008 + label-free decoy-marker name features (src/er_feat6.py; France builds its own table from test pairs). Full CV 0.98498 (+0.00244 vs 008). Implied France 0.929 -> 0.960. Commit f192703; LATEST_SUBMISSION.md (= Desktop handoff) updated, pushed cc3fbe2.
- **Upload 7 plan = 009 + retrieval route (Codex recipe, our harness) + France mass calibration.** Upload 8 = + lexical layer (+0.0003 CV) + confirmed agent findings.
- Upload-7 machinery (all staged in ~/aml_stage on A; proposed src in scratchpad src_proposed/, not yet in the repo):
  - src/er_retrieve.py: char-3-gram TF-IDF top-40 per view (name_core, addr_norm), pool in 125k blocks. Parity vs Codex cohort-6 edges: similarities equal, differences = ties only.
  - Test retrieval running: C (india addr shard 0/2, us addr 0/2, name india), E (india addr 1/2, us addr 1/2, name us, france addr+name). Done markers ~/aml/build_retC.done / build_retE.done. ETA ~03:30 IST.
  - F: train-sample retrieval (s10 S1) for our own CV of the route; name view re-queued after an output-dir bug; marker build_retF.done.
  - src/er_newpairs.py (union / fit / score): new pairs = retrieval minus existing candidates; 66-feature LightGBM (63 leaves, 300 rounds) trained on existing train pairs of the 10% owner sample (fold models + full model DONE on A: cache/models/010_route/); keep p > 0.001; append to 009 probabilities. Attempt 010_route (OLD=009_stage3m).
  - Mac orchestrator scratchpad/orch010.sh (log orch010.log, /tmp/orch010_train.log) waits for the markers, relays edges to A, runs union, er_feat/er_feat2/er_feat5 with --tag _new (test) and _s10new (train), scores, runs build/val010.py (paired 010 vs 009 on sample owners) and er_predict 010 + test guard.
  - src/er_masscal.py (written, not yet deployed): France-only logit shift delta >= 0 so mean sum-of-p per France S1 = train true matches per S1 in candidates (3.394) - 0.1. Validated in LOCO by the France agent (r3.md): +0.0094 (5/5), +0.024 when over-claiming, ~0 otherwise; 10% confirmation pending. Attempt 011 would = masscal over 010.
- Agents: France lane r3 resumed on D (confirming masscal on 10%, testing in-country masscal under doubled decoys). Codex-review + breakthrough agent on G (report docs/research/codex_leads.md). All other lanes paused (user: resume only when told).
- Codex session's chain on A is SIGSTOPped at the user's OK: resume with `ssh amlc-a 'kill -CONT $(cat ~/codex_paused_pids.txt)'`.
- Machines: A us-east-1 (on-demand), C us-west-2 (on-demand), D us-east-2 (on-demand), E us-west-2 (spot), F us-east-2 (spot), G us-east-1 (spot). All tagged; auto-shutdown 27 Sep 18:30 UTC.
- Uploads: 10 of 15 left; 26 Sep: 1 of 5 used.


## 25 Sep ~23:05 IST: submission 5 = 008_stage3, public LB 0.974 (006: 0.963)
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
