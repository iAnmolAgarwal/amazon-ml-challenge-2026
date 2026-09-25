# Live status / handoff (update at every milestone)

Last update: 2026-09-25 ~12:15 IST.
KEY FINDING (12:05): coherence (feat3) inflates decoy matches on TEST. House-number conflict among emitted pairs: US train OOF 7.4%; test sub-1 5.5%, sub-2 7.0%, sub-4 (004) 14.7%; France 1.5% -> 6.0%; India 15.9% -> 18.1%. Test has ~2x decoys per S1 (pool/S1 5.76 vs 4.68). Do NOT re-upload 004 (its upload failed on the portal due to server load; user may email support@unstop.com). Next submission = 005n_indic_nocoh (dictionary, no feat3), guard: test US conflict ~7%.
PIPELINE (12:20, 006 prioritised): A chain9 = 005n CV -> 005f --from-oof -> feat4 train -> 006 CV (verdict ~14:10). B bchain = after its test rebuild + train caches from A (TRAIN_CACHES_OK): predict 005f (stage-1 test probs) -> feat4 test -> after FEAT4_TRAIN_OK (Mac relays train_feat4 A->B) -> predict 006 (~15:00). If B dies: run feat4 test + predicts on A (test caches relayed to A by the Mac watcher). v3 (sub-7 candidate) armed: A chainA_v3 after EVAL006_DONE = v3 train rebuild (tag _v3) -> CV 007a_stage1 --tag _v3 -> feat4 v3 train -> CV 007_stage2 --tag _v3. B chainB_v3 after PREDICT006_DONE = v3 test rebuild -> (V3_TRAIN_OK from Mac relay) predict 007a -> feat4 v3 test -> (FEAT4V3_TRAIN_OK) predict 007. Mac relays: bpxmgo02k. Milestone watcher: scratchpad watch.sh (state watch.state).
SUPERSEDED (12:15): A chain7 = 005 CV -> 005n CV -> 005f (--from-oof 005n, FLOOR 0.6) -> wait for B test caches (~/B_TEST_RUNNING) or rebuild test -> predict 005f (sub-5, ~14:10). A chain8 = after PREDICT005F_DONE: er_feat4 train (src 005n OOF) / test (src test_005f) -> CV 006_stage2 -> predict 006 (sub-6, ~16:30). B (spot r7i us-east-1a) = test rebuild -> Mac watcher relays test norm/cand/feat/feat2/feat3 to A and touches ~/DICT_TEST_DONE; then run scratchpad dict/run_v3.sh on B (blocking v3 --rev-k 3 + k_sk,k_skh,k_fl, tag _v3) for sub-7. Floor auditor running -> docs/astra/09_floor_audit.md.
OLD: Running on A: 005_indic CV (dictionary + feat3, for measurement only) -> chain6: 005n CV (~13:15) -> test rebuild unless ~/DICT_TEST_DONE exists -> predict 005n --train-frac 1.0 (~PREDICT005N_DONE). Spot B (r7i.2xlarge us-east-1a) may run the test rebuild in parallel; if so relay test norm/cand/feat/feat2 to A and touch ~/DICT_TEST_DONE on A.
A parallel Codex pipeline exists (audit docs/astra/08_codex_pipeline_audit.md: CV India 0.9710 / US 0.9784; methodology doc must list the user too). Waiting: whether it has per-entity OOF scores (for a paired comparison / blend).

## Scores
| # | attempt | CV macro F0.5 | public LB | notes |
|---|---|---|---|---|
| sub-01 | 001_baseline (41 feats, train 20%) | 0.94292 +/- 0.00029 | 0.929 | commit 125b680; test norm without France fixes |
| (sub-4, pending upload) | 004_coherence_full (003 + feat3, train 100%) | gate 004b_coherence50 0.96351 (paired +0.00242 vs 003, 5/5) | - | submissions/03_004_coherence_full.* |
| (not submitted) | 003_trainfrac50 (002 trained on 50%) | 0.96109 +/- 0.00018 (paired +0.00059, SE 0.00004, 5/5) | - | predicting on A (~/predict_003.log, ~/PREDICT003_DONE) |
| sub-02 | 002_namefreq (+feat2: name freq, conservative names, gated addr) | 0.96050 +/- 0.00021 | 0.949 | France-gated test normalization; implied France ~0.907 |
Budget: 2/5 used on 25 Sep IST, 13/15 left. Log: submissions/log.csv, `python src/sub_status.py`.

## Running now
- EC2-A (on-demand r7a.2xlarge i-038c4d2ef3e919514, alias amlc-a): CV of 003_trainfrac50 (002 trained on 50%) with the NEW memory-lean harness. Log ~/eval_003.log, marker ~/EVAL003_DONE. The first 003 run was OOM-killed at 07:36 IST (old harness 42 GB + a concurrent job). RULE: only one heavy job on EC2-A at a time.
- EC2-B: train_feat3 DONE (97,811,967 rows, 329 s). After TEST_FEAT3_DONE, B auto-starts CV of 004_coherence_full (~/b_004.sh, log ~/eval_004.log, marker ~/EVAL004_DONE; B has cache/cand/train_s1.parquet copied from A, md5 e86b276a). If B is reclaimed or OOMs, run 004 CV on A after 003 predict. Relay of both feat3 files to A runs from the Mac after TEST_FEAT3_DONE.
- (history) EC2-B (spot r7a.2xlarge, id in scratchpad b-id, alias amlc-b) was building coherence features: `er_feat3.py --split train` (log ~/feat3_train.log, marker ~/TRAIN_FEAT3_DONE) then `--split test` (~/feat3_test.log, ~/TEST_FEAT3_DONE). When both markers exist: `scripts/s3_relay.sh amlc-b cache/feat/train_feat3.parquet amlc-a` and the same for test_feat3. If B is reclaimed: rerun the same two commands on A (A has all inputs) after A's current job.

## Next steps (approved by user)
1. 003 CV -> if significant paired gain over 002: `er_predict.py 003_trainfrac50 --train-frac 0.5` on A -> validate on A (`python3 validate_submission.py ... --test-dir valtest --check-ids`) -> copy TSV to Mac (checksum) -> `python src/er_package.py submissions/03_...tsv --skip-validate` -> cp to submissions/matching_results.tsv -> user uploads (sub-03).
2. 004 = 002 + feat3 (coherence) + TRAIN_FRAC 1.0 (+ optional per-country isotonic): create attempts/004_*, EXTRA_FEATS=["feat2","feat3"], EXCLUDE a_ratio/a_partial. CV on A (new harness, ~2 h at 100%), then predict with --train-frac 1.0 -> sub-04. Expected +0.004-0.005 CV over 002.
3. Then India word top-80 (P1), per-country isotonic (+0.00015).
4. Workers W1/W2 can be started from RUNBOOK once 004 is in flight.

## Key facts
- Gate 5 audit (docs/astra/05_*): coherence +0.0025-0.003 (sample, 3 seeds); learning curve 10%->20% +0.00071, 20%->40% +0.00088 (India +0.0014); calibration +0.0001-0.00015; Hungarian/ownership rejected; loss split All: retrieval .0102 / ownership .0046 / selection .0136 / FP .0112; singletons = 46% of FP damage.
- Harness: src/er_norm, er_block, er_feat, er_feat2 (side-car), er_feat3 (coherence side-car), er_model (to_matrix, full_dataset, fit_subset, predict_chunked, decide), er_eval (OOF CV, saves cache/oof/<id>.parquet), er_predict, er_package (code zip, no AI mentions, no notes), er_diag.
- User rules: every AWS change shown + approved (user pre-approved compute up to $100 total); no AI mentions in code zips; leaderboard never changes process; Astra 6 review at each gate + independent Opus audit; caffeinate until 26 Sep 07:45 IST.
- Quota requests pending: Standard on-demand 32 vCPU, Standard spot 32 vCPU.
