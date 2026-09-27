# Final-submission verification (independent verifier)

Scope: rebuild of the upload-10 chain on g1/g2 and the rewritten modules (023 comb, 024 keys, 025 veto, cascade, 026 blend), test guards, final package.
Tags: **M** = measured by me on data, **D** = derived (arithmetic on measured numbers), **G** = guessed / not confirmed. Times IST. Entries are appended; the latest verdict per stage wins.

## Stage verdict table (updated as stages land)

| Stage | Verdict | Evidence (short) |
|---|---|---|
| Code identity (core src vs code that built 019/020) | PASS | see 26 Sep 23:15 |
| Folds | PASS | identical to original (hash) |
| dict / norm / block (train+test) | PASS | 98,021,508 train / 78,924,662 test pairs; dict md5 = original; oracle per S1 identical |
| 005n stage-1 OOF | PASS | 0.97005 vs 0.97008; oracle identical per S1 |
| 005f / 006 stage 2 | PASS | 0.97425 vs 0.97423; oracle identical |
| 016 band | running (paused 02:45) | fold train pairs 10.36M = 4/5 of 12.9M |
| 017 route / 018 lexical | pending | |
| 019 vs original upload-10 file | pending | |
| 023 comb (rewritten) | pending (code review below) | |
| 024 keys | pending | |
| 025 veto | preview PASS (safe, near no-op) | 6 of 845,431 France pairs on the 020 output, all decoys |
| cascade | pending | |
| r18 meta (027) | GO with caveats | no leak; +0.00295 CV but LB about +0.0007-0.0012 over 024; package needs B code |
| 026 blend | open issues | parity fixed by lead 00:34; route/K9 kill not visible in CV; package needs the other pipeline code |
| Final package (029b_final) | GO (file); zip GO after doc fixes | byte-identical rerun of er_meta/er_france/er_predict on g2; validator PASS; guard PASS |

---

## 26 Sep 23:15 IST: baseline checks before the first stage lands

**Code identity (PASS, M).** The manifests of the original uploads (`submissions/archive/11_019_band.json`, `12_020_gen.json`) carry sha256 of every src file. Against HEAD: all modules of the 018/019 chain are byte-identical (er_block, er_dict, er_norm, er_feat, er_feat2, er_feat4, er_feat5, er_feat6, er_lift, er_stage3, er_lexical, er_model, er_eval, er_retrieve, common). Changed since: er_newpairs (VIEWS / EXCLUDE_OLD; default path unchanged for 017_route, which sets neither), er_predict (VETO, CAND_MINP/CAND_TOPK; no-ops unless the attempt sets them), er_package. g1 runs exactly HEAD for every file (md5 compare).
- Note 1: the Mac working tree has an **uncommitted** change to `src/er_retrieve.py` (a new `text` view). g1/g2 run the committed version. If the final package is zipped from the Mac working tree, the packaged er_retrieve.py will differ from the one that produced the file (harmless option, but the package should be built from the committed tree or the change committed first).
- Note 2: g2 has an older `er_newpairs.py` (before VIEWS/EXCLUDE_OLD). g2 only runs `union 017_route`, where both versions behave identically (017_route sets neither VIEWS nor EXCLUDE_OLD). If 024's union ever runs on g2, it must be updated first.
- Note 3: g2 also lacks er_keys.py / er_blend.py / post chains (not needed there).

**Folds (PASS, M).** g1 `data/folds.parquet` (built 17:09 UTC by er_eval) vs the fold column of the original `attempts/005n_indic_nocoh/entity_scores.parquet`: same 2,206,821 S1, md5 of the fold vector in s1_idx order identical (c6191f0e...), fold sizes 441,365 / 441,364 x4.

**Candidates (PASS on counts, M).** Train blocking on g1: 98,021,508 pairs (india 40,623,683 = 46.0/S1, us 57,397,825 = 43.4/S1) = n_train_pairs 98,021,508 in the original 006 manifest (`04_006_stage2.json`). Stage-1 log: 48 features, train_frac 0.5 (as the original 005n).

**README / PRE_STEPS defect (found, not blocking the build).** `attempts/018_lexical` and `019_band` PRE_STEPS say `python src/er_predict.py 006_stage2` (default TRAIN_FRAC 0.5). The original upload used `--train-frac 1.0` (`04_006_stage2.json`: train_frac 1.0), and rebuild_chain.sh uses 1.0 (correct). Fix for the lead: in both PRE_STEPS lists change the line to `python src/er_predict.py 006_stage2 --train-frac 1.0`. 023/024 inherit it via "all steps of 018_lexical".

## 26 Sep 23:20 IST: code review of the rewritten modules (static; numbers follow when stages land)

**Test-side inputs (PASS on counts, M).** g2: test_cand 78,924,662 pairs (France 13,129,681 = 50.6/S1, India 37,357,673 = 46.1, US 28,437,308 = 42.9) = 45.55/S1; plus the original route's 0.94/S1 gives the documented 46.49/S1 (D). `cache/dict/indic.json` md5 8d834aeb... on g1 and g2 = the documented byte-reproducible dictionary. test_s1 1,732,544 rows, test_pool 9,969,589.

**er_gen.py (no leakage found).** Every feature is pair-local (raw + normalized S1 record vs raw + normalized pool record); no label, no probability, nothing about other candidates. Row alignment raw vs norm is asserted (`entity_id` equality). Band pairs picked from the parent's OOF p (train) / test p (test), same 0.003-0.997 window on both sides. Minor, not blocking: `h_delta`/`s_delta` are raw integer differences cast to float32; numbers with > 38 digits become inf (LightGBM accepts inf; harmless).

**er_pool.py / er_comb.py (no leakage beyond the accepted second-order path).**
- Pool side only: pr_* are computed over the S1s that claim the same pool record (`over("pool_idx")`); pc_* contrasts likewise. Nothing compares a pair with its S1's other candidates. France is excluded from 023 (keeps p018).
- pr_* on train use the parent's OOF p of other S1s; those probabilities come from fold models that saw this S1's labels (the same second-order path accepted for feat4 and r11). Accepted, not new.
- CV: fold models train on `fold != f`, predict `fold == f`; model B (no base features, for route pairs) likewise. Test uses full-train models, feature-name order asserted equal.
- Deviation from the recorded original (G on impact): `attempts/021_comb` records `COMB_PARAMS = lr 0.05, 255 leaves, seed 42` (LightGBM defaults otherwise: min_data 20, no bagging, L2 0), while 023 uses min_data 300, feature_fraction 0.9, bagging 0.8, L2 10 for model A and adds a model B. If 023 lands below the +0.0011 alarm, this is the first thing to try (021's param set).

**er_keys.py / er_rtext.py / er_newpairs VIEWS (code equivalent to the measured r10 code per r10 section 6; the er_rtext patch is in HEAD).** But:
- **024 has no train-side check in any chain (UNVERIFIED).** post_chain.sh runs only the test side and reuses the 017 route models. The +0.00123 (owner sample, vs 018) is the original number with parent 018; with parent 023 the gain can be smaller (r10 section 6). Without the train run (`er_keys --split train --queries <owners> --tag _s10`, `er_rtext --split train --parent 023_comb --queries <owners> --tag _s10`, `union 024_keys --split train`, feat/feat2/feat5 `--tag _s10k`, `score 024_keys --split train`, paired 023 vs 024 on all S1 and on owners), 024's gain is unverified in this rebuild; only the test guard can be checked.
- Test reference for the guard (r10 section 5, parent 018): K9 kept 1,511,429 test pairs; new emitted pairs/S1 France +0.026, India +0.019, US +0.002 (train sample India +0.020, US +0.005).

**er_predict VETO (025).** Logic correct: France S1 only (raw `country` lower-cased = "france"; test has 259,452 France S1), p set to 0 when the pool name has the word and the S1 name does not; s1_idx/pool_idx = raw row order = norm order (er_norm keeps order; pool = source2 then source3). Vetoed pairs stay in the candidate set (allowed). Not CV-able (no French labels); only label-free checks apply (counts, house-shift share of vetoed emitted pairs).

**er_blend.py (026): three problems, one blocking for the package.**
1. **Candidate parity (bias, fix recommended).** Their test files hold only their cascade (max 10 pairs/S1; M: p_test max 10 per S1), their train files do not (M: p_train_US 5.21/S1, max 43; India 4.81/S1, max 167; 22.9% / 17.6% of train pairs have p2 <= 0.02). On test every pair outside their cascade gets FILL_EXT 0.0005, on train it gets their real q. The grid is therefore measured on a different blend than the one applied to test. Fix: in `theirs()` for split train, keep only `p2 > 0.02` and the top 10 per a_id by p2 before the join (as `audit_teammate_dir.md` section 2.4 prescribes).
2. **Route / K9 pairs are killed on test but barely present in train CV.** Our train OOF has route pairs only for the 10% owner sample and no K9/rtext pairs at all (OURS = 023 on train, OURS_TEST = 024 on test). On test, any of our pairs that they do not list gets logit 0.5*logit(po) + 0.5*logit(0.0005): po = 0.95 gives p = 0.06, i.e. dropped. So on test the blend removes most route and K9 emissions unless their candidates contain them, and the full-train grid sees about 1/10 of that effect for route pairs and none for K9. Required before any 026 decision: (a) the grid restricted to the owner sample (S1 with `sample_mask(s1_idx, 0.1)`), (b) on test, count 024 emitted pairs that 026 drops, split by source (base / route / K9-rtext) and country.
3. **Package blocker.** The organisers require the zip to regenerate the output from its own contents. 026 reads `cache/ext/p_*.parquet` (another pipeline's probabilities); that pipeline's code is not in SRC_FILES, and er_blend.py's docstring names a second pipeline. 026 cannot be the final package unless that code (and its GPU steps) is packaged and documented.

**er_package / README (fix before --final).**
- `docs/Documentation_template.md` still has `[update with the final upload]` (section 5) and `[final]` (appendix B). er_package's PLACEHOLDERS list does not contain either, so `--final` will not catch them. Fill them, and update the CV numbers (0.98758 etc. are the original 021's) to the rebuilt final attempt.
- For 023/024/025 the README is incomplete: `pre` is only the attempt's own PRE_STEPS, so 023 prints "# all steps of 018_lexical, then:" without the 018 steps; 025 and 026 have no PRE_STEPS at all; 023/024/025/026 have no EVAL, so the README's default line `python src/er_eval.py <attempt>` would train a plain stage-1 model instead of giving the CV. Fix: give each candidate final attempt a full PRE_STEPS (018's list + its own steps) and `EVAL = "python src/er_eval.py <id> --from-oof <id>"` (023) or a comment (024/025).
- 018/019 PRE_STEPS: `er_predict.py 006_stage2` must be `--train-frac 1.0` (see 23:15).
- final_checks compares src hashes with the predict-time manifest (all written on g1). Build the zip on g1, or make the Mac tree identical first: the Mac's uncommitted er_retrieve.py would make --final fail with "outputs or source differ".
- Names: code, config and attempts are clean (grep for AI/tool/person names; M). er_blend.py's docstring mentions "a second, independent pipeline" (only shipped if 026 is chosen; SRC_FILES always ships er_blend.py, so the phrase ships in any case; reword it if the rule forbids it). Documentation_template.md carries the team name and member names in the header fields the organisers' template asks for (human's call).

## 26 Sep 23:25 IST: expected numbers for this rebuild (read before judging 018/023/026)
- **The rebuild has no route pairs on train.** rebuild_chain.sh links `cache/oof/017_route.parquet -> 016_stage2b.parquet` and never runs `er_retrieve/er_newpairs` on the train owner sample. The original 018 CV (0.98618) contained route pairs for the 10% owners, worth +0.00126 on them, about +0.00013 on all S1 (D). Expected rebuilt 018 CV about 0.98605 (D), and verify's `paired 016 vs 018` should show the lexical gain alone, about +0.00016 (the original full nested number), not +0.0003.
- For the same reason, 023's model B (pairs without base features) never gets a validation fold on train (`has` is all true); on test it scores the route pairs. That matches how 021 was used, but its behaviour on route pairs has no CV evidence in this rebuild.
- For 026 this makes blend problem 2 worse: the train grid contains **no** route or K9 pairs at all, so the loss from the blend dropping our one-sided route/K9 emissions on test is invisible in CV. The test-side count of dropped emissions (by source) is the only evidence.

## 26 Sep 23:30 IST: tools checked, 025 veto previewed
- My TSV guard (`guard_tsv.py` in my scratchpad: pairs/S1, empty rate, house-conflict share from the matching TSV + norm house columns, so it covers veto and band, which `verify.py guard` does not) reproduces the reference exactly on the original 11_019_band.tsv: US 3.3624 / 0.0579 / 0.0863, India 3.3268 / 0.0585 / 0.1718, France 3.2585 / 0.0588 / 0.0100 (M). Original 019 vs 020: 1.26% of S1 differ (US/India ~1.5%, France 0); that is the scale of a real +0.0008 model change, for judging the rebuilt 019 against the original file.
- **025 veto preview (M, on the original 020 output; France is unchanged from 018 through 023):** the four words hit only **6 of 845,431** emitted France pairs (participations 2, holding 4, distribution 0, international 0). All 6 are clear decoys by eye (same number, different street: "43 Avenue Jean Mermoz" vs "43 AVENUE VICTOR HUGO"; "Enduro Ecole SAS" vs "Enduro Holding SAS"). Each S1 had only that match. Effect: right sign, about +0.000003 LB (D). Safe, and practically a no-op; not worth an upload of its own.

## 26 Sep 23:45 IST: 005n stage-1 OOF: PASS
- Rebuilt CV **0.97005** vs original 0.97008 (diff -0.000027, per-fold paired SE 0.000028, i.e. noise) (M). India 0.96735 vs 0.96737, US 0.97185 vs 0.97188; pair P 0.99175 vs 0.99178, R 0.93239 vs 0.93239.
- Per-entity comparison with the original `entity_scores.parquet` (M): all 2,206,821 S1 have the **same true count and the same blocking oracle** (100.00%), i.e. the rebuilt candidate set reaches exactly the same gold; 98.47% of S1 have an identical F0.5 (the rest is LightGBM run-to-run variation on the 50% training sample).

## 27 Sep 00:32 IST: BLOCKER (RESOLVED 00:33), g1 main chain crashed at s1_test (diagnosed; lead notified)
- `logs/rebuild/s1_test.log`: training done (98,021,508 pairs, 7,490,646 positives = the original 006 manifest's n_pos, M), then `FileNotFoundError: /home/amlc/aml/submissions/archive`. er_predict does `mkdir(exist_ok=True)` without `parents=True`, and g1 has no `submissions/` folder. The chain (set -e) stopped at 00:31 IST; post_chain / post2_chain are still waiting.
- `cache/oof/test_005f_floor.parquet` was written before the crash (571 MB, 00:31 IST), so no retrain is needed.
- Fix (lead applies on g1): `cd ~/aml && mkdir -p submissions/archive && touch logs/rebuild/s1_test.done && nohup scripts/rebuild_chain.sh main >> logs_main.txt 2>&1 &`. The mkdir also covers every later er_predict step. (Optional src hardening: `out.mkdir(parents=True, exist_ok=True)` in er_predict.py.)
- 00:33 IST: submissions/archive created on g1, s1_test marked done, chain restarted (feat4_train running). RESOLVED.

## 27 Sep 01:35 IST: 006 stage-2 OOF: PASS
- Rebuilt CV **0.97425** vs original 0.97423 (+0.000023, per-fold SE 0.000019: noise) (M). India 0.97212 vs 0.97207, US 0.97568 vs 0.97567; pair P 0.99319 vs 0.99322, R 0.93999 vs 0.93991. Oracle identical for every S1; 97.96% of S1 have an identical F0.5.
- s1_test (005f test probabilities, 100% train): 98,021,508 train pairs / 7,490,646 positives as in the original manifest; test file 78,924,662 rows = test_cand (M).

## 27 Sep 02:10 IST: 006 test output vs the original upload-4 file: PASS (run-to-run variation only)
- Rebuilt `01_006_stage2.tsv` (g1) vs original `04_006_stage2.tsv` (M): pair Jaccard 0.9909, 97.07% of S1 identical (France 95.7%, India 96.9%, US 97.8%); 5,786,760 vs 5,788,376 pairs.
- Guard, original -> rebuilt (pairs/S1, empty, house-conflict): US 3.3292/0.0589/0.0779 -> 3.3291/0.0588/0.0778; India 3.3165/0.0585/0.1678 -> 3.3160/0.0583/0.1679; France 3.4475/0.0499/0.0180 -> 3.4430/0.0498/0.0174 (M). Aggregates equal to 3-4 decimals.
- Same python/polars/numpy/lightgbm versions, same training sample hash (df5d0317...), same 8 threads, same n_train_pairs/n_pos in both manifests; the only difference is the CPU (AWS r7a vs GCP), so the 2.9% of S1 that differ are floating-point run-to-run variation compounded over two retrained stages. Consistent with the CV match (+0.00002). For judging 019 later: expect a similar 2-4% of S1 to differ from 11_019_band.tsv, with matching aggregates.

## 27 Sep 02:45 IST: paused by the lead (016 band CV on fold 4; retrieval on g2 still running). Remaining checks, in order
Tools (Mac scratchpad of this session): `cmp_tsv.py A.tsv B.tsv` (per-S1 / per-pair comparison by country), `guard_tsv.py f1.tsv f2.tsv ...` (pairs/S1, empty, house-conflict, added/dropped vs previous; validated on 11_019_band exactly), `veto_check.py f.tsv` (France veto hits + house signature), `vf/` (test house columns).
1. **016 band**: `results.json` vs original 0.98589 (per fold 0.98579/0.98588/0.98587/0.98592/0.98600; India 0.98466, US 0.98671); per-entity paired vs the original is not possible (no original entity_scores for 016), so check aggregates + oracle 0.99398 + P 0.99823 / R 0.96037.
2. **Route (g2 union, g1 fit/score)**: test new pairs kept with p > 0.001 about 0.94/S1 (original); fit uses 10% owner sample, 300 rounds.
3. **018 lexical --cv**: expect CV about 0.98605 (no train route pairs, see 23:25) and paired 016 -> 018 about +0.00016, >= 4/5 folds.
4. **019**: rebuilt `NN_019_band.tsv` vs original `11_019_band.tsv` (cmp_tsv; expect 2-4% of S1 differ, aggregates equal); guard vs reference US 3.3624/0.0579/0.0863, India 3.3268/0.0585/0.1718, France 3.2585/0.0588/0.0100; candidates about 7.25/S1 (12.6M).
5. **023 comb**: paired 018 -> 023 (alarm below +0.0011; original 021 +0.00140, US +0.00123, India +0.00165); per-fold 5/5; France unchanged on test (guard France identical to 019); guard US/India vs 020's (US 3.3663/0.0866, India 3.3303/0.1722).
6. **024 keys**: test guard vs 023; kept K9+rtext pairs (reference K9 alone 1,511,429); emitted additions per S1 vs reference France +0.026, India +0.019, US +0.002. Train side: UNVERIFIED unless the lead runs the owner-sample chain (23:20 list).
7. **025 veto**: `veto_check.py` on 024's output; expect a handful of France pairs (6 on the 020 output).
8. **Cascade**: `logs/rebuild/cascade_check.log`: diff vs full must be >= 0 or within noise at the chosen (minp, topk); candidate count per S1.
9. **026 blend**: grid log after the parity fix; require paired PASS on all 5 folds and on the owner sample; count 024 emissions dropped by 026 on test by source (base / route / K9-rtext) and country; package blocker (other pipeline's code) stands.
10. **Final package**: README from er_package for the chosen attempt (PRE_STEPS/EVAL gaps of 023-026, template placeholders `[update with the final upload]` / `[final]`, CV numbers), `validate_submission.py --check-ids` PASS on g1, candidates = decided set, every match inside it, zip file list and name/trace scan, manifest src hashes = src at package time (build on g1).

## 27 Sep 08:05 IST: adversarial audit of the r18 meta-model (m023 = 0.99037 CV, +0.00295 vs 023): interim
Scripts: g2 `~/vf18/a1.py`-`a4.py` (copies in my scratchpad). Inputs: g2 `cache/oof/{023_comb,test_023_comb,m023}.parquet` (md5 = g4's r18 copies), g4 `bc_{train,test}.parquet`, `t_m023.parquet`. Numbers below are M unless tagged.

**(1) Fold leakage: no first-order leak found.**
- meta.py: the fold-f model trains on `fold != f` of our folds and predicts `fold == f`. Every input is a probability, a CE score, or a label-free pool-side aggregate of them. No label, no fold-f fit.
- B's q (`stack_ce.py evaluate` with SAVE_IC=e1): outside the band, q = p2 of `exp_final_s25_spec`, predicted per row by the model of that S1's own crc32 fold (`predict_stream(row_fold=...)`, models trained on `f != k`, early stopping on a slice of the training folds). Inside the band, q = p3 from `oof_stage3` (fold `!= k` -> `== k`). The CE scores are heldout by fold parity (half h scored by the model trained on the other half). B's pair tables left-join the gold (`pairs.py`); no gold pair is injected into the train candidates, and no B feature file reads the ground truth except the transliteration dictionary (same in-sample mining as our er_dict).
- Second-order paths, all accepted elsewhere in our stack (feat4, r11, 023): (a) the q, CE and pool features of the meta's training rows come from B models that saw our fold f's labels (B folds are not our folds); (b) B's stacker sees CE features of training rows from the half model that saw its validation fold; (c) B chose its settings on the same OOF (self-reported optimism 0.001-0.004 at B level). None of these puts a validation row's own label into its features.
- Not verifiable: README step 9 says B's delivered export was an unlogged one-off; the q file reproduces B's reported OOF exactly (0.98783 vs 0.98782), which is consistent with the code above but does not prove it.
- Real train/test feature shift inside the meta (not a leak): e5i/e5b are single heldout models on train but the **mean of both half models on test** (prep_b.py). The means are smoother. r18 measured "no CE inputs" as equal on 005n (0.98884 both). Recommended fix: drop e5i/e5b from the meta (`--drop e5i,e5b`) unless m023 without them is > 2 SE worse.

**(2) Train/test parity: the cascade is built the same way; the gap comes from our side (A), and it means the train CV over-states the test gain.**
- B's cascade is the same on both splits: train keeps p2 > 0.02 and top 10 by p2 per S1 from B's OOF p2. The ranks among exported rows equal the full-table ranks for every row with p2 > 0.02. B rows/S1: US 3.94 train vs 4.09 test, India 3.88 vs 3.97.
- **Where the gain comes from.** Meta restricted to A's own candidate universe: +0.00042 vs 023 (US +0.00048, India +0.00033, 5/5). Rows outside A's 98M candidates (B-only rows): **+0.00253** (US +0.00210, India +0.00316). So 86% of the CV gain is union recall: 60,238 B-only pairs emitted on train (US 0.0227/S1 at 99.1% precision, India 0.0342/S1 at 98.5%).
- **Why test adds less.** Train A = 023 OOF with **no route pairs at all** (the rebuild links 017 -> 016 on train) and no K9. Test A = test_023_comb, which carries the route (A emits 0.0110/S1 US, 0.0122/S1 India route pairs on test). B-only rows/S1 relative to A's base candidates are the same on both splits: US 0.0505 train vs 0.043 test, and the meta emits about 45% vs 40% of them. But on test the route already holds half of them. Emitted additions that are new to test A: US 0.0063/S1 B-only + 0.0019 route pairs A did not emit, vs 0.0227 on train (0.36x); India 0.0253 + 0.0020 vs 0.0342 (0.80x). With 024 (K9 + reverse text) as A, test coverage grows further.
- Label-free check on those additions (house-conflict share of B-only additions): US 0.241 train vs 0.230 test; India 0.287 vs 0.318 (x1.11). Overall emitted conflict US 0.0864 -> 0.0875, India 0.1721 -> 0.1735 (same as r18's guard). No decoy signature.
- Running on g2: route (name/address char-3-gram top 40) and K9 reachability of the 60k train B-only additions, to measure directly how much of the +0.00253 route/K9 would deliver anyway.
- 08:35 **Route/K9 reachability of the train B-only additions (preliminary: K9 keys + route name view; address view running).** Reachable by K9: US 65%, India 77%; by the route name view: 28% / 28%; by either: **US 69%, India 80%**. Precision of the additions: reachable 99.2% / 98.9%, unreachable 98.9% / 96.7%.
  - Gain carried by the reachable B-only additions: +0.00188 (US +0.00143, India +0.00255). Gain carried by the unreachable ones: +0.00062 (5/5).
  - **Meta vs (023 + every reachable pair at the meta's probability): +0.00103 (SE 0.00001, 5/5; US +0.00113, India +0.00089).** This is a lower bound on the meta's gain over a perfect 024: it assumes 024's route classifier and decision emit every reachable pair as well as the meta does.
- 08:51 **Final reachability (K9 + route name + route address views, all 60,238 train B-only additions):** reachable **US 87%, India 83%** (K9 65% / 77%, route 65% / 32%). On test, K9 alone reaches 63% (US) / 72% (India) of the test B-only additions vs 65% / 77% on train, so the parity holds.
  - Gain carried by reachable B-only additions +0.00213 (US +0.00182, India +0.00261); by unreachable ones +0.00037 (US +0.00027, India +0.00052); within A's own candidates +0.00042.
  - Upper bound of what route + K9 could take from these pairs (023 + every reachable pair at the meta's probability): +0.00217 over 023.
  - **Meta vs that bound: +0.00078 (SE 0.00001, 5/5; US +0.00074, India +0.00083).** This is the floor of the meta's gain over a 024 that captured every reachable pair perfectly.

**(3) Realistic LB gain (D unless noted).** LB weights: US 663,106 / India 809,986 / France 259,452 of 1,732,544 test S1 (0.383 / 0.468 / 0.150). France keeps A's probabilities, so it adds 0.
- **Over 023 as uploaded (route, no K9):** scale the B-only part by the measured test/train addition ratio (US 0.36, India 0.80): US 0.00048 + 0.36 x 0.00210 = +0.0012, India 0.00033 + 0.80 x 0.00316 = +0.0029. That gives **LB about +0.0018**.
- **Over 024 (K9 + reverse text on test):** the floor is 0.383 x 0.00074 + 0.468 x 0.00083 = **+0.0007**. If 024 realises about 70% of the reachable-pair value (G; its own measured gains, route +0.00126 and K9 + rtext +0.00123, are the same order as the +0.00217 bound, and they also include pairs B lacks), US is about +0.0013 and India about +0.0016, so **LB about +0.0012**. Haircut 10-20% for test decoy density (India B-only additions: house-conflict 0.318 on test vs 0.287 on train) and B's self-reported OOF optimism. **Realistic: +0.0007 to +0.0012 over 024**, not the +0.0025 the headline CV suggests.
- **Over 028:** not measured (m028 has no result line on g4). 028's cross-encoder partly overlaps B's CE signal, so expect it at or below the 024 range (G).

**Verdict: GO WITH CAVEATS** (no leak; the CV gain is real but about 2.5-4x larger than what test can deliver on top of 024):
1. Build the final meta with A = the final test stack (024 or 028) and gate it on the test guard by class. Train cannot see route/K9, so CV will keep over-stating the gain; state the LB expectation as +0.0007 to +0.0012.
2. Drop e5i/e5b from the meta inputs (test = mean of two half models, train = one heldout model), or keep them only if `--drop e5i,e5b` costs > 2 SE on 023.
3. France: A's probabilities (the meta never saw a French label; r17 owns France).
4. **Package blocker (same as 026):** the output depends on B's probability files and CE scores. The zip must contain B's code path (CPU pipeline, Kaggle CE kernels, the export step) and B's README admits unlogged steps (export, kernel wrappers, non-bit-identical CE reruns). If the organisers' "regenerate from this folder" rule is enforced, this is a compliance risk that the human must accept explicitly. Without that acceptance: NO-GO for the final package, and 024/028 stays the final.
5. The g1 chain was not touched. On g2 I wrote only `~/vf18/*` and `cache/cand/{train,test}_ret_{keys,name,address}_vf.parquet` (tag `_vf`, safe to delete).

## 27 Sep 09:45 IST: FINAL GO / NO-GO for 029b_final (06_029b_final.tsv, Apprentice_submission.zip on g1)
**(1) Zip (M): PASS, with doc fixes below.**
- `output/matching_results.tsv` md5 eb8d8813... = `06_029b_final.tsv`; `output/candidate_pairs.tsv` md5 432b8f2c... = `06_029b_final.candidates.tsv` (byte-identical). Validator PASS (1,732,544 rows; 100,120 empty; candidates 32,049 empty S1).
- Contents: 85 code files + README + requirements + config + Documentation_template.md. No data, cache, credentials, keys, tokens or home paths. Scan of every code file for AI/tool names, person names, team or partner wording: clean. The only hits are the word "partners" in a filler-word list and "a 16 GB laptop" in `src/stack/README.md` (hardware note, harmless). Names appear only in Documentation_template.md (template header).
- README chain vs what ran: same steps and a valid order as rebuild_chain.sh (feat5/feat6/lift read no probabilities, so running them before stage 2 is fine); the 017_route copy (cp instead of the chain's symlink, equivalent) and `018_lexical --cv` are present; then er_gen, er_comb 023, the CE export/Kaggle/collect, er_ce 028, the stack family README, er_meta 027d, er_france 029b, er_predict. Every intra-package import resolves (ce/ce_export.py adds src to sys.path). Every attempt in DEPENDS is in the zip.
- Note: `src/stack/README.md` ships its section "Not reproducible from this code alone (honest gaps)" (unlogged export step, reconstructed flags, CE reruns not bit-identical). That is honest, but it is an explicit admission against the "regenerate from this folder" rule. The human should know it is in the zip.
- Note: g1's `cache/stack` has only the p_* files, not the `scores_{e5i,e5b}_*` files that `er_meta` (STACK_CE parity) reads. g4's `cache/ext` has them. The package is unaffected, but g1 alone cannot rerun 027d.

**(2) Reproduction from cache on g2 (isolated copy of the zip's code in `~/vf29`, M): PASS, byte-identical.**
- `er_meta.py 027d_meta` (inputs: 028_ce train/test, the stack p_* files with md5 = g1's, the e5i/e5b score files from g4): CV **0.99046** (US 0.99071, India 0.99009), 028_ce 0.98779. Train OOF md5 088345cf... = g1; test md5 1b83f607... = g1 (probabilities identical, max diff 0.0).
- `er_france.py 029b_final`: test_029b_final md5 f302fc8f... = g1 (712,367 France pairs vetoed to 0; 80,831 stack-only pairs).
- `er_predict.py 029b_final --from-probs 029b_final`: matching md5 **eb8d8813...** and candidates md5 **432b8f2c...** = g1's files.

**(3) Final guard vs upload 11 (12_020_gen, LB 0.985; = submissions/matching_results.tsv, md5 d32fe6d6...) (M):**
| country | pairs/S1 | empty | house-conflict |
|---|---|---|---|
| US | 3.3663 -> 3.3858 | 0.0579 -> 0.0577 | 0.0866 -> 0.0876 |
| India | 3.3303 -> 3.3677 | 0.0586 -> 0.0579 | 0.1722 -> 0.1737 |
| France | 3.2585 -> 3.3190 | 0.0588 -> 0.0579 | 0.0100 -> 0.0104 |
- Added 72,794 pairs (US 16,392, India 34,163, France 22,239); dropped 13,882. 95.3% of S1 unchanged (US 97.1%, India 95.6%, France 89.7%).
- Signature of the change (conflict share / share with a decoy house shift d in {1..21}): US added 0.318 / 0.172, dropped 0.505 / 0.265; India added 0.314 / 0.054, dropped 0.333 / 0.106; France added 0.048 / 0.008, dropped 0.080 / 0.063. The reference model change 019 -> 020 (LB went up) had US added 0.387 / 0.156, India 0.399 / 0.078. In every country the added pairs are cleaner than the dropped ones, and conflict per country rises by only 0.001-0.0015. **Guard PASS.** France moves the most (10% of S1, +0.060 pairs/S1). It has no labels, so r17's label-free evidence is the only support.

**(4) Documentation (fix before the zip is final; wording suggestions):**
- **Wrong for 029b:** section 4 says "extra candidates from exact name keys and reverse character retrieval scored by the route classifier", and appendix row "+ cross-encoder band score, extra key / reverse-retrieval candidates | 0.98779". 029b uses 023 pairs (no K9/rtext; no er_keys/er_rtext step in the README), and 0.98779 is 028_ce's CV, which has no K9. If 029b ships: delete that clause and relabel the row "+ cross-encoder band score | 0.98779". (If 029_final with 024 pairs replaces it, the wording holds, but the row's CV must be that attempt's.)
- **Contradiction with the design rule:** section 2.1 says "Every feature compares one pool record with its own S1", and 2.2 says "every stacked input out-of-fold". The second family's stack uses within-S1 cross-encoder rank/gap features and supporting-record features, and its OOF is on its own entity folds. Suggested: "Every feature of the main pipeline compares one pool record with its own S1 ...; the second model family's probabilities are out-of-fold on its own entity folds and enter the meta-model only as inputs."
- **Overclaim of the CV:** in section 5, after "0.99046 (US 0.99071, India 0.99009)", add: "The training candidates do not contain the retrieval-route pairs that the test candidates have, so part of the meta-model's +0.0027 CV gain over 028 is recall that the route already provides on test; we expect about +0.001 on the leaderboard." (My estimate: +0.0007 to +0.0012 over 024-class stacks, about +0.0018 over 023 as uploaded; D.)
- "The final model cut CV false-positive pairs from 12,352 to 7,578": not re-measured in the rebuild (it is the original 021 number). Re-measure on 027d or delete.
- Section 3: "The final candidate band (stage-2 probability >= 0.001 plus route pairs)": now also includes the second family's pairs. Add "and the second model family's pairs". Its "decisions are identical to deciding on the full set" was measured for 019 only.
- Section 5 LB line and appendix "this submission": fine as placeholders until the final LB is known. Update after upload.

**VERDICT: GO for 06_029b_final.tsv as the final upload** (reproducible byte-for-byte from the zip's code and cache, validator PASS, guard PASS). **GO for the zip after the documentation fixes above** (rebuild with er_package --final; the output files must stay md5 eb8d8813... / 432b8f2c...). The residual risk is compliance: the second family's reproduction relies on its README with admitted gaps and on Kaggle GPU runs. That is the human's call; it does not affect the matching file.
