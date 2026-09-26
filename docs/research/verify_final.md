# Final-submission verification (independent verifier)

Scope: rebuild of the upload-10 chain on g1/g2 and the rewritten modules (023 comb, 024 keys, 025 veto, cascade, 026 blend), test guards, final package.
Tags: **M** = measured by me on data, **D** = derived (arithmetic on measured numbers), **G** = guessed / not confirmed. Times IST. Entries are appended; the latest verdict per stage wins.

## Stage verdict table (updated as stages land)

| Stage | Verdict | Evidence (short) |
|---|---|---|
| Code identity (core src vs code that built 019/020) | PASS | see 26 Sep 23:15 |
| Folds | PASS | identical to original (hash) |
| dict / norm / block (train+test) | PASS (counts) | 98,021,508 train pairs = original manifest |
| 005n stage-1 OOF | PASS | 0.97005 vs 0.97008; oracle identical per S1 |
| 005f / 006 stage 2 | PASS | 0.97425 vs 0.97423; oracle identical |
| 016 band | pending | |
| 017 route / 018 lexical | pending | |
| 019 vs original upload-10 file | pending | |
| 023 comb (rewritten) | pending (code review below) | |
| 024 keys | pending | |
| 025 veto | pending | |
| cascade | pending | |
| 026 blend | pending | |
| Final package | pending | |

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
