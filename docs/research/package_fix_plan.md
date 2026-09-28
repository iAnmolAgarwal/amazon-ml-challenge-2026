# Final package fix plan (28 Sep 2026)

Goal: raise the final ZIP to a clear top-50-of-1000 package after three evaluator audits.
- Evaluator reports: `docs/research/eval_A_reproducibility.md` (A: not top-50 as shipped, borderline after HIGH fixes), `docs/research/eval_B_methodology.md` (B: 6.5/10 borderline; clarity 4.5, credibility 6), `docs/research/eval_C_compliance.md` (C: pending when this was written).
- Current package: `submissions/final_package/Apprentice_submission.zip` (105 files; built on g1 from attempt 029f6_final). The matching file is the best upload, LB 0.990203, md5 9261e5c681ddfb1b49ac5cc8822f737f. Also at the repo root as `matching_results.tsv`.
- **The matching file must stay byte-identical (md5 9261e5c6) after every rebuild.** Rebuild flow on g1: sync src/attempts/docs, then `er_predict.py 029f6_final --from-probs 030q_fix` (refreshes the manifest), then `er_package.py <tsv> --final --team Apprentice`, then verify md5 + validator, then copy to the Mac.

## Done so far
- `src/er_frwords.py` (new): label-free derivation of the France VETO/BROAD word lists from France test candidates (single inserted word, same street; decoy house shift vs equal house; legal forms excluded). It reproduces the shipped lists exactly:
  - VETO = distribution, holding, international, participations (about 17.6-17.7k shifted insertions each vs 51-60 at an equal house; shift share 0.997);
  - BROAD = developpement, france, groupe (about 16.5-17.8k shifted vs about 1.2k equal; share 0.93).
  - Still to do: add it to SRC_FILES and to the README chain before er_france, and mention it in the docs.

## Package fixes (A + B)
1. HIGH: `attempts/029f6_final/train.py` DEPENDS must include `029f5_final` (the README runs `er_france.py 029f5_final`; 030q_fix reads its test file). Remove `019_band` from DEPENDS (unused).
2. HIGH: ship the run manifest (`submissions/archive/18_029f6_final.json` on g1) as `code/business_entity_resolution/run_manifest.json`, plus SHA256 of both output TSVs. Add a README "Verification" section with expected stage CVs: 005n 0.97005, 006 0.97425, 016 0.98589, 018 0.98604, 023 0.98742, 028 0.98779, 027p 0.99177, 030p 0.99184; say the GPU steps reproduce closely but not bit-identically.
3. HIGH: the README must say which steps were rerun and which were reconstructed (the second family's export step and two command lines; see src/stack/README.md).
4. MEDIUM: `src/ce/ce_train.py` should pass `revision=` from ce_cfg.json (e5-small 614241f622f53c4eeff9890bdc4f31cfecc418b3, e5-base d128750597153bb5987e10b1c3493a34e5a4502a); default None.
5. MEDIUM: fix the README e5-base config line to include `"maxn": 6200000`.
6. MEDIUM: add a README compute section with a numbered table of the Kaggle runs. Cache disk measured on g1: 120 GB total (feat 52, oof 30, research arrays r19s/r19s1 32, cand 3.9, norm 2.7 GB). Evaluator A estimates about 30-55 CPU-hours on 12-vCPU / 96 GB machines plus about 17 GPU-hours (Kaggle 2x T4). State the working directory (code/business_entity_resolution/).
7. MEDIUM: `src/stack/Makefile` header: "historical, superseded by the README command list".
8. LOW: GPU pins in a separate `requirements-gpu.txt` (torch 2.14.0, transformers 5.17.0, pandas 3.0.6); leave er_feat3.py out of SRC_FILES; fix stale strings in `attempts/027p_meta/train.py` (PREDICT names 027l, PRE_STEPS names 029f2) and `attempts/024f_keys`; config.yaml unused keys (grep before removing).

## Documentation_template.md rewrite (B's H1-H7, M1-M7, L1-L4; see eval_B_methodology.md for the exact text)
- H1: the false-negative bullet is wrong. The empty address is on the POOL record (train S1 has 0 empty addresses). Unretrieved pairs cost at most 0.0020 of the 0.0082 loss (oracle 0.99798). Name-twin empty-address records = +0.00424 of the +0.00636 in-candidate headroom (r20_errors.md, on 027e). Single-match S1 score lowest (0.969 on 027e, r20).
- H2: the sibling-feature contradiction. Say instead: no feature compares a candidate's content with the S1's other candidates; score-based ranks and margins are allowed. The negative-result numbers (+0.0033 CV graph vs -0.037 at doubled decoys; +0.0028 stacker, 3.4x / 5.8x extra test pairs) must be VERIFIED in r7.md / r4.md / r10.md before use. r10.md has +0.0028 as the oracle of the identical-name empty-address class, a different number.
- H3: decoy shift spike verified in r9.md:30 (non-matches spike at exactly the ten shifts, 13.6k-15.8k pairs each vs 1-3k elsewhere). The "1.94x test decoy density" still needs a source.
- H4: split Section 4 into 4.1 pipeline diagram, 4.2 numbered models, 4.4 France subsection (derivations: LOCO numbers plus er_frwords.py), 4.5 decision rule.
- H5: a validation-protocol block. Do NOT claim "LB never used". Be honest: the France name-twin correction passed LOCO in one direction only (+0.00006 US->India 5/5, +0.00002 India->US) and was kept after it raised the public score.
- H6: oracle 0.9980 (not 0.9969); recompute the FP/FN counts from the final OOF (030p_fix) or cite the run.
- M1 executive summary in 3 sentences; M2 conclusion attribution + limitations; M3 one-model ablation table (r19_simple.md: chain 0.98779 = one LightGBM 0.98779; without gen+ce 0.98638; XGBoost 0.98778; GAM 16 features 0.98017); M4 negative-results table; M5 table footnotes; M6 runtime line; M7 glossary. L1 delete the leftover template "Note:" line; L2 candidate table; L3 France bound wording.
- Keep length at or under ~3,000 words; keep the team header (Apprentice: Anmol Agarwal (leader), Akshit Kumar Singh, Nitya Arora, Harsimran Singh).

## After fixes
Rebuild the zip, verify (validator PASS, matching md5 9261e5c6, README chain has no missing attempts, no names outside the Documentation header), then have evaluators A/B/C re-audit the new zip.
