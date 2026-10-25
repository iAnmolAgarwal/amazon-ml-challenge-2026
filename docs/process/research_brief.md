# Research brief: 10 parallel lanes toward >0.98 (evidence first, no builds)

## Goal and ground truth
- Current: submission 4 = attempt 006_stage2. CV 0.97423 (India 0.9721, US 0.9757), public LB 0.963, implied France 0.902.
- A separately built parallel pipeline scores LB 0.971 (OOF 0.98082; description `~/Downloads/OVERVIEW.md`).
- LB = 0.47·India + 0.38·US + 0.15·France (test composition); the LB−CV gap is steady at about −0.011.
- **Goal:** find approaches with evidence (measured on training labels with the paired 5-fold rule, plus label-free test checks) that can take the LB above 0.98.
- Reviewers so far call that a stretch: it needs roughly US 0.990, India 0.986, France 0.94. Each lane must state honestly how far its idea can move each country.

## Rules for every lane
- Read docs/STATUS.md, CLAUDE.md and docs/astra/10_*, 11_* first. Measure; don't guess. Label every number as measured, derived or guessed.
- **No test labels exist.** Decisions come from training CV only. Test may be used label-free (distributions, guards).
- **No edits to src/ or attempts/ and no commits.** Prototype in the scratchpad (`<scratchpad>/lanes/laneNN/`). Report to `docs/research/laneNN.md`.
- **No team or person names, and no mention of AI tools,** in any repo file.
- **Compute:**
  - Mac has 8 GB RAM and 10 lanes share it. Each lane stays under ~600 MB resident: polars lazy, small entity samples, one script at a time.
  - EC2-A (`ssh amlc-a`, 8 cores / 64 GB, repo at ~/aml, caches at ~/aml/cache) is shared. Heavy runs must take the lock: `ssh amlc-a 'flock ~/heavy.lock <command>'`, one at a time, ≤ 45 min each.
  - Prefer the cached OOF probabilities and features already on A over new full builds.
- **Useful caches on A:**
  - `cache/feat/train_feat*.parquet` (feat, feat2, feat4) and `cache/oof/{005n_indic_nocoh,006_stage2}.parquet` (s1_idx, pool_idx, prob);
  - `cache/norm/*`, `cache/cand/*`, `data/folds.parquet`;
  - `cache/oof/test_006_stage2.parquet` (test probabilities).
- **Evaluation helpers:** `src/er_model.py` (decide, per_entity_f05, load_gold, folds); `src/er_eval.py --from-oof` pattern; `src/common.py` (paired_gain).
- **Subsample protocol:** sample S1 entities, not pairs, and keep the full candidate graph of each sampled entity. When sampling changes rival sets (reverse retrieval, competition features), say so and correct for it.

## Lanes (one agent each)
1. **Name discrimination.** er_feat5 (draft, toy-checked; directional IDF masses, name/address cosines, skeleton chain size) + legal-form equal/conflict + name char-trigram cosine. Measure the stage-2 paired gain on a large entity sample with stage-1 held fixed (11_audit measured +0.006 US on the older base). India with the dictionary base. Memory-safe loader design.
2. **Missed matches.** 0.0164 of 0.0224 test-weighted loss; 86% of under-predicting entities miss exactly one. Root-cause the misses of 006: retrieval vs ownership vs floor vs low probability, by name commonness, empty address, source, operator. Propose targeted fixes with measured headroom.
3. **France.** Parser bugs from 10_audit_opus: 118k house numbers dropped, N°→north, et/and, EI, department vs region, glued web legal forms, missing street abbreviations. Quantify affected records and changed candidates on test (label-free). Design leave-one-country-out validation (train US→score India and the reverse, with the held-out country excluded from all supervised steps). Estimate France upside honestly.
4. **Blocking.** v3 (reverse top-3 + skeleton/first-last keys, `er_block.py --rev-k 3 --keys …`) vs current: oracle and pair recall on the same entities (full S1 index for reverse retrieval, sample only the evaluated entities). Candidate growth and memory. Is retrieval still worth anything (oracle 0.99398 now)?
5. **Model capacity and stacking.**
   - stage-1 OOF at 100% of the other folds (test uses 100%; CV used 40%);
   - rounds and learning rate, leaves, early stopping on a train slice;
   - a leak-free stage-2 protocol (the rival-probability leak);
   - seed averaging.

   Measure on samples; project full-data gains.
6. **Decision and calibration under test shift.** Test has 1.9x the unmatched pool per S1 (2.30 vs 1.22). Build the decoy-dense simulation properly: remove ~19% of S1, recompute graph, rival and competition features, re-score. Test floor, ratio and prior-shift rules. Only rules that also hold in ordinary CV may ship.
7. **House number and address relations.** Dropped leading digit (1995 vs 995: 24% of US gold conflicts, 0.3% of hard negatives), nearby number on the same street (40% of hard negatives), unit/suite, postal. Empty-address records are the hard core of ownership losses. Feature designs with measured separation.
8. **India residual.**
   - the 171 test-only native business words (14% of test native names unmapped): a label-free mapping with ≥97% precision on held-out known words;
   - Indic address components;
   - India-specific decoys.
9. **Ensembling.** Blend with the parallel pipeline (needs its OOF probabilities keyed by entity ids; its test predictions exist) and multi-seed/multi-param averaging. Measure error complementarity on training labels. The candidate-union protocol treats missing scores as missing, not zero.
10. **Generator reverse-engineering.** Catalogue every noise operator and its rate in train gold vs test (label-free): which operators our normalization/features do not invert (e.g. f/k/a, a/k/a, fka, "doing business as" missed by the alternate-name regex). Any operator with a large rate and no handling is a lever.

## Deliverable per lane (docs/research/laneNN.md)
1. **Finding:** a one-line verdict and expected LB gain (range, confidence) by country.
2. **Evidence:** the measurements, with sample sizes and paired fold numbers where applicable.
3. **Build spec:** exact code changes (files, functions), compute hours on one 8-core/64 GB machine, and memory.
4. **Interactions:** which other lanes it overlaps or conflicts with.
5. **Go/no-go test** for the full build, plus the label-free test guard to apply.

The leader then merges the lanes into one build plan and runs it only when the combined evidence supports it.
