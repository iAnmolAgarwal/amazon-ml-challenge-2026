# Leads from the parallel line, and one measured France lead

Status: complete (26 Sep). Compute: machine C (G was reclaimed mid-run; everything was re-run on C).
Tags: **M** measured, **R** their number re-read from their files, **D** derived arithmetic, **G** guess. Sources read (read-only): `/private/tmp/aml_perfect_round2/` (Mac),
`amlc-a:/var/tmp/aml_*` and `/tmp/aml_perfect_round2_*`, and the session transcripts (assistant summaries only).

## 1. What the parallel line found (ranked by credible LB value for us)

| # | Finding | Their numbers (R) | In our pipeline? |
|---|---|---|---|
| 1 | Stage-3 residual on logit(p006) with 32 pair features (specificity, house relations, legal form, char-3-gram) | +0.0079 on held-out cohort 3 (0.97395 -> 0.98185, 5/5) | **Yes** = our feat5 / 008 |
| 2 | Char-3-gram name+address retrieval (top-40 each) scored by a separate 66-feature pair classifier | cohort 6: 0.98411 -> 0.98537, +0.00127 (SE 0.00015, 5/5); +56 pairs/S1 | **Yes**: verified +0.00128 over 009, in upload 7 |
| 3 | Label-derived lexical likelihoods (42 features, 6 edit channels) on top of R5 | +0.0020 to +0.0022 on cohorts 3/4/5 (5/5 each) | **Ported** for upload 8 (+0.0003 over 009-type build); France-blind |
| 4 | French missing-house parser (padded / parenthesised numbers, street types ch, crs, cour, q, blvd, ...) | recovers 3,068 S1 + 79,072 pool houses; with a frozen R5 model France emits 5,035 fewer pairs (+193), 78.8% of removals carry a decoy house shift | **No.** r3 found the defect (sec. 3.5), unmeasured; 009 still uses the old parser |
| 5 | Country-normalised label-free marker scores (score / country max) for unseen-country transfer | LOCO, full rival closure, 1,078 S1: US 0.9196 -> 0.9376 (+0.018), India 0.8148 -> 0.8171 (+0.002). Raw (unnormalised) markers **hurt** US transfer (AP 0.970 -> 0.962) | **No.** 009 (LB 0.981) uses raw scores; France table built from test pairs |
| 6 | Lexical layer hurts unseen-country transfer | LOCO: US 0.9378 -> 0.9137 (-0.024), India 0.8268 -> 0.8227; markers win back +0.010 / +0.004 | Warning: keep lexical out of France |
| 7 | Reverse retrieval from baseline-unassigned pool records | oracle only: +0.0026 to +0.0029 (rank 3, +11M train pairs); cohort-8 bundle frozen, never evaluated | No (overlaps the route / r8) |
| 8 | Target pseudo-labels, synthetic positives | US -0.0076 / India +0.0100 (not replicated: -0.0028); synthetic -0.0022 / -0.0045 | Rejected; relevant to r3 recipe (d) |
| 9 | Neural cross-encoder (306M multilingual reranker, private GPU notebook) | GPU smoke only (311 pairs/s inference); first real run crashed on file names; **no quality number** | No. Pretrained external weights: rule status unclear |
| 10 | Clean end-to-end US outer validation (20% train / 80% reference) | still running on A (stage 2 frozen); no score | - |

Rejected by them (all R, paired on held-out cohorts): joint name/address aliases (+0.000003), raw formatting features
(+0.0001, CI spans 0), address char-3-gram cosine (-0.00007), 18-feature signed numeric channel (+0.00023; US +0.0006,
India -0.0003), Platt calibration (+0.00001), anchor-LLR token contrast (LOCO US -0.010, India -0.005 vs markers),
exact raw-name sibling propagation (18-25% precision), pool alias bridges (+0.00016 oracle), source-count balancing
(chance level), mapping the 171 native shop words (distractor-only, same as our r8).

Ceilings they measured (cohort 6, at 0.98537): add every retrieval miss 0.9899; remove every FP 0.9882; perfect
classification on expanded candidates 0.9957. 711 of 996 retrieved empty-address misses share the gold owner's exact
raw name with other S1 (no name evidence). With US/India at 0.995, France must reach 0.962 for 0.99 overall (D).

Files: `ROUND2_FINDINGS.md`, `NEXT_ITERATION_HANDOFF.md`, `france/FINDINGS.md` (secs. 4, 7, 8, 9, 10),
`france/marker_lexical_loco/REPORT.md`, `france/anchor_llr_loco/REPORT.md`, `france/parser_r5.py`,
`france/parser_r5_results.json`, `france/loco_macro_closure_results.json`, `france/fresh_loco_results.json`,
`france/neural_pilot/RUN_STATUS.md`, `tail/cohort6/residual/REPORT.md`, `audit/cohort{3..7}/evaluation.json`.

## 2. Already ours vs new
- Ours or in build: 1 (008), decoy house shift and marker tables (009, LB 0.981, implied France ~0.960), 2 (upload 7),
  3 (upload 8), native-word finding (r8).
- New and France-relevant: **4 (parser)** and **5 (scale normalisation of markers)**. Both touch the features 009 adds.
- New but unmeasured: 7 (oracle only). Not pursued here.
- Warnings to carry: 6 and 8 (lexical and pseudo-label recipes hurt transfer).

Status of the parallel line now (M, read-only): no new result since 18:55 UTC 25 Sep. The clean US outer chain is
still in its `r5_fit` step on A (started 19:16 UTC); the neural pilot's first real run failed on file names and was
not rerun; the cohort-8 reverse-retrieval bundle is frozen but never scored.

## 3. Part 2: chosen lead = scale-free marker scores for the unseen, decoy-denser country
Why this lead (M, label-free, from the feat6 tables 009 uses): the marker score (nd+1)/(nd+n0+10) depends on each
country's decoy density. Pair-level base rate nd/(nd+n0), added tokens: US train 0.382, India train 0.246,
**France test 0.439**; swaps 0.335 / 0.177 / **0.474**; drops 0.244 / 0.213 / **0.346**. Weighted median added-token
score: US 0.20, India 0.23, **France 0.42**. So France enters the 009 model on a shifted scale (test has ~1.9x decoys and
its table is built from test pairs), and the most frequent French operators (legal-form swaps sarl>sas 0.48,
sarl>eurl 0.61, sas>snc 0.84) sit in the model's ambiguous range. Fix: **lift** = logit(score) - logit(country base
rate), a per-country monotone map (exact on the max-aggregated features: dm_x_l = logit(dm_x) - logit(base_c,x)).

### 3.1 LOCO (France proxy): train on one country, score the other. Stage-1 48 + r5 32 + r9 5 + 3 marker features,
rk params, 600 rounds, rival=None, paired by target fold. Full 10% sample, 9.8M pairs (`loco_dm.py`), M:
| arm @ floor 0.6 | F0.5 | India | US | paired (SE, folds) |
|---|---|---|---|---|
| no markers | 0.94509 | 0.92555 | 0.95810 | |
| raw markers (009 recipe) | 0.94479 | 0.92596 | 0.95734 | vs none -0.00030 (0.00021, 1/5): raw markers do not transfer |
| **lift markers** | **0.95336** | **0.93735** | **0.96403** | **vs raw +0.00857 (0.00014, 5/5) PASS; India +0.0114, US +0.0067** |
| raw, target table at 2x decoy density | 0.93806 | 0.94170 | 0.93563 | vs raw -0.00673 (0.00013, 0/5); US -0.0217 |
| lift, target table at 2x decoy density | 0.95431 | 0.93995 | 0.96387 | **vs raw2 +0.01625 (0.00019, 5/5) PASS** |
Floor 0.9: lift vs raw +0.00817 (5/5), lift2 vs raw2 +0.02417 (5/5). The 2% subset gave the same picture
(lift vs raw +0.01017, SE 0.00041, 5/5).

### 3.2 In-country, 009-style stage 3 (init logit p006, band p006 >= 0.001, r5 32 + r9 5 + 3 markers, 31 leaves,
300 rounds; full sample, 220,678 S1, harness folds; `s3_dm.py`, `s3_pr.py`), M:
| arm | F0.5 | India | US | paired | pairs/S1 India, US | empty |
|---|---|---|---|---|---|---|
| raw markers (009) | 0.98460 | 0.98335 | 0.98543 | vs no markers +0.00154 (0.00006, 5/5) | 3.321, 3.329 | 5.80% |
| **lift markers** | **0.98464** | 0.98338 | 0.98547 | vs raw +0.00004 (0.00003, 3/5): **neutral** | 3.321, 3.330 | 5.80% |
| raw, tables at 2x density | 0.97832 | 0.97622 | 0.97971 | vs raw **-0.00628 (0.00010, 0/5)** | **3.243, 3.263** | **6.03%** |
| lift, tables at 2x density | 0.98443 | 0.98323 | 0.98523 | vs raw2 **+0.00611 (0.00013, 5/5)** | 3.323, 3.332 | 5.79% |
The density loss is recall: raw2 drops ~15k true pairs (FP falls only ~430). Lift costs nothing in-country.

### 3.3 Label-free test check (sample-trained stage 3 applied to the full test graph; `test_arms.py`), M
Reproduction: the raw arm matches the real 009 emissions at Jaccard 0.988 France / 0.995 US / 0.993 India.
| arm | France pairs/S1 | France empty | France conflict | France emitted with decoy shift* | US pairs/S1 | India pairs/S1 |
|---|---|---|---|---|---|---|
| raw (= 009) | 3.221 | 5.98% | 1.07% | 1,893 | 3.346 | 3.320 |
| lift | 3.276 | 5.77% | 1.12% | 2,262 | 3.346 | 3.308 |
| raw + French parser | 3.214 | 6.02% | 1.07% | 483 | same | same |
| **lift + French parser** | **3.268** | **5.82%** | 1.12% | **748** | 3.346 | 3.308 |
*house shift in {3,4,5,7,9,11,13,21} under the repaired parser. Train OOF reference: 3.30-3.32 pairs/S1, 5.8% empty.
- France under 009 emits 0.10-0.12 fewer pairs/S1 than US/India with a higher empty rate: the same signature as the
  2x-density simulation (3.2). Lift adds 15.7k and removes 1.6k French pairs; with the parser 15.0k / 2.9k.
- Lift alone re-admits 612 hidden decoy-shift pairs; the parser blocks them (146 left). **Ship both together.**
- Parser alone: -2,092 / +137 French pairs; 1,410 of the removals carry a decoy shift. US/India untouched.
- India shifts by -0.012 pairs/S1 under lift with a lower conflict share (17.21% -> 17.12%); re-check on full data.

### 3.4 Value of the parser, measured with labels (`sim_blank.py`, 2% subset, in-country CV, flat model)
Hiding the house of 6.1% of house-bearing pool records at inference (France: 5.5% of pool records are recovered by
the repaired parser, M) costs **-0.00199 (SE 0.00013, 5/5; US -0.0032, India -0.0002)**. Emitted pairs whose true
house shift is a decoy shift rise from 1 to 230 (US, 97% false).

## 4. Build spec (needs human approval for src edits)
1. `src/er_feat6.py`, `features()`: per country and kind, base = sum(nd) / sum(nd + n0) of the table in use (train
   table for US/India, the test-built table for France); emit dm_x = logit(clip(sc)) - logit(base). New tag `feat6l`
   (keeps 009 reproducible). Cheap path: map the existing `train_feat6` / `test_feat6` columns per country (max
   commutes with the monotone map; bases in `research/cx/f6tables`). About 5 min.
2. French house parser (France rows only, train unaffected): in `src/er_norm.py` `addr_views`, France fallback regex
   `\b0*(\d{1,4})\s*(?:bis|ter|quater|[a-z])?[\s,().-]*` + street types {existing, ch, crs, cr, cour, q, blvd, bvd, parvis,
   esplanade, lotissement, chaussee, mail, venelle, ruelle, sente, digue, corniche, grande rue, rond-point}, only
   when no house was parsed. Minimal integration: recompute the feat5 house columns (h_suffix ... h_in_s, hd,
   hd_shift) for France test pairs only (`house_feats` in `test_arms.py`; parity 0.0 on 300k pairs), not a full
   norm/blocking rebuild. About 5 min.
3. Stage 3 as in 009 / upload 7 with `feat6l` in place of `feat6`, then eval, predict and the test guard. Also feed
   lifted markers to the route classifier if it uses dm_* features. One 64 GB machine, about 1 h.
4. Guard expectations: France pairs/S1 3.22 -> ~3.27, empty 6.0% -> ~5.8%, France decoy-shift emissions ~1,900 -> ~750;
   US/India within +-0.015 pairs/S1; in-country CV change within +-0.0001.

Expected gain (D/G): France from lift = the in-country density analogue +0.006 (M) scaled by France's observed
under-emission (~70% of the simulated shift) = about +0.004, up to the LOCO transfer gain +0.009 (M, single-source;
pooled training should transfer better, G); parser +0.002 to +0.004 (US-like blanking loss, France has ~2x decoys).
**France +0.006 to +0.013 -> LB +0.001 to +0.002 (G); US/India 0 (M in-country).** It does not reach 0.99 alone:
with 009 at LB 0.981 and the route (+0.0013) and lexical (+0.0003), this makes about 0.983-0.985 (G).

## 5. Risks
- LOCO is a two-country proxy; French label noise and decoy vocabulary may differ (legal-form swaps dominate).
- The lift base is a token-weighted rate; if French test tables change (new candidates), recompute bases.
- Lift + parser change 17k French S1 (6.6%); if the upload regresses, the parser-only arm is the safe fallback.

Scripts (C: `~/aml/research/cx/`; copies in the Mac scratchpad `cx/`): `dm_build.py`, `loco_dm.py`, `s3_dm.py`,
`s3_pr.py`, `test_arms.py`, `sim_blank.py`, `audit_ops.py`; logs and `test_arms.json` in `cx/logs/`.
