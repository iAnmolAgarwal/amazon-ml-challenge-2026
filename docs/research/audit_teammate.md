# Audit: the parallel pipeline (P) against ours (O), and what to adopt in the last 26 h

Tags: **M** measured (source named: P = the parallel pipeline's own number, unverified; O = ours), **D** derived (arithmetic on M), **G** guess.
Inputs: the approach summary of P provided by the human (P's code, logs and OOF files were not available), LATEST_SUBMISSION.md, r1, r3, r4, r7, r10, r11, r12 and the leads note (§3, lifted markers).
O = 021_comb (being rebuilt as 023_comb on the GCP machine). As of 26 Sep evening IST. No code was run for this audit.

## 0. Verdict
- **P's summary is internally consistent where it can be checked.** Every LB step it reports falls inside the rounding window of its own calibration. Its loss accounting matches ours almost number for number: ~141k never-retrieved train gold (O: 142,973), the retrieved-but-rejected loss sitting on empty-address chain names (O: the r12 twins), flat decision grids and a harmful hard house-shift veto. I found no first-order leak. Two points could not be confirmed: the fold alignment of the 50%-model feature, and the train/test distribution of the CE scores.
- **On the LB the two pipelines are level.** O 021 is expected at **0.9851** (D: exact 0.98393 + 0.85 x 0.00140; range 0.9849-0.9855). That is below the 0.9855-0.986 in the brief, which needs the test gain to be 1.3-2x the CV gain. P v07 scored 0.984866 (M) and v08 is expected at ~0.98494 (D). **O leads in-country** by about +0.0006 to +0.0010 F0.5 on equal footing (D). **P leads in France** by about +0.0016 to +0.0042 (D). Both statements rest on one shared assumption: the in-country test gap is the same for both pipelines.
- **The value we can adopt is in France, and it is small.** Closing the whole France gap is worth +0.0002 to +0.0006 LB (D). Only one of P's ideas could move us by more than that: an in-country cross-encoder (G +0.0003 to +0.0009), which needs the Kaggle GPU and 8-12 h of wall time.
- **Adopt, in this order:**
  1. The France marker veto, restricted to pure decoy words. Extend it to the dual-use words only if a label-free estimate says they are at least 40% decoys.
  2. A LOCO harness for France extensions: P's no-spec 50/50 stage-2 blend, and our 023_comb applied to France.
  3. Optionally, an in-country e5-small cross-encoder on Kaggle.
- **Reject:**
  - France T=2 / min_p 0.6: it is a shrink, and our France does not over-claim (r3 §7.1).
  - The supporting-record/anchor sidecar: it is S1-side content context, the same family as our trap, and it has no signal on our largest loss bucket (r12 twins).
  - The 50%-sample p2 feature: already covered.
  - Cross-country e5-base for France: no time.
  - The cascade: not LB-scored.
- **Not from P, but it outranks everything above:** our own r10 K9 exact keys plus reverse text retrieval. It measured +0.00123 on the sample (5/5, M), about +0.0012 LB (D).
  - P's recall analysis agrees: half of P's loss is never retrieved.
  - P's rejection "exact-key routes <= 2% of misses" does not transfer to us: our keys catch 34% of misses (M, r10).
  - It should take the first CPU slot after 023.

## 1. Audit of P's claims

### 1.1 OOF vs LB (P: M; O: M/D)
| P version | in-country OOF | public LB | gap |
|---|---|---|---|
| v01 | 0.98082 | 0.970504 | -0.0103 |
| v03 | 0.98182 | 0.975 (3 dp) | -0.0068 +- 0.0005 |
| v05 | 0.98753 | 0.981 | -0.0065 +- 0.0005 |
| v06 | 0.98766 | 0.983 | -0.0047 +- 0.0005 |
| v07 | 0.98773 | 0.984866 | -0.0029 |
| v08 | 0.98782 | ~0.98494 (D, see 1.2) | -0.0029 |
- O's gaps show the same pattern: 008 -0.0085, 009 -0.004, 018 -0.0023, 020 ~-0.0024 (D).
- In both pipelines, the gap shrinks as decoy and France machinery arrives.

### 1.2 Calibration (0.85x in-country, 0.1275x France-LOCO)
- **0.85** is the US+India share of test. It assumes the in-country test gain equals the OOF gain 1:1, and our history agrees:
  - 018 -> 020 (in-country only): predicted 0.98393 + 0.85 x 0.00082 = 0.98463, shown as 0.985.
  - The in-country part of 008 -> 009 fits the same way.
- **0.1275 = 0.15 x 0.85.** It assumes the France gain is 0.85x the US->India LOCO gain. This is a structural assumption, not a slope fitted to many points.
- **Checks against P's own LB (D):**
  - v03 -> v05: predicted 0.85 x 0.00571 + 0.1275 x (0.0102 + 0.0038) = +0.0066; realised +0.005 to +0.007. OK.
  - v05 -> v06: predicted +0.0025; realised +0.001 to +0.003. OK.
  - v06 -> v07: predicted 0.85 x 0.00007 + 0.1275 x 0.01628 = +0.0021; realised +0.0014 to +0.0024. OK.
  - v01 -> v03: predicted >= +0.0014; realised +0.004 to +0.005. **Not checkable**: the v02 stacker with a cross-country CE changed France, and no LOCO number is reported for it.
- **Precision of the calibration.** Every LB except v01 and v07 is shown to 3 decimals, so the France slope is pinned only to about +-40% (0.08-0.14 x LOCO) (D). The one out-of-sample test, v07, came in 0.0002 under its forecast: a slight optimism.
- **The v08 estimate (0.9850-0.9851) is anchored on the v07 forecast.** Anchored on the realised v07 score it is 0.984866 + 0.85 x 0.00009 = **0.98494** (D).
  - The v06->v07 and v07->v08 in-country steps (+0.00007, +0.00009) are too small for our gate to resolve.
  - P's gate is ">= 4/5 folds in both countries", with no standard-error test.
- **Direction choice (mild fit to the public LB).** US->India is named "the calibrated predictor" after at most 4 rounded LB points; India->US is reported but not used.
  - It matters: the veto gains +0.0049 US->India and 0.0000 India->US.
  - US->India is also the over-claiming direction in our own LOCO (r3 §2.1-2.3: a US-trained model over-claims India, and India->US under-claims). Shrink rules (T=2, floors, vetoes) look best exactly in that direction.
  - The mean of the two directions is the fairer France proxy (v07: 0.97036).

### 1.3 LOCO numbers
- The progression matches the version table: v06 0.95088 is the "+ stacker" row; v07 0.96716 is the "+ marker veto" row.
- The veto arithmetic is plausible (D):
  - +0.0049 over ~880k India S1 needs ~12-15k emitted false pairs removed, at 0.19-0.35 F0.5 each (r7 §2.3; singletons are worth 1.0).
  - P vetoed 16k pairs, 28 of them true.
- **Not comparable with our LOCO levels.** Ours were stage-1-type models on the 10% sample (leads note, lift arm: India 0.937, US 0.964). We never ran LOCO on the full 018/021 stack.
  - P's held-out India at 0.967 is a transfer loss of only ~-0.019. That is strong.
  - Production France trains on both countries, so only LOCO differences carry over, not levels.

### 1.4 Leakage and fold hygiene
- **CE halves.** Each pair's CE score comes from a model that never saw its entity, so the scores are clean. The stacker's training rows carry CE scores from models that did see validation-fold entities. This second-order path is the same one we accept for feat4 and for r11's pr_* features, so it is acceptable.
- **CE on test** is presumably the average of the two half-models (P does not say). Averaged scores are smoother than one model's OOF scores, which is a train/test feature shift. P tried and rejected a "half-averaging fix", so P has seen this. It may explain the -0.0002 forecast miss. Could not confirm.
- **lp2b (50% model): fold alignment not stated.** If the 50% entity-sample model scored its own training entities, v05's OOF is inflated. v05's LB rose as calibrated, so any leak is small (G). Could not confirm.
- **Anchors (p2 >= 0.9, OOF).** All pairs of an S1 share its fold, so the anchors are OOF. The concern is test shift, not leakage (§3, item 8).
- **Marker lists and the veto** are computed from France test pairs, label-free. Our feat6 France table is built the same way. Allowed.
- **Stage 2 trains on a 25% entity sample.** This is not a leak, but it leaves data unused (our r6: 5x more data gives +0.0007).

### 1.5 Rule risk
- **Model licences and size.** multilingual-e5-small (118M) and -base (278M) are MIT, bge-reranker-v2-m3 is Apache-2.0 (568M), and multilingual-e5-large is MIT (560M). All fit README constraint 5 (MIT/Apache-2.0, <= 8B). Licences are taken from the model cards and were not re-checked today.
- **"No external data."** Pretrained weights embody external data. The licence clause implies pretrained models are allowed (G). The practical risks are:
  1. The package must regenerate its outputs "using only what is in this folder", so the fine-tuned weights must be bundled or the download documented.
  2. Reproduction needs a GPU (P's e5-base run took ~4.7 h on a T4).
  3. Our Documentation_template and methodology draft say "no pretrained models" and would need updating.
- **Public-LB overfit.** It sits in P's decision process (the calibration and the choice of LOCO direction), not in P's models. Mild.
- **Separate entries.** P's uploads are not counted against our 15, so P looks like a separately registered entry, and the organisers review top packages for fair play.
  - Lower-risk path: re-derive P's ideas in our own code and validate them on our CV/LOCO, as we did with the lifted markers.
  - Not an option: merging P's output rows or code into our package.
  - Could not confirm P's registration. Flagged for the human.

### 1.6 Could not confirm
- Every P number: no code, logs or OOF files were available.
- The exact LBs of v03, v05 and v06.
- P's per-country OOF split and fold scheme (CRC32 hash vs our stratified folds; OOF comparability +-0.0002, G).
- lp2b fold alignment; the definitions of the "decoy2" features.
- Whether the veto was even active in the India->US LOCO (0.0000).
- Whether P's France over-claims (that would explain why T=2 helps P).
- Which French veto words are pure decoys. See §3, item 1.

## 2. Side by side

| | O (021 = 023 recipe) | P (v07 / v08) |
|---|---|---|
| Public LB | 018 0.98393 (M); 020 shown as 0.985 (0.98463, D); **021 expected 0.9851** (D, range 0.9849-0.9855) | v07 **0.984866** (M); v08 ~0.98494 (D; P says 0.9850-0.9851) |
| In-country CV, train-weighted | 0.98758 (US 0.98819, India 0.98667) (M). Route pairs exist only for the 10% owner sample; with the route on every S1, as on test: **~0.9885-0.9889** (D, +0.0009 to +0.0013) | 0.98773 / 0.98782 (M, P); no country split given |
| In-country, test-weighted (India 0.47 / US 0.38) | 0.98735 (D); with the route ~0.9882-0.9886 | ~0.9875-0.9876 (G, assuming O's India-US gap) |
| France | frozen at 018's France (021 keeps France at p018): LB-implied 0.9657-0.9729, central **~0.967** (D) | P's calibrated estimate 0.971; LB-implied 0.9686-0.9699 (D) |
| Stronger in-country | **O, by ~+0.0006 to +0.0010** (D): generator features inside one integrated model (P got only +0.00009 from similar features on top of its CEs), pool-side competition, route, lexical | |
| Stronger in France | | **P, by ~+0.0016 to +0.0042** (D): in-country band stacker transferred to France, decoy-aware no-spec blend, cross-country CE, veto |
| Candidates per S1 | blocking ~45, final band **7.25** (019_band form, M) | blocking ~20.6, cascade **~4.15** (M, P) |
| Blocking recall | 142,973 train gold never retrieved (full train, route on 10% only; M); 98.68% retrieved on the sample with the route (M, r10) | ~141k never retrieved: India 97.5%, US 98.6% (M, P). **Equal recall; P's blocking is 2x more efficient; O adds route pairs on test (+0.94 per S1)** |
| Decision | owner + prefix 5Σp/(Σp+4k) vs Π(1-p), floor 0.6 everywhere (r4/r11: Bayes-optimal, flat) | same algebra (1.25Σp/(k+0.25Σp) is identical); min_p 0.5 in-country (O's r12 floor scan: 0.5 is 0.00003 worse than 0.6); France T=2, min_p 0.6 |
| Loss | never retrieved +0.0043 oracle; exchangeable twins +0.0033 (M) | "half never retrieved, half retrieved-but-rejected, 66-88% of those empty-address chain names" (M, P): same wall |
| Final model | LightGBM, 255 leaves, 800 rounds, 208 features; no pretrained models | LightGBM stacker, 127 leaves, 800 rounds, plus 4 e5 cross-encoders (Kaggle T4) |

How the France gap is derived (D, appendix A): LB = 0.85 x in-country + 0.15 x France; the in-country test score is taken as the test-weighted CV plus the route term for O; P's test-weighted score uses O's India-US gap. So ΔF = [0.000936 - 0.85 x (0.98750 - 0.98588 - r)] / 0.15, which is +0.0016 at r = 0.0008 and +0.0042 at r = 0.00126.

## 3. Ranked insights for our pipeline

Expected LB is shown two ways: first P's calibration applied to P's number (0.85x or 0.1275x), then our estimate for O after overlap.

| # | Insight | Verdict | LB, P calib. / for O | Compute | Risk |
|---|---|---|---|---|---|
| 1 | France marker veto, pure decoy words (dual-use words only if f̂ >= 0.4) | **adopt, gated** | +0.0006 (US->India), +0.0003 (two-direction mean) / **+0.0000 to +0.0002** (pure words), -0.0002 to +0.0005 (with dual-use words) | CPU < 1 h (+1-1.5 h LOCO) | low (pure words); medium (dual-use) |
| 2 | France no-spec 50/50 stage-2 blend, plus our 023_comb applied to France, on one LOCO harness | **adopt what passes LOCO both ways** | +0.0005 to +0.0013 (blend on P's weaker base) / **0 to +0.0005** | CPU 2-3 h | medium |
| 3 | In-country e5-small cross-encoder as a comb feature | **optional (Kaggle)** | +0.0038 (v04->v05 bundle) / **+0.0003 to +0.0009** | T4 ~3-5 h + CPU ~1.5 h; wall 8-12 h | high (schedule, packaging) |
| 4 | Cascade candidate set (p2 > 0.02 and top 10) | skip | 0 (not LB-scored) | - | packaging bug |
| 5 | France T=2 / min_p 0.6 | **reject** | +0.0005 / -0.0003 to +0.0001 | minutes | drops pairs from an under-emitting country |
| 6 | Cross-country e5-base CE for France | reject (time) | +0.0003 to +0.0006 / +0.0001 to +0.0004 | T4 2 x 4.7 h + LOCO | high |
| 7 | Supporting-record / anchor sidecar | **reject** | part of v04's +0.00127 OOF (x0.85) / <= +0.0001 in CV, LB risk negative | CPU 1-2 h | our trap family |
| 8 | 50%-sample p2 feature | reject | bundled / <= +0.0001 | CPU ~1 h | low |
| 9 | v08: all pair features + 4x capacity | already in O | +0.00008 / 0 | - | - |
| - | *(ours, r10)* K9 exact keys + reverse text retrieval | **take first** | / **+0.0012** (D from M) | CPU 3-5 h incl. rewrite | low-medium |

### 1. France label-free marker veto (hard p = 0), restricted to pure decoy words
**What.** After France's final probabilities and before `decide`, on France pairs only:
- **(a) Discovery.** Reuse the n_d / n_0 insertion tables that er_feat6 / er_lift already build from France test pairs. Markers are words with n >= 500 and lift = [n_d/(n_d+n_0)] / [Σn_d/Σ(n_d+n_0)] >= 1.5 (P's rule).
- **(b) Purity split (ours).**
  - Pure word: S1 count about 0 and pool count at the common decoy-injection level. r1 §2.6 (M): participations 34,279 pool vs 1 S1, holding 33,983 vs 17, distribution 34,205 vs 64, international 34,571 vs 203.
  - Dual-use word: pool count above that level after base-name echoes. groupe 53,799 vs 2,072 S1 and developpement 48,013 vs 815 S1 leave about +11-13k insertions beyond ~34k (D). associes has 10,616 vs 25 on its own. france is not counted in our notes.
- **(c) Veto.** A France pair whose pool name contains a pure marker absent from the S1 name gets p = 0.
- **(d) Dual-use words.** Veto them only if the estimated decoy share f̂ among our emitted insertions is >= 0.4. The break-even is 0.2-0.28 (D: removing a false pair gains 0.19-0.3, removing a true one costs ~0.074). Estimate f̂ as follows:
  - On train (labels allowed): the share of decoy-word insertion decoys that keep the S1's house (marker-only decoys), per country.
  - On France (label-free): split the insertion count into decoy (the pure level) and filler (the excess), with emission rates by house relation from our France output.

**Evidence.**
- P: US->India +0.0049 with 28 true lost of 16k vetoed; India->US 0.0000 (M, P). P's list includes groupe, developpement and france.
- O, covered already:
  - Lifted markers: LOCO +0.0086 (5/5, M).
  - The lexical layer already removes 2,537 French pairs, 27% of them groupe insertions, with house conflict 5.5% vs 1.0% kept (r3 §7.2, M).
- O, safety in-country: US/India decoy words are never inserted in true matches (group 0/543, holdings 0/498 per 10k pairs; 112 decoy-only words confirmed out-of-sample; r1 §2.3, M). A veto therefore costs ~0 true pairs in-country, so the in-country no-harm check is effectively pre-answered. Rerun it on the 023 OOF for the paired number (15 min).
- O, value (G):
  - Under 006, emitted French insertions were fils/et/groupe/france/developpement/cie/associes (4.3% of emitted pairs, r1 §2.10, M). The pure words are not on that list, so our model probably emits few of them already, and the pure-word veto may remove little.
  - r3 §7.3 (M): same-address one-word insertions are 26 per 1k French S1, emitted at mean p 0.88.
  - The real stake is the dual-use words, where the answer depends on the unknown share of marker-only decoys.

**Expected LB.** P: +0.0006 (one direction), +0.0003 (two-direction mean). O: pure words +0.0000 to +0.0002; with dual-use words -0.0002 to +0.0005 (G).
**Effort.** CPU, minutes for the veto and the diagnostics; +1-1.5 h if a US->India LOCO is run with a US-only band model (016 recipe, **feat4 excluded**, as r3 did, because stage-1 OOF saw target labels). Needs a small src step (human approval).
**Risk.** Low for pure words. For dual-use words: if f̂ is wrong by half, it costs up to ~-0.001 LB (D, worst case: ~24k true filler insertions vetoed).
**Guard.** France emitted pairs/S1 should stay >= ~3.2; list the vetoed emitted pairs per word, with their house-conflict and decoy-shift shares vs kept pairs (1.0% / 0.1%, r3 §7.2).

### 2. France LOCO harness: no-spec 50/50 blend, and 023_comb applied to France
**What.**
- **(a) No-spec blend.** Fit the 016 band model without the size-dependent "spec" columns: feat2 name-frequency counts, feat5 name/address specificity IDF masses, skeleton chain size. Keep every decoy column (house relations, hd shift, feat6l lifted markers). For France: p' = σ(0.5 logit p_spec + 0.5 logit p_nospec). P found weight 0.25 on spec worse than 0.5.
- **(b) Comb on France.** On the same harness, train 023_comb on one country and apply it to the other, full and drop-only (r11 s3loco protocol). This decides whether France should get the comb correction; today it keeps p018.
- **(c) Validation and shipping.** Run LOCO in both directions, paired by target fold, with feat4 excluded from all arms. Ship an arm only if it wins 5/5 folds in both directions (or 4/5 with > 2 SE). Then run the France guard.

**Evidence.**
- P: blend +0.0102 / +0.0038 at v05; "decoy-aware no-spec stage 2 + decoy2" +0.0088 / +0.0093, both directions (M, P). O's band model has been decoy-aware since 016, so only the no-spec half is new.
- O: count features are raw per split (r4 §2h, M). France has 259k S1 against 1.32M / 0.88M train S1, so French names look up to 5x rarer. Our only size test (r3 recipe c) was neutral, but US vs India spans only 1.5x (M), so it could not probe a 5x gap.
- O: r11 last-layer LOCO: full +0.00021 / +0.00012, drop-only +0.00005 / +0.00005 (M), so (b) is likely small.

**Expected LB.** P: +0.0005 to +0.0013. O: blend 0 to +0.0005, comb on France 0 to +0.0002 (G).
**Effort.** 4-6 band fits on half the data at 12 threads (~10-20 min each), decisions and code: 2-3 h CPU.
**Risk.** Medium. LOCO is a two-country proxy; the harness also serves item 1's LOCO.

### 3. In-country cross-encoder (e5-small, entity halves) as a comb feature
**What.**
- Model and settings: fine-tune multilingual-e5-small on "name ; address" pairs (P's settings: 1 epoch, max_len 96, lr 5e-5, batch 128, fp16 on T4).
- Halves: two halves that are unions of harness folds, e.g. folds {0,1,2} vs {3,4}. Each half scores the other; test gets the mean of the two models.
- Features: add the CE logit and **pool-side context only** (rank among the pool record's S1, margin vs the best other S1) to 023_comb.
- **Drop P's within-S1 rank/gap/margin of the CE.** That is S1-side context, the r4/r11 "PS" family that failed our guard.
- Accept on paired CV (> 2 SE, >= 4/5) plus guard3.

**Evidence.**
- P: v04 -> v05 +0.00444 OOF, bundled with lp2b and the France blend (M, P). P's in-country with CEs is 0.9878.
- Overlap: P's v08 generator-operator features added only +0.00009 on top of the CEs (M, P), and O without CEs is already ~0.9882-0.9886 (D). The overlap is heavy in at least one direction.

**Expected LB.** O: +0.0003 to +0.0009 (G).
**Effort.**
- Export: ~3M train + ~1.7M test band text pairs (~1 GB). Export them as soon as 006 exists, using a band wider than comb's.
- Kaggle T4: ~3-5 h for two e5-small halves plus scoring (G, scaled from P's ~4.7 h per e5-base run).
- CPU integration and CV: ~1.5 h.
- e5-base does not fit in the time.

**Risk.** High:
- Schedule: nothing is usable before ~15:00 IST 27 Sep.
- Packaging: GPU steps and weights in the zip, and the documentation changes.
- Gain unknown for O.
- Hard stop: if the feature file is not in hand by 17:00 IST, drop it.

### 4. Cascade candidate set (p2 > 0.02 and top 10, ~4 per S1)
- candidate_pairs.tsv is **not scored**. It is used to analyse the recall ceiling and reduction ratio (README).
- O's r6 already measured the same trade: stage-1 p >= 1e-3 and top 10 gives 5.85 per S1 at 0.00000 loss; p >= 3e-2 gives 4.09 per S1 at -0.00001 (M).
- O's 7.25 per S1 keeps the higher recall ceiling.
- Changing it on the last day risks a candidates/matches mismatch in the package. Skip.

### 5. France decision T=2 / min_p 0.6: reject
- p' = σ(logit p / 2) with min_p 0.6 means an effective raw floor of 0.692, plus flatter prefix and empty scores. It is a shrink.
- P measured it only US->India (+0.0039), which is the over-claiming direction (§1.2).
- O's France on 018 does not over-claim: mass excess +0.014, and after ownership it emits below US/India (3.26 vs 3.33-3.36 pairs/S1). The r3 §7.1 analogue (T - 0.1) is about -0.002 France (G).
- Every fixed floor failed across LOCO directions (r3 §2.2-2.3, M). In-country grids are flat for both pipelines.

### 6. Cross-country e5-base CE for France: reject for this deadline
- P: +0.0026 / +0.0044 LOCO (M, P).
- It needs two e5-base models (~4.7 h each on a T4) plus a France stacker validated by LOCO. It does not fit in 26 h next to the rebuild.

### 7. Supporting-record / anchor sidecar: reject
- **Is it our trap?** Partly. It is S1-side content context: the similarity of a pool record to the S1's other records with p2 >= 0.9. That breaks our rule "every feature compares one pool record with its S1 only".
- **What the p >= 0.9 gate does.** It blocks the cluster mechanism that killed coherence, where decoys support decoys. On test, shared-deviation candidates above 0.9 barely rise (US 0.121 -> 0.133 per S1, r7 §2.4, M), so anchors are mostly true.
- **What it leaves.** Decoys are near-copies of true records and copy their source-level noise (r7 §4, r9), so a decoy looks like the anchors. At 2x test decoy density, "looks like the entity's other records" is wrong more often than on train. That is the milder trap, the same direction as r4's S1-context stacker.
- **Where it would have to pay.** On the empty-address twins, the biggest reducible bucket, the local analysis found no anchor signal (r12, M):
  - similarity to the twin's other records: top-1 0.231 vs chance 0.232;
  - exact name equal to an anchor: 0.167 vs 0.163;
  - name noise shared with the owner's records: 4.2% vs 3.6%.
- **P's evidence.** The v04 sidecar (+0.00127 OOF, bundled with address evidence O already has) was never uploaded alone. Its LB effect is confounded with v05's CEs.
- **If ever tested:** run r4's alarm (test/train net-added pairs per S1 about 1x) and r7's clone_dep simulation.

### 8. 50%-sample p2 feature: reject
- P's main stage 2 uses a 25% entity sample, so a 50% model adds data P left unused. O's stage 2 and band models already train on full folds.
- Diversity features have been flat for O: 2 seeds +0.00001 and an XGBoost blend +0.00001 (r11, M).
- Fold alignment is also unconfirmed (§1.4).

### 9. P's v08: all pair features and 4x capacity: already in O
- 021 has all 100 base band features plus gf1-6 at 255 leaves / 800 rounds. The r12 capacity step gave +0.00005 (M).

## 4. Recommendation for the next 26 h (one 12 vCPU / 96 GB machine, optional Kaggle T4)

Principles: the rebuild chain is the critical path, so run only light jobs beside it. Decide France changes on LOCO plus the guard, never on the LB: 3-decimal LB rounding hides +-0.0005, and France steps are about +0.0002. Five uploads are left on 27 Sep; the plan uses at most four.

| step | when (IST) | what | gate | expected LB after |
|---|---|---|---|---|
| 0 | tonight -> ~08-10:00 | Rebuild -> 019_band -> er_gen -> 023_comb. Write the code for steps 1-2 while the chain runs (no CPU needed). | 023 vs 018 paired ~+0.0014 (5/5), guard3 clean, validator PASS | **0.9851** (D; 0.9849-0.9855) -> upload 12 |
| 1 *(ours, r10)* | ~10:00-14:00 | K9 exact keys (p6n, njn, p8, sk, fl, skh, tt; th India; tn US; key value <= 50 S1 and <= 50 pool) + reverse char-3-gram "name \| address" top-5 from unclaimed pool records, scored by the rebuilt route models (r10 §4). The r10 code was on AWS; the spec is exact enough to rewrite. | sample paired ~+0.0012 (5/5; r10 had +0.00123), route/K9 guard | **~0.9863** (D/G) -> upload 13 |
| 2 *(P insight 1)* | beside step 1 (light CPU) | Pure-word France veto; per-word purity and f̂ for groupe / developpement / france / associes; add dual-use words only if f̂ >= 0.4. | in-country no-harm on the 023 OOF; France guard (pairs/S1 >= ~3.2, vetoed pairs decoy-enriched) | **~0.9864** (+0.0000 to +0.0004, G) -> upload with step 3, or alone |
| 3 *(P insight 2)* | ~14:00-17:00 | LOCO harness: no-spec blend and comb on France (both directions, feat4 excluded); the same harness gives item 1's LOCO if wanted. | both directions 5/5 (or 4/5 with > 2 SE); France guard | **~0.9866** (G, 0 to +0.0005) -> upload 14 |
| opt *(P insight 3)* | Kaggle, parallel | e5-small CE halves on the band text exported after 006; integrate into comb only if the feature file is ready by 17:00 | paired CV > 2 SE, 4/5; guard3; weights and notebook in the package | **+0.0003 to +0.0009** (G) |
| final | 20:00 | Package the best build whose steps each passed CV/LOCO and the guard and whose code is in src (fallback: 023, then 019_band). | validator PASS; identity scan | - |

- **P's ideas only, without step 1:** 0.9851 -> ~0.9853 (veto) -> ~0.9855 (LOCO-passed France extension) -> ~0.986 with the CE (all G).
- **Expectation to share:** 021/023 and P's v08 should land within about +-0.0003 of each other on the LB (D). Our in-country lead and P's France lead cancel.

## 5. P's measured rejections that save us time (P: M)
| P rejected | Relevance to O |
|---|---|
| In-country T / min_p grids: 0 | Confirms r4/r11. Do not revisit. |
| Hard house-shift veto (-0.0005 to -0.0045 LOCO) | Confirms ours (India template houses). |
| Count-prior competition features: 0 | Confirms r12 e9 (+0.00001). |
| Utility-weighted stacker (-0.00012); domain importance weighting; dedicated HIGH model; France stacker bags | New to us; skip them. Seed/XGBoost blends were flat for us too. |
| Wider blocking by k (k_a 40 recovers 19% of misses for +18.8 cand/S1; k_r 8 recovers 14%) | Confirms r10 (forward text top 40/100 too costly). Go for targeted keys instead. |
| **Exact-key routes <= 2% of misses** | **Does not transfer.** O's K9 catches 34% of misses and gains +0.00097 (5/5, r10, M). |
| Char-trigram retrieval (+0.27% recall) | Does not transfer: P's blocking already has pool->S1 reverse top-3. O's route gains +0.00126 (M). |
| Lexical token tables | Does not transfer: O +0.00016 in-country (5/5) and France LOCO +0.0037 (r3 §7.2). |
| Count-free / rate features | Neutral for O too (r3 recipe c); the France 5x size gap stays untested, which is item 2. |
| Spec weight 0.25 in the France blend | Use 0.5 if item 2 is run. |
| Two CE features, e5bca, CE half-averaging fix, French-style CE augmentation | If item 3 is run: one e5-small CE, no averaging fix, no synthetic French pairs (our r3 self-training also failed). |
| In-country decoy-aware stage 2 as a feature: neutral | O's band model is already decoy-aware; no separate decoy model. |

## Appendix A: arithmetic (D)
- **021 expected LB:** 0.98393 + 0.85 x 0.00140 = 0.98512. Through 020: 0.98463 + 0.85 x 0.00058 = 0.98512.
  - 0.9855 needs a test/CV gain ratio of 1.3; 0.986 needs 2.0.
- **O in-country, test-weighted:**
  - 018: (0.47 x 0.98502 + 0.38 x 0.98695) / 0.85 = 0.98588.
  - 021: (0.47 x 0.98667 + 0.38 x 0.98819) / 0.85 = 0.98735.
  - Route term r = +0.0008 to +0.00126: +0.00126 was measured on 016 sample owners; the lower bound allows for more route false positives at 2x test decoys.
- **O France (018 = 021):** (0.98393 - 0.85 x (0.98588 + r)) / 0.15 = 0.9729 (r = 0) to 0.9657 (r = 0.00126); ~0.967 at r = 0.001.
- **P France:** (0.984866 - 0.85 x I) / 0.15 = 0.9699 (I = 0.98750, test-weighted G) to 0.9686 (I = 0.98773).
- **Veto value per 1k France S1:** ΔF ≈ (E/1000) x (0.2 f - 0.074 (1 - f)), with E the emitted pairs removed and f their decoy share.
  - Break-even f ≈ 0.27.
  - E = 45, f = 0.45: +0.0022 France (+0.0003 LB).
  - E = 45, f = 0.2: -0.0009 France (-0.0001 LB).
