# Audit 10: next iteration after submission 4 (006_stage2, LB 0.963)

Auditor: independent (Opus), 2026-09-25 ~16:15 IST. Brief: `10_next_iteration_prompt.md`, section 6.

**Inputs.** Raw train/test parquet, `submissions/matching_results.tsv` (sub-4) and the archived sub-1/2/3 TSVs, `attempts/*/entity_scores.parquet`, and `cache/oof/002_namefreq.parquet` (the only pair-level OOF on the Mac).

**What was touched.** EC2 was not touched. Nothing in `src/`, `attempts/` or git was changed. Test data was used label-free only.

**Scripts** (scratchpad `audit10/`):
- `buckets.py`: loss buckets.
- `normall.py`: `er_norm.normalize_rows` over all France/US/India test rows and US/India train rows.
- `tokdiff*.py`, `frcount.py`, `structprof.py`, `decoys2.py`, `suffix2.py`, `nearconf.py`, `missedgold.py`, `housep*.py`: France and decoy diagnostics.

Peak RSS stayed under 2 GB.

"tw" means test-weighted: 0.47 x India + 0.38 x US. France gets its own term where it is stated.

## 0. Bottom line

- **0.99+ is not reachable, in the next submission or by the deadline.** With perfect decisions on the current candidates, India and US are capped at their oracles (0.9918 / 0.9954). They then contribute 0.8444, so France would need 0.971 with no test shift at all. Add the observed LB-CV gap of -0.011 and France would need more than 1.0. Realistic targets: next submission **0.965-0.967**, final **0.967-0.972**.
- **74% of what CV loses is missed matches, not false positives.**
  - Under-prediction on non-empty answers: 0.0119 tw.
  - Empty answers on entities that do have matches: 0.0045.
  - Together that is 0.0164 of the 0.0224 tw lost.
  - 86% of the under-predicted entities miss exactly **one** match.
  - Many-match entities (t >= 6) hold only 0.0018. Singletons hold 0.0013.
- **The biggest cheap lever I found is not in the brief: a house-number relation feature.** 24% of US gold pairs with a house conflict are "front digit dropped" (1995 vs 995). Among negatives that are hard to reject, the same pattern is 0.3%. "Nearby number on the same street" is the reverse: 40% of those negatives and 6% of gold conflicts. Today only `house_prefix` exists.
- **France.** The label-free evidence does not support a large France-specific failure.
  - On every agreement metric, France's emitted pairs look like US gold pairs.
  - The France k histogram almost reproduces the train t histogram.
  - Same-street, nearby-number decoys are **half** as dense in France as in US/India test.

  I did find real France normalization bugs. The largest: the France-gated house fallback loses the house number on **8.2% of French pool records** (118k). Measured through the OOF proxy, though, each bug is worth only tenths of a point of France recall. A second France item: France returns an empty answer for 4.99% of S1, against a train singleton rate of 5.58%. If the singleton rate is the same in France, singleton over-answering alone costs France 0.006-0.010. Item d's "+0.01-0.05 France" is not supported by anything measurable. Plan for France +0.003-0.012, i.e. LB +0.0005-0.002.
- **The implied France 0.90 rests on an untested assumption: that US and India score on test exactly as in CV.** US and India emit about 0.04-0.05 more pairs per S1 on test than in CV (sub-2 and sub-4 both). One extra false positive per S1 costs about 0.25 macro F0.5 (measured on the 006 entity scores). So 0.045 extra false positives per S1 would by itself explain the whole -0.011 gap without France. The truth is somewhere between "France 0.90" and "France ≈ US". A 10-minute diagnostic on the test probabilities (§4.2) can narrow it.

## 1. Is 0.99+ reachable? Realistic next and final LB

**Formula:** LB = 0.47 x India + 0.38 x US + 0.15 x France.

**Ceiling arithmetic**
- Current candidates, perfect decisions: 0.47 x 0.9918 + 0.38 x 0.9954 = 0.8444. For LB 0.99, France needs (0.99 - 0.8444) / 0.15 = **0.971**, with India and US at their oracles on test.
- With the observed test gap applied to India and US (-0.011): France would need (0.99 - 0.47 x 0.9808 - 0.38 x 0.9844) / 0.15 = **1.03**. Impossible.
- Even with a perfect France (0.15), India and US would need 0.84 / 0.85 = 0.988 **on test**, i.e. about 0.999 CV. The current oracle is below that.

Verdict: not reachable. The realistic ceiling for this pipeline family in the remaining ~50 h is about 0.975 LB.

**Realistic next submission** (sub-5/6, v3 blocking + the bundle in §2):

| component | CV / tw gain | LB |
|---|---|---|
| sub-4 | | 0.963 |
| v3 blocking (running) | +0.001-0.002 | 0.964-0.965 |
| general bundle (house relation, alias fix, admin-area canon, skeleton features) | +0.001-0.002 | |
| France-gated fixes | France +0.003-0.012 | +0.0005-0.0018 |
| **next (with bundle)** | | **0.965-0.967** |

**Realistic final:** add stage-2 upgrades or a blend (+0.001-0.003) and seed averaging (+0.0003). Final is **0.967-0.972**.

**By country, final (two readings of the current gap).**
- **Reading A (France-only gap).** India 0.9721 -> ~0.976, US 0.9757 -> ~0.979, France 0.902 -> ~0.91. LB 0.47 x 0.976 + 0.38 x 0.979 + 0.15 x 0.91 = **0.968**.
- **Reading B (uniform test shift of -0.011).** India ~0.965, US ~0.968, France ~0.965 on test. LB also about **0.967**.

The plan is the same under both readings. Only the France-specific strictness question differs (§3.3).

## 2. Where sub-4's remaining CV loss sits (006 entity scores, 2,206,821 S1)

The total tw loss is 0.02237 (India mean loss 0.02793, US 0.02433). With France at the average, the full-weight equivalent is 0.0263.

| bucket | entities | tw loss | retrieval part (1-oracle) | decision part (oracle-F) |
|---|---|---|---|---|
| singletons (t=0; 3,289 answered wrongly) | 123,247 | 0.00133 | 0 | 0.00133 |
| matched, empty answer | 11,485 | 0.00446 | 0.00125 | 0.00321 |
| matched, all emitted pairs wrong | 196 | 0.00008 | 0.00002 | 0.00006 |
| **under-prediction only (no FP)** | 369,546 | **0.01192** | 0.00420 | **0.00771** |
| over-prediction only (no FN) | 33,812 | 0.00319 | 0 | 0.00319 |
| mixed FP+FN, plus 1.66M perfect entities | 1,668,535 | 0.00140 | 0.00011 | 0.00128 |
| **total** | | **0.02237** | **0.00558** | **0.01679** |

- Missing matches per under-predicted entity: 1 in 317,270 cases, 2 in 45,094, 3+ in 7,180. The loss is the **last noisy match**, not big clusters.
- Pair-level totals: 459,008 FN pairs vs 45,329 FP pairs on matched entities, plus 3,708 on singletons.
- By t, tw loss (gain to oracle):

  | t | tw loss (gain to oracle) | mean k-t | share of entities retrieval-limited |
  |---|---|---|---|
  | 1 | 0.00396 (0.00302) | | |
  | 2 | 0.00444 | | |
  | 3 | 0.00465 | | |
  | 4 | 0.00377 | | |
  | 5 | 0.00239 | | |
  | 6+ | 0.00184 (0.00132) | -0.41 | 10.9% |

- Oracle gains (tw): all singletons right +0.00133; empty-matched to oracle +0.00321; under-predicted to oracle +0.00771; t=1 +0.00302; t>=6 +0.00132; everything to oracle +0.01679.
- Which gold pairs are missed (002 OOF, the only pair-level OOF on the Mac; 006 will be better but the shape should hold):

  | pair type | emit rate at p > 0.6, US | India |
  |---|---|---|
  | pool address empty | 0.56 | 0.59 |
  | house conflict | 0.76 | 0.88 |
  | pool house missing | 0.98 | 0.98 |
  | house equal | 0.995 | 0.98 |

  Empty addresses and house conflicts are the two recall sinks.

## 3. France (label-free)

### 3.1 What French records look like (259,452 S1, 1,434,993 pool; sub-4 predictions)

**Names**
- Built from a small vocabulary of generic French words (club, école, amicale, comité, collège, maison, sportive, pharmacie...) plus short distinctive tokens (initials such as "QGO", city names, synthetic brands).
- 47.6% of S1 share their exact space-free name with another S1 (US test 39.9%).
- Legal forms: SARL, SAS, SASU, EURL, SA, SCI, SNC, EI. They also appear dotted (S.A.R.L., 5.2% of pool) or bracketed ([SARL], 1.7%).
- Connectors: &, + and "et" (1.1% of pool).
- Benign suffix noise that the generator adds to pool names: Cie/Compagnie, Ets, Société, & Fils, & Frères, Associés, Services, "(France)". Their assigned rates are 0.56-0.65, the same as the base rate of 0.62.
- Decoy markers: Holding (assigned 2.4%), Participations (1.6%), International (3.3%), Groupe (25%), Développement (20%). US analog in train: Holdings 13%, Group 31%, Ltd 28%, Co 35%, Partners 41%. **Do not strip these. The model already rejects them in France.**
- Other noise operators, all seen in the pool:
  - random Ç/À/È injected at word start (Çlub, Làcademie);
  - quote junk;
  - "<<" markers (also in S1);
  - aliases: DBA:, formerly:, t/a, a/k/a, f/k/a, "doing business as", née;
  - website forms (x.com, 3.5%) and handles (@x, #x, 0.5%).

  In website and handle forms, **accented letters are deleted, not folded**: "frres", "cole", "collge", "fdration". 11.9k records.

**Addresses**
- House templates in the pool:

  | template | share of pool |
  |---|---|
  | N° n / Nº n / No. n | 11.5% |
  | # n | 3.3% |
  | (n) | 1.6% |
  | "n -" | seen, share not counted |
  | zero-padded 00n | 3.3% |
  | n bis/ter | 3.8% |
  | n B / n t | seen, share not counted |

- Street types abbreviated: R./R (24.8% of pool), Av/Ave/Bd/Blvd/Pl/Imp/All/Rte (11.7%), Ch./Chem., Crs, Q./Qu, Pass/Psg, Rés.
- Typos: Aveune, Bousevard, Chmin, Blcd, Roue. Plurals: Allées. Source codes that also appear in S1: COUR2, CHEM1, VENE1.
- Components are rotated: "Tourcoing, Nord, 28 R. ...".
- Admin area: S1 always carries the **region** (Hauts-de-France / Nouvelle-Aquitaine / Pays de la Loire). The pool carries the region in 33%, the **department** (Nord, Pas-de-Calais, Gironde, Loire-Atlantique) in 32%, and neither in 35%. This is the France analog of US state code vs state name.
- City aliases: St-/St./Saint-.
- Apostrophes are straight and curly (d', l').
- Empty or "None" addresses: 3.0% of pool.
- CEDEX/BP: 0.1%. 5-digit postcodes: 0.5%. **Postal features are dead for France.**

**Structure**
- 58.8% of France S1 share house+street with another S1 (US 23%, India 54%). France is India-like in addresses and US-like in script.

### 3.2 Where `er_norm` mishandles French (measured with `normalize_rows`)

Ranked by estimated impact, with the number of French records affected:

| # | defect | France records | evidence |
|---|---|---|---|
| 1 | **France house fallback drops the house number.** `FR_STREET`-anchored regex; fails on "(4) RUE", "23 - RUE", Q./Crs/Ch., typo'd or unlisted types, plurals, "Grande Rue", "Le/La <lieu-dit>" | **118k pool (8.2%)**, 3.2% of S1 | the generic first-number path recovers 98.6% of them. Pool house present: France 83.7% of emitted pairs vs US 88.3%. Also breaks `street1` (anchor), `k_hs`, `k_njh`, `k_ih` |
| 2 | Department vs region never reconciled | 457k pool (31.8%) | lowers `a_tset`/`word_sim`; `street1` fallback picks region/city tokens when house is missing |
| 3 | Web/handle forms: English-only `strip_web_legal`; accent deletion; handles not flagged `is_web` | 46k pool (3.2%): 13.9k glued French legal ("qonlyceesarl", "mesclubsas"), 11.9k accent-deleted, 7.8k handles | the skeleton does not rescue "cole"/"ecole" (first letter differs) |
| 4 | "N°" -> token "n" -> `ADDR_ABBR` "north" | 70k pool (4.9%) | spurious token; appeared as `street1` in 46 of ~4.2k street mismatches in the 27k-pair sample |
| 5 | Street abbreviations missing from `FR_ABBR`/`FR_SKIP`: ch, crs, q, pass/psg, res, b (bis), t (ter), cour | ~20k pool (1.4%) | top `street1` mismatches in emitted pairs: ch 122, b 90, crs 61, q 59, res 18, psg, t |
| 6 | EI not in `LEGAL` | 16.4k pool, 4.2k S1 | "maison familles ei" |
| 7 | "et" not treated as "and" | 15.2k pool | "harmonieetfils" vs "harmoniefils" |
| 8 | Alias forms missed by `ALT` (**all countries**): f/k/a, a/k/a, n/k/a, fka, nka, "doing business as", née | France 4.2k; test 35.6k; train 51.1k (US 36.9k, India 14.2k) | core becomes "belonex f k a kung club"; `name_alt` stays empty |

**How much #1 is worth.** In the 002 OOF, gold pairs whose pool house is missing are emitted 97.8% of the time (US) vs 99.5% when the house is equal. Negatives with a missing house are emitted 0.03%, so a missing house does not create false positives. Unassigned French pool records that match an S1 exactly on name+street but have a missing house: 9.3 per 1000 S1, against 0.8 in US test. That makes #1 worth roughly 0.2-0.5 pp of France pair recall, i.e. **France +0.002-0.005**. The other items are smaller. **Sum of all eight: France +0.003-0.008.**

### 3.3 France profile vs US/India (sub-4, test)

| | France | US | India | train gold US | train gold India |
|---|---|---|---|---|---|
| k per S1 | 3.448 | 3.329 | 3.316 | t̄ 3.459 | t̄ 3.465 |
| empty rate | 4.99% | 5.89% | 5.85% | singletons 5.58% | 5.59% |
| name token_set (emitted pairs) | 91.9 | 93.0 | 93.9 | 92.7 | 93.4 |
| addr token_set | 92.4 | 91.2 | 94.1 | 89.1 | 92.3 |
| exact space-free name | 0.73 | 0.62 | 0.74 | 0.61 | 0.72 |
| street1 equal | 0.85 | 0.89 | 0.68 | 0.87 | 0.66 |
| house: prefix/suffix / near <= 20 / other | 0.2% / 0.6% / 0.5% | 3.9% / 0.6% / 1.9% | 3.5% / 1.0% / 10.0% | 4.3% / 0.6% / 2.3% | 3.8% / 0.9% / 10.3% |
| unassigned same-name, same-street, near-house decoys per 1000 S1 | 209 | 415 | 78 | 214 (train) | 39 (train) |

**Reading**
1. France emitted pairs look like gold pairs. No FP signature is visible.
2. France has almost no house-number noise (the unassigned side is equally clean: prefix/suffix 3.1 per 1000 S1 vs 13.8 in US test). The generator applies fewer house operators in France. **The brief's France house-conflict of 1.8% is not comparable with US.** It is deflated by defect #1 and by the rarer noise. It is not a decoy guard for France. Use "near <= 20" instead.
3. France sibling decoys are at train density, not 2x.
4. The France k histogram, k = 0..7+, is [.050 .062 .173 .241 .217 .144 .073 .039]. The train t histogram is [.056 .054 .170 .240 .219 .146 .075 .040]. US/India test k histograms match the CV k histograms instead (under-predicting).

   So France either recalls better than US at the same precision, which contradicts 0.90, or over-emits by about 0.12 pairs per S1. The one piece that is solid arithmetic is the emptiness gap. If France's singleton share is 5.58% and matched-empty is 0-0.5%, France answers wrongly on 0.6-1.1% of S1 that are singletons. The CV figure is 0.15%. **Excess France loss from singletons: 0.006-0.010.**

### 3.4 How to diagnose and validate an unseen country in a few hours

**1. Parse-rate audit (done above; 30 min).** Run `normalize_rows` on France and US test. Compare house/street1/alt/web/legal parse rates, and compare view agreement on emitted pairs.

**2. Fix the defects in §3.2.** France-gated rules (#1 and #3-#7): about 2 h of code. #2 and #8 are general.

**Rules to implement**
- **House, France.** Take the first 1-4 digit number, allowing `(`, `)`, `-`, `#`, `°`, `º` and "N°/Nº/No" before it. Skip numbers after appt/apt/étage/bât/bloc/esc/cedex/bp/cs and 5-digit numbers. Keep the `FR_STREET` anchor only as a tie-break when several numbers are present. Keep the bis/ter/quater and b/t suffixes out of the house value.
- **N°.** Drop "n"/"no" tokens directly before a number in France. Never map "n" to "north" there.
- **FR_ABBR/FR_SKIP.** Add ch, chem, crs->cours, q->quai, pass, psg->passage, res->residence, b->bis, t->ter, cour, allees->allee, chaussee.
- **Admin area.** Map the department to its region (nord and pas de calais -> hauts de france; gironde -> nouvelle aquitaine; loire atlantique -> pays de la loire), or drop both from `addr_norm`/`text`. Do the same generally for US state names -> codes: pool full name about 50% vs S1 code (CV-measurable).
- **Names.** Add EI/E.I. to `LEGAL`. Treat "et" and "+" like "&". Do the French legal strip inside website/handle tokens (sarl, sasu, sas, eurl, sci, snc, ei, sa). Treat @/# handles as web. For France, also key the website form against an S1 join built with accented letters **deleted**.
- **Aliases (all countries).** `ALT` += `[afn]/k/a`, `fka`, `nka`, `doing business as`, `n[ée]e`.
- **Do not strip** Holding/Participations/International/Groupe/Développement. They are decoy markers.

**3. Validate without labels**
- Parse targets on France test: pool house present >= 97% of addresses with digits (now 91.8%); `street1` agreement on sub-4 emitted pairs 0.85 -> >= 0.88.
- Silver pairs: sub-4 pairs with identical name_join and the same street. Their `house_eq`/`street_eq` should rise. On silver negatives (unassigned near-house siblings), `near<=20` must stay a conflict.
- Diff audit after predict: hand-check 100 France S1 whose answer changed. Accept if >= 80% of changes look right.
- Guards: France k/S1 3.45 ± 0.05; empty rate >= 4.9%; near <= 20 share of emitted pairs <= 0.8%. Compute before/after guards **with the same (fixed) extractor**.

**4. Leave-one-country-out (LOCO), only if §4.2 shows France is over-confident**
- Train stage-1 on US only and score India; then the reverse.
- Measure the unseen-country drift in Σp vs actual, k/t, singleton accuracy, and the best empty ratio r.
- Adopt a France-only strictness (ratio r on France) only if both directions agree and the paired gain on the held-out country passes the 2 SE / 4-of-5 rule.
- Cost: about 2 h of EC2.
- Caveat: India->US is the closer analog of France's script, US->India of France's address sharing.

## 4. Recommended combined iteration

### 4.1 Order and budget (one machine; stage-2 CV peaks at 62 GB, so jobs run one at a time)

| slot | job | wall | gate |
|---|---|---|---|
| now to ~21:30 | 007a/007 v3 blocking CV -> test predict (already queued) -> **sub-5** | running | paired CV (> 2 SE, >= 4/5); guards |
| now, Mac, in parallel | code bundle B1 + France rules; unit-test on Mac samples; France parse targets | 3 h | parse targets met |
| ~21:30, in a gap | Diagnostics D1 (§4.2), EC2 read-only | 0.3 h | |
| B1 train | rebuild train (1.2 h) -> stage-1 CV full (0.9 h) -> stage-1 CV excluding {house_rel, skeleton feats} (0.9 h) -> feat4 -> stage-2 CV (1 h) | ~4.5 h | stage-2 paired vs 007; the exclusion run attributes features vs normalization |
| B1 test | rebuild test (1.2 h) -> stage-1 predict -> feat4 -> stage-2 predict (1.5 h) -> **sub-6** | ~3 h | guards below |
| cycle 2 | stage-2 upgrades or blend (§4.3) -> **sub-7** | ~5-6 h | paired CV |
| last day | seed-average stage-2 (2-3 seeds), final pick, packaging | ~4 h | |

**Bundle B1 (one rebuild), CV-attributable parts**
- **house_rel**: categorical (equal / prefix / suffix / edit-1 / numeric diff <= 20 / other / missing). Gold conflicts vs confusable negatives, US: suffix 24% vs 0.3%; near <= 20 6% vs 40%. India: suffix 9% vs 2%. Expected **+0.0005-0.001 tw**.
- **ALT fix**: 51k train pool records. +0.0002-0.0005.
- **Admin-area canon**: US state names, France departments. +0.0002-0.0008 (guess).
- **Skeleton fuzzy features (b)**: +0.0003-0.001.

**Bundle B1, France-gated rules** (#1, #3-#7). Zero effect on train CV, so they do not confound attribution. **France +0.003-0.008 (LB +0.0005-0.0012)**, plus possibly France singletons, see D1.

**Item c** (test-only India words): include only if held-out precision is >= 97%. Otherwise leave it out: at 91% it will manufacture name agreement with wrong S1s. Realistic worth: India +0.0005-0.0015 (LB +0.0002-0.0007). Only about 2% of India pool records are partly unmapped.

**Kept separate** (each killable on its own):
- any test-only decision change (item i; France-only strictness);
- coherence (e);
- cross-encoder (h): no, 10+ h on a T4 does not fit;
- blend (j).

**Expected total, LB, from 0.963**
- v3 blocking: +0.001-0.002.
- B1 general: +0.001-0.002.
- France rules: +0.0005-0.0012.
- c: +0.0002-0.0007.
- cycle 2: +0.001-0.003.
- Sum: **+0.004-0.009, i.e. 0.967-0.972.**

**Go/no-go for every submission**
1. Paired CV vs parent: mean fold diff > 2 SE and >= 4/5 folds.
2. Label-free test guards vs sub-4:
   - US house-conflict among emitted pairs <= 8%;
   - **near <= 20 share <= 0.8% (US) and <= 1.2% (India)** (sub-4: 0.62% / 1.03%; gold 0.56% / 0.87%);
   - US/India k/S1 within +0.03 of sub-4;
   - France k/S1 <= 3.50 and empty rate >= 4.9%.

### 4.2 Diagnostics D1 (label-free, about 20 min on EC2, no training)

- **(a) Model-implied expected F0.5 by country on test vs CV.** Run the `decide()` best score per S1, i.e. max(expected F, p0), on the test probabilities and on the 006 OOF. If France's implied E[F] is close to US while its LB share implies 0.90, France is **over-confident**: do LOCO and a France-only stricter emptiness rule. If France's E[F] is low, the gap is genuine uncertainty: work on features.
- **(b) Test stage-1 p distribution vs OOF stage-1 p**, by country: quantiles of `st1_p`, of `cs_n` and of `cp_n`.
  - Stage-2 was trained on 50%-model OOF inputs, but at test time it receives inputs from a 100%-trained stage-1.
  - The US test k excess grew from +0.035 (sub-2) to +0.052 (sub-4), which fits a sharper test stage-1.
  - If the test quantiles are more extreme, compute the test stage-1 as the mean of the CV fold models (or one model at the same training fraction). This is a cheap consistency fix.
- **(c) Decoy-dense simulation with recomputed features.** Drop 19% of train S1, recompute feat4 from the stage-1 OOF on the reduced graph, rescore with the stage-2 fold models, and measure ΔF and the best ratio/floor. This is the only validation available for any test-time strictness change (item i). Without it, do not ship a stricter test rule. The measured US near <= 20 share is at gold level, so a blanket floor would mostly cut true matches.

### 4.3 Cycle 2 candidates, ranked by gain per hour

1. **Blend with the parallel pipeline (j).** A blend does **not** need identical folds. If its per-pair probabilities are truly out-of-fold for each S1, only the blend weight has to be chosen within our folds. Cost about 2 h if its train OOF pair probabilities exist; otherwise skip. Expected +0.001-0.003.
2. **Stage-2 at 100% entity training in CV, plus lr 0.03 / 1000 rounds.** +0.0003-0.001.
3. **Test stage-1 consistency (D1b).** A risk reducer, not validated by CV.
4. **Seed averaging for the final.** +0.0002-0.0005.

## 5. Candidate items contradicted or qualified by data

| item | finding |
|---|---|
| d (France +0.01-0.05) | **Not supported.** The fixable defects are worth France +0.003-0.008. Nothing label-free shows a large France failure; the France house-conflict metric is not comparable (§3.3). |
| brief: "test has 2x near-duplicate decoys" | **True for US/India** (same-name, same-street, near-house unassigned: 415 vs 214 per 1000 S1 in US). **False for France** (209). |
| guard "house-conflict among emitted pairs" | Weak. In US gold 8.5% of pairs conflict by generator noise, and the brief shows the rate hardly moves with the floor. Use near <= 20. |
| g (many-match under-prediction) | t >= 6 is 8% of the loss (0.0018 tw). The problem is one missing noisy match per entity across t = 1-5. Target it with house_rel and the empty-address pool records, not cardinality features. |
| i (stricter test floor or prior shift) | US test emits +0.05 pairs/S1 over CV, but sibling decoys are not inflated among emitted pairs (0.62% vs 0.56% gold). A blanket floor would mostly cut true matches. Validate only via D1c. |
| c (test-only native words) | Plausible but small (see §4.1); 91% precision is below break-even. |
| j (blend needs our folds) | Not required (§4.3). |
| a (v3 skeleton key) | Fine. It misses French website forms with deleted accents ("cole" vs "ecole"): the first letter differs. The France web key in §3.4 covers them. |

## 6. Cheap ideas not in the brief

1. The **house_rel** feature (§4.1): the best measured gain per hour.
2. **ALT regex** misses (51k train, 36k test pool records, all countries).
3. **Admin-area canonicalization** (US state name -> code; France department -> region).
4. **Test stage-1 consistency** for the stage-2 inputs (D1b).
5. **Model-implied E[F] by country** (D1a): decides whether France needs strictness at all.
6. Use **near <= 20** as the decoy guard.
