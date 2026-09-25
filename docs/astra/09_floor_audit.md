# Audit 09: probability floor and "all singletons right"

Auditor: independent (Opus), 2026-09-25. Input: `cache/oof/002_namefreq.parquet`, which holds out-of-fold probabilities for all 97.8M training candidate pairs, plus the raw sources and train gold. I did not retrain anything. EC2 was not touched. Nothing under src/ or attempts/ was edited. Scripts and outputs are in the scratchpad under `audit_floor/`:
- `owners.py`: finds each pool record's owner and second-best S1 on the full graph and on the simulated graphs.
- `lib.py` and `prep.py`: build the folds, the gold table and the expected-F prefix.
- `rules.py`, `ext.py` and `ebr_combo.py`: evaluate the rules.
- `house.py`: extracts pool house numbers with `er_norm.addr_views`.
- `report*.py`: print the tables. The raw tables are in `report*.txt`.

The question: "what if we raise the floor so that ~100% of no-match (singleton) entities are answered correctly?"

## 0. Bottom line

- **No.** Getting 99% of singletons right needs a floor of 0.95. That drops macro F0.5 from 0.9605 to **0.9367 (-0.024)**. Singletons are only 5.6% of S1 entities, and 90.8% of them are already right. Even a perfect oracle that fixes every singleton and changes nothing else adds only **+0.0052**. The floor pays for its singleton gain with recall on entities that do have matches. From floor 0.95 upward that costs about 6 times what it gains.
- **The global floor optimum is 0.59-0.61.** That plateau is flat, and 0.6 is on it. Floor 0.6 vs floor 0: **+0.00062** (SE 0.00006, 5/5 folds). Nothing from 0.65 upward beats 0.6.
- **One better rule:** decide emptiness per entity instead of per pair. Predict a non-empty answer only if its expected F0.5 is at least **2.5 times** the probability that the S1 has no match. Today's `decide()` uses a factor of 1. Drop the floor. Result: **0.96131, +0.00019 over floor 0.6 (SE 0.00002, 5/5)**. In nested CV the ratio is chosen on 4 folds and scored on the 5th, and the gain is still +0.00019, 5/5. Vs floor 0 it is **+0.00081**.
- **In the decoy-dense simulation, the best settings did not move.** I removed 18.75% of S1 entities, and their gold records became decoys. The best global floor is still 0.61. The ratio rule still gains **+0.00018 to +0.00021** over floor 0.6 on all 3 seeds, 5/5 folds each.
- **The simulation is weak.** With probabilities held fixed, the new decoys keep the low scores they earned while their true S1 was present. Only ~3.6k new decoy pairs get emitted, out of 5.7M. It therefore understates the test-time damage (see 3c).
- **A floor cannot fix the test's decoy problem.** 88% of train false positives have p > 0.7, and 22% of them have p > 0.95.
- **The house-number-conflict share hardly moves with the floor.** US: 6.74% at floor 0, 6.69% at 0.6, 5.78% at 0.9. Most conflicting pairs are true matches with noisy house numbers. So the 14.7% on submission 4 is a probability shift that no cutoff undoes. The metric cannot be used to tune the floor.
- **Ship:** `decide()` with the empty-ratio rule (ratio 2.5, floor 0). Validate it as an attempt on 005n's OOF (section 4). It needs a small harness change and the human's approval.

A note on the numbers in the brief (floor 0 -> 0.95989 on the 1/4 sample). `floor_sweep.py` runs `decide()` on the sampled S1 only. Each pool record's owner is then chosen among 25% of the S1s, which is the same as deleting 75% of the rival S1s. That makes the graph far more decoy-dense than train. Here, ownership is always computed on the full graph and scored on all S1s. Floor 0 gives **0.96050**, which matches the official 002 CV of 0.96050 exactly. All numbers below are on the full graph, over all 2.2M S1s.

Folds are rebuilt with the exact `folds()` code (StratifiedKFold, seed 42, country x match count 0-4+) and kept in the scratchpad. `data/folds.parquet` does not exist on the Mac and was not written. The paired statistics are "mean of 5 per-fold differences (SE of the mean, number of positive folds)", shown x1e-5.

## 1. Global floor sweep (full graph, 002 OOF)

Entity shares: singletons 5.6%, t=1 5.4%, t=2 17.0%, t=3 24.1%, t=4 21.9%, t=5+ 26.0%. "sing part" and "matched part" split the macro change vs floor 0 into the change on singletons and the change on matched entities (x1e-5).

| floor | macro | India | US | sing acc | F t=1 | t=2 | t=3 | t=4 | t=5+ | pair P | pair R | vs 0 | vs 0.6 | sing part | matched part |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.96050 | 0.94438 | 0.97126 | 0.9077 | 0.9121 | 0.9566 | 0.9656 | 0.9695 | 0.9722 | 0.9893 | 0.9154 | 0 | -62.2 (6.2, 0/5) | 0 | 0 |
| 0.50 | 0.96087 | 0.94476 | 0.97162 | 0.9212 | 0.9094 | 0.9557 | 0.9653 | 0.9694 | 0.9721 | 0.9896 | 0.9152 | +36.6 (3.2, 5/5) | -25.6 (3.4, 0/5) | +76 | -39 |
| 0.55 | 0.96109 | 0.94500 | 0.97182 | 0.9317 | 0.9060 | 0.9550 | 0.9652 | 0.9693 | 0.9721 | 0.9898 | 0.9150 | +58.5 (5.3, 5/5) | -3.7 (2.1, 1/5) | +134 | -76 |
| 0.57 | 0.96112 | 0.94501 | 0.97186 | 0.9353 | 0.9044 | 0.9546 | 0.9651 | 0.9693 | 0.9721 | 0.9899 | 0.9150 | +61.4 (6.2, 5/5) | -0.8 (1.2, 1/5) | +154 | -93 |
| 0.59 | 0.96112 | 0.94503 | 0.97187 | 0.9387 | 0.9028 | 0.9543 | 0.9650 | 0.9693 | 0.9721 | 0.9900 | 0.9149 | +62.3 (5.8, 5/5) | +0.1 (0.8, 3/5) | +173 | -111 |
| **0.60** | **0.96112** | 0.94500 | 0.97188 | 0.9403 | 0.9020 | 0.9541 | 0.9649 | 0.9692 | 0.9721 | 0.9900 | 0.9148 | +62.2 (6.2, 5/5) | 0 | +182 | -120 |
| 0.61 | 0.96113 | 0.94500 | 0.97189 | 0.9421 | 0.9012 | 0.9539 | 0.9649 | 0.9692 | 0.9721 | 0.9900 | 0.9148 | +62.9 (7.0, 5/5) | +0.7 (0.9, 4/5) | +192 | -129 |
| 0.63 | 0.96109 | 0.94490 | 0.97189 | 0.9453 | 0.8993 | 0.9536 | 0.9648 | 0.9691 | 0.9720 | 0.9901 | 0.9147 | +58.6 (7.4, 5/5) | -3.5 (1.7, 1/5) | +210 | -151 |
| 0.65 | 0.96104 | 0.94479 | 0.97189 | 0.9487 | 0.8976 | 0.9532 | 0.9646 | 0.9691 | 0.9720 | 0.9902 | 0.9145 | +54.2 (7.7, 5/5) | -8.0 (1.8, 0/5) | +229 | -175 |
| 0.67 | 0.96096 | 0.94463 | 0.97185 | 0.9518 | 0.8958 | 0.9527 | 0.9643 | 0.9689 | 0.9719 | 0.9903 | 0.9143 | +45.5 (7.8, 5/5) | -16.7 (1.8, 0/5) | +246 | -201 |
| 0.69 | 0.96082 | 0.94437 | 0.97179 | 0.9548 | 0.8940 | 0.9522 | 0.9641 | 0.9688 | 0.9719 | 0.9904 | 0.9140 | +31.6 (7.7, 5/5) | -30.6 (1.8, 0/5) | +263 | -232 |
| 0.70 | 0.96073 | 0.94420 | 0.97176 | 0.9562 | 0.8931 | 0.9519 | 0.9639 | 0.9687 | 0.9718 | 0.9905 | 0.9137 | +22.7 (8.0, 5/5) | -39.5 (2.2, 0/5) | +271 | -248 |
| 0.75 | 0.95992 | 0.94286 | 0.97131 | 0.9631 | 0.8890 | 0.9499 | 0.9627 | 0.9679 | 0.9712 | 0.9910 | 0.9115 | -58.0 (8.8, 0/5) | -120.2 (2.8, 0/5) | +310 | -368 |
| 0.80 | 0.95851 | 0.94066 | 0.97042 | 0.9694 | 0.8838 | 0.9477 | 0.9611 | 0.9663 | 0.9698 | 0.9923 | 0.9055 | -199 (10, 0/5) | -261 (4, 0/5) | +345 | -544 |
| 0.85 | 0.95583 | 0.93654 | 0.96871 | 0.9757 | 0.8748 | 0.9437 | 0.9583 | 0.9638 | 0.9673 | 0.9938 | 0.8963 | -467 (10, 0/5) | -529 (4, 0/5) | +380 | -847 |
| 0.90 | 0.95054 | 0.92847 | 0.96527 | 0.9826 | 0.8592 | 0.9361 | 0.9529 | 0.9591 | 0.9626 | 0.9955 | 0.8809 | -996 (13, 0/5) | -1059 (7, 0/5) | +418 | -1414 |
| 0.95 | 0.93667 | 0.90809 | 0.95575 | 0.9902 | 0.8241 | 0.9175 | 0.9389 | 0.9465 | 0.9507 | 0.9974 | 0.8463 | -2383 (12, 0/5) | -2445 (8, 0/5) | +461 | -2844 |
| 0.99 | 0.86683 | 0.80972 | 0.90494 | 0.9982 | 0.6808 | 0.8236 | 0.8667 | 0.8828 | 0.8922 | 0.9995 | 0.7116 | -9367 | -9429 | +506 | -9873 |

Oracle ceiling: floor 0 with every singleton answered empty gives 0.96566 (+0.0052).

**Why ~100% singleton accuracy is not worth it.** A singleton's F0.5 is all-or-nothing, but singletons are only 5.6% of entities. The most that singletons can ever add is 5.6% x 9.2% (current error) = +0.0052. A global floor cannot pick out singletons. The pairs it drops mostly belong to entities that do have matches.

- **Floor 0.6:** drops 9.4k emitted pairs. 5.3k of them are false positives, and 4.1k are true matches.
- **From floor 0.7 upward:** the pairs just above the cut are mostly true matches, spread over all entities. In the 0.7-0.95 band, 565k emitted pairs hold 50k false positives but 515k true matches.
- **Floor 0.95:** singleton accuracy reaches 99.0% (+0.0046 macro). Recall on matched entities falls from 0.915 to 0.846, which costs **-0.0284**. Net -0.024.
- **t=1 entities are hit hardest.** One lost true pair turns their F from 1 into 0. Their F0.5 falls from 0.912 to 0.824.

Past about 0.61, each extra singleton the floor fixes costs more F on matched entities than it adds. That is why the optimum is 0.59-0.61 and not higher.

## 2. Smarter rules (full graph; best of each family)

"Nested" means the parameter is chosen on 4 folds and scored on the held-out fold, repeated for all 5. It guards against grid-search optimism.

| family | best parameter | macro | India | US | sing acc | vs floor 0.6 (in-sample) | nested vs floor 0.6 |
|---|---|---|---|---|---|---|---|
| global floor | 0.61 | 0.96113 | 0.94500 | 0.97189 | 0.9421 | +0.7 (0.9, 4/5) | -0.5 (0.6, 3/5) |
| (a) emptiness threshold on the top pair, lower floor on the rest | t_empty 0.6, others 0 | 0.96116 | 0.94508 | 0.97190 | 0.9403 | +4.1 (0.3, 5/5) | +4.1 (0.3, 5/5) |
| (b) floor only on k>=2 | no emptiness threshold, any floor: worse than floor 0 (floor 0.6 on k>=2: -8.1 vs floor 0) | 0.96042 | | | | -70.3 | |
| (c) margin p - (pool record's 2nd-best S1 prob) | floor 0.6 and margin > 0.1 | 0.96115 | 0.94503 | 0.97191 | 0.9420 | +2.6 (1.6, 5/5) | +2.6 (1.6, 5/5) |
| (c') competition-adjusted p x (1 - p2), then re-decide | floor 0.6 | 0.96078 | 0.94417 | 0.97187 | 0.9468 | -34.2 (2.5, 0/5) | -34.2 |
| (c'') floor before the prefix (the prefix is recomputed) | 0.6 | 0.96093 | 0.94473 | 0.97174 | 0.9403 | -19.4 (0.6, 0/5) | -19.4 |
| (d) per-country floors | India 0.60, US 0.65 | 0.96113 | 0.94500 | 0.97189 | 0.9439 | +0.6 (0.8, 2/5) | -0.9 (0.4, 0/5) |
| (e) additive emptiness margin: best > p0 + d | d 0.35-0.40, floor 0 | 0.96128 | 0.94523 | 0.97200 | 0.9503 | +16.0 (1.5, 5/5) | +14.7 (1.4, 5/5) |
| **(e') emptiness ratio: best > r x p0** | **r = 2.5, floor 0** | **0.96131** | **0.94528** | **0.97201** | 0.9518 | **+18.9 (2.2, 5/5)** | **+18.9 (2.2, 5/5)** (always picks 2.5) |
| (e) + margin floor | d 0.35, margin > 0.05 | 0.96130 | 0.94528 | 0.97200 | 0.9465 | +17.9 (1.9, 5/5) | +16.4 (2.3, 5/5) |
| (e') + margin floor | r 2.5, margin > 0.05 | 0.96133 | | | 0.9523 | +20.2 (2.7, 5/5) | not nested |

Ratio sweep r (vs floor 0.6): 1.0 -62.2, 1.5 +3.5, 1.75 +14.9, 2.0 +16.6, **2.5 +18.9**, 3.0 +12.4, 4.0 -16.4. The optimum is smooth, and 2.0-3.0 is a broad plateau. Adding any floor on top of r = 2.5 does not help: floor 0.4 +18.7, 0.5 +17.5, 0.6 +7.0.

**Why the entity-level rule wins.** `decide()` already compares "answer something" (best expected F0.5) with "answer empty" (p0 = product of (1 - p)). The model's probabilities are too optimistic about whether an S1 has any match at all, so this comparison leans towards answering. The ratio corrects that lean, and only for weak entities:
- An S1 with one candidate now needs p > 0.70, where the current rule needs 0.47.
- When the evidence is spread over several candidates, p0 is smaller, so the rule is more lenient.
- For S1s with strong pairs, p0 is about 0 and nothing changes. In particular, the true tail pairs of confident entities are kept.

A floor instead strips the tail pairs of good entities: rule (b) shows that alone costs -8.1. Decomposition vs floor 0: floor 0.6 gives singletons +182 and matched entities -120. Ratio 2.5 gives singletons +246 and matched entities -165. The ratio buys more singleton accuracy (95.2%) per unit of matched-entity loss.

(c) Calibration: any per-country monotone recalibration followed by a floor is the same as a per-country floor (d), which gains nothing. Re-deciding on a competition-adjusted probability hurts. France has no labels, so any France-specific floor or ratio cannot be validated. Keep all such rules country-agnostic.

## 3. Test-like decoy density (simulation)

**3a. Method.** Remove a random 18.75% of S1 entities (seeds 0, 1, 2). That raises pool/S1 from 4.68 to 5.76, and the share of decoy pool records from 26% to ~40%. The removed S1s' gold pool records become decoys. Each pool record's owner and second-best S1 are recomputed over the remaining S1s, probabilities are held fixed, and the remaining S1s are scored on the same folds. The simulation uses no labels in any decision.

**3b. Results (mean of 3 seeds; range of the paired diff vs floor 0.6 across seeds):**

| rule | macro | India | US | sing acc | vs floor 0.6 (x1e-5) |
|---|---|---|---|---|---|
| floor 0 | 0.96036 | 0.94425 | 0.97110 | 0.9054 | -68.1 [-68.2, -67.9] |
| floor 0.5 | 0.96076 | 0.94466 | 0.97150 | 0.9198 | -27.9 |
| floor 0.55 | 0.96099 | 0.94491 | 0.97172 | 0.9305 | -4.8 |
| floor 0.60 | 0.96104 | 0.94492 | 0.97180 | 0.9395 | 0 |
| floor 0.61 (best floor, all seeds) | 0.96105 | 0.94492 | 0.97181 | 0.9413 | +1.2 [+0.8, +2.0] |
| floor 0.65 | 0.96097 | 0.94471 | 0.97182 | 0.9480 | -7.1 |
| floor 0.70 | 0.96066 | 0.94413 | 0.97168 | 0.9558 | -38.2 |
| floor 0.80 | 0.95846 | 0.94063 | 0.97036 | 0.9691 | -258 |
| floor 0.95 | 0.93667 | 0.90810 | 0.95573 | 0.9901 | -2437 |
| (a) t_empty 0.6, others 0 | 0.96108 | 0.94499 | 0.97181 | 0.9395 | +4.1 [+4.0, +4.2] |
| (c) floor 0.6 and margin > 0.1 | 0.96107 | 0.94495 | 0.97183 | 0.9409 | +3.3 [+3.0, +3.5] |
| (d) India 0.6 / US 0.65 | 0.96105 | 0.94492 | 0.97182 | 0.9432 | +1.3 [+0.5, +2.3] |
| (e) best > p0 + 0.4 | 0.96121 | 0.94515 | 0.97193 | 0.9496 | +17.0 [+15.7, +19.3] |
| **(e') best > 2.5 p0** | **0.96124** | 0.94520 | 0.97193 | 0.9509 | **+19.7 [+18.4, +20.9]**, 5/5 on every seed |
| (e') best > 3.0 p0 | 0.96119 | 0.94511 | 0.97191 | 0.9568 | +14.7 |

Nested tuning inside each simulated graph picks the same parameters: floor 0.59-0.61, ratio 2.5, d 0.35-0.4. The value of floor 0.6 over floor 0 rises slightly, from +62 to +68.

**3c. How the simulation is biased.** Holding probabilities fixed makes the simulated test look better than the real test will:
- The 002 model has pool-side graph features (`rev_rank`, `rev_gap`, `n_rival` over `pool_idx`). Before removal, those features saw the decoy's true S1 winning, so the decoy pairs were scored low.
- On the real test, a decoy's true S1 is absent. The remaining look-alike S1 is rank 1 for that pool record with a wide gap, so the model scores it higher.
- In the simulation, only ~2.3k (India) and ~1.2k (US) of the new decoy records are emitted, against ~61k false positives already present. The decoy increase barely reaches the output.
- So the simulation is a **lower bound** on the test decoy damage.

It still shows two useful things: the optimal cutoffs do not drift when the decoy density is raised on its own, and the ratio rule's advantage over floor 0.6 holds or grows slightly. It cannot tell whether a much higher floor would pay off on the real test. What would decide that is how many test decoys land in the 0.5-0.75 band. That band is where the ratio and a floor can act.

**3d. More singletons on test (sensitivity, not validatable).** The simulation does not change the singleton share. If the test had a larger share s of no-match S1s, the optimal floor would move up. Here the train singleton and matched F0.5 are re-weighted to a share s. Values are the gain vs floor 0.6, x1e-5.

| s | floor 0 | floor 0.65 | floor 0.70 | ratio 2.5 | ratio 3.0 | best floor | best ratio or d |
|---|---|---|---|---|---|---|---|
| 5.6% (train) | -63 | -8 | -39 | **+19** | +13 | 0.61 | r 2.5 |
| 8% | -144 | +13 | +2 | +48 | **+56** | 0.65 | d 0.5 |
| 10% | -212 | +31 | +37 | +71 | **+92** | 0.69 | d 0.55 |
| 15% | -381 | +76 | +123 | +131 | **+182** | 0.70 | d 0.6 |

The ratio rule beats floor 0.6 at every singleton share, and it beats the best floor for that share too. What is the label-free evidence on the test singleton share? Submission 2 predicted empty for 5.5% (India) and 5.5% (US) of test S1s, vs 5.8% and 5.6% on train OOF at floor 0. That suggests the test singleton share is near train's, unless test singleton accuracy has collapsed. This is weak evidence, so ratio 2.5 is the right default. Ratio 3.0 is a hedge that costs -6e-5 in CV.

**3e. House-number-conflict share among emitted pairs.** Pool houses are extracted with `er_norm.addr_views` and S1 houses come from `cache/norm`. A conflict means both houses are present and they differ.

| rule (full graph) | US emitted | US conflict | conflict among US false positives | US false positives | India conflict |
|---|---|---|---|---|---|
| floor 0 | 4.306M | 6.74% | 38.2% | 31,263 | 15.50% |
| floor 0.6 | 4.302M | 6.69% | 35.9% | 28,761 | 15.44% |
| floor 0.7 | 4.298M | 6.65% | 35.3% | 27,643 | 15.36% |
| floor 0.8 | 4.270M | 6.40% | 33.3% | 23,079 | 14.91% |
| floor 0.9 | 4.182M | 5.78% | 30.6% | 14,553 | 13.75% |
| ratio 2.5 (d 0.4 is similar) | 4.301M | 6.68% | 35.7% | 28,320 | 15.43% |

The simulated graphs give the same values to within ±0.01 points. Only ~0.7% of emitted US pairs are false positives, so ~96% of conflicting pairs are true matches with noisy house numbers. That is why the share barely moves with the floor.

The share is still a good alarm for a probability-level shift. Submission 2 on test was 7.0%, near train's 6.7-7.4%. Submission 4 was 14.7%, meaning about 8% of its emitted US pairs were extra conflicting pairs. But no floor or ratio can fix that: in train, even floor 0.9 moves the share by less than 1 point. Do not use the share to tune the cutoff. Use it only as a go/no-go guard on test.

## 4. Recommendation

**Ship `decide()` with the empty-ratio rule: r = 2.5, floor = 0.** An S1 gets a non-empty answer only if the best prefix's expected F0.5 exceeds 2.5 x p0. After that, the full expected-F prefix is emitted, with no per-pair floor.

- **Expected in-CV change** (measured on the 002 OOF): +0.00019 (SE 0.00002, 5/5) vs floor 0.6 (005f), and +0.00081 (SE ~0.00006, 5/5) vs floor 0 (005n). India gains +0.0009 and US +0.0008 vs floor 0. Both gains need re-measuring on 005n's OOF.
- **Expected direction on test:** the same as CV or better.
  - In the simulation, the advantage holds or grows slightly (+0.00018 to +0.00021).
  - If the test has more singletons, it grows (+0.0005 to +0.0009 at an 8-10% singleton share).
  - It does nothing about high-confidence decoy false positives. Those need model or feature fixes (for example, 005n's removal of the coherence features, or the stage-2 competition features from audit 08).
- **Implementation (harness change, needs the human's approval):** in `src/er_model.py` `decide()`, add a `ratio=1.0` parameter and change `best.filter(pl.col("best") > pl.col("p0"))` to `best.filter(pl.col("best") > ratio * pl.col("p0"))`. With the default ratio of 1.0, every existing output stays bit-identical. Add a `RATIO` attempt variable next to `FLOOR`, read by `er_eval --from-oof` and `er_predict`.
- **Validation under the paired rule:**
  1. Create an attempt, for example `005r_emptyratio`, with parent 005n, `RATIO = 2.5` and `FLOOR = 0`. Pre-register 2.5. It was chosen on 002, so on 005n it is an independent confirmation. Do not re-tune on 005n.
  2. Run `er_eval --from-oof 005n_indic_nocoh` for 005r (and 005f if it has not run yet). Compare paired per fold on the same folds against 005n, and against 005f if 005f passes.
  3. Pass condition: mean fold difference > 2 SE and positive in at least 4 of 5 folds, as `leaderboard.py` flags it. The expected margin is about 8 SE vs 005f and about 13 SE vs 005n, so a pass is very likely unless 005n's calibration differs materially.
  4. Label-free test guard after predicting: the predicted-empty share should rise by about +0.6 points (India) and +0.3 points (US). The train OOF equivalents are 5.79% -> 6.43% and 5.58% -> 5.91%. The US house-conflict share should stay at about submission 2's level.

**Which conclusions are validatable in ordinary CV, and which only in the simulation:**
- **Validatable in ordinary CV (they also hold under the simulation):**
  - ratio 2.5 and additive d 0.35-0.4;
  - emptiness threshold 0.6 with no floor on the other pairs (+0.00004);
  - margin > 0.1 on top of floor 0.6 (+0.00003).
  These are small, but they pass the paired rule in CV.
- **No rule helps only under the simulation.** The simulation agrees with CV everywhere.
- **Unvalidatable:**
  - Raising the cutoff above CV-optimal as a hedge for a higher test singleton share: ratio 3.0 (-6e-5 in CV) or floor 0.65-0.70 (-8e-5 to -40e-5).
  - Any France-specific setting.
  - Per-country floors: no CV gain (nested -0.9e-5). Reject them.
- **Rejected by CV:**
  - floor >= 0.65;
  - floor only on k >= 2;
  - floor before the prefix;
  - competition-adjusted re-decide.
