# 006_stage2

- parent: 005f_floor
- worker: L
- created: 2026-09-25

## Idea
Second-stage model (adapted from the team's parallel pipeline): 005f's features plus feat4 competition features computed from stage-1 probabilities (er_feat4.py): the stage-1 p, and for the pool record its rank among the S1 that retrieved it, best other S1 p, margin, number of competitors, sum of p; for the S1 its rank, gap to its best candidate, best other candidate, number of candidates with p >= 0.5 and sum of p. Train: stage-1 = 005n out-of-fold probabilities. Test: stage-1 = 005f full-train test probabilities.

## Evidence from history
Team pipeline: two-stage LightGBM with these features reaches OOF 0.9754 (India 0.9710, US 0.9784) with singleton accuracy 0.9725 (ours 0.939 with the floor). Gate 6 audit: 86% of the classification ceiling sits in pairs with p in [0.05, 0.95] (the ones competition context resolves).

## Expected gain / risk
+0.003 to +0.006. Risk: stacking optimism (stage-1 OOF rows scored by models that saw other folds), and a calibration gap between OOF (50%-trained) and test (100%-trained) stage-1 probabilities.

## Kill condition
No significant paired gain over 005f.

## Result
(fill after er_eval)
