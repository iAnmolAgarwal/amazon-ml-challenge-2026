# 006r_ratio

- parent: 006_stage2
- worker: L
- created: 2026-09-25

## Idea
Same model and probabilities as 006 (evaluated with er_eval --from-oof 006_stage2; test via er_predict --from-probs 006_stage2). Decision rule: an S1 gets any prediction only if its best expected F0.5 exceeds 2.5 x the probability that it has no match (EMPTY_RATIO 2.5), with no probability floor (FLOOR 0).

## Evidence from history
Floor audit (docs/astra/09_floor_audit.md), 002 OOF full graph: ratio 2.5 = 0.96131 vs floor 0.6 = 0.96112 (+0.00019, SE 0.00002, 5/5; nested +19e-5); decoy simulation +18 to +21e-5. Ratio fixed at 2.5 in advance.

## Kill condition
No significant paired gain over 006_stage2.

## Result
CV 0.97408 (ratio 2.5, no floor, same 006 probabilities). Paired vs 006: -0.00015, SE 0.00001, 0/5 -> worse. Keep floor 0.6 on stage-2 probabilities.
