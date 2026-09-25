# 003_trainfrac50

- parent: 002_namefreq
- worker: L
- created: 2026-09-25

## Idea
Same features and model as 002; train each fold's model on 50% of fold-train entities instead of 20%.

## Evidence from history
Gate 5 audit learning curve (fold 0, nested samples): 10%->20% +0.00071 (SE 0.00007), 20%->40% +0.00088 (SE 0.00007), India +0.00138. Curve not flattening.

## Why not a repeat
First change of training coverage.

## Expected gain / risk
About +0.001. Risk: memory (~45 GB peak with the current harness); runtime ~2.5x per fold.

## Kill condition
No significant paired gain over 002.

## Result
(fill after er_eval)
