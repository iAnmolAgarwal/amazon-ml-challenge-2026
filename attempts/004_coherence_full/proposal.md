# 004_coherence_full

- parent: 002_namefreq (also compared with 003_trainfrac50)
- worker: L
- created: 2026-09-25

## Idea
002 + feat3 coherence side-car (agreement of each candidate with the S1's label-free anchors: name/address/house/postal agreement and conflicts, support counts, cross-source support) + train each fold's model on 100% of fold-train entities.

## Evidence from history
Gate 5 audit: coherence +0.0025-0.003 on sample (3 seeds); learning curve 10->20% +0.00071, 20->40% +0.00088, not flattening.

## Why not a repeat
First use of pool-to-pool evidence; first full-coverage training.

## Expected gain / risk
+0.004-0.005 over 002. Risk: memory (60 features x 98M pairs float32 ~23.5 GB + bins); ~2 h CV on r7a.2xlarge.

## Kill condition
No significant paired gain over 002 and 003.

## Result
(fill after er_eval)
