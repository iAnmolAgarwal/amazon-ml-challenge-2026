# 001_baseline

- parent: none
- worker: L
- created: 2026-09-25

## Idea
Baseline entity-resolution pipeline (strategy v2/v3, docs/strategy.md):
normalization v2 (src/er_norm.py) -> blocking = chunked word-view TF-IDF top-40 + 6 exact keys, cap 50 (src/er_block.py)
-> 41 pair features incl. both-side ranks (src/er_feat.py) -> LightGBM defaults (src/er_model.py, trained on 20% of fold-train entities)
-> decision: pool record to its best S1, then per-S1 k maximizing expected F0.5 (k = 0 allowed).

## Evidence from history
Blocking on 20k S1 per country against the full train pool: oracle F0.5 0.994 US / 0.979 India, 45 candidates per S1.

## Why not a repeat
First real attempt.

## Expected gain / risk
Baseline. Risks: calibration of LightGBM probabilities drives the k selection; India Indic-script names have weak string features.

## Kill condition
n/a (baseline)

## Result
(fill after er_eval)
