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
Full 5-fold CV (2,206,821 entities, 97.8M candidate pairs, 41 features, LightGBM on 20% of fold-train entities), commit 85b2c3c + review fixes:
- macro F0.5 0.94292 +/- 0.00029 (folds 0.94249 / 0.94299 / 0.94289 / 0.94294 / 0.94329); test-weighted (US 0.38 / India 0.47) 0.93714.
- oracle (blocking ceiling) 0.98983 -> model + decision loss 0.047. Pair precision 0.9815, recall 0.8908: most loss is missed matches.
- India 0.9202 (oracle 0.9815), US 0.9581 (oracle 0.9954).
- By true matches: 0 (singletons) 0.842, 1: 0.885, 2: 0.939, 3: 0.951, 4: 0.956, 5: 0.959, 6+: 0.961. Singletons and 1-match entities lose most.
Next: 002_namefreq (name frequency + conservative names, measured +0.017 in the gate 4a audit).
