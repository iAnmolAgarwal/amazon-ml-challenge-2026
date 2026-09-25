# 010_route

- parent: 009_stage3m
- created: 2026-09-26

## Idea
Extra candidates from character 3-gram TF-IDF retrieval (top 40 per S1 for the name view and for the address view,
within country; src/er_retrieve.py). Pairs not already candidates are scored by a separate 66-feature LightGBM trained
on existing train pairs (er_feat base 34 + er_feat5 32); new pairs with p > 0.001 are appended to 009's probabilities,
which stay unchanged, and the usual decision runs on the union (src/er_newpairs.py).

## Evidence
Train 10% sample (220,678 S1; retrieval for these owners, fold-held-out classifiers): +0.00128 vs 009 (SE 0.00005, 5/5;
US +0.0015, India +0.0010). New candidates kept: 0.63 per sampled S1.

## Result (test)
94.5M new pairs retrieved, 1.63M kept (0.94 per S1), 28k above 0.6. Guard vs 009: pairs per S1 +0.01 (US/India),
France 3.215 -> 3.237, house-conflict share unchanged. Validator PASS.
