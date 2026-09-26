# 016_stage2b

- parent: 006_stage2 (band), compared with 009_stage3m and 013_stage3lp
- created: 2026-09-26

## Idea
Instead of a residual correction boosted from logit(p006), retrain a fresh LightGBM (INIT = False, USE_LOGIT = False)
on the band of pairs with p006 >= 0.001 (12.9M train pairs) with every feature set: feat, feat2, feat4 (competition
features), feat5p and feat6l; 127 leaves, 600 rounds. Pairs below the band keep p006.

## Result
Full CV 0.98589 (US 0.98671, India 0.98466); paired vs 009: +0.00092 (SE 0.00002, 5/5); vs 013: +0.00091
(SE 0.00002, 5/5). Oracle 0.99398, pair precision 0.9982, recall 0.9604.
