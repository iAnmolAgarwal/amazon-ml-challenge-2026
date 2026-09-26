# 013_stage3lp

- parent: 009_stage3m
- created: 2026-09-26

## Idea
Same stage 3 as 009 with two France-oriented feature fixes (src/er_lift.py): decoy-marker scores lifted by the
country's base rate (feat6l: logit(score) - log(sum nd / sum n0)), so France's test-built table sits on the train scale;
and France house columns recomputed after a French fallback house parser (feat5p). Train has no France, so in-country
the change should be neutral.

## Result
Full CV 0.98498; paired vs 009: +0.00001 (SE 0.00001, 3/5), neutral as intended. Test: France pairs per S1
3.232 -> 3.275, empty rate 5.95% -> 5.81%; US/India unchanged.
