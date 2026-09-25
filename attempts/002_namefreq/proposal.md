# 002_namefreq

- parent: 001_baseline
- worker: L
- created: 2026-09-25

## Idea
Add side-car features (src/er_feat2.py): label-free name frequencies (how many S1 / pool records in the same country share the S1's and the pool record's space-free name), conservative-name similarity (ratio, token_set, Jaro-Winkler on lightly normalized raw names that keep legal words and digits), and address ratio/partial gated to -1 when either address is empty (replacing a_ratio/a_partial).

## Evidence from history
Gate 4a audit (docs/astra/04a_*), measured on 4,000 fold-0 entities with exact full-graph features: name frequency +0.0076 (SE 0.0015), conservative names on top +0.0093, both +0.0168 (SE 0.0017); India +0.020, US +0.014. Common names are shared by >5 S1 in 34% (India) / 20% (US) of cases, so rarity should change how much address evidence is needed.

## Why not a repeat
First feature change on top of the baseline.

## Expected gain / risk
+0.01 to +0.017 macro F0.5 (may shrink with the full 20% training sample). Risk: name counts are computed per split (train vs test pool sizes differ by a few percent); log counts keep the scale comparable.

## Kill condition
No significant paired gain over 001_baseline on the full 5-fold CV.

## Result
Full 5-fold CV: macro F0.5 0.96050 +/- 0.00021 (folds 0.96036 / 0.96030 / 0.96039 / 0.96072 / 0.96073); test-weighted 0.95640.
Paired vs 001_baseline: +0.01758, SE 0.00010, wins 5/5 -> significant gain. India 0.9444 (+0.024), US 0.9713 (+0.013).
Singletons 0.908 (from 0.842). Pair precision 0.9893, recall 0.9154.
Stage losses (gold edges 7,638,365): retrieval 221,989 (2.9%), ownership 104,053 (1.4%), selection 320,470 (4.2%); false-positive edges 75,928.
k - t is negative and grows with t (-0.13 at t=2, -0.54 at t>=6): under-prediction for many-match entities. Calibration slightly underconfident for p in 0.7-0.95.
