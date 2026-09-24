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
(fill after er_eval)
