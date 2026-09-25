# 012_lexical

- parent: 010_route
- created: 2026-09-26

## Idea
Lexical correction of the existing-candidate probabilities (src/er_lexical.py): six token channels per pair from the raw
records (name words added / dropped, address words added / dropped, legal-word transition, character edits with one
character of context). Token likelihood ratios come from owner-deduplicated counts among true matches and non-matches of
other folds (cross-fitted), shrunk by n/(n+10), summarised as 42 features; a small LightGBM correction (100 rounds, 9
leaves) boosted from logit(p) on pairs whose 006 probability is >= 0.01. Route pairs are unchanged.

## Result
Full CV, nested cross-fitting (the outer fold is excluded from the token counts too), all 2,206,821 train S1:
+0.00031 vs 010_route (SE 0.00001, 5/5; US +0.0002, India +0.0005). Test: 8.79M pairs corrected; guard unchanged;
validator PASS.
