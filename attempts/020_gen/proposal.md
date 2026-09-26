# 020_gen

- parent: 018_lexical
- created: 2026-09-26

## Idea
A residual LightGBM boosted from logit(p018) on US/India pairs with 0.003 <= p018 <= 0.997, using pair-local,
label-free generator-operator features computed from the raw records: number-multiset differences and operation codes
(decoy shifts, truncation, digit drops, transpositions), name edit operations (typos, doubled letters, confusable
characters, acronyms, prefix drops), legal-form ids, reorder/join/split, fillers, honorifics, alias markers, web forms,
address-word differences. France and pairs outside the band keep p018. Candidate set as 019 (the band).

## Result
Full training data (all 2,206,821 S1, harness decide on all 98M pairs), paired vs 018: +0.00082 (SE 0.00001, 5/5;
US +0.00071, India +0.00099); CV 0.98700 (US 0.98766, India 0.98601). Test guard vs 018: India 3.327 -> 3.330
pairs/S1, US 3.362 -> 3.366; empty rate unchanged; house-conflict share 0.1718 -> 0.1722 / 0.0863 -> 0.0866; France
unchanged; 13,945 pairs added, 8,459 dropped.
