# 009_stage3m

- parent: 008_stage3
- created: 2026-09-26

## Idea
008's stage-3 correction with 5 more label-free name-operator features (src/er_feat6.py): the number of words the pool
name adds / drops relative to the S1 name, and decoy-marker scores for the added, dropped and swapped words. A marker
score contrasts, without labels, how often a word is added when the pool house number is the S1's shifted by one of the
generator's typical shifts versus when the house numbers are equal. Tables come from train pairs; France builds its
own from its test pairs.

## Evidence
10% sample: +0.0029 over the 008 feature set (SE 0.00009, 5/5); gains grow under doubled decoys.

## Result
Full CV 0.98498: +0.00244 vs 008 (SE 0.00002, 5/5; US +0.0019, India +0.0033). Test guard: US / India unchanged;
France emits 3.8% fewer pairs; sampled dropped France pairs are generator decoys (word swaps, decoy words, legal
swaps, shifted house numbers).
