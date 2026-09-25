# 008_stage3

- parent: 006_stage2
- created: 2026-09-25

## Idea
Stage 3 over 006's stage-2 probabilities: a LightGBM boosted from logit(p006) with 34 label-free pair features
(er_feat5): name and address token specificity (IDF-weighted shared / one-sided mass, cosines), house-number relations
(dropped leading digits, small numeric shifts, edit distance, signed shift and the generator's typical shift set),
unit/suite agreement, legal-form agreement, char-3-gram name cosine. Only pairs with p006 >= 0.001 are re-scored.
Every feature compares one pool record with one S1 (no sibling or coherence signal: test decoys come in clusters).

## Evidence
Research sample (10% of train S1, 220,678 S1): the same correction with the 32 base features (no signed shift)
gains +0.00716 over 006 (SE 0.00008, 5/5 folds; India +0.0075, US +0.0069). Feature port verified identical on
13.1M test pairs.

## Kill condition
No significant paired gain over 006 on the full out-of-fold CV, or a rise of the house-conflict share of emitted
test pairs relative to 006.
