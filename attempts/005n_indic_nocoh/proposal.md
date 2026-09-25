# 005n_indic_nocoh

- parent: 003_trainfrac50
- worker: L
- created: 2026-09-25

## Idea
003 (feat2, 50% training) on the rebuilt caches with the native-script dictionary (er_dict + er_norm latinize), without the feat3 coherence side-car.

## Evidence from history
Label-free test diagnostic (12:05 IST): house-number conflict rate among emitted pairs, test vs train OOF. US: train OOF 7.4%; test sub-1 5.5%, sub-2 7.0%, sub-4 (004, coherence) 14.7%. France 1.5% -> 6.0%, India 15.9% -> 18.1%. The pairs 004 adds on test are near-duplicate decoys (same street, nearby house number, similar name); test has ~2x the decoys per S1 of train (pool/S1 5.76 vs 4.68; ~40% of test pool unassigned vs 26% true decoys in train), so coherence anchors pick up decoys on test but not in CV. Coherence is set aside until a decoy-robust version passes a decoy-dense validation.

## Expected gain / risk
Dictionary: India +0.004-0.008 test-weighted. Guard before upload: test house-conflict rate of emitted US pairs near the train OOF rate (~7%).

## Kill condition
No significant paired gain over 003.

## Result
CV 0.97008 +/- (see results.json); test-weighted 0.96939; India 0.9674 (oracle 0.9815 -> 0.9918), US 0.9719. Paired vs 003: +0.00899, SE 0.00008, 5/5 -> significant.
