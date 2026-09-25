# 004b_coherence50

- parent: 003_trainfrac50
- worker: L
- created: 2026-09-25

## Idea
003 + feat3 coherence side-car, same 50% training fraction: isolates the coherence gain with a clean paired comparison against 003. Run because the 100% CV of 004 was lost to a spot reclaim; 50% runs in ~50 min instead of ~1 h 45.

## Evidence from history
Gate 5 audit: coherence +0.0025-0.003 on sample; 003 vs 002 (training fraction 20% -> 50%) significant.

## Expected gain / risk
+0.002-0.003 over 003.

## Kill condition
No significant paired gain over 003.

## Result
CV 0.96351 +/- 0.00010 (test-weighted 0.95947; India 0.9476, US 0.9741). Paired vs 003: +0.00242, SE 0.00006, 5/5 -> significant. Clears 004 (same features, 100% training) for submission 4.
