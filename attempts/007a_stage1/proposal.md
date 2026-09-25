# 007a_stage1

- parent: 005f_floor
- worker: L
- created: 2026-09-25

## Idea
005f on blocking v3 caches (tag _v3): reverse retrieval (each pool record's top-3 S1 by word cosine, flag k_rev) plus exact keys on the consonant skeleton of the space-free name (k_sk), skeleton + house (k_skh) and first|last name token (k_fl). Run with --tag _v3.

## Evidence from history
Gate 6 audit: retrieval loss 0.0102 of 0.0395 (002); a parallel pipeline with bidirectional retrieval and skeleton/combined keys reaches pair recall ~0.981 with ~22 candidates per S1 (ours ~0.971 with ~44). Unit test: "Novic Electric" <-> "Novuc Electric" now joined by k_sk/k_skh.

## Expected gain / risk
+0.001 to +0.003 (retrieval). Risk: more candidates per S1 (decoys) - check the test house-conflict guard.

## Kill condition
No significant paired gain over 005f_floor.

## Result
(fill after er_eval)
