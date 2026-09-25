# 005_indic

- parent: 004b_coherence50 (same model settings, CV at 50%; predict at 100% like 004)
- worker: L
- created: 2026-09-25

## Idea
Native-script -> Latin token dictionary learned from training gold pairs (er_dict.py): 1,347 Indic name tokens aligned by position to 147 Latin words, 17 native state tokens in addresses, 4 Latin spelling-variant classes (laxmi/lakshmi, jai/jay, sree/shree, shiv/shiva) on India names. Applied in er_norm before all views (name_lat keeps the latinized raw name for er_feat2/er_feat3). Full rebuild: norm, block, feat, feat2, feat3.

## Evidence from history
Gate 6 audit: 56% of India retrieval misses have Indic-script pool names; ceiling India +0.024 (test-weighted +0.0115), proxy +0.0083. Pilot on 30k native-script gold pairs: exact space-free name key match 0% -> 100%, name token_set 9 -> 100. Fold-local dictionaries equal the global one (2-6 of 1,312 targets differ; all tokens >= 20 owners). Test pool: 86% of native-script names become fully Latin (test uses 171 business words absent from train native text; phase 2).

## Confound (noted 11:25 IST)
The rebuilt train caches also pick up er_norm fixes made after the old train norm (22:03 UTC 24 Sep) but before the test norm used by submissions 2-4 (00:55 UTC 25 Sep): US train candidates change slightly (k_nj 6,281,544 -> 6,300,972). 005 vs 004b therefore measures dictionary + train/test normalization consistency; read the by-country deltas (US change = consistency only).

## Expected gain / risk
+0.004 to +0.008 test-weighted (India only; US/France no-op except canon on India names). Risk: blocking candidate set changes for India; memory as 004.

## Kill condition
No significant paired gain over 004b_coherence50.

## Result
(fill after er_eval)
