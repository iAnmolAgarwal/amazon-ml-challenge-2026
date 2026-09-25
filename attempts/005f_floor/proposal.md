# 005f_floor

- parent: 005n_indic_nocoh
- worker: L
- created: 2026-09-25

## Idea
005n with a probability floor in decide(): after the expected-F0.5 prefix is chosen, pairs with prob <= 0.6 are dropped (adapted from a parallel pipeline, where 0.6 was chosen by train-India -> score-US transfer). Evaluated on 005n's saved out-of-fold probabilities (er_eval --from-oof 005n_indic_nocoh), so the paired comparison isolates the floor.

## Evidence from history
002 OOF, 1/4 entity sample: floor 0.6 -> macro +0.00088 (0.95989 -> 0.96077), India +0.0010, US +0.0008, singleton accuracy 0.899 -> 0.939. The test has ~2x the decoys per S1 of train, so the floor should matter more on test.

## Kill condition
No significant paired gain over 005n.

## Result
(fill after er_eval)
