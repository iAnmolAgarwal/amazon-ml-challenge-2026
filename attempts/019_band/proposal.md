# 019_band

- parent: 018_lexical
- created: 2026-09-26

## Idea
Same probabilities as 018_lexical, but the candidate set is the band the final models score: pairs with 006 stage-2
probability >= 0.001 (the band stage 2, 016) plus the retrieval pairs added after it. Everything below the band keeps its
stage-2 probability < 0.001 and is never emitted, so stage 1/2 act as a learned filter. The decision runs on the band.

## Result
Train (out-of-fold, all 2,206,821 S1): 5.93 candidates per S1 instead of 44.48; F0.5 0.986181 for both (1 S1 differs);
gold pairs inside the candidates 98.117% vs 98.128%. Test: 7.25 candidates per S1 instead of 46.49 (US 6.79, India 6.81,
France 9.81); the 5,769,711 matched pairs are identical to 018.
