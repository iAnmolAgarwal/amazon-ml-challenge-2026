# r19: can a simpler model match the stage chain?

27 Sep 2026. All numbers are 5-fold CV macro F0.5 on the harness folds (US + India train S1), out-of-fold everywhere, same decision rule (`decide`, floor 0.6). "Paired" = `research/verify.py paired` (mean per-fold difference, SE, positive folds; PASS = mean > 2 SE and >= 4/5 folds). No test labels or leaderboard feedback were used.

Code: `research/r19s/` (build.py builds the band matrix, fit.py fits one model on it, sidecar.py writes a meta-model side-car). Data on the GCP machines under `cache/r19s/`.

## Setup

- **Band:** pairs with stage-2 probability `006_stage2` >= 0.001 (12.95M train pairs, all positives; 10.93M test pairs), as the band model 016 used. Pairs below the band keep the 006 probability.
- **Features (205):** every existing side-car family joined once: feat (41, without a_ratio / a_partial), feat2 (9), feat4 stage-1 competition (11), feat5p (34), feat6l (5), generator-operator gen (87), cross-encoder ce / ce_base (2), pool-side competition from the 006 probability (pr_rank, pr_other, pr_margin, pr_sum, pr_n01, pr_n50) and pool-side contrasts (11 pc_* columns), country. gen and ce exist only on the 018 band (0.003 <= p018 <= 0.997, US/India); elsewhere they are missing (LightGBM handles missing values natively). No feature compares a pair with the S1's other candidates beyond what 006/016 already used (cs_* from feat4).
- **One model:** a single LightGBM trained from scratch on the band (no init score, no parent logit), 255 leaves, lr 0.08, 500 rounds, min_data_in_leaf 300, feature_fraction 0.8, bagging 0.8, L2 10. Test = mean of the 5 fold models.

## Results: one model instead of the correction chain

| Variant | Stages after stage 2 | Features | Learner | CV (US / India) | Paired |
|---|---|---|---|---|---|
| 016_stage2b (band model, reference) | 1 | ~115 | LightGBM | 0.98589 | - |
| 023_comb (chain 016 → route → lexical → final correction) | 4 | - | LightGBM x4 | 0.98742 | - |
| 028_ce (023 + cross-encoder stack) | 5 | - | LightGBM x5 | 0.98779 (0.98831 / 0.98702) | - |
| **A: one LightGBM, all 205 features** | **1** | 205 | LightGBM | **0.98779** (0.98832 / 0.98699) | vs 023 **+0.00036** (SE 0.00000, 5/5, PASS); vs 028 −0.00001 (SE 0.00001, 2/5: equal); vs 016 +0.00191 (5/5) |
| X: one XGBoost, same 205 features (hist, lossguide 255 leaves, depth 10, eta 0.08, 500 rounds) | 1 | 205 | XGBoost | 0.98778 (0.98830 / 0.98700) | vs 028 −0.00002 (1/5); vs A −0.00001 (2/5): equal |
| mean logit of A and X | 1 | 205 | both | 0.98779 | vs A +0.00001 (3/5): nothing |
| B: A without gen and ce (only chain-free side-cars) | 1 | 116 | LightGBM | 0.98638 (0.98719 / 0.98517) | vs 016 +0.00050 (5/5); vs 018 +0.00034 (5/5); vs 023 −0.00104 (0/5) |
| S: compact LightGBM, A's top-25 features by gain, 127 leaves | 1 | 25 | LightGBM | 0.98736 (0.98795 / 0.98648) | not paired (stopped; about equal to 023's 0.98742) |
| G: logistic GAM (16 features, 16 quantile bins each, one-hot + L2 logistic, 30% train sample per fold) | 1 | 16 | logistic | 0.98017 (0.98071 / 0.97935) | vs 006 +0.00591 (5/5) |
| G2: same GAM, 40 features | 1 | 40 | logistic | 0.98050 (0.98093 / 0.97984) | - |

G's 16 features: n_tset, n_jw, c_jw, a_tset, house_eq, street_eq, postal_eq, h_near, hd_shift, ns_cos, as_cos, dm_add, lg_disj, cp_margin, pr_margin, ce.

A's top gain shares (fold-0 model): pr_margin 0.69, cp_margin 0.17, st1_p 0.047, ce 0.041, dm_add 0.010, ce_base 0.006, dm_swap 0.005, pr_sum 0.003, then house relations (h_suffix, shift_d, hd_shift, hd, h_delta, h_diffl) and decoy markers (dm_drop) at 0.001 each. The competition margins carry the earlier-stage probability, so they take most of the gain; the pair features decide the hard cases.

What this shows:
- The stages after stage 2 that train out-of-fold (band model, lexical correction, generator-operator + pool-side correction, cross-encoder stack; the route adds test pairs only) reduce to **one LightGBM on the union of the same inputs**: same CV as 028_ce (paired difference −0.00001, SE 0.00001) and +0.00036 over 023_comb (5/5). The chain's gains came from the features, not from the residual structure.
- gen + ce are worth +0.00141 inside the single model (A vs B, 5/5; US +0.00113, India +0.00182). B still beats 016 and 018 (5/5): pool-side competition and contrasts in the band model are worth about +0.0005 by themselves.
- A different learner adds nothing: XGBoost on the same features equals LightGBM (−0.00001), and averaging them adds +0.00001 (3/5). The ceiling is the information in the features, not the learner.
- A GAM with 16 features reaches 0.980 (40 features: 0.9805), about 0.008 below the trees: interactions between the similarity views, the house relations and the competition margins matter (for example, a decoy house shift matters only when the name matches).

## Diversity for the meta-model

Each model's out-of-fold band logit was added as one side-car column (`l_<name>`) to the meta-model (src/er_meta.py FEATS), everything else unchanged. Main pairs outside the 006 band (route and key pairs; 268k train, 2.98M test) get the main logit as the value so the column is never missing on main pairs.

| Meta variant | CV (US / India) | Paired vs its base |
|---|---|---|
| r19m0 = 027d_meta rerun on g1 (4 threads) | 0.99046 | identical to 027d (difference 0.00000) |
| 027d + GAM G | 0.99063 (0.99080 / 0.99038) | vs 027d **+0.00017** (SE 0.00001, 5/5, PASS) |
| 027h_meta (current base, adds ce + feat5 to the meta) | 0.99074 (0.99086 / 0.99055) | - |
| 027h + GAM G | 0.99074 | vs 027h +0.00001 (SE 0.00001, 3/5, FAIL) |
| **027h + A** (`r19s_ha`) | **0.99077** (0.99090 / 0.99058) | vs 027h **+0.00004** (SE 0.00001, 5/5, PASS; US +0.00004, India +0.00003) |
| 027h + XGBoost X (`r19s_hx`) | 0.99076 (0.99089 / 0.99057) | vs 027h **+0.00003** (SE 0.00001, 5/5, PASS) |
| 027h + A + X (`r19s_hax`) | 0.99077 (0.99090 / 0.99057) | vs 027h +0.00003 (5/5, PASS); vs 027h + A −0.00000 (2/5): X adds nothing on top of A |

The GAM's +0.00017 over 027d is the same information 027h already added (ce and feat5 inside the meta): over 027h it gives nothing. The single model A (or its XGBoost twin) passes the rule over 027h, but by only +0.00003 to +0.00004, and the two together add nothing more.

Test guard for 027h + A (label-free, `verify.py guard 027h_meta r19s_ha`): pairs per S1 US 3.3877 → 3.3878, India 3.3729 → 3.3732, France unchanged (kept from the main pipeline); empty rate unchanged; house-conflict share unchanged (US 0.0877, India 0.1743, France 0.0102); 1,832 pairs added and 1,539 dropped out of about 5.8M emitted. 027h + X and 027h + A + X: the same (US 3.3877-3.3878 pairs per S1, India 3.3732, house conflict unchanged; 1,639 / 1,447 and 1,842 / 1,548 pairs added / dropped).

## Runtime (12-vCPU / 96 GB GCP machines)

| Step | Time |
|---|---|
| build.py: join the 7 side-car families + pool-side features into one band matrix | 345 s train, 527 s test (10.7 GB + 9.0 GB float32) |
| A, 5-fold CV + test (205 features, 255 leaves, 500 rounds, 8 threads) | 5,063 s + 490 s, on a machine shared with two other jobs; B (116 features) took 1,695 s on a free machine with 12 threads |
| X, XGBoost 5-fold CV + test | 2,411 s + 250 s (free machine, 12 threads) |
| G / G2 (GAM, 5 folds + test) | about 350 s CV each, 2-4 threads |
| meta-model with one extra side-car column (5 folds + test) | about 12-13 min |

For comparison, the chain it replaces is 016 (10-15 min), lexical (about 10 min), er_gen + er_comb and er_ce (each 5-fold with a full refit). The single model needs one build and one fit, but the same feature extractors (feat5/6, lift, er_gen, the e5 cross-encoder) still have to run first.

Not finished (stopped to free the machines for the final uploads): A1 (the same single model with band and pool-side features from stage 1 instead of stage 2, i.e. stage 1 → one model), and C (A without any competition features).

## Recommendation

1. **Simpler pipeline at the same accuracy: yes.** stage 1 → stage 2 (006, for the band and the competition margins) → **one LightGBM on the band with the union of the existing feature families** (feat, feat2, feat4, feat5p, feat6l, gen, e5 cross-encoder score, pool-side competition of the 006 probability and 11 pool-side contrasts) gives CV 0.98779, identical to 028_ce (paired −0.00001, SE 0.00001) and +0.00036 over 023_comb (5/5). It replaces five models (016 band, lexical, 020/023 correction, 028 cross-encoder stack) with one, with no residual boosting from a parent logit. Two points for the documentation: (a) gen and the cross-encoder score were computed on the 018 band only, so the single model still inherits that band for those two columns; (b) test route pairs (017) are not in the 006 band, so a submission built from this model still needs the route classifier's probabilities appended (`test_r19_A.parquet` contains 006-band pairs only). The learner does not matter (XGBoost = LightGBM), and a 16-feature GAM reaches 0.980. The GAM shows that interactions carry the last 0.008.
2. **Verified gain over the current final (027h_meta): only +0.00003 to +0.00004.** Adding the single model's logit as a meta input (`r19s_ha`) passes the paired rule (+0.00004, SE 0.00001, 5/5) and the guard is clean, but the expected LB change is about +0.00004, well below the gap to 0.990. I recommend not changing the final for it. If it is used anyway, the files are:
   - meta output in harness format (s1_idx, pool_idx, prob): g3 `cache/oof/r19s_ha.parquet` (train OOF) and `cache/oof/test_r19s_ha.parquet` (test; France = 028k_ce main probabilities, so the France blend must be applied afterwards exactly as for 027h);
   - its side-car input: g3 `cache/feat/{train,test}_r19a.parquet` (column l_A), attempt `attempts/r19s_ha/train.py` = 027h_meta + FEATS "r19a";
   - the single model itself: g1 `cache/oof/r19_A.parquet`, `cache/oof/test_r19_A.parquet` (006 probabilities outside the band); XGBoost twin: g1 `cache/oof/r19_X.parquet`, `test_r19_X.parquet`.

Caveat on the files: `fit.py` saves test predictions as `cache/r19s/test_<name>.npy`, so the model named X overwrote the test feature matrix `test_X.npy` on g1, g2 and g3 (predictions and side-cars are correct; rebuild the matrix with build.py before refitting on test).
