TRAIN_FRAC = 0.5
EXTRA_FEATS = ["feat2", "feat4"]
EXCLUDE = ["a_ratio", "a_partial"]
FLOOR = 0.6
PARENT = "006_stage2"
FEATS3 = ["feat5", "feat6"]
BAND = 0.001
ROUNDS3 = 300
PARAMS3 = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42, num_threads=8)
DEPENDS = ["006_stage2", "005n_indic_nocoh", "005f_floor"]
PRE_STEPS = [
    "python src/er_eval.py 005n_indic_nocoh             # stage 1: out-of-fold probabilities -> cache/oof/005n_indic_nocoh.parquet",
    "python src/er_predict.py 005f_floor --train-frac 1.0  # stage 1: test probabilities -> cache/oof/test_005f_floor.parquet",
    "python src/er_feat4.py --split train --src 005n_indic_nocoh  # competition features (train)",
    "python src/er_feat4.py --split test --src 005f_floor         # competition features (test)",
    "python src/er_eval.py 006_stage2                   # stage 2: out-of-fold probabilities -> cache/oof/006_stage2.parquet",
    "python src/er_predict.py 006_stage2                # stage 2: test probabilities -> cache/oof/test_006_stage2.parquet",
    "python src/er_feat5.py --split train               # pair features: name/address specificity, house relations, legal forms",
    "python src/er_feat5.py --split test",
    "python src/er_feat6.py --split train               # name-operator features: label-free decoy-marker scores",
    "python src/er_feat6.py --split test",
    "python src/er_stage3.py 009_stage3m                 # stage 3: correction of stage-2 probabilities (CV + test)",
]
EVAL = "python src/er_eval.py 009_stage3m --from-oof 009_stage3m   # optional: 5-fold CV of stage 3, macro F0.5"
PREDICT = "python src/er_predict.py 009_stage3m --from-probs 009_stage3m  # decide on stage-3 test probabilities, write submissions/NN_009_stage3m.tsv (+ .candidates.tsv)"
