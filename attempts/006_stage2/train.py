TRAIN_FRAC = 0.5
EXTRA_FEATS = ["feat2", "feat4"]
EXCLUDE = ["a_ratio", "a_partial"]
FLOOR = 0.6
DEPENDS = ["005n_indic_nocoh", "005f_floor"]
PRE_STEPS = [
    "python src/er_eval.py 005n_indic_nocoh             # stage 1: out-of-fold probabilities -> cache/oof/005n_indic_nocoh.parquet",
    "python src/er_predict.py 005f_floor --train-frac 1.0  # stage 1: test probabilities -> cache/oof/test_005f_floor.parquet",
    "python src/er_feat4.py --split train --src 005n_indic_nocoh  # competition features (train)",
    "python src/er_feat4.py --split test --src 005f_floor         # competition features (test)",
]
