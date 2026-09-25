TRAIN_FRAC = 0.5
EXTRA_FEATS = ["feat2", "feat4"]
EXCLUDE = ["a_ratio", "a_partial"]
FLOOR = 0.6
DEPENDS = ["007a_stage1"]
PRE_STEPS = [
    "python src/er_eval.py 007a_stage1 --tag _v3            # stage 1: out-of-fold probabilities -> cache/oof/007a_stage1_v3.parquet",
    "python src/er_predict.py 007a_stage1 --tag _v3 --train-frac 1.0  # stage 1: test probabilities",
    "python src/er_feat4.py --split train --src 007a_stage1 --tag _v3",
    "python src/er_feat4.py --split test --src 007a_stage1 --tag _v3",
]
