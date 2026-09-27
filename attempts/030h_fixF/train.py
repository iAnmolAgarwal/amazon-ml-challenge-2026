PARENT = "027h_meta"
BAND = (0.01, 0.99)
USE = ["pr", "twin", "flags", "raw"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, seed=42, num_threads=10, verbose=-1, deterministic=True, force_col_wise=True)
ROUNDS = 300
NONEMPTY = "full"
