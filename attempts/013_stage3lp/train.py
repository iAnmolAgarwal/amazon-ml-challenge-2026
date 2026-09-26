TRAIN_FRAC = 0.5
EXTRA_FEATS = ["feat2", "feat4"]
EXCLUDE = ["a_ratio", "a_partial"]
FLOOR = 0.6
PARENT = "006_stage2"
FEATS3 = ["feat5p", "feat6l"]
BAND = 0.001
ROUNDS3 = 300
PARAMS3 = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42, num_threads=8)
DEPENDS = ["006_stage2", "005n_indic_nocoh", "005f_floor"]
