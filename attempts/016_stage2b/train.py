FLOOR = 0.6
PARENT = "006_stage2"
BAND = 0.001
FEATS3 = ["feat", "feat2", "feat4", "feat5p", "feat6l"]
EXCLUDE3 = ["a_ratio", "a_partial"]
INIT = False
USE_LOGIT = False
ROUNDS3 = 600
PARAMS3 = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42, num_threads=8)
DEPENDS = ["006_stage2", "005n_indic_nocoh", "005f_floor"]
