FLOOR = 0.6
CAND_BAND = ("006_stage2", 0.001)
GEN_PARENT = "018_lexical"
GEN_GROUPS = "12345"
GEN_BAND = (0.003, 0.997)
GEN_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=300, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, seed=42, verbose=-1)
GEN_ROUNDS = 400
DEPENDS = ["018_lexical", "017_route", "016_stage2b", "006_stage2", "005n_indic_nocoh", "005f_floor"]
