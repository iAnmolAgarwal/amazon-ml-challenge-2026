FLOOR = 0.6
CAND_BAND = ("006_stage2", 0.001)
PARENT = "018_lexical"
BAND = (0.005, 0.995)
FEATSP = ["feat", "feat2", "feat4", "feat5p", "feat6l"]
EXCLUDEP = ["a_ratio", "a_partial"]
PC_COLS = ["n_tset", "n_jw", "a_tset", "num_jacc", "house_eq", "street_eq", "c_jw", "ns_cos", "as_cos", "lg_eq", "dm_add"]
PARAMSP = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=42, num_threads=8, verbose=-1)
ROUNDSP = 600
FRANCE = "min"
DEPENDS = ["018_lexical", "017_route", "016_stage2b", "006_stage2", "005n_indic_nocoh", "005f_floor"]
