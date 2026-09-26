FLOOR = 0.6
CAND_BAND = ("006_stage2", 0.001)
PARENT = "018_lexical"
BAND = (0.003, 0.997)
FEATSC = ["feat", "feat2", "feat4", "feat5p", "feat6l"]
EXCLUDEC = ["a_ratio", "a_partial"]
PC_COLS = ["n_tset", "n_jw", "a_tset", "num_jacc", "house_eq", "street_eq", "c_jw", "ns_cos", "as_cos", "lg_eq", "dm_add"]
PARAMS_A = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=300, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, seed=42, num_threads=12, verbose=-1)
ROUNDS_A = 800
PARAMS_B = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=300, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, seed=42, num_threads=12, verbose=-1)
ROUNDS_B = 400
DEPENDS = ["018_lexical", "017_route", "016_stage2b", "006_stage2", "005n_indic_nocoh", "005f_floor"]
PRE_STEPS = [
    "# all steps of 018_lexical, then:",
    "python src/er_gen.py --split train --parent 018_lexical && python src/er_gen.py --split test --parent 018_lexical",
    "python src/er_comb.py 023_comb",
]
PREDICT = "python src/er_predict.py 023_comb --from-probs 023_comb  # write submissions/NN_023_comb.tsv (+ .candidates.tsv)"
