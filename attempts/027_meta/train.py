FLOOR = 0.6
MAIN = "028_ce"
MAIN_TEST = "028k_ce"
MIN_P = 0.001
STACK_DIR = "cache/stack"
CASCADE = (0.02, 10)
STACK_CE = "mean"
FEATS = []
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, seed=42, num_threads=12, verbose=-1, deterministic=True, force_col_wise=True)
ROUNDS = 300
DEPENDS = ["028k_ce", "028_ce", "024_keys", "023_comb", "018_lexical", "017_route", "016_stage2b", "006_stage2", "005n_indic_nocoh", "005f_floor"]
PRE_STEPS = [
    "# all steps of 028_ce and 028k_ce, then:",
    "# in-country stack model family (src/stack): its out-of-fold train and test pair probabilities and cross-encoder scores in cache/stack",
    "python src/er_meta.py 027_meta",
]
PREDICT = "python src/er_predict.py 027_meta --from-probs 027_meta"
