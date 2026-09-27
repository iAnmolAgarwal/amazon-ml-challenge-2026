FLOOR = 0.6
MAIN = "028e_ce"
MAIN_TEST = "028k_ce"
MIN_P = 0.001
STACK_DIR = "cache/stack"
CASCADE = (0.02, 10)
STACK_CE = "parity"
FEATS = ["ce2", "feat5"]
FEATS_TEST = {"feat5": "feat5p", "ce2": "ce2_par"}
FEATS_ROUTE = {"feat5": "feat5_new"}
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=500, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, seed=42, num_threads=12, verbose=-1, deterministic=True, force_col_wise=True)
ROUNDS = 600
DEPENDS = ["028e_ce", "028k_ce", "028_ce", "024_keys", "023_comb", "018_lexical", "017_route", "016_stage2b", "006_stage2", "005n_indic_nocoh", "005f_floor"]
PRE_STEPS = [
    "# all steps of 028e_ce and 028k_ce (cache/feat/{train,test}_ce.parquet, train_feat5, test_feat5p, test_feat5_new come from those chains; ce2 = the band cross-encoder scores plus the fill-in scores for pairs outside the band: train_ce2 = train_ce + train_ce_x, test_ce2_par = test_ce_par + test_ce_x_par), then:",
    "# in-country stack model family (src/stack): its out-of-fold train and test pair probabilities and cross-encoder scores in cache/stack",
    "python src/er_meta.py 027l_meta",
]
PREDICT = "python src/er_predict.py 027l_meta --from-probs 027l_meta"
