FLOOR = 0.6
CAND_BAND = ("006_stage2", 0.001)
PARENT = "023_comb"
TEST_PARENT = "024_keys"
BAND_SRC = "018_lexical"
BAND = (0.003, 0.997)
CE_COLS = ["ce"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=1.0, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=10, verbose=-1, seed=42)
ROUNDS = 300
DEPENDS = ["028_ce", "024_keys", "023_comb", "018_lexical", "017_route", "016_stage2b", "006_stage2", "005n_indic_nocoh", "005f_floor"]
PRE_STEPS = [
    "# all steps of 028_ce and 024_keys, then (test only; train CV is that of 028_ce):",
    "python src/er_ce.py 028k_ce --no-cv",
]
PREDICT = "python src/er_predict.py 028k_ce --from-probs 028k_ce"
