FLOOR = 0.6
CAND_BAND = ("006_stage2", 0.001)
PARENT = "023_comb"
BAND_SRC = "018_lexical"
BAND = (0.003, 0.997)
CE_COLS = ["ce"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=1.0, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=10, verbose=-1, seed=42)
ROUNDS = 300
DEPENDS = ["023_comb", "018_lexical", "017_route", "016_stage2b", "006_stage2", "005n_indic_nocoh", "005f_floor"]
PRE_STEPS = [
    "# all steps of 023_comb, then:",
    "python src/ce/ce_export.py records && python src/ce/ce_export.py pairs && python src/ce/ce_export.py band 018_lexical 0.003 0.997",
    "# upload cache/ce/{records,pairs,band} as private Kaggle datasets",
    "# Kaggle GPU T4 x2: src/ce/ce_train.py (e5-small, halves by folds {0,1} / {2,3,4}, ~170 min), then src/ce/ce_score.py (~22 min)",
    "python src/ce/ce_export.py collect <dir with ce_{A,B}_{train,test}.parquet> ce",
    "python src/er_ce.py 028_ce",
]
PREDICT = "python src/er_predict.py 028_ce --from-probs 028_ce"
