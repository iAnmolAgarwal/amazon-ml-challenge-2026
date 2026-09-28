FLOOR = 0.6
SRC = "017_route"
GATE_SRC = "006_stage2"
GATE = 0.01
LEX_PARAMS = dict(objective="binary", num_leaves=9, min_data_in_leaf=100, learning_rate=0.04, lambda_l2=3.0, seed=42, num_threads=8, verbose=-1, deterministic=True, force_col_wise=True)
LEX_ROUNDS = 100
DEPENDS = ["017_route", "016_stage2b", "006_stage2", "005n_indic_nocoh", "005f_floor"]
