FLOOR = 0.6
SRC = "010_route"
GATE_SRC = "006_stage2"
GATE = 0.01
LEX_PARAMS = dict(objective="binary", num_leaves=9, min_data_in_leaf=100, learning_rate=0.04, lambda_l2=3.0, seed=42, num_threads=8, verbose=-1, deterministic=True, force_col_wise=True)
LEX_ROUNDS = 100
DEPENDS = ["010_route", "009_stage3m", "006_stage2", "005n_indic_nocoh", "005f_floor"]
PRE_STEPS = [
    "# all steps of 010_route (see attempts/010_route/train.py PRE_STEPS), then:",
    "python src/er_newpairs.py score 010_route --split train     # route OOF on the train sample owners (optional, for the CV)",
    "python src/er_predict.py 010_route --from-probs 010_route   # test probabilities of 009 + route",
    "python src/er_lexical.py 012_lexical --cv                  # optional: cross-fitted out-of-fold train probabilities (CV)",
    "python src/er_lexical.py 012_lexical                       # lexical correction of test probabilities",
]
EVAL = "python src/er_eval.py 012_lexical --from-oof 012_lexical   # optional: CV of the lexical correction"
PREDICT = "python src/er_predict.py 012_lexical --from-probs 012_lexical  # write submissions/NN_012_lexical.tsv (+ .candidates.tsv)"
