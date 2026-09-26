FLOOR = 0.6
SRC = "017_route"
GATE_SRC = "006_stage2"
GATE = 0.01
LEX_PARAMS = dict(objective="binary", num_leaves=9, min_data_in_leaf=100, learning_rate=0.04, lambda_l2=3.0, seed=42, num_threads=8, verbose=-1, deterministic=True, force_col_wise=True)
LEX_ROUNDS = 100
DEPENDS = ["017_route", "016_stage2b", "006_stage2", "005n_indic_nocoh", "005f_floor"]
PRE_STEPS = [
    "python src/er_eval.py 005n_indic_nocoh             # stage 1: out-of-fold probabilities",
    "python src/er_predict.py 005f_floor --train-frac 1.0  # stage 1: test probabilities",
    "python src/er_feat4.py --split train --src 005n_indic_nocoh  # competition features (train)",
    "python src/er_feat4.py --split test --src 005f_floor         # competition features (test)",
    "python src/er_eval.py 006_stage2                   # stage 2: out-of-fold probabilities",
    "python src/er_predict.py 006_stage2 --train-frac 1.0  # stage 2: test probabilities",
    "python src/er_feat5.py --split train && python src/er_feat5.py --split test   # pair features",
    "python src/er_feat6.py --split train && python src/er_feat6.py --split test   # name-operator features",
    "python src/er_lift.py                              # country-lifted marker scores (feat6l), French house repair (feat5p)",
    "python src/er_stage3.py 016_stage2b                # band stage 2 with every feature set (CV + test)",
    "for v in name address; do python src/er_retrieve.py --split test --view $v; done   # char-3-gram retrieval (test)",
    "python src/er_newpairs.py union 017_route --split test          # new candidates = retrieved minus existing",
    "python src/er_feat.py --split test --tag _new && python src/er_feat2.py --split test --tag _new && python src/er_feat5.py --split test --tag _new",
    "python src/er_newpairs.py fit 017_route                         # new-pair classifier (existing train pairs)",
    "python src/er_newpairs.py score 017_route --split test          # keep new pairs with p > 0.001, append to 016",
    "python src/er_lexical.py 018_lexical --cv                      # optional: cross-fitted out-of-fold train probabilities (CV)",
    "python src/er_lexical.py 018_lexical                           # lexical correction of test probabilities",
]
EVAL = "python src/er_eval.py 018_lexical --from-oof 018_lexical   # optional: CV of the full chain"
PREDICT = "python src/er_predict.py 018_lexical --from-probs 018_lexical  # write submissions/NN_018_lexical.tsv (+ .candidates.tsv)"
