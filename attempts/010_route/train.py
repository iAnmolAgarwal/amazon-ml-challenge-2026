FLOOR = 0.6
OLD = "009_stage3m"
RET_TAG = {"train": "_s10", "test": ""}
NEW_TAG = {"train": "_s10new", "test": "_new"}
NEW_BASE = ["is_s3", "a_empty", "a_empty_p", "n_tok", "n_tok_p", "is_web_p", "n_ratio", "n_tset", "n_tsort", "n_partial", "n_jw", "j_ratio", "j_partial", "alt_ratio", "a_tset", "house_eq", "house_prefix", "street_eq", "postal_eq", "postal_conflict", "num_jacc", "nonlatin_name_p", "nonlatin_addr_p", "acronym", "join_eq", "nf_s1_s1", "nf_s1_pool", "nf_p_s1", "nf_p_pool", "c_ratio", "c_tset", "c_jw", "a_ratio_g", "a_partial_g"]
NEW_R5 = ["ns_sh", "ns_a1", "ns_b1", "ns_a1_frac", "ns_b1_frac", "ns_a1_max", "ns_b1_max", "ns_cos", "ns_n_a1", "ns_n_b1", "as_sh", "as_a1", "as_b1", "as_a1_max", "as_b1_max", "as_cos", "h_suffix", "h_diffl", "h_near", "h_near_st", "h_lev", "h_swap", "h_in_p", "h_in_s", "u_eq", "u_conf", "u_one", "lg_eq", "lg_sub", "lg_disj", "lg_one", "cos_nc3"]
NEW_PARAMS = dict(objective="binary", num_leaves=63, min_data_in_leaf=200, learning_rate=0.05, lambda_l2=1.0, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=42, num_threads=8, verbose=-1, deterministic=True, force_col_wise=True)
NEW_ROUNDS = 300
NEW_FILTER = 0.001
NEW_SAMPLE = 0.1
DEPENDS = ["009_stage3m", "006_stage2", "005n_indic_nocoh", "005f_floor"]
PRE_STEPS = [
    "python src/er_eval.py 005n_indic_nocoh             # stage 1: out-of-fold probabilities",
    "python src/er_predict.py 005f_floor --train-frac 1.0  # stage 1: test probabilities",
    "python src/er_feat4.py --split train --src 005n_indic_nocoh  # competition features (train)",
    "python src/er_feat4.py --split test --src 005f_floor         # competition features (test)",
    "python src/er_eval.py 006_stage2                   # stage 2: out-of-fold probabilities",
    "python src/er_predict.py 006_stage2                # stage 2: test probabilities",
    "python src/er_feat5.py --split train && python src/er_feat5.py --split test   # pair features",
    "python src/er_feat6.py --split train && python src/er_feat6.py --split test   # name-operator features",
    "python src/er_stage3.py 009_stage3m                # stage 3 (CV + test)",
    "for v in name address; do python src/er_retrieve.py --split test --view $v; done   # char-3-gram retrieval (test)",
    "python src/er_newpairs.py union 010_route --split test          # new candidates = retrieved minus existing",
    "python src/er_feat.py --split test --tag _new && python src/er_feat2.py --split test --tag _new && python src/er_feat5.py --split test --tag _new",
    "python src/er_newpairs.py fit 010_route                         # new-pair classifier (existing train pairs)",
    "python src/er_newpairs.py score 010_route --split test          # keep new pairs with p > 0.001, append to 009",
]
EVAL = "# CV of the route: run er_retrieve on the train owners of the 10% sample (--queries, --tag _s10), er_newpairs union/score --split train"
PREDICT = "python src/er_predict.py 010_route --from-probs 010_route   # decide on 009 + route probabilities, write submissions/NN_010_route.tsv (+ .candidates.tsv)"
