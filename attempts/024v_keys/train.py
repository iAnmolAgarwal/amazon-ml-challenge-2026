FLOOR = 0.6
CAND_BAND = ("006_stage2", 0.001)
OLD = "024_keys"
VIEWS = ("keysfr",)
EXCLUDE_OLD = True
RET_TAG = {"train": "_s10v", "test": "_v"}
NEW_TAG = {"train": "_s10kv", "test": "_kv"}
NEW_BASE = ["is_s3", "a_empty", "a_empty_p", "n_tok", "n_tok_p", "is_web_p", "n_ratio", "n_tset", "n_tsort", "n_partial", "n_jw", "j_ratio", "j_partial", "alt_ratio", "a_tset", "house_eq", "house_prefix", "street_eq", "postal_eq", "postal_conflict", "num_jacc", "nonlatin_name_p", "nonlatin_addr_p", "acronym", "join_eq", "nf_s1_s1", "nf_s1_pool", "nf_p_s1", "nf_p_pool", "c_ratio", "c_tset", "c_jw", "a_ratio_g", "a_partial_g"]
NEW_R5 = ["ns_sh", "ns_a1", "ns_b1", "ns_a1_frac", "ns_b1_frac", "ns_a1_max", "ns_b1_max", "ns_cos", "ns_n_a1", "ns_n_b1", "as_sh", "as_a1", "as_b1", "as_a1_max", "as_b1_max", "as_cos", "h_suffix", "h_diffl", "h_near", "h_near_st", "h_lev", "h_swap", "h_in_p", "h_in_s", "u_eq", "u_conf", "u_one", "lg_eq", "lg_sub", "lg_disj", "lg_one", "cos_nc3"]
NEW_PARAMS = dict(objective="binary", num_leaves=63, min_data_in_leaf=200, learning_rate=0.05, lambda_l2=1.0, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=42, num_threads=8, verbose=-1, deterministic=True, force_col_wise=True)
NEW_ROUNDS = 300
NEW_FILTER = 0.001
NEW_SAMPLE = 0.1
DEPENDS = ["024_keys", "023_comb", "018_lexical", "017_route", "016_stage2b", "006_stage2", "005n_indic_nocoh", "005f_floor"]
PRE_STEPS = [
    "# all steps of 024_keys, then (train check of the same key kinds where the country lacks them: th US, tn India):",
    "python src/er_keysfr.py --split train --keys 'us:th;india:tn' --queries cache/cand/s10_queries.parquet --tag _s10v",
    "python src/er_newpairs.py union 024v_keys --split train",
    "python src/er_feat.py --split train --tag _s10kv && python src/er_feat2.py --split train --tag _s10kv && python src/er_feat5.py --split train --tag _s10kv",
    "ln -s 017_route cache/models/024v_keys",
    "python src/er_newpairs.py score 024v_keys --split train",
]
