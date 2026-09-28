FLOOR = 0.6
CAND_BAND = ("006_stage2", 0.001)
COMB_PARENT = "018_lexical"
COMB_BAND = (0.003, 0.997)
COMB_FEATS = ["feat", "feat2", "feat4", "feat5p", "feat6l", "gen1-6", "pool_side_competition"]
COMB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255, seed=42, verbose=-1)
COMB_ROUNDS = 800
DEPENDS = ["020_gen", "018_lexical", "017_route", "016_stage2b", "006_stage2", "005n_indic_nocoh", "005f_floor"]
