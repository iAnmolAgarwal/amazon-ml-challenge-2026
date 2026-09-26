FLOOR = 0.6
CAND_BAND = ("006_stage2", 0.001)
VETO = {"france": ["participations", "holding", "distribution", "international"]}
DEPENDS = ["024_keys", "023_comb", "018_lexical", "017_route", "016_stage2b", "006_stage2", "005n_indic_nocoh", "005f_floor"]
PREDICT = "python src/er_predict.py 025_veto --from-probs 024_keys  # France pure-decoy-word veto on top of 024"
