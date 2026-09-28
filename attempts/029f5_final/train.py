FLOOR = 0.6
CAND_BAND = ("006_stage2", 0.001)
PARENT = "030p_fix"
FRANCE_BASE = "024_keys"
STACK_DIR = "cache/stack"
W = 0.5
RENAME = True
C = 0.1
THIRD = "cache/feat/test_ce.parquet"
THIRD_COL = "ce"
VETO_WORDS = ["participations", "holding", "distribution", "international"]
BROAD_WORDS = ["developpement", "france", "groupe"]
DEPENDS = ["030p_fix", "027p_meta", "024f_keys", "028k_ce", "028_ce", "024_keys", "023_comb", "018_lexical", "017_route", "016_stage2b", "006_stage2", "005n_indic_nocoh", "005f_floor"]
PRE_STEPS = [
    "# the command list of attempts/029f6_final up to python src/er_fix.py 030p_fix, then:",
    "python src/er_france.py 029f5_final",
]
PREDICT = "python src/er_predict.py 029f5_final --from-probs 029f5_final  # write submissions/archive/NN_029f5_final.tsv (+ .candidates.tsv)"
