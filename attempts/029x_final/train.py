FLOOR = 0.6
CAND_BAND = ("006_stage2", 0.001)
CAND_MINP = 0.001
CAND_OWNER = True
PARENT = "027h_meta"
FRANCE_BASE = "024_keys"
STACK_DIR = "cache/stack"
W = 0.5
VETO_WORDS = ["participations", "holding", "distribution", "international"]
BROAD_WORDS = ["developpement", "france", "groupe"]
DEPENDS = ["027h_meta", "028k_ce", "028_ce", "024_keys", "023_comb", "019_band", "018_lexical", "017_route", "016_stage2b", "006_stage2", "005n_indic_nocoh", "005f_floor"]
EVAL = "python src/er_eval.py 029x_final --from-oof 027h_meta   # optional: 5-fold CV macro F0.5 of the final out-of-fold probabilities (US + India; France has no labels)"
PRE_STEPS = [
    "python src/er_dict.py                              # native-script -> Latin token maps learned from training pairs",
    "python src/er_norm.py --split train && python src/er_norm.py --split test                 # normalization views",
    "python src/er_block.py --split train && python src/er_block.py --split test               # candidate generation (blocking)",
    "python src/er_feat.py --split train && python src/er_feat.py --split test                 # pair features",
    "python src/er_feat2.py --split train && python src/er_feat2.py --split test               # name frequency, conservative names",
    "python src/er_eval.py 005n_indic_nocoh             # stage 1: out-of-fold probabilities",
    "python src/er_predict.py 005f_floor --train-frac 1.0  # stage 1: test probabilities",
    "python src/er_feat5.py --split train && python src/er_feat5.py --split test               # pair features",
    "python src/er_feat6.py --split train && python src/er_feat6.py --split test               # name-operator features",
    "python src/er_feat4.py --split train --src 005n_indic_nocoh && python src/er_feat4.py --split test --src 005f_floor   # competition features",
    "python src/er_eval.py 006_stage2                   # stage 2: out-of-fold probabilities",
    "python src/er_predict.py 006_stage2 --train-frac 1.0  # stage 2: test probabilities",
    "python src/er_lift.py                              # country-lifted marker scores (feat6l), French house repair (feat5p)",
    "python src/er_stage3.py 016_stage2b                # band stage 2 with every feature set (CV + test)",
    "for v in name address; do python src/er_retrieve.py --split test --view $v; done   # char-3-gram retrieval (test)",
    "python src/er_newpairs.py union 017_route --split test",
    "python src/er_feat.py --split test --tag _new && python src/er_feat2.py --split test --tag _new && python src/er_feat5.py --split test --tag _new",
    "python src/er_newpairs.py fit 017_route && python src/er_newpairs.py score 017_route --split test   # retrieval-route classifier",
    "cp cache/oof/016_stage2b.parquet cache/oof/017_route.parquet   # the route adds test pairs only; train out-of-fold = 016",
    "python src/er_lexical.py 018_lexical --cv && python src/er_lexical.py 018_lexical   # lexical correction (train out-of-fold, test)",
    "python src/er_gen.py --split train --parent 018_lexical --workers 12 && python src/er_gen.py --split test --parent 018_lexical --workers 12",
    "python src/er_comb.py 023_comb                     # final correction (generator-operator + pool-side features)",
    "python src/er_keys.py --split test && python src/er_rtext.py --split test --parent 023_comb   # extra candidate sources",
    "python src/er_newpairs.py union 024_keys --split test",
    "python src/er_feat.py --split test --tag _k && python src/er_feat2.py --split test --tag _k && python src/er_feat5.py --split test --tag _k",
    "ln -s 017_route cache/models/024_keys               # extra pairs are scored with the retrieval-route classifier",
    "python src/er_newpairs.py score 024_keys --split test",
    "python src/er_extend.py sample 10 3 cache/cand/s10_queries.parquet   # label-free 10% S1 sample for train-side extra candidates",
    "python src/er_keys.py --split train --queries cache/cand/s10_queries.parquet --tag _s10 && python src/er_rtext.py --split train --parent 023_comb --queries cache/cand/s10_queries.parquet --tag _s10",
    "python src/er_newpairs.py union 024_keys --split train",
    "python src/er_feat.py --split train --tag _s10k && python src/er_feat2.py --split train --tag _s10k && python src/er_feat5.py --split train --tag _s10k",
    "python src/er_newpairs.py score 024_keys --split train          # out-of-fold route-classifier scores for the sampled S1",
    "python src/ce/ce_export.py records && python src/ce/ce_export.py pairs && python src/ce/ce_export.py band 018_lexical 0.003 0.997",
    "# GPU (Kaggle, 2x T4): upload cache/ce/{records,pairs,band}; run src/ce/ce_train.py (e5-small, halves by folds {0,1} / {2,3,4}, ~170 min), then src/ce/ce_score.py (~22 min)",
    "python src/ce/ce_export.py collect <dir with ce_{A,B}_{train,test}.parquet> ce",
    "# GPU again for the e5-base score: same notebooks with ce_cfg.json {\"model\": \"intfloat/multilingual-e5-base\", \"lr\": 3e-5, \"budget_min\": 140}",
    "python src/ce/ce_export.py collect <dir with the e5-base ce_{A,B}_{train,test}.parquet> ce_base   # second column of cache/feat/{split}_ce",
    "python src/er_ce.py 028_ce && python src/er_ce.py 028k_ce --no-cv   # cross-encoder stack on the band (train CV + test on 024 pairs)",
    "python src/er_extend.py union 028e_ce 028_ce 024_keys 023_comb   # train out-of-fold main file with the sampled extra candidates",
    "# in-country stack model family: run the command list in src/stack/README.md (CPU steps + Kaggle GPU cross-encoders); it writes cache/stack/",
    "python src/er_extend.py parity ce_par ce=<dir with the e5-small ce_{A,B}_test.parquet> ce_base=<dir with the e5-base ce_{A,B}_test.parquet>   # one half model per test S1, as on train",
    "python src/er_meta.py 027h_meta                    # meta combination with cross-encoder and pair side-cars, US + India (train CV + test)",
    "python src/er_france.py 029x_final                  # France: logit blend with the stack family, decoy-word veto",
]
PREDICT = "python src/er_predict.py 029x_final --from-probs 029x_final  # write submissions/archive/NN_029x_final.tsv (+ .candidates.tsv)"
