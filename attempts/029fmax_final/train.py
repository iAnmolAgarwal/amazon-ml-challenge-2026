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
DEPENDS = ["030q_fix", "030p_fix", "027p_meta", "024f_keys", "028k_ce", "028_ce", "024_keys", "023_comb", "019_band", "018_lexical", "017_route", "016_stage2b", "006_stage2", "005n_indic_nocoh", "005f_floor"]
EVAL = "python src/er_eval.py 029fmax_final --from-oof 030p_fix   # optional: 5-fold CV macro F0.5 of the final out-of-fold probabilities (US + India; France has no labels)"
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
    "# GPU (Kaggle, 2x T4): upload cache/ce/{records,pairs,band} as datasets; run src/ce/ce_train.py (e5-small, halves by folds {0,1} / {2,3,4}, ~170 min) and src/ce/ce_score.py (~22 min); download the scores to cache/ce/out/s",
    "# GPU again for e5-base: the same two scripts with ce_cfg.json {\"model\": \"intfloat/multilingual-e5-base\", \"lr\": 3e-5, \"budget_min\": 140}; download to cache/ce/out/b",
    "python src/ce/ce_export.py collect cache/ce/out/s ce && python src/ce/ce_export.py collect cache/ce/out/b ce_base   # cache/feat/{split}_ce",
    "python src/er_ce.py 028_ce && python src/er_ce.py 028k_ce --no-cv   # cross-encoder stack on the band (train CV + test on 024 pairs)",
    "python src/er_extend.py union 028e_ce 028_ce 024_keys 023_comb   # train file with the sampled extra candidates",
    "python src/er_keys.py --split train --tag _all && python src/er_rtext.py --split train --parent 023_comb --tag _all   # extra candidates for every train S1",
    "python src/er_newpairs.py union 024f_keys --split train",
    "python src/er_feat.py --split train --tag _allk && python src/er_feat2.py --split train --tag _allk && python src/er_feat5.py --split train --tag _allk",
    "ln -s 017_route cache/models/024f_keys",
    "python src/er_newpairs.py score 024f_keys --split train          # out-of-fold route-classifier scores",
    "python src/er_extend.py union 028f_ce 028_ce 024f_keys 023_comb   # meta-model train file with all extra candidates",
    "python src/ce/ce_export.py extra test 018_lexical 028k_ce && python src/ce/ce_export.py extra train 018_lexical 028e_ce 028f_ce   # added pairs to score",
    "# GPU: upload cache/ce/extra; run src/ce/ce_score.py with the e5-small models and score_cfg.json {\"train\": \"train_x1.parquet\", \"test\": \"test_x.parquet\"} -> cache/ce/out_x1/s, then {\"train\": \"train_x2.parquet\"} -> cache/ce/out_x2/s; with the e5-base models {\"train\": \"train_x1n.parquet\", \"test\": \"test_xn.parquet\"} -> cache/ce/out_x1/b, then {\"train\": \"train_x2n.parquet\"} -> cache/ce/out_x2/b",
    "python src/er_extend.py parity ce_par ce=cache/ce/out/s ce_base=cache/ce/out/b && python src/er_extend.py parity ce_x_par ce=cache/ce/out_x1/s ce_base=cache/ce/out_x1/b   # test: one half model per S1",
    "python src/ce/ce_export.py collectx ce=cache/ce/out_x1/s,cache/ce/out_x2/s ce_base=cache/ce/out_x1/b,cache/ce/out_x2/b && python src/ce/ce_export.py ce2   # cross-encoder side-car for all candidates",
    "# in-country stack model family: run the command list in src/stack/README.md (own environment, src/stack/requirements.txt; CPU steps + Kaggle GPU cross-encoders); it writes cache/stack/",
    "python src/single/build.py cache/r19s 006_stage2 006_stage2 train test && python src/single/fit.py A lgb - 800   # single flat LightGBM on the stage-2 band (out-of-fold + test)",
    "python src/single/sidecar.py A a2 028f_ce 028k_ce   # its logits as a meta-model side-car (cache/feat/{split}_r19a2)",
    "python src/er_meta.py 027p_meta                    # meta-model over both families, US + India (train CV + test)",
    "python src/er_fix.py 030p_fix                      # empty-address correction",
    "python src/er_france.py 029f5_final                # France: logit blend with the stack family and the cross-encoder, renamed-pair rule, decoy-word veto",
    "python src/er_fix.py 030q_fix --no-cv              # empty-address correction fitted on US + India, applied to France",
]
PREDICT = "python src/er_predict.py 029fmax_final --from-probs 029fmax_final  # write submissions/archive/NN_029fmax_final.tsv (+ .candidates.tsv)"
