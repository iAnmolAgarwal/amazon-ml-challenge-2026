# Reproduce the final submission end-to-end: data -> blocking -> features -> matching -> output.
# Run from this folder with the challenge data in ./dataset/{train,test}/ (see README.md).
PY ?= .venv/bin/python
export PYTHONUTF8 = 1

.PHONY: setup translit normalize blocking spec train train_aux predict band stack validate all

setup:               ## Python 3.11 venv with pinned requirements
	uv venv --python 3.11 .venv && uv pip install --python $(PY) -r requirements.txt

translit:            ## OPTIONAL: re-mine the non-Latin->Latin word dictionaries from TRAIN pairs (shipped in cache/)
	$(PY) src/translit.py names addr cov

normalize:           ## normalise names/addresses (applies cache/translit*.json) -> cache/norm_*.parquet
	$(PY) src/normalize.py train test

blocking:            ## candidate generation + pair features per country block, train and test
	$(PY) src/pairs.py train
	$(PY) src/pairs.py test

spec:                ## name-specificity / asymmetric-difference features added to the cached pairs
	$(PY) src/feat_spec.py --split train
	$(PY) src/feat_spec.py --split test

train:               ## two-stage LightGBM, 5 entity folds (OOF metrics) -> models/final_s25_spec
	$(PY) src/run_experiment.py --config configs/final_s25_spec.yaml

train_aux:           ## second stage-2 models: decoy-aware no-spec (France blend) and 50 % entity sample (stage-3 feature)
	$(PY) src/feat_decoy.py --split train
	$(PY) src/feat_decoy.py --split test --only France
	$(PY) src/run_experiment.py --config configs/final_s25dk.yaml
	$(PY) src/run_experiment.py --config configs/final_s50_spec.yaml
	$(PY) src/predict.py --models models/final_s25dk --n_models 5 --only France --out output_dk --work cache/predict_work_dk --rule '{"type": "expected_f", "min_p": 0.5}'
	$(PY) src/predict.py --models models/final_s50_spec --n_models 5 --out output_s50 --work cache/predict_work_s50 --rule '{"type": "expected_f", "min_p": 0.5}'

predict:             ## score test candidates, decide, write output/*.tsv
	$(PY) src/predict.py --models models/final_s25_spec --n_models 5 --rule '{"type": "expected_f", "min_p": 0.5}'

band:                ## export the uncertain band (0.02 < p < 0.98) of train OOF and test pairs for the pair model
	$(PY) src/export_band.py --split train --exp final_s25_spec --out ../kaggle/band
	$(PY) src/export_band.py --split test --out ../kaggle/band
	# then run kaggle/ce.py on a GPU (Kaggle 2xT4) with ../kaggle/band as input, three times; outputs -> ../kaggle/out:
	#   CE_MODE=country   (e5-small, per-country models, tag e5s; ~47 min)
	#   CE_MODE=incountry CE_TAG=e5i (e5-small, entity halves; ~2 h)
	#   CE_MODE=incountry CE_TAG=e5b CE_MODEL=intfloat/multilingual-e5-base CE_REVISION=d128750597153bb5987e10b1c3493a34e5a4502a CE_MAXLEN=128 CE_BATCH=64 CE_LR=3e-5 CE_TRAIN_MIN=200 (~4.7 h)
	# and kaggle/ce_france.py (e5-base cross-country CE per source country, tag e5bc): scores_e5bc_<US|India>_{heldout,france}.parquet

stack:               ## stage-3 sidecar features + stage 3: stack cross-encoder scores on the band, write output/*.tsv
	$(PY) src/stage3_feats.py --split train
	$(PY) src/stage3_feats.py --split test
	$(PY) src/export_extra.py --out ../kaggle/band_extra
	$(PY) src/decoy_feats.py --split train
	$(PY) src/decoy_feats.py --split test
	$(PY) src/stack_ce.py eval --tag e5s --tag2 e5i --tag3 e5b --extra --p2b cache/exp_final_s50_spec --decoy --decoy2
	$(PY) src/stack_ce.py test --tag e5s --tag2 e5i --tag3 e5b --extra --p2b cache/exp_final_s50_spec,cache/predict_work_s50 --cand_t 0.02 --cand_k 10 --rule_unseen '{"T": 2.0, "min_p": 0.6}' --unseen_blend cache/predict_work_dk --decoy --decoy2 --unseen_extra --unseen_ce e5bc --unseen_veto 1.5 --out output

validate:            ## official validator (must print PASS)
	$(PY) utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids

all: normalize blocking spec train train_aux predict band stack validate
