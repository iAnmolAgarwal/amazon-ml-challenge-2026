# In-country stack model family: source for the stack probability and cross-encoder score files

This folder has the code that produces the three file sets the final stack reads from `cache/stack/`:

| Produced here | Name in `cache/stack/` | Content |
|---|---|---|
| `p_train_{India,US}.parquet` | same | `a_id, r_id, q, p2`: train in-country out-of-fold stage-3 score `q` (equal to `p2` outside the band 0.02 < p2 < 0.98) and stage-2 OOF `p2`. Folds are `crc32(S1 id) % 5`. Rows are kept if `q >= 0.001` or the pair is cascade-kept (`p2 > 0.02` and top 10 by `p2` per S1). |
| `p_test_{France,India,US}.parquet` | same | `a_id, r_id, p`: final test probability of the cascade-kept pairs. It is the stage-3 score in the band and stage 2 elsewhere. France already has the marker veto applied (`p = 0` on vetoed pairs). |
| `scores_{e5i,e5b}_{0,1}_{heldout,test}.parquet` | same | Copies of `$CE_DIR/out/scores_{e5i,e5b}_half{0,1}_{heldout,test}.parquet`, renamed. These are in-country cross-encoder logits `ce`. `half{h}_heldout` is the half-`h` train band scored by the model trained on the other half. `half{h}_test` is the whole test band scored by the half-`h` model. |

No external data is used. The only pretrained weights are `intfloat/multilingual-e5-small` (rev `614241f622f53c4eeff9890bdc4f31cfecc418b3`) and `intfloat/multilingual-e5-base` (rev `d128750597153bb5987e10b1c3493a34e5a4502a`), both MIT-licensed. They are fine-tuned on the provided training pairs only.

## Layout
- `src/`: the CPU pipeline. It covers normalisation, blocking, pair features, the two-stage LightGBM, band export, stage-3 features, the stage-3 stack (`stack_ce.py`) and the export (`export_scores.py`).
- `kaggle/ce.py`: the in-country (`e5i`, `e5b`) and cross-country (`e5s`) cross-encoders.
- `kaggle/ce_france.py`: the cross-country cross-encoder for the country without labels (`e5bc`).
- `kaggle/requirements.txt`: pins for the GPU step.
- `configs/`: the stage-2 experiment configs used by the chain.
- `utils/validate_submission.py`: the organisers' validator.
- `Makefile`: the original make targets, kept unchanged. It does **not** cover every step below; use the command list.

## Setup
```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
export PYTHONUTF8=1 PY=.venv/bin/python CE_DIR=$PWD/ce_work     # CE_DIR keeps band/ and out/ inside this folder
export MEM_LIMIT_GB=48 MEM_MIN_AVAIL_GB=2                        # memory watchdog (src/memguard.py); defaults assume 8 GB
```
Data goes in `dataset/train/train_source{1,2,3}.tsv`, `dataset/train/train_ground_truth.tsv` and `dataset/test/test_source{1,2,3}.tsv`.

## Run order (raw data to the three file sets)
```bash
# 1. transliteration dictionaries mined from train pairs -> cache/translit.json, cache/translit_addr.json (required: normalize silently uses empty dictionaries without them)
$PY src/translit.py names addr
# 2. normalise, block, pair features, name-specificity features
$PY src/normalize.py train test
$PY src/pairs.py train && $PY src/pairs.py test
$PY src/feat_spec.py --split train && $PY src/feat_spec.py --split test
# 3. stage-2 LightGBM models (5 entity folds, OOF -> cache/exp_<name>/p2_<country>.npy) and their test p2
R='{"type": "expected_f", "min_p": 0.5}'
$PY src/run_experiment.py --config configs/final_s25_spec.yaml
$PY src/predict.py --models models/final_s25_spec --n_models 5 --rule "$R"                       # -> cache/predict_work
$PY src/feat_decoy.py --split train && $PY src/feat_decoy.py --split test --only France
$PY src/run_experiment.py --config configs/final_s25dk.yaml
$PY src/run_experiment.py --config configs/final_s50_spec.yaml
$PY src/run_experiment.py --config configs/final_s25.yaml
$PY src/predict.py --models models/final_s25dk --n_models 5 --only France --out output_dk --work cache/predict_work_dk --rule "$R"
$PY src/predict.py --models models/final_s50_spec --n_models 5 --out output_s50 --work cache/predict_work_s50 --rule "$R"
$PY src/predict.py --models models/final_s25 --n_models 5 --only France --out output_nospec --work cache/predict_work_nospec --rule "$R"
for e in loco_s25spec_in2us loco_s25spec_us2in loco_s25_in2us loco_s25_us2in; do $PY src/run_experiment.py --config configs/$e.yaml; done
# 4. uncertain-band exports (raw texts) for the cross-encoders
$PY src/export_band.py --split train --exp final_s25_spec --out $CE_DIR/band      # train_band.parquet (OOF p2, fold, y)
$PY src/export_band.py --split test --out $CE_DIR/band                            # test_band.parquet
$PY src/export_extra.py --out $CE_DIR/band_extra                                   # extra_heldout.parquet, france_band.parquet
# 5. GPU (Kaggle), see the next section -> $CE_DIR/out/scores_{e5s,e5i,e5b,e5bc}_*.parquet
# 6. stage-3 band features
for s in train test; do
  $PY src/stage3_feats.py --split $s; $PY src/decoy_feats.py --split $s
  $PY src/feat_ops.py --split $s;     $PY src/band_feats.py --split $s
done
# 7. p_train: in-country stage-3 OOF (saved to cache/ic_q/e1_{India,US}.npz)
SAVE_IC=e1 $PY src/stack_ce.py eval --tag e5s --tag2 e5i --tag3 e5b --extra --p2b cache/exp_final_s50_spec \
  --decoy --decoy2 --ops --pairf --params '{"num_leaves": 127, "min_data_in_leaf": 100}' --rounds 800
# 8. p_test: final test build (saved to output_v08/final_p_{France,India,US}.npz; also writes the TSVs)
STACK_PARAMS='{"num_leaves": 127, "min_data_in_leaf": 100}' STACK_ROUNDS=800 $PY src/stack_ce.py test \
  --tag e5s --tag2 e5i --tag3 e5b --extra --p2b cache/exp_final_s50_spec,cache/predict_work_s50 \
  --cand_t 0.02 --cand_k 10 --rule_unseen '{"T": 2.0, "min_p": 0.6}' --unseen_blend cache/predict_work_dk \
  --decoy --decoy2 --ops --pairf --unseen_extra --unseen_ce e5bc --unseen_veto 1.5 --out output_v08
# 9. keyed exports -> cache/stack/
$PY src/export_scores.py --ic e1 --build output_v08 --out ../../cache/stack
```
Dependencies between the file sets:
- **scores_e5i/e5b** need steps 1-5.
- **p_train** needs steps 1-7. It uses the `e5s`, `e5i` and `e5b` scores.
- **p_test** needs steps 1-8. India and US use `e5s`, `e5i` and `e5b`. France uses the DK blend, `e5bc`, the decoy stacker, the rule T 2 / min_p 0.6 and the marker veto at lift 1.5.

## GPU steps (Kaggle, 2 x T4, internet on for the model download or a pre-downloaded pinned revision)
Upload `$CE_DIR/band/{train,test}_band.parquet` (and, for `e5bc`, `$CE_DIR/band_extra/*.parquet`) as a private Kaggle dataset. The scripts find the files under `/kaggle/input/**`.

Run the script as a GPU kernel with these environment values. Set them in the kernel before the script runs, for example with `os.environ.setdefault(...)` at the top. Then download `/kaggle/working/scores_*.parquet` into `$CE_DIR/out/`.

| Tag | Script | Settings | Outputs | Time |
|---|---|---|---|---|
| e5s | `kaggle/ce.py` | defaults (`CE_MODE=country`, e5-small, `CE_MAXLEN=96 CE_BATCH=128 CE_LR=5e-5 CE_TRAIN_MIN=110`) | `scores_e5s_{US,India}_{heldout,test}` | ~47 min |
| e5i | `kaggle/ce.py` | `CE_MODE=incountry CE_TAG=e5i` (other values default) | `scores_e5i_half{0,1}_{heldout,test}` | ~70 min-2 h |
| e5b | `kaggle/ce.py` | `CE_MODE=incountry CE_TAG=e5b CE_MODEL=intfloat/multilingual-e5-base CE_REVISION=d128750597153bb5987e10b1c3493a34e5a4502a CE_MAXLEN=128 CE_BATCH=64 CE_LR=3e-5 CE_TRAIN_MIN=200` | `scores_e5b_half{0,1}_{heldout,test}` | ~4.7 h |
| e5bc | `kaggle/ce_france.py` | defaults (e5-base, `CE_TAG=e5bc CE_AUG=0 CE_MAXLEN=128 CE_BATCH=64 CE_LR=3e-5 CE_TRAIN_MIN=75`) | `scores_e5bc_{US,India}_{heldout,france}` | not recorded (75 min training budget per model) |

In-country halves are split by band fold parity: `fold % 2 == h` is half `h`. Each run uses seed 0, fp16 autocast, linear warmup of 5 %, one epoch, and stops early at the `CE_TRAIN_MIN` time budget.

## Hardware used originally
The steps ran on a mix of machines:
- A 16 GB laptop and AWS instances with 8-64 GB RAM ran the CPU steps. Blocking takes 25-60 min per country block. Stage-2 training takes ~35 min per model on 8 vCPU / 64 GB.
- Kaggle 2 x T4 ran the cross-encoders.

## Not reproducible from this code alone (honest gaps)
1. **Export step (step 9).**
   - The original export was an unrecorded one-off.
   - `src/export_scores.py` reconstructs it from the documented format, the `SAVE_IC` cache written by `stack_ce.py eval`, and the `final_p_*.npz` written by `stack_ce.py test`.
   - Its row filter matches the delivered files. Every row with `q < 0.001` has `p2 > 0.02` and a within-S1 `p2` rank of at most 10, and `q == p2` on 100 % of out-of-band rows. It has not been run end to end.
2. **Exact command lines of steps 7-8.**
   - These were not logged verbatim. The flags are rebuilt from the recorded configuration: `--decoy --decoy2 --ops --pairf`, 127 leaves, min_data 100, 800 rounds, the cache tag `e1` and the build directory `output_v08`.
   - At default settings, the current code behaves the same as the code of that build. The later additions only activate through the `MONO` and `UNSEEN_ADDR_SHIFT` environment variables, or write an extra `all_p_*.npy`.
3. **Kaggle kernels.**
   - The kernel wrappers for `e5s`, `e5i`, `e5b` and `e5bc`, and the dataset upload, were manual and are not in the code. Their settings come from the original Makefile notes and the script defaults.
   - Training is cut by a wall-clock budget (`CE_TRAIN_MIN`) under fp16. A rerun can see a different number of steps, so the scores will be close but not bit-identical to the delivered files.
4. **`e5bc` France band.** It was produced with `export_extra.py` at its defaults: the France band from the spec + no-spec (`final_s25`) blend, and LOCO extras from `loco_s25spec` / `loco_s25`. This is inferred from the change history. The `FR_NS`/`EXTRA_NS` variants were added later for other screens. Band pairs of the France DK blend that lack an `e5bc` score get the median score, by design.
5. **Transliteration dictionaries.**
   - The original run used dictionaries mined once and kept as data files; they are not included here.
   - Step 1 re-mines them with the same code from the train pairs. Byte-identity with the originals has not been checked.
6. **France pre-veto scores** are not stored anywhere. `p_test_France` has the veto applied.
7. **Makefile.** It is kept as found. It lacks the `final_s25` no-spec France prediction, the four LOCO experiments, `feat_ops.py`/`band_feats.py`, and the `--ops --pairf` / `STACK_PARAMS` / `STACK_ROUNDS` / `SAVE_IC` settings. The command list above is the complete order.
