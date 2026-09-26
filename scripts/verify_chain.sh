#!/usr/bin/env bash
# Runs on the aux worker: pulls each stage's outputs from g1 as they appear and checks them against the original
# measurements (reproduction), the paired CV rule (gains) and the label-free test guard. Log: logs/verify/verify.log
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
V=logs/verify; mkdir -p $V cache/oof
LOG=$V/verify.log
waitg1() { until ssh g1 "test -f aml/logs/rebuild/$1.done"; do sleep 120; done; }
pull() { for f in "$@"; do rsync -a g1:aml/cache/oof/$f cache/oof/ 2>/dev/null || rsync -aL g1:aml/cache/oof/$f cache/oof/; done; }
pullatt() { rsync -a g1:aml/attempts/$1/results.json attempts/$1/ 2>/dev/null; }
run() { echo "[$(date +%T)] $*" >> $LOG; "$@" >> $LOG 2>&1; }
rsync -a g1:aml/data/folds.parquet data/ 2>/dev/null
waitg1 s1_oof; pullatt 005n_indic_nocoh; rsync -a g1:aml/data/folds.parquet data/; run $PY research/verify.py expect 005n_indic_nocoh 0.97008 0.0015
waitg1 s2_oof; pullatt 006_stage2; run $PY research/verify.py expect 006_stage2 0.97423 0.0015
waitg1 band; pullatt 016_stage2b; run $PY research/verify.py expect 016_stage2b 0.98589 0.0015
waitg1 lexical_cv; pull 016_stage2b.parquet 018_lexical.parquet; run $PY research/verify.py paired 016_stage2b 018_lexical
waitg1 predict019; pull test_016_stage2b.parquet test_017_route.parquet test_018_lexical.parquet; run $PY research/verify.py guard 016_stage2b 017_route 018_lexical
waitg1 comb; pull 023_comb.parquet test_023_comb.parquet; run $PY research/verify.py paired 018_lexical 023_comb; run $PY research/verify.py guard 018_lexical 023_comb
until ssh g1 "test -f aml/logs/rebuild/predict024.done"; do sleep 120; done; pull test_024_keys.parquet; run $PY research/verify.py guard 023_comb 024_keys
echo "[$(date +%T)] verify chain done" >> $LOG
