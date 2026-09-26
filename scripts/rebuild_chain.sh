#!/usr/bin/env bash
# Rebuild the full pipeline up to 019_band on one machine (resumable: finished steps leave markers in logs/rebuild/).
# usage: scripts/rebuild_chain.sh [main|retrieval]   (retrieval = test char-3-gram retrieval only, for a second machine)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
L=logs/rebuild; mkdir -p $L
step() { local name=$1; shift; [ -f $L/$name.done ] && { echo "skip $name"; return; }; echo "[$(date +%T)] $name"; ( "$@" ) > $L/$name.log 2>&1; touch $L/$name.done; echo "[$(date +%T)] $name done"; }
if [ "${1:-main}" = "retrieval" ]; then
  step ret_name $PY src/er_retrieve.py --split test --view name
  step ret_address $PY src/er_retrieve.py --split test --view address
  exit 0
fi
step dict $PY src/er_dict.py
step norm_train $PY src/er_norm.py --split train
step norm_test $PY src/er_norm.py --split test
step block_train $PY src/er_block.py --split train
step block_test $PY src/er_block.py --split test
step feat_train $PY src/er_feat.py --split train
step feat_test $PY src/er_feat.py --split test
step feat2_train $PY src/er_feat2.py --split train
step feat2_test $PY src/er_feat2.py --split test
step s1_oof $PY src/er_eval.py 005n_indic_nocoh
step s1_test $PY src/er_predict.py 005f_floor --train-frac 1.0
step feat4_train $PY src/er_feat4.py --split train --src 005n_indic_nocoh
step feat4_test $PY src/er_feat4.py --split test --src 005f_floor
step s2_oof $PY src/er_eval.py 006_stage2
step s2_test $PY src/er_predict.py 006_stage2 --train-frac 1.0
step feat5_train $PY src/er_feat5.py --split train
step feat5_test $PY src/er_feat5.py --split test
step feat6_train $PY src/er_feat6.py --split train
step feat6_test $PY src/er_feat6.py --split test
step lift $PY src/er_lift.py
step band $PY src/er_stage3.py 016_stage2b
echo "[$(date +%T)] main chain done up to 016; route + lexical need the test retrieval files (cache/cand/test_ret_*.parquet)"
while [ ! -f cache/cand/test_ret_name.parquet ] || [ ! -f cache/cand/test_ret_address.parquet ]; do sleep 60; done
step union $PY src/er_newpairs.py union 017_route --split test
step newfeat bash -c "$PY src/er_feat.py --split test --tag _new && $PY src/er_feat2.py --split test --tag _new && $PY src/er_feat5.py --split test --tag _new"
step route_fit $PY src/er_newpairs.py fit 017_route
step route_score $PY src/er_newpairs.py score 017_route --split test
[ -e cache/oof/017_route.parquet ] || ln -s 016_stage2b.parquet cache/oof/017_route.parquet
step lexical $PY src/er_lexical.py 018_lexical
step predict $PY src/er_predict.py 019_band --from-probs 018_lexical
echo "[$(date +%T)] rebuild done"
