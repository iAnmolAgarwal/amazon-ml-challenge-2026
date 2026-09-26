#!/usr/bin/env bash
# Rebuild the pipeline on two workers (resumable: finished steps leave markers in logs/rebuild/).
# usage: scripts/rebuild_chain.sh main   (training side + models; waits for the test-side files)
#        scripts/rebuild_chain.sh aux    (test side: norm, blocking, features, retrieval; pushes files to host "g1")
#        scripts/rebuild_chain.sh single (everything on one machine)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
L=logs/rebuild; mkdir -p $L
ROLE=${1:-single}
step() { local name=$1; shift; [ -f $L/$name.done ] && { echo "skip $name"; return; }; echo "[$(date +%T)] $name"; ( "$@" ) > $L/$name.log 2>&1; touch $L/$name.done; echo "[$(date +%T)] $name done"; }
push() { local name=$1; shift; rsync -a --relative "$@" g1:aml/ && ssh g1 "mkdir -p aml/$L && touch aml/$L/$name.done"; echo "[$(date +%T)] pushed $name"; }
wait_for() { for m in "$@"; do while [ ! -f $L/$m.done ]; do sleep 30; done; done; }
test_side() {
  step norm_test $PY src/er_norm.py --split test
  step block_test $PY src/er_block.py --split test
  step feat_test $PY src/er_feat.py --split test
  step feat2_test $PY src/er_feat2.py --split test
  step feat5_test $PY src/er_feat5.py --split test
}
retrieval() {
  step ret_name $PY src/er_retrieve.py --split test --view name
  step ret_address $PY src/er_retrieve.py --split test --view address
  step union $PY src/er_newpairs.py union 017_route --split test
  step newfeat bash -c "$PY src/er_feat.py --split test --tag _new && $PY src/er_feat2.py --split test --tag _new && $PY src/er_feat5.py --split test --tag _new"
}
if [ "$ROLE" = aux ]; then
  step dict $PY src/er_dict.py
  test_side
  push test_side ./cache/norm/test_s1.parquet ./cache/norm/test_pool.parquet ./cache/cand/test_cand.parquet ./cache/feat/test_feat.parquet ./cache/feat/test_feat2.parquet ./cache/feat/test_feat5.parquet
  retrieval
  push retrieval ./cache/cand/test_ret_name.parquet ./cache/cand/test_ret_address.parquet ./cache/cand/test_cand_new.parquet ./cache/feat/test_feat_new.parquet ./cache/feat/test_feat2_new.parquet ./cache/feat/test_feat5_new.parquet
  exit 0
fi
step dict $PY src/er_dict.py
step norm_train $PY src/er_norm.py --split train
step block_train $PY src/er_block.py --split train
step feat_train $PY src/er_feat.py --split train
step feat2_train $PY src/er_feat2.py --split train
[ "$ROLE" = single ] && test_side
step s1_oof $PY src/er_eval.py 005n_indic_nocoh
step feat5_train $PY src/er_feat5.py --split train
step feat6_train $PY src/er_feat6.py --split train
wait_for test_side
step s1_test $PY src/er_predict.py 005f_floor --train-frac 1.0
step feat4_train $PY src/er_feat4.py --split train --src 005n_indic_nocoh
step feat4_test $PY src/er_feat4.py --split test --src 005f_floor
step s2_oof $PY src/er_eval.py 006_stage2
step s2_test $PY src/er_predict.py 006_stage2 --train-frac 1.0
step feat6_test $PY src/er_feat6.py --split test
step lift $PY src/er_lift.py
step band $PY src/er_stage3.py 016_stage2b
[ "$ROLE" = single ] && retrieval
wait_for retrieval
step route_fit $PY src/er_newpairs.py fit 017_route
step route_score $PY src/er_newpairs.py score 017_route --split test
[ -e cache/oof/017_route.parquet ] || ln -s 016_stage2b.parquet cache/oof/017_route.parquet
step lexical_cv $PY src/er_lexical.py 018_lexical --cv
step lexical $PY src/er_lexical.py 018_lexical
step predict019 $PY src/er_predict.py 019_band --from-probs 018_lexical
step gen_train $PY src/er_gen.py --split train --parent 018_lexical --workers 12
step gen_test $PY src/er_gen.py --split test --parent 018_lexical --workers 12
step comb $PY src/er_comb.py 023_comb
step predict023 $PY src/er_predict.py 023_comb --from-probs 023_comb
echo "[$(date +%T)] rebuild done"
