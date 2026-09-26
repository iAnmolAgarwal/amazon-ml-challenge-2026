#!/usr/bin/env bash
# After rebuild_chain.sh main: extra candidate sources (K9 keys + reverse text) on top of 023_comb -> 024_keys.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
L=logs/rebuild; mkdir -p $L
step() { local name=$1; shift; [ -f $L/$name.done ] && { echo "skip $name"; return; }; echo "[$(date +%T)] $name"; ( "$@" ) > $L/$name.log 2>&1; touch $L/$name.done; echo "[$(date +%T)] $name done"; }
while [ ! -f $L/predict023.done ]; do sleep 60; done
step keys_test $PY src/er_keys.py --split test
step rtext_test $PY src/er_rtext.py --split test --parent 023_comb
step union024 $PY src/er_newpairs.py union 024_keys --split test
step feat024 bash -c "$PY src/er_feat.py --split test --tag _k && $PY src/er_feat2.py --split test --tag _k && $PY src/er_feat5.py --split test --tag _k"
[ -e cache/models/024_keys ] || ln -s 017_route cache/models/024_keys
step score024 $PY src/er_newpairs.py score 024_keys --split test
step predict024 $PY src/er_predict.py 024_keys --from-probs 024_keys
step predict025 $PY src/er_predict.py 025_veto --from-probs 024_keys
echo "[$(date +%T)] post chain done"
