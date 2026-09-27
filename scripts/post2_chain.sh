#!/usr/bin/env bash
# After post_chain.sh: blend grid with the in-country stack model family (026_blend), then the candidate cascade check.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
L=logs/rebuild
step() { local name=$1; shift; [ -f $L/$name.done ] && { echo "skip $name"; return; }; echo "[$(date +%T)] $name"; ( "$@" ) > $L/$name.log 2>&1; touch $L/$name.done; echo "[$(date +%T)] $name done"; }
while [ ! -f $L/predict025.done ]; do sleep 60; done
step blend_grid $PY src/er_blend.py 026_blend --grid
step cascade_check $PY research/check_cascade.py 023_comb
echo "[$(date +%T)] post2 done"
