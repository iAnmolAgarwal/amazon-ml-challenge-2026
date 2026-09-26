#!/usr/bin/env bash
# Prepare a fresh GCP worker from the Mac: code, data, pinned Python env. No cloud credentials go to the worker.
# usage: scripts/gcp_bootstrap.sh <ssh-alias>
set -euo pipefail
H=$1
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for i in $(seq 1 30); do ssh -o ConnectTimeout=8 -o BatchMode=yes "$H" true 2>/dev/null && break; sleep 10; done
ssh "$H" 'sudo apt-get -qq update >/dev/null && sudo apt-get -qq install -y rsync tmux htop >/dev/null; command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; mkdir -p ~/aml/cache ~/aml/data'
rsync -az --exclude .venv --exclude cache --exclude data --exclude .git --exclude submissions --exclude logs "$ROOT/" "$H":~/aml/
rsync -az "$ROOT/data/student_resource" "$H":~/aml/data/
rsync -az "$ROOT/cache/raw" "$H":~/aml/cache/
ssh "$H" 'export PATH=$HOME/.local/bin:$PATH; cd ~/aml && [ -x .venv/bin/python ] || uv venv -q --python 3.12 .venv; uv pip install -q --python .venv/bin/python polars==1.44.2 polars-runtime-32==1.44.2 numpy==2.5.3 scipy==1.18.1 scikit-learn==1.9.1 lightgbm==4.7.0 rapidfuzz==3.14.6 sparse-dot-topn==1.2.0 pyarrow==25.0.1 pyyaml==6.0.3 psutil==7.2.2 && .venv/bin/python -c "import polars, lightgbm, sparse_dot_topn; print(\"env ok\")"; du -sh ~/aml/cache/raw ~/aml/data; nproc; free -g | sed -n 2p'
