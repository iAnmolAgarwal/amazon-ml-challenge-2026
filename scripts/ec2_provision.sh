#!/usr/bin/env bash
# Provision an EC2 worker from the Mac: point the SSH alias at its IP, copy code, pull data via
# short-lived presigned S3 URLs (no AWS credentials on the instance), build the Python env.
# usage: scripts/ec2_provision.sh <ssh-alias: amlc-a|amlc-b> <instance-id>
set -euo pipefail
ALIAS=$1; IID=$2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export AWS_PROFILE=${AWS_PROFILE:-amlc} AWS_REGION=${AWS_REGION:-us-east-1}
BUCKET=amlc-2026-c77a54ac
aws ec2 wait instance-running --instance-ids "$IID"
IP=$(aws ec2 describe-instances --instance-ids "$IID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
python3 - "$ALIAS" "$IP" <<'EOF'
import re, sys, pathlib
alias, ip = sys.argv[1], sys.argv[2]
p = pathlib.Path.home() / ".ssh" / "config"
s = p.read_text()
s = re.sub(rf"(Host {alias}\n  HostName )\S+", rf"\g<1>{ip}", s)
p.write_text(s)
EOF
ssh-keygen -R "$IP" >/dev/null 2>&1 || true
echo "$ALIAS -> $IP ($IID)"
for i in $(seq 1 30); do ssh -o ConnectTimeout=8 -o BatchMode=yes "$ALIAS" true 2>/dev/null && break; sleep 10; done
ssh "$ALIAS" 'cloud-init status --wait >/dev/null 2>&1; mkdir -p ~/aml/cache/raw'
rsync -az --exclude .venv --exclude cache --exclude data --exclude .git "$ROOT/" "$ALIAS":~/aml/
{
  echo 'set -e; cd ~/aml/cache/raw'
  for f in $(aws s3 ls "s3://$BUCKET/raw/" | awk '{print $4}'); do
    echo "[ -s $f ] || curl -sSf -o $f '$(aws s3 presign "s3://$BUCKET/raw/$f" --expires-in 1800)'"
  done
} | ssh "$ALIAS" bash -s
ssh "$ALIAS" 'export PATH=$HOME/.local/bin:$PATH; cd ~/aml && [ -x .venv/bin/python ] || uv venv -q --python 3.12 .venv; uv pip install -q --python .venv/bin/python polars numpy scipy scikit-learn lightgbm rapidfuzz sparse_dot_topn==1.2.0 pyyaml psutil pyarrow && .venv/bin/python -c "import polars, lightgbm, sparse_dot_topn; print(\"env ok\")"; du -sh ~/aml/cache/raw'
