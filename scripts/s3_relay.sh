#!/usr/bin/env bash
# Copy a file from one EC2 worker to another through the private S3 bucket, using short-lived
# presigned URLs generated on the Mac (no AWS credentials on the instances). Files are split into
# 1.5 GB parts (presigned PUT limit is 5 GB).
# usage: scripts/s3_relay.sh <src-alias> <remote-path-under-~/aml> <dst-alias>
set -euo pipefail
SRC=$1; REL=$2; DST=$3
export AWS_PROFILE=${AWS_PROFILE:-amlc} AWS_REGION=${AWS_REGION:-us-east-1}
BUCKET=amlc-2026-c77a54ac
RID="$(date +%s)$$$RANDOM"
KEY="relay/$RID/$(basename "$REL")"
TMP="/tmp/relay_$RID"
N=$(ssh "$SRC" "cd ~/aml && rm -rf $TMP && mkdir -p $TMP && split -b 1500M -d -a 3 '$REL' $TMP/part. && ls $TMP | wc -l")
echo "$REL: $N part(s)"
{
  echo 'set -e'
  for i in $(seq 0 $((N - 1))); do
    p=$(printf "part.%03d" "$i")
    url=$("$(dirname "$0")/../.venv/bin/python" - "$BUCKET" "$KEY.$p" <<'EOF'
import sys, boto3
print(boto3.client("s3", region_name="us-east-1").generate_presigned_url("put_object", Params={"Bucket": sys.argv[1], "Key": sys.argv[2]}, ExpiresIn=3600))
EOF
)
    echo "curl -sSf -X PUT -T $TMP/$p '$url'"
  done
  echo "rm -rf $TMP"
} | ssh "$SRC" bash -s
{
  echo "set -e; mkdir -p \"\$(dirname ~/aml/$REL)\"; : > ~/aml/$REL.tmp"
  for i in $(seq 0 $((N - 1))); do
    p=$(printf "part.%03d" "$i")
    echo "curl -sSf '$(aws s3 presign "s3://$BUCKET/$KEY.$p" --expires-in 3600)' >> ~/aml/$REL.tmp"
  done
  echo "mv ~/aml/$REL.tmp ~/aml/$REL; ls -la ~/aml/$REL"
} | ssh "$DST" bash -s
aws s3 rm "s3://$BUCKET/relay/$RID/" --recursive --only-show-errors
echo "relayed $REL from $SRC to $DST"
