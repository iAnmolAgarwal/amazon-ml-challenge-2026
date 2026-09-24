#!/usr/bin/env bash
# Run ON a fresh EC2 instance launched from the Deep Learning AMI (Ubuntu, PyTorch).
# Launch (from the Mac, profile amlc, see docs/aws-setup.md):
#   aws ec2 run-instances --profile amlc --region us-east-1 --instance-type g5.xlarge \
#     --image-id <DLAMI id> --key-name <key> --security-group-ids <sg> \
#     --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150}' \
#     --tag-specifications 'ResourceType=instance,Tags=[{Key=project,Value=amazon-ml-2026}]' \
#                          'ResourceType=volume,Tags=[{Key=project,Value=amazon-ml-2026}]'
# Then: scp this file up, ssh in, run:  GH_TOKEN=<token> bash setup_ec2.sh
# The token is read from the env only; nothing is written to disk except the git remote URL (no token).
set -euo pipefail
: "${GH_TOKEN:?set GH_TOKEN (GitHub fine-grained token with repo read)}"
REPO="${REPO:-iAnmolAgarwal/amazon-ml-2026}"
sudo apt-get update -qq && sudo apt-get install -y -qq task-spooler git >/dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone "https://x-access-token:${GH_TOKEN}@github.com/${REPO}.git" ~/amazon-ml-2026
cd ~/amazon-ml-2026
git remote set-url origin "https://github.com/${REPO}.git"
uv venv --python 3.12 && uv sync
sed -i 's/^device: .*/device: cuda/; s/^num_workers: .*/num_workers: 4/' config.yaml
mkdir -p data cache logs
echo "done. put data in ~/amazon-ml-2026/data (aws s3 sync s3://amlc-2026-c77a54ac/data data), then:"
echo "  source .venv/bin/activate && scripts/queue.sh python src/embed.py --model clip-ViT-B-32 --modality image --input data/train.csv --col image --image-root data/images"
echo "  aws s3 sync cache s3://amlc-2026-c77a54ac/cache   # bring embeddings back, then terminate (see docs/aws-setup.md)"
