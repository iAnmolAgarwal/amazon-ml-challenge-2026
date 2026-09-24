# AWS setup (Amazon ML Challenge 2026)

No secrets live in this repo. Access keys are only in `~/.aws/credentials` under profile `amlc`.

| Item | Value |
|---|---|
| Account | `267150805041` |
| Region | `us-east-1` (EC2 in every other region is explicitly denied) |
| Working profile | `amlc` (IAM user `amlc-agent`, inline policy `amlc-agent-guardrails`) |
| Admin profile | `agent-toolkit` (root, browser `aws login`); use only for IAM/billing changes |
| S3 bucket | `amlc-2026-c77a54ac` (private, Block Public Access on) - **not yet created, account S3 not activated** |
| Allowed instance types | `g5.xlarge`, `g5.2xlarge`, `t3.medium` |
| Required launch tag | `project=amazon-ml-2026` (untagged launches are denied; start/stop/terminate only on tagged instances) |
| Budget | `amlc-2026`, 100 USD/month, email alerts at 50/80/100% actual and 100% forecasted |
| GPU quota (us-east-1) | On-Demand G/VT (L-DB2E81BA): 0 vCPU; Spot G/VT (L-3819A6DF): 0 vCPU. Increase to 8 **not yet requested** (pending EC2 activation) |

## Launching an instance

Tag the instance (and its volumes) at launch:

```bash
aws ec2 run-instances --profile amlc --region us-east-1 --instance-type g5.xlarge ... \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=project,Value=amazon-ml-2026}]' \
                       'ResourceType=volume,Tags=[{Key=project,Value=amazon-ml-2026}]'
```

## Kill everything

```bash
aws ec2 terminate-instances --profile amlc --region us-east-1 --instance-ids $(aws ec2 describe-instances --profile amlc --region us-east-1 --filters Name=tag:project,Values=amazon-ml-2026 Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].InstanceId' --output text)
```
