# AWS setup (Amazon ML Challenge 2026)

No secrets live in this repo. Access keys are only in `~/.aws/credentials` under profile `amlc`.

| Item | Value |
|---|---|
| Account | `267150805041` |
| Region | `us-east-1` (EC2 in every other region is explicitly denied) |
| Working profile | `amlc` (IAM user `amlc-agent`, inline policy `amlc-agent-guardrails`; allowed instance types g5.xlarge, g5.2xlarge, t3.medium, r7a.2xlarge, m7a.2xlarge, r6a.2xlarge, r7i.2xlarge as of 2026-09-25 11:10 IST, user-approved) |
| Admin profile | `agent-toolkit` (root, browser `aws login`); use only for IAM/billing changes |
| S3 bucket | `amlc-2026-c77a54ac` (us-east-1, private, Block Public Access on all four settings; `amlc` can list/put/get/delete) |
| Allowed instance types | `g5.xlarge`, `g5.2xlarge`, `t3.medium` |
| Required launch tag | `project=amazon-ml-2026` (untagged launches are denied; start/stop/terminate only on tagged instances) |
| Budget | `amlc-2026`, 100 USD/month, email alerts at 50/80/100% actual and 100% forecasted |
| GPU quota (us-east-1) | On-Demand G/VT (L-DB2E81BA) and Spot G/VT (L-3819A6DF): 0 vCPU today; increase to 8 vCPU **requested 2026-09-24, PENDING** (request ids c88d311d..., 4e6a0722...). Check: `aws service-quotas get-service-quota --profile agent-toolkit --region us-east-1 --service-code ec2 --quota-code L-DB2E81BA --query Quota.Value` |

Account activated 2026-09-24 (support case 179026914900549). Policy verified with `run-instances --dry-run`: g5.xlarge/t3.medium tagged in us-east-1 pass; p4d, untagged, and other-Region launches are denied.

## SageMaker
| Item | Value |
|---|---|
| Execution role | `amlc-sagemaker-exec` (trust: sagemaker.amazonaws.com; S3 only on the project bucket, logs under /aws/sagemaker/*) |
| GPU notebook quota | ml.g5.xlarge for notebook instance usage (L-E8917BB7): requested 1 on 2026-09-24, PENDING |
| CPU notebook quota | ml.t3.medium notebook instances: 2 (usable now) |
| Studio domains | L-B683BCB0 applied 0 (default 2); restore requested on support case 179026914900549 |

Create the notebook in the console (root): SageMaker > Notebook instances > Create, type ml.g5.xlarge, role amlc-sagemaker-exec, tag project=amazon-ml-2026. Stop it whenever idle; it bills hourly.

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
