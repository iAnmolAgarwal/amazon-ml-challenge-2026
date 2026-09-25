# Latest submission

Only the team leader uploads. This file is updated after every scored upload, so check it before starting new work.

## Current best: submission 5, `008_stage3`, public LB **0.974**

| | |
|---|---|
| Uploaded | 25 Sep 2026, ~23:00 IST |
| Attempt | `attempts/008_stage3` (parent `006_stage2`) |
| 5-fold CV macro F0.5 | **0.98254** (India 0.98029, US 0.98403); paired vs 006: +0.00831, SE 0.00002, 5/5 folds |
| Public LB | **0.974** (previous best 0.963) |
| Code commit | `5cea04d` (feature and stage-3 code), docs after |
| matching_results.tsv | sha256 `bf456199656d6af5851345ecc808ffebd8b0eaf2a7dfcabc874925c5fc2560c3` (md5 `c674f878b78c8fcc093c5110b8ec492e`), 1,732,544 rows, 5,748,901 matches |
| Validator | organisers' `validate_submission.py --check-ids`: PASS (matching + candidates) |
| Test guard (house-conflict share of emitted pairs) | US 8.4%, India 17.0%, France 1.0% (006: 7.8 / 16.8 / 1.8) |

**What it is:** 006 plus a stage-3 LightGBM correction, boosted from logit(stage-2 probability), with 34 label-free pair features (`src/er_feat5.py`, `src/er_stage3.py`):
- name and address specificity;
- house-number relations, including the generator's decoy shift set;
- unit and legal-form agreement;
- char-3-gram name cosine.

**Reproduce:** the steps are in the README inside the code zip; the repo is identical at commit `5cea04d`. Put the official dataset under `data/student_resource/dataset/{train,test}/`.

## History

| # | Date (IST) | Attempt | CV | Public LB | Change |
|---|---|---|---|---|---|
| 1 | 25 Sep 05:54 | 001_baseline | 0.94292 | 0.929 | blocking + LightGBM |
| 2 | 25 Sep 07:08 | 002_namefreq | 0.96050 | 0.949 | name frequency, conservative names |
| 3 | 25 Sep ~11:00 | 004_coherence_full | - | not scored | portal failure under load |
| 4 | 25 Sep 15:22 | 006_stage2 | 0.97423 | 0.963 | native-script dictionary, floor 0.6, stage-2 competition features |
| **5** | **25 Sep 23:00** | **008_stage3** | **0.98254** | **0.974** | **stage-3 correction with label-free pair features** |

Uploads left: 11 of 15 overall (max 5 per IST day). Deadline 27 Sep 2026 23:59 IST.

## Working on the next one

Read these first:
1. `docs/STATUS.md`: the live status.
2. `docs/TASKS.md`: the task list.
3. `docs/research/`: evidence for every idea.

Rules every candidate must pass before upload:
1. **Paired CV rule:** same folds; mean per-fold gain > 2 SE and positive in ≥ 4 of 5 folds.
2. **Label-free test guard:** no rise in the house-conflict share or in added pairs per S1 beyond the train rate.
3. **No sibling or coherence features:** test decoys come in clusters.
