# Approach: Amazon ML Challenge 2026

Filled in from hour ~60. Every number cites an attempt id whose results.json backs it.

## 1. Problem
- Task, inputs (text / image / tabular), target, official metric, train/test sizes.
- Submission format and constraints (5/day, 15 total, ties by earlier submission).

## 2. Data insights
- Key EDA findings (distributions, missingness, duplicates, label noise, train/test shift).
- Failure slices of the baseline (from error analysis) that drove the direction choices.

## 3. Candidate generation / blocking
- (If retrieval/matching task) how candidates were generated, recall@k on CV, why this blocking.
- (Otherwise) "n/a" and how the problem was framed.

## 4. Features
- Handcrafted features (list, with the attempt that introduced each and its CV delta).
- Frozen embeddings (model, modality, dims, where computed: Mac / Kaggle / EC2, runtime).
- Preprocessing and how leakage was avoided (fit inside folds).

## 5. Models
- Model families tried, heads on top of embeddings, loss/objective, key hyperparameters.
- Ensembling / stacking and how weights were chosen (CV only).

## 6. Experiments
| attempt | parent | change | CV mean ± std | Δ vs parent | public LB | kept? |
|---|---|---|---|---|---|---|
| 000_example | - | baseline | | | | |

## 7. Validation
- CV scheme (type, folds, seed, group column), why it mirrors the private test.
- CV vs public LB correlation across submissions.

## 8. Conclusion
- What worked, what did not and why, what we would do with 72 more hours.
- Final submission: attempt id, commit, CV, and the fallback submission.
