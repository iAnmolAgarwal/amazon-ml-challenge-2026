# 000_example

- parent: none
- worker: L
- created: 2026-09-24

## Idea
Reference attempt showing the `fit_predict(train, test, cfg) -> predictions` contract.
TF-IDF (1-2 grams) on a text column + two log-scaled numeric columns, logistic regression.
Real attempts branch from this with `src/new_attempt.py --parent 000_example --worker W1`.

## Evidence from history
None (first attempt). Smoke-tested on a 20 Newsgroups subset (4 classes, 2,000 rows, text + n_chars/n_words).

## Why not a repeat
n/a

## Expected gain / risk
n/a. Risk: the contract assumes a single target column and row-aligned predictions.

## Kill condition
n/a

## Result
Smoke test (not the competition data): see results.json. Delete this attempt's results.json once the real data is in and re-evaluate.
