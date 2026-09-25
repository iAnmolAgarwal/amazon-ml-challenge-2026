You are a strict competition compliance reviewer. Check whether my Amazon ML Challenge 2026 submissions are being produced and packaged correctly, and whether anything could get them rejected, disqualified at final review, or silently scored lower than they should be. List concrete problems only, most severe first, each with the exact fix. If something is fine, say so in one line. No praise.

# Official rules (quoted from the organisers' README)
- Output 1, `matching_results.tsv` (the only file scored; uploaded to the portal): tab-separated, header `source1_entity_id<TAB>matched_entity_ids`. One row per test Source 1 entity. Comma-separated S2-/S3- ids, no quoting, empty for singletons. No duplicates within a list and no duplicate S1 rows. Ids must exist in the test set.
- Output 2, `candidate_pairs.tsv` (not scored; used to audit blocking): same format with `candidate_entity_ids`. It must be "the exact set of records you feed into your matching model for inference — the final candidate list just before the ML model scores them". If there are several blocking/filtering stages, it is the last one. Every matched id should appear in it.
- Final package `<team_name>_submission.zip`:
  - `output/matching_results.tsv` and `output/candidate_pairs.tsv`
  - `code/business_entity_resolution/src/` (all source), plus README.md (exact end-to-end run instructions) and requirements.txt (pinned)
  - `Documentation_template.md` (filled in)
  - "Anyone should be able to regenerate both output files from the training/test data using only what is in this folder." Top teams' packages are reviewed in detail (reproducibility, blocking audit, fair play, model licence).
- Country is an open set: "do not hard-code, filter, or one-hot your pipeline to only {US, India}"; every test entity, France included, must appear.
- The final model must be MIT/Apache 2.0 licensed, up to 8B parameters.
- Metric: macro F0.5 per S1. A singleton scores 1.0 for an empty list, else 0.
- Public LB uses a subset of test; the private LB (the rest) decides the final ranking.
- Portal (per upload): the matching_results TSV plus a code zip. 5 uploads per day.

# What I do
- Pipeline:
  1. prepare_raw (TSV -> parquet, all strings)
  2. normalization (er_norm). Language rules are applied only to France-labelled rows (French street types, st->saint, BP/CEDEX). There are no country restrictions elsewhere, and blocking runs per country value found in the data.
  3. blocking (er_block): TF-IDF top-40 + exact keys
  4. pair features (er_feat, er_feat2, er_feat3). No ids, row positions or country labels are features.
  5. LightGBM binary model trained on training-set pairs
  6. decision: each pool record goes to its best S1; per-S1 expected-F0.5 cut
- `candidate_pairs.tsv` = every (S1, pool) pair the model scored, union the emitted matches (a subset anyway), S2/S3 ids only. Written in test_source1 order.
- Each upload is validated with the organisers' `validate_submission.py --check-ids` (PASS) and its checksums compared after copying.
- Per-upload code zip contains:
  - `code/business_entity_resolution/src/` (all pipeline modules plus a generated prepare_raw.py)
  - `config.yaml`
  - `attempts/<id>/train.py` and `results.json` (CV results)
  - README.md with setup and the exact command sequence (prepare_raw, er_norm, er_block, er_feat, er_feat2, er_feat3, er_eval optional, `er_predict <attempt> --train-frac <f>`)
  - pinned requirements.txt (polars 1.44.2, numpy 2.5.3, scikit-learn 1.9.1, lightgbm 4.7.0, rapidfuzz 3.14.6, sparse-dot-topn 1.2.0, pyarrow, pyyaml, psutil)
  - Python 3.12, x86-64 Linux, 64 GB RAM
- The zip has no data, no outputs and no notes. The final package adds `output/` and `Documentation_template.md`.
- No external data, no pretrained models, no web lookups. Random seeds are fixed (42). Training uses a seeded sample of training entities (20% / 50% / 100%).
- Results so far:
  - Upload 1: CV 0.9429 -> public LB 0.929
  - Upload 2: CV 0.9605 -> public LB 0.949
  - The LB-CV gap of about -0.012 is attributed to France (15% of test, unseen in training; implied France F0.5 about 0.907 if US/India transfer).

# Questions
1. Is anything in the above likely to cause rejection, a lower score than intended, or a problem at the final package review? Consider: candidate_pairs semantics; row order; France-specific normalization vs the "open set" rule; the per-upload code zip contents; reproducibility of training with a sampled fraction; memory/hardware requirements in the README; what the final package must contain that the per-upload zip lacks.
2. Is attributing the stable -0.012 LB-CV gap to France reasonable, or which other systematic causes should I rule out? Name the cheap checks (e.g. test vs train candidate-count distributions, singleton rate predicted on test vs train, per-country predicted matches per S1).
3. What should the Documentation_template.md emphasise so a detailed reviewer can verify fair play quickly?
