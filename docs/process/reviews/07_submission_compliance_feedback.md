VERDICT: FIX REQUIRED — the archived reproduction instructions and final-package builder have concrete defects. I found no confirmed invalid scored TSV.

I inspected both local code zips and the packaging, prediction and validation code. I did not rerun training or verify files on EC2.

1. HIGH — Both uploaded READMEs select a different training fraction from their attempt configuration.
   In the upload 1 archive (submissions/01_001_baseline_code.zip), packaged README.md:21 omits --train-frac; upload 2's packaged README (submissions/02_002_namefreq_code.zip) does the same at line 23. Both packaged predictors default to 0.3, while both attempt configurations specify 0.2.
   The predictor loads the attempt module but samples using the CLI value directly: src/er_predict.py:41, src/er_predict.py:62. Following the README therefore trains on 30%. I cannot verify which fraction produced the uploaded TSVs.
   Exact fix: recover the actual inference command for each upload and put its explicit fraction in that archive's instructions. Make the CLI default resolve to the attempt's TRAIN_FRAC. Record the effective fraction, sampled-entity checksum, feature list and parameters alongside every inference run. The current packager's updated README template does not repair the already uploaded archives.

2. HIGH — The final builder can omit mandatory documentation, accept an unfilled template, and overwrite an existing code zip before failing.
   The methodology file still contains placeholders: docs/Documentation_template.md:3, :30, :43. The builder silently omits it if absent: src/er_package.py:129.
   --final still targets NN_attempt_code.zip and opens it for replacement before adding the required outputs: src/er_package.py:113. Neither corresponding .candidates.tsv exists locally, so final packaging here would fail when adding that file at line 127, after overwriting the previous archive. The candidate files may exist on EC2; not checked.
   Exact fix: retrieve the candidate file from the same inference run and verify its checksum; complete the methodology document; preflight all required files before opening an archive. Build a distinct <team_name>_submission.zip through a temporary file, then rename after successful validation. Package the source snapshot that produced the chosen outputs — the current builder simply copies today's workspace at line 117.

3. HIGH — Your automated validation gate can accept missing candidates or unchecked IDs.
   The packaging wrapper omits --check-ids and checks only the exit code: src/er_package.py:92.
   The official validator returns warnings, rather than failure, when: a pool source file is missing, even with --check-ids (utils/validate_submission.py:80); the candidate TSV is missing (:251); matches are absent from candidates (:261). Those warnings can still precede PASS (:337).
   Exact fix: add --check-ids; require all three test source files and both outputs; fail on skipped ID checks, missing candidates or containment violations. Independently assert that exported candidates equal the scored-pair set and that emitted matches are its subset. Do not use unioning emitted matches into candidates as a repair mechanism.

4. MEDIUM — Fixed seeds do not fully specify reproducible training or output bytes.
   Feature and label joins have no explicit ordering guarantee before matrix construction: src/er_predict.py:55, :65. Training uses row bagging: src/er_model.py:14. Polars warns that unspecified join ordering can vary between runs; a fixed seed does not compensate for reordered training rows.
   Candidate serialization also follows unordered grouping at src/er_predict.py:84. That can change checksums without changing candidate sets.
   Exact fix: sort entity IDs before sampling; sort training pairs by (s1_idx, pool_idx) after joins; fix thread count and LightGBM's deterministic histogram configuration; sort candidate IDs when writing. Perform one clean extraction-to-output reproduction using the final configuration.

Compliant with the quoted rules: candidate boundary and row order (src/er_predict.py:73); France handling (src/er_norm.py:165, src/er_block.py:130); intermediate archives may omit outputs and methodology; dependencies pinned; LightGBM MIT, sampled training and 64 GB requirement fine; local matching_results.tsv checksum matches upload 2 (submissions/log.csv:3).

5. MEDIUM — Attributing the entire CV–LB gap to France overlooks a measurable country-composition effect.
   US/India test-weighted CV is 0.95640 vs overall CV 0.96050 (attempts/002_namefreq/results.json:19), so ~0.0041 of the 0.0115 gap comes from the US/India mix alone. F_France = (0.949 - 0.38*0.9713 - 0.47*0.9444)/0.15 ≈ 0.907 is conditional on unchanged US/India performance and public-set country proportions matching the full test set.
   Cheap checks before assigning the remaining gap to France: training/inference parity (fraction, parameters, feature names/order/dtypes, normalization version, cache provenance); ID mapping round-trip of sampled scored pairs to original ids and text; blocking distributions per country (zero-candidate rate, candidate-count quantiles, key hit/skip rates, cosine, reverse ranks, rival counts); prediction distributions per country (empty rate, matches per S1, S2/S3 split, ownership rejection) test vs OOF; input/feature quality (missing rates, script mix, house/postal extraction, nonfinite values). Do not force test rates to equal training rates.

Documentation_template.md should make explicit: data provenance (files, checksums, label use, label-free test statistics, hand-written rules vs learned mappings, no external data/pretrained weights/lookups); exact producing run (attempt, effective fraction, sample checksum, seeds, features, params, source snapshot); blocking audit (TF-IDF settings, top-40, chunking, exact keys, oversized-block skipping, dynamic countries, scored-pair boundary, candidate counts by country, containment assertions); model and decisions (licence, size, ownership rule, expected-F selection vs official metric); reproduction (clean-extraction commands, pinned env, threads, peak RAM/disk/runtime, output paths); evidence (checksums, validator command and log, coverage, CV protocol and results; France estimate labelled unvalidated).

No rules violation from external-data use is established by this inspection; a clean end-to-end reproduction remains unverified.
