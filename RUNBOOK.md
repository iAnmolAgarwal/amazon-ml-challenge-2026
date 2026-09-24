# RUNBOOK

Solo, 72h, 25 Sep 00:00 IST start. 5 submissions/day, 15 total. Private LB is the goal. Repo is PRIVATE; never publish.

## Layout
```
config.yaml          data_dir, cache_dir, device (mps -> cpu fallback), dtype, id/target/metric, cv scheme
src/                 harness (edit only with the human's approval)
attempts/<id>/       train.py (fit_predict(train, test, cfg) -> preds), proposal.md, results.json
prompts/             leader.md, worker.md, AGENT_PROMPT.md, codex_audit.md
orchestration/       assignments.md (leader writes), checkpoints.md (leader appends)
research/insights.md EDA / error-analysis findings, append-only
submissions/         NN_<attempt>.csv (ignored) + log.csv (tracked)
scripts/             queue.sh (tsp), memcheck.sh, setup_ec2.sh, bootstrap_notebook.py
data/  cache/        ignored; shared by all worktrees (resolved to the main checkout automatically)
```
Setup once: `uv venv --python 3.12 && uv sync && source .venv/bin/activate` (`brew install task-spooler`; binary is `ts` on mac, `tsp` on linux, queue.sh handles both).

## Measured Mac throughput (M3, 8GB, mps, float32, batch 64 / 32)
Smoke test 24 Sep 2026, machine already at 1.5 GB free + 3 GB swap from other apps.

| job | model | items/s steady | items/s incl. model load | 100k items |
|---|---|---|---|---|
| text embedding | all-MiniLM-L6-v2 (384d) | ~190 | 187 (500 texts, <=2000 chars) | ~9 min |
| image embedding | clip-ViT-B-32 (512d) | ~63 | 63 (200 jpg 224px, from disk) | ~27 min |
| TF-IDF + logreg CV | 000_example | 2000 rows, 5 folds | 4 s total | - |

Rules of thumb: MiniLM/e5-small text up to ~500k items stays on the Mac (<1h). CLIP-B/32 images up to ~100k stay on the Mac; beyond that, or any ViT-L / fine-tuning, go to Kaggle (T4/P100: expect 5-15x). Re-measure on real data (first chunk prints items/s and ETA) before deciding; write the number here.

## Memory rules (8GB unified)
polars not pandas; float32 everywhere; stream in chunks; never hold all images in memory; every embedding/feature matrix goes to `cache_dir` as .npy/.parquet and is never recomputed if the file exists (embed.py checks; train.py must too); one heavy job at a time via `scripts/queue.sh`; close Chrome tabs before big jobs. Run `scripts/memcheck.sh` first: if free < 2 GB or swap > 2 GB, wait or move the job to Kaggle.

## Starting the sessions
Three terminals, three Claude Code sessions:
1. Leader, in `~/Amazon_ML_Challenge` (branch main): paste `prompts/leader.md`.
2. W1, in `~/aml-w1` (branch w1): paste `prompts/worker.md` with `W1`/`w1` filled in.
3. W2, in `~/aml-w2` (branch w2): paste `prompts/worker.md` with `W2`/`w2` filled in.
Workers commit on their branch; leader merges `w1`/`w2` into main after each checkpoint (`git merge w1 w2`), workers then `git merge main`. `.venv` in the worktrees is a symlink to the main one.

## Queueing jobs and checking memory
```
scripts/memcheck.sh                                   # free mem, python procs, tsp queue
scripts/queue.sh python src/evaluate.py 012_w1_xyz    # queue full CV (one at a time)
scripts/queue.sh ls | tail <id> | rm <id> | kill      # inspect / follow / drop / kill running
```
Finished logs land in `logs/tsp-<id>.log`. Quick sanity check before queueing: `python src/evaluate.py <id> --subsample 0.1`.

## Heavy job on Kaggle / Colab
1. Push the branch. Put a GitHub fine-grained token (repo read only, this repo only) into Kaggle Secrets / Colab userdata as `GH_TOKEN`. Never paste it into a notebook cell.
2. Upload data as a Kaggle dataset / to Drive `MyDrive/amlc/data` (train.csv, test.csv, images/).
3. First cell: upload `scripts/bootstrap_notebook.py`, then
   `%run bootstrap_notebook.py --job embed --data /kaggle/input/<ds> --model clip-ViT-B-32 --modality image --input data/train.csv --col image --image-root data/images --tag train`
4. Download the .npy from the Output tab (Kaggle) or Drive into `~/Amazon_ML_Challenge/cache/` with the same file name. Workers then just use it (cache hit).
EC2 (only if activated + quota): see `scripts/setup_ec2.sh` and `docs/aws-setup.md`; move data/cache via the S3 bucket, terminate when done.

## Checkpoint (every 4-6h)
Human types `checkpoint` in the leader session. Leader reads leaderboard (`python src/leaderboard.py`), log.csv, assignments, queue; appends to `orchestration/checkpoints.md`; rewrites `orchestration/assignments.md`. Human relays new assignments to W1/W2 ("read your section of assignments.md").

## Submission sequence (exact)
1. Leader proposes a candidate (significant paired gain over the best submitted attempt) and prints only: commit hash, attempt ids.
2. Human fills `{COMMIT_HASH}` `{ATTEMPT_IDS}` in `prompts/codex_audit.md`, runs Codex CLI in the repo with that prompt.
3. Human pastes Codex's verdict verbatim into the leader session. Leader answers every finding; "fix first" -> worker fixes, re-evaluate, back to step 1.
4. `python src/predict.py <attempt_id>` -> `submissions/NN_<attempt_id>.tsv` (matches), `.candidates.tsv`, `_code.zip`; runs the official validator and stops on FAIL.
5. `python src/sub_status.py` -> confirm a submission is left today. Human uploads the .tsv AND the _code.zip on Unstop (both are required), clicks Submit & Evaluate.
6. Immediately: `python src/submit_log.py submissions/NN_<attempt_id>.tsv --notes "<what changed>"` (budget check, log row, git tag sub-NN). When the score shows: `python src/submit_log.py --set-lb NN <score>`. Commit log.csv.
7. `python src/sub_status.py` again: LB-CV gap, rank agreement, and whether an unsubmitted attempt now shows a significant paired gain over the best submitted attempt. The leader reads this at every checkpoint.

Budget: 5 per IST day (reset assumed at 00:00 IST, unused ones do not carry over), 15 total. Plan: day 1 baseline within ~3h + up to 4 audited improvements; day 2 up to 5; day 3 up to 3 + final ensemble + safer fallback. Never submit two candidates without a significant paired gain between them: the LB score of the second teaches nothing. Ties go to the earlier submission.

## 72h timeline (IST)
- 00:00-03:00 (h0-3): human reads rules; set config.yaml (id/target/metric/cv); `python src/split.py`; leader + workers run EDA into research/insights.md; baseline attempt; audit; submission #1.
- h3-60: workers loop within assigned directions; checkpoint at ~h6, h12, h18, h24, h30, h36, h42, h48, h54; queue long jobs (embeddings, full CV of slow models) via tsp before sleep, Kaggle jobs overnight.
- h60-72: freeze. No new directions. Ensemble + fallback, final Codex audit, docstrings on every function in the submitted code, docs/approach.md, last 2 submissions.
