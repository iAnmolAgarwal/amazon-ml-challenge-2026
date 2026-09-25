# Amazon ML Challenge 2026: business entity resolution

**Latest submission and score: see [LATEST_SUBMISSION.md](LATEST_SUBMISSION.md)** (updated after every scored upload; only the team leader uploads).

## Setup
1. Python 3.12: `python3.12 -m venv .venv && . .venv/bin/activate && pip install -e .` (or `uv sync`).
2. Download the official `student_resource` dataset from the challenge portal. Unzip it so the files sit at `data/student_resource/dataset/{train,test}/*.tsv`. The dataset is not in git: `data/` and `cache/` are ignored.
3. Run the pipeline steps listed in `LATEST_SUBMISSION.md` / the code-zip README. Heavy steps need a 64 GB machine.

## Where things are
- `src/`: the pipeline (normalize → block → features → stage 1/2/3 → decide → write TSVs; `er_package.py` validates and packages).
- `attempts/<id>/`: one folder per experiment (settings in `train.py`, idea and result in `proposal.md`, CV in `results.json`).
- `docs/STATUS.md`: live status. `docs/TASKS.md`: task list. `docs/research/`: measured evidence per research lane.
