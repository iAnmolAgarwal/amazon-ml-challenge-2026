# Amazon ML Challenge 2026: business entity resolution

**#66 out of 10,000+ submissions.** Team Apprentice's entry to the Amazon ML Challenge 2026, built in the 72-hour window (25–28 September 2026).

| | |
|---|---|
| Public leaderboard (macro F0.5) | **0.990203** |
| 5-fold CV, US + India | 0.99184 (fold SD 0.00003) |
| Test candidate pairs | 11.84M (6.83 per business) |

## The problem

Three sources list the same businesses in different ways. For each business in Source 1, return every record in Sources 2 and 3 that is the same real business. There could be none, one or several.

- **Size:** 2.2M training businesses against 10.3M records. The test set has 1.73M businesses against about 10M records.
- **Noise:** typos, aliases ("aka", "dba"), reordered words, Indic-script names and missing addresses.
- **France:** 15% of the test set is French, and France never appears in training, so there are no labels for it.
- **Scoring:** F0.5 per business, averaged. Precision counts twice as much as recall.

## What worked

1. **Working out how the fake records were made.** The unmatched records are near-copies of real businesses. The house number is shifted by one of ten fixed steps (+1, 2, 3, 4, 5, 7, 9, 11, 13, 21), a word is inserted, or the legal form is swapped. We found this from the data: non-matches spike at exactly those ten shifts and nowhere else. That turned into "decoy-marker" features that need no labels, so France could build its own table from its own test candidates.
2. **Checking every change against the test set's shape, without labels.** On test, candidate pairs with one of the ten shifts were 1.94x as common as on train, so the test set has about twice the decoys. Some features looked great in CV and would have failed there. One example is judging a record by the business's other candidates: it gained +0.0033 CV and lost 0.037 once train decoys were cloned to test density. Before every upload, a label-free guard compared test and train on pairs emitted per business. Any change that emitted far more on test was thrown out.
3. **France with no labels.** Every France rule was set by training on one country and scoring the other (leave-one-country-out), plus label-free counts on France's own candidates.

Every US/India change had to beat its parent on paired 5-fold CV: a gain above 2 standard errors, and positive in at least 4 of 5 folds.

## Pipeline

```
Source 1 + pool records, per country
 -> normalise (native-script words -> Latin, dictionary learned from training pairs)
 -> blocking: word TF-IDF top 40 + exact keys + character 3-gram route + reverse retrieval
 -> stage 1 / stage 2 LightGBM (+ competition features)
 -> band model -> lexical correction -> generator-operator correction
 -> fine-tuned multilingual-e5 cross-encoders on the uncertain band
 -> meta-model over two independently built model families
 -> empty-address correction -> expected-F0.5 decision (each record gets at most one owner)
```

## How the score moved

| Stage | CV | Public LB |
|---|---|---|
| Blocking + one LightGBM | 0.943 | 0.929 |
| + name frequency, native-script dictionary, competition features | 0.974 | 0.963 |
| + pair features (name specificity, house relations, legal form) | 0.983 | 0.974 |
| + decoy-marker name features | 0.985 | 0.981 |
| + generator-operator correction | 0.987 | 0.985 |
| + meta-model with the second model family, France blend | 0.99046 | 0.988787 |
| + extra retrieval routes, cross-encoder inputs, empty-address correction | 0.99184 | **0.990203** |

One LightGBM on all 205 features scores the same as the whole correction chain (0.98779 for both). The gains came from the features, not the stage structure.

## Repository

```
src/                 pipeline (er_*.py), cross-encoders (src/ce), second model family (src/stack)
attempts/<id>/       one folder per experiment: settings, idea, CV result
docs/                full write-up (Documentation_template.md), approach and research notes
docs/process/        how the 72 hours were run (see below)
research/            analysis scripts behind the write-up's numbers
submissions/         upload log with scores
```

The full write-up, with error analysis and negative results, is [docs/Documentation_template.md](docs/Documentation_template.md).

## Running it

1. Python 3.12: `uv sync` (or `pip install -e .`).
2. Download the official dataset from the challenge portal. Put it at `data/student_resource/dataset/{train,test}/*.tsv`. The data isn't in this repo.
3. CV for an experiment: `python src/er_eval.py <attempt_id>`. Test output: `python src/er_predict.py <attempt_id>`. The final run is `attempts/029f6_final`. It builds on earlier attempts, and the write-up's appendix A lists the full order.

The heavy steps need a 64 GB machine. We used about 30–55 CPU hours on cloud VMs and 15–20 GPU hours on Kaggle T4s.

## Team and how it was built

Team Apprentice: Anmol Agarwal (team leader), Akshit Kumar Singh, Nitya Arora and Harsimran Singh.

The main pipeline was built in Claude Code: a lead session planned the work and handed experiments to two worker sessions, and Codex CLI audited each stage. Every change was accepted only on paired CV. The prompts, run book and review notes are kept as they were written in [docs/process/](docs/process/).
