You are a senior machine-learning engineer with deep experience in large-scale entity resolution (record linkage) and in winning time-boxed ML competitions. Review the competition plan below adversarially. Assume the plan is wrong until the reasoning proves otherwise. I will act only on findings that are concrete, specific and justified.

# 1. The competition (complete problem statement)

**Amazon ML Challenge 2026: Business Entity Resolution.** Solo participant. Window: 25 Sep 2026 00:00 IST to 27 Sep 2026 23:59 IST (72 hours). 5 leaderboard submissions per day, 15 in total. Ties are broken by earlier submission time. Final ranking uses the private leaderboard (the part of the test set not used for the public leaderboard); predictions are always made for the full test set.

**Task.** Business records come from 3 independent sources with no shared identifiers. Source 1 is the deduplicated reference source. For every Source 1 entity, find all matching records in Source 2 and Source 3. A Source 1 entity may match zero, one or many records.

**Data (tab-separated .tsv).** Each source file has: `entity_id` (prefix S1-, S2-, S3- gives the source), `business_name`, `business_address`, `country`. Training ground truth: `source1_entity_id`, `matched_entity_ids` (comma-separated S2/S3 ids, empty when no match). The test set has no labels.
- Country: training covers US and India. The test set also contains France, which never appears in training. Country must be treated as an open set of labels: no hard-coding, filtering or one-hot encoding to {US, India}; every test entity, France included, must appear in the output.
- Stated noise: name abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited), legal suffix inconsistencies, DBA/trade names, punctuation (& vs and), word-order transpositions, typos; address abbreviations (Rd/Road, St/Street), transliteration variants, missing components (no PIN, no state), landmark references (Near SBI ATM), municipal numbering formats, component reordering.

**Output.** Two tab-separated files:
1. `matching_results.tsv`: `source1_entity_id`, `matched_entity_ids`. Exactly one row per test Source 1 entity; empty list for singletons; no duplicate ids in a list; only S2/S3 ids that exist in the test set. This is the only scored file.
2. `candidate_pairs.tsv`: `source1_entity_id`, `candidate_entity_ids`. The exact final candidate set the matching model scores (the last blocking/filtering stage). Not scored, but used to audit blocking (recall ceiling, reduction ratio). Every matched id must also be a candidate.
An official stdlib validator checks both files. Each leaderboard upload on the portal requires the matching TSV **and** a zip of the code.

**Metric.** F-beta with beta = 0.5, computed per Source 1 entity and macro-averaged over all Source 1 entities: F0.5 = 1.25·P·R / (0.25·P + R). An entity with no true matches scores 1.0 for an empty prediction and 0.0 for any prediction. Precision is weighted 2x over recall.

**Rules.**
- Strictly no external data lookup: no entity-resolution APIs, no government business registries, no geocoding APIs, no augmentation from internet sources. Only the provided training data. Violations mean disqualification; top teams' code is reviewed.
- Final model must be MIT or Apache-2.0 licensed and at most 8B parameters.
- Final package: zip with `output/` (both TSVs), `code/business_entity_resolution/` (src/, README with exact end-to-end reproduction, pinned requirements.txt), and a filled methodology document (methodology, blocking strategy, model architecture and features, other details).

# 2. Measured facts about the data (from training labels; test only row and country counts)

| | Source 1 | Source 2 | Source 3 |
|---|---|---|---|
| Train rows | 2,206,821 | 5,034,616 | 5,285,603 |
| Test rows | 1,732,544 | 4,887,273 | 5,082,316 |

- Train countries: US 60%, India 40% of Source 1. Test: India 47%, US 38%, France 15% (about 259k Source 1 entities and 1.4M pool records).
- Every S2/S3 record matches **at most one** Source 1 entity (7,638,365 gold pairs, all pool ids unique).
- **No gold pair crosses countries.**
- Matches per Source 1 entity: 0: 5.6%, 1: 5.4%, 2: 17%, 3: 24%, 4: 22%, 5: 15%, 6: 7.5%, 7+: 4%. Mean 3.46.
- 26% of pool records match nothing (distractors). S2 and S3 contribute about equally.
- Noise appears to be produced by a fixed set of generator operators. Observed in sampled clusters:
  - Names: website form (`starswelfaresociety.com`, `mvdata.com`), acronym (`SWS` for Stars Welfare Society), honorific prefixes (Mr, Smt, Shri), junk prefixes (`***`, `--`), "formerly: <old name>" / DBA, added filler words (Center, Partners, Corp), digit-for-letter typos (`Michae1ine`), transpositions (`Crumrnie`), case changes, legal suffixes added/removed/moved (`Llc Deluca Ace Crumrine`), names in Devanagari script.
  - Addresses: state names in native script (Telugu, Devanagari), state codes (TG, MH, OH) vs full names, city aliases (Poona/Pune), house-number formats (`Hno. 102`, `Door No 0102`, `#422`, `NO 422`), reordered components, extra locality (`SHEFFIELD TWP, PO BOX 5597`), missing parts, the literal string `None`, altered house numbers (634 vs 6345), city typos.

Example cluster (train):
```
S1  Stars Welfare Society | 422 Rao Saheb Kedari Road Camp, Pune, Maharashtra | India
S2  Mr Stars Welfare Society Corporation | Maharashtra, NO 422 RAO SAHEB KEDARI ROAD CAMP, PUNE
S2  *** Stars Welfare Society Corp | NO 422 RAO SAHEB KEDARI ROAD CAMP, POONA, PUNE, Maharashtra
S2  Smt starswelfaresociety.com | 422 RAO SAHEB KEDARI ROAD CAMP, PUNE, Maharashtra
S2  Stars-Welfare Society | #422 RAO SAHEB KEDARI ROAD CAMP, PUNE, महाराष्ट्र
S3  Shri Stars Welfare Partners | #422 Rao Saheb Kedari Road Camp, Poona, Pune, MH
S3  SWS | 422 Rao Saheb Kedari Road Camp, Pune, Maharashtra
```

# 3. Compute constraints

- Main machine: MacBook Air M3, **8 GB unified memory**, about 7 GB free disk. Python, polars, LightGBM, rapidfuzz, scikit-learn, sentence-transformers.
- Kaggle/Colab notebooks available (Kaggle CPU sessions have about 30 GB RAM; free GPUs for a limited number of hours per week).
- AWS GPU quota requests were denied or are pending; no plan may depend on them.
- One human, 72 hours, with two parallel AI coding agents (workers) and one AI coordinator, plus an independent AI code auditor before every submission.

# 4. Planned strategy

**Pipeline.**
1. **Normalization.** Unicode NFKD, strip accents, lowercase, remove punctuation and junk prefixes, expand abbreviations, drop honorifics and legal suffixes to get a "core name". Build a space-free name (to match website forms) and initials (to match acronyms). Map state codes, full state names and native-script state names to one canonical token, and city aliases, using mappings **learned from training gold pairs** (co-occurrence of address tokens in matched pairs), plus a small hand-written French layer (rue, avenue, boulevard, SARL, SAS, EURL, cedex). Extract house numbers and postal codes.
2. **Blocking (candidate generation), within country only.** Key-based joins, since nearest-neighbour search of 2M x 10M records does not fit in 8 GB:
   - house number + first street token
   - space-free core name (catches website forms)
   - initials + house number (catches acronyms)
   - rare name tokens (inverse-frequency filtered) + city/state token
   Union of all keys, with a maximum block size per key; if needed, a secondary char n-gram TF-IDF view restricted within blocks. Target: over 97% pair recall at a manageable candidate count. Every run reports pair recall, the oracle F0.5 ceiling and average candidates per entity.
3. **Pair features.** rapidfuzz ratio, token-set, token-sort, partial ratio and Jaro-Winkler on core and raw names; the same on addresses; TF-IDF cosine; house-number and postal-code agreement; website-form and acronym matches; first-token equality; length differences; S2 vs S3 source; rank and score gap of the candidate within its Source 1 entity, and of the Source 1 entity within the candidate's list (reverse rank).
4. **Model.** LightGBM binary classifier on labelled candidate pairs.
5. **Decision rule.** Assign each pool record only to its highest-scoring Source 1 entity (exact constraint from the data), then keep pairs above a probability threshold tuned directly on macro F0.5 with inner out-of-fold predictions. Later: per-country or confidence-dependent thresholds, and consistency between S2 and S3 records (a pool record that strongly matches another already-assigned pool record inherits its assignment).
6. **France (unseen country).** Country-agnostic features only; hand-written French normalization; a leave-one-country-out check (train on US, validate on India) to estimate degradation for an unseen country; a slightly more conservative threshold for countries not seen in training.
7. **Scale plan.** Develop and cross-validate on about 10% of Source 1 entities while always blocking against the full pool; full training and test inference in chunks by country. Move the full run to a Kaggle CPU notebook if the Mac runs out of memory.

**Validation.** 5-fold CV over Source 1 entities, stratified by country x match count. The candidate pool is always the full training pool, so records belonging to other folds act as distractors, as in the test set. Labels of pool records are never used as features. A change counts only if the CV mean beats its parent by more than one standard deviation across folds.

**Organisation.** Worker 1 owns blocking recall, normalization and France robustness. Worker 2 owns features, the model and the decision rule. The coordinator assigns directions, runs checkpoints every 4-6 hours and plans submissions; an independent auditor checks every candidate for leakage before submission.

**Submission plan (15 total).** Day 1: baseline within about 3 hours (confirms format and the CV-to-leaderboard relationship), then up to 4 audited improvements. Day 2: up to 5. Day 3: up to 3, then the final ensemble and a safer fallback. Never submit two candidates whose CV differs by less than one standard deviation. Every submission is logged with CV mean/std, public score, commit hash and code-zip checksum; the CV-vs-leaderboard gap and rank agreement are tracked after each score.

# 5. What I want from you

Return findings ranked by expected impact on the **private leaderboard F0.5**. For each finding give:
1. **Claim**: what is wrong, missing or suboptimal (one or two sentences).
2. **Why**: the reasoning or evidence, referring to the facts above.
3. **Change**: the concrete change you recommend (specific enough to implement).
4. **Expected impact**: rough F0.5 effect and how confident you are (high / medium / low).
5. **Cost**: implementation time and compute on an 8 GB Mac.

Cover at least: blocking design and its recall at this scale; validation design and leakage risks (including anything in the assignment constraint or reverse-rank features that could inflate CV); the decision rule under macro F0.5 with 3.5 matches per entity; France generalisation; exploiting the apparent synthetic-generator structure within the rules; memory and runtime risks; and submission-budget strategy. Also list anything in the plan that would breach the competition rules (external data, model licence, output format).

Do not praise the plan and do not summarise it back. If you think a part of the plan is already right, say nothing about it.
