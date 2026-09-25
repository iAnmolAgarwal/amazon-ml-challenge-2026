You are a senior engineer who has won large-scale entity-resolution competitions. My pipeline is a well-tuned classical one, and incremental work on it gives +0.001 to +0.005. I need a step change of about +0.025 macro F0.5 on the hidden test set. Do not polish my current approach. Tell me which fundamentally different approaches, or which missing big components, can deliver a gain that large on this data in the time left. Be concrete and quantitative, and say where your numbers come from. Where you are guessing, say so.

# 1. Task
Amazon ML Challenge 2026, business entity resolution. For each Source 1 (S1) record, return all matching Source 2/3 ("pool") records.
- Fields: entity_id (the prefix gives the source), business_name, business_address, country. There are no other columns.
- Metric: F0.5 per S1, macro-averaged over all S1. A singleton scores 1 for an empty prediction, else 0.
- Rules: no external data or lookups. Models must be MIT/Apache licensed with at most 8B parameters. Code must be reproducible, and a code zip goes with each submission.
- Data sizes: train has 2.21M S1 and 10.32M pool; test has 1.73M S1 and 9.97M pool.
- Countries: train is US 60% / India 40%. Test is India 47% / US 38% / France 15%. France never appears in training.
- Structure: each pool record matches at most one S1, and no match crosses countries. 5.6% of S1 are singletons. Match counts per S1: 1: 5.4%, 2: 17%, 3: 24%, 4: 22%, 5: 15%, 6: 7.5%, 7+: 4% (mean 3.46). 26% of pool records match nothing.
- The data is clearly synthetic. The noise comes from a fixed operator set with small vocabularies (rates per gold pair):
  - address uppercased 30%, native-script address 9.0%, native-script name 7.4% (Devanagari, Telugu, Kannada, ...)
  - case-only name change 5.9%, website form 5.2% (always concatenated lowercase + ".com"), token reorder 5.0%
  - acronym; honorific/junk prefixes (the, m, mr, shri, sri, dr, smt, @ # *** -- >> ...)
  - "formerly:"/DBA with a different name; word splits ("Li mited")
  - ~20 house-number templates (#N, no N, door no N, h.no N, plot no., flat no.)
  - state code vs full name vs native script; city aliases; "None" addresses 4.5%
- Entity ids and row order carry no signal (checked).
- 34% of India and 20% of US S1 names are shared by more than 5 S1.

# 2. Current pipeline
1. **Normalization** inverts the operators explicitly. Views:
   - core name, space-free join, sorted tokens, initials, alternate (formerly/DBA) name
   - normalized address, house number, first street token, postal code, numbers
   - France-specific rules, hand-written without labels: street types, particles, st->saint, BP/CEDEX
2. **Blocking** is done per country. Word-level TF-IDF top-40 (hashed word 1-2 grams of name + address, sparse_dot_topn) is unioned with 6 exact keys. Keys with more than 50 rows are skipped.
   - About 44 candidates per S1.
   - Oracle macro F0.5 on train: 0.9898 overall (US 0.9954, India 0.9815).
   - 56% of India retrieval misses have Indic-script pool names. Char 3-grams recover none of them. Word top-80 recovers 18% of India misses.
3. **Pair features (~60)**:
   - word cosine; both-side ranks and gaps; candidate and rival counts
   - rapidfuzz ratio / token_set / partial / Jaro-Winkler on several name views
   - address token_set; house/street/postal equality and conflict; number Jaccard; flags
   - label-free name frequencies (how many S1 and pool records share each name)
   - coherence: agreement of each candidate with the S1's other strong "anchor" candidates (new, in CV now)
4. **Model**: LightGBM binary (lr 0.05, 127 leaves, 600 rounds), trained on candidates of a 20-100% sample of training entities.
5. **Decision**: each pool record goes to its highest-probability S1. Per S1, pick the k maximizing 5*sum(p_1..p_k)/(E[T]+4k), compared against k=0 scoring prod(1-p).

# 3. Results
| Submission | CV macro F0.5 (train, 5-fold OOF) | India CV | US CV | Public LB |
|---|---|---|---|---|
| 1: baseline | 0.9429 | 0.9202 | 0.9581 | 0.929 |
| 2: + name frequency + conservative names | 0.9605 | 0.9444 | 0.9713 | 0.949 |
| In CV now: + 50% training / + coherence + 100% training | expected +0.001 / +0.004-0.005 | | | |

- If US and India transfer at their CV values, submission 2 implies **France about 0.907**.
- The best public LB score is **0.977**.
- To reach 0.977, roughly: US 0.99, India 0.975, France 0.96.

Loss split for submission 2 on train (F0.5 points lost, total 0.0395):
- retrieval 0.0102; ownership 0.0046 (the gold owner is not the argmax S1); k-selection 0.0136; false positives 0.0112
- Singletons are 46% of the false-positive damage.

By true-match count t:
- F0.5 is 0.908 at t=0, 0.912 at t=1 and 0.97 at t=5+.
- Mean (k - t) runs from +0.10 at t=0 to -0.54 at t=6+ (under-prediction for many-match entities).

Already measured, each worth under 0.003 on the full graph:
- exact expected-F decision, tuned thresholds, per-country isotonic calibration
- Hungarian / joint assignment
- a learned utility model
- a separate singleton classifier (+0.003 before the name-frequency features, ~0 after)

# 4. Constraints
- About 62 hours left and 13 submissions (5 per day).
- CPU: one on-demand 8-core/64 GB machine plus one spot 8-core/64 GB machine; up to ~$100 total AWS spend, so a couple more CPU boxes are possible.
- No AWS GPU yet (quota requests pending). Kaggle T4/P100 are available (~30 h/week).
- A 5-fold CV run takes 25-120 minutes. A feature build over 98M train pairs takes 5-10 minutes. Blocking reruns take about 1 hour.
- Unlabeled test data may be used transductively (e.g. clustering the test pool). Tell me if you think a method crosses a rule line.

# 5. What I want
For each approach below and any better one you know, give:
- mechanism
- which loss bucket it attacks, and how many points it could realistically recover there
- expected LB gain (range) and confidence
- cost (hours, compute, GPU or not)
- how to validate it with 5-fold OOF CV (acceptance: mean paired fold difference > 2 SE, positive in >= 4/5 folds)
- leakage and France risks

Approaches:
1. **Reverse-engineering the generator.** Since the operators are few and fixed, can each pool record be deterministically canonicalized back to its S1 form (inverse operators, including transliteration of native script learned from gold pairs), turning most matching into exact or near-exact lookup? How far can that go, and how would you build the Indic-script -> Latin mapping from training pairs alone?
2. **Neural representation.** Options: a multilingual/char-level bi-encoder (<=8B, MIT/Apache, e.g. small e5/LaBSE-class or a byte-level model) fine-tuned on gold pairs for retrieval, and/or a cross-encoder re-ranker on the top candidates. Is it worth it on Kaggle T4s or on CPU for 10M+ pairs? Which model, what sequence length, how many pairs to score, and expected gain over LightGBM + rapidfuzz?
3. **Pool-side clustering.** Cluster S2/S3 records first (records of one business agree with each other), then assign clusters to S1. Does this fix k-selection and ownership losses better than my per-pair model + coherence features?
4. **Set-level / listwise decision model.** A second-stage model over each S1's whole sorted candidate list (e.g. a small transformer or GBM over candidate sequences) that predicts the emitted set directly.
5. **France (15% of test, estimated 0.907).** Options: synthetic French training pairs made by applying the measured operator catalogue to the test France S1 records; self-training / pseudo-labels on confident test France matches; country-agnostic feature design. Which is safest and largest, and how do I validate without French labels (e.g. hold out US or India as a "new country" analogue)?
6. **Anything a 0.977 pipeline on generated ER data almost certainly does that mine does not.**

Then give:
- the 3 highest-expected-value bets, in order, with a go/no-go checkpoint for each (the smallest experiment that proves or kills it within 2-3 hours)
- how to split the remaining 13 submissions between safe increments and these bets

No praise, no summary of my text.
