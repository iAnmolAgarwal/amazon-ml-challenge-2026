# Final strategy (v2, 2026-09-25)

Source: plan v1 (docs/review_prompt.md), external review (docs/review_astra6.md), independent audit with measurements on training data. Only findings the audit agreed with are included.

## Facts driving the design (measured, train only)
- 2.2M train S1 / 10.3M train pool; 1.7M test S1 / 10.0M test pool; France = 15% of test S1, unseen in train.
- Each pool record matches at most one S1; no cross-country matches; 5.6% singletons; mean 3.46 matches; 26% of pool are distractors.
- 18.3% of India pool names and 22.5% of India addresses contain Indic script (Devanagari, Telugu, Kannada, Gujarati, Tamil, Bengali...). 53% of India gold pairs missed by exact keys have an Indic-script name.
- 4.6% (India) / 5.6% (US) of gold pool names are website forms.
- 52% of India and 43% of US S1 entities share a normalized name with another S1 in the same country; 34% / 20% share it with more than 5. Name-only keys are ambiguous; address agreement and the assignment constraint decide.
- Blocking measured on 30k S1: the 4 exact keys alone give pair recall 0.909 US / 0.808 India and oracle F0.5 0.966 / 0.899; adding TF-IDF top-k retrieval lifts oracle to about 0.995 / 0.979.

## Pipeline
1. **Normalization, multiple views kept side by side** (never one destructive string):
   raw; Latin-folded (accents stripped only for Latin letters, Indic marks preserved); core name (honorifics, junk prefixes, legal suffixes removed); website view (TLD stripped, e.g. `starswelfaresociety.com` -> `starswelfaresociety`); space-free key built from sorted tokens; initials; alternate name from "formerly:"/DBA; house numbers and postal codes.
2. **Indic-script view.** Learn an Indic-to-Latin token dictionary from aligned gold names on outer-train folds only (e.g. प्राइवेट -> private), minimum support 20 distinct identities. Address-only retrieval views cover what the dictionary misses.
3. **Blocking, per country:** TF-IDF top-k retrieval on name char n-grams, name+address words and address char 3-grams, unioned with exact keys (house number + street token, space-free sorted name, initials + house number). Oversized blocks are similarity-ranked, never truncated by row order. **Acceptance criterion: oracle macro F0.5 of the candidate set**, reported per country and per match count.
4. **Pair features + LightGBM**, as planned, plus entity-level features (number of candidates with p > 0.5, sum of p) and reverse-rank features computed on the full train graph.
5. **Decision rule:** assign each pool record to its best S1 (test S1 only at inference); then for each S1, sort candidates by probability and choose k = 0..n maximizing expected F0.5 ≈ 5·ΣTP / (E[T] + 4k), with explicit empty-prediction probability for singletons. No global threshold.
6. **Alias mining** from set differences between gold names/addresses (not co-occurrence), outer-train folds only, support >= 20.
7. **Output contract:** candidates written as candidates ∪ matches; official validator on the exact packaged files.
8. **Deferred (P2):** pool-to-pool assignment inheritance, US<->India transfer checks, any pretrained model (licence and <=8B verified before use).

## Harness changes required (need human approval: src/)
- fit_predict receives the train pool and the test pool separately; in predict mode, train S1 is blocked against the train pool (for labels) and test S1 against the test pool; only test S1 compete in assignment.
- Blocking for all train S1 runs once (label-free) and is cached; CV subsamples only the entities trained on or scored, keeping full-graph rivals.
- Per-entity scores saved; acceptance on paired per-fold differences (mean and std of candidate - parent), replacing the ">1 std of fold means" rule in CLAUDE.md.
- CV also reported re-weighted to test country shares (US 0.38 / India 0.47, France unknown).

## Compute
- Mac: development, CV on subsampled entities, bounded batches.
- Kaggle CPU (30 GB): full training + test inference, set up on day 1. Pair scoring in batches keeping only (s1, cand, prob); assignment in a separate global pass by pool id.

## Submissions
- Each upload has a written purpose in its log notes (e.g. baseline format + CV/LB link; retrieval coverage; decision rule; Indic view).
- Confirm with the organisers whether the final ranking uses the best or the latest submission, and which timestamp breaks ties.
- Submit the strongest validated pipeline early; keep the last two slots for the final ensemble and a safer fallback.

## Priority order
P0 (before submission 1): harness split pools + test-only assignment; TF-IDF + key blocking with oracle F0.5 acceptance; TLD strip / sorted space-free key / honorifics; batched scoring + global assignment (Kaggle path ready); candidates ∪ matches.
P1 (day 1-2): Indic view; full-graph cached candidates + reverse rank; alias mining; expected-F0.5 k selection; paired acceptance rule + test-weighted CV; submission purposes + final-selection rule.
P2: transfer checks; pool-to-pool expansion; pretrained models.
