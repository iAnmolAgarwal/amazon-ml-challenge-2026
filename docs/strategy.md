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

## Compute (v3, audited 2026-09-25; docs/review_astra6_compute.md + independent audit)
Measured on the Mac (8 threads, 2M-row same-country pool slices, 20k S1 per country):
- Search cost grows faster than linearly with pool size in one call (name 3-grams: 1629 -> 600 -> 388 queries/s per 1M pool rows at 0.5M / 1M / 2M). **Split the pool into 125-250k-row chunks and merge with `zip_sp_matmul_topn`**: linear scaling, 3-5x faster.
- The **name+address word 1-2 gram view is primary**: best recall and fastest (US recall 0.962, 5200 q/s per 1M rows; India 0.923, 1360). Name char 3-grams are weakest (US 0.719, India 0.584). Address char 3-grams: US 0.901, India slowest (407) because Indic scripts create 53-65 non-zeros per row.
- Keys + word view: oracle F0.5 0.991 US / 0.977 India on the slice, 23 / 37 candidates per S1. Adding the other two views adds only about 0.003-0.006. At full country pools expect about 0.01 lower (recall drops about 0.01 per pool doubling).
- Raising word top-k from 20 to 40 costs almost nothing (4.0 s vs 3.8 s) and gives about the same recall as adding a second view. Pruning vectors to the top-8 to top-12 grams loses 0.02-0.08 recall: rejected.
- sparse_dot_topn is Apache-2.0 and has no Linux arm64 wheel: stay on x86.

Plan:
1. Mac: chunked retrieval code, word view primary (top-30/40).
2. EC2-A, on-demand r7a.2xlarge (8 physical cores, 64 GB, x86): check chunk size and oracle (go: at least 0.985 US / 0.97 India on 20k S1 against full train pools); keys + word for all test S1 and all train S1 (about 1-1.5 h, estimated); features, LightGBM, decision pass, CV; **submission 1 at about 4-5 h**. Then development and cached submission builds (20-35 min).
3. EC2-B, spot r7a.2xlarge: name 3-gram and address 3-gram views for train and test, sharded and checkpointed (about 12 Mac-hours). Adopted only if full-pool oracle and CV show a significant paired gain. Release when done.
4. Mac: editing, agents, orchestration only. Kaggle not needed.
5. Data and code move over SSH (rsync); no AWS credentials on the instances. Both instances shut down automatically at 2026-09-27 18:30 UTC (deadline). Cost estimate about $40-60.

## Submissions
- Each upload has a written purpose in its log notes (e.g. baseline format + CV/LB link; retrieval coverage; decision rule; Indic view).
- Confirm with the organisers whether the final ranking uses the best or the latest submission, and which timestamp breaks ties.
- Submit the strongest validated pipeline early; keep the last two slots for the final ensemble and a safer fallback.

## Priority order
P0 (before submission 1): harness split pools + test-only assignment; TF-IDF + key blocking with oracle F0.5 acceptance; TLD strip / sorted space-free key / honorifics; batched scoring + global assignment (Kaggle path ready); candidates ∪ matches.
P1 (day 1-2): Indic view; full-graph cached candidates + reverse rank; alias mining; expected-F0.5 k selection; paired acceptance rule + test-weighted CV; submission purposes + final-selection rule.
P2: transfer checks; pool-to-pool expansion; pretrained models.
