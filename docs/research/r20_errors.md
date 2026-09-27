# r20: where 027e_meta still loses, and what can be fixed

Parent: `027e_meta` train OOF (CV 0.99051; US 0.99074, India 0.99017), harness decide (one owner per pool record,
expected-F0.5 prefix, floor 0.6). Tags: **M** measured, **D** derived. Code and logs: g1 `~/aml/research/r20/`
(`decomp.py`, `calib.py`, `miss.py`, `oracle.py`, `fncount.py`, `ex.py`, `floorseg.py`, `abl.py`). Times IST, 27 Sep.

## 1. Verdict
- **There is no big gain left inside the candidate set.** Of the 0.00949 loss, 0.00313 is true pairs never retrieved,
  0.00092 false positives, and 0.00545 true pairs that are candidates but get p < 0.6. 84% of those in-candidate misses
  (112k of 134k pairs) are pool records with an empty address, and 96.6k of them are claimed by a name twin: another S1
  with the same normalized name. The empty-address twin oracle is +0.00424 of the +0.00636 in-candidate oracle, and it
  is not reachable: the twins are exchangeable (r11/r12, confirmed here on 240 examples), and the model is calibrated in
  every segment.
- The one-owner rule is not the problem: only 1,329 true pairs lose the pool record to an emitted wrong pair.
  Segment floors (empty-address 0.5-0.7, non-empty 0.55-0.7) are all flat (|diff| <= 0.00001, M).
- **One small fix passes: `030_fix`**, a pool-context correction boosted from logit(parent p): name-twin structure
  among the pool record's claimers, pool-record flags, raw-name pair features. Delivered rule (`NONEMPTY = "min"`):
  only empty-address records may gain probability (decoys never have an empty address); every other pair can only go
  down. On 027e: **+0.00010 (SE 0.00001, 5/5; US +0.00006, India +0.00016) PASS**. On the new base 027h_meta
  (CV 0.99074, which already carries part of this signal): **`030h_fix` +0.00006 (SE 0.00001, 5/5; US +0.00007,
  India +0.00004) PASS**, CV 0.99079, test guard clean. Expected LB +0.00003 to +0.00006 (D). Far from 0.9945.
- Broader retrieval does not pay either (section 5b): on 20,113 missing true pairs, after tie-aware ranking (a twin
  with the same score counts as ahead), a reverse name char-3gram + text char-3gram view queried from unclaimed pool
  records finds 17% of all misses at k = 3 (7.8 new pairs per train S1) but only 4.9% of the misses that remain after
  the existing extra key / reverse-text candidates (4.3% of their loss, about +0.0001 oracle). The remaining misses are
  name twins and scrambled names, not a missing key.

## 2. Loss decomposition (M, `decomp.py`)
Per S1, the F0.5 loss is split by fixing error types in a fixed order: remove false positives, then add true pairs
lost to another owner, then true pairs cut by the prefix (p > 0.6), then true pairs below the floor, then true pairs
missing from the candidates. The terms add up to the total loss 1 - 0.99051 = 0.00949.

| bucket | pairs | loss (F0.5 points) | US | India |
|---|---|---|---|---|
| (c) false positives | 7,545 | 0.00092 | 0.00050 | 0.00042 |
| (d) true pair's pool record owned by another S1 (higher p, almost always also below the floor) | 86,504 | 0.00353 | 0.00240 | 0.00114 |
| (b1) true pair owned, p > 0.6, cut by the prefix | 8,072 | 0.00026 | 0.00016 | 0.00009 |
| (b2) true pair owned, p <= 0.6 | 39,538 | 0.00166 | 0.00113 | 0.00053 |
| (a) true pair missing from the candidates | 78,793 | 0.00313 | 0.00137 | 0.00176 |
| total | | 0.00949 | 0.00556 | 0.00394 |

Country columns are the bucket's loss summed over that country's S1 and divided by all 2,206,821 S1, so US + India =
all. Per-country F0.5: US 0.99074, India 0.99017.

What (d) really is: only 1,329 of the 86,504 owner-conflict pairs lose the pool record to an emitted pair (a
false positive on the other S1). In the other 85k cases both S1 are below the floor. So the one-owner rule itself costs
almost nothing; (d) and (b2) together are "true pair in the candidates with a low probability". All in-candidate
misses by probability (M): p < 0.3: 78,746; 0.3-0.6: 47,234; >= 0.6 (cut by the prefix): 8,134.

By true match count t (loss per bucket, all S1 as denominator):

| t | S1 | F0.5 | FP | owner | cut | floor | missing |
|---|---|---|---|---|---|---|---|
| 0 | 123,247 | 0.99632 | 0.00021 | 0 | 0 | 0 | 0 |
| 1 | 119,157 | 0.96908 | 0.00013 | 0.00079 | 0 | 0.00019 | 0.00056 |
| 2 | 375,212 | 0.98888 | 0.00020 | 0.00057 | 0.00006 | 0.00046 | 0.00061 |
| 3 | 530,841 | 0.99129 | 0.00018 | 0.00071 | 0.00007 | 0.00044 | 0.00069 |
| 4 | 484,115 | 0.99212 | 0.00011 | 0.00066 | 0.00006 | 0.00030 | 0.00059 |
| 5 | 321,957 | 0.99254 | 0.00006 | 0.00045 | 0.00004 | 0.00016 | 0.00038 |
| 6 | 164,868 | 0.99281 | 0.00002 | 0.00023 | 0.00002 | 0.00007 | 0.00019 |
| 7+ | 87,424 | 0.99297 | 0.00001 | 0.00013 | 0.00001 | 0.00003 | 0.00010 |

Singletons (t = 1) are the worst per S1 (0.969): one missed pair costs the whole entity. S1 predicted empty
(126,820 S1, most of them true empties; F0.5 0.969 India / 0.968 US) carry 0.00095 + 0.00022 + 0.00066 of the owner /
floor / missing loss.
S1-side traits: S1 records are clean (no native script, no empty address in S1). India S1 with no house number are
better than average (0.992 vs 0.990). Name length matters in both countries (2-token names 0.987-0.988, 5+ tokens
0.993-0.994).

## 3. The pool record is what matters (M, `calib.py`, `fncount.py`, `oracle.py`)
Candidate pairs split by the pool record: A = address empty / "None" (name only); B = native-script name;
C = name shares no token with the S1 (pseudo-word or unrelated name, address-only evidence); D = the rest.

| segment | candidate pairs | gold | in-candidate misses (p < 0.6 or cut) | of which a name twin claims the record | emitted FP |
|---|---|---|---|---|---|
| A empty address | 2,902,893 | 294,804 | 112,191 | 96,602 | 2,520 |
| B native name | 591,973 | 546,845 | 629 | 195 | 350 |
| C no name overlap | 869,843 | 689,917 | 14,935 | 35 | 865 |
| D other | 6,677,723 | 6,028,006 | 6,359 | 204 | 3,810 |

"Name twin" = another S1 with the same `name_join` also claims the pool record with p > 0.01.
Calibration: gold rate = mean p within 0.01-0.03 in every decile of every segment (A: 0.142/0.142, 0.248/0.245,
0.349/0.344, 0.460/0.456, 0.530/0.534 for the 0.1-0.6 deciles). The model is not under-confident anywhere.

Oracles (set p = y for a subset of candidate pairs, re-decide; M):

| subset fixed | F0.5 | gain | US | India |
|---|---|---|---|---|
| all candidate pairs | 0.99687 | +0.00636 | 0.99771 | 0.99561 |
| A empty address, all | 0.99534 | +0.00484 | 0.99620 | 0.99406 |
| A empty address, name twin claims it | 0.99475 | **+0.00424** | 0.99550 | 0.99362 |
| A empty address, no twin | 0.99111 | +0.00060 | 0.99145 | 0.99060 |
| non-empty, all (B + C + D) | 0.99207 | +0.00156 | 0.99228 | 0.99175 |
| C no name overlap | 0.99121 | +0.00070 | | |
| B native name | 0.99059 | +0.00008 | | |
| D other | 0.99130 | +0.00079 | | |
| false positives only (y = 0, p >= 0.5 set to 0) | 0.99143 | +0.00092 | 0.99157 | 0.99121 |

**Two thirds of the in-candidate headroom (+0.00424 of +0.00636) is empty-address pool records shared by name
twins.** r11/r12 showed these are exchangeable: twins never differ in case, accents or spacing, and the best raw-string
signal picks the owner 0.317 of the time against 0.239 by chance. This is the ceiling problem, not a modelling gap.

Missing candidates (a, 78,793 pairs, loss 0.00313; `miss.py`): empty-address pool records 42,214 pairs (loss 0.00164;
39,623 with a name that differs from the S1's after normalization, i.e. typos or scrambles on a name-only record);
address present 36,579 (0.00148; 3,596 native-script names whose transliteration equals a twin-shared name).
r8 measured reverse retrieval for empty-address misses: only 13% are caught at k = 3.

## 4. Categorised sample (200+ examples, M, `ex.py`)
240 random in-candidate misses (p < 0.6 or cut), with raw strings of the S1, the pool record and the S1 that owns it:

| category | count | share |
|---|---|---|
| empty address, another S1 with the identical name (up to case/punctuation) owns the record | 113 | 47% |
| empty address, the owning S1 differs only by a legal / filler word ("Family Program" vs "Family Program LLC") | 22 | 9% |
| empty address, another near-name S1 owns it | 12 | 5% |
| empty address, the true S1 is the top claimer, same name (twins below it) | 19 | 8% |
| empty address, the true S1 is the top claimer, name edited (typo, accent, suffix / filler added, truncation) | 33 | 14% |
| pseudo-word name ("Viozephdelta", "Wexlum"), address-only evidence, other S1 at the same street address (different unit) owns it | 17 | 7% |
| pseudo-word name, address-only, no other claimer | 15 | 6% |
| name ok, house number changed or truncated (231 -> 23, 105 -> 10, 236 -> 144) | 8 | 3% |
| native-script name | 1 | 0.4% |

83% of the sample has an empty address. Typical pairs:
- `Office of Public Works | 1560 Liberty Street, Harrisonburg City, VA` vs pool `Office of Public Works | None`; the
  owner is `Office of Public Works | 1133 14th Street, Washington, DC` (p 0.010 vs 0.011).
- `Corner Bike Shop` (true, p 0.006) vs `Corner Bike Shop Partners` (owner, p 0.407); pool `Corner Bike Shop [Partners]`:
  the generator added a bracketed filler word that happens to be the twin's real word.
- `Anchor | 2392 Traymore Road, University Heights, OH` vs pool `Viozephdelta | TRAYMORE ROAD, UNIVERSITY HEIGHTS, OH`
  (name replaced by a pseudo-word and house number dropped), p 0.493.
- `Norton, Lorna R., DMD | 320 11th Avenue, Unit 324, Nashville` vs pool `UMBRAONYX | 320 11TH AVE, NASHVILLE, TN`;
  owner `Feldman & Streeter LLC | 320 11th Avenue, Unit 204`: the unit is dropped, so every S1 in the building is a
  candidate.
False positives (80 sampled) mirror this: decoys with a shifted or truncated house number (2801 -> 281, 432 -> 43,
10106 -> 10108 plus a street typo) and empty-address records given to the wrong twin.

Patterns the features miss: none that is pair-local and not already covered by er_gen / gf1-gf8 (suffix / filler
add-drop, brackets, accents, case, truncation, digit-for-letter). The misses are low-probability by construction:
the evidence names more than one S1 equally well.

## 5. Fixes tested (paired vs 027e_meta on the harness folds, verify rule: mean > 2 SE and >= 4/5 folds)
| fix | what | paired vs 027e_meta | US / India | verdict |
|---|---|---|---|---|
| floor 0.5 / 0.55 / 0.65 / 0.7 on empty-address pairs only | decision rule by sub-population | -0.00000 / -0.00000 / -0.00000 / -0.00001 (<= 3/5) | | flat |
| floor 0.55 / 0.65 / 0.7 on non-empty pairs only | decision rule by sub-population | -0.00000 (1/5 each) | | flat |
| exact expected F0.5, unowned mass, ownership relaxation | decision rule | measured in r4 / r11: <= +0.00001 | | not redone |
| `030_fix` full (`NONEMPTY = "full"`, file `030_fixF`) | pool-context correction, 33 inputs | **+0.00011** (SE 0.00001, 5/5) | +0.00006 / +0.00017 | PASS |
| `030_fix` empty-address rows only (non-empty keep p) | same model | +0.00003 (SE 0.00001, 5/5) | +0.00003 / +0.00002 | PASS, small |
| **`030_fix` delivered (`NONEMPTY = "min"`)** | empty-address rows corrected, non-empty rows min(p, corrected) | **+0.00010** (SE 0.00001, 5/5) | +0.00006 / +0.00016 | **PASS** |
| `030_fix` feature groups: pool-side only (`pr`) | ablation on 027e | +0.00001 (SE 0.00000, 5/5) | +0.00001 / +0.00001 | FAIL: the gain needs the twin / flag / raw-name inputs |
| **`030h_fix` = delivered rule on the new parent 027h_meta** | same code, PARENT = 027h_meta | **+0.00006** (SE 0.00001, 5/5), CV 0.99079 | +0.00007 / +0.00004 | **PASS** |
| `030h_fixF` = full version on 027h_meta | | +0.00006 (SE 0.00001, 5/5) | +0.00007 / +0.00005 | PASS, same gain, riskier on test |

Model (`src/er_fix.py`, attempt `attempts/030_fix/train.py`): rows = US/India pairs with 0.01 <= p <= 0.99
(1,731,650 train, 227,552 positives); LightGBM 63 leaves, 300 rounds, lr 0.05, boosted from logit(p), 5-fold cross-fit
on the harness folds; test = one model on all train band rows. Inputs:
- pool side (er_pool, recomputed from 027e's own p): rank, best other claimer, margin, sum of others, claimers > 0.01 / > 0.5;
- name twins among the pool record's claimers (p > 0.01): number of claimers with this S1's `name_join`, best other
  twin's p, probability mass on claimers with a different name, twin share of the mass;
- pool-record flags: empty address, native-script name / address, brackets, all upper / lower case, accents, token count;
  S1 flags: address has a unit, token count;
- raw-name pair features: exact / lowercase equality, punctuation-set equality (the r12 lead), punctuation counts,
  `name_join` equality, core-token Jaccard, last-token equality, containment both ways, length ratio.
Nothing compares a pair with the S1's other candidates. Top gain (test model): pr_sum, len_r, lp, pr_margin, pr_other,
core_jac, pr_rank, p_ntok, tw_best, s_ntok, pr_n01, punct_s.
Where the gain comes from (027e parent, train, delivered version, per 1k S1): adds 0.99 US / 0.81 India empty-address pairs at 79% /
76% precision; drops 0.47 / 1.45 non-empty pairs of which 38% / 53% were false; drops 0.46 / 0.36 empty-address pairs
(62% / 60% of them true).

Test guard on 027h (M; per 1k S1, train -> test): `030h_fix` adds empty-address pairs US 0.91 -> 1.19, India 0.74 ->
1.00 (train precision 80% / 79%); adds non-empty 0.006 -> 0.011 / 0.002 -> 0.006; drops non-empty US 0.27 (conflict
0.62) -> 0.28 (0.77), India 0.41 (0.31) -> 0.50 (0.37); drops empty 0.54 -> 0.39 / 0.36 -> 0.23. Pairs/S1 US 3.3877 ->
3.3882, India 3.3729 -> 3.3732; empty S1 rate 0.0577 unchanged; emitted conflict US 0.0877 -> 0.0876, India 0.1743 ->
0.1742; France untouched. The full version `030h_fixF` adds non-empty pairs at 1.9x the train rate on test (India 0.52
-> 0.99, conflict 0.42 -> 0.54) for the same CV, so the restricted rule is the one to use.

Test guard on 027e (M; `verify.py guard`, `diff.py`; per 1k S1, train -> test):

| | empty-address added | non-empty added | non-empty dropped (house-conflict share) | empty-address dropped |
|---|---|---|---|---|
| US | 0.99 -> 1.73 | 0.003 -> 0.008 | 0.47 (0.68) -> 0.56 (0.71) | 0.46 -> 0.37 |
| India | 0.81 -> 1.20 | 0.009 -> 0.007 | 1.45 (0.46) -> 2.09 (0.44) | 0.36 -> 0.29 |

Pairs/S1 on test: US 3.3863 -> 3.3871, India 3.3735 -> 3.3724; empty S1 rate 0.0576 -> 0.0576 / 0.0577; emitted
house-conflict US 0.0877 -> 0.0875, India 0.1744 -> 0.1742; France untouched. Non-empty drops are larger on test
(the conservative direction, as expected with more decoys). The one flag: empty-address additions per S1 are 1.5-1.8x
the train rate. Empty-address records are not decoys (r12: same empty-address rate per S1 on test while pool/S1 grows),
so the risk is giving a record to the wrong twin, not emitting a decoy. A plausible cause is that test has fewer S1 per
name (US test has half as many S1), so fewer twins claim each record and the correction trusts the top claimer more;
that is the direction the parent cannot see (G). The full version (`030_fixF`) adds 1.07 US / 1.17 India non-empty
pairs per 1k test S1 (train 0.55 / 0.59, conflict 0.53 / 0.45): that asymmetry is why it is not the delivered one.

## 5b. Retrieval for the missing bucket (a) (M, `retr.py`, `retr2.py`)
Sample: 20,113 missing true pairs (all 5,113 in the extra-candidate slice s1_idx % 10 == 3, plus 15,000 others);
56.5% have an empty-address pool record, 97.2% have a pool record that no S1 claims with p >= 0.5 ("unclaimed").
Unclaimed pool records are 27.6% of the pool, so a reverse top-k view queried only from them costs about
k x 0.276 x 4.68 = 1.3k new pairs per train S1 (more on test, where pool/S1 is 5.8 and decoys are unclaimed).
Ranks are tie-aware: every S1 with a score >= the true S1's counts as ahead, because a tied name twin would get the
same probability and never be emitted. (Counting ties in the S1's favour inflates recall@1 from 0.05 to 0.31: that
difference is exactly the twin problem.)

| view (unclaimed pool records -> S1 of the country) | recall @1 / @3 / @5 / @10, all misses | same, extra-candidate slice | new pairs per train S1 at k = 3 |
|---|---|---|---|
| name char-3gram TF-IDF | 0.051 / 0.087 / 0.110 / 0.151 | 0.020 / 0.040 / 0.057 / 0.088 | 3.9 |
| text (name + address) char-3gram | 0.087 / 0.119 / 0.137 / 0.167 | 0.004 / 0.012 / 0.019 / 0.035 | 3.9 |
| union of both | 0.120 / 0.173 / 0.201 / 0.252 | 0.023 / 0.049 / 0.070 / 0.110 | 7.8 |

Share of the slice's missing-pair loss covered by the union: 2.1% / 4.3% / 6.5% / 10.0% at k = 1 / 3 / 5 / 10. With
slice missing loss 0.00203 and the measured conversion of about 1/3 (the extras cut missing loss by 0.00122 on the
slice for +0.00041 F0.5), the realistic gain at k = 5 is about +0.00004 (D) for 13 new pairs per S1.
Exact keys (block size <= 20 S1): name_join / name_skel / name_sorted / name_fl recall 0.09-0.11 of the slice's
misses at 3.7-6.2 new pairs per S1, but most hits are twin names; initials + house, house + street1, nums + street1,
street1 + first token recall 0.003-0.02 of the slice. Text word TF-IDF and address char-3gram views add little even with ties counted in the S1's favour (slice recall @5
0.05; address view 0.00 on empty-address records).
Verdict: no cheap key or view closes bucket (a); more machines would not change that.

## 6. Files
All on g1 `~/aml/cache/oof/` (harness format s1_idx, pool_idx, prob; same keys as the parent; md5 checked g2 = g1):
- `030_fix.parquet` / `test_030_fix.parquet` (md5 ec3ff3cc / 80887feb): delivered rule on parent 027e_meta, written by
  `python src/er_fix.py 030_fix` (g2).
- `030h_fix.parquet` / `test_030h_fix.parquet` (md5 7b92c3cb / 0bc653a8): delivered rule on parent 027h_meta, written by
  `python src/er_fix.py 030h_fix` (g2). **This is the one to stack on the current base.**
- Research files: g1 `030_fixF` / `030_fixE` / `030_fixD` (027e: full, empty-only, restricted via `combo.py`), g4
  `030h_fixF` (027h, full).
- France rows keep the parent's p in every test file.
Code: `src/er_fix.py`, `attempts/030_fix/train.py`, `attempts/030h_fix/train.py`, `attempts/030h_fixF/train.py` (Mac
repo; copies on g1, g2, g4; not committed). Research scripts: g1 `~/aml/research/r20/` (retrieval: g4). Runtime on an idle
12-core machine: 2 min for train CV + test. Nothing uploaded.
