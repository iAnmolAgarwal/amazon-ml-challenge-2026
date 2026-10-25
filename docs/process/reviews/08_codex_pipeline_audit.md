# Gate 8 audit: the second (Codex-built) pipeline vs the main pipeline

Auditor: independent (Opus), 2026-09-25 ~11:35 IST. Inputs: `~/Downloads/matching_results.tsv`, the parallel pipeline's package zip in Downloads (263 MB). The zip was extracted to the session scratchpad (`task_audit/`). Nothing from that pipeline was trained or run. Test files were used label-free only; train labels only for reference rates. Scripts are in scratchpad `task_audit/work/` (`cand_check.py`, `compare.py`, `diff.py`, `samples.py`, `house.py`, `m24.py`). "Main" = `submissions/matching_results.tsv` = `archive/03_004_coherence_full.tsv` (sha256 e40a4b47...). "Codex" = the second pipeline.

## 0. Bottom line

- **Validity: PASS.** The official validator with `--check-ids` passes on the Downloads file. Its only warning is that no candidate file was passed. The candidate file is too large for the stdlib validator on 8 GB (38.8M ids in Python sets), so every rule was re-checked in polars and all pass. Every match is inside the candidates. The Downloads file is byte-identical to the zip's `output/matching_results.tsv`.
- **Package: not submittable as-is.** Its methodology document names a team and members that do not match this harness's submissions. The cloud scripts hard-code an AWS account id in a bucket name. `make predict` would not reproduce the shipped output. There is no AI-tool mention anywhere in the package.
- **Which is better: probably the Codex pipeline, by roughly +0.01 on US+India.** Its reported leak-light OOF CV is India 0.9710 / US 0.9784, against main 004b's India 0.9476 / US 0.9741. The folds differ, so this is not a paired comparison. The India gap is too large to be noise and is mostly the Indic dictionary, which main 005 is adding now.
- **Red flag for main 004 (independent of Codex).** On test, 004 predicts +0.30 more matches per US S1 than 002 (3.59 vs 3.29), and +0.16 more in France. In CV the whole change was about +0.025. Only 18.6% of the 216k US pairs that 004 added share any address number with the S1, against 92.8% of train gold pairs. For France the figure is 8.6%. This looks like near-miss distractors being accepted, i.e. a test-time feature shift, most likely in the feat3 coherence features. Check it before building 005 on top of 004.
- **Worth adopting:** (1) stage-2 "competition" features from OOF stage-1 probabilities; (2) a probability floor in `decide()`, tuned with leave-one-country-out; (3) record→S1 reverse retrieval plus conjunction keys. Probability-level blending of the two pipelines is also worth a CV test. A union of the match sets is not.

## 1. Format and compliance

### 1a. Validator and file checks

| Check | Result |
|---|---|
| Official validator, `--check-ids`, Downloads `matching_results.tsv` | **PASS**. 1,732,544 rows (98,583 empty). One WARNING: `output/candidate_pairs.tsv not found - skipping candidate checks` (the candidate file was not passed on this run). Peak about 2.9 GB. |
| Bundled `utils/validate_submission.py` | identical to the official file |
| Downloads file vs zip `output/matching_results.tsv` | **identical**, sha256 `3b4b35bd568c...a2ce8` |
| `candidate_pairs.tsv` (official validator not run: about 4-5 GB of Python sets on 8 GB; re-checked in polars) | header correct; 1,732,544 rows = unique S1 = every test S1; 38,837,173 ids; 0 malformed / empty tokens; 0 duplicates within a list; 0 ids missing from test S2/S3; 0 cross-country pairs; 0 leading zeros, spaces or CR; 1 row with an empty list |
| Matches inside candidates | **0 of 5,797,550 outside** (every match is a candidate) |
| One pool record, one S1 | 0 pool ids appear under more than one S1, in both Codex and main |
| Candidate semantics | `predict.py` writes every blocked pair of every block, which is exactly what the model scored. This matches the README's "exact set fed to the model". |

### 1b. Zip structure vs README "Final Submission Package"

| Required | Present | Note |
|---|---|---|
| `output/matching_results.tsv`, `output/candidate_pairs.tsv` | yes | 97 MB + 523 MB (263 MB zipped) |
| `code/business_entity_resolution/src/` | yes | 15 .py files, 75 KB |
| `README.md` with run instructions | yes | the `make` pipeline |
| `requirements.txt` pinned | yes | all `==`. `jellyfish` and `tabulate` are pinned but never imported (harmless). |
| `Documentation_template.md` filled | yes | every template section filled, no placeholders left |
| extras | `configs/`, `cloud/`, `utils/validate_submission.py`, `Makefile`, `cache/translit.json` (1,318 token maps, 45 KB), `cache/translit_addr.json` (16 state names) | the JSON files are derived from train gold and rebuilt by `make translit`: acceptable, no raw data |

### 1c. Issues to fix before the zip could be used

1. **Identity (blocking).** `Documentation_template.md` carries a team name and member list that do not match this harness's submissions. For a solo registration this is wrong and could read as misrepresentation. It must match the portal registration exactly, and the zip name `<team_name>_submission.zip` must use the registered team name.
2. **AWS identifiers.** `cloud/run_pipeline.sh` and `cloud/run_pairs_test.sh` hard-code `s3://amazon-ml-er-(redacted)` (a 12-digit account id), region ap-south-1 and "Free-tier worker". These are not credentials, but they should be parametrised or removed. Note that the doc's "three 8 GB cloud workers" is fine: it is compute, not data.
3. **Reproducibility.** The shipped output used `predict.py --models models/final --n_models 2 --rule '{"type":"expected_f","min_p":0.6}'` (README). But `make predict` runs the default rule (`expected_f`, no floor) with 5 models, and `cloud/run_pipeline.sh` uses `final_cloud` with min_p 0.5. `make` therefore does not regenerate the submitted file. The Makefile must be aligned to the rule actually used. The doc also cites `notes/eda.md` and `notes/france_check.md`, which are not in the zip. `package.py` expects a `README_pipeline.md`.
4. **Licences.** No pretrained model; the matcher is LightGBM (MIT). `unidecode` is GPL-2.0. It is a library, not a model, so constraint 5 (MIT/Apache model of 8B parameters or fewer) is not violated, but a strict reviewer could ask about it. The main pipeline avoids unidecode. sparse_dot_topn is Apache-2.0; rapidfuzz is MIT.
5. **External data.** None found. No network calls in `src/`; the only downloads are packages and the team's own S3 copy of the challenge data. The dictionaries are mined from train gold. Blocking and feature IDF are fitted on test S1 text: label-free and transductive, and the doc says so. The main pipeline does the same.
6. **AI-assistant traces: none.** Case-insensitive grep over all code, docs, README, configs and scripts for claude, anthropic, codex, gpt, openai, chatgpt, copilot, llm, agent, assistant, gemini, cursor and "generated by": only false positives (the variable `ai` = S1 positions; "auto-generated by src/eda.py"). No zip or file comments, no `__MACOSX` or `.DS_Store`, no notebooks, no caches except the two dictionary JSONs, no credentials or keys.

## 2. Approach of the Codex pipeline

- **Normalization** (`normalize.py`): the mined dictionary is applied first (non-Latin word → Latin word; non-Latin trailing address component → state), then `unidecode`, lowercase, `&`/`+`→and, alias split (formerly/fka/dba/aka/t/a; the alias is kept as a separate field), domain and handle stripping, 7+ digit runs removed, abbreviation canon (pvt→private, ltd→limited, corp, tech…; street/st/saint→st, avenue/ave/av→av, rue/r→r, boulevard/bd…). Legal forms (incl. SARL/SAS/EURL/SASU/SCI/EI/LLC/LLP) go to their own field, and misspelled long legal words are canonicalised through the skeleton. House numbers lose leading zeros, ordinals are stripped, and every word gets a phonetic **consonant skeleton**. There is no per-country branch: French and Indian vocabularies are applied to every record.
- **Blocking** (`block_feats.py`, `blocking.py`): about 37 hashed keys per record: name tokens, skeleton tokens, adjacent name-token pairs, first×last token, space-free name and space-free skeleton, skeleton×first house number, address tokens, address bigrams, house number×street word. Keys are IDF-weighted on the S1 of each country block, keys with S1 df > 3000 are dropped, and vectors are cosine-normalised. Retrieval runs in both directions: **each pool record → top-3 S1** and **each S1 → top-15 pool records** (min cosine 0.1), then the union. Test candidates per S1: France 23.5, India 22.7, US 21.7 (main: 50.6 / 45.8 / 42.9). Reported train pair-recall ceilings: India 0.9747 (0.960 without the dictionary), US 0.9859.
- **Features** (`features.py`): about 55 pairwise features. Hashed TF-IDF cosines (char-3gram on the space-free name, name words, skeleton words, skeleton char 2-3grams, address words); number and legal-form overlap (shared / Jaccard); rapidfuzz ratio / token_set / token_sort / partial / Jaro-Winkler on core names, space-free names and skeletons; alias-vs-name; address token_set / sort / partial_token_set / ratio; first-number equality and missingness; lengths; flags (non-Latin, domain, alias, S3); blocking cosine and both-direction ranks. Country is never a feature.
- **Model** (`model.py`, `run_experiment.py`): LightGBM (127 leaves, lr 0.08, up to 800 rounds, early stopping), in two stages. **Stage 2 adds 11 "competition" features computed from OOF stage-1 probabilities**: for the pool record, rank, best other S1's p, margin to it, number of competitors and sum of p; for the S1, rank, gap to the best, best-other, count and sum of p. Folds are crc32(S1 id) mod 5. Training uses **entity-level sampling: 7% of entities for the final model**, 15% for the India-only runs.
- **Decision** (`decision.py`, `run_experiment.decide`): each pool record keeps only its best S1 (others set to 0; this handles the one-record-one-S1 constraint). Per S1, sort by p and take the prefix maximising 5·ΣTP/(4k + E[N]); k = 0 is worth Π(1−p). This is the same rule as main `decide()`. On top, **a floor p > 0.6**, chosen on leave-one-country-out (train India → score US), where it raised unseen-country singleton accuracy from 0.928 to 0.942 at F within 0.0001.
- **France:** no French data and no country branch. Generic French abbreviations and legal forms are in the dictionaries; the stricter floor was chosen on leave-one-country-out (LOCO). The same absolute threshold was kept after checking that the France score histogram is as bimodal as the known countries'.
- **Native script:** a positional-alignment dictionary mined from train gold (same token count; support ≥ 3, majority share ≥ 0.6; 1,318 tokens, 7 scripts). It covers 94.7% of non-Latin test name tokens. On top: a state-name map, a unidecode fallback for unmapped words, and skeleton keys that absorb transliteration spelling variants.

**Reported validation** (doc §5, App. B; not re-run by me):

| Setting | macro F0.5 | P | R | singleton acc |
|---|---|---|---|---|
| India only, 5-fold OOF, expected-F min 0.6 | 0.9727 (fold std 0.0003) | 0.992 | 0.943 | 0.971 |
| Train India → score US (LOCO, unseen country), min 0.6 | 0.9627 | 0.985 | 0.932 | 0.942 |
| Final model (US+India, 7% sample), OOF | India **0.9710**, US **0.9784** | | | |

How it is computed (checked in code): `entity_scores()` covers every S1 of the block. The true count n_true comes from the ground truth, so blocking misses count as false negatives. Singletons score 1 only when the prediction is empty. It is a correct macro F0.5 per S1, singletons included. Predictions are out-of-fold per entity group, and stage-2 context uses OOF stage-1 scores (standard stacking).

Optimism sources: (a) the dictionary is mined from **all** train gold with support ≥ 3, so held-out entities' own pairs contribute. Main `er_dict` has the same design with ≥ 5 owners. The 06 audit found 100% of held-out Indic tokens covered by other folds at ≥ 5 owners, so the effect should be small. (b) The decision variant and floor were picked on the same OOF/LOCO numbers, a within-0.001 effect. **No public LB number is documented anywhere in the zip.**

## 3. Label-free comparison on test (Codex vs main 004)

### 3a. Match sets by country

| Country | S1 | identical sets | mean Jaccard | empty: Codex / main / both | mean k: Codex / main | S3 share: Codex / main | Codex pairs also in main | main pairs also in Codex |
|---|---|---|---|---|---|---|---|---|
| France | 259,452 | 60.2% | 0.862 | 4.66% / 4.13% / 3.59% | 3.52 / 3.59 | 0.520 / 0.514 | 93.4% | 91.7% |
| India | 809,986 | 63.9% | 0.862 | 5.91% / 5.37% / 4.58% | 3.30 / 3.27 | 0.511 / 0.512 | 92.2% | 92.8% |
| US | 663,106 | 68.1% | 0.885 | 5.82% / 4.67% / 4.53% | 3.34 / **3.59** | 0.516 / 0.515 | 97.4% | 90.5% |
| All | 1,732,544 | 65.0% | 0.871 | 5.69% / 4.92% / 4.41% | 3.35 / 3.44 | 0.515 / 0.514 | 94.4% | 91.7% |

Reference: train gold has mean 3.46 and 5.6% singletons in both countries, and main OOF k − t is about −0.23. Share of S1 with k ≥ 5: Codex 27.1 / 22.6 / 23.4% (France / India / US); main 28.4 / 22.9 / 28.4%.

Disagreement types (S1 count; "X superset" = X's set strictly contains the other's):

| Country | main superset | Codex superset | mixed overlap | Codex empty only | main empty only | disjoint |
|---|---|---|---|---|---|---|
| France | 47,465 | 40,258 | 10,782 | 2,780 | 1,395 | 483 |
| India | 116,898 | 127,902 | 28,229 | 10,747 | 6,368 | 2,063 |
| US | **149,078** | 37,971 | 14,223 | 8,591 | 937 | 488 |

### 3b. What the disagreeing pairs look like

Raw-text token_set similarity after accent folding (Indic text is left unfolded, so Indic names score low by construction):

| Country | group | pairs | name tset | addr tset | name ≥ 90 | Indic name | shares an address number with S1 |
|---|---|---|---|---|---|---|---|
| France | both (sample) | 62k | 92.0 | 84.6 | 0.78 | 0 | 0.989 |
| France | Codex only | 60k | 71.6 | 78.5 | 0.28 | 0 | 0.719 |
| France | main only | 77k | 86.1 | 79.3 | 0.61 | 0 | **0.278** |
| India | both (sample) | 181k | 78.4 | 90.0 | 0.61 | 0.155 | 0.984 |
| India | Codex only | 208k | 46.0 | 79.2 | 0.21 | **0.538** | 0.899 |
| India | main only | 190k | 76.4 | 85.4 | 0.41 | 0.129 | 0.717 |
| US | both (sample) | 157k | 91.9 | 86.2 | 0.76 | 0 | 0.947 |
| US | Codex only | 58k | 84.1 | 70.4 | 0.56 | 0 | 0.515 |
| US | main only | 226k | 90.8 | 83.1 | 0.66 | 0 | **0.218** |

Train gold pairs share an address number 92.8% (US) / 97.2% (India) of the time (40k sampled S1 per country, pairs where both addresses contain digits).

Reading:
- **India:** Codex's extra pairs are mostly Indic-script names that main 004 cannot read. There are 93k Codex matches outside main's candidates (3.5% of Codex's India matches). Most of these should be fixed by main 005's dictionary.
- **US and France:** main's extra pairs are mostly same-name-plus-extra-word records at a nearby but different house number (1025 vs 1029 Bird St, 2109 vs 2111 Moonlight Dr, 1905 vs 1914 Market St). The Codex doc lists exactly this pattern as the typical distractor, and the number-agreement rate (22% / 28%) is far below gold. Codex's extra pairs look more gold-like on this measure (52% / 72%).

### 3c. Main 004 vs main 002 on test

| Country | pairs 004 added vs 002 | of which also chosen by Codex | share an address number | pairs 004 removed vs 002 | removed: share a number |
|---|---|---|---|---|---|
| France | 52,886 | 10.7% | **8.6%** | 10,497 | 43.9% |
| India | 133,530 | 26.6% | 62.1% | 40,834 | 88.2% |
| US | 216,210 | 10.2% | **18.6%** | 15,345 | 51.6% |

Mean predicted k on test by upload: 001 3.33/3.14/3.23, 002 3.43/3.16/3.29, 004 **3.59/3.27/3.59** (France / India / US). In CV (004b vs 002) mean k − t moved only from −0.259 to −0.233. The test jump in US is about 12 times the CV shift. The added pairs are mostly ones that share no address number with the S1, and only 10% of them are also chosen by Codex. This is consistent with a train/test shift in the new feat3 coherence features. A plausible mechanism: test has 5.76 pool records per S1 against 4.68 in train, so more near-miss distractors qualify as "anchors" (name tset ≥ 90 with address tset ≥ 80) and inflate `coh_support`. The 100%-train refit is another candidate cause. **Recommended check (cheap, label-free):** compare the distributions of coh_n_anchor, coh_support and coh_house_conflict on test vs train OOF by country, and 004's train-OOF predicted k by country vs test. If 004's LB comes back at or below 002's 0.949, this is the first suspect.

### 3d. Blocking recall proxy

| Country | Codex candidates/S1 | main candidates/S1 | main matches outside Codex candidates | Codex matches outside main candidates |
|---|---|---|---|---|
| France | 23.5 | 50.6 | 7,720 (0.83%, 7,299 S1) | 22,650 (2.48%, 21,079 S1) |
| India | 22.7 | 45.8 | 20,723 (0.78%, 18,948 S1) | **93,458 (3.50%, 73,582 S1)** |
| US | 21.7 | 42.9 | 4,678 (0.20%, 4,549 S1) | 11,181 (0.51%, 10,735 S1) |

With half the candidates, Codex's set covers the other pipeline's matches better: main's candidates miss 2.4-4.5 times more Codex matches than Codex's candidates miss main matches. India is the dictionary effect. France and US show that the reverse-direction retrieval and conjunction keys also add reach.

### 3e. Sampled disagreements (10 random S1 per country; seed 7)

#### France
- S1-54355250 "Cordeliers Sport SARL" / "49 Rue des Salines, Calais" (Codex k=3, main k=5, shared 3). Main only: S2-201740000 "Cordeliers Sport SAS" / "70 R DES SALINES, CALAIS"; S2-796776707 "Cordeliers Sport International [SARL]" / "70 R. DES SALINES, CALAIS".
- S1-170082417 "Mouv Atelier SARL" / "21 Rue des Frères Grimm, Saint-Herblain" (4/3/3). Codex only: S3-717421539 "Mouv Gestion SARL" / "24 R Des Frères Grimm".
- S1-717955467 "Fédération des Afrika" / "22 Rue de Lavaud, Bordeaux" (4/6/4). Main only: S3-315452339 "Fédération des Afrika SASU" / "029 Rue De Lavaud"; S2-616774765 "Fédération des Àfrika Distribution" / "29 RUE DE LAVAUD".
- S1-427977412 "IQ Societe SAS" / "279 Rue Léon Gambetta, Lille" (7/6/6). Codex only: S3-461330396 "IQ Societe S.A.R.L." / "290 Rue Leon Gambetta".
- S1-963419135 "Cercle (France) Ecole SARL" / "67 Rue Robert Deschamps, Calais" (2/5/2). Main only: three records at "76 Rue Robert Deschamps" ("…Elémentaire Sarl", "…Ecole Développement SARL", "SARL … Ècole Groupe").
- S1-466382064 "Norbert & Frères SAS" / "22 Rue d'Elbeuf, Tourcoing" (3/3/2). Codex only: S2-825781455 "Norbert & Frères France SAS" / "27 RUE D'ELBEUF". Main only: S2-915590432 "Norbert Frères EURL France" / "27 RUE D'ELBEUF".
- S1-724107607 "Ak Centre" / "14 Rue des Cordeliers, Bordeaux" (5/4/4). Codex only: S3-179231202 "Ak Parents" / "No 14 R. Des Cordelaers".
- S1-329675444 "Temps Patrimoine SASU" / "4 Rue El-Alamein, Bordeaux" (4/3/3). Codex only: S3-809066823 "Temps Primaire SASU" / "6 Rue El-alamein".
- S1-982826717 "La Baule-Escoublac Lycée SAS" / "48 Avenue des Saphirs" (2/3/2). Main only: S2-47200440 "La Baule-Escoublac Lycée EURL" / "69 AVENUE DES SAPHIRS".
- S1-71837965 "Commercants Pharmacie EURL" / "4 et 6 RUE Boileau, Nantes" (3/4/3). Main only: S3-577741609 "CP" / "4 Et 6 R Bileau, Nantes" (acronym plus same address: plausibly gold).

#### India
- S1-55721884 "Versha Traders" / "Door No. 2397/A, East End, … Kozhikode" (3/4/3). Main only: S2-505075916 "Versha Traders [Company]" / "DOOR NO. 397/A, EAST END, …" (truncated number, plausibly gold).
- S1-173188885 "Ventures Tsk Logistics Private Limited" / "Mumbai, … 2Nd Floor, Peninsula Business Park" (2/4/2). Main only: S3-108626337 "venturestsklogistics.com" / "Plot 207 Flat No. A-201, Mumbai"; S2-294976729 "VENTURES TSK BOTTLES PRIVATE" / "FLAT NO. A-204, 2ND FLOOR, PENINSULA BUSINESS PARK…".
- S1-719639056 "Sada Solutions Clinic" / "83- Kennedy Avenue, Amritsar" (3/4/3). Main only: S2-807768677 "Clinic Constech Sada" / "C-104- KENNEDY AVENUE, AMRITSAR".
- S1-428285665 "Tech Industries Private Limited" / "Lok Bhavan, … Marol Maroshi Road, And…" (4/3/3). Codex only: S3-863116194 "टेक इंडस्ट्रीज प्राइवेट लिमिटेड" / "Lok Bhavan, Mumbai, Andheri (East), महाराष्ट्र".
- S1-964305432 "Right Real" / "Purva Eternity, E1804 …, Ernakulam" (2/1/1). Codex only: S2-630377126 "Mr Right (Real)" / "" (empty address).
- S1-466463154 "Maad Commerce Limited" / "House No. E-7/163, Arera Colony, Bhopal" (5/6/5). Main only: S3-522310308 "Yumacalofaye" / "Arera Colony, House No. E-7/163, Huzur, Bhopal" (random DBA name, same address).
- S1-725842170 "Raj Projects Private Limited" / "Ag- 20, 21 Radhe Business, Empire, Daskroi, Ahmedabad" (4/3/3). Codex only: S2-193521450 "રાજ પ્રોજેક્ટ્સ પ્રાઇવેટ લિમિટેડ" / "Gujarat, AHMEDABAD, DASKROI, AG- 18".
- S1-331094034 "Hcs Trust" / "S No 77/1A (Part) Mundhwa, Pune" (6/4/4). Codex only: S2-815205144 "HCS GROUP [TRUST]" / "S NO 77/10A (PART) MUNDHWA"; S3-267663345 "Hcs Trust LLP" / "S No 77/10A (Part) Mundhwa, Pune, महाराष्ट्र".
- S1-982847580 "Premier Smart Infra Pvt Ltd" / "9, Om Vihar Colony, … Indore" (4/2/2). Codex only: S2-325729996 and S3-785896974 "प्रीमियर स्मार्ट इंफ्रा प्रा. लि." / "9, … Indore".
- S1-73857258 "Friends Foods Ltd" / "01 Mankara, Moradabad" (3/4/3). Main only: S3-919597831 "Pvt Friends Foods Ltd" / "H.no 14 Mankara, Moradabad, UP".

#### US
- S1-55629539 "ZD Reliable Bitwise" / "12599 72nd Place, Fl 1, Scottsdale, AZ" (1/2/1). Main only: S3-563318359 "ZD Reliable 8itwise Corp." / "…, Fl 1, 12606 72nd Place".
- S1-172016663 "Russell & Osborn California LLC" / "1025 Bird Street, City Of Wausau, WI" (2/3/2). Main only: S2-580624586 "Russell & Osborn Development LLC" / "1029 BIRD ST".
- S1-717526048 "Shandy Sam Heartland Sound LLC" / "1905 Market Street, Wichita" (3/5/3). Main only: S3-143349639 "Shandy Sam Heartland Sound Group" / "1914 Market St"; S2-636237344 "Shandy Sam Heartland Family Practice LLC" / "1914 MARKET STREET".
- S1-427667711 "The Dent Hair Studio" / "9439 Willow Bend Lane, Phoenix, AZ" (3/5/3). Main only: S2-975947484 "The Dent Hair Studio Corp" / "009443 WILLOW BEND LN, MARYVALE"; S3-709331042 "The Dent Hair Stúdio Ltd" / "9443 1/2 Willow Bend Lane".
- S1-963246620 "Secure Genomic Consultants Inc" / "1146 Lodge Road, Northumberland County, VA" (5/4/4). Codex only: S3-856568723 "Secure Genomic" / "1146 Ldoge Road, Calllao, Virginia".
- S1-466159466 "Hughlett Highview LLC" / "2109 Moonlight Drive, Village Of Suamico" (3/4/3). Main only: S2-254268709 "Hughlett Highview Corp." / "2111 MOONLIGHT DRIVE".
- S1-723607714 "Foot & Ankle Horizon Associates" / "2021 7, Bainbridge, NY" (2/4/2). Main only: S3-763537016 "… Ássociates Corp" and S3-141095154 "… Associates Uptown" / "2024 7, Bainbridge".
- S1-329711742 "Mutz & Howton LLC" / "116 Bentley Court, Hagerstown, MD" (3/4/3). Main only: S3-500301989 "Mutz & Howton Corp" / "00119 Bentley Ct".
- S1-982058990 "Ruthy Haralson Tri-State Praetorian Inc" / "2406 Hurley Circle" (5/6/5). Main only: S2-709390631 "… Praetorian Partners" / "2427 HURLEY CIRCLE".
- S1-73617085 "Geissler & Glenn" / "78 R Mount Vernon Street, Somerville, MA" (3/4/3). Main only: S2-84881408 "Geissler & Glenn Holdings" / "87 R MOUNT VERNON STREET".

Pattern: in 9 of 10 US samples, main alone adds a record whose house number differs slightly from the S1's (+2 to +21), usually with an extra name word. Codex's extras are typos of the same number ("Ldoge Road", "Cordelaers"), or Indic transliterations in India.

## 4. Verdict

**Likely better now: the Codex pipeline.** The evidence, in order of weight:
1. **CV** (not paired, different folds, both OOF over all S1 with singletons): India 0.9710 vs 0.9476 (+0.023), US 0.9784 vs 0.9741 (+0.004). The known-country part of the test-weighted CV is about +0.0126 in Codex's favour (0.47×0.023 + 0.38×0.004). The 06 audit's dictionary proxy (+0.0177 India) explains most of the India gap, so after 005 lands the expected gap shrinks to roughly +0.003-0.006.
2. **Unseen-country evidence:** Codex measured a LOCO drop of 0.978 → 0.963 (India-trained → US) and chose a stricter floor for it. Main has no LOCO measurement.
3. **Test behaviour:** main 004's US/France extra predictions look distractor-like (§3b-c) and sit far outside what its CV implied. Codex's predicted-k profile is close to train truth (5.7% empty; k 3.30-3.52).

Against Codex: it trains on only a 7% entity sample (main's learning curve says more data helps by +0.0007-0.0009 per doubling). It has no France-specific normalisation (its France-only pairs have lower name similarity: 71.6 vs 86.1). Its CV was not independently reproduced. There is no LB evidence. The package has the §1c issues.

Under CLAUDE.md rules this is not yet a decision. A decision needs a paired comparison. Both pipelines produce OOF predictions for every train S1, so per-entity F0.5 from Codex (recomputable from its saved `cache/exp_final/p2_<block>.npy` plus the pair keys) can be joined to main's `attempts/<id>/entity_scores.parquet` by entity_id and grouped by main's folds (`data/folds.parquet`). That gives the paired per-fold difference, mean > 2 SE and 5/5 folds, without retraining either pipeline. The caveat: Codex's models were trained on different splits, which is fine because both sets of predictions are out-of-fold.

### Ideas the main pipeline lacks (ranked)

| # | Idea | Expected gain (test-weighted) | Test under the paired 5-fold rule |
|---|---|---|---|
| 1 | **Stage-2 competition features** from OOF stage-1 p: per pool record, rank, best-other-S1 p, margin, count and Σp; per S1, rank, gap to best, best-other, count and Σp. Retrain LightGBM on base + context features. Main's graph features use only the blocking `word_sim`, never model probabilities. | +0.002 to +0.005 (targets the 06 audit's ownership 0.0046 + selection 0.0136 losses) | Take 004b's (or 005's) OOF `prob` from `cache/oof/<id>.parquet` and compute context features per country graph. Train fold-k stage-2 models on folds ≠ k with the same `folds.parquet`, predict fold k, run `decide()`, and compare per fold with the parent. EC2, about 1-2 h, side-car only, so no `src/` edit needed to pilot it. |
| 2 | **Probability floor in `decide()`** (keep only p > t, t ∈ {0.3, 0.5, 0.6}), chosen with a LOCO check (train US → score India and the reverse) | CV about ±0.0002; France 0 to +0.003 (the 06 audit's France over-merge lean; the floor would also cut the §3c distractor pairs if they have mid-range p) | A `decide()` sweep on existing OOF gives the paired per-fold CV at no cost. LOCO needs two LightGBM fits on existing features (about 1 h). Adopt only if CV is neutral (≥ −1 SE) and both LOCO directions improve. |
| 3 | **Reverse retrieval + conjunction keys:** each pool record → top-3 S1 with S1-side IDF and a df cap; keys for address bigrams, house number×street word, name-token pairs, skeleton×house number | +0.001 to +0.003 (main's candidates miss 2.4-4.5 times more of Codex's matches than the reverse, with twice the candidates; US/France 11k and 23k pairs) | Measure oracle F0.5 on train (labels allowed) with and without the added pairs. Go if oracle gains ≥ +0.002 in some country, then rebuild features and run paired CV. |
| 4 | Phonetic consonant skeleton (cross-script spelling variants) as a key and a feature; non-Latin state map | +0.0005 to +0.002 India, after 005 | Ablation on 005: paired CV |
| 5 | **Probability blend with Codex** (average p on the union of scored pairs; a pair scored by one side only keeps that side's p, or gets 0) | +0.002 to +0.006 (65% set agreement means high diversity) | Needs Codex OOF p for all train pairs (it has them, `p2_*.npy`), joined by (entity_id, pool id) to main OOF. Grid w ∈ {0, .25, .5, .75, 1}, `decide()`, paired per fold on main's folds. Choose w on folds 0-1 and confirm on 2-4. |

- **Union of match sets: not worth testing.** F0.5 weighs precision double, and the one-sided pairs include the distractor-like US/France set.
- **Intersection:** would cost recall on about 250k India Indic pairs. Also not worth it.
- **Only via CV:** any ensemble needs both pipelines' OOF on the same entities, which both already have.

### Immediate actions (priority)
1. Before 005 is packaged on top of 004, run the §3c check (feat3 train-vs-test drift, predicted k OOF vs test by country). If the drift is confirmed, fix it or drop feat3 from test-time scoring. It may be costing more on the LB than 004's +0.0024 CV gain.
2. The paired OOF comparison Codex vs main (no training, about 30 min).
3. Pilot idea 1 on existing OOF.
4. If the human wants the Codex package as a fallback, fix §1c items 1-3 first. Only the human submits.
