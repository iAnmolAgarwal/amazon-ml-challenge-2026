# Task list (live; tick as done)

Finish line: public LB >= 0.99 (bar for any build: evidence of a large gain in paired CV + label-free test guard).
Stop-and-ask points: src/ edits (human approval), AWS changes beyond pre-approved compute, uploads (human only).

## Tonight (25 Sep, uploads left today: 2, expire 23:59 IST)
- [x] Research sample + kit (~/aml/rk), 9 lanes, reports docs/research/r1..r9.md
- [x] feat5 port verified exact vs the lane's features (13.1M test-France pairs, 0 mismatches)
- [ ] Chain on A (build/chain2.sh): feat5 train/test -> stage-3 CV on full train (arms ctrl / r5 / all) -> test TSV + guard
- [ ] Verify: stage-3 full-train CV vs 006 (paired rule), guard (house conflict, empty rate, per S1) vs 006
- [ ] Human approval: src/er_feat5.py + src/er_stage3.py + attempt 008 -> validate + package -> upload candidate

## Next build (26 Sep)
- [ ] Port decoy name-marker features (r9 N: label-free dm_add/dm_drop/dm_swap scores; France gets its own list) + signed house delta
- [ ] Add r1 address "new number" features and legal-change code; house-edit (r7) overlap check
- [ ] Name-frequency counts per country size (r4 finding: US test has half the S1)
- [ ] Full integration: features into stage 1 + stage 2 (join-free loader first; scratchpad loader_draft.py)
- [ ] Candidate set: stage-1 band as a learned filter so the final model and candidate_pairs.tsv use ~6 pairs/S1 instead of ~45 (guideline: smaller candidate sets rank higher)
- [ ] Retrieval: reverse top-1 + name-prefix+house key (r8, +0.0015 LB est.) only if the build has time
- [ ] France: house-conflict strictness under decoy density (r3); LOCO numbers (r3 loco_q)

## Rejected (evidence in reports)
- Sibling/coherence/support features incl. house agreement and source balance (test decoy clusters, r7/r2/r4)
- Decision-rule changes (r4: already Bayes-optimal); India native-word mapping (r8: decoy-only words)

## Research lanes, round 2 (resumed 25 Sep ~22:05 IST; each reports in docs/research/rN.md)
- [ ] r3: LOCO recipes for the unseen country: generator features, country-size normalisation, self-training (France is the biggest lever: LOCO loses 0.05-0.07)
- [ ] r9: full-data decoy-marker features (er_feat6), verified vs sample; France's own marker table from test
- [ ] r6: candidate-set reduction (learned stage-1 filter), candidates/S1 vs F0.5 loss
- [ ] r2: test guard decides house-agreement / source-balance; decoy-robust variant
- [ ] r7: decoy-cloning simulation for the stage-3 arms (p5, p59)
- [ ] r1: address-operator + legal-change add-on over the shipped set
- [ ] r5: stage 3 vs full integration (stage-2 retrain with features); France legal-form check
- [ ] r4: name-frequency country-size shift simulation (US test has half the S1)
- [ ] r8: cheap retrieval (reverse top-1, p6h) priced in candidates/S1
