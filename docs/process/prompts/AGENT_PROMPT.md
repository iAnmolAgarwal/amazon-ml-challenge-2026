Before proposing anything, read every attempt's proposal.md and results.json (not a sample, not just your branch). Trust measured results over claims in proposals.
Never load, inspect, or evaluate on test data.
Improvement = significant paired gain over the parent on the same folds (mean per-fold difference > 2x its standard error, positive in at least 4 of 5 folds). Smaller or inconsistent differences are noise; don't build on them. Also check the test-weighted country breakdown (US 0.38 / India 0.47) for regressions.
For every failed or weak attempt decide: bad idea, or good idea with a flawed implementation? Check leakage, preprocessing fit on validation, feature/label misalignment, wrong loss or metric, LR/epochs, class imbalance. Retry a promising idea only after locating the bug in the code.
If recent attempts are hyperparameter tweaks with shrinking gains, stop tuning: propose a structural change (features, data cleaning, model family, retrieval/blocking, ensembling).
Prefer compute-efficient approaches: frozen pretrained embeddings cached once + GBDT/small MLP heads + handcrafted features. Propose fine-tuning only with an estimate of its runtime on this Mac vs Kaggle.
Before proposing, inspect where the current best fails (confusion matrix, worst slices, high-loss examples) and target that.
Write proposal.md first, then code. Never claim improvement before evaluation.
