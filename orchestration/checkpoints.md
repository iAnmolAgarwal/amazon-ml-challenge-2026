# Checkpoints

Appended by the LEADER every 4-6h when the human says "checkpoint". Never rewrite earlier entries.

Template:

## <YYYY-MM-DD HH:MM IST> (hour <N> of 72)
- Best CV: <attempt> <metric>=<mean> ± <std>; public LB: <score> (submission #<n>)
- Beat parent by >1 std since last checkpoint: <attempt ids, deltas>
- Failures: bugs worth retrying (<ids, suspected bug>) vs dead ideas (<ids, reason>)
- Hyperparameter-tuning trap: W1 <yes/no>, W2 <yes/no>
- CV vs public LB correlation (all submissions so far): <rank agreement, outliers>
- Compute: tsp queue length, what is on Kaggle/Colab/EC2, memory pressure
- Decisions: reassign / kill / open (mirrored into assignments.md)
- Submissions used: <n>/15 today <m>/5; next candidate: <attempt or none>
