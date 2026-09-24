WORKER W{N}. Read CLAUDE.md, prompts/AGENT_PROMPT.md, research/insights.md, and your section of orchestration/assignments.md. Work only in your git worktree on branch w{n}.
Loop: read all attempts -> src/new_attempt.py -> write proposal.md -> implement train.py -> evaluate --subsample on CPU/mps as a sanity check -> run scripts/memcheck.sh -> queue full CV via scripts/queue.sh -> append "Result" section to proposal.md (CV vs parent, what it shows, next idea) -> repeat within your direction.
Reuse cached embeddings/features from cache_dir; never recompute what exists.
If your kill condition is hit, write "DIRECTION EXHAUSTED: <reason>" in assignments.md and stop.
Never: touch test data, submit, edit other workers' attempts, modify src/ without asking, decide from differences <1 std.
