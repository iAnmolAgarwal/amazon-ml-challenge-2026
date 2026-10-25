You previously recommended a compute plan for a large sparse top-k retrieval job (TF-IDF cosine, sparse_dot_topn 1.2.0 `sp_matmul_topn`, top_n=20, 8 threads) and assumed cost scales linearly with pool size (505 queries/s per 1M pool rows, measured with one call on a 1M pool). An independent auditor measured something different and claims a 3-5x speedup. Verify or refute it. Assume the auditor may be wrong.

## Auditor's claim
1. A single `sp_matmul_topn(Q, P.T, top_n=20, n_threads=8, sort=True)` call gets SLOWER per pool row as the pool grows (superlinear), and
2. splitting the pool into row chunks, running `sp_matmul_topn` per chunk, and merging with `sparse_dot_topn.zip_sp_matmul_topn(top_n, [per-chunk results])` restores linear scaling and is 3-5x faster, with identical top-k results.

## Measurements (Apple M3, 8 threads; 20,000 US queries; name char 3-grams, TfidfVectorizer(analyzer='char', ngram_range=(3,3), sublinear_tf=True, float32); ~20 non-zeros per row; pool = gold matches + random same-country distractors)

| Method | Queries/s per 1M pool rows |
|---|---|
| One call, 0.5M pool | 1629 |
| One call, 1M pool | 579-613 |
| One call, 2M pool | 388 |
| 2M pool, 500k-row chunks + zip merge | 1330 |
| 2M pool, 250k-row chunks + zip merge | 1709 |
| 2M pool, 125k-row chunks + zip merge | 2107 |

Other views at 2M, chunked 250k (queries/s per 1M rows): name+address word 1-2 grams (HashingVectorizer 2^23 + TfidfTransformer, ~17 non-zeros/row US, ~24-27 India): 5200 US / 1360 India; address char 3-grams: 1076 US, 407 India (India 53-65 non-zeros/row due to Indic scripts).

## Benchmark code (exact)
```python
PTs = [P[i:i+ch].T.tocsr() for i in range(0, P.shape[0], ch)]   # built OUTSIDE the timer
t = time.time()
C = zip_sp_matmul_topn(K, [sp_matmul_topn(Q, x, top_n=K, n_threads=8, sort=True) for x in PTs])
tq = time.time() - t
# unchunked: PT = P.T.tocsr(); C = sp_matmul_topn(Q, PT, top_n=K, n_threads=8, sort=True)
```
Caveats I already know: per-chunk transposes are excluded from timing (one-time per index); IDF was fit on the 2M slice, not the full pool; measured on the Mac, not on the target EC2 r7a.2xlarge (AMD Zen 4, 8 physical cores).

## Planned use
Full country pools: US ~6.2M train / 3.8M test rows, India ~4.1M / 4.7M, France ~1.4M test; queries 0.26M-1.3M per country. Plan: chunk pools at 125-250k rows, merge with zip_sp_matmul_topn, pick the chunk size with a 5-minute benchmark on the EC2 box.

## Answer
1. Is the superlinear slowdown of a single call plausible for this library/algorithm? Give the mechanism (e.g. per-thread dense accumulator of size n_cols, cache misses, top-k heap/insert behaviour), citing how sp_matmul_topn is implemented if you know it.
2. Is chunk-and-merge exactly equivalent (same top-k set and scores, tie handling) to one call? Any correctness trap with zip_sp_matmul_topn (column offsets, sorting, duplicate handling)?
3. Is the benchmark sound? What would make it misleading (excluded transposes, warm caches, thread contention, query batch size, float32)?
4. Would the speedup carry to Zen 4 (1 MB L2 per core, 32 MB L3), and what chunk sizes should be tested there?
5. Anything better: e.g. also chunking queries, or parallelising chunks across processes instead of threads inside one call.
VERDICT first (CONFIRMED / PARTLY / REFUTED), then the answers. No praise, no summary.
