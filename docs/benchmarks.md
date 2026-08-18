# Benchmarks

This is a template for reporting retrieval and end-to-end metrics.

## Retrieval Latency

| Variant | p50 (ms) | p95 (ms) | p99 (ms) | Notes |
| --- | --- | --- | --- | --- |
| baseline | TBD | TBD | TBD | single node |

## Retrieval Quality

| Variant | Recall@1 | Recall@3 | Recall@5 | MRR | NDCG |
| --- | --- | --- | --- | --- | --- |
| baseline | TBD | TBD | TBD | TBD | TBD |

## How to Run

1. Add queries to docs/bench_queries.txt (one per line)
2. Run: python scripts/run_benchmark.py --queries docs/bench_queries.txt
3. Results will be written to docs/benchmarks/results.csv
