import argparse
import csv
import os
import time
from statistics import mean

from retrieval.retriever import UltimateRetriever


def load_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    k = int(round((p / 100.0) * (len(values) - 1)))
    return values[k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="docs/bench_queries.txt")
    parser.add_argument("--output", default="docs/benchmarks/results.csv")
    args = parser.parse_args()

    queries = load_queries(args.queries)
    engine = UltimateRetriever()

    latencies = []
    rows = []
    for q in queries:
        start = time.perf_counter()
        results = engine.search(q)
        elapsed = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed)
        rows.append({"query": q, "latency_ms": "%.3f" % elapsed, "results": len(results)})

    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "latency_ms", "results"])
        writer.writeheader()
        writer.writerows(rows)

    print("p50=%.2fms p95=%.2fms p99=%.2fms mean=%.2fms" % (p50, p95, p99, mean(latencies)))


if __name__ == "__main__":
    main()
