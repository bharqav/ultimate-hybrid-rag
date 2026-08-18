import numpy as np

from config.logging import setup_logging
from retrieval.retriever import UltimateRetriever

log = setup_logging()


def build_eval_data():
    return [
        {"query": "Model RX-78-2 v4", "relevant_chunk_ids": ["hash_of_that_chunk"]},
        {"query": "What is RAG?", "relevant_chunk_ids": ["hash1", "hash2"]},
    ]


def evaluate():
    engine = UltimateRetriever()
    test_set = build_eval_data()
    recall_at_k = {1: [], 3: [], 5: []}
    mrr_total = 0
    ndcg_total = 0
    for item in test_set:
        results = engine.search(item["query"])
        for rank, r in enumerate(results, start=1):
            if r.chunk.chunk_id in item["relevant_chunk_ids"]:
                mrr_total += 1.0 / rank
                break
        for k in [1, 3, 5]:
            recall = sum(1 for r in results[:k] if r.chunk.chunk_id in item["relevant_chunk_ids"]) / len(
                item["relevant_chunk_ids"]
            )
            recall_at_k[k].append(recall)
        dcg = 0
        idcg = sum(1 / np.log2(i + 2) for i in range(len(item["relevant_chunk_ids"])))
        for rank, r in enumerate(results[:10], start=1):
            if r.chunk.chunk_id in item["relevant_chunk_ids"]:
                dcg += 1 / np.log2(rank + 1)
        ndcg = dcg / idcg if idcg else 0
        ndcg_total += ndcg
    n = len(test_set)
    log.info(
        "Recall@1: %.3f, Recall@3: %.3f, Recall@5: %.3f",
        np.mean(recall_at_k[1]),
        np.mean(recall_at_k[3]),
        np.mean(recall_at_k[5]),
    )
    log.info("MRR: %.3f, NDCG: %.3f", mrr_total / n, ndcg_total / n)
