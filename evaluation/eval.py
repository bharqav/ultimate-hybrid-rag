import os
import shutil
import numpy as np
from typing import List

from config.logging import setup_logging
from config.settings import get_settings
from core.types import DocumentChunk
from indexing.build import build_indexes
from ingestion.ingest import embed_chunk_batch
from retrieval.retriever import UltimateRetriever

log = setup_logging()

def build_eval_data(num_docs=200, num_queries=15):
    log.info("Generating fully offline benchmark dataset...")
    
    eval_docs = []
    for i in range(num_docs):
        eval_docs.append({
            "_id": f"doc_{i}",
            "title": f"Scientific Paper {i}",
            "text": f"This document contains important research regarding topic {i}. It explores the implications of hybrid retrieval methods and dense vectors."
        })
        
    eval_queries = []
    qrels = []
    
    # Generate queries that explicitly target specific documents
    for i in range(num_queries):
        target_doc_index = i * (num_docs // num_queries)
        q_id = f"q_{i}"
        eval_queries.append({
            "_id": q_id,
            "text": f"What does the research say about topic {target_doc_index}?"
        })
        qrels.append({
            "query-id": q_id,
            "corpus-id": f"doc_{target_doc_index}"
        })
        
    log.info(f"Generated {len(eval_docs)} documents and {len(eval_queries)} queries for evaluation.")
    return eval_docs, eval_queries, qrels


def evaluate():
    settings = get_settings()
    eval_db_dir = os.path.join(settings.db_dir, "eval_db")
    if os.path.exists(eval_db_dir):
        shutil.rmtree(eval_db_dir)
    os.makedirs(eval_db_dir, exist_ok=True)
    
    eval_docs, eval_queries, qrels = build_eval_data(num_docs=200, num_queries=15)
    if not eval_docs:
        return
        
    log.info("Chunking and embedding evaluation documents...")
    chunks: List[DocumentChunk] = []
    for d in eval_docs:
        chunk = DocumentChunk(
            chunk_id=d["_id"],
            text=f"{d['title']}\n{d['text']}",
            source_document="synthetic_benchmark",
            page_number=1,
            token_count=len(d["text"].split())
        )
        chunks.append(chunk)
        
    texts = [c.text for c in chunks]
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), settings.gpu_embed_batch_size):
        batch = texts[i : i + settings.gpu_embed_batch_size]
        emb = embed_chunk_batch(batch)
        embeddings.extend(emb)

    for i, chunk in enumerate(chunks):
        chunk.dense_embedding = embeddings[i]

    log.info("Building vector, bm25, splade, and colbert indexes...")
    build_indexes(chunks, db_dir=eval_db_dir)
    
    settings.db_dir = eval_db_dir
    engine = UltimateRetriever()
    
    log.info("Running evaluation queries...")
    recall_at_k = {1: [], 3: [], 5: [], 10: []}
    mrr_total = 0
    ndcg_total = 0
    
    for item in qrels:
        q_text = next(q["text"] for q in eval_queries if q["_id"] == item["query-id"])
        target_doc_id = str(item["corpus-id"])
        
        results = engine.search(q_text)
        
        for rank, r in enumerate(results, start=1):
            if r.chunk.chunk_id == target_doc_id:
                mrr_total += 1.0 / rank
                break
                
        for k in [1, 3, 5, 10]:
            recall = sum(1 for r in results[:k] if r.chunk.chunk_id == target_doc_id)
            recall_at_k[k].append(recall)
            
        dcg = 0
        idcg = 1.0
        for rank, r in enumerate(results[:10], start=1):
            if r.chunk.chunk_id == target_doc_id:
                dcg += 1 / np.log2(rank + 1)
        ndcg = dcg / idcg if idcg else 0
        ndcg_total += ndcg

    n = len(qrels)
    log.info("=== EVALUATION BENCHMARK RESULTS ===")
    log.info("Dataset: Synthetic Offline Benchmark (200 docs, 15 queries)")
    log.info(
        "Recall@1: %.3f | Recall@3: %.3f | Recall@5: %.3f | Recall@10: %.3f",
        np.mean(recall_at_k[1]),
        np.mean(recall_at_k[3]),
        np.mean(recall_at_k[5]),
        np.mean(recall_at_k[10]),
    )
    log.info("MRR: %.3f | NDCG@10: %.3f", mrr_total / n, ndcg_total / n)
    log.info("====================================")
