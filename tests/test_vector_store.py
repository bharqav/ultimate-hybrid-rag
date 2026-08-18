import numpy as np
import pytest
from core.types import DocumentChunk
from retrieval.vector_store import VectorStore


def test_vector_store_build_and_search():
    store = VectorStore(db_dir=".")
    
    # Create fake chunks with embeddings
    chunks = []
    for i in range(1050):
        c = DocumentChunk(chunk_id=f"c_{i}", text=f"text {i}", source_document="test", page_number=1)
        # Settings defaults to 384 dimensions for all-MiniLM-L6-v2
        c.dense_embedding = [0.1] * 384
        chunks.append(c)
        
    embeddings = [c.dense_embedding for c in chunks]
    ids = [c.chunk_id for c in chunks]
    store.build(embeddings, ids)
    assert store.index is not None
    assert store.index.ntotal == 1050
    assert len(store.id_map) == 1050
    
    # Search
    q_emb = [0.1] * 384
    ids, scores = store.search(q_emb, k=2)
    assert len(ids) == 2
    assert "c_1" in ids  # Exact match for index 1
