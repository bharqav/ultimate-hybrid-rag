from typing import List

from core.types import DocumentChunk
from retrieval.bm25_store import BM25Store
from retrieval.colbert_store import ColBERTStore
from retrieval.splade_index import SPLADEIndex
from retrieval.vector_store import VectorStore


def build_indexes(chunks: List[DocumentChunk], db_dir=None):
    chunk_ids = [c.chunk_id for c in chunks]

    vec_store = VectorStore(db_dir=db_dir)
    vec_store.build([c.dense_embedding for c in chunks], chunk_ids)

    bm25_store = BM25Store(db_dir=db_dir)
    bm25_store.build(chunks)

    splade_idx = SPLADEIndex(db_dir=db_dir)
    splade_idx.build(chunks)

    colbert_store = ColBERTStore(db_dir=db_dir)
    colbert_store.build(chunks)
