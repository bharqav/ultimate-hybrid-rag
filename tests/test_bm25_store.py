import pytest
from core.types import DocumentChunk
from retrieval.bm25_store import BM25Store


def test_bm25_store_build_and_search():
    store = BM25Store(db_dir=".")
    
    chunks = [
        DocumentChunk(chunk_id="c_1", text="The quick brown fox", source_document="test", page_number=1),
        DocumentChunk(chunk_id="c_2", text="jumps over the lazy dog", source_document="test", page_number=1),
        DocumentChunk(chunk_id="c_3", text="foxes are quick", source_document="test", page_number=1),
    ]
    
    store.build(chunks)
    assert len(store.metadata) == 3
    assert store.bm25 is not None
    
    # Search
    results = store.search("quick fox", k=2)
    assert len(results) == 2
    # Expect c_1 to be in results due to exact match
    ids = [r[0] for r in results]
    assert "c_1" in ids
