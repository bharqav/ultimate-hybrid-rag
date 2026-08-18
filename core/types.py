from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source_document: str
    page_number: int
    section_title: str = ""
    token_count: int = 0
    dense_embedding: Optional[List[float]] = None
    splade_weights: Optional[Dict[str, float]] = None
    colbert_doc_id: int = -1


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    vector_score: float = 0.0
    bm25_score: float = 0.0
    splade_score: float = 0.0
    colbert_score: float = 0.0
    fused_score: float = 0.0
