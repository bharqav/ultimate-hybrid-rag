import os
import pickle
from typing import List, Tuple

import numpy as np

from config.settings import get_settings
from core.deps import BM25Okapi
from core.types import DocumentChunk


class BM25Store:
    def __init__(self, db_dir=None, path=None):
        settings = get_settings()
        db_dir = db_dir or settings.db_dir
        self.path = path or os.path.join(db_dir, "bm25.pkl")
        self.corpus = []
        self.metadata = []
        self.bm25 = None

    def build(self, chunks: List[DocumentChunk]):
        if BM25Okapi is None:
            raise RuntimeError("rank-bm25 not installed")
        tokenized = [chunk.text.lower().split() for chunk in chunks]
        self.corpus = tokenized
        self.metadata = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "source": c.source_document,
                "page": c.page_number,
                "section": c.section_title,
                "token_count": c.token_count,
            }
            for c in chunks
        ]
        self.bm25 = BM25Okapi(tokenized)
        with open(self.path, "wb") as f:
            pickle.dump({"corpus": self.corpus, "metadata": self.metadata}, f)

    def load(self):
        if BM25Okapi is None:
            raise RuntimeError("rank-bm25 not installed")
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        self.corpus = data["corpus"]
        self.metadata = data["metadata"]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(self.metadata[i]["chunk_id"], float(scores[i])) for i in top_indices]
